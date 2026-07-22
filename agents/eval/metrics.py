"""
agents/eval/metrics.py
STEP3-1: 규칙 기반(LLM 불필요) 지표 계산

설계 문서 'STEP3-1: Metric 진단' 구현.
세 지표(Recall@k, token F1, Oracle F1)와 무응답 판별을 제공한다. 이 지표들의 조합으로
원인을 판정하는 로직은 signals/diagnose 로 이관됐다(브랜치리스). 전량 순수 파이썬이라
외부 API·모델 없이 항상 동작한다.

[구현 포인트]
  - 정답 매칭은 KorQuAD 공식 지표를 따른다: normalize_answer(구두점 제거·소문자화·공백정리)
    후 '문자 단위(bag-of-characters) F1'. 한국어는 완벽한 형태소 분석기가 없고 어절 단위
    F1이 F1 취지를 못 살려서, KorQuAD 1.0 이 문자 단위를 표준으로 채택했다.
    (근거: KorQuAD 1.0 논문 / evaluate-v1.0.py)
  - 문자 단위라 조사·어미·숫자 포맷(1,450↔1450, 14:33↔14시33분) 차이가 대부분 흡수된다.
    단 부정·근접오답 같은 의미 판정은 문자 겹침으로 못 잡으므로 tier3(RAGAS)가 담당한다.
  - 설계상 추가 지표(Precision, MRR, nDCG, No-Hit, EHR, Number Match 등)는
    여기에 함수로 덧붙여 확장한다.
"""
from __future__ import annotations

import string
from collections import Counter


# ── 정규화 (KorQuAD 공식 normalize_answer) ────────────────────────
# KorQuAD 1.0 evaluate-v1.0.py 와 동일: 따옴표·괄호류 → 공백, 소문자화, ASCII 구두점 제거, 공백 정리.
# 숫자·날짜 특수 규칙은 두지 않는다 — 콤마·콜론·하이픈은 구두점 제거로 흡수되고(1,450→1450,
# 14:33→1433), 조사·어미·포맷 차이는 아래 '문자 단위' 채점이 흡수한다.

_REMOVE_CHARS = "'\"《》<>〈〉()" + "‘’“”"   # remove_: 따옴표·괄호류
_REMOVE_TABLE = {ord(c): " " for c in _REMOVE_CHARS}
_PUNCT_SET = set(string.punctuation)


def _normalize(text: str) -> str:
    """KorQuAD normalize_answer: 따옴표·괄호류 → 공백 → 소문자화 → ASCII 구두점 제거 → 공백 정리."""
    if not text:
        return ""
    text = text.translate(_REMOVE_TABLE)                     # remove_
    text = text.lower()                                      # lower
    text = "".join(c for c in text if c not in _PUNCT_SET)   # remove_punc
    return " ".join(text.split())                            # white_space_fix


def _chars(text: str) -> list[str]:
    """정규화 후 '문자 리스트'(공백 제외) — KorQuAD F1 의 채점 단위(bag of characters)."""
    return [c for tok in _normalize(text).split() for c in tok]


# ── 생성 지표 (KorQuAD 문자 단위) ─────────────────────────────────

def char_f1(prediction: str, reference: str) -> float:
    """KorQuAD 공식 문자 단위 F1 (bag-of-characters).
        Precision = 겹친 문자 수 / 답변 문자 수,  Recall = 겹친 문자 수 / 정답 문자 수,  F1 = 2PR/(P+R)
    한국어는 완벽한 형태소 분석기가 없어 어절 단위 F1이 부적절 → KorQuAD가 문자 단위를 표준으로 쓴다.
    reference 없으면(정답 미보유) 0.0."""
    pred = _chars(prediction)
    ref = _chars(reference)
    if not pred or not ref:
        return 0.0
    num_same = sum((Counter(pred) & Counter(ref)).values())   # 멀티셋 교집합(문자 중복 고려)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred)
    recall = num_same / len(ref)
    return 2 * precision * recall / (precision + recall)


# 하위호환 별칭 — 기존 호출부(signals._ablation_helps, tests)가 token_f1 이름을 쓴다.
# 이제 어절이 아니라 KorQuAD 문자 단위 F1이다.
token_f1 = char_f1


def exact_match(prediction: str, reference: str) -> bool:
    """KorQuAD 공식 EM — 정규화 문자열 완전 일치."""
    return _normalize(prediction) == _normalize(reference)


# ── 정답 매칭 (KorQuAD 문자 F1 + 짧은 정답 문자-recall) ───────────
# char_f1 도 결국 F1(P·R 조화평균)이라, gold 가 짧은 추출형 span 인데 답변이 완결 문장이면
# 프레이밍 문자에 precision 이 깎인다. 짧은 정답은 recall(정답 문자가 답변에 담겼나)만 보는 게
# 추출형 QA 의 정석 — 표준 char-F1 위에 얹는 확장이다.

_SHORT_REF_MAX_CHARS = 10   # 정규화 후 정답 문자 수 이하면 '짧은 정답'으로 보고 recall 경로 허용
_CONTAINMENT_MIN = 0.9      # 짧은 정답은 recall 이 이 이상(정답이 거의 다 담김)일 때만 recall 로 통과


def char_recall(prediction: str, reference: str) -> float:
    """정답 문자가 답변에 담긴 비율(문자 단위 recall = 겹친 문자 / 정답 문자).
    Counter 멀티셋 교집합으로 겹친 문자 수(중복 고려)를 세고 정답 문자 수로 나눈다.
    답변 길이·순서·위치는 보지 않는다 — 완결 문장의 프레이밍 문자에 precision 이 깎이는 짧은 정답용."""
    ref = _chars(reference)
    if not ref:
        return 0.0
    pred = _chars(prediction)
    num_same = sum((Counter(pred) & Counter(ref)).values())
    return num_same / len(ref)


def answer_match(prediction: str, reference: str) -> float:
    """정답 매칭 점수(규칙 기반 tier1). 기준은 KorQuAD 문자 단위 F1이고, 짧은 정답(정규화 후 ≤10자)은
    '정답 문자가 답변에 거의 다 담겼을 때(recall ≥ 0.9, containment)'에 한해 recall 로 통과시켜
    완결 문장의 precision 감점을 피한다. reference 없으면 0.0.

    recall 문턱(containment)을 둔 이유: 문턱 없이 max(f1, recall) 이면 정답 문자를 일부만 공유하는
    근접 오답(gold '145' ↔ 답 '150' → recall 0.67)도 통과해 너무 후해진다. recall 을 '거의 완전
    포함'으로 제한하면 verbose 정답(recall≈1.0, '332cm입니다')은 살리고 near-miss 오답은 char_f1 로 떨어진다.

    [남는 한계] 부정/모순('사망'⊂'사망하지 않았다', recall=1.0)과 '3월'↔'3일'(char_f1=0.5)은
    문자 단위로 못 거른다 → 의미 판정은 tier3(RAGAS), 관측은 EM 병기가 담당한다."""
    ref = _chars(reference)
    if not ref:
        return 0.0
    f1 = char_f1(prediction, reference)
    if len(ref) <= _SHORT_REF_MAX_CHARS:
        rc = char_recall(prediction, reference)
        if rc >= _CONTAINMENT_MIN:              # 정답이 거의 다 담김 → recall 로 precision 감점 상쇄
            return max(f1, rc)
    return f1


# ── 검색 지표 ─────────────────────────────────────────────────────

def recall_at_k(gold_chunk_ids: list[str], retrieved_chunk_ids: list[str]) -> float:
    """
    상위 k개 검색 결과 안에 gold 청크가 얼마나 포함됐는지 비율.
    단일홉(gold 1개)이면 0/1, 멀티홉(gold n개)이면 0~1의 부분 점수.
    gold 가 없으면(무응답 케이스 등) 계산 불가 → -1.0 (판정에서 별도 처리).
    """
    if not gold_chunk_ids:
        return -1.0
    retrieved = set(retrieved_chunk_ids)
    hit = sum(1 for g in gold_chunk_ids if g in retrieved)
    return hit / len(gold_chunk_ids)


def span_recall_at_k(
    gold_spans: list[dict],
    retrieved_chunk_ids: list[str],
    chunks: list,
) -> float | None:
    """검색된 청크 좌표가 gold span을 얼마나 온전히 덮는지 계산한다.

    한 청크가 span 전체를 포함해도 성공이고, 여러 검색 청크의 좌표 합집합이
    빈틈없이 span을 덮어도 성공이다. 청크 좌표가 없어 계산할 수 없는 legacy
    환경에서는 None을 반환해 호출부가 기존 chunk-id Recall로 폴백하게 한다.
    """

    valid_spans: list[tuple[str, int, int]] = []
    for span in gold_spans:
        if not isinstance(span, dict):
            continue
        doc_id = span.get("doc_id")
        start = span.get("start")
        end = span.get("end")
        if (
            isinstance(doc_id, str)
            and isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and start >= 0
            and end > start
        ):
            valid_spans.append((doc_id, start, end))
    if not valid_spans:
        return None

    all_positions: dict[str, list[tuple[int, int]]] = {}
    retrieved_positions: dict[str, list[tuple[int, int]]] = {}
    retrieved = set(retrieved_chunk_ids)
    gold_doc_ids = {doc_id for doc_id, _start, _end in valid_spans}
    retrieved_gold_chunk_without_position = False
    for chunk in chunks:
        doc_id = getattr(chunk, "doc_id", None)
        chunk_id = getattr(chunk, "chunk_id", None)
        raw = getattr(chunk, "char_span", None)
        metadata = getattr(chunk, "metadata", None)
        if raw is None and isinstance(metadata, dict):
            raw = metadata.get("char_span")
        if (
            not isinstance(raw, (list, tuple))
            or len(raw) != 2
            or isinstance(raw[0], bool)
            or isinstance(raw[1], bool)
            or not isinstance(raw[0], int)
            or not isinstance(raw[1], int)
            or raw[0] < 0
            or raw[1] <= raw[0]
        ):
            if doc_id in gold_doc_ids and chunk_id in retrieved:
                retrieved_gold_chunk_without_position = True
            continue
        if not isinstance(doc_id, str):
            continue
        position = (raw[0], raw[1])
        all_positions.setdefault(doc_id, []).append(position)
        if chunk_id in retrieved:
            retrieved_positions.setdefault(doc_id, []).append(position)

    # gold 문서 좌표가 없거나 검색된 gold 문서 청크의 좌표가 일부라도 빠지면
    # span 기반 0점으로 단정하지 않고 기존 chunk-id Recall로 폴백한다.
    if retrieved_gold_chunk_without_position or any(
        not all_positions.get(doc_id) for doc_id, _start, _end in valid_spans
    ):
        return None

    covered = 0
    for doc_id, start, end in valid_spans:
        intersections = sorted(
            (max(start, c_start), min(end, c_end))
            for c_start, c_end in retrieved_positions.get(doc_id, [])
            if c_start < end and c_end > start
        )
        cursor = start
        for covered_start, covered_end in intersections:
            if covered_start > cursor:
                break
            cursor = max(cursor, covered_end)
            if cursor >= end:
                break
        if cursor >= end:
            covered += 1
    return covered / len(valid_spans)


# ── 무응답(기권) 판별 ─────────────────────────────────────────────

_ABSTENTION_MARKERS = (
    "모른다", "모르겠", "알 수 없", "답변할 수 없", "답변하기 어렵",
    "정보가 없", "정보를 찾을 수 없", "제공된 정보로는 알 수 없", "확인할 수 없",
    "don't know", "do not know", "cannot answer", "no information",
    "not available", "unable to answer",
)


def is_abstention(answer: str) -> bool:
    """
    답변이 '모른다/답할 수 없다'류의 기권인지 휴리스틱 판별.
    [구현 포인트] 설계의 Non-Answer Critic(RAGAS AspectCritic)으로 정밀화 가능.
    """
    if not answer or not answer.strip():
        return True
    low = answer.lower()
    return any(m in low for m in _ABSTENTION_MARKERS)
