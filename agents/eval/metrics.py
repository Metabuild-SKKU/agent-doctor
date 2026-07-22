"""
agents/eval/metrics.py
STEP3-1: 규칙 기반(LLM 불필요) 지표 계산

설계 문서 'STEP3-1: Metric 진단' 구현.
세 지표(Recall@k, token F1, Oracle F1)와 무응답 판별을 제공한다. 이 지표들의 조합으로
원인을 판정하는 로직은 signals/diagnose 로 이관됐다(브랜치리스). 전량 순수 파이썬이라
외부 API·모델 없이 항상 동작한다.

[구현 포인트]
  - 토큰화가 공백 기준이라 한국어 조사/어미를 분리하지 못한다.
    형태소 분석기(kiwi, mecab) 도입 시 `_tokenize()` 만 교체하면 된다.
  - 설계상 추가 지표(Precision, MRR, nDCG, No-Hit, EHR, Number Match 등)는
    여기에 함수로 덧붙여 확장한다.
"""
from __future__ import annotations

import re
from collections import Counter


# ── 토큰화 ────────────────────────────────────────────────────────

_PUNCT = re.compile(r"[^\w가-힣]+")
_COMMA_IN_NUM = re.compile(r"(?<=\d),(?=\d)")                    # 1,450 → 1450 (천단위 콤마)
_DATE_KO = re.compile(r"(\d+)\s*년\s*(\d+)\s*월\s*(\d+)\s*일")   # 2018년 3월 27일
_DATE_ISO = re.compile(r"(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})")  # 2018-03-27, 2018.3.27
_TIME_KO = re.compile(r"(\d+)\s*시\s*(\d+)\s*분")               # 14시 33분
_TIME_COLON = re.compile(r"(\d{1,2}):(\d{2})")                  # 14:33


def _ymd(m) -> str:
    """날짜 매치 → ' Y M D '(선행 0 제거). 2018-03-27 ↔ 2018년 3월 27일 을 같은 토큰열로."""
    y, mo, d = (int(g) for g in m.groups())
    return f" {y} {mo} {d} "


def _hm(m) -> str:
    """시간 매치 → ' H M '(선행 0 제거). 14:33 ↔ 14시 33분 을 같은 토큰열로."""
    h, mi = (int(g) for g in m.groups())
    return f" {h} {mi} "


def _normalize(text: str) -> str:
    """소문자화 + 천단위 콤마 제거 + 날짜/시간 '패턴'을 숫자열로 정규화 + 구두점 제거.
    포맷 차이만 흡수한다: '2018년 3월 27일'↔'2018-03-27', '14시 33분'↔'14:33'.
    단위 통삭제가 아니라 '완결 패턴'만 바꾸므로 단위 구분은 보존된다 — 단독 '3월'·'3일'·'14시'는
    collapse 되지 않는다(예전 년/월/일/시/분/초 통삭제가 만들던 3월=3일=3 오정규화 교정).
    [구현 포인트] 형태소 분석기(kiwi/mecab) 도입 시 여기서 조사/어미까지 분리하면 된다."""
    if not text:
        return ""
    text = text.lower()
    text = _COMMA_IN_NUM.sub("", text)   # 1,450 → 1450
    text = _DATE_KO.sub(_ymd, text)      # 2018년 3월 27일 → 2018 3 27
    text = _DATE_ISO.sub(_ymd, text)     # 2018-03-27    → 2018 3 27
    text = _TIME_KO.sub(_hm, text)       # 14시 33분     → 14 33
    text = _TIME_COLON.sub(_hm, text)    # 14:33         → 14 33
    return _PUNCT.sub(" ", text)         # 나머지 구두점 → 공백


def _tokenize(text: str) -> list[str]:
    """정규화 후 공백 분리. [구현 포인트] 형태소 분석기로 교체 가능."""
    return [t for t in _normalize(text).split() if t]


# ── 생성 지표 ─────────────────────────────────────────────────────

def token_f1(prediction: str, reference: str) -> float:
    """
    답변(prediction)과 정답(reference)의 토큰 겹침 F1.
        Precision = 겹친 토큰 수 / 답변 토큰 수
        Recall    = 겹친 토큰 수 / 정답 토큰 수
        F1        = 2PR / (P+R)
    reference 가 없으면(=정답 미보유) 계산 불가 → 0.0.
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    # 다중 등장 토큰까지 고려한 겹침 수(멀티셋 교집합)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ── 정답 매칭 (recall/포함 위주 — 표면형 강건) ────────────────────
# token_f1 은 precision 을 함께 보므로, gold 가 짧은 추출형 span 인데 답변이 완결 문장이면
# 프레이밍 토큰('…시점은 …입니다')에 precision 이 깎여 의미가 맞아도 점수가 낮게 나온다.
# 짧은 정답은 recall(정답이 답변에 담겼나)만 보는 게 추출형 QA(KorQuAD 등)의 정석이다.

_SHORT_REF_MAX_TOKENS = 5   # 이하 토큰이면 '추출형 짧은 정답'으로 보고 recall 위주로 채점


def _covers(pred_tok: str, gold_tok: str) -> bool:
    """pred 토큰이 gold 토큰을 '커버'하나 — 조사/어미 부착은 허용, 숫자 오확장은 금지.
      · 단어: 접두 일치('사이프러스입니다' ⊇ '사이프러스').
      · 숫자: gold 뒤에 숫자가 더 붙으면 다른 수(145 ≠ 1450) → 불인정."""
    if not pred_tok.startswith(gold_tok):
        return False
    rest = pred_tok[len(gold_tok):]
    if gold_tok[-1:].isdigit() and rest[:1].isdigit():
        return False
    return True


def token_recall(prediction: str, reference: str) -> float:
    """정답(reference) 토큰이 답변(prediction)에 얼마나 담겼는지(포함 비율).
    조사/어미 부착·프레이밍 문장에 강건 — 각 gold 토큰이 pred 토큰 하나에 커버되면 인정."""
    ref = _tokenize(reference)
    if not ref:
        return 0.0
    pred = _tokenize(prediction)
    covered = sum(1 for g in ref if any(_covers(p, g) for p in pred))
    return covered / len(ref)


def answer_match(prediction: str, reference: str) -> float:
    """정답 매칭 점수(규칙 기반 tier1). 짧은 정답은 recall(포함) 위주, 긴 정답은 token F1.
    숫자·시간 포맷, 조사·어미, 프레이밍 문장 같은 표면형 차이에 강건하다.
    reference 없으면(정답 미보유) 0.0.

    [트레이드오프] 짧은 정답에서 recall 위주라, gold 를 담고도 다른 답을 말하는 드문 경우를
    정답으로 볼 수 있다(false negative↓·false positive↑). 그런 애매 케이스는 tier3 의미
    유사도 게이트(signals._answer_correct)가 승급 검증한다."""
    ref = _tokenize(reference)
    if not ref:
        return 0.0
    f1 = token_f1(prediction, reference)
    if len(ref) <= _SHORT_REF_MAX_TOKENS:
        return max(f1, token_recall(prediction, reference))
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
