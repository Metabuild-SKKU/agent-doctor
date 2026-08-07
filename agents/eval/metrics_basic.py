"""
agents/eval/metrics_basic.py
[tier1] 추가 자원 없이 계산되는 측정을 모은 파일. (STEP3-1: 규칙 기반 지표)

LLM 호출도, 추가 검색 쿼리도 쓰지 않는다 — 이미 가진 답변·정답·청크 좌표만으로 계산하므로
모드 게이트 없이 FAST 부터 항상 동작한다. 담는 것:
  · 정답 매칭 지표 : char_f1 / best_window_char_f1 / answer_match / exact_match
  · 검색 지표      : recall_at_k / span_recall_at_k
  · 무응답 판별    : is_abstention
  · 진입 시 계산   : _compute_metrics (record 에 recall/f1/oracle_f1/EM 저장)
  · 청크 경계 측정 : _gold_span_boundary_analysis (원문 절대좌표 비교)

여기는 '측정'만 한다 — 임계값 판정과 라벨 부여는 diagnose 소관이다.
추가 검색이 필요한 측정은 metrics_search(tier2), LLM 이 필요한 측정은 metrics_ragas(tier3).

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

from agents.eval.types import EvalRecord
from agents.eval.metrics_common import _ctx, _cache


# ── 정규화 (KorQuAD 공식 normalize_answer) ────────────────────────
# KorQuAD 1.0 evaluate-v1.0.py 와 동일: 따옴표·괄호류 → 공백, 소문자화, ASCII 구두점 제거, 공백 정리.

_REMOVE_CHARS = "'\"《》<>〈〉()" + "‘’“”"   # remove_: 따옴표·괄호류
_REMOVE_TABLE = {ord(c): " " for c in _REMOVE_CHARS}
_PUNCT_SET = set(string.punctuation)

def _normalize(text: str) -> str:
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
    """
    KorQuAD 공식 문자 단위 F1
    - Precision = 겹친 문자 수 / 답변 문자 수
    - Recall = 겹친 문자 수 / 정답 문자 수
    - F1 = 2PR/(P+R)
    한국어는 완벽한 형태소 분석기가 없어 어절 단위 F1이 부적절 → KorQuAD가 문자 단위를 표준으로 쓴다.
    """
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


# 하위호환 별칭 — 기존 호출부(tests)가 token_f1 이름을 쓴다.
# 이제 어절이 아니라 KorQuAD 문자 단위 F1이다.
token_f1 = char_f1


def exact_match(prediction: str, reference: str) -> bool:
    """KorQuAD 공식 EM — 정규화 문자열 완전 일치."""
    return _normalize(prediction) == _normalize(reference)


# ── 정답 매칭 (KorQuAD 문자 F1 + 짧은 정답 문자-recall + 긴 정답 창 F1) ───────────
# 정답이 짧고, 답변이 문장이면 precision이 감소함
# 짧은 정답은 recall(정답 문자가 답변에 담겼나)만 보는 게 표준이다.

_SHORT_REF_MAX_CHARS = 10   # 정규화 후 정답 문자 수 이하면 '짧은 정답'으로 보고 recall 경로 허용
_CONTAINMENT_MIN = 0.9      # 짧은 정답은 recall 이 이 이상(정답이 거의 다 담김)일 때만 recall 로 통과

# 긴 서술형 정답용 창(window) 경로 — 아래 best_window_char_f1 참고.
_LONG_REF_MIN_CHARS = 30    # 정규화 후 정답이 이보다 길면 '서술형'으로 보고 창 경로 허용
_VERBOSE_RATIO_MIN = 1.3    # 답변이 정답보다 이 배수 이상 길 때만(=verbose) 창 경로를 켠다
_WINDOW_STRIDE_DIV = 8      # 창 이동 간격 = 정답 길이 / 이 값 (작을수록 촘촘·느림)


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


def best_window_char_f1(prediction: str, reference: str) -> float:
    """답변에서 '정답과 가장 잘 맞는 구간'만 떼어내 잰 문자 F1(0~1).

    왜 필요한가: char_f1 의 precision 분모가 답변 전체 길이라, 긴 서술형 gold 를 상대로 근거·
    소제목·부연을 갖춘 정답 답변(gold 의 3~5배 길이)은 정답을 다 담아도 F1 이 0.3~0.4 로
    깎인다. 이건 답이 틀린 게 아니라 채점 단위가 어긋난 것이다(KorQuAD 는 추출형 짧은
    정답용 지표) — 그래서 답변 전체가 아니라 정답 길이만큼의 창(window) 안에서 precision 을
    잰다. 창 크기를 정답 길이로 고정하므로 '길게 써서 점수를 버는' 경로는 생기지 않는다
    (분모가 답변 길이와 무관하게 len(ref) 로 고정된다).

    창을 len(ref) 로 두면 P=R 이라 F1 = (창 안에서 겹친 정답 문자)/len(ref) 로 수렴한다 —
    즉 '정답 내용이 답변의 어느 한 구간에 얼마나 응집해 있나'를 재는 값이다.
    답변이 정답보다 짧으면 창을 못 잡으므로 char_f1 을 그대로 돌려준다."""
    pred = _chars(prediction)
    ref = _chars(reference)
    if not pred or not ref:
        return 0.0
    width = len(ref)
    if len(pred) <= width:
        return char_f1(prediction, reference)

    ref_counter = Counter(ref)
    stride = max(1, width // _WINDOW_STRIDE_DIV)
    starts = list(range(0, len(pred) - width + 1, stride))
    if starts[-1] != len(pred) - width:
        starts.append(len(pred) - width)          # 끝까지 훑도록 마지막 창을 붙인다

    best = 0.0
    for start in starts:
        num_same = sum((Counter(pred[start:start + width]) & ref_counter).values())
        if num_same == 0:
            continue
        precision = num_same / width
        recall = num_same / len(ref)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def answer_match(prediction: str, reference: str) -> float:
    """정답 매칭 점수(규칙 기반 tier1). 기준은 KorQuAD 문자 단위 F1이고, 정답 길이에 따라 두
    보정 경로를 둔다. reference 없으면 0.0.

    · 짧은 정답(정규화 후 ≤10자): '정답 문자가 답변에 거의 다 담겼을 때(recall ≥ 0.9,
      containment)'에 한해 recall 로 통과시켜 완결 문장의 precision 감점을 피한다.
      recall 문턱을 둔 이유: 문턱 없이 max(f1, recall) 이면 정답 문자를 일부만 공유하는
      근접 오답(gold '145' ↔ 답 '150' → recall 0.67)도 통과해 너무 후해진다. recall 을 '거의 완전
      포함'으로 제한하면 verbose 정답(recall≈1.0, '332cm입니다')은 살리고 near-miss 오답은 char_f1 로 떨어진다.
    · 긴 서술형 정답(≥30자)에 답변이 1.3배 이상 길면: 창 F1(best_window_char_f1)을 함께 보고
      큰 값을 쓴다. 답변 전체를 precision 분모로 쓰면 정답을 다 담은 서술형 답변이 구조적으로
      0.3~0.4 로 깎여, 맞은 답이 실패로 판정되고 C그룹(context) 원인으로 오진되기 때문이다.
      창 크기가 len(ref) 로 고정돼 '길게 쓰면 이득'은 생기지 않는다.

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
    pred = _chars(prediction)
    if len(ref) >= _LONG_REF_MIN_CHARS and len(pred) >= len(ref) * _VERBOSE_RATIO_MIN:
        return max(f1, best_window_char_f1(prediction, reference))
    return f1


_SAFE_TERM_VARIANTS = (
    ("자산총계", "총자산"),
    ("부채총계", "총부채"),
    ("자본총계", "총자본"),
    ("당분기말", "분기말"),
    ("전기말", "직전 기말"),
    ("금 액", "금액"),
    ("비 율", "비율"),
)


def gold_answer_variants(reference: str) -> list[str]:
    """Build safe reference variants for lexical matching.

    사실을 보존하는 용어 별칭만 만든다 — 숫자 하나 겹친다고 오답이 통과하면 안 된다.

    표기 변형(표 마크업 `|`·`:` 제거, 숫자 콤마 제거)은 만들지 않는다: _normalize 가 ASCII
    구두점을 지우고 _chars 가 공백을 버리므로 원본과 문자 멀티셋이 완전히 같아져 점수를
    바꿀 수 없다(실측: 표 gold 0.7027=0.7027, 창 경로 0.75=0.75). 만들면 채점만 늘고
    variant_count 도 실제 신호를 과대 표시한다.
    """
    base = (reference or "").strip()
    if not base:
        return []

    variants: list[str] = []

    def add(value: str) -> None:
        value = " ".join((value or "").strip().split())
        if value and value not in variants:
            variants.append(value)

    add(base)

    for left, right in _SAFE_TERM_VARIANTS:
        if left in base:
            add(base.replace(left, right))
        if right in base:
            add(base.replace(right, left))

    # Keep the candidate set small and auditable.
    return variants[:8]


def best_answer_match(prediction: str, reference: str) -> tuple[float, str, float, int]:
    """Return best lexical match over safe gold-answer variants.

    Returns: best_score, best_variant, raw_score_against_original, variant_count.
    best 는 관측 전용이다 — 게이트·점수가 쓰는 값은 raw 다(_compute_metrics 참고).
    """
    variants = gold_answer_variants(reference)
    if not variants:
        return 0.0, "", 0.0, 0
    scored = [(answer_match(prediction, variant), variant) for variant in variants]
    best_score, best_variant = max(scored, key=lambda item: item[0])
    return best_score, best_variant, scored[0][0], len(variants)


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
    """검색된 청크 좌표가 gold span 을 얼마나 덮는지 — span 별 **문자 커버리지 비율**의 평균.

    청크 좌표가 없어 계산할 수 없는 legacy 환경에서는 None 을 반환해 호출부가 기존
    chunk-id Recall 로 폴백하게 한다.

    ## 왜 부분점수인가 (2026-08-07 변경)

    예전에는 span 을 **빈틈없이 다 덮어야** 1, 아니면 0 인 이진 판정이었다. 그 판정이
    "정답을 맞혔는데 검색 실패로 집계되는" 대량 오탐을 만들었다 — 실측
    (`output/logs/corpus_20260804_103059.txt`) 30문항 중 **14건(47%)** 이 recall=0 이고,
    그중 다수가 골드 청크 2~4개를 전부 top-k 에 넣어야만 1점이 되는 구조적 불가능이었다.

        missed_gold_ranks=[chunk_015:8, chunk_014:9, chunk_016:12, chunk_013:17]
        → 청크 4개를 전부 top-5 안에 넣어야 1점

    RAGChecker(NeurIPS 2024)가 골드를 원자 단위로 쪼개 "덮은 것 / 전체" 로 재는 것과
    같은 취지다. 다만 여기서 쪼갤 대상은 답변 텍스트가 아니라 **골드 구간의 문자**라
    LLM 이 필요 없다 — 좌표 산수로 끝난다.

    ## 계산

        span 하나  : 검색 청크들과의 교집합 **합집합** 길이 / span 길이
        전체       : 위 비율들의 평균

    합집합을 쓰는 이유는 청킹 overlap 때문이다. 인접 청크가 같은 문자를 공유하므로
    단순 합이면 비율이 1을 넘는다.

    ## 판정과의 관계 — 라벨 배정은 바뀌지 않는다

    전부 덮으면 정확히 1.0 이고(int 나눗셈이라 오차 없음), 하나도 못 덮으면 0.0 이다.
    그래서 `_recall_ok`(`>= 1`)·A슬롯 진입(`0 <= recall < 1`)의 참거짓이 이 변경으로
    뒤집히지 않는다. 달라지는 것은 **그 사이 값**뿐이다.

    ⚠️ 대신 점수는 올라간다 — `scoring.reliability_score` 가 이 값을 그대로 곱하므로
    부분 커버리지 probe 의 신뢰도가 0 에서 부분점수로 오른다. **이 변경 이전 실행과
    composite 를 직접 비교하면 개선으로 오독된다**(agents/eval/README.md 참고).
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

    ratio_sum = 0.0
    for doc_id, start, end in valid_spans:
        intersections = sorted(
            (max(start, c_start), min(end, c_end))
            for c_start, c_end in retrieved_positions.get(doc_id, [])
            if c_start < end and c_end > start
        )
        # 겹치는 구간은 합쳐서 문자를 두 번 세지 않는다. 청킹 overlap 이 켜져 있으면
        # 인접 청크가 같은 문자를 공유하므로, 합집합이 아니면 비율이 1을 넘는다.
        # intersections 가 시작 좌표로 정렬돼 있어 cursor 하나로 합집합이 계산된다.
        covered_chars = 0
        cursor = start
        for covered_start, covered_end in intersections:
            if covered_end <= cursor:
                continue                      # 이미 센 구간 안에 들어옴
            covered_chars += covered_end - max(covered_start, cursor)
            cursor = covered_end
        ratio_sum += covered_chars / (end - start)
    return ratio_sum / len(valid_spans)


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
    [추후 구현 포인트] 설계의 Non-Answer Critic(RAGAS AspectCritic)으로 정밀화 가능.
    """
    if not answer or not answer.strip():
        return True
    low = answer.lower()
    return any(m in low for m in _ABSTENTION_MARKERS)


# ══════════════════════════════════════════════════════════════════
#  진입 시 지표 계산 (diagnose 진입 시 1회 — 판정 전, 스킵 없음)
# ══════════════════════════════════════════════════════════════════

def _compute_metrics(record: EvalRecord) -> None:
    """
    STEP3-1: 규칙 지표(recall/f1/oracle_f1/EM)를 record 에 계산·저장.
    """
    gt = record.probe.ground_truth
    span_recall = span_recall_at_k(
        record.probe.gold_spans,
        record.retrieved_chunk_ids,
        _ctx.chunks,
    )
    record.recall_at_k = (
        span_recall
        if span_recall is not None
        else recall_at_k(record.probe.gold_chunk_ids, record.retrieved_chunk_ids)
    )
    record.recall_basis = "span" if span_recall is not None else "chunk"
    # answer_match: KorQuAD char-F1 (+짧은 정답 recall).
    if gt:
        score, best_variant, raw_score, variant_count = best_answer_match(record.generated_answer, gt)
        record.f1_score = raw_score
        record.raw_f1_score = raw_score
        record.best_gold_answer_f1 = score
        record.best_gold_answer = best_variant
        record.gold_answer_variant_count = variant_count
    else:
        record.f1_score = 0.0
        record.raw_f1_score = 0.0
        record.best_gold_answer_f1 = 0.0
        record.best_gold_answer = None
        record.gold_answer_variant_count = 0

    if gt and record.oracle_answer:
        score, best_variant, raw_score, _variant_count = best_answer_match(record.oracle_answer, gt)
        record.oracle_f1 = raw_score
        record.raw_oracle_f1 = raw_score
        record.best_oracle_gold_answer_f1 = score
        record.best_oracle_gold_answer = best_variant
    else:
        record.oracle_f1 = 0.0
        record.raw_oracle_f1 = 0.0
        record.best_oracle_gold_answer_f1 = 0.0
        record.best_oracle_gold_answer = None
    # KorQuAD 공식 EM — 관측용으로만 남김.
    record.exact_match = exact_match(record.generated_answer, gt) if gt else False


# ══════════════════════════════════════════════════════════════════
#  청크 경계 측정 — gold span 이 현재 청크 경계에 나뉘었나
#    LLM·추가 검색 없이 이미 가진 원문 절대좌표만 비교한다(그래서 tier1).
#    diagnose.chunking_context_mismatch 가 _gold_span_boundary_analysis 를 소비한다.
# ══════════════════════════════════════════════════════════════════

def _chunk_char_span(chunk) -> tuple[int, int] | None:
    """청크가 원문에서 차지하는 절대좌표 [start, end) 를 안전하게 읽는다. 없거나 형식 불량이면 None."""
    raw = getattr(chunk, "char_span", None)
    if raw is None and isinstance(getattr(chunk, "metadata", None), dict):
        raw = chunk.metadata.get("char_span")
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
        return None
    return raw[0], raw[1]


def _exact_probe_gold_spans(record: EvalRecord) -> list[dict]:
    """경계 진단에 쓸 수 있는 'exact' gold span 만 고른다.

    chunk_fallback 은 정확한 정답 위치를 못 찾아 청크 전체를 대신 기록한 값이라,
    그대로 쓰면 항상 '한 청크에 담김'으로 나와 경계 판정 근거가 되지 못한다.
    품질 메타데이터가 없는 기존 Probe 는 하위 호환으로 exact 취급한다.
    """
    grounding = record.probe.metadata.get("span_grounding", {})
    if not isinstance(grounding, dict):
        grounding = {}
    raw_qualities = grounding.get("span_qualities")
    qualities = raw_qualities if isinstance(raw_qualities, list) else []
    status = grounding.get("status")
    spans: list[dict] = []
    for index, span in enumerate(record.probe.gold_spans):
        if not isinstance(span, dict):
            continue
        doc_id = span.get("doc_id")
        start = span.get("start")
        end = span.get("end")
        if (
            not isinstance(doc_id, str)
            or isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            continue
        quality = qualities[index] if index < len(qualities) else None
        if quality == "chunk_fallback" or (
            quality is None and status in {"chunk_fallback", "partial"}
        ):
            continue
        spans.append({"doc_id": doc_id, "start": start, "end": end})
    return spans


def _gold_span_boundary_analysis(record: EvalRecord):
    """gold span 이 현재 인접 청크 경계에 나뉘었는지 분석한다(코퍼스 전체 좌표 기준).

    span 마다 셋 중 하나로 센다:
      contained      한 청크가 span 을 통째로 포함 → 정상
      boundary_split 단일 청크엔 안 들어가지만 인접 청크들의 합집합이 빈틈없이 덮음 → 경계에 잘림
      uncovered      합쳐도 못 덮음(틈 있음/겹치는 청크 없음) → 누락
    청크 좌표가 없는 환경은 미확정(None).
    """
    if not _ctx.chunks:
        return None

    def compute():
        spans = _exact_probe_gold_spans(record)
        if not spans:
            return None
        chunks_by_doc: dict[str, list[tuple[int, int]]] = {}
        for chunk in _ctx.chunks:
            position = _chunk_char_span(chunk)
            doc_id = getattr(chunk, "doc_id", None)
            if position is None or not isinstance(doc_id, str):
                continue
            chunks_by_doc.setdefault(doc_id, []).append(position)
        for positions in chunks_by_doc.values():
            positions.sort()

        contained_count = 0
        split_count = 0
        uncovered_count = 0
        for span in spans:
            start, end = span["start"], span["end"]
            positions = chunks_by_doc.get(span["doc_id"], [])
            if any(c_start <= start and c_end >= end for c_start, c_end in positions):
                contained_count += 1
                continue

            # span 과 겹치는 조각만 모아 좌표순으로 훑으며 빈틈 없이 이어지는지 본다.
            intersections = sorted(
                (max(start, c_start), min(end, c_end))
                for c_start, c_end in positions
                if c_start < end and c_end > start
            )
            cursor = start
            for covered_start, covered_end in intersections:
                if covered_start > cursor:   # 틈 발견 → 덮지 못함
                    break
                cursor = max(cursor, covered_end)
                if cursor >= end:
                    break
            if intersections and cursor >= end:
                split_count += 1
            else:
                uncovered_count += 1

        return {
            "span_count": len(spans),
            "contained_count": contained_count,
            "boundary_split_count": split_count,
            "uncovered_count": uncovered_count,
        }

    return _cache(record, "gold_span_boundary_analysis", compute)


def _gold_chunk_evidence_density(record: EvalRecord):
    """검색된 gold 청크 안에서 실제 근거(span)가 차지하는 비율. 낮을수록 청크가 근거보다 크다.

    청크 '사이' 노이즈(비-gold 청크)가 아니라 청크 '안'의 노이즈를 잰다 —
    chunking_underchunking 과 context_noise_interference 를 가르는 신호다.
    gold 를 안 담은 청크는 분모에서 뺀다(그건 top_k·리랭커 문제지 청크 크기 문제가 아니다).
    반환: 0~1 / 좌표·span 없으면 None.
    """
    if not _ctx.chunks:
        return None

    def compute():
        spans = _exact_probe_gold_spans(record)
        if not spans:
            return None
        by_doc: dict[str, list[tuple[int, int]]] = {}
        for span in spans:
            by_doc.setdefault(span["doc_id"], []).append((span["start"], span["end"]))

        retrieved = set(record.retrieved_chunk_ids)
        evidence = 0
        total = 0
        for chunk in _ctx.chunks:
            if getattr(chunk, "chunk_id", None) not in retrieved:
                continue
            position = _chunk_char_span(chunk)
            doc_id = getattr(chunk, "doc_id", None)
            if position is None or doc_id not in by_doc:
                continue
            c_start, c_end = position
            covered = sum(max(0, min(c_end, s_end) - max(c_start, s_start))
                          for s_start, s_end in by_doc[doc_id])
            if covered <= 0:
                continue                      # gold 를 안 담은 청크 → 분모 제외
            evidence += covered
            total += c_end - c_start
        return evidence / total if total > 0 else None

    return _cache(record, "gold_chunk_evidence_density", compute)


def _context_char_total(record: EvalRecord) -> int:
    """LLM 에 들어간 검색 context 총 길이(문자). too_long_context 판별용."""
    return sum(len(text or "") for text in record.retrieved_context)


def _gold_position_band(record: EvalRecord):
    """검색 결과 안 gold 청크의 상대 위치(0=맨 앞, 1=맨 뒤) 중 가장 가장자리에 가까운 값.

    lost_in_the_middle 판별용 — gold 가 하나라도 양끝에 있으면 '중간이라 못 봤다'가 성립하지
    않으므로 가장 유리한 위치를 대표로 쓴다.
    반환: 0~1 / 결과가 3건 미만이거나 gold 미검색이면 None(앞·중간·뒤가 안 갈림).
    """
    ids = record.retrieved_chunk_ids
    if len(ids) < 3:
        return None
    golds = set(record.probe.gold_chunk_ids)
    positions = [i / (len(ids) - 1) for i, cid in enumerate(ids) if cid in golds]
    if not positions:
        return None
    return min(positions, key=lambda p: min(p, 1 - p))


def _oversized_gold_spans(record: EvalRecord):
    """현재 청크로는 한 청크에 담길 수 없는 gold span 통계.

    span 길이가 최장 청크보다 길면 겹침을 아무리 늘려도 한 청크에 담기지 않는다 —
    청크 i 가 [i·(c-o), i·(c-o)+c) 를 덮으므로 담김 가능 조건이 L <= c 이기 때문이다(기하 사실).
    그래서 처방이 overlap 이 아니라 chunk_size 증가다 — chunking_overchunking 판별.
    반환: {"oversized_count", "max_chunk_len", "max_span_len"} / 재료 없으면 None.
    """
    if not _ctx.chunks:
        return None

    def compute():
        spans = _exact_probe_gold_spans(record)
        if not spans:
            return None
        chunk_lengths = []
        for chunk in _ctx.chunks:
            position = _chunk_char_span(chunk)
            if position is not None:
                chunk_lengths.append(position[1] - position[0])
        if not chunk_lengths:
            return None
        max_chunk = max(chunk_lengths)
        span_lengths = [span["end"] - span["start"] for span in spans]
        return {
            "oversized_count": sum(1 for length in span_lengths if length > max_chunk),
            "max_chunk_len": max_chunk,
            "max_span_len": max(span_lengths),
        }

    return _cache(record, "oversized_gold_spans", compute)
