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


def _tokenize(text: str) -> list[str]:
    """소문자화 + 구두점 제거 후 공백 분리. [구현 포인트] 형태소 분석기로 교체 가능."""
    if not text:
        return []
    text = _PUNCT.sub(" ", text.lower())
    return [t for t in text.split() if t]


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
    for chunk in chunks:
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
            continue
        doc_id = getattr(chunk, "doc_id", None)
        if not isinstance(doc_id, str):
            continue
        position = (raw[0], raw[1])
        all_positions.setdefault(doc_id, []).append(position)
        if getattr(chunk, "chunk_id", None) in retrieved:
            retrieved_positions.setdefault(doc_id, []).append(position)

    # gold 문서의 현재 청크 좌표 자체가 없으면 span 기반 판정이 불가능하다.
    if any(not all_positions.get(doc_id) for doc_id, _start, _end in valid_spans):
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
