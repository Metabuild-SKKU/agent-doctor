"""
agents/eval/metrics_chunk.py
청크 경계 측정 — gold span이 현재 청크 경계에 나뉘었는지 원문 절대좌표만으로 저비용 분석한다.

측정 레이어(metrics_*)의 청크 경계 담당 모듈. 공유 자원(_ctx: 주입된 청크, _cache: memoize)은
metrics_common 에서 가져온다. diagnose.chunking_context_mismatch 가 _gold_span_boundary_analysis 를 소비한다.
"""
from __future__ import annotations

from agents.eval.types import EvalRecord
from agents.eval.metrics_common import _ctx, _cache


def _chunk_char_span(chunk) -> tuple[int, int] | None:
    """현재 청크의 원문 절대좌표를 안전하게 읽는다."""

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
    """경계 진단에 사용할 exact gold span만 고른다.

    chunk_fallback은 기존 청크 전체를 정답 위치로 대신 기록한 값이라 경계가
    잘렸는지 판정할 근거가 될 수 없다. 품질 메타데이터가 없는 기존 Probe는
    하위 호환을 위해 exact로 취급한다.
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
    """gold span이 현재 인접 청크 경계에 나뉘었는지 저비용으로 분석한다.

    LLM이나 추가 검색을 호출하지 않고, Eval이 이미 가진 원문 절대좌표만 쓴다.
    한 청크가 span 전체를 포함하면 정상이고, 그렇지 않지만 현재 청크들의 합집합이
    span 전체를 덮으면 경계 분할로 본다. 좌표가 없는 환경은 미확정(None)이다.
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

            intersections = sorted(
                (max(start, c_start), min(end, c_end))
                for c_start, c_end in positions
                if c_start < end and c_end > start
            )
            cursor = start
            for covered_start, covered_end in intersections:
                if covered_start > cursor:
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
