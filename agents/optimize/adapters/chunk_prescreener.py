"""
agents/optimize/adapters/chunk_prescreener.py

Chunk size/overlap 후보를 DB 쓰기와 임베딩 없이 사전선별한다.

읽기: OptimizationRequest의 baseline_config, 단일 축 search_space,
      metadata.chunk_precheck_context(documents, gold_spans, chunk_strategy)
쓰기: AgentDoctorState를 수정하지 않고 InternalAdapterResult만 반환

이 결과는 검색·생성 품질의 최종 증명이 아니라 기하 조건으로 고른 예상 best다.
실제 개선 여부는 선택값 하나를 Index/Eval한 뒤 overall_score로 판정한다.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from core.schema import Document
from agents.optimize.config_mapper import canonicalize_path, get_current_value
from agents.optimize.schemas import (
    InternalAdapterResult,
    InternalTrialResult,
    OptimizationRequest,
)


Previewer = Callable[[Document, int, int, str | int], list[tuple[int, int]]]
_CHUNK_PATHS = {"chunker.chunk_size", "chunker.chunk_overlap"}


def run(
    request: OptimizationRequest,
    *,
    previewer: Previewer | None = None,
) -> InternalAdapterResult:
    """실제 청커 경계와 gold span으로 chunk 후보 하나를 선택한다."""

    try:
        path, values = _search_axis(request.search_space)
        context = _precheck_context(request)
        documents = _documents(context)
        gold_spans = _gold_spans(context, documents)
        strategy = context.get(
            "chunk_strategy",
            request.baseline_config.get("chunk_strategy", "markdown_recursive"),
        )
    except (TypeError, ValueError) as exc:
        return _result(
            request,
            status="skipped",
            error=str(exc),
            path=None,
            search_space={},
            metadata={"error_code": "missing_chunk_precheck_context"},
        )

    preview = previewer or _default_previewer
    baseline_value = get_current_value(request.baseline_config, path)
    candidates = [baseline_value, *values[: request.max_trials]]
    trials: list[InternalTrialResult] = []
    ranked: list[tuple[tuple[float, ...], Any, dict[str, Any], bool]] = []
    baseline_contained_indices: set[int] | None = None

    try:
        for index, value in enumerate(candidates):
            is_baseline = index == 0
            chunk_size, chunk_overlap = _candidate_config(
                request,
                path,
                value,
            )
            metrics = _measure_candidate(
                documents=documents,
                gold_spans=gold_spans,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strategy=strategy,
                previewer=preview,
            )
            contained_indices = set(metrics.pop("_contained_span_indices"))
            if is_baseline:
                baseline_contained_indices = set(contained_indices)
            all_indices = set(range(int(metrics["total_span_count"])))
            baseline_missing_indices = all_indices - (baseline_contained_indices or set())
            recovered = len(contained_indices & baseline_missing_indices)
            baseline_missing = len(baseline_missing_indices)
            metrics["boundary_recovery_rate"] = (
                recovered / baseline_missing if baseline_missing else 1.0
            )
            metrics["unrecovered_cut_rate"] = (
                max(0, baseline_missing - recovered) / baseline_missing
                if baseline_missing
                else 0.0
            )
            key = _ranking_key(
                path=path,
                value=value,
                baseline_value=baseline_value,
                metrics=metrics,
            )
            config = {} if is_baseline else {path: value}
            fingerprint = _fingerprint(
                {
                    **request.baseline_config,
                    path: value,
                }
            )
            trials.append(
                InternalTrialResult(
                    trial_id=f"{request.request_id}:precheck:{index}",
                    config=config,
                    score=float(metrics["full_span_containment"]),
                    metrics=metrics,
                    status="completed",
                    is_baseline=is_baseline,
                    fingerprint=fingerprint,
                    metadata={
                        "proxy_only": True,
                        "candidate_value": value,
                        "effective_chunk_size": chunk_size,
                        "effective_chunk_overlap": chunk_overlap,
                    },
                )
            )
            ranked.append((key, value, metrics, is_baseline))
    except Exception as exc:
        return _result(
            request,
            status="failed",
            error=f"chunk 사전검증 실패: {exc}",
            path=path,
            search_space={path: list(values)},
            trials=trials,
            metadata={"error_code": "chunk_precheck_failed"},
        )

    _best_key, best_value, best_metrics, best_is_baseline = max(
        ranked,
        key=lambda item: item[0],
    )
    return _result(
        request,
        status="completed",
        path=path,
        search_space={path: list(values)},
        best_config={path: best_value},
        best_score=float(best_metrics["full_span_containment"]),
        trials=trials,
        metadata={
            "proxy_only": True,
            "best_is_baseline": best_is_baseline,
            "stop_reason": "chunk_precheck_finished",
            "budget_used": len(values[: request.max_trials]),
            "max_trials": request.max_trials,
            "selection_reason": _selection_reason(path, best_value, best_metrics),
            "candidate_metrics": [
                {
                    "value": value,
                    "is_baseline": is_baseline,
                    **metrics,
                }
                for _key, value, metrics, is_baseline in ranked
            ],
        },
    )


def _search_axis(search_space: dict[str, Any]) -> tuple[str, list[Any]]:
    if not isinstance(search_space, dict) or len(search_space) != 1:
        raise ValueError("chunk 사전검증은 config 축 하나가 필요합니다.")
    raw_path, raw_values = next(iter(search_space.items()))
    path = canonicalize_path(raw_path)
    if path not in _CHUNK_PATHS:
        raise ValueError(f"chunk 사전검증이 지원하지 않는 경로입니다: {path}")
    values = list(raw_values) if isinstance(raw_values, (list, tuple)) else [raw_values]
    if not values:
        raise ValueError("chunk 후보가 없습니다.")
    return path, values


def _precheck_context(request: OptimizationRequest) -> dict[str, Any]:
    context = request.metadata.get("chunk_precheck_context")
    if not isinstance(context, dict):
        raise ValueError("chunk_precheck_context가 없어 자동 사전검증을 건너뜁니다.")
    return context


def _documents(context: dict[str, Any]) -> dict[str, Document]:
    raw_documents = context.get("documents")
    if not isinstance(raw_documents, (list, tuple)):
        raise ValueError("chunk_precheck_context.documents는 문서 목록이어야 합니다.")
    documents = {
        document.doc_id: document
        for document in raw_documents
        if isinstance(document, Document)
    }
    if not documents:
        raise ValueError("chunk 사전검증에 사용할 문서가 없습니다.")
    return documents


def _gold_spans(
    context: dict[str, Any],
    documents: dict[str, Document],
) -> list[dict[str, Any]]:
    raw_spans = context.get("gold_spans")
    if not isinstance(raw_spans, (list, tuple)):
        raise ValueError("chunk_precheck_context.gold_spans는 목록이어야 합니다.")
    spans: list[dict[str, Any]] = []
    for raw in raw_spans:
        if not isinstance(raw, dict):
            continue
        doc_id = raw.get("doc_id")
        start = raw.get("start")
        end = raw.get("end")
        document = documents.get(doc_id)
        if (
            document is None
            or isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or end > len(document.content)
        ):
            continue
        spans.append({"doc_id": doc_id, "start": start, "end": end})
    if not spans:
        raise ValueError("유효한 gold_spans가 없어 chunk 자동 사전검증을 건너뜁니다.")
    return spans


def _candidate_config(
    request: OptimizationRequest,
    path: str,
    value: Any,
) -> tuple[int, int]:
    chunk_size = int(
        value
        if path == "chunker.chunk_size"
        else get_current_value(request.baseline_config, "chunker.chunk_size", 512)
    )
    chunk_overlap = int(
        value
        if path == "chunker.chunk_overlap"
        else get_current_value(request.baseline_config, "chunker.chunk_overlap", 50)
    )
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            f"유효하지 않은 chunk 후보입니다: size={chunk_size}, overlap={chunk_overlap}"
        )
    return chunk_size, chunk_overlap


def _measure_candidate(
    *,
    documents: dict[str, Document],
    gold_spans: list[dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
    strategy: str | int,
    previewer: Previewer,
) -> dict[str, Any]:
    previews = {
        doc_id: previewer(document, chunk_size, chunk_overlap, strategy)
        for doc_id, document in documents.items()
    }
    contained = 0
    span_fit = 0
    wastes: list[int] = []
    contained_indices: list[int] = []
    for span_index, span in enumerate(gold_spans):
        span_length = span["end"] - span["start"]
        if span_length <= chunk_size:
            span_fit += 1
        containing = [
            (start, end)
            for start, end in previews.get(span["doc_id"], [])
            if start <= span["start"] and end >= span["end"]
        ]
        if containing:
            contained += 1
            contained_indices.append(span_index)
            smallest = min(end - start for start, end in containing)
            wastes.append(max(0, smallest - span_length))

    total_spans = len(gold_spans)
    total_document_chars = sum(len(document.content) for document in documents.values())
    total_chunk_chars = sum(
        end - start
        for spans in previews.values()
        for start, end in spans
    )
    chunk_count = sum(len(spans) for spans in previews.values())
    containment = contained / total_spans
    return {
        "_contained_span_indices": contained_indices,
        "contained_span_count": contained,
        "total_span_count": total_spans,
        "full_span_containment": containment,
        "boundary_cut_rate": 1.0 - containment,
        "span_fit_rate": span_fit / total_spans,
        "context_waste": (
            sum(wastes) / len(wastes) if wastes else float(total_document_chars)
        ),
        "chunk_count": chunk_count,
        "duplication_ratio": (
            max(0, total_chunk_chars - total_document_chars) / total_document_chars
            if total_document_chars
            else 0.0
        ),
    }


def _ranking_key(
    *,
    path: str,
    value: Any,
    baseline_value: Any,
    metrics: dict[str, Any],
) -> tuple[float, ...]:
    delta = abs(float(value) - float(baseline_value))
    if path == "chunker.chunk_size":
        return (
            float(metrics["full_span_containment"]),
            float(metrics["span_fit_rate"]),
            -float(metrics["context_waste"]),
            -float(metrics["chunk_count"]),
            -delta,
        )
    return (
        float(metrics["full_span_containment"]),
        -float(metrics["duplication_ratio"]),
        -float(metrics["chunk_count"]),
        -delta,
    )


def _selection_reason(path: str, value: Any, metrics: dict[str, Any]) -> str:
    if path == "chunker.chunk_size":
        return (
            f"정답 전체 포함률 {metrics['full_span_containment']:.1%}를 우선하고 "
            f"동률에서 문맥 낭비가 작은 chunk_size={value}를 선택"
        )
    return (
        f"정답 전체 포함률 {metrics['full_span_containment']:.1%}를 우선하고 "
        f"동률에서 중복량이 작은 chunk_overlap={value}를 선택"
    )


def _default_previewer(
    document: Document,
    chunk_size: int,
    chunk_overlap: int,
    strategy: str | int,
) -> list[tuple[int, int]]:
    # TODO(index-합의): private 함수 직접 import를 공개 preview API로 교체한다.
    from agents.index.agent import _chunk_document

    return [
        (draft.start, draft.end)
        for draft in _chunk_document(
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
        )
    ]


def _fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _result(
    request: OptimizationRequest,
    *,
    status: str,
    path: str | None,
    search_space: dict[str, list[Any]],
    best_config: dict[str, Any] | None = None,
    best_score: float | None = None,
    trials: list[InternalTrialResult] | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> InternalAdapterResult:
    return InternalAdapterResult(
        request_id=request.request_id,
        status=status,  # type: ignore[arg-type]
        best_config=best_config,
        best_score=best_score,
        trial_results=list(trials or []),
        objective_metric=(
            "chunk_geometry" if path in _CHUNK_PATHS else "overall_score"
        ),
        direction="maximize",
        search_space={key: list(values) for key, values in search_space.items()},
        error=error,
        metadata=dict(metadata or {}),
    )
