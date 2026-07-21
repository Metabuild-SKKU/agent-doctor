"""
agents/optimize/optimizer.py

OptimizationRequest를 검증하고 실행 backend를 조율하는 계층이다.

[책임]
  - Planner가 만든 search space를 canonical 경로로 정규화한다.
  - 사용자 pipeline capability와 backend capability의 교집합만 허용한다.
  - 안전 범위와 현재 config를 기준으로 실행 가능한 후보만 남긴다.
  - 파라미터 경로에 맞는 internal 정책, rules 또는 RAGBuilder를 실행한다.
  - backend별 반환값을 OptimizationResult로 정규화한다.
  - 외부 backend 실패 시 이미 검증된 rules 후보로만 안전하게 fallback한다.

[중요한 입력 계약]
  search space 생성과 applies_when 판정은 Planner 책임이다. 이 모듈은
  PrescriptionCandidate.search_space 또는 OptimizationRequest.search_space가
  채워져 있다고 가정하며, "increase", "decrease", "upgrade" 같은 symbolic
  patch를 임의의 숫자나 모델명으로 해석하지 않는다.

[상태 계약]
  이 모듈은 AgentDoctorState를 읽거나 수정하지 않는다. 실제 index_config 반영은
  config_mapper와 agent.py가 담당하며, 여기서 성공한 결과도 적용 전에는
  status="proposed", improved=None을 유지한다.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from agents.optimize.config_mapper import canonicalize_path, get_current_value
from agents.optimize.schemas import (
    ConfigPatch,
    InternalAdapterResult,
    OptimizationRequest,
    OptimizationResult,
    PrescriptionCandidate,
    RAGBuilderResult,
)


BackendRunner = Callable[[OptimizationRequest], Any]


# optimizer가 후보값을 실험하기 전에 적용하는 안전 범위다.
DEFAULT_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "retriever.top_k": {"min": 1, "max": 20},
    "reranker.candidate_count": {"min": 4, "max": 100},
    "chunker.chunk_size": {"min": 200, "max": 1500},
    "chunker.chunk_overlap": {"min": 0, "max": 300},
    "chunker.strategy": {"allowed": ["recursive_sentence"]},
}


# 현재 pipeline이 실제로 변경값을 소비하는 기능만 기본 허용한다.
# RAGBuilder가 탐색할 수 있다는 사실과 사용자 pipeline이 적용할 수 있다는 사실은
# 다르므로, 외부 backend 지원 범위와 이 capability를 별도로 검사한다.
DEFAULT_CAPABILITIES: dict[str, bool] = {
    # Eval Agent가 state.index_config.top_k를 실제 검색 개수로 사용한다.
    # (Index도 청크 metadata에 기록한다 — 소비처가 확인돼 허용으로 전환.)
    "retriever.top_k": True,
    "hybrid_search": False,
    "chunking": True,
    "embedding_model": False,
    "reranker": False,
    "context_compression": False,
    "chunking_strategy": False,
}


# canonical config 경로가 요구하는 사용자 pipeline capability다.
PATH_CAPABILITIES: dict[str, str] = {
    "retriever.top_k": "retriever.top_k",
    "retriever.search_type": "hybrid_search",
    "reranker.enabled": "reranker",
    "reranker.candidate_count": "reranker",
    "context.compression.enabled": "context_compression",
    "chunker.chunk_size": "chunking",
    "chunker.chunk_overlap": "chunking",
    "chunker.strategy": "chunking_strategy",
    "embedding.model": "embedding_model",
}


# 현재 config_mapper가 실제 state.index_config로 변환할 수 있는 경로다.
# downstream capability가 추가되더라도 mapper 계약이 함께 갱신되기 전에는 적용하지 않는다.
STATE_MAPPABLE_PATHS: set[str] = {
    "retriever.top_k",
    "retriever.search_type",
    "chunker.chunk_size",
    "chunker.chunk_overlap",
    "embedding.model",
}


# backend별 실행 가능 경로다. 최종 적용 후보는 아래 지원 범위뿐 아니라
# STATE_MAPPABLE_PATHS와 사용자 pipeline capability도 모두 통과해야 한다.
BACKEND_SUPPORTED_PATHS: dict[str, set[str]] = {
    "rules": set(STATE_MAPPABLE_PATHS),
    "internal": {
        "retriever.top_k",
        "chunker.chunk_size",
        "chunker.chunk_overlap",
    },
    "ragbuilder": {
        "retriever.top_k",
        "retriever.search_type",
        "reranker.enabled",
        "chunker.chunk_size",
        "chunker.chunk_overlap",
        "chunker.strategy",
    },
}


REINDEX_PATHS: set[str] = {
    "embedding.model",
    "chunker.chunk_size",
    "chunker.chunk_overlap",
    "chunker.strategy",
}


# 임베딩 모델 교체는 벡터 차원이 바뀔 수 있어, Qdrant 컬렉션을 안전하게 재생성하도록
# 이 부수 키를 함께 실어 보내야 하는 경로. planner의 search_space는 단일 축(embedding.model)만
# 다루므로, 최종 ConfigPatch를 만드는 이 시점에만 추가한다(_run_rules 참고).
_RECREATE_ON_MISMATCH_PATHS: set[str] = {"embedding.model"}


# 공개 진입점 ---------------------------------------------------------------
def run(
    request: OptimizationRequest,
    *,
    backend_runners: dict[str, BackendRunner] | None = None,
) -> OptimizationResult:
    """최적화 요청을 검증하고 선택된 backend의 표준 결과를 반환한다."""

    backend = request.optimizer
    if backend not in {"rules", "internal", "ragbuilder", "autorag"}:
        return _failed_result(request, "unsupported_backend", f"지원하지 않는 backend: {backend}")

    if backend == "autorag":
        return _failed_result(
            request,
            "backend_not_implemented",
            f"아직 구현되지 않은 backend: {backend}",
        )

    try:
        prepared = _prepare_candidate(request, backend)
    except (TypeError, ValueError) as exc:
        return _failed_result(request, "invalid_search_space", str(exc))

    candidate, search_space, skipped, skip_reason = prepared
    if not search_space:
        return OptimizationResult(
            request_id=request.request_id,
            status="skipped",
            optimizer=backend,
            improved=None,
            message="실행 가능한 search space가 없어 최적화를 건너뜁니다.",
            metadata={
                "error_code": skip_reason or "missing_search_space",
                "requested_backend": backend,
                "skipped_candidates": skipped,
            },
        )

    if backend == "rules":
        return _run_rules(request, candidate, search_space, skipped=skipped)

    if backend == "internal":
        return _run_internal(
            request,
            candidate,
            search_space,
            skipped=skipped,
            backend_runners=backend_runners,
        )

    return _run_ragbuilder(
        request,
        candidate,
        search_space,
        skipped=skipped,
        backend_runners=backend_runners,
    )


# capability 정책 -----------------------------------------------------------
def merge_capabilities(capabilities: dict[str, Any] | None = None) -> dict[str, Any]:
    """호출자가 넘긴 pipeline capability를 보수적 기본값 위에 병합한다."""

    merged: dict[str, Any] = dict(DEFAULT_CAPABILITIES)
    merged.update(capabilities or {})
    return merged


def merge_constraints(
    constraints: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """호출자가 넘긴 constraint를 optimizer 기본 제약 위에 병합한다."""

    merged: dict[str, dict[str, Any]] = {
        path: dict(rule) for path, rule in DEFAULT_CONSTRAINTS.items()
    }
    for path, rule in (constraints or {}).items():
        canonical_path = canonicalize_path(path)
        if isinstance(rule, dict):
            merged.setdefault(canonical_path, {}).update(rule)
    return merged


def is_capability_supported(
    capability: str | None,
    capabilities: dict[str, Any] | None = None,
) -> tuple[bool, str | None]:
    """현재 사용자 pipeline이 특정 capability를 지원하는지 판단한다."""

    if not capability:
        return True, None

    merged = merge_capabilities(capabilities)
    if not bool(merged.get(capability, False)):
        return False, "unsupported_capability"
    return True, None


# search space 준비 ---------------------------------------------------------
def _prepare_candidate(
    request: OptimizationRequest,
    backend: str,
) -> tuple[
    PrescriptionCandidate | None,
    dict[str, list[Any]],
    list[dict[str, Any]],
    str | None,
]:
    """Planner가 만든 후보를 순서대로 검사해 첫 실행 가능 후보를 반환한다."""

    capabilities = merge_capabilities(request.metadata.get("capabilities"))
    constraints = request.metadata.get("constraints")
    skipped: list[dict[str, Any]] = []

    if request.candidates:
        for candidate in request.candidates:
            if candidate.status != "ready":
                skipped.append({"prescription_id": candidate.id, "reason": "not_ready"})
                continue

            # candidate별 search space를 우선한다. request.search_space는 Planner가
            # 후보 하나를 이미 활성화했거나 후보가 하나뿐일 때의 보조 계약이다.
            raw_search_space = candidate.search_space
            if not raw_search_space and len(request.candidates) == 1:
                raw_search_space = request.search_space
            if not raw_search_space:
                skipped.append(
                    {"prescription_id": candidate.id, "reason": "missing_search_space"}
                )
                continue

            prepared, reason = _prepare_search_space(
                raw_search_space,
                request.baseline_config,
                backend,
                capabilities,
                constraints,
            )
            if prepared:
                return candidate, prepared, skipped, None
            skipped.append({"prescription_id": candidate.id, "reason": reason})

        last_reason = skipped[-1]["reason"] if skipped else "missing_search_space"
        return None, {}, skipped, last_reason

    if not request.search_space:
        return None, {}, skipped, "missing_search_space"

    prepared, _reason = _prepare_search_space(
        request.search_space,
        request.baseline_config,
        backend,
        capabilities,
        constraints,
    )
    if not prepared:
        return None, {}, skipped, _reason
    return None, prepared, skipped, None


def _prepare_search_space(
    raw_search_space: dict[str, Any],
    baseline_config: dict[str, Any],
    backend: str,
    capabilities: dict[str, Any],
    constraints: dict[str, Any] | None,
) -> tuple[dict[str, list[Any]], str]:
    """search space를 canonical 경로로 정규화하고 안전한 후보만 남긴다."""

    if not isinstance(raw_search_space, dict):
        raise TypeError("search_space는 dict여야 합니다.")

    normalized: dict[str, list[Any]] = {}
    for path, raw_values in raw_search_space.items():
        if not isinstance(path, str) or not path:
            raise ValueError("search_space의 config 경로는 비어 있지 않은 문자열이어야 합니다.")
        if isinstance(raw_values, dict):
            raise TypeError(f"후보값은 scalar 또는 list여야 합니다: {path}")

        values = list(raw_values) if isinstance(raw_values, (list, tuple)) else [raw_values]
        canonical_path = canonicalize_path(path)
        normalized.setdefault(canonical_path, []).extend(values)

    # AgentDoctor는 효과 귀속을 위해 한 번에 처방 하나와 config 축 하나만 바꾼다.
    if len(normalized) != 1:
        return {}, "multi_axis_search_space"

    path, values = next(iter(normalized.items()))
    if path not in BACKEND_SUPPORTED_PATHS.get(backend, set()):
        return {}, "unsupported_backend_path"
    if path not in STATE_MAPPABLE_PATHS:
        return {}, "unsupported_state_mapping"

    capability = PATH_CAPABILITIES.get(path)
    supported, reason = is_capability_supported(capability, capabilities)
    if not supported:
        return {}, reason or "unsupported_capability"

    filtered = filter_candidate_values(path, values, baseline_config, constraints)
    current_value = get_current_value(baseline_config, path)
    filtered = [value for value in filtered if value != current_value]
    if not filtered:
        return {}, "no_valid_candidate_values"

    return {path: filtered}, ""


# constraint 정책 -----------------------------------------------------------
def filter_candidate_values(
    path: str,
    values: list[Any],
    current_config: dict[str, Any],
    constraints: dict[str, Any] | None = None,
) -> list[Any]:
    """optimizer 제약 조건을 적용해 후보값 목록을 걸러낸다."""

    canonical_path = canonicalize_path(path)
    merged_constraints = merge_constraints(constraints)
    rule = merged_constraints.get(canonical_path, {})
    allowed = rule.get("allowed")
    minimum = rule.get("min")
    maximum = rule.get("max")

    if canonical_path == "chunker.chunk_overlap":
        chunk_size = _int_or_default(
            get_current_value(current_config, "chunker.chunk_size"),
            512,
        )
        derived_max = int(chunk_size * 0.4)
        maximum = min(maximum, derived_max) if maximum is not None else derived_max

    filtered: list[Any] = []
    for value in values:
        if allowed is not None and value not in allowed:
            continue
        if minimum is not None or maximum is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if minimum is not None and value < minimum:
                continue
            if maximum is not None and value > maximum:
                continue
        filtered.append(value)

    return dedupe_values(filtered)


# rules backend -------------------------------------------------------------
def _run_rules(
    request: OptimizationRequest,
    candidate: PrescriptionCandidate | None,
    search_space: dict[str, list[Any]],
    *,
    skipped: list[dict[str, Any]],
    fallback_reason: str | None = None,
) -> OptimizationResult:
    """검증된 search space에서 가장 작은 변경 후보 하나를 제안한다."""

    path, values = next(iter(search_space.items()))
    value = values[0]
    prescription_id = candidate.id if candidate else None
    description = candidate.patch.description if candidate and candidate.patch else ""
    reindex_required = bool(
        path in REINDEX_PATHS
        or (candidate and candidate.patch and candidate.patch.reindex_required)
    )
    changes: dict[str, Any] = {path: value}
    if path in _RECREATE_ON_MISMATCH_PATHS:
        changes["embedding.recreate_on_mismatch"] = True
    patch = ConfigPatch(
        changes=changes,
        reindex_required=reindex_required,
        description=description or f"{path} 값을 {value!r}(으)로 변경",
        metadata={"prescription_id": prescription_id} if prescription_id else {},
    )

    metadata: dict[str, Any] = {
        "requested_backend": request.optimizer,
        "effective_backend": "rules",
        "filtered_search_space": search_space,
        "skipped_candidates": skipped,
        "propose_only": request.propose_only,
    }
    if fallback_reason:
        metadata["fallback_reason"] = fallback_reason

    return OptimizationResult(
        request_id=request.request_id,
        status="proposed",
        optimizer="rules",
        selected_candidate=candidate,
        config_patch=patch,
        improved=None,
        needs_reindex=reindex_required,
        message="검증된 rules 후보 하나를 선택했습니다.",
        metadata=metadata,
    )


# internal backend ----------------------------------------------------------
def _run_internal(
    request: OptimizationRequest,
    candidate: PrescriptionCandidate | None,
    search_space: dict[str, list[Any]],
    *,
    skipped: list[dict[str, Any]],
    backend_runners: dict[str, BackendRunner] | None,
) -> OptimizationResult:
    """파라미터 경로별 internal 실행 결과를 공통 결과로 변환한다."""

    path = next(iter(search_space))
    runner = (backend_runners or {}).get("internal")
    if runner is None:
        if path in {"chunker.chunk_size", "chunker.chunk_overlap"}:
            from agents.optimize.adapters.chunk_prescreener import run as runner
        else:
            from agents.optimize.adapters.internal_adapter import run as runner

    prepared_request = replace(
        request,
        candidates=[candidate] if candidate else [],
        search_space={key: list(values) for key, values in search_space.items()},
    )

    try:
        adapter_result = runner(prepared_request)
    except Exception as exc:  # backend 주입 경계의 마지막 안전망
        return _failed_result(
            request,
            "internal_exception",
            f"internal backend 실행 실패: {exc}",
        )

    if not isinstance(adapter_result, InternalAdapterResult):
        return _failed_result(
            request,
            "invalid_internal_result_type",
            "internal backend가 InternalAdapterResult를 반환하지 않았습니다.",
        )

    metadata = _internal_result_metadata(
        request=request,
        result=adapter_result,
        path=path,
        search_space=search_space,
        skipped=skipped,
    )

    if adapter_result.status == "needs_evaluation":
        next_config = _validate_internal_config(
            adapter_result.next_config,
            search_space,
            request.baseline_config,
            allow_baseline=False,
        )
        if next_config is None:
            return _failed_result(
                request,
                "invalid_internal_next_config",
                "internal backend의 next_config가 검증된 후보 범위 밖입니다.",
            )
        return _internal_proposed_result(
            request=request,
            candidate=candidate,
            config=next_config,
            metadata=metadata,
            message="다음 실제 Eval이 필요한 internal 후보를 선택했습니다.",
        )

    if adapter_result.status == "completed":
        best_config = _validate_internal_config(
            adapter_result.best_config,
            search_space,
            request.baseline_config,
            allow_baseline=True,
        )
        if best_config is None:
            return _failed_result(
                request,
                "invalid_internal_best_config",
                "internal backend의 best_config가 후보 또는 baseline 범위 밖입니다.",
            )

        if _is_baseline_config(best_config, request.baseline_config):
            metadata["error_code"] = "baseline_selected"
            return OptimizationResult(
                request_id=request.request_id,
                status="skipped",
                optimizer="internal",
                selected_candidate=candidate,
                best_config=best_config,
                improved=False,
                message="후보가 baseline을 개선하지 못해 현재 설정을 유지합니다.",
                metadata=metadata,
            )

        return _internal_proposed_result(
            request=request,
            candidate=candidate,
            config=best_config,
            metadata=metadata,
            message="internal 평가에서 선택한 best 후보를 제안합니다.",
        )

    if adapter_result.status == "skipped":
        metadata["error_code"] = adapter_result.metadata.get(
            "error_code",
            "internal_skipped",
        )
        return OptimizationResult(
            request_id=request.request_id,
            status="skipped",
            optimizer="internal",
            selected_candidate=candidate,
            improved=None,
            message=adapter_result.error or "internal 평가를 건너뜁니다.",
            metadata=metadata,
        )

    return OptimizationResult(
        request_id=request.request_id,
        status="failed",
        optimizer="internal",
        selected_candidate=candidate,
        improved=None,
        message=adapter_result.error or "internal 평가에 실패했습니다.",
        error=adapter_result.error or "internal 평가에 실패했습니다.",
        metadata={**metadata, "error_code": "internal_failed"},
    )


def _internal_proposed_result(
    *,
    request: OptimizationRequest,
    candidate: PrescriptionCandidate | None,
    config: dict[str, Any],
    metadata: dict[str, Any],
    message: str,
) -> OptimizationResult:
    """검증된 internal config 하나를 실제 적용 전 제안으로 만든다."""

    path, value = next(iter(config.items()))
    needs_reindex = path in REINDEX_PATHS
    if candidate and candidate.patch and candidate.patch.reindex_required:
        needs_reindex = True
    prescription_id = candidate.id if candidate else None
    patch = ConfigPatch(
        changes={path: value},
        reindex_required=needs_reindex,
        description=(
            candidate.patch.description
            if candidate and candidate.patch and candidate.patch.description
            else f"{path} 값을 {value!r}(으)로 변경"
        ),
        metadata={"prescription_id": prescription_id} if prescription_id else {},
    )
    return OptimizationResult(
        request_id=request.request_id,
        status="proposed",
        optimizer="internal",
        selected_candidate=candidate,
        config_patch=patch,
        best_config=dict(config),
        improved=None,
        needs_reindex=needs_reindex,
        message=message,
        metadata=metadata,
    )


def _internal_result_metadata(
    *,
    request: OptimizationRequest,
    result: InternalAdapterResult,
    path: str,
    search_space: dict[str, list[Any]],
    skipped: list[dict[str, Any]],
) -> dict[str, Any]:
    """다음 Optimize 방문에 필요한 internal 상태만 복사한다."""

    return {
        "requested_backend": request.optimizer,
        "effective_backend": "internal",
        "parameter_path": path,
        "filtered_search_space": {
            key: list(values) for key, values in search_space.items()
        },
        "skipped_candidates": list(skipped),
        "adapter_status": result.status,
        "adapter_warnings": list(result.warnings),
        "trial_results": list(result.trial_results),
        "study_baseline_config": dict(
            request.metadata.get("study_baseline_config", request.baseline_config)
        ),
        "objective_metric": result.objective_metric,
        "direction": result.direction,
        "best_score": result.best_score,
        "propose_only": request.propose_only,
        **dict(result.metadata),
    }


def _validate_internal_config(
    config: dict[str, Any] | None,
    search_space: dict[str, list[Any]],
    baseline_config: dict[str, Any],
    *,
    allow_baseline: bool,
) -> dict[str, Any] | None:
    """internal config가 요청 후보 또는 허용된 baseline 값인지 확인한다."""

    if not isinstance(config, dict) or len(config) != 1:
        return None
    raw_path, value = next(iter(config.items()))
    path = canonicalize_path(raw_path)
    allowed_values = search_space.get(path)
    if allowed_values is None:
        return None
    if value in allowed_values:
        return {path: value}
    if allow_baseline and value == get_current_value(baseline_config, path):
        return {path: value}
    return None


def _is_baseline_config(
    config: dict[str, Any],
    baseline_config: dict[str, Any],
) -> bool:
    """단일 축 config가 study 시작 baseline과 같은지 확인한다."""

    path, value = next(iter(config.items()))
    return value == get_current_value(baseline_config, path)


# RAGBuilder backend --------------------------------------------------------
def _run_ragbuilder(
    request: OptimizationRequest,
    candidate: PrescriptionCandidate | None,
    search_space: dict[str, list[Any]],
    *,
    skipped: list[dict[str, Any]],
    backend_runners: dict[str, BackendRunner] | None,
) -> OptimizationResult:
    """검증된 search space로 RAGBuilder를 실행하고 결과를 정규화한다."""

    runner = (backend_runners or {}).get("ragbuilder")
    if runner is None:
        from agents.optimize.adapters.ragbuilder_adapter import run as runner

    prepared_request = replace(
        request,
        candidates=[candidate] if candidate else [],
        search_space={path: list(values) for path, values in search_space.items()},
    )

    try:
        adapter_result = runner(prepared_request)
    except Exception as exc:  # 주입 runner와 외부 adapter 경계의 마지막 안전망
        return _run_rules(
            request,
            candidate,
            search_space,
            skipped=skipped,
            fallback_reason=f"ragbuilder_exception:{exc}",
        )

    if not isinstance(adapter_result, RAGBuilderResult):
        return _run_rules(
            request,
            candidate,
            search_space,
            skipped=skipped,
            fallback_reason="invalid_ragbuilder_result_type",
        )

    if adapter_result.status != "completed" or not adapter_result.best_config:
        reason = adapter_result.error or f"ragbuilder_status:{adapter_result.status}"
        return _run_rules(
            request,
            candidate,
            search_space,
            skipped=skipped,
            fallback_reason=reason,
        )

    best_config = _validate_best_config(adapter_result.best_config, search_space)
    if best_config is None:
        return _run_rules(
            request,
            candidate,
            search_space,
            skipped=skipped,
            fallback_reason="best_config_outside_search_space",
        )

    needs_reindex = any(path in REINDEX_PATHS for path in best_config)
    if candidate and candidate.patch and candidate.patch.reindex_required:
        needs_reindex = True

    return OptimizationResult(
        request_id=request.request_id,
        status="proposed",
        optimizer="ragbuilder",
        selected_candidate=candidate,
        best_config=best_config,
        improved=None,
        needs_reindex=needs_reindex,
        message="RAGBuilder가 검증된 search space에서 후보 설정을 선택했습니다.",
        metadata={
            "requested_backend": "ragbuilder",
            "effective_backend": "ragbuilder",
            "filtered_search_space": search_space,
            "skipped_candidates": skipped,
            "best_score": adapter_result.best_score,
            "adapter_status": adapter_result.status,
            "adapter_warnings": list(adapter_result.warnings),
            "adapter_metadata": dict(adapter_result.metadata),
            "propose_only": request.propose_only,
        },
    )


def _validate_best_config(
    best_config: dict[str, Any],
    search_space: dict[str, list[Any]],
) -> dict[str, Any] | None:
    """외부 best config가 요청한 단일 config 축과 후보값 안에 있는지 확인한다."""

    if not isinstance(best_config, dict) or len(best_config) != 1:
        return None
    path, value = next(iter(best_config.items()))
    canonical_path = canonicalize_path(path)
    allowed_values = search_space.get(canonical_path)
    if allowed_values is None or value not in allowed_values:
        return None
    return {canonical_path: value}


# 결과/유틸 ---------------------------------------------------------------
def _failed_result(
    request: OptimizationRequest,
    error_code: str,
    message: str,
) -> OptimizationResult:
    """실행할 수 없는 요청을 공통 실패 결과로 변환한다."""

    return OptimizationResult(
        request_id=request.request_id,
        status="failed",
        optimizer=request.optimizer,
        improved=None,
        message=message,
        error=message,
        metadata={
            "error_code": error_code,
            "requested_backend": request.optimizer,
        },
    )


def dedupe_values(values: list[Any]) -> list[Any]:
    """후보 순서를 유지하면서 중복 값을 제거한다."""

    deduped: list[Any] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _int_or_default(value: Any, default: int) -> int:
    """bool을 숫자로 취급하지 않으면서 config 값을 안전하게 int로 변환한다."""

    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
