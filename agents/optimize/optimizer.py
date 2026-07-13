"""
agents/optimize/optimizer.py

OptimizationRequest를 실행 가능한 범위로 제한하고, backend adapter 실행을
조율하는 계층이다.

현재 파일에는 config_mapper.py에 섞여 있던 optimizer 성격의 정책만 옮겨 두었다.
planner가 search_space/request 생성을 맡고, config_mapper가 state config 적용을
맡으면 이 파일은 capability/constraint 검증과 backend 선택을 담당한다.
"""
from __future__ import annotations

from typing import Any

from agents.optimize.config_mapper import canonicalize_path, get_current_value


# optimizer가 후보값을 실험하기 전에 적용하는 안전 범위다.
# planner가 후보를 만들더라도 여기서 과도하거나 불가능한 값을 한 번 더 거른다.
DEFAULT_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "retriever.top_k": {"min": 1, "max": 20},
    "reranker.candidate_count": {"min": 4, "max": 100},
    "chunker.chunk_size": {"min": 200, "max": 1500},
    "chunker.chunk_overlap": {"min": 0, "max": 300},
    "chunker.strategy": {"allowed": ["recursive_sentence"]},
}


# 현재 AgentDoctor pipeline이 기본적으로 지원한다고 보는 기능 목록이다.
# 명시 입력이 없을 때는 보수적으로 비지원 기능을 False로 둔다.
DEFAULT_CAPABILITIES: dict[str, bool] = {
    "retriever.top_k": True,
    "hybrid_search": True,
    "chunking": True,
    "reranker": False,
    "context_compression": False,
    "chunking_strategy": False,
}


# capability 정책 -----------------------------------------------------------
def merge_capabilities(capabilities: dict[str, Any] | None = None) -> dict[str, Any]:
    """호출자가 넘긴 capability를 보수적 기본값 위에 병합한다."""

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
    """현재 optimizer/backend가 특정 capability를 지원하는지 판단한다."""

    if not capability:
        return True, None

    merged = merge_capabilities(capabilities)
    if not bool(merged.get(capability, False)):
        return False, "unsupported_capability"
    return True, None


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
        if isinstance(value, (int, float)):
            if minimum is not None and value < minimum:
                continue
            if maximum is not None and value > maximum:
                continue
        filtered.append(value)

    return dedupe_values(filtered)


# 후보값 정리 유틸 ----------------------------------------------------------
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
