"""
agents/optimize/config_mapper.py

optimizer/adapter가 만든 표준 config 변경을 AgentDoctor의 실제
state.index_config 형태로 변환하고 적용한다.

살릴 가치가 있는 기존 아이디어:
- optimizer 내부에서는 "retriever.top_k" 같은 표준 경로를 쓰고,
  AgentDoctor state에는 "top_k" 같은 현재 구현 key를 쓴다.
- 지원하지 않는 key는 억지로 state에 넣지 않고 ignored/warning으로 남긴다.
- 적용 전후 ConfigDiff를 만들어 reporter/history가 같은 정보를 재사용한다.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from agents.optimize.schemas import ConfigDiff, ConfigPatch


CANONICAL_INDEX_CONFIG_KEYS: dict[str, str] = {
    "retriever.top_k": "top_k",
    # 융합 가중치. 필드는 이미 state.index_config 에 있고 retriever 가 소비하지만
    # 표준 경로가 없어 Optimize 가 건드리지 못했다(retrieval_rank_fusion_loss 처방).
    "retriever.hybrid_dense_weight": "hybrid_dense_weight",
    "retriever.mmr": "use_mmr",
    "reranker.enabled": "use_reranker",
    "reranker.candidate_count": "rerank_candidates",
    "context.compression.enabled": "context_compression",
    "chunker.chunk_size": "chunk_size",
    "chunker.chunk_overlap": "chunk_overlap",
    "chunker.strategy": "chunk_strategy",
    "embedding.model": "embedding_model",
    "embedding_model": "embedding_model",
    "embedding.recreate_on_mismatch": "recreate_collection_on_dimension_mismatch",
    # 생성(B그룹) 설정. index_config에 함께 담기므로 여기 매핑한다(재색인은 키별
    # 결정이라 generation.* 는 REINDEX_PATHS에 없어 재색인을 유발하지 않는다).
    "generation.temperature": "temperature",
    "generation.grounding_strict": "grounding_strict",
    "generation.require_citation": "require_citation",
    "generation.restate_question": "restate_question",
    "generation.completeness_mode": "completeness_mode",
    "generation.abstention_strict": "abstention_strict",
    "generation.abstention_relaxed": "abstention_relaxed",
    "generation.model": "generation_model",
}


# 한 축의 양방향이라 동시에 켜질 수 없는 index_config 플래그 쌍.
#
# index_config 는 회차 간 누적되므로 켜는 쪽만 쓰면 반대쪽 True 가 그대로 남는다. 그러면
# 나중 처방이 플래그는 바꿔도(optimizer 는 baseline 과 다르기만 하면 통과시킨다) 소비처가
# 둘 중 하나만 반영하므로 프롬프트가 안 변하고, 점수가 그대로라 롤백 + blacklist 로 빠진다
# — 그 라벨의 레버가 영구히 죽는다. 예: relax_abstention 이 유지된 뒤의
# strengthen_abstention(환각 억제)은 아무것도 못 하고 사라진다.
#
# 우선순위 대신 '마지막에 쓴 쪽이 이긴다' 로 푼다. 어느 방향을 특별대우하지 않으므로
# 양쪽 레버가 모두 살아 있고, 최신 진단이 항상 반영된다.
EXCLUSIVE_FLAG_PAIRS: tuple[tuple[str, str], ...] = (
    ("abstention_strict", "abstention_relaxed"),
)


def _resolve_exclusive_flags(mapped_changes: dict[str, Any]) -> None:
    """상호배제 쌍 중 한쪽을 켜는 변경이면 반대쪽을 같은 묶음에서 내린다(제자리 수정).

    한 묶음이 양쪽을 동시에 켜는 모순은 어느 쪽이 '나중'인지 정할 수 없어 건드리지 않는다
    (처방은 단일 축이라 정상 경로에선 안 생기고, 소비처의 백스톱이 하나만 싣는다).
    """
    for flag, opposite in EXCLUSIVE_FLAG_PAIRS:
        turned_on = [key for key in (flag, opposite) if mapped_changes.get(key) is True]
        if len(turned_on) != 1:
            continue
        mapped_changes[opposite if turned_on[0] == flag else flag] = False


# 표준 config 경로를 읽을 때 허용할 기존 flat key alias 목록이다.
# 예: "chunker.chunk_size"를 읽을 때 현재 state의 "chunk_size"도 함께 본다.
CONFIG_READ_PATHS: dict[str, tuple[str, ...]] = {
    "retriever.top_k": ("retriever.top_k", "top_k"),
    "retriever.hybrid_dense_weight": (
        "retriever.hybrid_dense_weight",
        "hybrid_dense_weight",
    ),
    "retriever.mmr": ("retriever.mmr", "use_mmr", "mmr"),
    "retriever.search_type": ("retriever.search_type", "search_type", "use_hybrid"),
    "reranker.enabled": ("reranker.enabled", "use_reranker"),
    "reranker.candidate_count": (
        "reranker.candidate_count",
        "rerank_candidates",
        "rerank_top_n",
    ),
    "context.compression.enabled": (
        "context.compression.enabled",
        "context_compression",
    ),
    "chunker.chunk_size": ("chunker.chunk_size", "chunk_size"),
    "chunker.chunk_overlap": ("chunker.chunk_overlap", "chunk_overlap"),
    "chunker.strategy": ("chunker.strategy", "chunk_strategy"),
    "embedding.model": ("embedding.model", "embedding_model"),
    "embedding.recreate_on_mismatch": (
        "embedding.recreate_on_mismatch",
        "recreate_collection_on_dimension_mismatch",
    ),
    "generation.temperature": ("generation.temperature", "temperature"),
    "generation.grounding_strict": ("generation.grounding_strict", "grounding_strict"),
    "generation.require_citation": ("generation.require_citation", "require_citation"),
    "generation.restate_question": ("generation.restate_question", "restate_question"),
    "generation.completeness_mode": ("generation.completeness_mode", "completeness_mode"),
    "generation.abstention_strict": ("generation.abstention_strict", "abstention_strict"),
    "generation.abstention_relaxed": ("generation.abstention_relaxed", "abstention_relaxed"),
    "generation.model": ("generation.model", "generation_model"),
}


# 현재 config 조회 -----------------------------------------------------------
def get_current_value(
    current_config: dict[str, Any],
    path: str,
    default: Any | None = None,
) -> Any:
    """현재 index_config에서 표준 경로 또는 flat key 값을 읽는다."""

    for candidate_path in CONFIG_READ_PATHS.get(path, (path,)):
        found, value = _read_path(current_config, candidate_path)
        if found:
            if path == "retriever.search_type" and candidate_path == "use_hybrid":
                return "hybrid" if bool(value) else "dense"
            return value
    return default


# 표준 변경을 index_config 변경으로 변환 ------------------------------------
def map_canonical_change(path: str, value: Any) -> tuple[str, Any] | None:
    """표준 config 변경 하나를 AgentDoctor index_config key/value로 바꾼다."""

    canonical_path = canonicalize_path(path)

    if canonical_path == "retriever.search_type":
        return "use_hybrid", str(value).lower() == "hybrid"

    state_key = CANONICAL_INDEX_CONFIG_KEYS.get(canonical_path)
    if state_key is None:
        return None

    return state_key, value


def map_changes_to_index_config(
    changes: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    """표준 config 변경 묶음을 state.index_config 변경 묶음으로 변환한다."""

    mapped_changes: dict[str, Any] = {}
    ignored_keys: list[str] = []
    warnings: list[str] = []

    for path, value in changes.items():
        mapped = map_canonical_change(path, value)
        if mapped is None:
            ignored_keys.append(path)
            warnings.append(f"지원하지 않는 index_config key를 무시함: {path}")
            continue

        key, mapped_value = mapped
        mapped_changes[key] = mapped_value

    _resolve_exclusive_flags(mapped_changes)

    return mapped_changes, ignored_keys, warnings


# ConfigPatch/best_config 적용 ----------------------------------------------
def apply_config_patch(
    index_config: dict[str, Any],
    patch: ConfigPatch,
    *,
    mutate: bool = True,
) -> ConfigDiff:
    """ConfigPatch를 index_config에 적용하고 적용 전후 diff를 반환한다."""

    before_config = deepcopy(index_config)
    after_config = deepcopy(index_config)
    ignored_keys: list[str] = []
    warnings: list[str] = []

    if patch.target != "index_config":
        ignored_keys = list(patch.changes)
        warnings.append(f"지원하지 않는 config target을 무시함: {patch.target}")
    else:
        mapped_changes, ignored_keys, warnings = map_changes_to_index_config(
            patch.changes
        )
        after_config.update(mapped_changes)

    if mutate:
        index_config.clear()
        index_config.update(after_config)

    return build_config_diff(
        before_config=before_config,
        after_config=after_config,
        ignored_keys=ignored_keys,
        warnings=[*warnings, *patch.warnings],
        metadata=dict(patch.metadata),
    )


def apply_best_config(
    index_config: dict[str, Any],
    best_config: dict[str, Any],
    *,
    mutate: bool = True,
) -> ConfigDiff:
    """optimizer가 반환한 best_config dict를 index_config에 적용한다."""

    patch = ConfigPatch(changes=best_config, target="index_config")
    return apply_config_patch(index_config, patch, mutate=mutate)


# 적용 전후 diff 생성 --------------------------------------------------------
def build_config_diff(
    before_config: dict[str, Any],
    after_config: dict[str, Any],
    ignored_keys: list[str] | None = None,
    warnings: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ConfigDiff:
    """두 index_config 스냅샷을 비교해 ConfigDiff를 만든다."""

    before_keys = set(before_config)
    after_keys = set(after_config)

    changed_keys = [
        key
        for key in sorted(before_keys & after_keys)
        if before_config.get(key) != after_config.get(key)
    ]
    added_keys = sorted(after_keys - before_keys)
    removed_keys = sorted(before_keys - after_keys)

    return ConfigDiff(
        before_config=before_config,
        after_config=after_config,
        changed_keys=changed_keys,
        added_keys=added_keys,
        removed_keys=removed_keys,
        ignored_keys=list(ignored_keys or []),
        warnings=list(warnings or []),
        metadata=dict(metadata or {}),
    )


# canonical 스냅샷 -----------------------------------------------------------
def canonical_config_view(config: dict[str, Any]) -> dict[str, Any]:
    """Optimize가 실제로 바꿀 수 있는 축만 표준 경로로 읽어 정렬해 돌려준다.

    같은 config를 flat key로 담았는지 표준 경로로 담았는지에 따라 baseline이 달라
    보이면 fingerprint가 흔들린다. 비교·해시의 기준을 여기 하나로 모은다.
    미지정 축은 넣지 않는다 — 키 유무 자체가 baseline 정체성의 일부다.
    """

    view: dict[str, Any] = {}
    for canonical in sorted(CONFIG_READ_PATHS):
        sentinel = object()
        value = get_current_value(config, canonical, sentinel)
        if value is not sentinel:
            view[canonical] = value
    return view


# 경로 정규화와 dict path 읽기 ----------------------------------------------
def canonicalize_path(path: str) -> str:
    """flat key나 alias를 알고 있는 표준 config 경로로 정규화한다."""

    for canonical, read_paths in CONFIG_READ_PATHS.items():
        if path == canonical or path in read_paths:
            return canonical
    return path


def _read_path(config: dict[str, Any], path: str) -> tuple[bool, Any]:
    """dict에서 flat key 또는 dot path 기반 nested key를 읽는다."""

    if path in config:
        return True, config[path]

    current: Any = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current
