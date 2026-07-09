"""
agents/optimize/config_mapper.py

이번 MVP에서 주의할 점:
  - 이 모듈은 판단기가 아니라 변환기다. rules.py에서 나온 prescription id를
    AgentDoctor 내부 표준 ConfigPatch 후보와 search space 값으로 바꾼다.
  - label -> prescription 선택 책임은 rules.py에 남긴다.
  - RAGBuilder config key 같은 adapter 전용 이름은 이 파일에 넣지 않는다.
  - 지원하지 않는 capability, 알 수 없는 prescription, constraint로 인해
    후보가 모두 제거된 경우, 서로 충돌하는 prescription은 복잡하게 해결하지
    않고 warning/skipped로 처리한다.

이번 MVP에서 구현한 범위:
  - rules.py에서 현재 쓰는 retrieval/chunking/context 계열 prescription id:
    increase_top_k, decrease_top_k, dynamic_top_k, enable_reranker,
    disable_reranker, widen_rerank_candidates, enable_hybrid,
    context_compression, increase_chunk_size, decrease_chunk_size,
    shrink_chunk_size, adjust_chunk_size, increase_chunk_overlap,
    switch_chunking_strategy.
  - 현재 config 값을 반영한 숫자 후보 생성.
  - 동일 patch 중복 제거.
  - 단순 min/max/allowed constraint와 명시적 capability skip 처리.

나중으로 미룬 범위:
  - 여러 label 사이의 우선순위 계산과 충돌 해결.
  - adapter별 search space 변환.
  - metric/cost를 고려한 후보 ranking.
  - embedding/reranker 모델 교체를 위한 model catalog 처리.
  - history 기반 blacklist와 재시도 억제.
  - patch를 AgentDoctorState에 실제 적용하는 로직.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from agents.optimize.schemas import (
    ConfigMappingResult,
    ConfigPatch,
    ConfigTarget,
    SkippedPrescription,
)


CandidateKind = Literal[
    "static",
    "increase_top_k",
    "decrease_top_k",
    "increase_chunk_size",
    "decrease_chunk_size",
    "adjust_chunk_size",
    "increase_chunk_overlap",
    "widen_rerank_candidates",
    "switch_chunking_strategy",
]


# PrescriptionSpec.kind에 들어갈 값은 "후보값 생성 방식"을 뜻한다.
# prescription id별로 if/else를 크게 벌리지 않기 위해, id -> spec 테이블에는
# target path와 후보 생성 방식을 선언만 해두고 실제 값 생성은 _candidate_values가 맡는다.
@dataclass(frozen=True)
class PrescriptionSpec:
    target_path: str
    kind: CandidateKind
    reindex_required: bool = False
    target: ConfigTarget = "index_config"
    capability: str | None = None
    static_values: tuple[Any, ...] = ()


# rules.py에서 내려오는 prescription id를 AgentDoctor 내부 표준 config path로 매핑한다.
# 이 테이블은 외부 모듈이 직접 읽는 public contract가 아니라 config_mapper 내부 구현이다.
# 외부로 전달해야 하는 prescription id 목록은 ConfigPatch.metadata["prescription_ids"]에 싣는다.
PRESCRIPTION_SPECS: dict[str, PrescriptionSpec] = {
    "increase_top_k": PrescriptionSpec(
        target_path="retriever.top_k",
        kind="increase_top_k",
        capability="retriever.top_k",
    ),
    "dynamic_top_k": PrescriptionSpec(
        target_path="retriever.top_k",
        kind="increase_top_k",
        capability="retriever.top_k",
    ),
    "decrease_top_k": PrescriptionSpec(
        target_path="retriever.top_k",
        kind="decrease_top_k",
        capability="retriever.top_k",
    ),
    "enable_reranker": PrescriptionSpec(
        target_path="reranker.enabled",
        kind="static",
        capability="reranker",
        static_values=(True,),
    ),
    "disable_reranker": PrescriptionSpec(
        target_path="reranker.enabled",
        kind="static",
        capability="reranker",
        static_values=(False,),
    ),
    "widen_rerank_candidates": PrescriptionSpec(
        target_path="reranker.candidate_count",
        kind="widen_rerank_candidates",
        capability="reranker",
    ),
    "enable_hybrid": PrescriptionSpec(
        target_path="retriever.search_type",
        kind="static",
        capability="hybrid_search",
        static_values=("hybrid",),
    ),
    "context_compression": PrescriptionSpec(
        target_path="context.compression.enabled",
        kind="static",
        capability="context_compression",
        static_values=(True,),
    ),
    "increase_chunk_size": PrescriptionSpec(
        target_path="chunker.chunk_size",
        kind="increase_chunk_size",
        capability="chunking",
        reindex_required=True,
    ),
    "decrease_chunk_size": PrescriptionSpec(
        target_path="chunker.chunk_size",
        kind="decrease_chunk_size",
        capability="chunking",
        reindex_required=True,
    ),
    "shrink_chunk_size": PrescriptionSpec(
        target_path="chunker.chunk_size",
        kind="decrease_chunk_size",
        capability="chunking",
        reindex_required=True,
    ),
    "adjust_chunk_size": PrescriptionSpec(
        target_path="chunker.chunk_size",
        kind="adjust_chunk_size",
        capability="chunking",
        reindex_required=True,
    ),
    "increase_chunk_overlap": PrescriptionSpec(
        target_path="chunker.chunk_overlap",
        kind="increase_chunk_overlap",
        capability="chunking",
        reindex_required=True,
    ),
    "switch_chunking_strategy": PrescriptionSpec(
        target_path="chunker.strategy",
        kind="switch_chunking_strategy",
        capability="chunking_strategy",
        reindex_required=True,
        static_values=("recursive_sentence",),
    ),
}


# mapper가 만들어내는 patch는 "retriever.top_k" 같은 내부 표준 path를 사용한다.
# 반면 현재 AgentDoctorState.index_config는 아직 "chunk_size", "use_hybrid"처럼 flat한 key를 쓴다.
# CONFIG_READ_PATHS는 후보값을 만들기 위해 현재 config 값을 읽을 때만 사용하는 조회 순서다.
# 예: "chunker.chunk_size"를 읽을 때 nested key가 없으면 기존 flat key "chunk_size"를 확인한다.
# 이 테이블은 output path를 바꾸지 않는다. 생성되는 patch key는 항상 PRESCRIPTION_SPECS.target_path다.
CONFIG_READ_PATHS: dict[str, tuple[str, ...]] = {
    "retriever.top_k": ("retriever.top_k", "top_k"),
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
    "chunker.strategy": ("chunker.strategy", "chunking_strategy"),
}


# 별도 constraints가 들어오지 않았을 때 사용하는 안전한 기본 범위다.
# optimizer가 무의미하거나 과도한 후보를 실험하지 않도록 mapper 단계에서 1차로 걸러낸다.
DEFAULT_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "retriever.top_k": {"min": 1, "max": 20},
    "reranker.candidate_count": {"min": 4, "max": 100},
    "chunker.chunk_size": {"min": 200, "max": 1500},
    "chunker.chunk_overlap": {"min": 0, "max": 300},
    "chunker.strategy": {"allowed": ["recursive_sentence"]},
}


# MVP에서는 복잡한 conflict resolution을 하지 않는다.
# 서로 반대 방향의 prescription이 동시에 들어오면 둘 다 skipped로 보내고 warning만 남긴다.
# 같은 방향 처방(increase_top_k + dynamic_top_k)은 conflict가 아니라 dedupe 대상으로 처리한다.
CONFLICT_PAIRS: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (
        frozenset({"increase_top_k", "dynamic_top_k"}),
        frozenset({"decrease_top_k"}),
    ),
    (
        frozenset({"increase_chunk_size"}),
        frozenset({"decrease_chunk_size", "shrink_chunk_size"}),
    ),
    (
        frozenset({"enable_reranker"}),
        frozenset({"disable_reranker"}),
    ),
)


def map_prescriptions_to_config(
    prescription_ids: list[str],
    current_config: dict[str, Any],
    capabilities: dict[str, Any] | None = None,
    constraints: dict[str, Any] | None = None,
) -> ConfigMappingResult:
    """rules.py의 prescription id를 실행 가능한 config patch 후보로 변환한다."""

    result = ConfigMappingResult()
    capabilities = capabilities or {}
    constraints = _merge_constraints(constraints or {})

    # 1. 서로 반대 방향의 처방이 함께 들어온 경우, mapper가 임의로 선택하지 않는다.
    #    MVP에서는 정상 입력이 아니라고 보고 해당 prescription들을 모두 skip한다.
    conflict_ids = _find_conflicting_prescriptions(prescription_ids)
    if conflict_ids:
        warning = (
            "서로 충돌하는 prescription을 건너뜀: "
            + ", ".join(sorted(conflict_ids))
        )
        result.warnings.append(warning)
        for prescription_id in sorted(conflict_ids):
            result.skipped.append(
                SkippedPrescription(
                    prescription_id=prescription_id,
                    reason="conflicting_prescription",
                )
            )

    patches: list[ConfigPatch] = []
    seen_prescriptions: set[str] = set()

    for prescription_id in prescription_ids:
        # 2. conflict로 판정된 prescription은 이미 skipped에 기록했으므로 여기서는 무시한다.
        if prescription_id in conflict_ids:
            continue

        # 3. 동일 prescription이 반복 입력되면 한 번만 처리한다.
        #    같은 patch가 여러 번 생기는 것을 줄이고, 마지막에 한 번 더 patch dedupe를 수행한다.
        if prescription_id in seen_prescriptions:
            continue
        seen_prescriptions.add(prescription_id)

        # 4. rules.py에 없는 prescription id는 mapper가 해석하지 않는다.
        #    alias를 두지 않고, source of truth인 rules.py id만 지원한다.
        spec = PRESCRIPTION_SPECS.get(prescription_id)
        if not spec:
            result.skipped.append(
                SkippedPrescription(
                    prescription_id=prescription_id,
                    reason="unsupported_prescription",
                )
            )
            continue

        # 5. 현재 pipeline이 해당 기능을 지원하지 않으면 실행 불가능한 처방으로 skip한다.
        #    capabilities에 key가 없으면 "알 수 없음"이 아니라 MVP 기본값인 "허용"으로 본다.
        supported, reason = _is_supported(spec, capabilities)
        if not supported:
            result.skipped.append(
                SkippedPrescription(
                    prescription_id=prescription_id,
                    reason=reason or "unsupported_capability",
                    target=spec.target_path,
                )
            )
            continue

        # 6. 현재 config 값을 기준으로 후보값을 만들고, constraints로 범위를 좁힌다.
        #    예: top_k=4 + increase_top_k -> [6, 8, 10].
        values = _candidate_values(prescription_id, spec, current_config, constraints)
        values = _filter_candidate_values(spec.target_path, values, constraints, current_config)

        # 7. constraints를 적용한 뒤 남는 값이 없으면 실행 후보를 만들 수 없다.
        if not values:
            result.skipped.append(
                SkippedPrescription(
                    prescription_id=prescription_id,
                    reason="no_valid_candidate_values",
                    target=spec.target_path,
                )
            )
            continue

        current_value = get_current_value(current_config, spec.target_path)
        executable_values = [
            value for value in values if not _values_equal(current_value, value)
        ]

        # 8. 현재값과 같은 후보는 실행할 필요가 없다.
        #    search_space와 patches가 서로 다른 후보 집합을 가리키지 않도록 여기서 같이 제거한다.
        if not executable_values:
            result.skipped.append(
                SkippedPrescription(
                    prescription_id=prescription_id,
                    reason="already_satisfied",
                    target=spec.target_path,
                    metadata={"current_value": current_value},
                )
            )
            continue

        # 9. optimizer/adapter가 search space로 볼 수 있게 path별 후보값을 모은다.
        result.search_space.setdefault(spec.target_path, [])
        result.search_space[spec.target_path].extend(executable_values)

        # 10. 각 후보값은 독립적인 ConfigPatch 후보가 된다.
        #     reporter 등 후속 단계가 내부 spec table을 import하지 않도록 prescription_ids는 metadata에 싣는다.
        for value in executable_values:
            patches.append(
                ConfigPatch(
                    changes={spec.target_path: value},
                    target=spec.target,
                    reindex_required=spec.reindex_required,
                    metadata={"prescription_ids": [prescription_id]},
                )
            )

    # 11. 여러 prescription이 같은 target/value를 만들 수 있으므로 마지막에 한 번 더 중복 제거한다.
    result.search_space = {
        path: _dedupe_values(values) for path, values in result.search_space.items()
    }
    result.patches = _dedupe_patches(patches)
    result.metadata["input_count"] = len(prescription_ids)
    result.metadata["patch_count"] = len(result.patches)
    return result


def get_current_value(
    current_config: dict[str, Any],
    path: str,
    default: Any | None = None,
) -> Any:
    """nested 또는 flat config dict에서 표준 path 값을 읽는다."""

    # CONFIG_READ_PATHS에 정의된 순서대로 현재 config를 조회한다.
    # current_config가 이미 nested 표준 path를 쓰면 그대로 읽고, 아니면 기존 flat key를 fallback으로 읽는다.
    for candidate_path in CONFIG_READ_PATHS.get(path, (path,)):
        found, value = _read_path(current_config, candidate_path)
        if found:
            # 기존 state.index_config의 use_hybrid는 bool이지만, mapper 표준 output은 search_type 문자열이다.
            # 현재값 비교가 자연스럽게 되도록 읽는 순간 "hybrid"/"dense"로 변환한다.
            if path == "retriever.search_type" and candidate_path == "use_hybrid":
                return "hybrid" if bool(value) else "dense"
            return value
    return default


def _candidate_values(
    prescription_id: str,
    spec: PrescriptionSpec,
    current_config: dict[str, Any],
    constraints: dict[str, dict[str, Any]],
) -> list[Any]:
    """prescription 종류와 현재 config 값을 기준으로 변경 후보값 목록을 만든다."""

    # prescription별 후보값 생성 규칙을 한 곳에 모은다.
    # 여기서는 값 후보만 만들고, min/max/allowed 같은 제약 적용은 _filter_candidate_values에서 처리한다.
    current_value = get_current_value(current_config, spec.target_path)

    # enable/disable처럼 목표값이 고정된 처방은 spec.static_values를 그대로 후보로 쓴다.
    if spec.kind == "static":
        return list(spec.static_values)

    # chunking strategy는 constraints.allowed가 있으면 그 목록에서 현재 strategy를 제외하고 시도한다.
    # allowed가 없으면 rules.py에서 의도한 기본 전략인 recursive_sentence만 사용한다.
    if spec.kind == "switch_chunking_strategy":
        allowed = constraints.get(spec.target_path, {}).get("allowed")
        if allowed:
            current = get_current_value(current_config, spec.target_path)
            return [value for value in allowed if value != current]
        return list(spec.static_values)

    # 검색 결과가 부족한 경우: 작은 폭으로 top_k를 늘려 후보를 만든다.
    if spec.kind == "increase_top_k":
        current = _int_or_default(current_value, 4)
        return [current + 2, current + 4, current + 6]

    # context가 너무 긴 경우: top_k를 줄여 전달 context 양을 낮춘다.
    if spec.kind == "decrease_top_k":
        current = _int_or_default(current_value, 4)
        return [current - 2, current - 3]

    # chunk가 너무 잘게 쪼개졌다고 판단된 경우: chunk_size를 키운다.
    if spec.kind == "increase_chunk_size":
        current = _int_or_default(current_value, 512)
        return [current + 200, current + 400]

    # chunk 하나가 너무 커서 의미 단위가 흐려진 경우: chunk_size를 줄인다.
    if spec.kind == "decrease_chunk_size":
        current = _int_or_default(current_value, 512)
        return [current - 150, current - 250]

    # rules.py의 adjust_chunk_size는 방향이 확정되지 않은 처방이므로 양방향 후보를 함께 만든다.
    if spec.kind == "adjust_chunk_size":
        current = _int_or_default(current_value, 512)
        return [current - 150, current + 200, current + 400]

    # gold chunk가 경계에서 잘리는 경우를 줄이기 위해 overlap 후보를 늘린다.
    if spec.kind == "increase_chunk_overlap":
        current = _int_or_default(current_value, 50)
        return [current + 25, current + 50, current + 100]

    # reranker를 켰지만 recall이 낮은 경우, reranker에 넘기는 후보 수를 늘린다.
    # 현재값이 없으면 top_k 기반의 보수적 기본값을 사용한다.
    if spec.kind == "widen_rerank_candidates":
        current = _int_or_default(current_value, 0)
        top_k = _int_or_default(get_current_value(current_config, "retriever.top_k"), 4)
        base = current if current > 0 else max(top_k * 5, 20)
        return [base + 10, base + 20]

    # 이후 추가될 spec을 위한 방어적 fallback.
    return []


def _filter_candidate_values(
    path: str,
    values: list[Any],
    constraints: dict[str, dict[str, Any]],
    current_config: dict[str, Any],
) -> list[Any]:
    """생성된 후보값에서 constraints를 벗어나는 값을 제거한다."""

    # 후보 생성 규칙은 일부러 단순하게 두고, 실제 실행 가능한 범위 검사는 여기서 한 번에 처리한다.
    # constraints는 canonical path와 flat key 모두 받을 수 있고 _merge_constraints에서 표준 path로 정규화된다.
    rule = constraints.get(path, {})
    allowed = rule.get("allowed")
    minimum = rule.get("min")
    maximum = rule.get("max")

    if path == "chunker.chunk_overlap":
        # overlap은 절대값 max뿐 아니라 chunk_size에 대한 상대 비율도 제한한다.
        # 너무 큰 overlap은 chunk 수를 과도하게 늘리고 거의 중복 chunk를 만들 수 있다.
        chunk_size = _int_or_default(get_current_value(current_config, "chunker.chunk_size"), 512)
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
    return _dedupe_values(filtered)


def _is_supported(
    spec: PrescriptionSpec,
    capabilities: dict[str, Any],
) -> tuple[bool, str | None]:
    """현재 pipeline capabilities로 prescription target을 실행할 수 있는지 확인한다."""

    # capabilities는 adapter/pipeline이 실제로 제공하는 기능 목록이다.
    # key가 명시적으로 False일 때만 skip하고, key가 없으면 MVP 기본값으로 허용한다.
    if not spec.capability:
        return True, None
    if spec.capability in capabilities and not bool(capabilities[spec.capability]):
        return False, "unsupported_capability"
    return True, None


def _find_conflicting_prescriptions(prescription_ids: list[str]) -> set[str]:
    """동시에 실행하면 의미가 충돌하는 prescription id 집합을 찾는다."""

    # "증가"와 "감소", "enable"과 "disable"처럼 동시에 적용하면 의미가 충돌하는 처방을 찾는다.
    # 어떤 쪽을 우선할지는 planner/optimizer의 책임이므로 mapper는 양쪽 모두 skipped 처리한다.
    ids = set(prescription_ids)
    conflicting: set[str] = set()
    for left_group, right_group in CONFLICT_PAIRS:
        left_matched = left_group & ids
        right_matched = right_group & ids
        if left_matched and right_matched:
            conflicting.update(left_matched)
            conflicting.update(right_matched)
    return conflicting


def _merge_constraints(
    constraints: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """기본 constraints와 입력 constraints를 표준 path 기준으로 병합한다."""

    # 사용자/adapter가 넘긴 constraints를 DEFAULT_CONSTRAINTS 위에 덮어쓴다.
    # "top_k" 같은 flat key로 들어온 constraint도 canonical path로 정규화한다.
    merged: dict[str, dict[str, Any]] = {
        path: dict(rule) for path, rule in DEFAULT_CONSTRAINTS.items()
    }
    for path, rule in constraints.items():
        canonical_path = _canonical_path(path)
        if isinstance(rule, dict):
            merged.setdefault(canonical_path, {}).update(rule)
    return merged


def _canonical_path(path: str) -> str:
    """flat key나 alias key를 mapper 내부 표준 config path로 바꾼다."""

    # constraints 입력은 "chunk_size"처럼 현재 state key로 들어올 수도 있다.
    # mapper 내부에서는 항상 "chunker.chunk_size" 같은 표준 path로 다룬다.
    for canonical, read_paths in CONFIG_READ_PATHS.items():
        if path == canonical or path in read_paths:
            return canonical
    return path


def _read_path(config: dict[str, Any], path: str) -> tuple[bool, Any]:
    """flat key 또는 dot path로 dict 값을 읽고, 존재 여부와 값을 함께 반환한다."""

    # 먼저 flat key를 그대로 확인한다. 예: {"chunk_size": 512}
    if path in config:
        return True, config[path]

    # 없으면 dot path를 nested dict로 해석한다. 예: {"chunker": {"chunk_size": 512}}
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _int_or_default(value: Any, default: int) -> int:
    """값을 int로 변환하되 실패하거나 bool이면 기본값을 반환한다."""

    # bool은 int의 subclass라 int(True) == 1이 된다.
    # config flag를 숫자 후보 생성에 잘못 쓰지 않도록 bool은 default 처리한다.
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _values_equal(left: Any, right: Any) -> bool:
    """현재값과 후보값이 같은지 비교한다."""

    # 지금은 단순 equality만 쓰지만, 추후 model id normalization 같은 비교 규칙을 넣을 수 있게 분리해둔다.
    return left == right


def _dedupe_values(values: list[Any]) -> list[Any]:
    """입력 순서를 유지하면서 중복 후보값을 제거한다."""

    # 후보값 순서는 실험 순서로 이어질 수 있으므로 set으로 바꾸지 않고 입력 순서를 보존한다.
    deduped: list[Any] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _dedupe_patches(patches: list[ConfigPatch]) -> list[ConfigPatch]:
    """동일한 config 변경을 만드는 중복 ConfigPatch를 제거한다."""

    # 서로 다른 prescription이 같은 target/value patch를 만들 수 있다.
    # patch 단위 dedupe에서는 실제 config 변화가 같은지를 기준으로 본다.
    # 단, provenance는 잃지 않도록 metadata["prescription_ids"]를 병합한다.
    deduped: list[ConfigPatch] = []
    seen: dict[tuple[Any, ...], ConfigPatch] = {}

    for patch in patches:
        key = (
            patch.target,
            tuple(sorted(patch.changes.items())),
            patch.reindex_required,
        )
        existing = seen.get(key)
        if existing:
            _merge_patch_metadata(existing, patch)
            continue
        seen[key] = patch
        deduped.append(patch)

    return deduped


def _merge_patch_metadata(target_patch: ConfigPatch, source_patch: ConfigPatch) -> None:
    """중복 patch를 합칠 때 prescription provenance metadata를 보존한다."""

    target_ids = list(target_patch.metadata.get("prescription_ids", []))
    source_ids = list(source_patch.metadata.get("prescription_ids", []))

    for prescription_id in source_ids:
        if prescription_id not in target_ids:
            target_ids.append(prescription_id)

    if target_ids:
        target_patch.metadata["prescription_ids"] = target_ids
