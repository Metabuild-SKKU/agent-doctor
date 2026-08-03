"""
agents/optimize/eligibility.py
"이 action 을 지금 실제로 적용할 수 있는가" 를 판정하는 공유 계층.

[왜 필요한가]
  선택 단위가 action 으로 바뀌면 **점수를 매기기 전에** 실행 가능 여부를 알아야 한다.
  실행할 수 없는 action 이 표를 받으면 실행 가능한 action 이 밀려나고(starvation),
  선택된 뒤 optimizer 에서 탈락하면 그 방문이 통째로 낭비된다.

  기존에는 이 판정이 optimizer 안에서 실행 직전에만 이뤄졌고 planner 는 알 수 없었다.
  이 모듈은 같은 정책을 planner·aggregator 도 쓸 수 있게 조합해 노출한다.

[정책의 단일 진실 원천은 optimizer 다]
  상수와 기본 검사(capability 병합, constraint 병합, 후보값 필터)는 optimizer 가
  소유하며 여기서 재정의하지 않는다. **값을 바꾸면 planner 와 optimizer 의 판정이
  갈려 "선택했는데 실행 못 하는" 상태가 생긴다.**

[판정 순서]  계획서 §4.6
  catalog 차단 → state mapping → backend 지원 → pipeline capability
  → runtime capability → baseline prerequisite → constraint → no-op 제거 → 단일 축
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents.optimize import optimizer as _optimizer
from agents.optimize.config_mapper import canonicalize_path, get_current_value
from agents.optimize.schemas import ActionDefinition

# 정책 재노출 — 호출자가 optimizer 를 직접 import 하지 않아도 되게 한다.
# 정의는 optimizer 가 소유한다(여기서 값을 새로 만들지 않는다).
BACKEND_SUPPORTED_PATHS = _optimizer.BACKEND_SUPPORTED_PATHS
DEFAULT_CAPABILITIES = _optimizer.DEFAULT_CAPABILITIES
DEFAULT_CONSTRAINTS = _optimizer.DEFAULT_CONSTRAINTS
PATH_CAPABILITIES = _optimizer.PATH_CAPABILITIES
REINDEX_PATHS = _optimizer.REINDEX_PATHS
STATE_MAPPABLE_PATHS = _optimizer.STATE_MAPPABLE_PATHS

filter_candidate_values = _optimizer.filter_candidate_values
is_capability_supported = _optimizer.is_capability_supported
merge_capabilities = _optimizer.merge_capabilities
merge_constraints = _optimizer.merge_constraints


# 실행 불가 사유. optimizer 가 쓰는 문자열과 같은 어휘를 유지해 로그·리포트에서
# 두 계층의 판정을 대조할 수 있게 한다.
REASON_CATALOG_BLOCKED = "catalog_blocked"
REASON_STATE_MAPPING = "unsupported_state_mapping"
REASON_BACKEND = "unsupported_backend_path"
REASON_CAPABILITY = "unsupported_capability"
REASON_RUNTIME = "runtime_capability_unavailable"
REASON_PREREQUISITE = "prerequisite_unmet"
REASON_NO_CANDIDATE = "no_candidate_value"
# 단일 축 보장은 optimizer._prepare_search_space 가 소유한다. 여기서 같은 검사를
# 다시 구현하면 이 모듈의 전제("정책을 재정의하지 않는다")를 스스로 어긴다 —
# 사유 문자열만 공유해 두 계층의 보고가 갈리지 않게 한다.
REASON_MULTI_AXIS = "multi_axis_search_space"


@dataclass
class ActionEligibility:
    """한 action 의 실행 가능 판정 결과."""

    action_key: str
    eligible: bool
    reason: str | None = None
    # constraint·no-op 을 통과한 최종 후보값. eligible 이면 최소 1개다.
    search_space: dict[str, list[Any]] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def candidate_count(self) -> int:
        if not self.search_space:
            return 0
        return len(next(iter(self.search_space.values())))


def _runtime_verified(path: str, runtime_capabilities: dict[str, Any] | None) -> bool:
    """runtime 이 이 경로를 실행할 수 없다고 **명시**했는지 확인한다.

    reranker 처럼 Index 가 실제 로드·추론해 봐야 아는 optional 모델이 대상이다.

    ⚠️ **정보 부재는 차단이 아니다.** runtime capability 는 Index 가 생산하므로 첫
    방문에는 비어 있다. 그때 막아 버리면 reranker 계열 action 이 한 번도 선택되지
    못하고, deferred 기록조차 남지 않아 사용자가 이유를 알 수 없다.

    최종 판정은 optimizer 가 실행 직전에 한다(같은 정책을 재검증한다). 여기서는
    "이미 못 한다고 밝혀진" 경우만 걸러 헛된 선택을 줄인다. runtime 미확인은
    품질 실패가 아니므로 호출자가 영구 blacklist 로 다루면 안 된다.
    """
    if not path.startswith("reranker."):
        return True
    if not isinstance(runtime_capabilities, dict) or not runtime_capabilities:
        return True                      # 정보 없음 → optimizer 가 판정한다
    capability = runtime_capabilities.get("reranker")
    if not isinstance(capability, dict):
        return True                      # 이 기능에 대한 관측이 없음
    return capability.get("status") == "verified"


def _prerequisites_met(
    definition: ActionDefinition,
    baseline_config: dict[str, Any],
) -> tuple[bool, str]:
    """baseline 이 이 action 의 선행 조건을 만족하는지."""
    for prerequisite in definition.prerequisites:
        if not bool(get_current_value(baseline_config, prerequisite)):
            return False, prerequisite
    return True, ""


def evaluate_action(
    definition: ActionDefinition,
    candidate_values: list[Any],
    baseline_config: dict[str, Any],
    *,
    backend: str = "rules",
    capabilities: dict[str, Any] | None = None,
    runtime_capabilities: dict[str, Any] | None = None,
    constraints: dict[str, Any] | None = None,
) -> ActionEligibility:
    """action 하나가 지금 실행 가능한지 판정하고 안전한 후보값만 남긴다.

    optimizer 가 실행 직전에 하는 검사와 같은 정책을 쓴다. 순서가 중요하다 —
    싼 검사(정적 계약)를 먼저 하고 비싼 검사(후보값 계산)를 나중에 한다.
    """
    path = canonicalize_path(definition.canonical_path)
    detail: dict[str, Any] = {"canonical_path": path}

    def reject(reason: str, **extra: Any) -> ActionEligibility:
        detail.update(extra)
        return ActionEligibility(definition.key, False, reason, {}, detail)

    # 1. catalog 가 이미 차단한 action (mapper 계약 부재 / capability off)
    if definition.is_blocked:
        return reject(
            REASON_CATALOG_BLOCKED,
            blocked_reason=definition.blocked_reason,
            blocked_detail=definition.blocked_detail,
        )

    # 2. config_mapper 가 state 로 바꿀 수 있는 경로인가
    if path not in STATE_MAPPABLE_PATHS:
        return reject(REASON_STATE_MAPPING)

    # 3. 이 backend 가 이 축을 다룰 수 있는가
    if path not in BACKEND_SUPPORTED_PATHS.get(backend, set()):
        return reject(REASON_BACKEND, backend=backend)

    # 4. 사용자 pipeline 이 이 기능을 실제로 소비하는가
    supported, reason = is_capability_supported(definition.capability, capabilities)
    if not supported:
        return reject(reason or REASON_CAPABILITY, capability=definition.capability)

    # 5. runtime 이 실제 실행 가능함을 확인해 줬는가 (optional 모델)
    if not _runtime_verified(path, runtime_capabilities):
        return reject(REASON_RUNTIME)

    # 6. baseline 이 선행 조건을 만족하는가
    met, unmet = _prerequisites_met(definition, baseline_config)
    if not met:
        return reject(REASON_PREREQUISITE, unmet_prerequisite=unmet)

    # 7. constraint 적용 (allowed / min / max)
    filtered = filter_candidate_values(
        path, list(candidate_values), baseline_config, constraints
    )

    # 8. 현재값과 같은 no-op 제거.
    #    ⚠️ filter_candidate_values 는 constraint 만 본다. no-op 제거는 optimizer 의
    #    _prepare_search_space 에 있으므로 같은 정책을 여기서도 적용해야 한다.
    #    이 단계가 빠지면 boolean 축의 자동 배타가 깨진다 — 리랭커가 꺼져 있는데도
    #    disable 이 후보로 남아 enable 과 경쟁하고, 2:1 지지에서 충돌 보류에 걸려
    #    그 축이 영구히 닫힌다.
    current_value = get_current_value(baseline_config, path)
    filtered = [value for value in filtered if value != current_value]

    detail["filtered_from"] = len(candidate_values)
    detail["current_value"] = current_value
    if not filtered:
        return reject(REASON_NO_CANDIDATE)

    return ActionEligibility(
        action_key=definition.key,
        eligible=True,
        reason=None,
        search_space={path: filtered},
        detail=detail,
    )


def is_sweepable(path: str, backend: str = "internal") -> bool:
    """여러 후보를 한 study 에서 비교할 수 있는 축인지."""
    return canonicalize_path(path) in BACKEND_SUPPORTED_PATHS.get(backend, set())
