"""
agents/optimize/action_catalog.py
config 변경(action)의 단일 진실 원천.

[이 파일의 역할]
  rules.py는 "어떤 label이 어떤 변경을 원하는가"를 선언하고, 이 파일은 "그 변경이
  실제로 무엇이며 실행 가능한가"를 정의한다. 같은 config 변경이 여러 label에서 다른
  이름으로 선언되던 것을 canonical action 하나로 모은다.

  reindex/cost/capability/conflict family가 여기 한 번만 정의된다. 이전에는 같은
  정보가 label마다 중복 선언돼 서로 어긋날 수 있었다.

[왜 하드코딩하지 않고 파생하는가]
  action의 실행 가능 여부는 optimizer.py의 정책(STATE_MAPPABLE_PATHS,
  PATH_CAPABILITIES, DEFAULT_CAPABILITIES, REINDEX_PATHS)이 이미 결정한다. 목록을
  손으로 적으면 그 정책과 어긋나는 순간 조용히 틀린다. 그래서 rules 선언 + optimizer
  정책에서 파생하고, **결과가 바뀌면 tests/test_action_inventory.py가 깨지도록** 했다.

[action key 규칙]  <canonical_path>:<operation>
  label이나 prescription id를 넣지 않는다. 고정값도 넣지 않는다 —
  같은 축의 값들을 key로 쪼개면 지지 label 집합이 같아져 점수가 영원히 동률이 되고
  그 축이 한 번도 선택되지 않는다(starvation). 값은 후보값으로 넘긴다.
"""
from __future__ import annotations

from typing import Any

from agents.optimize import optimizer as _optimizer
from agents.optimize import rules as _rules
from agents.optimize.config_mapper import canonicalize_path
from agents.optimize.schemas import (
    ActionBlockedReason,
    ActionDefinition,
    ActionOperation,
)


# 방향·폭이 진단 실측으로 정해지는 심볼릭 지시어의 접두사.
# 예: retriever.hybrid_dense_weight = "shift_to_favored_channel"
#     → Eval이 실측한 favored_channel로 방향이 정해진다(planner가 후보값을 만든다).
_ADJUST_PREFIX = "shift_to"

# 비용 유도. rules.py의 cost가 전부 None이라 재색인 여부로만 가른다.
# 근거 없는 숫자를 새로 만들지 않는다는 것이 전환의 전제다.
COST_REINDEX = 3.0
COST_RUNTIME = 1.0


def derive_operation(value: Any) -> ActionOperation:
    """patch 값에서 action operation을 유도한다."""
    if value in ("increase", "decrease"):
        return value
    if value is True:
        return "enable"
    if value is False:
        return "disable"
    if isinstance(value, str) and value.startswith(_ADJUST_PREFIX):
        return "adjust"
    return "replace"


def build_action_key(raw_path: str, value: Any) -> str:
    """rules.py의 (patch 경로, 값)에서 canonical action key를 만든다.

    raw_path는 flat("top_k")일 수도 canonical("retriever.top_k")일 수도 있다.
    canonicalize_path가 흡수하므로 rules.py를 전면 재작성할 필요는 없다.
    """
    return f"{canonicalize_path(raw_path)}:{derive_operation(value)}"


def _blocked(path: str) -> tuple[ActionBlockedReason | None, str]:
    """실행을 막는 사유. optimizer가 실행 직전에 쓰는 정책과 같은 기준이다."""
    if path not in _optimizer.STATE_MAPPABLE_PATHS:
        return "not_state_mappable", "config_mapper가 이 경로를 state로 변환하지 못한다"
    capability = _optimizer.PATH_CAPABILITIES.get(path)
    if capability and not _optimizer.DEFAULT_CAPABILITIES.get(capability, False):
        return "capability_off", capability
    return None, ""


def _prerequisites(path: str) -> tuple[str, ...]:
    """실행 전에 만족해야 하는 baseline 조건.

    optimizer가 이미 검사하는 것을 catalog에도 드러내, 리포트가 "왜 못 썼는지"를
    설명할 수 있게 한다.
    """
    if path == "reranker.candidate_count":
        # 후보 수 확대는 reranker가 켜져 있을 때만 의미가 있다. 꺼진 상태에서 바꾸면
        # Eval 실행 0회로 롤백돼 같은 처방이 다시 뽑히는 no-progress 순환이 생긴다.
        return ("reranker.enabled",)
    return ()


def _describe(path: str, operation: ActionOperation, values: list[Any]) -> str:
    verb = {
        "increase": "값을 늘린다",
        "decrease": "값을 줄인다",
        "enable": "기능을 켠다",
        "disable": "기능을 끈다",
        "replace": "값을 교체한다",
        "adjust": "진단이 가리키는 방향으로 조정한다",
    }[operation]
    if operation == "replace" and values:
        readable = ", ".join(str(v) for v in values)
        return f"{path} {verb} (후보: {readable})"
    return f"{path} {verb}"


def _build_catalog() -> dict[str, ActionDefinition]:
    """rules 선언 + optimizer 정책에서 catalog를 파생한다."""
    # 경로별로 값·재색인 플래그를 모은다(같은 action을 여러 label이 선언할 수 있다).
    collected: dict[str, dict[str, Any]] = {}

    for rule in _rules.LABEL_TO_PRESCRIPTIONS.values():
        if rule.get("status") != "ready":
            continue
        for prescription in rule.get("prescriptions") or []:
            reindex = bool(prescription.get("reindex"))
            for raw_path, value in (prescription.get("patch") or {}).items():
                path = canonicalize_path(raw_path)
                key = build_action_key(raw_path, value)
                entry = collected.setdefault(key, {
                    "path": path,
                    "operation": derive_operation(value),
                    "values": [],
                    "reindex": False,
                })
                if value not in entry["values"]:
                    entry["values"].append(value)
                # 같은 action을 두 label이 다른 reindex 플래그로 선언하면 안전한 쪽
                # (재색인 필요)으로 수렴시킨다.
                entry["reindex"] = entry["reindex"] or reindex

    catalog: dict[str, ActionDefinition] = {}
    for key, entry in sorted(collected.items()):
        path = entry["path"]
        operation = entry["operation"]
        reason, detail = _blocked(path)
        # 재색인 판정은 REINDEX_PATHS를 정본으로 삼고, rules 선언을 보조로 쓴다.
        reindex = path in _optimizer.REINDEX_PATHS or entry["reindex"]
        catalog[key] = ActionDefinition(
            key=key,
            canonical_path=path,
            operation=operation,
            description=_describe(path, operation, entry["values"]),
            reindex_required=reindex,
            base_cost=COST_REINDEX if reindex else COST_RUNTIME,
            capability=_optimizer.PATH_CAPABILITIES.get(path),
            conflict_family=path,
            prerequisites=_prerequisites(path),
            blocked_reason=reason,
            blocked_detail=detail,
        )
    return catalog


# 모듈 로드 시 한 번만 만든다. rules/optimizer가 정적 선언이라 안전하다.
ACTION_CATALOG: dict[str, ActionDefinition] = _build_catalog()


# ── 조회 ──────────────────────────────────────────────────────────

def get_action(key: str) -> ActionDefinition | None:
    """action key로 정의를 찾는다. 없으면 None."""
    return ACTION_CATALOG.get(key)


def all_actions() -> list[ActionDefinition]:
    """등록된 모든 action(차단 포함). key 사전순."""
    return [ACTION_CATALOG[k] for k in sorted(ACTION_CATALOG)]


def executable_actions() -> list[ActionDefinition]:
    """실행 가능한 action만. key 사전순."""
    return [a for a in all_actions() if not a.is_blocked]


def blocked_actions() -> list[ActionDefinition]:
    """차단된 action만. key 사전순."""
    return [a for a in all_actions() if a.is_blocked]


def actions_on_axis(canonical_path: str) -> list[ActionDefinition]:
    """같은 config 축을 공유하는 action들.

    같은 축에 둘 이상이면 경쟁 가능성이 있다. 다만 boolean 축(enable/disable)은
    현재 상태가 한쪽을 no-op으로 만들어 자동 배타되므로 충돌 정책 대상이 아니다.
    """
    return [a for a in all_actions() if a.canonical_path == canonical_path]


def is_sweepable(key: str) -> bool:
    """internal sweep으로 여러 후보를 한 study에서 비교할 수 있는 축인지."""
    action = ACTION_CATALOG.get(key)
    if action is None:
        return False
    return action.canonical_path in _optimizer.BACKEND_SUPPORTED_PATHS.get(
        "internal", set()
    )


# ── 무결성 검증 ───────────────────────────────────────────────────

def validate_catalog() -> list[str]:
    """catalog 계약 위반을 문자열 목록으로 반환한다. 비면 통과.

    테스트가 호출하며, 위반 시 어떤 계약이 깨졌는지 그대로 보고한다.
    """
    problems: list[str] = []

    for key, action in sorted(ACTION_CATALOG.items()):
        # key는 <canonical_path>:<operation> 형태여야 한다.
        expected = f"{action.canonical_path}:{action.operation}"
        if key != expected:
            problems.append(f"{key}: key가 {expected} 와 다르다")

        # 고정값을 key에 넣지 않았는지(starvation 방지).
        if key.count(":") != 1:
            problems.append(f"{key}: key에 값이 포함된 것으로 보인다(구분자 2개 이상)")

        # 비용은 재색인 여부에서 유도된다.
        expected_cost = COST_REINDEX if action.reindex_required else COST_RUNTIME
        if action.base_cost != expected_cost:
            problems.append(
                f"{key}: base_cost {action.base_cost} 가 재색인 여부와 어긋난다"
            )

        # 재색인 판정이 optimizer 정본과 일치해야 한다.
        in_reindex_paths = action.canonical_path in _optimizer.REINDEX_PATHS
        if in_reindex_paths and not action.reindex_required:
            problems.append(f"{key}: REINDEX_PATHS 에 있으나 reindex_required 가 False")

        # 실행 가능한 action은 state mapping과 capability를 모두 통과해야 한다.
        if not action.is_blocked:
            if action.canonical_path not in _optimizer.STATE_MAPPABLE_PATHS:
                problems.append(f"{key}: 실행 가능인데 STATE_MAPPABLE_PATHS 에 없다")
            if action.capability and not _optimizer.DEFAULT_CAPABILITIES.get(
                action.capability, False
            ):
                problems.append(
                    f"{key}: 실행 가능인데 capability {action.capability} 가 꺼져 있다"
                )
        elif not action.blocked_detail:
            problems.append(f"{key}: 차단됐는데 사유 상세가 비어 있다")

        # 한 action은 canonical 축 하나만 소유한다.
        if action.conflict_family != action.canonical_path:
            problems.append(f"{key}: conflict_family 가 canonical_path 와 다르다")

    # 같은 축에 replace 가 둘 이상이면 starvation 이 생긴다.
    replace_per_axis: dict[str, list[str]] = {}
    for action in ACTION_CATALOG.values():
        if action.operation == "replace":
            replace_per_axis.setdefault(action.canonical_path, []).append(action.key)
    for axis, keys in sorted(replace_per_axis.items()):
        if len(keys) > 1:
            problems.append(
                f"{axis}: replace action 이 {len(keys)}개다({', '.join(sorted(keys))}). "
                "값을 후보로 통합해야 한다"
            )

    # rules 가 참조하는 모든 action 이 catalog 에 있어야 한다.
    for label, rule in _rules.LABEL_TO_PRESCRIPTIONS.items():
        if rule.get("status") != "ready":
            continue
        for prescription in rule.get("prescriptions") or []:
            for raw_path, value in (prescription.get("patch") or {}).items():
                key = build_action_key(raw_path, value)
                if key not in ACTION_CATALOG:
                    problems.append(f"{label}/{prescription.get('id')}: {key} 미등록")

    return problems
