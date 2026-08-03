"""
agents/optimize/planner.py
Optimize 모듈의 "결정" 계층.

[역할]
  Eval의 진단 리포트(state.report.findings)를 받아서
    1) 자동 처방 대상인지 3분류하고 (manual / actionable / 스킵)
    2) 최적화 흐름을 결정하고 (OptimizeDecision: 제안/적용/유지/수동)
    3) 같은 라벨의 finding 을 묶어 최상위 우선순위 라벨 하나를 골라 처방 후보를 만들어
    4) OptimizationRequest 로 묶어 optimizer 에게 넘긴다.

  Eval 은 Finding 을 probe 마다 따로 만든다(같은 원인이 probe 10개에서 터지면
  Finding 도 10개). 처방은 라벨 단위이므로 planner 는 먼저 라벨로 묶고,
  그 묶음 전체의 측정값으로 점수와 근거값을 계산한다.

  "무엇을, 어느 순서로" 까지가 planner 의 책임이다.
  "그 처방을 실제 config 값으로 바꾸는 일"은 config_mapper/adapters,
  "적용 후 좋아졌는지 판단·롤백"은 optimizer 결과 + history 소관이다.

[읽는 것]  state.report(Finding 목록), state.index_config, state.iteration, blacklist
[쓰는 것]  (state 를 직접 수정하지 않음. agent.py 가 반환값을 받아 반영한다.)

[MVP 결정 사항]  (planner 설계 시 확정, 나중에 재검토 가능)
  - 우선순위: 1차로 그룹(A>C>B) 정렬, 2차로 점수 정렬.
    D그룹은 manual 이라 자동처방 대상에서 빠진다.
  - 예비(confirmed=False) finding 은 자동 처방 대상에서 제외한다(_split_findings 참고).
  - 점수 = 빈도 × 진단신뢰도 ÷ 처방비용
      빈도   = 라벨 묶음이 영향을 준 probe 수 (최소 1)
      신뢰도 = rules.py diagnosis_confidence (None 이면 1.0 fallback)
      비용   = rules.py cost 가 None 이므로 reindex 로 유도 (런타임=1, 재색인=3)
  - 후보값은 진단 측정값에서 계산한다(_GROUNDED_VALUES). 근거가 없는 라벨만
    방향 키워드(×2/÷2) 추측으로 폴백한다.
  - target_metrics 는 rules.py 라벨의 target_metrics 를 읽어 실어 보낸다.
    guardrail 은 폐기 — 롤백은 전역 하한선 체크 + 점수 비교로 대체(history.py).
  - propose_only(제안만) 모드는 뼈대만. 현재 기본은 apply_optimize.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any

from core.state import AgentDoctorState
from core.schema import Document, Finding
from agents.optimize import rules
from agents.optimize import eligibility
from agents.optimize import gate
from agents.optimize import history
from agents.optimize.config_mapper import canonicalize_path, get_current_value
from agents.optimize.evidence_window import build_evidence_windows
from agents.optimize.schemas import (
    ActionAttemptKey,
    ActionStudyKey,
    ConfigPatch,
    OptimizationRequest,
    OptimizeDecision,
    PrescriptionCandidate,
)

# ── 후보값 계산은 candidate_values 로 분리했다 ────────────────────
# planner 는 "무엇을 어느 순서로" 를 정하고, "얼마나" 는 그쪽이 계산한다.
# 아래 이름들은 기존 코드·테스트가 planner 를 통해 접근하던 것이라 그대로 re-export
# 한다(경로를 깨지 않기 위한 호환 계층이며, 계산 내용은 바뀌지 않았다).
from agents.optimize.candidate_values import (  # noqa: F401  (re-export)
    _DEFAULT_CURRENT,
    _DEFAULT_HYBRID_DENSE_WEIGHT,
    _DEFAULT_MAX_RERANK_CANDIDATES,
    _DIRECTION_STEP,
    _EvidenceAnalysis,
    _GROUNDED_ONLY,
    _GROUNDED_VALUES,
    _MAX_STEP_PER_PROBE,
    _MAX_SWEEP_CANDIDATES,
    _SYMBOLIC_FALLBACK_ALLOWED,
    _WEIGHT_MAX,
    _WEIGHT_MIN,
    _WEIGHT_STEPS,
    _allows_symbolic_fallback,
    _build_search_space,
    _chunk_candidate_limits,
    _chunk_candidate_policy,
    _chunk_overlap_candidate_policy,
    _chunk_positions_by_doc,
    _chunk_safety_boundary,
    _clamped_chunk_candidate,
    _concrete_values,
    _evidence_windows,
    _finding_search_space,
    _ground_chunk_overlap_candidates,
    _ground_chunk_size_candidates,
    _ground_hybrid_dense_weight,
    _ground_rerank_candidates,
    _ground_top_k_from_gold,
    _grounded_search_space,
    _knee,
    _knee_candidates,
    _percentile_nearest_rank,
    _probe_required_candidates,
    _probe_required_top_k,
    _round_to_step,
    _supplied_candidates,
    _valid_gold_spans,
)




# ── 상수 ──────────────────────────────────────────────────────────
# 그룹 1차 우선순위. 값이 작을수록 먼저 처리.
# D(데이터 문제)는 manual 이라 자동처방 대상이 아니므로 실질 순서는 A > C > B.
_GROUP_ORDER: dict[str, int] = {"A": 0, "C": 1, "B": 2, "D": 3}

# MVP fallback 상수 (rules.py 값이 None 일 때 사용)
_DEFAULT_CONFIDENCE = 1.0
_COST_RUNTIME = 1
_COST_REINDEX = 3




# topic_cluster 신호 소비 스위치 — 현재 OFF(관측용 신호로만 유지).
# 신호 생산(Eval)·대조 로직(_prescription_applies)·테스트는 모두 배선돼 있으나,
# 소비(applies_when 으로 후보를 실제로 거르는 일)는 의도적으로 꺼 둔다. 이유:
#   1) 신호가 고르는 유일한 처방 swap_embedding_model 은 optimizer 의
#      DEFAULT_CAPABILITIES["embedding_model"]=False 로 항상 거절돼(unsupported_capability),
#      spread/concentrated 를 활성화해도 결국 청킹으로 완화된다 — 분기가 config 적용까지
#      이어지지 않는다.
#   2) 임계값(rules.py TOPIC_CLUSTER_*_RATIO)이 아직 캘리브레이션 전 임의값이고,
#      실측상 추정량 분산(stdev~0.5)이 none 대 폭(~0.2)보다 커, 신호 없는 회차도
#      spread/concentrated 로 튄다. 소비를 켜면 그 노이즈가 비싼 재색인을 잘못 발동시킨다.
# 따라서 임베딩 교체 실행(capability 활성화 + 검증 모델 후보 + 차원/재색인 통합검증)과
# 임계값 캘리브레이션(신뢰도 게이트)이 준비된 뒤 이 플래그를 True 로 켠다. False 인 동안
# planner 는 신호를 무시하고 전 처방을 순서대로 시도한다(신호 배선 이전과 동일).
_CONSUME_TOPIC_CLUSTER_SIGNAL = False

# 사전검증(prescreener)으로 후보를 미리 거를 수 있는 축.
# chunker.strategy 는 여기 넣지 않는다: prescreener 는 gold span 이 청크 경계에
# 걸리는지를 기하로 판정하는데, 전략 교체는 경계 생성 규칙 자체를 바꾸므로 기존
# 경계 좌표를 전제로 한 span-경계 계산이 성립하지 않는다. 전략은 사전검증 없이
# internal sweep 의 실측(composite_score)으로 판정한다 — 그래서 use_internal 분기도
# 후보 수(candidate_count > 1)만 보는 아래 경로를 탄다.
_CHUNK_PRECHECK_PATHS = frozenset({
    "chunker.chunk_size",
    "chunker.chunk_overlap",
})
_CHUNK_PRECHECK_GROUNDING_STATUSES = frozenset({
    "grounded",
    "explicit_candidates",
})



_CONTEXT_NOISE_LABEL = "context_noise_interference"
_TOP_K_EXPANSION_LABELS = {
    "retrieval_incomplete_enumeration",
    "retrieval_missing_gold",
}





# ── 진입점 ────────────────────────────────────────────────────────

def plan(
    state: AgentDoctorState,
    blacklist: set[tuple[str, str]] | None = None,
) -> tuple[OptimizationRequest | None, OptimizeDecision]:
    """
    진단 리포트를 보고 (최적화 요청, 흐름 결정)을 만든다.

    Args:
        state: 공유 상태. state.report 가 있어야 한다.
        blacklist: 이미 실패해 재시도 금지된 (label, prescription_id) 조합.
            history.py 가 나중에 채워 넘긴다. None 이면 빈 집합.

    Returns:
        (request, decision)
          - decision.mode != "apply_optimize" 이면 request 는 None.
          - apply_optimize 이면 request 에 처방 후보가 담긴다.
    """
    blacklist = blacklist or set() # set은 blacklist가 None인 경우 필요

    if state.report is None:
        return None, OptimizeDecision(
            mode="use_current",
            status="skipped",
            requires_user_confirmation=False,
            next_route="serve",
            reason="진단 리포트가 없음 — 최적화 스킵",
        )

    manual, actionable = _split_findings(state.report.findings) #draft 부분은 사라짐
    decision = _decide_mode(state, actionable, manual)

    if decision.mode != "apply_optimize":
        return None, decision

    selection = _select_action(state, actionable, blacklist)
    if selection is None or selection.selected is None:
        # 실을 request 가 없으므로 "왜 아무것도 못 했는지"를 decision 에 담는다.
        # 후보값 근거 계산 실패(direction_conflict·insufficient_spans 등)도 여기로
        # 흘러오며, 이게 없으면 사용자에게는 "처방 없음"만 남는다.
        return None, OptimizeDecision(
            mode="use_current",
            status="skipped",
            requires_user_confirmation=False,
            next_route="serve",
            reason="적용 가능한 action 이 없음",
            manual_labels=decision.manual_labels,
            metadata=_selection_diagnostics(selection),
        )

    request = _build_action_request(selection, state)
    decision.request_id = request.request_id
    return request, decision


# ── action 중심 선택 ──────────────────────────────────────────────
# label 을 먼저 고르고 그 label 의 처방 순서를 따르던 방식을 대체한다.
# 모든 활성 label 이 지지하는 action 을 만들고, 같은 실제 config 변경을 하나로 통합한
# 뒤 action 끼리 경쟁시킨다. label 은 근거·probe·목표 metric 만 제공한다.


@dataclass
class _ActionSelection:
    """선택 결과와 그 근거. 요청을 만들고 리포트를 쓰는 데 필요한 것만 담는다."""

    selected: Any = None
    ranked: list[Any] = field(default_factory=list)
    rejected: list[Any] = field(default_factory=list)
    deferred: list[dict[str, Any]] = field(default_factory=list)
    evidence_cache: dict[str, Any] = field(default_factory=dict)
    findings_by_label: dict[str, list[Finding]] = field(default_factory=dict)


def _select_action(
    state: AgentDoctorState,
    actionable: list[Finding],
    exclusions: set[str] | None,
) -> _ActionSelection | None:
    """활성 Finding 에서 적용할 action 하나를 고른다.

    실행 가능성 판정이 **점수 경쟁보다 먼저**다. 실행할 수 없는 action 이 표를 받으면
    실행 가능한 action 이 밀려나고, 선택된 뒤 optimizer 에서 탈락하면 그 방문이
    통째로 낭비된다.
    """
    from agents.optimize import action_aggregator

    grouped = _group_by_label(actionable)
    evidence_cache: dict[str, Any] = {}

    def search_space_for(label, findings, changes):
        # 청크 경계 분석은 비싸다. 라벨 단위로 캐시해 같은 분석을 반복하지 않는다.
        analysis = None
        if any(canonicalize_path(path) == "chunker.chunk_size" for path in changes):
            if label not in evidence_cache:
                evidence_cache[label] = _evidence_windows(state, findings)
            analysis = evidence_cache[label]
        return _finding_search_space(findings, changes, state, analysis)

    blocked_keys, blocked_attempts = _normalize_exclusions(exclusions, state)
    supports = action_aggregator.build_action_supports(
        grouped, state, search_space_for=search_space_for
    )
    candidates = action_aggregator.aggregate_action_candidates(
        supports, state, exclusions=blocked_keys
    )
    eligible, rejected = action_aggregator.filter_ineligible_actions(
        candidates,
        state,
        backend="rules",
        runtime_capabilities=state.runtime_capabilities,
        constraints=_policy_constraints(state),
        blocked_attempts=blocked_attempts,
    )
    kept, deferred = action_aggregator.resolve_action_conflicts(eligible)
    ranked = action_aggregator.rank_action_candidates(kept)

    return _ActionSelection(
        selected=ranked[0] if ranked else None,
        ranked=ranked,
        rejected=rejected,
        deferred=deferred,
        evidence_cache=evidence_cache,
        findings_by_label=grouped,
    )


def _policy_constraints(state: AgentDoctorState) -> dict[str, Any]:
    """config 정책에서 나오는 후보값 제약. optimizer 의 정적 제약 위에 얹는다.

    후보창 상한은 사용자 config(rerank_candidate_policy)의 값이라 optimizer 의
    DEFAULT_CONSTRAINTS 로는 표현할 수 없다. **선택 전 eligibility 와 실행 직전
    optimizer 가 같은 값을 봐야 한다** — 한쪽만 보면 근거값 계산은 상한을 지키는데
    방향 폴백(현재값×2)이 상한을 넘겨 그대로 config 에 박힌다(30×2=60 > 50).
    """
    policy = state.index_config.get("rerank_candidate_policy") or {}
    return {
        "reranker.candidate_count": {
            "max": int(policy.get("max_candidates", _DEFAULT_MAX_RERANK_CANDIDATES)),
        },
    }


def _rejected_action_entries(selection: _ActionSelection) -> list[dict[str, Any]]:
    """제외된 action 과 사유. 후보값 근거(grounding)까지 함께 남긴다."""
    entries: list[dict[str, Any]] = []
    for candidate in selection.rejected:
        entry: dict[str, Any] = {
            "action_key": candidate.action_key,
            "reason": candidate.reason,
        }
        grounding = _selected_grounding(candidate, candidate.definition.canonical_path)
        if grounding:
            entry["candidate_grounding"] = grounding
        entries.append(entry)
    return entries


def _selection_diagnostics(
    selection: _ActionSelection | None,
) -> dict[str, Any]:
    """선택이 비었을 때 사용자에게 남길 근거."""
    if selection is None:
        return {}
    return {
        "rejected_actions": _rejected_action_entries(selection),
        "deferred_axes": list(selection.deferred),
    }


def _normalize_exclusions(
    exclusions: set[Any] | None,
    state: AgentDoctorState,
) -> tuple[set[str], set[Any]]:
    """호출자가 넘긴 제외 목록을 (차단 action key, 차단 전이) 로 가른다.

    받는 형태는 네 가지다.
      - ``str``              : action key. 이번 방문에서만 막는 visit-local 제외
      - ``ActionStudyKey``   : 그 baseline 에서 결론이 난 탐색 → action 통째로 차단
      - ``ActionAttemptKey`` : **정확한 전이 하나만** 차단 → 후보값 단위로 거른다
      - ``(label, id)`` 튜플 : 구버전 blacklist. action key 로 환산해 받아준다

    ⚠️ study/attempt key 는 baseline fingerprint 를 담고 있어, baseline 이 달라지면
    자동으로 풀린다(구현계획 §5.1). 여기서 baseline 이 다른 study key 를 action 차단으로
    올리면 그 완화가 무너지므로 현재 baseline 과 일치할 때만 차단한다.
    """
    from agents.optimize import action_catalog

    blocked_keys: set[str] = set()
    blocked_attempts: set[Any] = set()

    for item in exclusions or set():
        if isinstance(item, str):
            blocked_keys.add(item)
        elif isinstance(item, ActionAttemptKey):
            blocked_attempts.add(item)
        elif isinstance(item, ActionStudyKey):
            current = history.baseline_fingerprint(
                state.index_config,
                item.action_key,
                state.runtime_capabilities,
            )
            if item.baseline_fingerprint == current:
                blocked_keys.add(item.action_key)
        elif isinstance(item, tuple) and len(item) == 2:
            label, prescription_id = item
            rule = rules.get_rule(label)
            for prescription in (rule or {}).get("prescriptions") or []:
                if prescription.get("id") != prescription_id:
                    continue
                for raw_path, value in (prescription.get("patch") or {}).items():
                    blocked_keys.add(action_catalog.build_action_key(raw_path, value))

    return blocked_keys, blocked_attempts


def _build_action_request(
    selection: _ActionSelection,
    state: AgentDoctorState,
) -> OptimizationRequest:
    """선택된 action 을 OptimizationRequest 로 묶는다.

    action 필드가 정본이고, `failure_label`·`candidates` 같은 기존 필드는 여기서
    파생해 함께 채운다(dual-read). optimizer·agent·history·reporter 가 아직 그
    필드들을 읽기 때문이며, 소비처 전환이 끝나면 파생을 걷어낸다.
    """
    action = selection.selected
    definition = action.definition
    path = definition.canonical_path
    values = list(action.search_space.get(path, []))

    # 대표 라벨 — 설명용이다. 인과 등급이 가장 높은 support 를 고른다.
    primary_support = min(
        action.supports,
        key=lambda s: (
            {"A": 0, "C": 1, "B": 2, "D": 3}.get(s.group or "D", 99),
            s.label,
        ),
    )
    primary_label = primary_support.label
    findings = selection.findings_by_label.get(primary_label, [])

    # 후보가 여러 개면 sweep 이 실측으로 고른다. chunk 축은 사전검증 결과가
    # 있어야 internal 로 넘긴다(기존 정책 유지).
    chunk_precheck_context = None
    if path in _CHUNK_PRECHECK_PATHS:
        chunk_precheck_context = _chunk_precheck_context(
            state,
            findings,
            path=path,
            evidence_analysis=selection.evidence_cache.get(primary_label),
        )
    grounding = _selected_grounding(action, path)
    use_internal = _should_sweep(
        path, values, grounding, chunk_precheck_context
    )

    patch = ConfigPatch(
        changes={path: values[0] if len(values) == 1 else definition.operation},
        reindex_required=definition.reindex_required,
        description=f"{action.action_key} ({', '.join(action.supporting_labels)})",
        metadata={
            "action_key": action.action_key,
            "prescription_id": primary_support.prescription_id,
        },
    )
    legacy_candidate = PrescriptionCandidate(
        id=primary_support.prescription_id or action.action_key,
        failure_label=primary_label,
        group=primary_support.group,
        status="ready",
        patch=patch,
        search_space={path: values},
        cost=definition.base_cost,
        priority=action.score,
        target_metrics=list(action.target_metrics),
        applies_when=dict(primary_support.applies_when),
        reason=primary_support.reason,
        metadata=(
            {"candidate_grounding": grounding} if grounding else {}
        ),
    )

    metadata: dict[str, Any] = {
        "primary_metric": "composite_score",
        "min_delta": history.MIN_IMPROVEMENT_MARGIN,
        "study_baseline_config": dict(state.index_config),
        "baseline_metrics": _report_metrics(state),
        "trial_results": [],
        "runtime_capabilities": {
            name: dict(capability)
            for name, capability in state.runtime_capabilities.items()
            if isinstance(capability, dict)
        },
        # 후보창 상한 같은 config 정책 제약. planner 의 eligibility 와 optimizer 의
        # 실행 직전 재검증이 같은 값을 봐야 한다(_policy_constraints 주석 참고).
        "constraints": _policy_constraints(state),
        "action_key": action.action_key,
        "action_score_breakdown": dict(action.score_breakdown),
        "candidate_support": dict(action.metadata.get("candidate_support", {})),
        "rejected_actions": _rejected_action_entries(selection),
        "deferred_axes": list(selection.deferred),
        "runner_up_actions": [
            {"action_key": c.action_key, "score": round(c.score, 6)}
            for c in selection.ranked[1:4]
        ],
    }
    if grounding:
        metadata["candidate_grounding"] = dict(grounding)
    if use_internal and chunk_precheck_context is not None:
        metadata["chunk_precheck_context"] = chunk_precheck_context

    return OptimizationRequest(
        request_id=str(uuid.uuid4()),
        iteration=state.iteration,
        baseline_config=dict(state.index_config),
        failure_label=primary_label,
        related_failure_labels=[
            label for label in action.supporting_labels if label != primary_label
        ],
        candidates=[legacy_candidate],
        search_space={path: values},
        target_metrics=list(action.target_metrics),
        target_profile="balanced",
        optimizer="internal" if use_internal else "rules",
        max_trials=max(len(values), 1),
        reason=(
            f"action {action.action_key} "
            f"(지지 라벨 {len(action.supporting_labels)}개, "
            f"probe {len(action.supporting_probes)}개)"
        ),
        propose_only=False,
        metadata=metadata,
        action_key=action.action_key,
        action=action,
        supporting_labels=list(action.supporting_labels),
        supporting_probes=sorted(action.supporting_probes),
        opposing_labels=list(action.opposing_labels),
        action_score=action.score,
        action_score_breakdown=dict(action.score_breakdown),
    )


def _selected_grounding(action: Any, path: str) -> dict[str, Any] | None:
    """선택된 action 의 후보값이 어떻게 나왔는지(실측 근거 여부)."""
    for support in action.supports:
        if support.grounding_metadata:
            return dict(support.grounding_metadata)
    return None


def _should_sweep(
    path: str,
    values: list[Any],
    grounding: dict[str, Any] | None,
    chunk_precheck_context: dict[str, Any] | None,
) -> bool:
    """internal sweep 으로 넘길지 판단한다(기존 정책을 그대로 옮긴 것).

    chunk 축은 사전검증 입력이 갖춰졌을 때만 sweep 한다 — prescreener 가 실제 청커를
    dry-run 해 후보를 거르기 때문이다. 그 외 축은 후보가 여럿이면 sweep 한다.
    """
    if not eligibility.is_sweepable(path):
        return False
    if path in _CHUNK_PRECHECK_PATHS:
        return (
            bool(values)
            and grounding is not None
            and grounding.get("status") in _CHUNK_PRECHECK_GROUNDING_STATUSES
            and _has_chunk_precheck_inputs(chunk_precheck_context)
        )
    return len(values) > 1


def _rule_uses_chunk_size(rule: dict[str, Any]) -> bool:
    """선택된 라벨의 처방 중 chunk_size 축이 있는지 확인한다."""

    return any(
        canonicalize_path(path) == "chunker.chunk_size"
        for prescription in rule.get("prescriptions", [])
        if isinstance(prescription, dict)
        for path in prescription.get("patch", {})
        if isinstance(path, str)
    )


# ── 1. 분류 ───────────────────────────────────────────────────────

def _split_findings(
    findings: list[Finding],
) -> tuple[list[Finding], list[Finding]]:
    """
    finding 을 (manual, actionable) 로 나눈다.
      - manual     : is_manual (D그룹) → 자동처방 불가, reporter 로 넘어감
      - actionable : is_actionable (ready + 처방 있음) + 확정(confirmed) → 점수 경쟁 대상
      - 나머지     : draft/unassigned/라벨없음/예비 → 지금은 실행 불가, 스킵

    예비(confirmed=False)는 Eval 이 자원(진단 모드/tier) 부족으로 확신하지 못한
    의심 원인이다. 처방 1회는 파이프라인 전체 재평가(LLM 호출 다수)를 유발하므로,
    확신 없는 진단에는 그 비용을 쓰지 않는다. 더 깊은 EVAL_MODE 에서 확정되면
    그때 처방 대상이 된다. (manual 은 비용을 쓰지 않으므로 예비여도 사용자에게
    알리기 위해 그대로 넘긴다.)
    """
    manual: list[Finding] = []
    actionable: list[Finding] = []
    for f in findings:
        label = f.label
        if not label:
            continue  # 세분화 라벨이 없으면 rules.py 매핑 불가
        if rules.is_manual(label):
            manual.append(f)
        elif rules.is_actionable(label) and f.confirmed:
            actionable.append(f)
        # draft/unassigned/예비 는 의도적으로 스킵
    return manual, actionable


# ── 2. 흐름 결정 (3-way / 4-way) ──────────────────────────────────

def _decide_mode(
    state: AgentDoctorState,
    actionable: list[Finding],
    manual: list[Finding],
) -> OptimizeDecision:
    """
    최적화 흐름을 결정한다.
      - 이미 임계값 통과            → use_current(already_optimal) → serve
      - 자동처방 없음 + manual 있음 → manual_required → serve (사람 개입)
      - 자동처방 없음 + manual 없음 → use_current(skipped) → serve
      - 자동처방 있음               → apply_optimize → index
    """
    manual_labels = [f.label for f in manual if f.label]

    report = state.report
    if gate.passes_report(report):
        return OptimizeDecision(
            mode="use_current",
            status="already_optimal",
            requires_user_confirmation=False,
            next_route="serve",
            reason="모든 임계값 달성 — 최적화 불필요",
            manual_labels=manual_labels,
        )

    if not actionable:
        if manual:
            return OptimizeDecision(
                mode="manual_required",
                status="manual_required",
                requires_user_confirmation=True,
                next_route="serve",
                reason="자동 처방 가능한 라벨 없음 — 사람 개입 필요(D그룹)",
                manual_labels=manual_labels,
            )
        return OptimizeDecision(
            mode="use_current",
            status="skipped",
            requires_user_confirmation=False,
            next_route="serve",
            reason="처방 가능한 finding 없음",
            manual_labels=manual_labels,
        )

    return OptimizeDecision(
        mode="apply_optimize",
        status="proposed",
        requires_user_confirmation=False,
        next_route="index",
        reason="처방 가능한 finding 존재 → 최적화 진행",
        manual_labels=manual_labels,
    )


# ── 3. 우선순위 점수 ──────────────────────────────────────────────

def _derive_cost(prescription: dict) -> int:
    """처방비용을 reindex 로 유도한다 (런타임=1, 재색인=3).
    rules.py 의 cost 가 아직 전부 None 이라 MVP 는 reindex 플래그로 계산."""
    return _COST_REINDEX if prescription.get("reindex") else _COST_RUNTIME


def _label_cost(rule: dict) -> int:
    """라벨 점수 계산용 비용. 처방 리스트의 첫(가장 가벼운) 처방 기준."""
    prescriptions = rule.get("prescriptions") or []
    if not prescriptions:
        return _COST_RUNTIME
    return _derive_cost(prescriptions[0])


def _group_by_label(actionable: list[Finding]) -> dict[str, list[Finding]]:
    """같은 라벨의 finding 을 묶는다.

    Eval 은 Finding 을 probe 마다 따로 만든다(affected_probes 는 항상 1개).
    같은 원인이 probe 10개에서 터지면 Finding 객체도 10개가 된다. 처방은 라벨
    단위이므로, 점수·근거값 계산 모두 이 묶음을 대상으로 해야 한다.
    """
    groups: dict[str, list[Finding]] = {}
    for f in actionable:
        groups.setdefault(f.label, []).append(f)
    return groups


def _score(findings: list[Finding], rule: dict) -> float:
    """우선순위점수 = 빈도 × 진단신뢰도 ÷ 처방비용.

    빈도는 이 라벨이 영향을 준 probe 수다. Finding 하나당 probe 하나이므로
    묶음 전체의 affected_probes 를 합쳐야 실제 빈도가 나온다(중복 제거).
    """
    probes = {p for f in findings for p in f.affected_probes}
    frequency = max(len(probes), 1)
    confidence = rule.get("diagnosis_confidence")
    if confidence is None:
        confidence = _DEFAULT_CONFIDENCE
    cost = _label_cost(rule)
    return (frequency * confidence) / cost


def _rank_groups(
    groups: dict[str, list[Finding]],
) -> list[tuple[str, list[Finding], dict, float]]:
    """라벨 묶음을 (그룹순서, 점수내림차순)으로 정렬.
    1차 키 = 그룹(A>C>B), 2차 키 = 점수 높은 순."""
    ranked: list[tuple[str, list[Finding], dict, float]] = []
    for label, findings in groups.items():
        rule = rules.get_rule(label)
        if not rule:
            continue
        ranked.append((label, findings, rule, _score(findings, rule)))
    noise_precedes_top_k = _context_noise_precedes_top_k_expansion(ranked)
    ranked.sort(
        key=lambda item: (
            _effective_group_order(item, noise_precedes_top_k),
            -item[3],
            0 if item[0] == _CONTEXT_NOISE_LABEL else 1,
        )
    )
    return ranked


def _context_noise_precedes_top_k_expansion(
    ranked: list[tuple[str, list[Finding], dict, float]],
) -> bool:
    """노이즈가 top-k 확장 라벨만큼 강하면 컨텍스트 압축을 먼저 검증한다.

    top_k 증가는 누락된 근거를 회수할 수 있지만, 동시에 모델에 넣는 잡음도 늘린다.
    따라서 context_noise_interference가 이미 같은 수준 이상으로 확인된 방문에서는
    먼저 재색인 없는 압축/필터링 처방을 실험해 노이즈 악화를 막는다.
    """
    noise_score = next(
        (score for label, _findings, _rule, score in ranked if label == _CONTEXT_NOISE_LABEL),
        None,
    )
    if noise_score is None:
        return False
    top_k_scores = [
        score
        for label, _findings, _rule, score in ranked
        if label in _TOP_K_EXPANSION_LABELS
    ]
    return bool(top_k_scores) and noise_score >= max(top_k_scores)


def _effective_group_order(
    item: tuple[str, list[Finding], dict, float],
    noise_precedes_top_k: bool,
) -> int:
    label, _findings, rule, _score_value = item
    if noise_precedes_top_k and label == _CONTEXT_NOISE_LABEL:
        return _GROUP_ORDER["A"]
    return _GROUP_ORDER.get(rule.get("group"), 99)


# ── 4. 최상위 선택 + 블랙리스트 ───────────────────────────────────

def _finding_signal(findings: list[Finding], key: str) -> str | None:
    """findings metadata 에서 신호값 하나를 읽는다(applies_when 대조용).

    같은 라벨의 findings 는 Eval 이 라벨 단위로 같은 신호를 실으므로(예: topic_cluster),
    첫 값을 대표로 쓴다. 신호가 없으면 None → 호출부가 '태그 무시(순차 fallback)'로 처리.
    """
    for f in findings:
        val = f.metadata.get(key)
        if val is not None:
            return val
    return None


def _prescription_applies(pres: dict, findings: list[Finding]) -> bool:
    """처방의 applies_when 신호 조건을 finding metadata 와 대조한다.

    계약(schemas.py PrescriptionCandidate.applies_when / rules.py 주석):
      - 처방에 applies_when 이 없으면          → 항상 적용(신호 무관 처방)
      - 신호 키는 있는데 finding 에 값이 없으면 → 적용(미측정 = 순차 fallback, 기존 동작)
      - 값이 있으면 허용 리스트 membership 검사 → 포함될 때만 적용
    키가 여러 개면 전부(AND) 만족해야 한다. 지금 유일한 소비 키는 topic_cluster.
    """
    applies_when = pres.get("applies_when") or {}
    for key, allowed in applies_when.items():
        signal = _finding_signal(findings, key)
        if signal is None:
            continue                    # 미측정 → 이 조건은 통과(fallback)
        if signal not in allowed:
            return False
    return True


def _available_prescriptions(
    rule: dict, label: str, blacklist: set[tuple[str, str]],
    findings: list[Finding] | None = None,
) -> list[dict]:
    """블랙리스트·applies_when 신호에 걸리지 않은 처방만 순서대로 반환.

    findings 를 주면 applies_when(topic_cluster 등) 신호로 후보를 거른다. 안 주면(레거시
    호출) 블랙리스트만 본다 — 신호 대조는 findings 가 있어야 성립하기 때문.

    ⚠️ 현재 신호 소비는 _CONSUME_TOPIC_CLUSTER_SIGNAL=False 로 꺼져 있어, findings 를 줘도
    applies_when 대조를 건너뛰고 블랙리스트만 본다(관측용 신호로만 유지 — 상수 정의부 주석
    참고). 아래 완화 로직은 소비를 켰을 때를 위한 것으로, 소비가 켜지면 다시 활성화된다.

    (소비 ON 일 때의 계약) 신호로 걸러 후보가 0개가 되면 신호 조건을 완화해 블랙리스트만
    적용한 목록을 돌려준다. 신호는 처방 순서를 '선호'하게 만들 뿐 라벨을 통째로 막아선 안
    되기 때문이다 — 예를 들어 topic_cluster=spread 인데 swap_embedding_model 이 블랙리스트에
    오르면, 완화가 없으면 이 라벨의 후보가 전부 사라져 _pick_top 이 라벨 자체를 건너뛴다
    (신호 배선 이전에는 청킹 처방으로 넘어가던 경로라 회귀). 임계값이 아직 캘리브레이션 안
    된 임의값이라 더욱, 신호 때문에 고칠 기회를 잃는 쪽보다 덜 맞는 처방이라도 시도하는
    쪽이 안전하다.
    """
    unblacklisted = [
        p for p in rule.get("prescriptions", []) if (label, p["id"]) not in blacklist
    ]
    if findings is None or not _CONSUME_TOPIC_CLUSTER_SIGNAL:
        return unblacklisted
    preferred = [p for p in unblacklisted if _prescription_applies(p, findings)]
    return preferred or unblacklisted


def _pick_top(
    ranked: list[tuple[str, list[Finding], dict, float]],
    blacklist: set[tuple[str, str]],
) -> tuple[str, list[Finding], dict, float] | None:
    """정렬된 목록에서 '아직 시도할 처방이 남은' 최상위 라벨 묶음을 고른다.

    라벨이 스킵되는 조건은 블랙리스트로 처방이 전부 소진됐을 때뿐이다. 신호 소비가 켜져
    있어도(_CONSUME_TOPIC_CLUSTER_SIGNAL) applies_when 은 _available_prescriptions 안에서
    후보가 0개가 되면 완화되므로, 신호만으로는 라벨이 통째로 건너뛰어지지 않는다. 소비가
    꺼진 현재는 신호 자체를 보지 않아 늘 블랙리스트만 기준이 된다.
    """
    for label, findings, rule, score in ranked:
        if _available_prescriptions(rule, label, blacklist, findings):
            return label, findings, rule, score
    return None
def _build_candidates(
    label: str,
    findings: list[Finding],
    rule: dict,
    blacklist: set[tuple[str, str]],
    state: AgentDoctorState,
    *,
    evidence_analysis: _EvidenceAnalysis | None = None,
) -> list[PrescriptionCandidate]:
    """
    rules.py 의 raw dict 처방들을 PrescriptionCandidate 객체로 변환한다.
    rules.py 에 적힌 순서(가벼운 것 먼저)를 그대로 유지한다.
    블랙리스트에 걸린 처방은 제외한다.
    """
    candidates: list[PrescriptionCandidate] = []
    target_metrics = list(rule.get("target_metrics", []))
    reason = findings[0].description if findings else ""
    for pres in _available_prescriptions(rule, label, blacklist, findings):
        changes = dict(pres.get("patch", {}))
        search_space, grounding_metadata = _finding_search_space(
            findings,
            changes,
            state,
            evidence_analysis,
        )
        patch = ConfigPatch(
            changes=changes,
            reindex_required=bool(pres.get("reindex")),
            description=f"{label} → {pres['id']}",
            metadata={"prescription_id": pres["id"]},
        )
        candidates.append(
            PrescriptionCandidate(
                id=pres["id"],
                failure_label=label,
                group=rule.get("group"),
                status=rule.get("status"),
                patch=patch,
                # optimizer 가 소비할 구체 후보값.
                # 우선순위: Finding.metadata 후보 > 근거값 계산 > 방향 키워드 추측.
                search_space=search_space,
                cost=float(_derive_cost(pres)),
                priority=0.0,          # 후보 개별 우선순위는 MVP 미사용
                target_metrics=list(target_metrics),  # rules.py 라벨의 target_metrics
                # 신호기반 택1(retrieval_semantic_mismatch 등). 없으면 빈 dict
                # → optimizer 가 순서대로 순차 시도(fallback).
                applies_when=dict(pres.get("applies_when", {})),
                reason=reason,
                metadata=(
                    {"candidate_grounding": grounding_metadata}
                    if grounding_metadata is not None
                    else {}
                ),
            )
        )
    return candidates


# ── 6. 요청서 포장 ────────────────────────────────────────────────

def _build_request(
    label: str,
    findings: list[Finding],
    rule: dict,
    candidates: list[PrescriptionCandidate],
    ranked: list[tuple[str, list[Finding], dict, float]],
    state: AgentDoctorState,
    *,
    evidence_analysis: _EvidenceAnalysis | None = None,
) -> OptimizationRequest:
    """선택된 라벨과 처방 후보를 OptimizationRequest 로 묶는다."""
    related = [lbl for lbl, _fs, _rule, _s in ranked if lbl != label]
    probes = {p for f in findings for p in f.affected_probes}
    selected_space = candidates[0].search_space if candidates else {}
    candidate_count = (
        len(next(iter(selected_space.values())))
        if len(selected_space) == 1
        and isinstance(next(iter(selected_space.values())), (list, tuple))
        else 1
    )
    selected_path = _space_path(selected_space)
    candidate_grounding = (
        candidates[0].metadata.get("candidate_grounding")
        if candidates
        else None
    )
    grounding_status = (
        candidate_grounding.get("status")
        if isinstance(candidate_grounding, dict)
        else None
    )
    chunk_precheck_context = (
        _chunk_precheck_context(
            state,
            findings,
            path=selected_path,
            evidence_analysis=evidence_analysis,
        )
        if selected_path in _CHUNK_PRECHECK_PATHS
        else None
    )
    use_chunk_precheck = (
        candidate_count > 0
        and selected_path in _CHUNK_PRECHECK_PATHS
        and grounding_status in _CHUNK_PRECHECK_GROUNDING_STATUSES
        and _has_chunk_precheck_inputs(chunk_precheck_context)
    )
    use_internal = (
        use_chunk_precheck
        if selected_path in _CHUNK_PRECHECK_PATHS
        else candidate_count > 1
    )
    metadata: dict[str, Any] = {
        # 후보별 trade-off의 최종 심판은 Eval의 정규화 composite_score(0~1)다.
        # 신뢰도 축이 연속값이 된 뒤 composite 이 매끄러워져, 표시·게이트와 같은 지표로
        # 탐색까지 통일한다(과거 overall 을 따로 쓴 이유였던 '이진 신뢰도의 계단'이 사라짐).
        # 근거: history.judge 주석 + scoring.reliability_score.
        "primary_metric": "composite_score",
        # sweep 이 "baseline 을 이겼다"고 판정하는 최소 상승폭. judge(유지/롤백)와 같은
        # 값이어야 같은 점수 변화가 경로에 따라 다르게 보고되지 않는다(sweep 승자는
        # _finish_internal_study 가 그 자리에서 확정해 judge 를 거치지 않는다).
        # primary_metric 이 정규화 composite(0~1)이라 마진도 같은 스케일이다.
        "min_delta": history.MIN_IMPROVEMENT_MARGIN,
        "study_baseline_config": dict(state.index_config),
        "baseline_metrics": _report_metrics(state),
        "trial_results": [],
        # Optional 모델의 실제 준비 상태는 Index가 생산한다. Optimizer는 이
        # 스냅샷을 보고 실행할 수 없는 처방을 config 적용 전에 건너뛴다.
        "runtime_capabilities": {
            name: dict(capability)
            for name, capability in state.runtime_capabilities.items()
            if isinstance(capability, dict)
        },
    }
    # 후보창 상한은 config 정책값이라 optimizer 의 정적 DEFAULT_CONSTRAINTS 로는 표현할 수 없다.
    # 여기서 실어 보내야 근거값·방향 폴백(현재값×2)·Eval 이 직접 넘긴 후보까지 한 지점에서
    # 걸린다 — 근거값 계산에서만 상한을 보면 폴백 경로가 상한을 넘겨 config 에 박힌다
    # (현재값 30 이면 ×2=60 > 50).
    policy = state.index_config.get("rerank_candidate_policy") or {}
    metadata["constraints"] = {
        "reranker.candidate_count": {
            "max": int(policy.get("max_candidates", _DEFAULT_MAX_RERANK_CANDIDATES)),
        },
    }
    if isinstance(candidate_grounding, dict):
        metadata["candidate_grounding"] = dict(candidate_grounding)
    if use_internal and chunk_precheck_context is not None:
        metadata["chunk_precheck_context"] = chunk_precheck_context
    return OptimizationRequest(
        request_id=str(uuid.uuid4()),
        iteration=state.iteration,
        baseline_config=dict(state.index_config),
        failure_label=label,
        related_failure_labels=related,
        candidates=candidates,
        search_space={
            path: list(values) if isinstance(values, (list, tuple)) else [values]
            for path, values in selected_space.items()
        },
        target_metrics=list(rule.get("target_metrics", [])),  # 라벨의 target_metrics
        target_profile="balanced",
        # 후보가 여러 개면 internal backend 가 방문에 걸쳐 sweep 한다.
        optimizer="internal" if use_internal else "rules",
        max_trials=candidate_count,
        reason=f"우선순위 최상위 라벨: {label} (probe {len(probes)}개 영향)",
        propose_only=False,
        metadata=metadata,
    )


def _space_path(search_space: dict[str, Any]) -> str | None:
    """단일 축 search space의 canonical 경로를 반환한다."""

    if len(search_space) != 1:
        return None
    path = next(iter(search_space))
    return canonicalize_path(path) if isinstance(path, str) else None


def _report_metrics(state: AgentDoctorState) -> dict[str, Any]:
    """Internal Adapter가 고정 baseline으로 사용할 Eval 지표를 복사한다."""

    if state.report is None:
        return {}
    metrics: dict[str, Any] = dict(state.report.ragas_scores)
    if state.report.overall_score is not None:
        metrics["overall_score"] = state.report.overall_score
    # 탐색 objective 는 composite_score(0~1). baseline 관측값에도 실어 sweep 이 같은
    # 지표로 후보와 baseline 을 비교하게 한다(없으면 _extract_score 가 overall 로 폴백).
    composite_total = (state.report.composite_score or {}).get("total")
    if composite_total is not None:
        metrics["composite_total"] = float(composite_total)
        metrics["composite_score"] = float(composite_total) / 100.0
    metrics["pass_threshold"] = gate.passes_report(state.report)
    return metrics


def _chunk_precheck_context(
    state: AgentDoctorState,
    findings: list[Finding],
    *,
    path: str | None,
    evidence_analysis: _EvidenceAnalysis | None = None,
) -> dict[str, Any]:
    """Chunk 사전검증에 원문과 축에 맞는 evidence 좌표를 전달한다."""

    gold_spans = _valid_gold_spans(state, findings)
    measured_spans = gold_spans
    span_source = "gold_spans"
    if path == "chunker.chunk_size":
        evidence_windows, _metadata = (
            evidence_analysis
            if evidence_analysis is not None
            else _evidence_windows(state, findings)
        )
        if evidence_windows:
            measured_spans = evidence_windows
            span_source = "structural_evidence_windows"
    affected_doc_ids = {
        span.get("doc_id")
        for span in measured_spans
        if isinstance(span.get("doc_id"), str)
    }
    documents = [
        document
        for document in state.documents
        if not affected_doc_ids or document.doc_id in affected_doc_ids
    ]
    return {
        "documents": documents,
        # 새 계약은 측정 대상을 정확히 표현한다. prescreener는 오래된 저장 요청의
        # ``gold_spans``도 하위호환 입력으로 계속 읽는다.
        "evidence_spans": measured_spans,
        "span_source": span_source,
        "chunk_strategy": state.index_config.get(
            "chunk_strategy",
            state.index_config.get("chunk_stage", "markdown_recursive"),
        ),
    }


def _has_chunk_precheck_inputs(context: dict[str, Any] | None) -> bool:
    """사전검사가 실제 측정 가능한 원문과 evidence 좌표를 가졌는지 확인한다."""

    if not isinstance(context, dict):
        return False
    documents = context.get("documents")
    spans = context.get("evidence_spans")
    if not isinstance(documents, (list, tuple)) or not isinstance(
        spans,
        (list, tuple),
    ):
        return False

    document_lengths = {
        document.doc_id: len(document.content)
        for document in documents
        if (
            isinstance(document, Document)
            and isinstance(document.doc_id, str)
            and isinstance(document.content, str)
        )
    }
    if not document_lengths:
        return False

    return any(
        isinstance(span, dict)
        and isinstance(span.get("doc_id"), str)
        and isinstance(span.get("start"), int)
        and not isinstance(span.get("start"), bool)
        and isinstance(span.get("end"), int)
        and not isinstance(span.get("end"), bool)
        and 0 <= span["start"] < span["end"] <= document_lengths.get(
            span["doc_id"],
            -1,
        )
        for span in spans
    )
