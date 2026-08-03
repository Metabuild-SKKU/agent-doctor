"""
agents/optimize/planner.py
Optimize 모듈의 "결정" 계층.

[역할]
  Eval의 진단 리포트(state.report.findings)를 받아서
    1) 자동 처방 대상인지 3분류하고 (manual / actionable / 스킵)
    2) 최적화 흐름을 결정하고 (OptimizeDecision: 제안/적용/유지/수동)
    3) 활성 라벨이 지지하는 **action** 을 모아 하나를 고르고
    4) OptimizationRequest 로 묶어 optimizer 에게 넘긴다.

  planner 는 얇은 조율자다. 점수·통합·충돌·정렬은 action_aggregator, 실행 가능성은
  eligibility, 후보값 수학은 candidate_values 가 소유한다. 여기 남는 것은 분기,
  preliminary 제외, manual 보존, ID 생성, backend 선택이다.

  ⚠️ 라벨은 실행 순서를 소유하지 않는다. 라벨이 제공하는 것은 진단 근거·영향 probe·
  목표 metric 뿐이고, "무엇을 먼저 바꿀지"는 action 끼리의 경쟁이 정한다. 라벨을
  먼저 고른 뒤 그 라벨의 처방을 선언 순서대로 시도하던 코드는 전환으로 제거됐다.

[읽는 것]  state.report(Finding 목록), state.index_config, state.iteration, exclusions
[쓰는 것]  (state 를 직접 수정하지 않음. agent.py 가 반환값을 받아 반영한다.)

[MVP 결정 사항]  (나중에 재검토 가능)
  - 예비(confirmed=False) finding 은 자동 처방 대상에서 제외한다(_split_findings 참고).
  - 후보값은 진단 측정값에서 계산한다(_GROUNDED_VALUES). 근거가 없는 축만
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
    """제외된 action 과 사유.

    후보값 근거(grounding)와 함께 **어느 선언에서 왔는지**(지지 라벨·처방 id)도 남긴다.
    실행 불가 판정이 optimizer 에서 planner 앞단으로 옮겨오면서, 그 사유를 사용자에게
    설명할 책임도 함께 옮겨왔기 때문이다 — agent 가 이 목록을 읽어 runtime 보류
    (deferred)를 리포트에 싣는다(구현계획 §8.3).
    """
    entries: list[dict[str, Any]] = []
    for candidate in selection.rejected:
        entry: dict[str, Any] = {
            "action_key": candidate.action_key,
            "reason": candidate.reason,
            "supporting_labels": list(candidate.supporting_labels),
            "prescription_ids": list(
                dict.fromkeys(
                    support.prescription_id
                    for support in candidate.supports
                    if support.prescription_id
                )
            ),
            "eligibility": dict(candidate.metadata.get("eligibility") or {}),
        }
        grounding = _selected_grounding(candidate)
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
    # 사전검증 입력은 **지지 라벨 전체**의 finding 에서 모은다. 대표 라벨 하나만 쓰면
    # 같은 chunk 축을 함께 지지한 다른 라벨의 gold span 이 측정에서 빠져, 그 라벨이
    # 요구하는 경계까지 만족하는지 확인하지 못한 채 후보가 통과한다.
    findings = _supporting_findings(selection, action.supporting_labels)

    # 후보가 여러 개면 sweep 이 실측으로 고른다. chunk 축은 사전검증 결과가
    # 있어야 internal 로 넘긴다(기존 정책 유지).
    chunk_precheck_context = None
    if path in _CHUNK_PRECHECK_PATHS:
        chunk_precheck_context = _chunk_precheck_context(
            state,
            findings,
            path=path,
            evidence_analysis=_merged_evidence_analysis(
                state, selection, action.supporting_labels, findings
            ),
        )
    grounding = _selected_grounding(action)
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
        prescription_id=primary_support.prescription_id or None,
    )


def _supporting_findings(
    selection: _ActionSelection,
    labels: list[str],
) -> list[Finding]:
    """지지 라벨들의 Finding 을 순서를 지키며 합친다(중복 객체 제거)."""
    merged: list[Finding] = []
    seen: set[str] = set()
    for label in labels:
        for finding in selection.findings_by_label.get(label, []):
            if finding.finding_id in seen:
                continue
            seen.add(finding.finding_id)
            merged.append(finding)
    return merged


def _merged_evidence_analysis(
    state: AgentDoctorState,
    selection: _ActionSelection,
    labels: list[str],
    findings: list[Finding],
) -> _EvidenceAnalysis | None:
    """지지 라벨 전체에 대한 evidence window 분석. 라벨 하나면 캐시를 재사용한다.

    청크 경계 분석은 비싸다. 라벨이 하나뿐인 흔한 경우에는 후보값 계산이 이미 만들어
    둔 결과를 그대로 쓰고, 여러 라벨이 같은 축을 지지할 때만 합친 입력으로 다시 계산해
    조합 키로 캐시한다.
    """
    if not labels:
        return None
    if len(labels) == 1:
        cached = selection.evidence_cache.get(labels[0])
        if cached is not None:
            return cached
    key = "\x00".join(sorted(labels))
    if key not in selection.evidence_cache:
        selection.evidence_cache[key] = _evidence_windows(state, findings)
    return selection.evidence_cache[key]


def _selected_grounding(action: Any) -> dict[str, Any] | None:
    """선택된 action 의 후보값이 **어떻게 나왔는지**.

    ⚠️ 이 값은 표시용이 아니다. `_should_sweep` 이 chunk 축의 사전검증 여부를 이걸로
    가른다. 예전에는 support 목록의 첫 원소를 그대로 돌려줘서, 같은 축을 여러 라벨이
    지지할 때 근거 계산에 **실패한** 쪽이 앞에 오면 실패 status 가 보고됐다 —
    실측 후보가 있는데도 사전검증을 건너뛰고 rules 로 떨어졌다. **finding 순서가
    backend 를 바꾼 것**이다.

    그래서 두 가지를 지킨다.
      1. 살아남은 후보값을 **실제로 제공한** support 의 근거를 쓴다 — 값과 설명이
         어긋나면 리포트가 시험하지도 않은 계산을 근거로 든다.
      2. 그중 실측(`is_grounded`)을 우선하고, 남는 동률은 라벨 사전순으로 가른다.
         어떤 선택이든 **입력 순서와 무관해야** 하기 때문이다.
    """
    values = set(map(str, next(iter(action.search_space.values()), [])))

    def rank(support: Any) -> tuple[int, int, str]:
        contributed = bool(values & set(map(str, support.candidate_values)))
        return (0 if contributed else 1, 0 if support.is_grounded else 1, support.label)

    for support in sorted(action.supports, key=rank):
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
