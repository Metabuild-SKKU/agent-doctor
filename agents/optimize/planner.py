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
from typing import Any

from core.state import AgentDoctorState
from core.schema import Document, Finding
from agents.optimize import rules
from agents.optimize import gate
from agents.optimize import history
from agents.optimize.config_mapper import canonicalize_path, get_current_value
from agents.optimize.evidence_window import build_evidence_windows
from agents.optimize.schemas import (
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

    ranked = _rank_groups(_group_by_label(actionable))
    picked = _pick_top(ranked, blacklist)
    if picked is None:
        # 점수는 났지만 후보가 전부 블랙리스트 → 처방할 게 없음
        return None, OptimizeDecision(
            mode="use_current",
            status="skipped",
            requires_user_confirmation=False,
            next_route="serve",
            reason="처방 후보가 모두 블랙리스트에 걸림",
            manual_labels=decision.manual_labels,
        )

    label, findings, rule, _score_val = picked
    evidence_analysis = (
        _evidence_windows(state, findings)
        if _rule_uses_chunk_size(rule)
        else None
    )
    candidates = _build_candidates(
        label,
        findings,
        rule,
        blacklist,
        state,
        evidence_analysis=evidence_analysis,
    )
    request = _build_request(
        label,
        findings,
        rule,
        candidates,
        ranked,
        state,
        evidence_analysis=evidence_analysis,
    )
    # legacy 가 이미 계산한 evidence 분석을 shadow 가 재사용하도록 캐시로 넘긴다.
    # 관측 때문에 비싼 청크 경계 분석을 두 번 돌리면 안 된다.
    evidence_cache: dict[str, Any] = {}
    if evidence_analysis is not None:
        evidence_cache[label] = evidence_analysis
    request.metadata["shadow_action_selection"] = _shadow_action_selection(
        state,
        actionable,
        label,
        candidates[0].id if candidates else None,
        evidence_cache,
    )
    decision.request_id = request.request_id
    return request, decision


# ── shadow mode: action 선택을 함께 계산해 비교만 한다 ─────────────
# 실제 적용은 위의 legacy 경로가 그대로 한다. 여기서는 "action 중심으로 골랐다면
# 무엇을 골랐을까" 를 계산해 request metadata 에 남긴다.
#
# 목적은 전환 전에 **실익을 관측**하는 것이다. 선택이 전혀 달라지지 않거나 tie-break
# 차이뿐이라면 선택 로직 전환을 보류한다(계획서 §8 단계 3 중단 기준). 구조(catalog·
# eligibility·aggregator)는 중복 선언 제거와 정책 단일화만으로도 유지 가치가 있다.

def _shadow_action_selection(
    state: AgentDoctorState,
    actionable: list[Finding],
    legacy_label: str,
    legacy_prescription_id: str | None,
    evidence_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """legacy 선택과 action 선택을 비교한다.

    **실패해도 legacy 경로를 깨지 않는다.** shadow 는 관측용이므로 어떤 예외도
    최적화 자체를 막아서는 안 된다.
    """
    try:
        return _compute_shadow_selection(
            state,
            actionable,
            legacy_label,
            legacy_prescription_id,
            evidence_cache if evidence_cache is not None else {},
        )
    except Exception as exc:  # noqa: BLE001 - 관측 실패가 최적화를 막으면 안 된다
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _compute_shadow_selection(
    state: AgentDoctorState,
    actionable: list[Finding],
    legacy_label: str,
    legacy_prescription_id: str | None,
    evidence_cache: dict[str, Any],
) -> dict[str, Any]:
    from agents.optimize import action_aggregator, action_catalog

    grouped = _group_by_label(actionable)

    def search_space_for(label, findings, changes):
        # 청크 경계 분석은 비싸다. 라벨 단위로 캐시해 legacy 가 이미 계산한 것을
        # 재사용하고, shadow 때문에 같은 분석이 두 번 돌지 않게 한다.
        analysis = None
        if any(canonicalize_path(path) == "chunker.chunk_size" for path in changes):
            if label not in evidence_cache:
                evidence_cache[label] = _evidence_windows(state, findings)
            analysis = evidence_cache[label]
        return _finding_search_space(findings, changes, state, analysis)

    supports = action_aggregator.build_action_supports(
        grouped, state, search_space_for=search_space_for
    )
    candidates = action_aggregator.aggregate_action_candidates(supports, state)
    eligible, rejected = action_aggregator.filter_ineligible_actions(
        candidates,
        state,
        backend="rules",
        runtime_capabilities=state.runtime_capabilities,
    )
    kept, deferred = action_aggregator.resolve_action_conflicts(eligible)
    selected = action_aggregator.select_action(kept)

    legacy_key = _legacy_action_key(legacy_label, legacy_prescription_id)
    shadow_key = selected.action_key if selected else None

    return {
        "status": "ok",
        "legacy_label": legacy_label,
        "legacy_prescription_id": legacy_prescription_id,
        "legacy_action_key": legacy_key,
        "shadow_action_key": shadow_key,
        "agrees": bool(legacy_key) and legacy_key == shadow_key,
        "divergence_reason": _divergence_reason(legacy_key, shadow_key, selected),
        "shadow_score": round(selected.score, 6) if selected else None,
        "shadow_supporting_labels": (
            list(selected.supporting_labels) if selected else []
        ),
        "shadow_supporting_probe_count": (
            len(selected.supporting_probes) if selected else 0
        ),
        "shadow_score_breakdown": (
            dict(selected.score_breakdown) if selected else {}
        ),
        # 여러 label 이 같은 변경을 지지해 하나로 합쳐진 사례. 전환의 실익이 여기서
        # 관측된다 — 0 건이면 통합할 것이 없었다는 뜻이다.
        "merged_action_count": sum(
            1 for c in candidates if len(c.supporting_labels) > 1
        ),
        "eligible_action_count": len(kept),
        "rejected_actions": [
            {"action_key": c.action_key, "reason": c.reason} for c in rejected
        ],
        "deferred_axes": deferred,
        "catalog_size": len(action_catalog.ACTION_CATALOG),
    }


def _legacy_action_key(label: str, prescription_id: str | None) -> str | None:
    """legacy 가 고른 처방을 action key 로 환산한다(같은 잣대로 비교하기 위해)."""
    from agents.optimize import action_catalog

    rule = rules.get_rule(label)
    if not rule or not prescription_id:
        return None
    for prescription in rule.get("prescriptions") or []:
        if prescription.get("id") != prescription_id:
            continue
        for raw_path, value in (prescription.get("patch") or {}).items():
            return action_catalog.build_action_key(raw_path, value)
    return None


def _divergence_reason(
    legacy_key: str | None,
    shadow_key: str | None,
    selected: Any,
) -> str | None:
    """선택이 갈린 이유를 거칠게 분류한다(중단 기준 판정용)."""
    if legacy_key == shadow_key:
        return None
    if shadow_key is None:
        return "shadow_found_no_eligible_action"
    if legacy_key is None:
        return "legacy_prescription_not_in_catalog"
    if selected is not None and len(selected.supporting_labels) > 1:
        # 여러 label 이 지지해 통합된 action 이 이긴 경우 — 전환의 진짜 실익이다.
        return "shared_support_won"
    if legacy_key.split(":")[0] != shadow_key.split(":")[0]:
        return "different_axis"
    return "different_operation"


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
