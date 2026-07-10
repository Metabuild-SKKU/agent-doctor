"""
agents/optimize/planner.py
Optimize 모듈의 "결정" 계층.

[역할]
  Eval의 진단 리포트(state.report.findings)를 받아서
    1) 자동 처방 대상인지 3분류하고 (manual / actionable / 스킵)
    2) 최적화 흐름을 결정하고 (OptimizeDecision: 제안/적용/유지/수동)
    3) 최상위 우선순위 라벨 하나를 골라 처방 후보를 만들어
    4) OptimizationRequest 로 묶어 optimizer 에게 넘긴다.

  "무엇을, 어느 순서로" 까지가 planner 의 책임이다.
  "그 처방을 실제 config 값으로 바꾸는 일"은 config_mapper/adapters,
  "적용 후 좋아졌는지 판단·롤백"은 optimizer 결과 + history 소관이다.

[읽는 것]  state.report(Finding 목록), state.index_config, state.iteration, blacklist
[쓰는 것]  (state 를 직접 수정하지 않음. agent.py 가 반환값을 받아 반영한다.)

[MVP 결정 사항]  (planner 설계 시 확정, 나중에 재검토 가능)
  - 우선순위: 1차로 그룹(A>C>B) 정렬, 2차로 점수 정렬.
    D그룹은 manual 이라 자동처방 대상에서 빠진다.
  - 점수 = 빈도 × 진단신뢰도 ÷ 처방비용
      빈도   = len(finding.affected_probes) (최소 1)
      신뢰도 = rules.py diagnosis_confidence (None 이면 1.0 fallback)
      비용   = rules.py cost 가 None 이므로 reindex 로 유도 (런타임=1, 재색인=3)
  - target_metrics 는 rules.py 라벨의 target_metrics 를 읽어 실어 보낸다.
    guardrail 은 폐기 — 롤백은 전역 하한선 체크 + 점수 비교로 대체(history.py).
  - propose_only(제안만) 모드는 뼈대만. 현재 기본은 apply_optimize.
"""
from __future__ import annotations

import uuid

from core.state import AgentDoctorState
from core.schema import Finding
from agents.optimize import rules
from agents.optimize.schemas import (
    ConfigPatch,
    OptimizationRequest,
    OptimizeDecision,
    PrescriptionCandidate,
)


# ── 상수 ──────────────────────────────────────────────────────────
# 그룹 1차 우선순위. 값이 작을수록 먼저 처리.
# D(데이터 문제)는 manual 이라 자동처방 대상이 아니므로 실질 순서는 A > C > B.
_GROUP_ORDER: dict[str, int] = {"A": 0, "C": 1, "B": 2, "D": 3}

# MVP fallback 상수 (rules.py 값이 None 일 때 사용)
_DEFAULT_CONFIDENCE = 1.0
_COST_RUNTIME = 1
_COST_REINDEX = 3


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
    blacklist = blacklist or set()

    if state.report is None:
        return None, OptimizeDecision(
            mode="use_current",
            status="skipped",
            requires_user_confirmation=False,
            next_route="serve",
            reason="진단 리포트가 없음 — 최적화 스킵",
        )

    manual, actionable = _split_findings(state.report.findings)
    decision = _decide_mode(state, actionable, manual)

    if decision.mode != "apply_optimize":
        return None, decision

    ranked = _rank_findings(actionable)
    picked = _pick_top(ranked, blacklist)
    if picked is None:
        # 점수는 났지만 후보가 전부 블랙리스트 → 처방할 게 없음
        return None, OptimizeDecision(
            mode="use_current",
            status="skipped",
            requires_user_confirmation=False,
            next_route="serve",
            reason="처방 후보가 모두 블랙리스트에 걸림",
        )

    finding, rule, _score_val = picked
    candidates = _build_candidates(finding, rule, blacklist)
    request = _build_request(finding, rule, candidates, ranked, state)
    decision.request_id = request.request_id
    return request, decision


# ── 1. 분류 ───────────────────────────────────────────────────────

def _split_findings(
    findings: list[Finding],
) -> tuple[list[Finding], list[Finding]]:
    """
    finding 을 (manual, actionable) 로 나눈다.
      - manual     : is_manual (D그룹) → 자동처방 불가, reporter 로 넘어감
      - actionable : is_actionable (ready + 처방 있음) → 점수 경쟁 대상
      - 나머지     : draft/unassigned/라벨없음 → 지금은 실행 불가, 스킵
    """
    manual: list[Finding] = []
    actionable: list[Finding] = []
    for f in findings:
        label = f.label
        if not label:
            continue  # 세분화 라벨이 없으면 rules.py 매핑 불가
        if rules.is_manual(label):
            manual.append(f)
        elif rules.is_actionable(label):
            actionable.append(f)
        # draft/unassigned 은 의도적으로 스킵
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
    report = state.report
    if report is not None and report.pass_threshold:
        return OptimizeDecision(
            mode="use_current",
            status="already_optimal",
            requires_user_confirmation=False,
            next_route="serve",
            reason="모든 임계값 달성 — 최적화 불필요",
        )

    if not actionable:
        if manual:
            return OptimizeDecision(
                mode="manual_required",
                status="manual_required",
                requires_user_confirmation=True,
                next_route="serve",
                reason="자동 처방 가능한 라벨 없음 — 사람 개입 필요(D그룹)",
            )
        return OptimizeDecision(
            mode="use_current",
            status="skipped",
            requires_user_confirmation=False,
            next_route="serve",
            reason="처방 가능한 finding 없음",
        )

    return OptimizeDecision(
        mode="apply_optimize",
        status="proposed",
        requires_user_confirmation=False,
        next_route="index",
        reason="처방 가능한 finding 존재 → 최적화 진행",
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


def _score(finding: Finding, rule: dict) -> float:
    """우선순위점수 = 빈도 × 진단신뢰도 ÷ 처방비용."""
    frequency = max(len(finding.affected_probes), 1)
    confidence = rule.get("diagnosis_confidence")
    if confidence is None:
        confidence = _DEFAULT_CONFIDENCE
    cost = _label_cost(rule)
    return (frequency * confidence) / cost


def _rank_findings(
    actionable: list[Finding],
) -> list[tuple[Finding, dict, float]]:
    """actionable finding 을 (그룹순서, 점수내림차순)으로 정렬.
    1차 키 = 그룹(A>C>B), 2차 키 = 점수 높은 순."""
    ranked: list[tuple[Finding, dict, float]] = []
    for f in actionable:
        rule = rules.get_rule(f.label)
        if not rule:
            continue
        ranked.append((f, rule, _score(f, rule)))
    ranked.sort(
        key=lambda item: (
            _GROUP_ORDER.get(item[1].get("group"), 99),
            -item[2],
        )
    )
    return ranked


# ── 4. 최상위 선택 + 블랙리스트 ───────────────────────────────────

def _available_prescriptions(
    rule: dict, label: str, blacklist: set[tuple[str, str]]
) -> list[dict]:
    """블랙리스트에 걸리지 않은 처방만 순서대로 반환."""
    return [
        p
        for p in rule.get("prescriptions", [])
        if (label, p["id"]) not in blacklist
    ]


def _pick_top(
    ranked: list[tuple[Finding, dict, float]],
    blacklist: set[tuple[str, str]],
) -> tuple[Finding, dict, float] | None:
    """정렬된 목록에서 '아직 시도할 처방이 남은' 최상위 finding 을 고른다."""
    for finding, rule, score in ranked:
        if _available_prescriptions(rule, finding.label, blacklist):
            return finding, rule, score
    return None


# ── 5. 처방 후보 생성 (rules.py → PrescriptionCandidate) ──────────

def _build_candidates(
    finding: Finding,
    rule: dict,
    blacklist: set[tuple[str, str]],
) -> list[PrescriptionCandidate]:
    """
    rules.py 의 raw dict 처방들을 PrescriptionCandidate 객체로 변환한다.
    rules.py 에 적힌 순서(가벼운 것 먼저)를 그대로 유지한다.
    블랙리스트에 걸린 처방은 제외한다.
    """
    candidates: list[PrescriptionCandidate] = []
    target_metrics = list(rule.get("target_metrics", []))
    for pres in _available_prescriptions(rule, finding.label, blacklist):
        patch = ConfigPatch(
            changes=dict(pres.get("patch", {})),
            reindex_required=bool(pres.get("reindex")),
            description=f"{finding.label} → {pres['id']}",
            metadata={"prescription_id": pres["id"]},
        )
        # applies_when(신호기반 택1) 은 optimizer 가 finding.metadata 와
        # 대조해 쓰도록 후보 metadata 로 실어 보낸다. 없으면 넣지 않는다.
        meta: dict = {}
        if pres.get("applies_when"):
            meta["applies_when"] = pres["applies_when"]

        candidates.append(
            PrescriptionCandidate(
                id=pres["id"],
                failure_label=finding.label,
                group=rule.get("group"),
                status=rule.get("status"),
                patch=patch,
                cost=float(_derive_cost(pres)),
                priority=0.0,          # 후보 개별 우선순위는 MVP 미사용
                target_metrics=list(target_metrics),  # rules.py 라벨의 target_metrics
                reason=finding.description,
                metadata=meta,
            )
        )
    return candidates


# ── 6. 요청서 포장 ────────────────────────────────────────────────

def _build_request(
    finding: Finding,
    rule: dict,
    candidates: list[PrescriptionCandidate],
    ranked: list[tuple[Finding, dict, float]],
    state: AgentDoctorState,
) -> OptimizationRequest:
    """선택된 라벨과 처방 후보를 OptimizationRequest 로 묶는다."""
    related = [f.label for f, _rule, _s in ranked if f.label != finding.label]
    return OptimizationRequest(
        request_id=str(uuid.uuid4()),
        iteration=state.iteration,
        baseline_config=dict(state.index_config),
        failure_label=finding.label,
        related_failure_labels=related,
        candidates=candidates,
        target_metrics=list(rule.get("target_metrics", [])),  # 라벨의 target_metrics
        target_profile="balanced",
        optimizer="internal",
        max_trials=1,
        reason=f"우선순위 최상위 라벨: {finding.label}",
        propose_only=False,
    )
