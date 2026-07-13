"""
agents/optimize/history.py
Optimize 모듈의 "판정·기록" 계층.

[역할]
  planner 가 고른 처방이 실제로 적용된 뒤(Index→Eval 재측정 후) 다시 Optimize 로
  돌아왔을 때, 지난 처방이 좋았는지 나빴는지 판정하고 이력 항목을 만들어 준다.
    1) check_floor        : 모든 지표가 하한선 위인지 검사 (위반 지표 반환)
    2) judge              : 하한선 위반 → 무조건 롤백, 아니면 점수 비교로 유지/롤백
    3) build_history_item : 이번 시도를 OptimizationHistoryItem 으로 만들어 반환

  "판정과 이력 항목 생성"까지가 history 의 책임이다. state 를 직접 수정하지 않고
  값만 반환하며, 실제 반영(블랙리스트 추가·이력 append·config 롤백)은 agent.py 가 한다.
  (planner 가 (request, decision) 만 반환하고 state 는 안 건드린 것과 같은 방식.)
  "무엇을 처방할지"는 planner, "그 처방을 config 로 바꾸는 일"은 config_mapper 소관.

[읽는 것]  before/after 리포트(DiagnosticReport.overall_score, ragas_scores), state.iteration
[쓰는 것]  없음 — 반환값을 agent.py 가 state.optimization_history / state.blacklist 에 반영

[MVP 결정 사항]  (나중에 재검토 가능)
  - 단일 점수(overall_score)는 Eval 이 계산한다. history 는 그 값을 읽어 비교만 한다.
    (여기서 점수 공식을 다시 만들지 않는다 — Eval 과 다른 값이 나오면 안 되므로.)
  - FLOORS(하한선)는 Optimize 고유의 롤백 안전망이라 여기서 관리한다. 지금은 임시
    더미값이며, Eval 임계값이 확정되면 그에 맞춰 조정한다.
  - 롤백 판정은 원값(반올림 전)으로 한다. 사용자 표시용 반올림은 reporter 소관.
"""
from __future__ import annotations

import uuid

from core.schema import DiagnosticReport
from core.state import AgentDoctorState
from agents.optimize.schemas import (
    OptimizationHistoryItem,
    OptimizationRequest,
    Verdict,
)


# ── 하한선 기준값 (Optimize 롤백 안전망) ──────────────────────────
# RAGAS 6개 지표. noise_sensitivity 만 "낮을수록 좋음".
_LOWER_IS_BETTER: set[str] = {"noise_sensitivity"}

# 하한선: 이 밑(낮을수록 좋은 지표는 이 위)으로 떨어지면 무조건 롤백.
# "통과 기준"과 별개인 '절대 넘으면 안 되는 안전선'이라 Optimize 가 관리한다.
# TODO: ⚠️ 전부 임시 더미값. Eval 임계값 확정 시 그에 맞춰 조정.
_FLOORS: dict[str, float] = {
    "faithfulness": 0.50,
    "answer_relevancy": 0.50,
    "context_recall": 0.45,
    "context_precision": 0.45,
    "context_utilization": 0.40,
    "noise_sensitivity": 0.60,   # 낮을수록 좋음 → 0.60 초과면 위반
}


# ── 1. 하한선 검사 ────────────────────────────────────────────────

def check_floor(metrics: dict[str, float]) -> list[str]:
    """모든 지표를 하한선과 비교해 위반한 지표명을 반환. (비면 통과)"""
    violations: list[str] = []
    for metric, value in metrics.items():
        floor = _FLOORS.get(metric)
        if floor is None:
            continue
        if metric in _LOWER_IS_BETTER:
            if value > floor:       # 낮을수록 좋음 → 상한 초과가 위반
                violations.append(metric)
        else:
            if value < floor:       # 높을수록 좋음 → 하한 미달이 위반
                violations.append(metric)
    return violations


# ── 2. 유지/롤백 판정 ─────────────────────────────────────────────

def _read_score(report: DiagnosticReport) -> float:
    """Eval 이 계산한 단일 점수(overall_score)를 읽는다.
    아직 없으면 0.0 으로 보수적으로 취급한다."""
    return report.overall_score if report.overall_score is not None else 0.0


def judge(
    before_report: DiagnosticReport,
    after_report: DiagnosticReport,
) -> Verdict:
    """
    처방 전후 Eval 리포트를 비교해 유지/롤백을 판정한다. (CONTEXT.md 5번)
      ① after 가 하한선을 위반하면 → 무조건 롤백 (지표별 원값으로 검사)
      ② Eval 단일 점수가 올랐으면 → 유지, 아니면 → 롤백
    단일 점수는 Eval 이 계산한 overall_score 를 그대로 쓴다(여기서 재계산 안 함).
    """
    before_score = _read_score(before_report)
    after_score = _read_score(after_report)

    violations = check_floor(after_report.ragas_scores)
    if violations:
        return Verdict(
            keep=False,
            before_score=before_score,
            after_score=after_score,
            floor_violations=violations,
            reason=f"하한선 위반 {violations} → 무조건 롤백",
        )

    if after_score > before_score:
        return Verdict(
            keep=True,
            before_score=before_score,
            after_score=after_score,
            reason=f"점수 상승 {before_score:.1f}→{after_score:.1f} → 유지",
        )

    return Verdict(
        keep=False,
        before_score=before_score,
        after_score=after_score,
        reason=f"점수 미상승 {before_score:.1f}→{after_score:.1f} → 롤백",
    )


# ── 3. 이력 항목 생성 ─────────────────────────────────────────────
# 블랙리스트 추가·이력 append 는 여기서 하지 않는다. agent.py 가 반환값을 받아
#   state.blacklist.add((label, prescription_id))
#   state.optimization_history.append(item)
# 형태로 직접 반영한다. (판정만 하고 반영은 조율자가 — planner 와 같은 규칙)

def build_history_item(
    state: AgentDoctorState,
    request: OptimizationRequest,
    prescription_id: str,
    verdict: Verdict,
    before_config: dict,
    after_config: dict,
    before_metrics: dict[str, float],
    after_metrics: dict[str, float],
) -> OptimizationHistoryItem:
    """이번 처방 시도를 OptimizationHistoryItem 으로 만들어 반환한다.
    state 에 append 하지 않는다 — 반영은 agent.py 가 한다."""
    return OptimizationHistoryItem(
        trial_id=str(uuid.uuid4()),
        request_id=request.request_id,
        iteration=state.iteration,
        failure_labels=[request.failure_label],
        optimizer=request.optimizer,
        status="applied" if verdict.keep else "failed",
        selected_prescription_id=prescription_id,
        before_config=dict(before_config),
        after_config=dict(after_config),
        before_metrics=dict(before_metrics),
        after_metrics=dict(after_metrics),
        target_metrics=list(request.target_metrics),
        reason=request.reason,
        rollback_reason=None if verdict.keep else verdict.reason,
        metadata={
            "before_score": verdict.before_score,
            "after_score": verdict.after_score,
            "floor_violations": verdict.floor_violations,
        },
    )
