"""
agents/optimize/history.py
Optimize 모듈의 "판정·기록" 계층.

[역할]
  planner 가 고른 처방이 실제로 적용된 뒤(Index→Eval 재측정 후) 다시 Optimize 로
  돌아왔을 때, 지난 처방이 좋았는지 나빴는지 판정하고 이력 항목을 만들어 준다.
    1) check_floor         : 모든 지표가 하한선 위인지 검사 (위반 지표 반환)
    2) judge               : 하한선 위반 → 무조건 롤백, 아니면 점수 비교로 유지/롤백
    3) create_pending_item : 처방 적용 시점의 pending 이력 항목 생성(before 만 채움)
    4) finalize_item       : 다음 방문에서 판정 결과로 pending 항목 확정(after+verdict)
    5) find_pending        : 판정 대기 중(pending)인 최근 이력 항목 조회

  판정이 다음 Eval 재측정 후에야 가능하므로(시점 문제), 이력은 두 단계로 기록한다.
  적용 시점엔 before 만 담은 pending 항목을 만들고, 다음 방문에서 확정한다.
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


def _read_composite(report: DiagnosticReport | None) -> float | None:
    """설계 종합점수(composite_score.total, 0~100)를 읽는다. 표시·게이트용.
    (판정은 overall 로 하고 이 값은 리포트에 함께 실어주기 위한 것 — 없으면 None.)"""
    if report is None:
        return None
    total = (report.composite_score or {}).get("total")
    return float(total) if total is not None else None


def judge(
    before_report: DiagnosticReport,
    after_report: DiagnosticReport,
) -> Verdict:
    """
    처방 전후 Eval 리포트를 비교해 유지/롤백을 판정한다. (CONTEXT.md 5번)
      ① after 가 하한선을 위반하면 → 무조건 롤백 (지표별 원값으로 검사)
      ② Eval 단일 점수가 올랐으면 → 유지, 아니면 → 롤백
    단일 점수는 Eval 이 계산한 overall_score 를 그대로 쓴다(여기서 재계산 안 함).

    ── 탐색 신호는 overall 이어야 한다 (composite 로 바꾸지 말 것) ──────────────
    표시·게이트(gate.py)는 설계 종합점수 composite(품질×신뢰도 조화평균)를 쓰지만,
    최적화 탐색(이 judge 의 유지/롤백)은 반드시 overall(품질 단일축, 연속값)을 써야 한다.
    이유: composite 의 조화평균은 신뢰도가 낮은 구간(=최적화가 가장 필요한 지점)에서
    거의 평평해지거나 0 으로 붕괴해, "합격선을 아직 못 넘은 부분 진전"을 신호로 못 준다.
    그러면 통과선 직전의 좋은 처방까지 롤백돼 최적화가 구멍에서 못 나온다.
    overall 은 통과 판정을 이루는 지표(recall·faithfulness 등)의 연속 평균이라 신뢰도가
    오를 방향으로 매끄럽게 움직인다 → 등반용 나침반으로 적합. "매끄러운 대리지표로 탐색,
    정직한 종합점수(composite)로 게이트/표시"라는 분리가 이 설계의 핵심이다.
    (composite 를 탐색에 넣으려면 조화평균이 아닌 매끄러운 blend 여야 하며, 근거 측정
     없이는 도입하지 않는다.)
    """
    before_score = _read_score(before_report)
    after_score = _read_score(after_report)
    # 판정은 overall(탐색 신호)로 하되, 표시·게이트용 composite 도 함께 실어 보낸다.
    before_composite = _read_composite(before_report)
    after_composite = _read_composite(after_report)

    violations = check_floor(after_report.ragas_scores)
    if violations:
        return Verdict(
            keep=False,
            before_score=before_score,
            after_score=after_score,
            before_composite=before_composite,
            after_composite=after_composite,
            floor_violations=violations,
            reason=f"하한선 위반 {violations} → 무조건 롤백",
        )

    if after_score > before_score:
        return Verdict(
            keep=True,
            before_score=before_score,
            after_score=after_score,
            before_composite=before_composite,
            after_composite=after_composite,
            reason=f"점수 상승 {before_score:.1f}→{after_score:.1f} → 유지",
        )

    return Verdict(
        keep=False,
        before_score=before_score,
        after_score=after_score,
        before_composite=before_composite,
        after_composite=after_composite,
        reason=f"점수 미상승 {before_score:.1f}→{after_score:.1f} → 롤백",
    )


# ── 3. 이력 항목: pending 생성 → 확정 (2단계) ──────────────────────
# 처방이 좋았는지는 다음 Eval 재측정 후에야 알 수 있어(시점 문제) 이력을 두 단계로
# 나눈다. 적용 시점엔 create_pending_item(before 만), 다음 방문에서 finalize_item
# (after+verdict)으로 확정한다. state.optimization_history 에 append/수정하는 것은
# agent.py 가 한다(판정만 하고 반영은 조율자가 — planner 와 같은 규칙).

def create_pending_item(
    state: AgentDoctorState,
    request: OptimizationRequest,
    prescription_id: str,
    before_config: dict,
    before_report: DiagnosticReport | None,
) -> OptimizationHistoryItem:
    """처방 적용 시점의 'pending' 이력 항목을 만든다.
    after/verdict 는 아직 없다 — 다음 방문에서 finalize_item 으로 채운다.
    before_report 는 다음 방문의 judge 에 쓰려고 metadata 에 잠시 보관한다."""
    before_scores = dict(before_report.ragas_scores) if before_report else {}
    return OptimizationHistoryItem(
        trial_id=str(uuid.uuid4()),
        request_id=request.request_id,
        iteration=state.iteration,
        failure_labels=[request.failure_label],
        optimizer=request.optimizer,
        status="applied",   # config 적용됨, 다음 Eval 검증 대기 (AGENTS.md §8)
        selected_prescription_id=prescription_id,
        before_config=dict(before_config),
        after_config={},           # 미확정
        before_metrics=before_scores,
        after_metrics={},          # 비어있음
        target_metrics=list(request.target_metrics),
        reason=request.reason,
        rollback_reason=None,
        metadata={
            "pending": True,               # 아직 판정되지 않음
            "before_report": before_report,  # judge 용(MVP: 객체 참조 보관)
            "before_composite": _read_composite(before_report),  # 표시용 baseline 종합점수
        },
    )


def find_pending(
    history: list[OptimizationHistoryItem],
) -> OptimizationHistoryItem | None:
    """판정 대기 중(pending)인 가장 최근 이력 항목을 찾는다. 없으면 None."""
    for item in reversed(history):
        if item.metadata.get("pending"):
            return item
    return None


def find_active_study(
    history: list[OptimizationHistoryItem],
) -> OptimizationHistoryItem | None:
    """후보 sweep를 진행 중인 가장 최근 라벨 study를 찾는다.

    별도 ActiveStudy 스키마를 추가하지 않고 기존 이력 항목의 metadata를
    사용한다. ``pending``은 다음 Eval을 기다린다는 뜻이고,
    ``active_study``는 개별 후보가 아니라 전체 후보 묶음의 판정이 아직
    끝나지 않았다는 뜻이다.
    """
    for item in reversed(history):
        if item.metadata.get("pending") and item.metadata.get("active_study"):
            return item
    return None


def last_failure_label(
    history: list[OptimizationHistoryItem],
) -> str | None:
    """가장 최근에 실제 적용을 시작한 라벨을 반환한다."""
    for item in reversed(history):
        if item.failure_labels:
            return item.failure_labels[0]
    return None


def finalize_item(
    item: OptimizationHistoryItem,
    verdict: Verdict,
    after_config: dict,
    after_report: DiagnosticReport | None,
) -> None:
    """pending 항목을 판정 결과로 확정한다(제자리 수정).
    유지면 status='applied', 롤백이면 'failed'+rollback_reason 을 기록한다.
    after_config 는 롤백 전의 '실제 적용된(=측정된)' config 다."""
    item.after_config = dict(after_config)
    item.after_metrics = dict(after_report.ragas_scores) if after_report else {}
    item.status = "applied" if verdict.keep else "failed"
    item.rollback_reason = None if verdict.keep else verdict.reason
    item.metadata["pending"] = False
    item.metadata["before_score"] = verdict.before_score
    item.metadata["after_score"] = verdict.after_score
    # 표시·게이트용 종합점수(0~100). verdict 가 실어줬으면 그걸, 아니면 리포트에서 직접.
    before_report = item.metadata.get("before_report")
    item.metadata["before_composite"] = (
        verdict.before_composite if verdict.before_composite is not None
        else _read_composite(before_report)
    )
    item.metadata["after_composite"] = (
        verdict.after_composite if verdict.after_composite is not None
        else _read_composite(after_report)
    )
    item.metadata["floor_violations"] = verdict.floor_violations
    item.metadata.pop("before_report", None)   # 판정 끝 — 무거운 참조 제거
