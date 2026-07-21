"""
agents/optimize/agent.py
Optimize 노드의 진입점(오케스트레이션 계층).

[역할]
  Eval 진단 리포트를 받아 planner → optimizer → config_mapper → history 를 엮는다.
  각 방문은 2단계다:
    (1) 지난 방문에 적용한 처방을 판정(judge)한다. 나빠졌으면 config 를 되돌리고
        (label, prescription_id)를 blacklist 에 넣는다. (방문 간 롤백)
    (2) 새 처방 하나를 골라 적용하고 pending 이력을 남긴다.
  모든 경로에서 같은 state 를 반환한다(AGENTS.md 2절 계약).

[읽는 것]  state.report, state.index_config, state.iteration, state.max_iterations,
           state.blacklist, state.optimization_history
[쓰는 것]  state.index_config, state.iteration, state.status, state.error,
           state.current_agent, state.blacklist, state.optimization_history,
           state.optimization_report

[state.status 신호]  (graph 라우팅이 참고)
  - "applied"      : 새 처방을 적용함 → 재색인 필요(Index)
  - "rolled_back"  : 롤백으로 config 를 되돌림 → 재색인 필요(Index)
  - 그 외(skipped/manual_required/already_optimal/verified/error) → 변경 없음(Serve)

[사용자 리포트]  매 방문마다 state.optimization_report 에 이번 방문 결과를 번역해
  저장한다(reporter). 방문마다 덮어써지며 Serve 가 마지막 방문 리포트를 읽는다.
  리포트가 설명하는 처방과 점수가 어긋나지 않도록, '판정'은 이력 항목 기반
  trial 리포트로, '새 적용/수동/유지'는 decision 기반 리포트로 나눠 만든다.
"""
from __future__ import annotations

from dataclasses import replace

from core.state import AgentDoctorState
from agents.optimize import planner, optimizer, config_mapper, history, reporter, gate
from agents.optimize.schemas import (
    OptimizationHistoryItem,
    OptimizationRequest,
    OptimizationResult,
    OptimizeDecision,
    Verdict,
)


def run(state: AgentDoctorState) -> AgentDoctorState:
    """Optimize 노드 진입점. 성공·스킵·수동·오류 어느 경로든 같은 state 를 반환한다."""
    state.current_agent = "optimize"
    try:
        # top_k sweep는 후보 하나의 성공/실패를 곧바로 확정하지 않는다.
        # 직전 후보의 Eval 결과를 같은 study에 넣고 다음 후보 또는 best를 고른다.
        active_study = history.find_active_study(state.optimization_history)
        if active_study is not None:
            return _continue_internal_study(state, active_study)

        # (1) 지난 처방 판정 + (나빴으면) 롤백/블랙리스트
        judged_item, verdict = _judge_pending_trial(state)
        rolled_back = judged_item is not None and not verdict.keep

        # (2) 새 처방 선택. 저비용 사전검증에서 baseline이 이기면 현재 처방을
        # 소진 처리하고, 재색인·iteration 증가 없이 같은 방문에서 다음 처방을 고른다.
        while True:
            request, decision = planner.plan(state, blacklist=state.blacklist)
            if decision.mode != "apply_optimize" or request is None:
                state.status = "rolled_back" if rolled_back else decision.status
                # 롤백이 있었으면 그게 headline, 아니면 흐름 결정(수동/유지)을 보고.
                state.optimization_report = (
                    reporter.build_trial_report(judged_item, verdict)
                    if rolled_back
                    else reporter.build_report(decision, request)
                )
                return state

            previous_label = history.last_failure_label(state.optimization_history)
            starts_new_label = (
                previous_label is None or previous_label != request.failure_label
            )
            if starts_new_label and state.iteration >= state.max_iterations:
                state.status = "rolled_back" if rolled_back else "verified"
                if judged_item is not None:
                    state.optimization_report = reporter.build_trial_report(
                        judged_item, verdict
                    )
                return state

            result = optimizer.run(request)
            if result.metadata.get("error_code") != "baseline_selected":
                break

            prescription_id = (
                result.selected_candidate.id if result.selected_candidate else None
            )
            rejection = (request.failure_label, prescription_id)
            if not prescription_id or rejection in state.blacklist:
                state.status = "rolled_back" if rolled_back else "skipped"
                if rolled_back:
                    state.optimization_report = reporter.build_trial_report(
                        judged_item, verdict
                    )
                return state
            state.blacklist.add(rejection)

        if result.status != "proposed" or result.config_patch is None:
            # optimizer 가 적용 가능한 patch 를 못 만듦(skipped/failed)
            state.status = "rolled_back" if rolled_back else result.status
            if result.status == "failed":
                state.error = result.error or result.message
            # 롤백만 리포트로 남긴다. 적용 실패 자체는 status/error 로 전달(MVP 한계).
            if rolled_back:
                state.optimization_report = reporter.build_trial_report(judged_item, verdict)
            return state

        # 적용 직전 스냅샷(롤백이 있었다면 이미 되돌려진 config 가 before 가 된다)
        before_config = dict(state.index_config)
        before_report = state.report

        # 검증된 처방을 실제 index_config 에 반영(canonical→flat 변환은 mapper 담당)
        config_mapper.apply_config_patch(state.index_config, result.config_patch)
        state.reindex_required = bool(result.needs_reindex)

        # pending 이력 생성(다음 방문에서 finalize) + iteration 1회 증가
        prescription_id = (
            result.selected_candidate.id if result.selected_candidate else None
        )
        item = history.create_pending_item(
            state, request, prescription_id, before_config, before_report
        )
        item.metadata["reindex_required"] = bool(result.needs_reindex)
        state.optimization_history.append(item)
        if starts_new_label:
            state.iteration += 1

        # 여러 top_k 후보의 첫 적용이면 같은 이력 항목을 active study로 사용한다.
        # 후보별 결과는 다음 방문마다 metadata.trial_results에 누적한다.
        if result.metadata.get("adapter_status") == "needs_evaluation":
            item.metadata.update(
                {
                    "active_study": True,
                    "study_request": request,
                    "study_decision": decision,
                    "trial_results": list(result.metadata.get("trial_results", [])),
                    "current_candidate": dict(result.config_patch.changes),
                    "study_baseline_config": dict(before_config),
                    "reindex_required": bool(result.needs_reindex),
                }
            )
        state.status = "applied"
        # 새 처방을 적용함 → "적용, 다음 검증 대기" 리포트(verdict 없음).
        state.optimization_report = reporter.build_report(decision, request)
        return state

    except Exception as exc:  # 예외를 밖으로 전파하지 않고 state 에 기록(AGENTS.md 2절)
        state.status = "error"
        state.error = f"optimize 실행 실패: {exc}"
        return state


def _continue_internal_study(
    state: AgentDoctorState,
    item: OptimizationHistoryItem,
) -> AgentDoctorState:
    """직전 top_k 후보 결과를 기록하고 같은 라벨의 다음 후보를 진행한다."""
    request = item.metadata.get("study_request")
    if not isinstance(request, OptimizationRequest):
        return _fail_active_study(
            state,
            item,
            "active study에 원본 OptimizationRequest가 없습니다.",
        )

    current_candidate = item.metadata.get("current_candidate")
    if not isinstance(current_candidate, dict) or len(current_candidate) != 1:
        return _fail_active_study(
            state,
            item,
            "active study의 현재 후보가 올바르지 않습니다.",
        )

    observed_trials = list(item.metadata.get("trial_results", []))
    observed_trials.append(
        {
            "trial_id": f"{item.trial_id}:candidate:{len(observed_trials)}",
            "config": dict(current_candidate),
            "metrics": _report_metrics(state),
            "status": "completed" if state.report is not None else "failed",
            "error": None if state.report is not None else "Eval report가 없습니다.",
        }
    )
    resumed_request = replace(
        request,
        metadata={
            **dict(request.metadata),
            "study_baseline_config": dict(item.before_config),
            "trial_results": observed_trials,
        },
    )
    result = optimizer.run(resumed_request)
    item.metadata["trial_results"] = list(
        result.metadata.get("trial_results", observed_trials)
    )
    item.metadata["adapter_status"] = result.metadata.get("adapter_status")

    if (
        result.status == "proposed"
        and result.config_patch is not None
        and result.metadata.get("adapter_status") == "needs_evaluation"
    ):
        config_mapper.apply_config_patch(state.index_config, result.config_patch)
        state.reindex_required = bool(result.needs_reindex)
        item.metadata["current_candidate"] = dict(result.config_patch.changes)
        item.after_config = dict(state.index_config)
        state.status = "applied"
        decision = item.metadata.get("study_decision")
        if isinstance(decision, OptimizeDecision):
            state.optimization_report = reporter.build_report(decision, resumed_request)
        return state

    if result.metadata.get("adapter_status") == "completed":
        return _finish_internal_study(state, item, result)

    return _fail_active_study(
        state,
        item,
        result.error or result.message or "internal study를 완료하지 못했습니다.",
        retryable=True,
    )


def _finish_internal_study(
    state: AgentDoctorState,
    item: OptimizationHistoryItem,
    result: OptimizationResult,
) -> AgentDoctorState:
    """모든 후보 평가 뒤 best를 적용하거나 study baseline을 복원한다."""
    before_score = _report_score(item.metadata.get("before_report"))
    best_score = result.metadata.get("best_score")
    after_score = float(best_score) if isinstance(best_score, (int, float)) else before_score
    baseline_selected = result.metadata.get("error_code") == "baseline_selected"
    best_metrics = _best_trial_metrics(
        item.metadata.get("trial_results", []),
        result,
    )
    floor_violations = _study_floor_violations(best_metrics)

    if baseline_selected:
        changed = state.index_config != item.before_config
        state.index_config = dict(item.before_config)
        verdict = Verdict(
            keep=False,
            before_score=before_score,
            after_score=after_score,
            reason="모든 top_k 후보 평가 후 baseline이 가장 좋아 원래 설정을 유지",
        )
        label = item.failure_labels[0] if item.failure_labels else ""
        if label and item.selected_prescription_id:
            state.blacklist.add((label, item.selected_prescription_id))
        state.status = "rolled_back" if changed else "verified"
        state.reindex_required = bool(item.metadata.get("reindex_required", True))
    elif result.status == "proposed" and result.config_patch is not None:
        if floor_violations:
            changed = state.index_config != item.before_config
            state.index_config = dict(item.before_config)
            verdict = Verdict(
                keep=False,
                before_score=before_score,
                after_score=after_score,
                floor_violations=floor_violations,
                reason=f"sweep 최적 후보가 하한선을 위반함 {floor_violations} → baseline 복원",
            )
            label = item.failure_labels[0] if item.failure_labels else ""
            if label and item.selected_prescription_id:
                state.blacklist.add((label, item.selected_prescription_id))
            state.status = "rolled_back" if changed else "verified"
            state.reindex_required = bool(item.metadata.get("reindex_required", True))
        else:
            config_mapper.apply_config_patch(state.index_config, result.config_patch)
            verdict = Verdict(
                keep=True,
                before_score=before_score,
                after_score=after_score,
                reason="모든 top_k 후보 평가 후 가장 좋은 후보를 선택",
            )
            state.status = (
                "applied"
                if state.index_config != item.after_config
                else "verified"
            )
            state.reindex_required = bool(result.needs_reindex)
    else:
        return _fail_active_study(
            state,
            item,
            result.error or result.message or "best 후보를 적용할 수 없습니다.",
        )

    item.after_config = dict(state.index_config)
    if baseline_selected:
        baseline_report = item.metadata.get("before_report")
        item.after_metrics = dict(getattr(baseline_report, "ragas_scores", {}) or {})
    else:
        item.after_metrics = best_metrics
    item.status = "applied" if verdict.keep else "failed"
    item.rollback_reason = None if verdict.keep else verdict.reason
    item.metadata.update(
        {
            "pending": False,
            "active_study": False,
            "before_score": verdict.before_score,
            "after_score": verdict.after_score,
            "best_config": dict(result.best_config or {}),
        }
    )
    item.metadata.pop("before_report", None)
    state.optimization_report = reporter.build_trial_report(item, verdict)
    return state


def _fail_active_study(
    state: AgentDoctorState,
    item: OptimizationHistoryItem,
    reason: str,
    *,
    retryable: bool = False,
) -> AgentDoctorState:
    """study 오류 시 baseline으로 복원하고 재시도 가능 여부를 구분한다."""
    changed = state.index_config != item.before_config
    state.index_config = dict(item.before_config)
    item.status = "failed"
    item.rollback_reason = reason
    item.after_config = dict(state.index_config)
    item.metadata["pending"] = False
    item.metadata["active_study"] = False
    item.metadata["study_error"] = reason
    item.metadata["study_retryable"] = retryable
    item.metadata.pop("before_report", None)
    label = item.failure_labels[0] if item.failure_labels else ""
    prescription_id = item.selected_prescription_id
    previous_same_errors = sum(
        1
        for previous in state.optimization_history
        if previous is not item
        and previous.selected_prescription_id == prescription_id
        and (previous.failure_labels[0] if previous.failure_labels else "") == label
        and previous.metadata.get("study_error")
    )
    # 상태 계약 손상은 같은 처방으로 회복되지 않으므로 즉시 차단한다. 일시적인
    # adapter 오류는 한 번 재시도하되 반복되면 무한 루프 방지를 위해 차단한다.
    if (
        label
        and prescription_id
        and (not retryable or previous_same_errors >= 1)
    ):
        state.blacklist.add((label, prescription_id))
    state.status = "rolled_back" if changed else "error"
    state.reindex_required = bool(item.metadata.get("reindex_required", True))
    state.error = None if changed else reason
    return state


def _report_metrics(state: AgentDoctorState) -> dict:
    """현재 Eval report를 internal trial 관측값으로 변환한다."""
    if state.report is None:
        return {}
    metrics = dict(state.report.ragas_scores)
    if state.report.overall_score is not None:
        metrics["overall_score"] = state.report.overall_score
    metrics["pass_threshold"] = gate.passes_report(state.report)
    return metrics


def _report_score(report) -> float:
    """저장된 baseline report에서 overall_score를 안전하게 읽는다."""
    score = getattr(report, "overall_score", None)
    return float(score) if isinstance(score, (int, float)) else 0.0


def _best_trial_metrics(trials: list, result: OptimizationResult) -> dict:
    """adapter가 선택한 best trial의 지표를 이력에 남긴다."""
    best_config = result.best_config or {}
    for trial in trials:
        trial_config = getattr(trial, "config", None)
        trial_metrics = getattr(trial, "metrics", None)
        if trial_config == best_config and isinstance(trial_metrics, dict):
            return dict(trial_metrics)
        if isinstance(trial, dict) and trial.get("config") == best_config:
            return dict(trial.get("metrics") or {})
    return {}


def _study_floor_violations(metrics: dict) -> list[str]:
    """sweep 승자에도 일반 처방과 같은 지표 하한선을 적용한다."""
    numeric_metrics = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if "context_recall" not in numeric_metrics and "mean_recall_at_k" in numeric_metrics:
        numeric_metrics["context_recall"] = numeric_metrics["mean_recall_at_k"]
    return history.check_floor(numeric_metrics)


def _judge_pending_trial(
    state: AgentDoctorState,
) -> tuple[OptimizationHistoryItem | None, Verdict | None]:
    """직전에 적용한 처방(pending)을 판정한다. 나빴으면 config 롤백 + blacklist.
    판정한 이력 항목과 Verdict 를 반환한다(판정할 게 없으면 (None, None)).
    호출부가 이 둘로 롤백 여부(not verdict.keep) 판단 + 사용자 리포트를 만든다."""
    pending = history.find_pending(state.optimization_history)
    if pending is None:
        return None, None

    before_report = pending.metadata.get("before_report")
    after_report = state.report

    if before_report is None or after_report is None:
        # 비교할 점수가 없어 판정 불가 → 보수적으로 유지 처리하고 확정만.
        verdict = Verdict(
            keep=True, before_score=0.0, after_score=0.0,
            reason="판정 불가(리포트 없음) — 유지",
        )
    else:
        verdict = history.judge(before_report, after_report)

    # 롤백 전의 '실제 적용되어 측정된' config 를 이력에 남긴다.
    after_config = dict(state.index_config)

    if not verdict.keep:
        restored = dict(pending.before_config)  # config 되돌리기(롤백)
        # 임베딩 모델을 바꿨던 처방을 되돌릴 때는 컬렉션 차원도 원래대로
        # 재생성해야 한다. before_config의 플래그는 적용 전 스냅샷(False)이라
        # 그대로 복원하면 영구 Qdrant에서 dimension mismatch로 죽는다.
        if restored.get("embedding_model") != after_config.get("embedding_model"):
            restored["recreate_collection_on_dimension_mismatch"] = True
        state.index_config = restored
        state.reindex_required = bool(pending.metadata.get("reindex_required", True))
        label = pending.failure_labels[0] if pending.failure_labels else ""
        if label and pending.selected_prescription_id:
            state.blacklist.add((label, pending.selected_prescription_id))

    history.finalize_item(pending, verdict, after_config, after_report)
    return pending, verdict
