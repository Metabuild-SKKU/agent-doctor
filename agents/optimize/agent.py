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
           state.optimize_visit_count, state.max_optimize_visits,
           state.blacklist, state.completed_prescriptions, state.optimization_history,
           state.active_index_key, state.active_eval_key,
           state.runtime_capabilities(Planner request를 통해 소비)
[쓰는 것]  state.index_config, state.iteration, state.optimize_visit_count,
           state.status, state.error, state.current_agent, state.blacklist,
           state.completed_prescriptions, state.optimization_history,
           state.optimization_report, state.reindex_required

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

from collections import Counter
from dataclasses import replace
from typing import Any

from core.state import AgentDoctorState
from core.schema import DiagnosticReport
from agents.optimize import planner, optimizer, config_mapper, history, reporter, gate
from agents.optimize.schemas import (
    ConfigDiff,
    OptimizationHistoryItem,
    OptimizationRequest,
    OptimizationResult,
    OptimizeDecision,
    Verdict,
)


_MAX_UNJUDGEABLE_ATTEMPTS = 1
_OPTIMIZE_VISIT_LIMIT_REASON = "Optimize 절대 방문 상한 도달"


def _restore_history_item_baseline(
    state: AgentDoctorState,
    item: OptimizationHistoryItem,
    reason: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], DiagnosticReport | None, bool]:
    """처방 이력을 종료하고 적용 전 baseline 설정으로 안전하게 복원한다."""
    before_config = dict(state.index_config)
    before_report = item.metadata.get("before_report")
    state.index_config = dict(item.before_config)
    item.after_config = dict(state.index_config)
    item.status = "failed"
    item.rollback_reason = reason
    item.metadata.update(
        {
            "pending": False,
            "active_study": False,
            **(metadata or {}),
        }
    )
    item.metadata.pop("before_report", None)
    state.reindex_required = bool(item.metadata.get("reindex_required", True))
    return before_config, before_report, before_config != state.index_config


def _stop_at_optimize_visit_limit(
    state: AgentDoctorState,
) -> AgentDoctorState:
    """Optimize 절대 방문 상한에 도달하면 진행 중 설정을 복원하고 종료한다."""
    pending = history.find_active_study(state.optimization_history)
    if pending is None:
        pending = history.find_pending(state.optimization_history)

    changed = False
    if pending is not None:
        _before_config, before_report, changed = _restore_history_item_baseline(
            state,
            pending,
            _OPTIMIZE_VISIT_LIMIT_REASON,
            metadata={"visit_limit_reached": True},
        )
        before_score = _report_score(before_report)
        verdict = Verdict(
            keep=False,
            before_score=before_score,
            after_score=before_score,
            reason=_OPTIMIZE_VISIT_LIMIT_REASON,
        )
        pending.metadata.update(
            {
                "before_score": verdict.before_score,
                "after_score": verdict.after_score,
                "before_composite": history._read_composite(before_report),
                "after_composite": history._read_composite(before_report),
            }
        )
        state.optimization_report = reporter.build_trial_report(pending, verdict)
    else:
        decision = OptimizeDecision(
            mode="use_current",
            status="skipped",
            requires_user_confirmation=False,
            next_route="serve",
            reason=_OPTIMIZE_VISIT_LIMIT_REASON,
        )
        state.optimization_report = reporter.build_report(decision)

    state.status = "rolled_back" if changed else "verified"
    state.error = None
    print(
        "[Optimize] 절대 방문 상한 도달: "
        f"{state.optimize_visit_count}/{state.max_optimize_visits}"
    )
    print(
        "[Optimize] 진행 중 설정을 baseline으로 복원"
        if changed
        else "[Optimize] 추가 처방 없이 종료"
    )
    return state


def _unjudgeable_exclusions(
    optimization_history: list[OptimizationHistoryItem],
) -> set[tuple[str, str]]:
    """효과를 검증하지 못한 동일 처방의 재선택을 제한한다.

    품질 악화가 확인된 것은 아니므로 영구 blacklist에는 넣지 않는다. 대신 현재
    파이프라인 실행의 이력에서 같은 처방이 이미 측정 불가로 끝났다면 이후 Optimize
    방문에서 제외한다. 새 파이프라인 실행은 이력이 비어 있으므로 다시 시도할 수 있다.
    """
    attempts: Counter[tuple[str, str]] = Counter()
    for item in optimization_history:
        if not item.metadata.get("unjudgeable"):
            continue
        label = item.failure_labels[0] if item.failure_labels else ""
        prescription_id = item.selected_prescription_id or ""
        if label and prescription_id:
            attempts[(label, prescription_id)] += 1
    return {
        key
        for key, count in attempts.items()
        if count >= _MAX_UNJUDGEABLE_ATTEMPTS
    }


def _fmt_bool(value: bool) -> str:
    return "true" if value else "false"


def _fmt_score(value: float | None) -> str:
    return "None" if value is None else f"{value:.2f}"


def _fmt_composite(report: DiagnosticReport | None) -> str:
    if report is None or not isinstance(report.composite_score, dict):
        return "-"
    total = report.composite_score.get("total")
    if not isinstance(total, (int, float)):
        return "-"
    return f"{float(total):.1f}"


def _fmt_findings_summary(report: DiagnosticReport | None) -> str:
    if report is None or not report.findings:
        return "없음"

    labels = Counter(
        finding.label
        for finding in report.findings
        if finding.confirmed and finding.label
    )

    if not labels:
        return "없음"

    return ", ".join(f"{label} {count}건" for label, count in labels.items())


def _fmt_config_values(config: dict[str, Any], keys: list[str]) -> str:
    if not keys:
        return "변경 없음"
    return ", ".join(f"{key}={config.get(key)!r}" for key in keys)


def _fmt_mapping(values: dict[str, Any] | None) -> str:
    """후보 탐색 범위나 patch를 한 줄 로그로 표시한다."""
    if not values:
        return "{}"
    return "{" + ", ".join(f"{key}={value!r}" for key, value in values.items()) + "}"


def _diff_visible_keys(diff: ConfigDiff) -> list[str]:
    seen: set[str] = set()
    keys: list[str] = []
    for key in [*diff.changed_keys, *diff.added_keys, *diff.removed_keys]:
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _config_change_next_step(needs_reindex: bool) -> str:
    if needs_reindex:
        return "Index 재색인 후 Eval 재실행"
    return "Index 경유(물리 재색인 생략) 후 Eval 재실행"


def _log_eval_line(report: DiagnosticReport | None) -> None:
    pass_result = gate.passes_report(report) if report is not None else False
    print(
        f"[Optimize] Eval 결과: overall={_fmt_score(getattr(report, 'overall_score', None))}, "
        f"composite={_fmt_composite(report)}, pass={_fmt_bool(pass_result)}"
    )


def _log_optimize_input(state: AgentDoctorState) -> None:
    """이번 Optimize 방문의 진단 입력과 이미 소진된 처방을 출력한다."""
    print(f"[Optimize] 반복 횟수: {state.iteration}/{state.max_iterations} (진단 입력)")
    _log_eval_line(state.report)
    print(f"[Optimize] 발견된 문제: {_fmt_findings_summary(state.report)}")
    if state.blacklist:
        blocked = ", ".join(
            f"{label}/{prescription_id}"
            for label, prescription_id in sorted(state.blacklist)
        )
        print(f"[Optimize] 제외된 처방: [{blocked}]")


def _log_candidate_list(request: OptimizationRequest) -> None:
    """Planner가 만든 처방 후보와 후보별 근거·탐색 범위를 출력한다."""
    print(
        f"[Optimize] 후보 생성: {len(request.candidates)}개 "
        f"(label={request.failure_label}, optimizer={request.optimizer})"
    )
    distinct_reasons = list(
        dict.fromkeys(
            candidate.reason
            for candidate in request.candidates
            if candidate.reason
        )
    )
    shared_reason = distinct_reasons[0] if len(distinct_reasons) == 1 else None
    if shared_reason:
        print(f"[Optimize] 후보 제안 근거: {shared_reason}")
    for index, candidate in enumerate(request.candidates, 1):
        reindex = bool(candidate.patch and candidate.patch.reindex_required)
        patch = candidate.patch.changes if candidate.patch else {}
        print(
            f"[Optimize] 후보 {index}/{len(request.candidates)}: "
            f"id={candidate.id}, status={candidate.status}, "
            f"cost={candidate.cost!r}, "
            f"patch={_fmt_mapping(patch)}, "
            f"search_space={_fmt_mapping(candidate.search_space)}, "
            f"reindex={_fmt_bool(reindex)}"
        )
        if candidate.reason and candidate.reason != shared_reason:
            print(f"[Optimize]   제안 근거: {candidate.reason}")


def _log_candidate_review(result: OptimizationResult) -> None:
    """Optimizer가 후보를 거르거나 선택한 결과와 이유를 출력한다."""
    for skipped in result.metadata.get("skipped_candidates", []):
        if not isinstance(skipped, dict):
            continue
        print(
            f"[Optimize] 후보 SKIP: "
            f"id={skipped.get('prescription_id') or '-'}, "
            f"reason={skipped.get('reason') or 'unknown'}"
        )

    selected = result.selected_candidate
    if result.status == "failed":
        reason = result.metadata.get("error_code") or result.error or result.message
        if selected is None:
            print(f"[Optimize] 요청 FAIL: reason={reason or 'unknown'}")
        else:
            print(
                f"[Optimize] 후보 FAIL: id={selected.id}, "
                f"reason={reason or 'unknown'}"
            )
        return

    if result.status == "skipped":
        reason = result.metadata.get("error_code") or result.error or result.message
        if selected is None:
            print(f"[Optimize] 요청 SKIP: reason={reason or 'unknown'}")
            return
        prescription_id = selected.id
        print(
            f"[Optimize] 후보 SKIP: id={prescription_id}, "
            f"reason={reason or 'unknown'}"
        )
        return

    if selected is not None:
        print(f"[Optimize] 후보 SELECT: id={selected.id}")


def _log_optimize_transition(
    *,
    label: str | None,
    prescription_id: str | None,
    before_config: dict[str, Any],
    after_config: dict[str, Any],
    changed_keys: list[str],
    reindex_required: bool,
    next_step: str,
    include_reindex: bool = True,
    include_next_step: bool = True,
) -> None:
    print(f"[Optimize] 선택한 라벨: {label or '-'}")
    print(f"[Optimize] 선택한 처방: {prescription_id or '-'}")
    print(f"[Optimize] 변경 전 config: {_fmt_config_values(before_config, changed_keys)}")
    print(f"[Optimize] 변경 후 config: {_fmt_config_values(after_config, changed_keys)}")
    if include_reindex:
        print(f"[Optimize] reindex_required={_fmt_bool(reindex_required)}")
    if include_next_step:
        print(f"[Optimize] 다음 단계: {next_step}")


def _log_optimize_application(
    state: AgentDoctorState,
    request: OptimizationRequest,
    result: OptimizationResult,
    before_config: dict[str, Any],
    after_config: dict[str, Any],
    changed_keys: list[str],
    prescription_id: str | None,
) -> None:
    selected = result.selected_candidate
    print(f"[Optimize] 반복 횟수: {state.iteration}/{state.max_iterations} (처방 적용 후)")
    print(
        f"[Optimize] 처방 적용: id={prescription_id or '-'}, "
        f"label={request.failure_label}"
    )
    if selected is not None:
        reason = result.message or selected.reason or request.reason
        if reason:
            print(f"[Optimize] 선택 근거: {reason}")
    _log_optimize_transition(
        label=request.failure_label,
        prescription_id=prescription_id,
        before_config=before_config,
        after_config=after_config,
        changed_keys=changed_keys,
        reindex_required=bool(state.reindex_required),
        next_step=_config_change_next_step(bool(state.reindex_required)),
    )


def _log_optimize_verdict(
    state: AgentDoctorState,
    item: OptimizationHistoryItem,
    verdict: Verdict,
    *,
    next_step: str | None = None,
) -> None:
    if verdict.keep:
        before_config = item.before_config
        after_config = item.after_config
        reindex_required = False
    else:
        before_config = item.after_config
        after_config = dict(state.index_config)
        reindex_required = bool(state.reindex_required)
    diff = config_mapper.build_config_diff(before_config, after_config)
    action = "KEEP" if verdict.keep else "ROLLBACK"
    print(
        f"[Optimize] 이전 처방 판정: {action}, "
        f"prescription={item.selected_prescription_id or '-'}, "
        f"before={_fmt_score(verdict.before_score)}, "
        f"after={_fmt_score(verdict.after_score)}"
    )
    print(f"[Optimize] 판정 근거: {verdict.reason or '-'}")
    _log_optimize_transition(
        label=item.failure_labels[0] if item.failure_labels else None,
        prescription_id=item.selected_prescription_id,
        before_config=before_config,
        after_config=after_config,
        changed_keys=_diff_visible_keys(diff),
        reindex_required=reindex_required,
        next_step=next_step or _config_change_next_step(reindex_required),
        include_reindex=next_step is not None,
        include_next_step=next_step is not None,
    )
    print(
        f"[Optimize] 판정 결과: keep={_fmt_bool(verdict.keep)}, "
        f"before={_fmt_score(verdict.before_score)}, "
        f"after={_fmt_score(verdict.after_score)}"
    )


def _log_optimize_decision(
    state: AgentDoctorState,
    decision: OptimizeDecision,
) -> None:
    next_step = "Serve 이동" if decision.next_route == "serve" else decision.next_route
    action = "SKIP" if decision.status == "skipped" else decision.status.upper()
    print(f"[Optimize] 행동 결정: {action}, reason={decision.reason or '-'}")

    _log_optimize_transition(
        label=None,
        prescription_id=None,
        before_config={},
        after_config={},
        changed_keys=[],
        reindex_required=bool(state.reindex_required),
        next_step=f"{next_step} ({decision.status})",
        include_reindex=False,
    )


def _attach_runtime_deferred(
    report: object | None,
    deferred: list[dict[str, Any]],
) -> None:
    """현재 Optimize 방문에서만 보류한 runtime 처방을 사용자 리포트에 남긴다."""
    if report is None or not deferred:
        return
    metadata = getattr(report, "metadata", None)
    if isinstance(metadata, dict):
        metadata["runtime_deferred_prescriptions"] = list(deferred)


def run(state: AgentDoctorState) -> AgentDoctorState:
    """Optimize 노드 진입점. 성공·스킵·수동·오류 어느 경로든 같은 state 를 반환한다."""
    state.current_agent = "optimize"
    try:
        _log_optimize_input(state)
        if state.optimize_visit_count >= state.max_optimize_visits:
            return _stop_at_optimize_visit_limit(state)
        state.optimize_visit_count += 1
        if state.optimize_visit_count >= state.max_optimize_visits:
            return _stop_at_optimize_visit_limit(state)

        # top_k sweep는 후보 하나의 성공/실패를 곧바로 확정하지 않는다.
        # 직전 후보의 Eval 결과를 같은 study에 넣고 다음 후보 또는 best를 고른다.
        active_study = history.find_active_study(state.optimization_history)
        if active_study is not None:
            return _continue_internal_study(state, active_study)

        # (1) 지난 처방 판정 + (나빴으면) 롤백/블랙리스트
        judged_item, verdict, rollback_baseline_report = _judge_pending_trial(state)
        rolled_back = judged_item is not None and not verdict.keep
        visit_exclusions = set(state.blacklist)
        visit_exclusions.update(state.completed_prescriptions)
        visit_exclusions.update(
            _unjudgeable_exclusions(state.optimization_history)
        )
        deferred_runtime: list[dict[str, Any]] = []
        if (
            judged_item is not None
            and judged_item.metadata.get("reranker_execution_incomplete")
        ):
            label = judged_item.failure_labels[0] if judged_item.failure_labels else ""
            prescription_id = judged_item.selected_prescription_id
            if label and prescription_id:
                visit_exclusions.add((label, prescription_id))
                deferred_runtime.append(
                    {
                        "failure_label": label,
                        "prescription_id": prescription_id,
                        "reason": "reranker_execution_incomplete",
                        "retryable": True,
                    }
                )

        # route_after_eval 이 '품질 통과 + 판정 대기' 상태에서 이리로 보낸 경우:
        # 방금 판정한 처방을 유지(keep)했다면 이미 목표 품질에 도달했으므로 새 처방을
        # 더 붙이지 않고 그대로 Serve 로 확정한다(치료경과·유지 카운트 마감용 1회 방문).
        # 롤백됐다면 복원된 config 로 재색인·재평가가 필요하니 이 단축을 타지 않는다.
        if (
            judged_item is not None
            and not rolled_back
            and gate.passes_report(state.report)
        ):
            state.status = "verified"
            _log_optimize_verdict(
                state,
                judged_item,
                verdict,
                next_step="Serve 이동 (verified, 품질 통과)",
            )
            state.optimization_report = reporter.build_trial_report(judged_item, verdict)
            _attach_runtime_deferred(state.optimization_report, deferred_runtime)
            return state

        # (2) 새 처방 선택. 저비용 사전검증에서 baseline이 이기면 현재 처방을
        # 소진 처리하고, 재색인·iteration 증가 없이 같은 방문에서 다음 처방을 고른다.
        while True:
            request, decision = planner.plan(state, blacklist=visit_exclusions)
            if decision.mode != "apply_optimize" or request is None:
                state.status = "rolled_back" if rolled_back else decision.status
                if judged_item is not None:
                    _log_optimize_verdict(state, judged_item, verdict)
                else:
                    _log_optimize_decision(state, decision)
                # 롤백이 있었으면 그게 headline, 아니면 흐름 결정(수동/유지)을 보고.
                state.optimization_report = (
                    reporter.build_trial_report(judged_item, verdict)
                    if rolled_back
                    else reporter.build_report(decision, request)
                )
                _attach_runtime_deferred(
                    state.optimization_report,
                    deferred_runtime,
                )
                return state
            _log_candidate_list(request)

            previous_label = history.last_failure_label(state.optimization_history)
            starts_new_label = (
                previous_label is None or previous_label != request.failure_label
            )
            if starts_new_label and state.iteration >= state.max_iterations:
                state.status = "rolled_back" if rolled_back else "verified"
                if judged_item is not None:
                    verdict_next_step = (
                        _config_change_next_step(bool(state.reindex_required))
                        if rolled_back
                        else "Serve 이동 (verified, 반복 예산 소진)"
                    )
                    _log_optimize_verdict(
                        state,
                        judged_item,
                        verdict,
                        next_step=verdict_next_step,
                    )
                else:
                    _log_optimize_decision(
                        state,
                        OptimizeDecision(
                            mode="use_current",
                            status=state.status,
                            requires_user_confirmation=False,
                            next_route="serve",
                            reason="반복 예산 소진",
                        ),
                    )
                if judged_item is not None:
                    state.optimization_report = reporter.build_trial_report(
                        judged_item, verdict
                    )
                return state

            result = optimizer.run(request)
            _log_candidate_review(result)
            # skipped 처방(baseline 무개선·적용 불가 경로·빈 search space)이면 포기하지 않고
            # 그 처방을 블랙리스트에 넣어 다음 우선순위 처방으로 넘어간다. 한 라벨의 처방이
            # 막혀도(예: enable_hybrid 는 pipeline capability 미지원) 다른 actionable
            # finding 이 처방받을 기회를 준다. (issue #26)
            if result.status != "skipped":
                break

            prescription_id = (
                result.selected_candidate.id
                if result.selected_candidate
                else (request.candidates[0].id if request.candidates else None)
            )
            rejection = (request.failure_label, prescription_id)
            if not prescription_id or rejection in visit_exclusions:
                state.status = "rolled_back" if rolled_back else "skipped"
                if judged_item is not None:
                    _log_optimize_verdict(state, judged_item, verdict)
                if rolled_back:
                    state.optimization_report = reporter.build_trial_report(
                        judged_item, verdict
                    )
                elif judged_item is None:
                    _log_optimize_decision(
                        state,
                        OptimizeDecision(
                            mode="use_current",
                            status="skipped",
                            requires_user_confirmation=False,
                            next_route="serve",
                            reason="적용 가능한 처방 없음",
                        ),
                    )
                return state
            error_code = str(result.metadata.get("error_code") or "")
            visit_exclusions.add(rejection)
            if error_code in {
                "runtime_capability_unavailable",
                "reranker_disabled",
            }:
                capability = (
                    request.metadata.get("runtime_capabilities", {})
                    .get("reranker", {})
                )
                deferred_runtime.append(
                    {
                        "failure_label": request.failure_label,
                        "prescription_id": prescription_id,
                        "reason": (
                            error_code
                            if error_code == "reranker_disabled"
                            else capability.get(
                                "reason",
                                "runtime_capability_unavailable",
                            )
                        ),
                        "retryable": (
                            False
                            if error_code == "reranker_disabled"
                            else bool(capability.get("retryable", True))
                        ),
                    }
                )
            else:
                state.blacklist.add(rejection)

        if result.status != "proposed" or result.config_patch is None:
            # optimizer 가 적용 가능한 patch 를 못 만듦(skipped/failed)
            state.status = "rolled_back" if rolled_back else result.status
            if result.status == "failed":
                state.error = result.error or result.message
            # 롤백만 리포트로 남긴다. 적용 실패 자체는 status/error 로 전달(MVP 한계).
            if rolled_back:
                _log_optimize_verdict(state, judged_item, verdict)
                state.optimization_report = reporter.build_trial_report(judged_item, verdict)
            elif judged_item is not None:
                _log_optimize_verdict(state, judged_item, verdict)
            else:
                _log_optimize_decision(
                    state,
                    OptimizeDecision(
                        mode="use_current",
                        status=result.status,
                        requires_user_confirmation=False,
                        next_route="serve",
                        reason=result.error or result.message,
                    ),
                )
            return state

        # 적용 직전 스냅샷(롤백이 있었다면 이미 되돌려진 config 가 before 가 된다)
        before_config = dict(state.index_config)
        # 롤백 직후라면 비교 기준을 복원된 baseline 리포트로 잡는다. state.report 는
        # 롤백 전의 열화된 Eval 이라, 그걸 baseline 으로 쓰면 이 처방이 원래보다
        # 나빠도 '개선'으로 오판해 유지된다(#2). 롤백이 없었으면 현재 report 가 기준.
        before_report = rollback_baseline_report if rolled_back else state.report
        # 롤백이 요구한 재색인(_judge_pending_trial 이 state.reindex_required 에 세팅)을
        # 기억해둔다. index-time 처방을 롤백하면 config 는 되돌아가지만 실제 Qdrant
        # 인덱스는 아직 열화 상태라 이 재색인이 반드시 일어나야 한다.
        rollback_reindex_required = state.reindex_required if rolled_back else False

        if judged_item is not None:
            _log_optimize_verdict(state, judged_item, verdict)

        # 검증된 처방을 실제 index_config 에 반영(canonical→flat 변환은 mapper 담당)
        config_diff = config_mapper.apply_config_patch(state.index_config, result.config_patch)
        # 새 처방의 재색인 필요 여부와 '롤백이 요구한 재색인'을 OR 로 합친다. 검색시점
        # 처방(needs_reindex=False)이 롤백의 재색인 요구를 덮어써, baseline 청킹이 인덱스에
        # 복원되지 않고 열화된 인덱스가 그대로 재사용되던 버그(config/인덱스 불일치)를 막는다.
        state.reindex_required = bool(result.needs_reindex) or rollback_reindex_required

        # pending 이력 생성(다음 방문에서 finalize) + iteration 1회 증가
        prescription_id = (
            result.selected_candidate.id if result.selected_candidate else None
        )
        item = history.create_pending_item(
            state, request, prescription_id, before_config, before_report
        )
        item.metadata["reindex_required"] = bool(result.needs_reindex)
        if prescription_id in {
            "enable_reranker",
            "widen_rerank_candidates",
        }:
            item.metadata["runtime_capability"] = dict(
                request.metadata.get("runtime_capabilities", {}).get(
                    "reranker",
                    {},
                )
            )
        if rolled_back and judged_item is not None:
            item.metadata["before_index_key"] = judged_item.metadata.get(
                "before_index_key", ""
            )
            item.metadata["before_eval_key"] = judged_item.metadata.get(
                "before_eval_key", ""
            )
        else:
            item.metadata["before_index_key"] = state.active_index_key
            item.metadata["before_eval_key"] = state.active_eval_key
        state.optimization_history.append(item)
        if starts_new_label:
            state.iteration += 1

        _log_optimize_application(
            state,
            request,
            result,
            before_config,
            dict(state.index_config),
            _diff_visible_keys(config_diff),
            prescription_id,
        )

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
        state.optimization_report = reporter.build_report(
            decision,
            request,
            diff=config_diff,
        )
        state.optimization_report.metadata.update(
            {
                "failure_label": request.failure_label,
                "selected_prescription": prescription_id,
                "reindex_required": bool(result.needs_reindex),
            }
        )
        if prescription_id in {
            "enable_reranker",
            "widen_rerank_candidates",
        }:
            state.optimization_report.metadata["runtime_capability"] = dict(
                request.metadata.get("runtime_capabilities", {}).get(
                    "reranker",
                    {},
                )
            )
        _attach_runtime_deferred(
            state.optimization_report,
            deferred_runtime,
        )
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
    print(
        f"[Optimize] 후보 평가 완료: config={_fmt_mapping(current_candidate)}, "
        f"metrics={_fmt_mapping(observed_trials[-1]['metrics'])}, "
        f"status={observed_trials[-1]['status']}"
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
    _log_candidate_review(result)
    item.metadata["trial_results"] = list(
        result.metadata.get("trial_results", observed_trials)
    )
    item.metadata["adapter_status"] = result.metadata.get("adapter_status")

    if (
        result.status == "proposed"
        and result.config_patch is not None
        and result.metadata.get("adapter_status") == "needs_evaluation"
    ):
        before_config = dict(state.index_config)
        config_diff = config_mapper.apply_config_patch(state.index_config, result.config_patch)
        state.reindex_required = bool(result.needs_reindex)
        item.metadata["current_candidate"] = dict(result.config_patch.changes)
        item.after_config = dict(state.index_config)
        state.status = "applied"
        _log_optimize_application(
            state,
            resumed_request,
            result,
            before_config,
            dict(state.index_config),
            _diff_visible_keys(config_diff),
            item.selected_prescription_id,
        )
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
    before_config_for_log = dict(state.index_config)
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
            # 비교 기준은 before_config(study baseline)가 아니라 after_config 가 맞다.
            # after_config 는 '마지막으로 적용·재색인된 후보'(=현재 물리 인덱스가 반영
            # 중인 config)의 스냅샷이므로, best 가 마지막 후보와 같으면 이미 색인돼 있어
            # 재색인 불필요("verified"), 다르면 "applied" 로 Index 재색인을 태운다.
            # (첫 후보 단계의 after_config={} 는 비교가 항상 True 라 "applied" 로 안전.)
            state.status = (
                "applied"
                if state.index_config != item.after_config
                else "verified"
            )
            state.reindex_required = bool(result.needs_reindex)
            label = item.failure_labels[0] if item.failure_labels else ""
            if label and item.selected_prescription_id:
                state.completed_prescriptions.add(
                    (label, item.selected_prescription_id)
                )
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
    # 표시·게이트용 종합점수(0~100). before 는 baseline 리포트에서, after 는 baseline
    # 복원 시 before 와 동일(설정을 되돌렸으므로), 아니면 sweep 이 full report 를 남기지
    # 않아 미상(None) — 표시부가 fallback 한다.
    before_composite = history._read_composite(item.metadata.get("before_report"))
    # baseline 복원이면 설정을 되돌렸으니 after=before. 새 후보가 이겼으면 그 후보의
    # 관측값에 실린 composite_total 을 쓴다(_report_metrics 가 실어둠). 없으면 None →
    # 표시부가 overall×100 으로 폴백.
    after_composite = (
        before_composite if baseline_selected
        else best_metrics.get("composite_total")
    )
    item.metadata.update(
        {
            "pending": False,
            "active_study": False,
            "before_score": verdict.before_score,
            "after_score": verdict.after_score,
            "before_composite": before_composite,
            "after_composite": after_composite,
            "best_config": dict(result.best_config or {}),
        }
    )
    item.metadata.pop("before_report", None)
    state.optimization_report = reporter.build_trial_report(item, verdict)
    diff = config_mapper.build_config_diff(before_config_for_log, state.index_config)
    next_step = (
        _config_change_next_step(bool(state.reindex_required))
        if state.status in ("applied", "rolled_back")
        else f"Serve 이동 ({state.status})"
    )
    _log_optimize_transition(
        label=item.failure_labels[0] if item.failure_labels else None,
        prescription_id=item.selected_prescription_id,
        before_config=before_config_for_log,
        after_config=dict(state.index_config),
        changed_keys=_diff_visible_keys(diff),
        reindex_required=bool(state.reindex_required),
        next_step=next_step,
    )
    return state


def _fail_active_study(
    state: AgentDoctorState,
    item: OptimizationHistoryItem,
    reason: str,
    *,
    retryable: bool = False,
) -> AgentDoctorState:
    """study 오류 시 baseline으로 복원하고 재시도 가능 여부를 구분한다."""
    before_config_for_log, _before_report, changed = _restore_history_item_baseline(
        state,
        item,
        reason,
        metadata={
            "study_error": reason,
            "study_retryable": retryable,
        },
    )
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
    state.error = None if changed else reason
    diff = config_mapper.build_config_diff(before_config_for_log, state.index_config)
    next_step = (
        _config_change_next_step(bool(state.reindex_required))
        if state.status == "rolled_back"
        else f"Serve 이동 ({state.status})"
    )
    _log_optimize_transition(
        label=label,
        prescription_id=prescription_id,
        before_config=before_config_for_log,
        after_config=dict(state.index_config),
        changed_keys=_diff_visible_keys(diff),
        reindex_required=bool(state.reindex_required),
        next_step=next_step,
    )
    return state


def _report_metrics(state: AgentDoctorState) -> dict:
    """현재 Eval report를 internal trial 관측값으로 변환한다."""
    if state.report is None:
        return {}
    metrics = dict(state.report.ragas_scores)
    if state.report.overall_score is not None:
        metrics["overall_score"] = state.report.overall_score
    # 표시·게이트용 종합점수(0~100)도 관측값에 실어, sweep 승자의 after_composite 를
    # 표시부가 복원하게 한다(없으면 overall×100 폴백 → before/after 스케일 뒤섞임).
    composite_total = (state.report.composite_score or {}).get("total")
    if composite_total is not None:
        metrics["composite_total"] = float(composite_total)
        # 탐색 objective(sweep 승자 선택)용 정규화 composite(0~1). overall_score 와 같은
        # 스케일이라 objective 미측정 시 overall 폴백이 스케일-안전하다(internal_adapter).
        metrics["composite_score"] = float(composite_total) / 100.0
    metrics["pass_threshold"] = gate.passes_report(state.report)
    return metrics


def _report_score(report) -> float:
    """baseline report의 탐색 점수 — 정규화 composite(0~1). sweep 승자 best_score(=composite)
    와 같은 지표라야 before/after 가 같은 축에서 비교된다. composite 미측정이면 overall 폴백."""
    total = (getattr(report, "composite_score", None) or {}).get("total")
    if total is not None:
        return float(total) / 100.0
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
) -> tuple[OptimizationHistoryItem | None, Verdict | None, DiagnosticReport | None]:
    """직전에 적용한 처방(pending)을 판정한다. 나빴으면 config 롤백 + blacklist.
    판정한 이력 항목과 Verdict 를 반환한다(판정할 게 없으면 (None, None, None)).
    호출부가 이 둘로 롤백 여부(not verdict.keep) 판단 + 사용자 리포트를 만든다.

    3번째 반환값(rollback_baseline_report): 롤백했을 때 '복원된 config 가 실제로
    받았던 점수'를 담은 리포트(=이 처방의 before_report). 롤백 후 같은 방문에서
    이어서 제안되는 다음 처방의 비교 기준(before_report)으로 써야 한다. state.report
    는 롤백 직전의 열화된 Eval 이라 baseline 으로 쓰면 원래보다 나빠도 '개선'으로
    오판해 유지해버린다(#2). 유지/판정없음이면 None."""
    pending = history.find_pending(state.optimization_history)
    if pending is None:
        return None, None, None

    before_report = pending.metadata.get("before_report")
    after_report = state.report

    if before_report is None or after_report is None:
        # 비교할 점수가 없어 판정 불가 → 안전망 취지대로 롤백(원래 설정 복원)한다.
        # before_config 스냅샷은 리포트와 무관하게 항상 있으므로 복원은 안전하며,
        # '유지'로 두면 성능을 떨어뜨린 처방도 검증 없이 굳어질 수 있다(방향이 위험).
        verdict = Verdict(
            keep=False, before_score=0.0, after_score=0.0,
            reason="판정 불가(리포트 없음) — 롤백", unjudgeable=True,
        )
    elif _reranker_execution_incomplete(pending, after_report):
        runtime = dict(
            (after_report.runtime_summary or {}).get("reranker") or {}
        )
        pending.metadata["reranker_execution_incomplete"] = True
        pending.metadata["reranker_runtime"] = runtime
        verdict = Verdict(
            keep=False,
            before_score=_report_score(before_report),
            after_score=_report_score(after_report),
            reason=(
                "reranker가 모든 평가 검색에서 실행되지 않아 효과를 판정할 수 없음 "
                f"(시도 {runtime.get('attempted', 0)}, "
                f"성공 {runtime.get('applied', 0)}) — baseline 복원"
            ),
            unjudgeable=True,
        )
    else:
        verdict = history.judge(before_report, after_report)
        verdict = _relax_reranker_precision_floor(
            pending,
            before_report,
            after_report,
            verdict,
        )

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
        # 측정이 없어(unjudgeable) 롤백한 경우는 '나빴다는 증거'가 아니므로 블랙리스트
        # 등록을 건너뛴다. 영구 차단하면 리포트 부재만으로 처방이 소진되고, before_report
        # None 이 다음 방문으로 전파돼 판정 불가→롤백→차단이 연쇄될 수 있다(리뷰 #36).
        label = pending.failure_labels[0] if pending.failure_labels else ""
        if label and pending.selected_prescription_id and not verdict.unjudgeable:
            state.blacklist.add((label, pending.selected_prescription_id))

    history.finalize_item(pending, verdict, after_config, after_report)
    rollback_baseline_report = before_report if not verdict.keep else None
    return pending, verdict, rollback_baseline_report


def _relax_reranker_precision_floor(
    pending: OptimizationHistoryItem,
    before_report: DiagnosticReport,
    after_report: DiagnosticReport,
    verdict: Verdict,
) -> Verdict:
    """Reranker 처방은 검색 순위 개선 신호가 있으면 precision 단독 위반을 완화한다.

    Reranker는 관련 청크를 더 위로 올리는 과정에서 context_precision이 일시적으로
    흔들릴 수 있다. 그런데 종합점수와 low-rank 라벨이 함께 개선됐는데도
    context_precision 하나만으로 롤백하면 실제 검색 개선 처방을 학습하지 못한다.
    """
    if verdict.keep:
        return verdict
    if pending.selected_prescription_id not in {
        "enable_reranker",
        "widen_rerank_candidates",
    }:
        return verdict
    if verdict.floor_violations != ["context_precision"]:
        return verdict
    if verdict.after_score <= verdict.before_score:
        return verdict

    before_low_rank = _label_count(before_report, "retrieval_low_rank")
    after_low_rank = _label_count(after_report, "retrieval_low_rank")
    if before_low_rank <= 0 or after_low_rank >= before_low_rank:
        return verdict

    return Verdict(
        keep=True,
        before_score=verdict.before_score,
        after_score=verdict.after_score,
        before_composite=verdict.before_composite,
        after_composite=verdict.after_composite,
        floor_violations=[],
        reason=(
            "reranker 적용 후 context_precision 단독 하한선 위반이 있었지만 "
            f"종합점수 상승 {verdict.before_score:.3f}→{verdict.after_score:.3f}, "
            f"retrieval_low_rank 감소 {before_low_rank}→{after_low_rank}로 유지"
        ),
        unjudgeable=verdict.unjudgeable,
    )


def _label_count(report: DiagnosticReport, label: str) -> int:
    """리포트에서 특정 라벨의 확정 finding 개수를 센다."""
    return sum(1 for finding in report.findings if finding.label == label)


def _reranker_execution_incomplete(
    pending: OptimizationHistoryItem,
    after_report: DiagnosticReport,
) -> bool:
    """reranker 처방 후 Eval이 일부라도 실제 CrossEncoder 점수를 못 만들었는지 본다."""
    if pending.selected_prescription_id not in {
        "enable_reranker",
        "widen_rerank_candidates",
    }:
        return False
    runtime = (after_report.runtime_summary or {}).get("reranker")
    if not isinstance(runtime, dict) or not runtime.get("enabled"):
        return True
    try:
        attempted = int(runtime.get("attempted", 0))
        applied = int(runtime.get("applied", 0))
    except (TypeError, ValueError):
        return True
    return attempted == 0 or applied < attempted
