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
from agents.optimize import (
    action_aggregator,
    planner,
    optimizer,
    config_mapper,
    history,
    reporter,
    gate,
    rules,
)
from agents.optimize.schemas import (
    ConfigDiff,
    OptimizationHistoryItem,
    OptimizationRequest,
    OptimizationResult,
    OptimizeDecision,
    Verdict,
)


# 리포트 부재로 판정 자체가 없었던 경우. 다음 방문에도 리포트가 없을 가능성이 커
# 재시도 여지가 없다고 보고 1회로 제한한다.
_MAX_UNJUDGEABLE_ATTEMPTS = 1
# sweep이 baseline을 측정하지 못해 study가 끝난 경우(missing_scorable_baseline).
# baseline 관측값은 방문마다 갱신되므로 다음 Eval에서 성립할 수 있다. min_delta 도입
# 전 이 실패는 retryable 경로로 2회까지 허용됐고, 그 예산을 그대로 유지한다.
_MAX_MEASUREMENT_FAILURE_ATTEMPTS = 2
# 품질 저하가 확인되어 롤백된 action 의 재선택 허용 횟수. 1회로 막으면 "다른 baseline
# 에서는 다시 시도한다"(구현계획 §5.1)가 통째로 죽고, 제한이 없으면 baseline 이 조금만
# 움직여도 같은 처방을 계속 재시도한다. 옮겨간 baseline 에서 한 번만 더 확인한다.
_MAX_CONFIRMED_ROLLBACK_ATTEMPTS = 2
_OPTIMIZE_VISIT_LIMIT_REASON = "Optimize 절대 방문 상한 도달"

# ── reranker guardrail 의 대상 action ──────────────────────────────
# reranker 는 Index 가 실제 모델을 로드·추론해 봐야 쓸 수 있는지 알 수 있는 유일한
# 축이라, 다른 축에는 없는 특수 처리가 붙는다(실행 검증·deferred·precision floor 완화).
# 세 지점이 같은 집합을 보므로 상수 하나로 묶는다 — 어긋나면 guardrail 이 반쪽만 걸린다.
_RERANKER_ACTION_KEYS = frozenset({
    "reranker.enabled:enable",
    "reranker.candidate_count:increase",
})
# 구버전 이력에는 action_key 가 없다. 저장된 state 를 이어받아도 guardrail 이
# 그대로 걸리도록 처방 id 도 함께 인식한다(단계 8 에서 제거).
_LEGACY_RERANKER_PRESCRIPTIONS = frozenset({
    "enable_reranker",
    "widen_rerank_candidates",
})

# planner 의 eligibility 가 "품질 실패가 아닌 이유"로 거른 action 은 차단이 아니라
# 보류로 보고해야 한다. 실행 판정이 optimizer 에서 planner 앞단으로 옮겨오면서
# 이 사유들도 함께 앞당겨졌다(구현계획 §8.3).
#   runtime_capability_unavailable : Index 가 모델을 못 올렸다 → 다음 실행에 풀릴 수 있다
#   prerequisite_unmet             : 선행 조건 미충족(예: 리랭커가 꺼진 채 후보창 확대)
#
# ⚠️ 나머지 사유(catalog_blocked·unsupported_state_mapping·unsupported_backend_path·
# unsupported_capability·no_candidate_value·multi_axis_search_space)는 **의도적으로
# 보류에서 뺀다.** 그것들은 "지금은 못 한다"가 아니라 "이 파이프라인에서는 불가능하다"
# 이므로 매 방문마다 반복 보고하면 소음이 된다. 사용자에게 필요한 형태는 방문 단위
# 보류가 아니라 catalog 수준의 "차단된 기능 목록"이며, 그건 별도 작업이다.
# 제외된 사유 전체는 request/decision metadata 의 rejected_actions 에 남는다.
_DEFERRABLE_ELIGIBILITY_REASONS = {
    "runtime_capability_unavailable",
    "prerequisite_unmet",
}

# 선행 조건별 사유 문구. optimizer 가 실행 직전에 쓰는 어휘와 같은 값을 쓴다 —
# 두 계층의 보고가 갈리면 사용자가 같은 상황을 다른 이름으로 두 번 본다.
_PREREQUISITE_REASONS = {
    "reranker.enabled": "reranker_disabled",
}


def _is_reranker_item(item: OptimizationHistoryItem) -> bool:
    """이 이력 항목이 reranker guardrail 대상인가."""
    if item.action_key:
        return item.action_key in _RERANKER_ACTION_KEYS
    return item.selected_prescription_id in _LEGACY_RERANKER_PRESCRIPTIONS


def _deferred_from_rejected_actions(
    rejected: list[dict[str, Any]] | None,
    runtime_capabilities: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """planner 가 실행 불가로 거른 action 중 '보류'로 보고할 것을 뽑는다.

    품질 실패와 구분하는 것이 요점이다. runtime 이 준비되지 않았거나 선행 조건이
    아직 아닌 것은 처방이 나빴다는 증거가 아니므로 차단 집합에 넣지 않고, 사용자
    리포트에만 "지금은 못 한다"로 남긴다.
    """
    deferred: list[dict[str, Any]] = []
    capability = (runtime_capabilities or {}).get("reranker") or {}
    for entry in rejected or []:
        reason = entry.get("reason")
        if reason not in _DEFERRABLE_ELIGIBILITY_REASONS:
            continue
        if reason == "prerequisite_unmet":
            # 어느 선행 조건이 막았는지는 eligibility 가 detail 에 채워 준다.
            # 하드코딩하면 선행 조건 축이 늘어나는 순간 조용히 틀린 사유를 보고한다.
            unmet = (entry.get("eligibility") or {}).get("unmet_prerequisite", "")
            detail = _PREREQUISITE_REASONS.get(unmet) or f"prerequisite_unmet:{unmet}"
            retryable = False
        else:
            detail = capability.get("reason", "runtime_capability_unavailable")
            retryable = bool(capability.get("retryable", True))
        for prescription_id in entry.get("prescription_ids") or [None]:
            deferred.append(
                {
                    "action_key": entry.get("action_key"),
                    "failure_label": next(
                        iter(entry.get("supporting_labels") or []), ""
                    ),
                    "prescription_id": prescription_id,
                    "reason": detail,
                    "retryable": retryable,
                }
            )
    return deferred


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
) -> set[str]:
    """효과를 검증하지 못한 동일 action 의 재선택을 제한한다. 반환값은 action key.

    품질 악화가 확인된 것은 아니므로 blocked_action_attempts 에는 넣지 않는다.
    두 집합과 optimization_history 는 같은 AgentDoctorState 필드라 수명이 같으므로
    "새 실행에서 재시도"는 어느 쪽이든 마찬가지다 — 이 구분의 실익은 수명이 아니라
    **분류**다. 측정 불가를 품질 실패로 기록하면 리포트가 처방을 잘못 비난한다.

    차단 단위가 attempt(정확한 전이)가 아니라 action key 인 것도 그래서다. 측정이
    성립하지 않은 것은 그 전이의 문제가 아니라 이 방문의 문제라, 같은 축의 다른
    후보값으로 바꿔 봐야 똑같이 못 잰다.

    허용 횟수는 측정이 실패한 이유에 따라 다르다. 리포트 부재는 다음 방문에도 같을
    가능성이 크지만, sweep의 baseline 미측정은 관측값이 갱신되면 풀린다.
    """
    report_absent: Counter[str] = Counter()
    measurement_failure: Counter[str] = Counter()
    for item in optimization_history:
        if not item.metadata.get("unjudgeable"):
            continue
        action_key = item.action_key or ""
        if not action_key:
            continue
        # study_error 는 _fail_active_study 가 남긴다(= sweep 측정 실패).
        if item.metadata.get("study_error"):
            measurement_failure[action_key] += 1
        else:
            report_absent[action_key] += 1
    excluded = {
        key
        for key, count in report_absent.items()
        if count >= _MAX_UNJUDGEABLE_ATTEMPTS
    }
    excluded.update(
        key
        for key, count in measurement_failure.items()
        if count >= _MAX_MEASUREMENT_FAILURE_ATTEMPTS
    )
    return excluded


def _rollback_action_cooldown_exclusions(
    optimization_history: list[OptimizationHistoryItem],
) -> set[str]:
    """품질 저하가 확인된 롤백이 쌓인 action 을 같은 실행 안에서 제외한다.

    ``blocked_action_attempts`` 는 의도적으로 baseline 이 바뀌면 풀리는 정확한 전이
    차단이다. 그런데 baseline 지문이 config 전체를 보므로 무관한 축이 조금만 움직여도
    풀려, 방금 점수를 떨어뜨린 처방이 곧바로 다시 선택된다. 그 의미는 그대로 두고
    history 기반 횟수 제한을 따로 얹는다.

    세는 대상은 **품질이 실제로 나빠진 롤백**뿐이다. 측정 불가(unjudgeable)·study
    오류·방문 상한은 처방의 잘못이 아니고, 마진 미달(margin_rejected)은 점수가 오르긴
    한 시도라 그 축이 나쁘다는 증거가 아니다.
    """
    confirmed: Counter[str] = Counter()
    for item in optimization_history:
        action_key = item.action_key or ""
        if not action_key or item.status != "failed":
            continue
        if item.rollback_reason is None:
            continue
        metadata = item.metadata
        if metadata.get("unjudgeable"):
            continue
        if metadata.get("study_error") or metadata.get("visit_limit_reached"):
            continue
        if metadata.get("margin_rejected"):
            continue
        confirmed[action_key] += 1
    return {
        key
        for key, count in confirmed.items()
        if count >= _MAX_CONFIRMED_ROLLBACK_ATTEMPTS
    }


def _active_excluded_action_keys(state: AgentDoctorState) -> set[str]:
    """지금 baseline 에서 실제로 닫혀 있는 action key 를 로그용으로 모은다.

    판정·롤백이 끝난 뒤에 불러야 한다. 롤백은 index_config 를 되돌리므로 그 전에
    부르면 planner 가 서지도 않을 baseline 으로 지문을 비교하게 된다.
    """
    active = set(_unjudgeable_exclusions(state.optimization_history))
    active.update(_rollback_action_cooldown_exclusions(state.optimization_history))

    # 지문은 action key 마다 한 번만 계산한다(sweep 이 후보마다 attempt key 를 쌓는다).
    fingerprints: dict[str, str] = {}
    for key in (*state.blocked_action_attempts, *state.completed_action_studies):
        current = fingerprints.get(key.action_key)
        if current is None:
            current = history.baseline_fingerprint(
                state.index_config,
                key.action_key,
                state.runtime_capabilities,
            )
            fingerprints[key.action_key] = current
        if key.baseline_fingerprint == current:
            active.add(key.action_key)

    return active


def _block_action_study(
    state: AgentDoctorState,
    item: OptimizationHistoryItem,
) -> None:
    """이 baseline 에서 결론이 난 탐색을 기록한다(같은 study 재시작 방지).

    sweep 정상 완료뿐 아니라 "이 baseline 에서는 적용할 수 없다"는 판정도 여기에
    들어간다. baseline fingerprint 를 담고 있어 config 가 바뀌면 자동으로 풀린다.
    """
    if item.action_study_key is not None:
        state.completed_action_studies.add(item.action_study_key)


def _block_evaluated_sweep_candidates(
    state: AgentDoctorState,
    item: OptimizationHistoryItem,
) -> None:
    """sweep 이 실제로 측정한 후보들을 **결과 baseline 기준으로도** 차단한다.

    study key 만으로는 부족하다. sweep 승자가 적용되면 baseline 이 그 action
    자신 때문에 움직여, 다음 방문에서 study fingerprint 가 어긋나 같은 축을 곧바로
    다시 훑는다(이미 잰 값을 다시 재는 순수 낭비다). 기존
    ``completed_prescriptions`` 는 baseline 무관 영구 차단이라 이 문제가 없었다.

    후보값을 결과 baseline 에도 기록하면 "후보 소진으로 축이 닫힌다"(구현계획 §5.1)가
    실제로 성립한다. 다른 축이 baseline 을 바꾸면 fingerprint 가 달라져 다시 열린다 —
    "상류를 고친 뒤 하류를 재평가한다"는 설계 의도는 그대로다.

    이 기록이 정당한 이유: sweep 후보값은 절대값(top_k=7)이지 baseline 대비 변화량이
    아니다. 같은 값을 다른 baseline 에서 다시 적용해도 도달하는 config 는 같다.
    """
    if not item.action_key:
        return
    for trial in item.metadata.get("trial_results") or []:
        config = (
            trial.get("config") if isinstance(trial, dict) else getattr(trial, "config", None)
        )
        if not isinstance(config, dict) or not config:
            continue
        for baseline in (item.before_config, state.index_config):
            state.blocked_action_attempts.add(
                history.build_attempt_key(
                    item.action_key,
                    baseline,
                    config,
                    state.runtime_capabilities,
                )
            )


def _block_action_attempt(
    state: AgentDoctorState,
    item: OptimizationHistoryItem,
) -> None:
    """품질이 나빠진 **정확한 전이** 하나를 차단한다.

    같은 action 의 다른 후보값과 다른 baseline 에서의 재시도는 열어 둔다(§5.1).
    """
    if item.action_attempt_key is not None:
        state.blocked_action_attempts.add(item.action_attempt_key)


def _record_legacy_block(
    state: AgentDoctorState,
    item: OptimizationHistoryItem,
    *,
    completed: bool = False,
) -> None:
    """구버전 (label, prescription_id) 집합을 계속 채운다.

    ⚠️ 실행 제어는 더 이상 이 값을 읽지 않는다(planner 에 넘기는 제외 목록은 action
    식별자뿐이다). 저장된 이력을 읽는 구버전 리포트 경로와의 호환을 위해 기록만 한다.
    단계 8 에서 소비처를 걷어낸 뒤 제거한다.
    """
    label = item.failure_labels[0] if item.failure_labels else ""
    prescription_id = item.selected_prescription_id
    if not (label and prescription_id):
        return
    if completed:
        state.completed_prescriptions.add((label, prescription_id))
    else:
        state.blacklist.add((label, prescription_id))


def _fmt_bool(value: bool) -> str:
    return "true" if value else "false"


def _fmt_score(value: float | None) -> str:
    return "None" if value is None else f"{value:.2f}"


def _fmt_verdict_score(value: float | None) -> str:
    """판정 로그용 점수. 탐색 신호(0~1)와 표시 스케일(0~100)을 함께 찍는다.

    로그는 개발자용이라 판정에 실제로 쓰인 0~1 값을 숨기면 안 되지만, 리포트는
    0~100 으로 나간다. 두 스케일을 한 줄에 두어야 실행 로그와 리포트를 나란히 놓고
    원인을 찾을 때 대조가 된다(before=0.75 와 "75.0→" 이 같은 사건으로 보이게).
    """
    if value is None:
        return "None"
    return f"{value:.2f}({value * 100:.1f})"


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
    """이번 Optimize 방문의 진단 입력을 출력한다."""
    print(f"[Optimize] 반복 횟수: {state.iteration}/{state.max_iterations} (진단 입력)")
    _log_eval_line(state.report)
    print(f"[Optimize] 발견된 문제: {_fmt_findings_summary(state.report)}")
    _log_measurement_noise(state)


def _log_measurement_noise(state: AgentDoctorState) -> None:
    """재측정 편차가 개선 마진을 삼킬 만큼 크면 경고한다.

    **마진 이상일 때만 찍는다.** 이 값은 평소엔 알 필요 없는 내부 관측이고, 매 방문
    출력하면 정작 문제일 때 묻힌다. 편차가 마진에 닿았다는 것은 "개선"과 "노이즈"가
    구분되지 않는다는 뜻이라, 그때만 소리를 낸다.
    """
    spread = history.max_repeated_measurement_spread(state.optimization_history)
    if spread is None or spread < history.MIN_IMPROVEMENT_MARGIN:
        return
    print(
        f"[Optimize] ⚠ 재측정 편차 {spread:.3f} ≥ 개선 마진 "
        f"{history.MIN_IMPROVEMENT_MARGIN:.3f} — 같은 config 가 다르게 측정됐습니다. "
        "이 구간의 '개선' 판정은 노이즈와 구분되지 않으므로 마진 재보정이 필요합니다."
    )


def _log_excluded_actions(state: AgentDoctorState) -> None:
    """planner 를 부르기 직전, 이번 선택에서 닫혀 있는 action 을 출력한다.

    baseline 이 바뀌어 이미 풀린 과거 차단은 빼고 보여준다 — 로그에 찍힌 action 이
    바로 아래에서 선택되면 원인 추적이 꼬인다. 진단 입력 로그가 아니라 여기서 찍는
    것은 롤백이 index_config 를 되돌린 **뒤**라야 비교 기준이 planner 와 같아지기
    때문이다.
    """
    blocked_keys = sorted(_active_excluded_action_keys(state))
    if blocked_keys:
        print(f"[Optimize] 제외된 action: [{', '.join(blocked_keys)}]")


# 처방 선택을 견제한 사유. eligibility 실패("이 파이프라인에서는 불가능")와 달리
# "지금은 시도할 가치가 없다"는 판정이라, 로그에서 구분해 보여줘야 사용자가
# "왜 뻔한 처방을 안 했나"를 되짚을 수 있다.
_GUARD_REASONS = {
    action_aggregator.REASON_NO_PROGRESS_CONFIG: "이미 측정한 config 로 되돌아감",
    action_aggregator.REASON_KEEP_PROTECTED_AXIS: "유지 판정된 변경의 되돌리기",
}


def _log_guard_rejections(metadata: dict[str, Any]) -> None:
    """무진전·되돌리기 견제로 제외된 action 을 출력한다."""
    for entry in metadata.get("rejected_actions") or []:
        headline = _GUARD_REASONS.get(entry.get("reason"))
        if headline is None:
            continue
        detail = ""
        protection = entry.get("keep_protection")
        if protection:
            detail = (
                f", 지지 {protection.get('candidate_support')}"
                f" ≤ 필요 {protection.get('required_support')}"
            )
        print(
            f"[Optimize]   견제된 action: {entry.get('action_key') or '-'} "
            f"({headline}{detail})"
        )


def _log_selected_action(request: OptimizationRequest) -> None:
    """Planner 가 고른 action 과 그 근거·탐색 범위를 출력한다.

    "후보 목록 N개 중 하나"가 아니라 **이미 정해진 변경 하나**를 보고한다. 경쟁은
    planner 안에서 끝났고, 밀린 후보는 runner-up 으로 따로 남긴다.
    """
    breakdown = request.action_score_breakdown
    print(
        f"[Optimize] 선택된 action: {request.action_key or '-'} "
        f"(지지 라벨 {len(request.supporting_labels)}개, "
        f"probe {len(request.supporting_probes)}개, "
        f"optimizer={request.optimizer})"
    )
    if breakdown:
        print(
            f"[Optimize] 선택 점수: {request.action_score:.3f} "
            f"(가중 지지 {breakdown.get('weighted_probe_support')}, "
            f"비용 {breakdown.get('base_cost')}, "
            f"출처 {breakdown.get('cost_source')}/{breakdown.get('confidence_source')})"
        )
    print(
        f"[Optimize] 탐색 범위: {_fmt_mapping(request.search_space)}, "
        f"후보 {request.max_trials}개"
    )
    for runner_up in request.metadata.get("runner_up_actions", []):
        print(
            f"[Optimize]   밀린 action: {runner_up['action_key']} "
            f"(점수 {runner_up['score']})"
        )
    for deferred in request.metadata.get("deferred_axes", []):
        print(
            f"[Optimize]   보류된 축: {deferred.get('axis')} "
            f"({deferred.get('reason')})"
        )
    _log_guard_rejections(request.metadata)


def _log_action_review(result: OptimizationResult) -> None:
    """Optimizer 의 실행 직전 재검증 결과를 출력한다."""
    for skipped in result.metadata.get("skipped_candidates", []):
        if not isinstance(skipped, dict):
            continue
        print(
            f"[Optimize] action SKIP: "
            f"{skipped.get('action_key') or skipped.get('prescription_id') or '-'}, "
            f"reason={skipped.get('reason') or 'unknown'}"
        )

    selected = result.selected_action
    if result.status == "failed":
        reason = result.metadata.get("error_code") or result.error or result.message
        if selected is None:
            print(f"[Optimize] 요청 FAIL: reason={reason or 'unknown'}")
        else:
            print(
                f"[Optimize] action FAIL: {selected.action_key or '-'}, "
                f"reason={reason or 'unknown'}"
            )
        return

    if result.status == "skipped":
        reason = result.metadata.get("error_code") or result.error or result.message
        if selected is None:
            print(f"[Optimize] 요청 SKIP: reason={reason or 'unknown'}")
            return
        print(
            f"[Optimize] action SKIP: {selected.action_key or '-'}, "
            f"reason={reason or 'unknown'}"
        )
        return

    if selected is not None:
        print(f"[Optimize] action SELECT: {selected.action_key or '-'}")


def _log_optimize_transition(
    *,
    action_key: str | None,
    supporting_labels: list[str] | None,
    prescription_id: str | None,
    before_config: dict[str, Any],
    after_config: dict[str, Any],
    changed_keys: list[str],
    reindex_required: bool,
    next_step: str,
    include_reindex: bool = True,
    include_next_step: bool = True,
) -> None:
    # 선택 단위는 action 이다. 라벨은 "왜 이 변경이 지지받았는가"를 설명할 뿐이라
    # 하나만 뽑지 않고 전부 보여준다. 처방 id 는 rules.py 선언으로 되짚기 위한 표시다.
    origin = f" (처방 {prescription_id})" if prescription_id else ""
    print(f"[Optimize] 선택한 action: {action_key or '-'}{origin}")
    print(f"[Optimize] 지지 라벨: {', '.join(supporting_labels or []) or '-'}")
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
    selected = result.selected_action
    print(f"[Optimize] 반복 횟수: {state.iteration}/{state.max_iterations} (처방 적용 후)")
    print(
        f"[Optimize] action 적용: {request.action_key or '-'}, "
        f"처방={prescription_id or '-'}"
    )
    if selected is not None:
        reason = result.message or selected.description or request.reason
        if reason:
            print(f"[Optimize] 선택 근거: {reason}")
    _log_optimize_transition(
        action_key=request.action_key,
        supporting_labels=list(request.supporting_labels),
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
    verdict_label = "KEEP" if verdict.keep else "ROLLBACK"
    print(
        f"[Optimize] 이전 처방 판정: {verdict_label}, "
        f"action={item.action_key or '-'}, "
        f"before={_fmt_verdict_score(verdict.before_score)}, "
        f"after={_fmt_verdict_score(verdict.after_score)}"
    )
    print(f"[Optimize] 판정 근거: {verdict.reason or '-'}")
    _log_optimize_transition(
        action_key=item.action_key,
        supporting_labels=list(item.supporting_labels or item.failure_labels),
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
        f"before={_fmt_verdict_score(verdict.before_score)}, "
        f"after={_fmt_verdict_score(verdict.after_score)}"
    )


def _log_manual_prescriptions(decision: OptimizeDecision) -> None:
    """D그룹(manual) 라벨의 사람 조치를 로그에 남긴다. config 처방과 달리 자동 적용되지
    않으므로, 어떤 라벨에 무슨 매뉴얼 스텝이 필요한지 출력로그에도 드러낸다."""
    labels = getattr(decision, "manual_labels", None) or []
    for label in labels:
        rule = rules.get_rule(label) or {}
        headline = (rule.get("manual_action", "") or "").strip()
        print(f"[Optimize] 수동 조치 필요: {label}" + (f" — {headline}" if headline else ""))
        for i, presc in enumerate(rule.get("prescriptions", []), start=1):
            if not presc.get("manual"):
                continue
            action = presc.get("action", "") or presc.get("id", "")
            print(f"[Optimize]   {i}. {action}")


def _log_optimize_decision(
    state: AgentDoctorState,
    decision: OptimizeDecision,
) -> None:
    next_step = "Serve 이동" if decision.next_route == "serve" else decision.next_route
    action = "SKIP" if decision.status == "skipped" else decision.status.upper()
    print(f"[Optimize] 행동 결정: {action}, reason={decision.reason or '-'}")
    # 요청이 만들어지지 않은 방문에서도 견제 사유는 남겨야 한다 — "처방 없음"만
    # 보이면 견제 때문인지 후보가 없어서인지 구분되지 않는다.
    _log_guard_rejections(decision.metadata)
    _log_manual_prescriptions(decision)

    _log_optimize_transition(
        action_key=None,
        supporting_labels=[],
        prescription_id=None,
        before_config={},
        after_config={},
        changed_keys=[],
        reindex_required=bool(state.reindex_required),
        next_step=f"{next_step} ({decision.status})",
        include_reindex=False,
    )


def _extend_deferred(
    deferred: list[dict[str, Any]],
    entries: list[dict[str, Any]],
) -> None:
    """보류 목록에 새 항목만 더한다.

    한 방문 안에서 planner 를 여러 번 호출하므로(적용 불가 action 을 건너뛰며),
    같은 보류가 매번 다시 나온다. 중복을 그대로 실으면 리포트가 같은 말을 반복한다.
    """
    seen = {
        (item.get("action_key"), item.get("prescription_id"), item.get("reason"))
        for item in deferred
    }
    for entry in entries:
        key = (entry.get("action_key"), entry.get("prescription_id"), entry.get("reason"))
        if key not in seen:
            seen.add(key)
            deferred.append(entry)


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
    """Optimize 노드 진입점. 성공·스킵·수동·오류 어느 경로든 같은 state 를 반환한다.

    ── 종료 불변조건 ─────────────────────────────────────────────────
    iteration 은 "새 ActionStudy 를 적용한 횟수"라 **소비하지 않는 경로가 여럿**이다.

      1. sweep 중간 후보 진행 (_continue_internal_study)   → status=applied
      2. sweep 종료 (_finish_internal_study)               → applied/verified/rolled_back
      3. study 오류 복원 (_fail_active_study)              → rolled_back/error
      4. 자동 처방 대상 없음 (decision != apply_optimize)  → skipped/manual_required/…
      5. 예산 소진 후 판정만                                → verified/rolled_back
      6. 적용 가능한 action 소진                            → skipped/rolled_back
      7. optimizer 가 patch 를 못 만듦                      → skipped/failed/rolled_back
      8. 품질 통과 + 유지 판정                              → verified

    그래서 실제 종료를 보장하는 것은 iteration 이 아니라 **optimize_visit_count** 다.
    아래 증가는 위 어떤 분기보다도 먼저, 조건 없이 실행된다 — 즉 방문이 한 번이라도
    본문에 들어오면 반드시 소비된다. 유일하게 증가하지 않는 경로는 상한 도달
    (_stop_at_optimize_visit_limit)이며, 그 경로는 진행 중 설정을 복원하고 종료한다.
    복원이 config 를 바꿔 status=rolled_back 이 되면 Index→Eval 을 한 바퀴 더 도는데,
    그때는 복원할 pending 이 없어 status=verified 로 Serve 에 도달한다(수렴 1회).

    최악 방문 수는 max_optimize_visits(20) + 상한 처리 1회다. graph.py 는 이 값을
    읽지 않지만 그래서 오히려 안전하다 — 라우팅을 고치지 않아도 무조건 닫힌다.
    """
    state.current_agent = "optimize"
    try:
        # ⚠️ 로깅보다 **먼저** 소비한다. 콘솔 인코딩 오류 같은 표시 계층 예외가
        # 방문을 공짜로 만들면 안 된다 — 그 순간 이 불변조건이 무너진다.
        if state.optimize_visit_count >= state.max_optimize_visits:
            _log_optimize_input(state)
            return _stop_at_optimize_visit_limit(state)
        state.optimize_visit_count += 1
        _log_optimize_input(state)
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
        # planner 에 넘기는 제외 목록은 action 식별자뿐이다. 구버전 (label,
        # prescription_id) 튜플은 실행 제어에 쓰지 않는다(구현계획 §8.2 완료 조건).
        #   ActionAttemptKey  → 정확한 전이만 차단(후보값 단위)
        #   ActionStudyKey    → 그 baseline 에서 결론이 난 탐색을 차단
        #   action key 문자열 → 이번 방문에서만 막는 visit-local 제외
        visit_exclusions: set[Any] = set(state.blocked_action_attempts)
        visit_exclusions.update(state.completed_action_studies)
        visit_exclusions.update(
            _unjudgeable_exclusions(state.optimization_history)
        )
        visit_exclusions.update(
            _rollback_action_cooldown_exclusions(state.optimization_history)
        )
        deferred_runtime: list[dict[str, Any]] = []
        if (
            judged_item is not None
            and judged_item.metadata.get("reranker_execution_incomplete")
        ):
            action_key = judged_item.action_key
            if action_key:
                # runtime 이 못 돌아 효과를 못 잰 것이라 품질 실패가 아니다.
                # 이번 방문에서만 막고 다음 파이프라인 실행에서는 다시 시도한다.
                visit_exclusions.add(action_key)
                deferred_runtime.append(
                    {
                        "action_key": action_key,
                        "failure_label": (
                            judged_item.failure_labels[0]
                            if judged_item.failure_labels
                            else ""
                        ),
                        "prescription_id": judged_item.selected_prescription_id,
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
        _log_excluded_actions(state)
        while True:
            request, decision = planner.plan(state, blacklist=visit_exclusions)
            # planner 가 실행 불가로 거른 것 중 runtime·선행조건 사유는 보류로 보고한다.
            # 요청이 만들어졌든 아니든 사유가 나오는 곳이 다를 뿐 보고 책임은 같다.
            _extend_deferred(
                deferred_runtime,
                _deferred_from_rejected_actions(
                    (request.metadata if request is not None else decision.metadata)
                    .get("rejected_actions"),
                    state.runtime_capabilities,
                ),
            )
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
            _log_selected_action(request)

            # iteration 은 "새 ActionStudy 를 적용한 횟수"다(구현계획 §5.3).
            # study key 는 baseline fingerprint 를 담고 있으므로
            #   - 같은 action 의 내부 sweep → _continue_internal_study 가 이 경로를
            #     타지 않아 iteration 하나로 묶인다
            #   - rollback 후 다른 action  → 새 study → 새 iteration
            #   - 같은 action 이라도 새 baseline → 새 study → 새 iteration
            # 이 셋이 모두 자동으로 성립한다. 라벨 비교를 쓰던 예전에는 같은 라벨의
            # 다른 처방이 예산을 소비하지 않아, 라벨 하나가 예산 밖에서 무한히
            # 처방을 갈아 끼울 수 있었다.
            previous_study = history.last_action_study_key(state.optimization_history)
            new_study = history.study_key_for_request(
                request, state.runtime_capabilities
            )
            starts_new_action_study = (
                new_study is None or previous_study != new_study
            )
            if starts_new_action_study and state.iteration >= state.max_iterations:
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
            _log_action_review(result)
            # skipped action(baseline 무개선·적용 불가 경로·빈 search space)이면 포기하지
            # 않고 그 탐색을 소진 처리한 뒤 다음 순위 action 으로 넘어간다. 한 축이
            # 막혀도(예: enable_hybrid 는 이미 hybrid 라 no-op) 다른 actionable finding 이
            # 처방받을 기회를 준다. (issue #26)
            if result.status != "skipped":
                break

            prescription_id = request.prescription_id
            rejected_action = request.action_key
            if not rejected_action or rejected_action in visit_exclusions:
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
            visit_exclusions.add(rejected_action)
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
                        "action_key": rejected_action,
                        "failure_label": next(iter(request.supporting_labels), ""),
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
                # optimizer 가 "이 baseline 에서는 이 탐색을 적용할 수 없다"고 판정한
                # 것이므로 방문을 넘어 남긴다. baseline fingerprint 를 담고 있어
                # config 가 바뀌면 같은 축을 다시 볼 수 있다(§5.1).
                rejected_study = history.study_key_for_request(
                    request, state.runtime_capabilities
                )
                if rejected_study is not None:
                    state.completed_action_studies.add(rejected_study)
                if prescription_id:
                    label = next(iter(request.supporting_labels), "")
                    if label:
                        state.blacklist.add((label, prescription_id))

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
        prescription_id = request.prescription_id
        item = history.create_pending_item(
            state,
            request,
            prescription_id,
            before_config,
            before_report,
            # attempt fingerprint 는 '이번에 적용한 값'으로 만든다. 탐색 범위 전체를
            # 넣으면 sweep 중간 후보가 서로를 차단한다.
            applied_changes=dict(result.config_patch.changes),
        )
        item.metadata["reindex_required"] = bool(result.needs_reindex)
        if request.action_key in _RERANKER_ACTION_KEYS:
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
        if starts_new_action_study:
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
                "action_key": request.action_key,
                "supporting_labels": list(request.supporting_labels),
                # 구버전 리포트 소비처 호환. 단계 7 에서 action 표시로 전환한다.
                "failure_label": next(iter(request.supporting_labels), ""),
                "selected_prescription": prescription_id,
                "reindex_required": bool(result.needs_reindex),
            }
        )
        if request.action_key in _RERANKER_ACTION_KEYS:
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
    """직전 후보 결과를 기록하고 같은 action 의 다음 후보를 진행한다.

    축을 가리지 않는다 — study 는 단일 canonical 축 하나를 sweep 하며, 어느 축인지는
    item.action_key 가 말해 준다. 이 경로 전체가 iteration 하나에 속하므로(§5.3)
    여기서는 예산을 소비하지 않는다. 대신 방문마다 optimize_visit_count 는 run() 진입부
    에서 이미 증가했다 — 그것이 이 루프의 종료를 보장하는 절대 안전선이다.
    """
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
    _log_action_review(result)
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
        item.metadata["applied_changes"] = dict(result.config_patch.changes)
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

    # min_delta 비교에 쓸 scorable baseline 이 없어 sweep 이 끝나지 못한 경우는
    # '처방이 나빴다'가 아니라 '측정이 성립하지 않았다'이다. 품질 blacklist 대신
    # unjudgeable 로 기록해 다음 파이프라인 실행에서 재시도할 수 있게 한다.
    unjudgeable = (
        result.metadata.get("stop_reason") == "missing_scorable_baseline"
    )
    return _fail_active_study(
        state,
        item,
        result.error or result.message or "internal study를 완료하지 못했습니다.",
        retryable=True,
        unjudgeable=unjudgeable,
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
    # chunk 사전검증(prescreener)이 고른 후보의 best_score 는 종합점수가 아니라 정답
    # span 포함률이다. after_score 자리에 다른 축이 들어온다는 뜻이므로, 표시 계층이
    # 이 값을 "종합점수"라고 부르지 않도록 판정·이력에 함께 실어 보낸다.
    proxy_only = bool(result.metadata.get("proxy_only"))
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
            proxy_only=proxy_only,
            reason="모든 후보 평가 후 baseline이 가장 좋아 원래 설정을 유지",
        )
        # 탐색은 정상적으로 끝났다(결론이 baseline 이었을 뿐) → study 로 기록한다.
        # 개별 후보를 attempt 로 막지 않는 이유: baseline 이 달라지면 같은 후보가
        # 다시 이길 수 있고, 이 study key 자체가 그 baseline 에서의 재시작을 막는다.
        _block_action_study(state, item)
        _record_legacy_block(state, item)
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
                proxy_only=proxy_only,
                reason=f"sweep 최적 후보가 하한선을 위반함 {floor_violations} → baseline 복원",
            )
            # 하한선 위반은 품질 실패다 — 승자 전이를 attempt 로 막고, 같은 탐색을
            # 이 baseline 에서 다시 돌리지도 않는다.
            if item.action_key:
                state.blocked_action_attempts.add(
                    history.build_attempt_key(
                        item.action_key,
                        item.before_config,
                        dict(result.config_patch.changes),
                        state.runtime_capabilities,
                    )
                )
            _block_action_study(state, item)
            _record_legacy_block(state, item)
            state.status = "rolled_back" if changed else "verified"
            state.reindex_required = bool(item.metadata.get("reindex_required", True))
        else:
            config_mapper.apply_config_patch(state.index_config, result.config_patch)
            verdict = Verdict(
                keep=True,
                before_score=before_score,
                after_score=after_score,
                proxy_only=proxy_only,
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
            # 이 baseline 에서 탐색이 정상 종료됐다 → 같은 study 를 다시 시작하지 않는다.
            _block_action_study(state, item)
            _record_legacy_block(state, item, completed=True)
    else:
        return _fail_active_study(
            state,
            item,
            result.error or result.message or "best 후보를 적용할 수 없습니다.",
        )

    item.after_config = dict(state.index_config)
    # 세 종료 분기(baseline 승·하한선 위반·정상 승자) 모두 "이 탐색은 끝났다"이므로
    # 측정한 후보를 여기 한 곳에서 소진 처리한다.
    _block_evaluated_sweep_candidates(state, item)
    if baseline_selected:
        baseline_report = item.metadata.get("before_report")
        item.after_metrics = dict(getattr(baseline_report, "ragas_scores", {}) or {})
    else:
        item.after_metrics = best_metrics
    item.status = "applied" if verdict.keep else "failed"
    item.rollback_reason = None if verdict.keep else verdict.reason
    # 표시·게이트용 종합점수(0~100). before 는 baseline 리포트에서, after 는 baseline
    # 복원 시 before 와 동일(설정을 되돌렸으므로), 아니면 sweep 이 full report 를 남기지
    # 않아 미상(None).
    before_composite = history._read_composite(item.metadata.get("before_report"))
    # baseline 복원이면 설정을 되돌렸으니 after=before. 새 후보가 이겼으면 그 후보의
    # 관측값에 실린 composite_total 을 쓴다(_report_metrics 가 실어둠). 없으면 None
    # — 이때 표시부는 after_score×100 으로 채우지 않는다(그 값이 composite 라는 보장이
    # 없다). score_display 가 한쪽만 있는 쌍을 표시하지 않는 이유다.
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
            # 표시 계층이 프록시 지표를 "종합점수"로 부르지 않게 하는 신호.
            "proxy_only": proxy_only,
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
        action_key=item.action_key,
        supporting_labels=list(item.supporting_labels or item.failure_labels),
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
    unjudgeable: bool = False,
) -> AgentDoctorState:
    """study 오류 시 baseline으로 복원하고 재시도 가능 여부를 구분한다.

    ``unjudgeable``은 '측정 자체가 성립하지 않음'(예: min_delta 비교에 필요한
    scorable baseline 부재)이다. 처방이 나빴다는 증거가 아니므로 차단 집합에
    넣지 않고, 같은 실행 안에서만 _unjudgeable_exclusions로 재선택을 제한한다.
    """
    before_config_for_log, _before_report, changed = _restore_history_item_baseline(
        state,
        item,
        reason,
        metadata={
            "study_error": reason,
            "study_retryable": retryable,
            "unjudgeable": unjudgeable,
        },
    )
    action_key = item.action_key or ""
    previous_same_errors = sum(
        1
        for previous in state.optimization_history
        if previous is not item
        and previous.action_key == action_key
        and previous.metadata.get("study_error")
    )
    # 상태 계약 손상은 같은 탐색으로 회복되지 않으므로 즉시 차단한다. 일시적인
    # adapter 오류는 한 번 재시도하되 반복되면 무한 루프 방지를 위해 차단한다.
    # 측정 불가(unjudgeable)는 품질 증거가 없으므로 차단 대신 실행 범위
    # 제한(_unjudgeable_exclusions)에 맡긴다.
    # 차단 단위가 attempt 가 아니라 study 인 이유: 실패한 것은 후보값 하나가 아니라
    # 탐색 자체이며, 같은 baseline 에서 후보만 바꿔도 같은 오류가 재현된다.
    if (
        action_key
        and not unjudgeable
        and (not retryable or previous_same_errors >= 1)
    ):
        _block_action_study(state, item)
        _record_legacy_block(state, item)
    state.status = "rolled_back" if changed else "error"
    state.error = None if changed else reason
    diff = config_mapper.build_config_diff(before_config_for_log, state.index_config)
    next_step = (
        _config_change_next_step(bool(state.reindex_required))
        if state.status == "rolled_back"
        else f"Serve 이동 ({state.status})"
    )
    _log_optimize_transition(
        action_key=item.action_key,
        supporting_labels=list(item.supporting_labels or item.failure_labels),
        prescription_id=item.selected_prescription_id,
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
    """직전에 적용한 처방(pending)을 판정한다. 나빴으면 config·진단서 롤백 + blacklist.
    판정한 이력 항목과 Verdict 를 반환한다(판정할 게 없으면 (None, None, None)).
    호출부가 이 둘로 롤백 여부(not verdict.keep) 판단 + 사용자 리포트를 만든다.

    롤백은 **config 와 state.report 를 함께** 되돌린다. 둘 중 하나만 되돌리면 같은
    방문의 나머지 로직이 서로 다른 config 를 가정한다 — 복원된 설정으로 처방을
    고르면서 판단 근거는 되돌리기 전 finding 인 상태가 된다.

    3번째 반환값(rollback_baseline_report): 롤백했을 때 '복원된 config 가 실제로
    받았던 점수'를 담은 리포트(=이 처방의 before_report). 롤백 후 같은 방문에서
    이어서 제안되는 다음 처방의 비교 기준(before_report)으로 써야 한다. state.report
    를 되돌린 지금은 같은 값이지만, 비교 기준을 호출부에서 눈에 보이게 두는 편이
    낫다 — 측정이 없어 진단서를 못 되돌린 경우(before_report 부재)에는 둘이 갈린다.
    유지/판정없음이면 None."""
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
        # 진단서도 함께 되돌린다. config 를 복원해 놓고 리포트를 그대로 두면, 같은
        # 방문에서 이어지는 처방 선택이 **존재하지 않는 config 의 finding** 을 보고
        # 결정한다. 실제로 abstention 처방을 롤백한 방문에서, 그 처방이 스스로 만든
        # generation_wrongful_abstention 3건이 남은 리포트로 다음 처방을 골랐다
        # (그 증상을 되돌리는 abstention_relaxed 가 후보 2위까지 올라왔다).
        #
        # 이 복원은 Eval 이 다음 방문에서 하는 일과 같다 — 롤백 진단 캐시가 복원된
        # config 의 리포트를 그대로 돌려준다(agents/eval/agent.py 의 '롤백 진단 캐시
        # 복원'). 즉 한 스텝 뒤에 성립할 상태를 지금 맞춰 주는 것이지 새 값을 지어
        # 내는 것이 아니다. 측정이 없어(before_report 부재) 롤백한 경우는 되돌릴
        # 진단서 자체가 없으므로 건드리지 않는다.
        #
        # ⚠️ 되돌아가는 것은 **state.report 포인터뿐**이다. 열화된 측정 자체는 이 처방의
        # 이력 항목에 그대로 남는다 — 바로 위에서 after_report 를 먼저 잡아 두고(복원 전),
        # finalize_item 이 after_score·after_composite·after_metrics·after_config 로
        # 기록한다. 즉 "무엇을 쟀는가"는 보존되고 "지금 서 있는 config 의 진단서가
        # 무엇인가"만 복원된다. 마지막 Eval 원본이 필요한 소비처는 그 이력을 봐야 한다.
        if before_report is not None:
            state.report = before_report
        # 측정이 없어(unjudgeable) 롤백한 경우는 '나빴다는 증거'가 아니므로 차단
        # 등록을 건너뛴다. 차단하면 리포트 부재만으로 action 이 소진되고, before_report
        # None 이 다음 방문으로 전파돼 판정 불가→롤백→차단이 연쇄될 수 있다(리뷰 #36).
        #
        # 품질 악화가 확인된 경우에만 **정확한 전이 하나**를 막는다. 같은 축의 다른
        # 후보값과, config 가 바뀐 뒤의 같은 값은 다시 시도할 수 있다(§5.1).
        if not verdict.unjudgeable:
            _block_action_attempt(state, pending)
            _record_legacy_block(state, pending)

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
    if not _is_reranker_item(pending):
        return verdict
    if verdict.floor_violations != ["context_precision"]:
        return verdict
    # 하한선 위반이 있으면 judge가 floor 판정으로 먼저 반환하므로, 여기가 이 경로의
    # 유일한 점수 관문이다. "종합점수가 올랐다"를 judge와 같은 기준으로 재야 한다 —
    # 아니면 노이즈 수준의 상승(+0.001)이 하한선 위반 처방을 되살린다.
    if not history.meets_improvement_margin(
        verdict.after_score - verdict.before_score
    ):
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
        # judge 의 reason 과 같은 규약 — "종합점수"라고 이름을 대는 문장의 숫자는
        # 리포트 헤드라인과 같은 0~100 이어야 한다(판정은 계속 0~1 탐색 신호로 한다).
        reason=(
            "reranker 적용 후 context_precision 단독 하한선 위반이 있었지만 "
            f"종합점수 상승 {history.to_display_scale(verdict.before_score):.1f}"
            f"→{history.to_display_scale(verdict.after_score):.1f}"
            f"(마진 {history.to_display_scale(history.MIN_IMPROVEMENT_MARGIN):.1f} 충족), "
            f"retrieval_low_rank 감소 {before_low_rank}→{after_low_rank}로 유지"
        ),
        unjudgeable=verdict.unjudgeable,
    )


def _label_count(report: DiagnosticReport, label: str) -> int:
    """리포트에서 특정 라벨의 확정 finding 개수를 센다."""
    return sum(
        1
        for finding in report.findings
        if finding.confirmed and finding.label == label
    )


def _reranker_execution_incomplete(
    pending: OptimizationHistoryItem,
    after_report: DiagnosticReport,
) -> bool:
    """reranker 처방 후 Eval이 일부라도 실제 CrossEncoder 점수를 못 만들었는지 본다."""
    if not _is_reranker_item(pending):
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
