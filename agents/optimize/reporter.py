"""
agents/optimize/reporter.py
Optimize 모듈의 "사용자 리포트" 계층.

[역할]
  planner/history가 만든 내부 결정·판정 결과를 받아, 사용자가 읽기 쉬운
  OptimizationReport 로 번역한다. 무엇이 문제였고, 어떤 처방을 적용/제안했으며,
  config가 어떻게 바뀌었고, 사람이 직접 해야 할 조치가 무엇인지 정리한다.

  "결과의 사람 친화적 번역"까지가 reporter의 책임이다. 실제 config 변경은
  config_mapper, 유지/롤백 판정은 history, 처방 선택은 planner 소관이다.

[읽는 것]  OptimizeDecision(필수), OptimizationRequest/Verdict/ConfigDiff(선택),
           rules.py(manual 라벨의 사람용 조치 문구)
[쓰는 것]  없음 — OptimizationReport 를 만들어 반환만 한다(planner/history와 동일 규칙).

[MVP 결정 사항]  (나중에 재검토 가능)
  - manual 라벨의 조치 문구는 rules.py 의 manual_action 필드에서 읽는다.
    문구가 비어있으면 일반 fallback 문장을 쓴다.
  - selected_prescription 은 요청 후보의 첫 번째(가장 가벼운) 처방 id 로 잡는다.
    실제 적용된 처방을 정확히 추적하는 건 optimizer/agent.py 완성 후 개선.
  - propose_only 는 뼈대만. 현재 기본 흐름은 apply_optimize.
"""
from __future__ import annotations

import uuid

from agents.optimize import rules
from agents.optimize.schemas import (
    ConfigDiff,
    OptimizationReport,
    OptimizationRequest,
    OptimizeDecision,
    Verdict,
)


# ── 진입점 ────────────────────────────────────────────────────────

def build_report(
    decision: OptimizeDecision,
    request: OptimizationRequest | None = None,
    verdict: Verdict | None = None,
    diff: ConfigDiff | None = None,
) -> OptimizationReport:
    """흐름 결정(decision)을 중심으로 상황별 OptimizationReport 를 만든다."""
    if decision.mode == "apply_optimize":
        return _report_apply(decision, request, verdict, diff)
    if decision.mode == "manual_required":
        return _report_manual(decision)
    if decision.mode == "propose_only":
        return _report_propose(decision, request)
    if decision.mode == "use_current":
        return _report_use_current(decision)
    # 예상 못한 mode는 조용히 넘기지 않고 바로 드러낸다(planner 버그 조기 발견).
    raise ValueError(f"알 수 없는 decision.mode: {decision.mode}")


# ── 상황별 리포트 ─────────────────────────────────────────────────

def _report_apply(
    decision: OptimizeDecision,
    request: OptimizationRequest | None,
    verdict: Verdict | None,
    diff: ConfigDiff | None,
) -> OptimizationReport:
    """자동 처방을 적용한 경우. verdict가 있으면 유지/롤백 결과까지 반영."""
    prescription = _selected_prescription(request)
    kept = verdict.keep if verdict is not None else None

    if verdict is None:
        status = "proposed"
        summary = f"'{_label(request)}' 문제에 '{prescription}' 처방을 적용했습니다."
    elif kept:
        status = "applied"
        summary = (
            f"'{prescription}' 처방으로 점수가 "
            f"{verdict.before_score:.1f}→{verdict.after_score:.1f}로 올라 적용을 유지했습니다."
        )
    else:
        status = "failed"
        summary = (
            f"'{prescription}' 처방을 시도했으나 개선되지 않아 되돌렸습니다. "
            f"({verdict.reason})"
        )

    return OptimizationReport(
        report_id=_new_id(),
        request_id=_request_id(decision, request),
        status=status,
        summary=summary,
        problem=_problem_text(request),
        selected_prescription=prescription,
        config_changes=_config_changes(diff, request),
        expected_tradeoffs=_tradeoffs(request),
        manual_actions=_manual_actions(decision.manual_labels),
        next_steps=_next_steps_apply(kept),
        diff=diff,
        metadata=_score_metadata(verdict),
    )


def _report_manual(decision: OptimizeDecision) -> OptimizationReport:
    """자동 처방 없이 사람 개입이 필요한 경우."""
    actions = _manual_actions(decision.manual_labels)
    return OptimizationReport(
        report_id=_new_id(),
        request_id=_request_id(decision, None),
        status="manual_required",
        summary=(
            f"자동으로 고칠 수 있는 문제는 없고, 사람이 직접 조치해야 할 "
            f"항목이 {len(actions)}개 있습니다."
        ),
        manual_actions=actions,
        next_steps=["안내된 조치를 완료한 뒤 다시 진단을 실행하세요."],
    )


def _report_propose(
    decision: OptimizeDecision,
    request: OptimizationRequest | None,
) -> OptimizationReport:
    """제안만 하고 적용하지 않는 경우(propose_only). MVP 뼈대."""
    prescription = _selected_prescription(request)
    return OptimizationReport(
        report_id=_new_id(),
        request_id=_request_id(decision, request),
        status="proposed",
        summary=f"'{_label(request)}' 문제에 '{prescription}' 처방을 제안합니다(자동 적용하지 않음).",
        problem=_problem_text(request),
        selected_prescription=prescription,
        config_changes=_config_changes(None, request),
        expected_tradeoffs=_tradeoffs(request),
        manual_actions=_manual_actions(decision.manual_labels),
        next_steps=["제안을 검토하고 승인하면 적용합니다."],
    )


def _report_use_current(decision: OptimizeDecision) -> OptimizationReport:
    """변경 없이 현재 설정을 유지하는 경우(already_optimal / skipped)."""
    if decision.status == "already_optimal":
        summary = "이미 모든 목표 지표를 달성해 설정을 변경하지 않았습니다."
    else:
        summary = "적용할 수 있는 처방이 없어 현재 설정을 유지합니다."
    return OptimizationReport(
        report_id=_new_id(),
        request_id=_request_id(decision, None),
        status=decision.status,
        summary=summary,
        manual_actions=_manual_actions(decision.manual_labels),
        next_steps=["현재 설정으로 서빙을 진행합니다."],
    )


# ── 보조 함수 ─────────────────────────────────────────────────────

def _new_id() -> str:
    return str(uuid.uuid4())


def _request_id(
    decision: OptimizeDecision, request: OptimizationRequest | None
) -> str:
    """리포트에 실을 request_id. request가 없으면 decision에 연결된 것 사용."""
    if request is not None:
        return request.request_id
    return decision.request_id or ""


def _label(request: OptimizationRequest | None) -> str:
    """이번에 다룬 대표 진단 라벨."""
    return request.failure_label if request is not None else ""


def _problem_text(request: OptimizationRequest | None) -> str:
    """문제 설명. request.reason 우선, 없으면 라벨명."""
    if request is None:
        return ""
    return request.reason or request.failure_label


def _selected_prescription(request: OptimizationRequest | None) -> str | None:
    """이번에 적용/제안한 대표 처방 id. MVP는 첫(가장 가벼운) 후보 기준."""
    if request is None or not request.candidates:
        return None
    return request.candidates[0].id


def _config_changes(
    diff: ConfigDiff | None, request: OptimizationRequest | None
) -> list[str]:
    """config 변경을 사람이 읽는 문자열 목록으로. diff 우선, 없으면 후보 patch."""
    if diff is not None:
        lines: list[str] = []
        for key in diff.changed_keys:
            before = diff.before_config.get(key)
            after = diff.after_config.get(key)
            lines.append(f"{key}: {before} → {after}")
        for key in diff.added_keys:
            lines.append(f"{key}: (신규) {diff.after_config.get(key)}")
        return lines
    # diff가 없으면(제안만 등) 후보의 제안 변경을 그대로 보여준다.
    if request is None or not request.candidates:
        return []
    patch = request.candidates[0].patch
    if patch is None:
        return []
    return [f"{k}: {v}" for k, v in patch.changes.items()]


def _tradeoffs(request: OptimizationRequest | None) -> list[str]:
    """대표 후보 처방의 예상 부작용."""
    if request is None or not request.candidates:
        return []
    return list(request.candidates[0].tradeoffs)


def _manual_actions(labels: list[str]) -> list[str]:
    """manual 라벨을 사람용 조치 문구로. rules.py 의 manual_action 을 읽는다."""
    actions: list[str] = []
    for label in labels:
        rule = rules.get_rule(label)
        text = (rule.get("manual_action", "") or "").strip() if rule else ""
        if text:
            actions.append(f"[{label}] {text}")
        else:
            actions.append(f"[{label}] 사람의 직접 확인이 필요합니다.")
    return actions


def _next_steps_apply(kept: bool | None) -> list[str]:
    """적용 결과에 따른 다음 안내."""
    if kept is False:
        return ["처방을 되돌렸습니다. 다음 후보로 재시도하거나 서빙을 진행합니다."]
    return ["변경을 반영하고 서빙을 진행합니다."]


def _score_metadata(verdict: Verdict | None) -> dict:
    """UI 표시용 점수/위반 정보."""
    if verdict is None:
        return {}
    return {
        "before_score": verdict.before_score,
        "after_score": verdict.after_score,
        "floor_violations": verdict.floor_violations,
    }
