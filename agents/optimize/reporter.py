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

[설명의 단위는 action 이다]
  "어떤 라벨을 골랐나"가 아니라 **"어떤 config 변경을 왜 골랐나"**를 설명한다.
  라벨은 그 변경을 지지한 근거이므로 대표 하나가 아니라 전체를 보여준다 — 여러
  라벨이 같은 변경을 지지했다는 사실 자체가 선택 근거이기 때문이다.

  함께 드러내는 것: 같은 축의 반대편(opposing), 근소한 차이로 보류된 축(deferred),
  점수 구성(score breakdown), 그리고 지지받은 라벨 중 **실제로 해결된 것**.

[MVP 결정 사항]  (나중에 재검토 가능)
  - manual 라벨의 조치 문구는 rules.py 의 manual_action 필드에서 읽는다.
    문구가 비어있으면 일반 fallback 문장을 쓴다.
  - propose_only 는 뼈대만. 현재 기본 흐름은 apply_optimize.
"""
from __future__ import annotations

import uuid

from agents.optimize import rules
from agents.optimize.schemas import (
    ConfigDiff,
    OptimizationHistoryItem,
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


def build_trial_report(
    item: OptimizationHistoryItem,
    verdict: Verdict,
) -> OptimizationReport:
    """판정이 끝난(finalize된) 처방 하나를 사용자 리포트로 번역한다.

    build_report 가 planner 의 '흐름 결정(decision)'을 번역하는 것과 달리,
    이쪽은 history 의 '유지/롤백 판정(verdict)'을 번역한다. 판정은 지난 방문에
    적용한 처방(=이력 항목)을 대상으로 하므로, 리포트가 설명하는 처방 이름과
    점수가 서로 어긋나지 않도록 입력을 이력 항목에서만 뽑는다.
    """
    prescription = item.selected_prescription_id
    # 구버전 이력에는 action_key 가 없다. 그때는 처방 id 를 이름으로 쓴다.
    supporting = list(item.supporting_labels or item.failure_labels)
    subject = _action_phrase(item.action_key, prescription)
    resolved = list(item.metadata.get("resolved_labels") or [])
    remaining = list(item.metadata.get("remaining_labels") or [])

    if verdict.keep:
        status = "applied"
        before, after = _display_scores(verdict)
        summary = (
            f"{subject}(으)로 점수가 "
            f"{before:.1f}→{after:.1f}로 올라 적용을 유지했습니다."
        )
        # "지지받았다"와 "해결됐다"는 다른 사실이다. 유지된 경우에만 해결 여부를
        # 말할 수 있고(롤백은 설정을 되돌렸으므로 귀속 자체가 성립하지 않는다),
        # 실제로 사라진 라벨이 있을 때만 덧붙인다.
        if resolved:
            summary += f" 해결된 문제: {', '.join(resolved)}."
        if remaining:
            summary += f" 남은 문제: {', '.join(remaining)}."
    else:
        status = "failed"
        summary = (
            f"{subject}을(를) 시도했으나 개선되지 않아 되돌렸습니다. "
            f"({verdict.reason})"
        )

    return OptimizationReport(
        report_id=_new_id(),
        request_id=item.request_id,
        status=status,
        summary=summary,
        problem=item.reason or ", ".join(supporting),
        selected_prescription=prescription,
        config_changes=_config_changes_from_configs(
            item.before_config, item.after_config
        ),
        next_steps=_next_steps_apply(verdict.keep),
        metadata=_score_metadata(verdict),
        action_key=item.action_key,
        supporting_labels=supporting,
        # 롤백된 처방에 "해결됐다"를 붙이지 않는다 — 설정을 되돌렸으므로 그 개선은
        # 지금 config 에 남아 있지 않다.
        resolved_labels=resolved if verdict.keep else [],
        remaining_labels=remaining if verdict.keep else supporting,
    )


# ── 상황별 리포트 ─────────────────────────────────────────────────

def _report_apply(
    decision: OptimizeDecision,
    request: OptimizationRequest | None,
    verdict: Verdict | None,
    diff: ConfigDiff | None,
) -> OptimizationReport:
    """자동 처방을 적용한 경우. verdict가 있으면 유지/롤백 결과까지 반영."""
    prescription = _selected_prescription(request)
    action_key = request.action_key if request is not None else None
    supporting = list(request.supporting_labels) if request is not None else []
    subject = _action_phrase(action_key, prescription)
    kept = verdict.keep if verdict is not None else None

    if verdict is None:
        status = "proposed"
        summary = f"{_support_phrase(supporting, request)} {subject}을(를) 적용했습니다."
    elif kept:
        status = "applied"
        before, after = _display_scores(verdict)
        summary = (
            f"{subject}(으)로 점수가 "
            f"{before:.1f}→{after:.1f}로 올라 적용을 유지했습니다."
        )
    else:
        status = "failed"
        summary = (
            f"{subject}을(를) 시도했으나 개선되지 않아 되돌렸습니다. "
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
        action_key=action_key,
        supporting_labels=supporting,
        opposing_labels=list(request.opposing_labels) if request is not None else [],
        score_breakdown=(
            dict(request.action_score_breakdown) if request is not None else {}
        ),
        deferred_axes=_deferred_axes(request),
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
    action_key = request.action_key if request is not None else None
    supporting = list(request.supporting_labels) if request is not None else []
    subject = _action_phrase(action_key, prescription)
    return OptimizationReport(
        report_id=_new_id(),
        request_id=_request_id(decision, request),
        status="proposed",
        summary=(
            f"{_support_phrase(supporting, request)} {subject}을(를) "
            "제안합니다(자동 적용하지 않음)."
        ),
        problem=_problem_text(request),
        selected_prescription=prescription,
        config_changes=_config_changes(None, request),
        expected_tradeoffs=_tradeoffs(request),
        manual_actions=_manual_actions(decision.manual_labels),
        next_steps=["제안을 검토하고 승인하면 적용합니다."],
        action_key=action_key,
        supporting_labels=supporting,
        opposing_labels=list(request.opposing_labels) if request is not None else [],
        score_breakdown=(
            dict(request.action_score_breakdown) if request is not None else {}
        ),
        deferred_axes=_deferred_axes(request),
    )


def _report_use_current(decision: OptimizeDecision) -> OptimizationReport:
    """변경 없이 현재 설정을 유지하는 경우(already_optimal / skipped)."""
    if decision.status == "already_optimal":
        summary = "이미 모든 목표 지표를 달성해 설정을 변경하지 않았습니다."
    elif decision.reason:
        summary = decision.reason
    else:
        summary = "적용할 수 있는 처방이 없어 현재 설정을 유지합니다."
    # 실행 가능한 action 이 하나도 없어서 아무것도 못 한 경우다. 그 사유를 실을 곳이
    # request 가 아니라 decision 뿐이므로(planner._selection_diagnostics) 여기서 옮긴다.
    # 없으면 사용자에게는 "처방 없음"만 남고 왜인지 알 수 없다.
    rejected = decision.metadata.get("rejected_actions") or []
    if rejected:
        summary += f" (검토했으나 실행할 수 없던 변경 {len(rejected)}건)"
    return OptimizationReport(
        report_id=_new_id(),
        request_id=_request_id(decision, None),
        status=decision.status,
        summary=summary,
        manual_actions=_manual_actions(decision.manual_labels),
        next_steps=["현재 설정으로 서빙을 진행합니다."],
        metadata={"rejected_actions": list(rejected)} if rejected else {},
        deferred_axes=list(decision.metadata.get("deferred_axes") or []),
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


def _problem_text(request: OptimizationRequest | None) -> str:
    """문제 설명. request.reason 우선, 없으면 지지 라벨 목록."""
    if request is None:
        return ""
    return request.reason or ", ".join(request.supporting_labels)


def _action_phrase(action_key: str | None, prescription: str | None) -> str:
    """사용자에게 보여줄 '무엇을 바꿨나' 한 조각.

    action key 는 `retriever.top_k:increase` 처럼 기계용이라, 처방 id 를 함께
    보여줘 rules.py 선언으로 되짚을 수 있게 한다. 구버전 이력처럼 action 이
    없으면 처방 id 만 쓴다.
    """
    if action_key and prescription:
        return f"'{action_key}'({prescription}) 변경"
    if action_key:
        return f"'{action_key}' 변경"
    return f"'{prescription}' 처방"


def _support_phrase(
    supporting: list[str],
    request: OptimizationRequest | None,
) -> str:
    """'무엇이 이 변경을 지지했나'. 대표 라벨 하나로 좁히지 않는다.

    여러 라벨이 같은 변경을 지지했다는 사실 자체가 선택 근거이므로, 그 수를 함께
    보여주면 사용자가 "왜 하필 이걸" 을 이해할 수 있다.
    """
    if not supporting:
        return "진단 결과에 따라"
    if len(supporting) == 1:
        return f"'{supporting[0]}' 문제에"
    return f"{len(supporting)}개 라벨({', '.join(supporting)})이 함께 지지한"


def _deferred_axes(request: OptimizationRequest | None) -> list[dict]:
    """근소한 차이로 이번 방문에서 보류한 축. 왜 안 골랐는지의 설명이다."""
    if request is None:
        return []
    return list(request.metadata.get("deferred_axes") or [])


def _selected_prescription(request: OptimizationRequest | None) -> str | None:
    """이번에 **실제로 선택된** action 의 출처 처방 id.

    전환 전에는 "후보 목록의 첫(가장 가벼운) 것"으로 추측했다. 지금은 요청이 선택된
    action 하나만 담으므로 추측할 자리가 없다 — 요청이 직접 들고 있는 값을 읽는다.
    """
    return request.prescription_id if request is not None else None


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
    # diff 가 없으면(제안만 등) 탐색 범위를 그대로 보여준다.
    if request is None or not request.search_space:
        return []
    return [
        f"{path}: {values[0] if len(values) == 1 else values}"
        for path, values in request.search_space.items()
    ]


def _config_changes_from_configs(
    before: dict, after: dict
) -> list[str]:
    """before/after config 두 dict를 비교해 사람이 읽는 변경 목록으로.
    이력 항목은 ConfigDiff 대신 before/after config를 통째로 들고 있어 여기서 비교한다.
    롤백된 경우 after_config == before_config 라 목록이 비고, 그게 정상이다."""
    lines: list[str] = []
    for key in after:
        if key not in before:
            lines.append(f"{key}: (신규) {after[key]}")
        elif before[key] != after[key]:
            lines.append(f"{key}: {before[key]} → {after[key]}")
    return lines


def _tradeoffs(request: OptimizationRequest | None) -> list[str]:
    """선택된 action 의 예상 부작용. catalog 정의가 단일 진실 원천이다."""
    if request is None or request.action is None:
        return []
    return list(request.action.definition.tradeoffs)


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


def _display_scores(verdict: Verdict) -> tuple[float, float]:
    """사용자 요약에 쓰는 표시용 점수 쌍(0~100).

    before_score/after_score 는 마진 판정용 탐색 신호(0~1)라 그대로 표시하면
    "0.7→0.8" 처럼 뭉개진다. 표시용 composite(0~100)를 우선 쓰고, composite
    미측정이면 report_view._to_100 과 같은 규약으로 탐색 신호×100 으로 폴백한다.
    """
    before = (
        verdict.before_composite
        if verdict.before_composite is not None
        else verdict.before_score * 100
    )
    after = (
        verdict.after_composite
        if verdict.after_composite is not None
        else verdict.after_score * 100
    )
    return before, after


def _score_metadata(verdict: Verdict | None) -> dict:
    """UI 표시용 점수/위반 정보."""
    if verdict is None:
        return {}
    return {
        "before_score": verdict.before_score,
        "after_score": verdict.after_score,
        # 표시용 종합점수(0~100). 하류 UI 가 0~1 탐색 신호를 스케일 오인하지
        # 않도록 표시 값은 여기서 함께 싣는다(없으면 None — 미측정).
        "before_composite": verdict.before_composite,
        "after_composite": verdict.after_composite,
        "floor_violations": verdict.floor_violations,
    }
