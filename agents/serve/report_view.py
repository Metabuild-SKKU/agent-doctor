"""
agents/serve/report_view.py
완료된 AgentDoctorState 를 web/prototype/report.html 이 기대하는 JSON 모양으로 변환한다.

I/O 없는 순수 함수 모음. report.html 의 렌더 함수(mHtml/rxCard/dxList/qaList 등)는
그대로 두고, 이 모듈이 만든 값만 그 자리에 꽂아 넣는 용도.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from core.state import AgentDoctorState

_EVAL_MODE_LABELS = {
    "fast": "빠른 검진",
    "standard": "표준 검진",
    "deep": "정밀 검진",
    "full": "정밀 검진",
}

_METRIC_LABELS = {
    "faithfulness": ("충실도", "답이 근거 문서에 충실한 정도. 낮으면 지어낼 위험이 있습니다."),
    "context_recall": ("정답 회수율", "필요한 정답 조각을 검색이 얼마나 가져왔는지입니다."),
    "context_precision": ("검색 정확도", "가져온 조각 중 실제로 정답에 쓸모 있는 비율입니다."),
    "response_relevancy": ("답변 관련성", "답이 질문에 얼마나 들어맞는지입니다."),
}


def build_report_view(state: AgentDoctorState, depth: Optional[str] = None) -> dict[str, Any]:
    report = state.report
    history = state.optimization_history or []

    findings = report.findings if report else []
    confirmed = [f for f in findings if f.confirmed]

    kept = sum(1 for h in history if h.status == "applied" and not h.metadata.get("pending"))
    rolled = sum(1 for h in history if h.status == "failed")
    pending = sum(1 for h in history if h.metadata.get("pending"))

    overall_after = report.overall_score if report and report.overall_score is not None else 0.0
    overall_before = _first_score(history, overall_after)

    depth_key = (depth or os.getenv("EVAL_MODE", "")).strip().lower()
    return {
        "meta": {
            "corpus": _corpus_label(state),
            "depth": _EVAL_MODE_LABELS.get(depth_key, "표준 검진"),
            "question_count": len(state.probes),
            "created_at": report.created_at.isoformat() if report else "",
        },
        "score": {
            "before": _to_100(overall_before),
            "after": _to_100(overall_after),
            "delta": round(_to_100(overall_after) - _to_100(overall_before), 1),
            "pass_threshold": bool(report and report.pass_threshold),
            "findings_count": len(findings),
            "kept": kept,
            "rolled": rolled,
            "pending": pending,
        },
        "priority": _build_priority(confirmed),
        "metrics": _build_metrics(report, history),
        "course": _build_course(history, overall_before),
        "rxs": _build_rxs(history),
        "dxs": _build_dxs(findings),
        "qas": _build_qas(state, findings),
        "transparency": {
            "duration_label": "",
            "question_count": len(state.probes),
            "rx_count": len(history),
            "rx_kept": kept,
            "rx_rolled": rolled,
            "chunk_count": len(state.chunks),
        },
    }


def _corpus_label(state: AgentDoctorState) -> str:
    if state.documents:
        title = state.documents[0].metadata.get("title")
        if title:
            return title
    return state.source_url or "업로드된 문서"


def _to_100(score_0_to_1: float) -> float:
    """Eval overall_score(0~1 스케일)를 리포트 표시용 100점 만점으로 변환한다."""
    return round(score_0_to_1 * 100, 1)


def _first_score(history: list, fallback: float) -> float:
    for item in history:
        before = item.metadata.get("before_score")
        if before is not None:
            return before
    return fallback


def _build_priority(confirmed_findings: list) -> list[dict[str, Any]]:
    ranked = sorted(
        confirmed_findings,
        key=lambda f: (0 if f.severity == "critical" else 1, -len(f.affected_probes)),
    )
    out = []
    for f in ranked[:3]:
        out.append({
            "group": f.metadata.get("group", ""),
            "severity": f.severity,
            "title": f.description.split("\n")[0][:60],
            "desc": f.description,
            "confirmed": f.confirmed,
            "affected": len(f.affected_probes),
        })
    return out


def _build_metrics(report, history: list) -> list[dict[str, Any]]:
    if report is None:
        return []
    after_scores = dict(report.ragas_scores or {})
    before_scores = dict(history[0].before_metrics) if history else after_scores
    if history:
        last_after = history[-1].after_metrics
        if last_after:
            after_scores = {**after_scores, **last_after}

    out = []
    for key, (name, tip) in _METRIC_LABELS.items():
        if key not in after_scores and key not in before_scores:
            continue
        before = before_scores.get(key, after_scores.get(key, 0.0))
        after = after_scores.get(key, before)
        out.append({
            "name": name,
            "en": key,
            "tip": tip,
            "before": round(float(before), 3),
            "after": round(float(after), 3),
        })
    return out


def _build_course(history: list, baseline_score: float) -> list[dict[str, Any]]:
    points = [{"label": "기준선", "score": _to_100(baseline_score), "kept": True}]
    for idx, item in enumerate(history, start=1):
        before = item.metadata.get("before_score", baseline_score)
        after = item.metadata.get("after_score", before)
        kept = item.status == "applied" and not item.metadata.get("pending")
        point = {
            "label": f"Rx{idx} · {item.selected_prescription_id or ''}",
            "score": _to_100(after if kept else before),
            "kept": kept,
        }
        if not kept:
            point["roll"] = _to_100(after)
        points.append(point)
    return points


def _changed_keys(before: dict, after: dict) -> list[str]:
    keys = set(before.keys()) | set(after.keys())
    return [k for k in keys if before.get(k) != after.get(k)]


def _build_rxs(history: list) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate(history, start=1):
        kept = item.status == "applied" and not item.metadata.get("pending")
        rolled_back = item.status == "failed"
        state_key = "kept" if kept else ("rolled" if rolled_back else "pending")

        changed = _changed_keys(item.before_config, item.after_config)
        if changed:
            key = changed[0]
            change = [key, str(item.before_config.get(key, "")), str(item.after_config.get(key, ""))]
        else:
            change = [item.selected_prescription_id or "설정 변경", "", ""]

        before_score = item.metadata.get("before_score", 0.0)
        after_score = item.metadata.get("after_score", before_score)
        direction = "up" if after_score >= before_score else "down"

        out.append({
            "state": state_key,
            "num": f"{idx:02d}",
            "change": change,
            "target": ", ".join(item.failure_labels),
            "reason": ["처방 근거", item.reason or ""],
            "score": [
                str(_to_100(before_score)),
                str(_to_100(after_score)),
                direction,
            ],
            "verdict": (
                ["keep", "유지"] if kept else
                ["roll", "롤백"] if rolled_back else
                ["pending", "판정 대기"]
            ),
            "drill": {
                "label": "판정 근거",
                "rows": [],
                "caption": item.rollback_reason or item.reason or "",
            },
        })
    return out


def _build_dxs(findings: list) -> list[dict[str, Any]]:
    out = []
    for f in findings:
        out.append({
            "grp": f.metadata.get("group", ""),
            "title": f.description.split("\n")[0][:60],
            "code": f.label or f.type,
            "badge": ["confirm", "확정"] if f.confirmed else ["prelim", "의심"],
            "desc": f.description,
            "foot": f"질문 {len(f.affected_probes)}건 영향",
            "rx": f.prescription or "미처방",
        })
    return out


def _build_qas(state: AgentDoctorState, findings: list) -> list[dict[str, Any]]:
    """근사치 구성: 실제 생성 답변 텍스트는 state 에 남지 않으므로, Probe/Finding 데이터를
    조합해 질문·기대정답·처방 전후 상태를 재구성한다(문자 그대로의 답변 비교가 아님).
    state.report 는 최신 Eval 방문 결과만 담으므로, 여기 남은 confirmed finding 은 아직
    미해결이다. optimization_history 에서 유지(kept)된 처방의 failure_labels 는 그 라벨의
    문제가 해결됐다고 보고 별도로 "해결됨" 카드를 만든다."""
    probes_by_id = {p.probe_id: p for p in state.probes}
    unresolved_labels = {f.label for f in findings if f.confirmed and f.label}

    out = []

    for item in state.optimization_history or []:
        kept = item.status == "applied" and not item.metadata.get("pending")
        if not kept:
            continue
        for label in item.failure_labels:
            if label in unresolved_labels:
                continue  # 나중에 다시 발견됨 → 미해결 쪽에서 다룬다
            out.append({
                "label": label,
                "solved": True,
                "q": "",
                "gold": "",
                "before": f"처방 전 진단 라벨: {label}",
                "bnote": "",
                "after": f"처방({item.selected_prescription_id or ''}) 적용 후 재검증 통과",
                "fix": item.selected_prescription_id or "",
            })

    for f in findings:
        if not f.confirmed:
            continue
        for probe_id in f.affected_probes:
            probe = probes_by_id.get(probe_id)
            if probe is None:
                continue
            out.append({
                "label": f"{f.metadata.get('group', '')} · {f.label or f.type}",
                "solved": False,
                "q": probe.question,
                "gold": probe.ground_truth or "",
                "before": f.description,
                "bnote": "",
                "after": "처방 후에도 재현됨 — 여전히 미해결" if f.prescription else "아직 처방되지 않음",
                "fix": f.prescription or "미처방",
            })

    return out[:6]
