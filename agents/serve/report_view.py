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

    # 헤드라인 '종합 점수' = 설계 종합점수(composite, 0~100). 없으면 overall×100 폴백.
    headline_after = _headline_score(report)
    headline_before = _first_headline(history, headline_after)

    depth_key = (depth or os.getenv("EVAL_MODE", "")).strip().lower()
    return {
        "meta": {
            "corpus": _corpus_label(state),
            "depth": _EVAL_MODE_LABELS.get(depth_key, "표준 검진"),
            "question_count": len(state.probes),
            "created_at": report.created_at.isoformat() if report else "",
        },
        "score": {
            "before": headline_before,
            "after": headline_after,
            "delta": round(headline_after - headline_before, 1),
            "pass_threshold": bool(report and report.pass_threshold),
            "findings_count": len(findings),
            "kept": kept,
            "rolled": rolled,
            "pending": pending,
        },
        "priority": _build_priority(confirmed),
        "metrics": _build_metrics(report, history),
        "course": _build_course(history, headline_before),
        "rxs": _build_rxs(history),
        "dxs": _build_dxs(findings),
        "qas": _build_qas(state, findings),
        "recommendations": _build_recommendations(state, findings),
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


def _composite_total(report) -> Optional[float]:
    """설계 종합점수(composite_score.total, 이미 0~100). 없으면 None."""
    if report is None:
        return None
    total = (report.composite_score or {}).get("total")
    return round(float(total), 1) if total is not None else None


def _headline_score(report) -> float:
    """리포트 헤드라인 '종합 점수'(0~100). 설계 종합점수(composite=품질×신뢰도)를
    우선 쓰고, 아직 없으면(측정 불가·구버전 리포트) overall_score×100 로 폴백한다.
    overall 은 품질 단일축이라 통과율이 낮아도 높게 나와 시스템 실제 성능을 과대평가하므로,
    사용자에게 보여주는 헤드라인은 composite 를 정본으로 삼는다."""
    total = _composite_total(report)
    if total is not None:
        return total
    overall = report.overall_score if report and report.overall_score is not None else 0.0
    return _to_100(overall)


def _first_headline(history: list, fallback: float) -> float:
    """최적화 이전(baseline) 헤드라인 종합점수. history 에 저장된 baseline composite 를
    우선 쓰고, 없으면 구버전 overall 기반 before_score 로 폴백한다."""
    for item in history:
        bc = item.metadata.get("before_composite")
        if bc is not None:
            return round(float(bc), 1)
        bs = item.metadata.get("before_score")
        if bs is not None:
            return _to_100(bs)
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


def _course_point_score(item, key: str, fallback: float) -> float:
    """치료경과 한 점의 헤드라인 점수(0~100). 설계 종합점수(composite, 이미 0~100)를
    우선 쓰고, 없으면 구버전 overall(0~1)×100 로 폴백. key 는 'before'|'after'."""
    comp = item.metadata.get(f"{key}_composite")
    if comp is not None:
        return round(float(comp), 1)
    raw = item.metadata.get(f"{key}_score")
    return _to_100(raw) if raw is not None else fallback


def _build_course(history: list, baseline_score: float) -> list[dict[str, Any]]:
    # baseline_score 는 이미 헤드라인 스케일(0~100) — 여기서 다시 _to_100 하지 않는다.
    points = [{"label": "기준선", "score": baseline_score, "kept": True}]
    for idx, item in enumerate(history, start=1):
        kept = item.status == "applied" and not item.metadata.get("pending")
        before = _course_point_score(item, "before", baseline_score)
        after = _course_point_score(item, "after", before)
        point = {
            "label": f"Rx{idx} · {item.selected_prescription_id or ''}",
            "score": after if kept else before,
            "kept": kept,
        }
        if not kept:
            point["roll"] = after
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

        # 헤드라인(composite)과 일관되게 처방 카드 점수도 종합점수로 표시.
        # (유지/롤백 판정 자체는 overall 탐색 신호 기준이므로 direction 과 verdict 이
        #  드물게 어긋날 수 있으나, 표시 점수는 사용자가 보는 종합점수로 통일한다.)
        before_head = _course_point_score(item, "before", 0.0)
        after_head = _course_point_score(item, "after", before_head)
        direction = "up" if after_head >= before_head else "down"

        out.append({
            "state": state_key,
            "num": f"{idx:02d}",
            "change": change,
            "target": ", ".join(item.failure_labels),
            "reason": ["처방 근거", item.reason or ""],
            "score": [
                str(before_head),
                str(after_head),
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


# ── 남은 권고 (자동 처방으로 끝나지 않는 항목) ──────────────────────
# 두 부류를 모은다:
#   - manual(D그룹): config로 못 고침 → 사람이 문서 보강/probe 재생성. rules.py 의
#     매뉴얼 스텝과 "어디가 문제인지"(질문·근거 문서)를 함께 실어 구체화한다.
#   - preliminary(의심): 표준 검진으론 원인 미확정 → 정밀 검진 필요.
# 확정 자동처방 대상(dxs/rxs에서 다룸)은 여기서 제외한다.

_REC_TITLES = {
    "corpus_gap": "코퍼스에 근거가 없는 질문 {n}건",
    "corpus_gap_partial_hop": "일부 단계 근거가 없는 질문 {n}건",
    "bad_gold_answer": "정답셋이 의심되는 질문 {n}건",
}
_REC_CTAS = {
    "corpus_gap": "문서 보강 필요",
    "corpus_gap_partial_hop": "문서 보강 필요",
    "bad_gold_answer": "probe 재생성 필요",
}
# manual finding.metadata["group"] → 배지. D 이외(예비가 A/B/C일 수 있음)는 prelim에서 처리.
_REC_GROUP_BADGES = {"D": ["data", "D · 데이터"]}


def _rec_source_docs(probe) -> list[str]:
    """probe 가 근거로 삼는 원본 문서들(재청킹에도 안 깨지는 gold_doc_id / gold_spans.doc_id).
    누락 gold 를 콕 집으려면 finding.metadata['missing_gold_ids'] 가 필요하나(Eval 계약, PR #55),
    아직 없으므로 probe 의 gold 문서 전체를 degradation 으로 보여준다."""
    docs: list[str] = []
    if getattr(probe, "gold_doc_id", None):
        docs.append(probe.gold_doc_id)
    for span in getattr(probe, "gold_spans", None) or []:
        doc = span.get("doc_id") if isinstance(span, dict) else None
        if doc and doc not in docs:
            docs.append(doc)
    return docs


def _rec_manual_steps(rule: dict) -> list[dict[str, str]]:
    """rules.py 의 매뉴얼 처방 스텝(manual=True)을 렌더용으로."""
    steps = []
    for p in rule.get("prescriptions", []):
        if not p.get("manual"):
            continue
        steps.append({"action": p.get("action", ""), "detail": p.get("detail", "")})
    return steps


# bad_gold_answer 소스별 조치 — 사용자 제공 정답은 자동 재생성 대상이 아니라 사람 검수,
# 우리가 만든 probe 는 재생성 대상. 분기 로직은 rules.py(선언)가 아니라 여기(resolver)에 둔다.
# 실제 제외·재평가 실행은 Eval 재평가 루프 몫이므로, 여기서는 소스에 맞는 안내만 한다.
_BAD_GOLD_BY_SOURCE = {
    "user_log": "사용자 제공 정답 — 자동 재생성 대상 아님. 검수·수정 요청",
}
_BAD_GOLD_DEFAULT_ACTION = "자동 생성 probe — 재생성 후 재평가 대상"


def _rec_items(label: str, findings: list, probes_by_id: dict) -> list[dict[str, str]]:
    """이 권고가 걸린 질문들을 '어디가 문제인지'와 함께 per-probe 로.
    corpus_gap 계열은 근거 문서를, bad_gold_answer 는 기대 정답 + 소스별 조치를 보여준다."""
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for f in findings:
        for pid in f.affected_probes:
            if pid in seen:
                continue
            seen.add(pid)
            probe = probes_by_id.get(pid)
            if probe is None:
                continue
            if label == "bad_gold_answer":
                action = _BAD_GOLD_BY_SOURCE.get(getattr(probe, "source", ""), _BAD_GOLD_DEFAULT_ACTION)
                items.append({
                    "q": probe.question,
                    "where": action,          # 소스별 조치를 위치 슬롯에 실어 렌더 재사용
                    "gold": probe.ground_truth or "",
                })
            else:
                docs = _rec_source_docs(probe)
                where = ("근거 문서: " + ", ".join(docs)) if docs else "근거 문서 미상"
                items.append({"q": probe.question, "where": where, "gold": ""})
    return items


def _build_recommendations(state: AgentDoctorState, findings: list) -> list[dict[str, Any]]:
    from agents.optimize.rules import get_rule, is_manual

    probes_by_id = {p.probe_id: p for p in state.probes}
    groups: dict[str, list] = {}
    order: list[str] = []
    for f in findings:
        label = f.label
        if not label:
            continue
        # manual(D) 또는 예비(의심)만. 확정 자동처방 대상은 dxs/rxs 몫.
        if not (is_manual(label) or not f.confirmed):
            continue
        if label not in groups:
            groups[label] = []
            order.append(label)
        groups[label].append(f)

    out: list[dict[str, Any]] = []
    for label in order:
        fs = groups[label]
        rule = get_rule(label) or {}
        n = len({pid for f in fs for pid in f.affected_probes})
        if is_manual(label):
            title = _REC_TITLES.get(label, "사람 조치가 필요한 질문 {n}건").format(n=n)
            out.append({
                "kind": "manual",
                "code": label,
                "badge": _REC_GROUP_BADGES.get(rule.get("group"), ["data", "D · 데이터"]),
                "title": title,
                "desc": rule.get("manual_action", ""),
                "cta": _REC_CTAS.get(label, "사람 조치 필요"),
                "steps": _rec_manual_steps(rule),
                "items": _rec_items(label, fs, probes_by_id),
            })
        else:
            # 예비(의심): 원인 미확정 → 정밀 검진 안내. 제목은 진단 설명 첫 줄에서 군더더기 제거.
            title = fs[0].description.split("\n")[0]
            for token in ("[예비] ", "[A그룹] ", "[B그룹] ", "[C그룹] ", "[D그룹] "):
                title = title.replace(token, "")
            out.append({
                "kind": "prelim",
                "code": label,
                "badge": ["prelim", "의심"],
                "title": f"{title[:50]} · 질문 {n}건",
                "desc": "표준 검진으로는 원인을 확정할 수 없어 정밀 검진이 필요합니다.",
                "cta": "정밀 검진 실행 →",
                "steps": [],
                "items": _rec_items(label, fs, probes_by_id),
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
