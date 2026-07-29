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

_PRESCRIPTION_POINT_LABELS = {
    "enable_reranker": "리랭커 활성화",
    "widen_rerank_candidates": "후보군 확대",
    "enable_hybrid": "하이브리드 검색",
    "swap_embedding_model": "임베딩 교체",
    "shrink_chunk_size": "청크 축소",
    "decrease_chunk_size": "청크 축소",
    "increase_chunk_size": "청크 확대",
    "increase_chunk_overlap": "겹침 확대",
    "switch_chunking_strategy": "청킹 전략 변경",
    "increase_top_k": "top_k 확대",
    "decrease_top_k": "top_k 축소",
    "dynamic_top_k": "동적 top_k",
    "expand_query": "질의 확장",
    "enable_query_decomposition": "질의 분해",
    "expand_bridge_entity_query": "연결어 확장",
    "enable_mmr": "MMR 활성화",
    "enable_adaptive_retrieval": "적응형 검색",
    "relax_reranker_threshold": "리랭커 완화",
    "tighten_reranker_threshold": "리랭커 강화",
    "swap_reranker_model": "리랭커 교체",
    "lower_temperature": "생성 온도 낮춤",
    "strict_grounding_prompt": "근거 지시 강화",
    "upgrade_generation_model": "생성 모델 교체",
    "completeness_prompt": "완전성 지시",
    "checklist_review_step": "체크리스트 검증",
    "llm_verification_pass": "LLM 재검증",
    "restate_question": "질문 재진술",
    "strengthen_abstention_prompt": "기권 기준 강화",
    "require_citation": "인용 필수화",
    "require_numeric_citation": "수치 인용 강화",
    "enable_calculation_check": "계산 검증",
    "force_hop_evidence_binding": "근거 연결 강화",
    "enable_bridge_entity_verifier": "연결어 검증",
    "context_compression": "컨텍스트 압축",
    "reorder_context_edges": "컨텍스트 재정렬",
    "enable_noise_filter": "노이즈 필터",
    "strict_conflict_prompt": "충돌 검증",
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


def _course_point_label(item, index: int) -> str:
    """차트 폭에 맞는 짧은 처방명. Rx 순번 대신 실제 처방의 의미를 보여준다."""
    prescription_id = item.selected_prescription_id or ""
    if prescription_id in _PRESCRIPTION_POINT_LABELS:
        return _PRESCRIPTION_POINT_LABELS[prescription_id]
    if prescription_id:
        return prescription_id.replace("_", " ")[:18]
    return f"처방 {index}"


def _build_course(history: list, baseline_score: float) -> list[dict[str, Any]]:
    # baseline_score 는 이미 헤드라인 스케일(0~100) — 여기서 다시 _to_100 하지 않는다.
    points = [{
        "label": "기준선",
        "score": baseline_score,
        "kept": True,
        "kind": "baseline",
    }]
    for idx, item in enumerate(history, start=1):
        kept = item.status == "applied" and not item.metadata.get("pending")
        rolled_back = item.status == "failed"
        before = _course_point_score(item, "before", baseline_score)
        after = _course_point_score(item, "after", before)
        # 같은 진단 라벨에 여러 처방을 시도해도 Rx 순번만으로 뭉뚱그리지 않고,
        # 차트에서는 실제 처방 이름을 점 이름으로 쓴다.
        label = _course_point_label(item, idx)

        if rolled_back:
            # 실패한 처방은 롤백 전 실측 점수와 복원된 점수를 각각 한 점으로 남긴다.
            # 이전에는 after를 세로 보조선에만 넣어 본선에서 점수 변화가 보이지 않았다.
            points.extend([
                {
                    "label": f"{label} 실패",
                    "score": after,
                    "kept": False,
                    "kind": "failed",
                },
                {
                    "label": "원상 복구",
                    "score": before,
                    "kept": True,
                    "kind": "rollback",
                },
            ])
            continue

        points.append({
            "label": label,
            "score": after if kept else before,
            "kept": kept,
            "kind": "kept" if kept else "pending",
        })
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


def _build_qas(state: AgentDoctorState, findings: list) -> list[dict[str, Any]]:
    """실패한 검증 질문을 실제 Eval 답변과 함께 UI 데이터로 변환한다.

    한 probe에 Finding이 여러 개여도 질문 카드는 하나만 만들고 진단 설명과 처방을
    합친다. 예비 Finding도 평가상 실패한 질문이므로 숨기지 않는다.
    """
    report = state.report
    failed_questions = getattr(report, "failed_questions", []) if report else []
    findings_by_probe: dict[str, list] = {}
    for finding in findings:
        for probe_id in finding.affected_probes:
            findings_by_probe.setdefault(probe_id, []).append(finding)

    out = []
    for result in failed_questions:
        probe_id = result.get("probe_id", "")
        related = findings_by_probe.get(probe_id, [])
        labels = [
            f"{finding.metadata.get('group', '')} · {finding.label or finding.type}".strip(" ·")
            for finding in related
        ]
        descriptions = list(dict.fromkeys(finding.description for finding in related))
        prescriptions = list(dict.fromkeys(
            finding.prescription or "미처방" for finding in related
        ))
        out.append({
            "label": " / ".join(labels) or "검증 실패",
            "q": result.get("question", ""),
            "gold": result.get("expected_answer", ""),
            "actual": result.get("actual_answer", ""),
            "diagnosis": " ".join(descriptions) or "실패 원인을 확인하지 못했습니다.",
            "fix": " / ".join(prescriptions) or "미처방",
        })

    return out
