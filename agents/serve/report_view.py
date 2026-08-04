"""
agents/serve/report_view.py
완료된 AgentDoctorState 를 web/prototype/report.html 이 기대하는 JSON 모양으로 변환한다.

I/O 없는 순수 함수 모음. report.html 의 렌더 함수(mHtml/rxCard/dxList/qaList 등)는
그대로 두고, 이 모듈이 만든 값만 그 자리에 꽂아 넣는 용도.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from agents.optimize import gate
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

# ── action 을 사람 말로 ────────────────────────────────────────────
# 선택 단위가 action 이 된 뒤로, 차트 점 이름도 "어떤 처방을 골랐나"가 아니라
# "무엇을 바꿨나"를 보여준다. 축 이름 + 동작 동사로 조립하므로 catalog 에 새 action 이
# 생겨도 표를 손보지 않아도 된다(축만 등록하면 된다).
_AXIS_NOUNS = {
    "retriever.top_k": "top_k",
    "retriever.mmr": "MMR",
    "retriever.hybrid_dense_weight": "하이브리드 가중치",
    "retriever.search_type": "하이브리드 검색",
    "reranker.enabled": "리랭커",
    "reranker.candidate_count": "리랭커 후보군",
    "context.compression.enabled": "컨텍스트 압축",
    "chunker.chunk_size": "청크 크기",
    "chunker.chunk_overlap": "청크 겹침",
    "chunker.strategy": "청킹 전략",
    "embedding.model": "임베딩 모델",
    "generation.temperature": "생성 온도",
    "generation.grounding_strict": "근거 지시",
    "generation.require_citation": "인용 표시",
    "generation.restate_question": "질문 재진술",
    "generation.completeness_mode": "완전성 모드",
    "generation.abstention_strict": "기권 기준",
    "generation.abstention_relaxed": "기권 완화",
    "generation.model": "생성 모델",
}
_OPERATION_VERBS = {
    "increase": "확대",
    "decrease": "축소",
    "enable": "활성화",
    "disable": "비활성화",
    "replace": "교체",
    "adjust": "조정",
}

# ⚠️ 구버전 이력 fallback. 이전 실행이 저장한 항목에는 action_key 가 없어
# 처방 id 만 남아 있다. 그 리포트도 계속 읽혀야 하므로 표를 유지한다.
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
    errors = sum(1 for h in history if _is_study_error(h))
    rolled = sum(
        1 for h in history
        if h.status == "failed" and not _is_study_error(h)
    )
    pending = sum(1 for h in history if h.metadata.get("pending"))

    # 헤드라인 '종합 점수' = 설계 종합점수(composite, 0~100). 없으면 overall×100 폴백.
    headline_after = _headline_score(report)
    headline_before = _first_headline(history, headline_after)
    gate_summary = _gate_summary(report)

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
            "pass_threshold": gate_summary["pass"],
            "gate_pass": gate_summary["pass"],
            "eval_pass_threshold": gate_summary["eval_pass_threshold"],
            "gate": gate_summary,
            "findings_count": len(findings),
            "kept": kept,
            "rolled": rolled,
            "errors": errors,
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
            "rx_errors": errors,
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


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_recall_at_k(report) -> Optional[float]:
    scores = getattr(report, "ragas_scores", None) or {}
    return _safe_float(scores.get("mean_recall_at_k"))


def _gate_summary(report) -> dict[str, Any]:
    if report is None:
        return {
            "pass": False,
            "reason": "report_missing",
            "eval_pass_threshold": False,
            "score_pass": False,
            "score_source": "missing",
            "composite_total": None,
            "composite_threshold": gate.COMPOSITE_PASS_THRESHOLD,
            "mean_recall_at_k": None,
            "recall_floor": gate.RECALL_FLOOR,
            "recall_pass": None,
        }

    composite_total = _safe_float((getattr(report, "composite_score", None) or {}).get("total"))
    eval_pass = bool(getattr(report, "pass_threshold", False))
    if composite_total is None:
        score_pass = eval_pass
        score_source = "report.pass_threshold"
    else:
        score_pass = composite_total >= gate.COMPOSITE_PASS_THRESHOLD
        score_source = "composite_score.total"

    recall = _mean_recall_at_k(report)
    recall_pass = None if recall is None else recall >= gate.RECALL_FLOOR
    gate_pass = gate.passes_report(report)

    if gate_pass:
        reason = "passed"
    elif not score_pass:
        reason = (
            "composite_below_threshold"
            if composite_total is not None
            else "eval_pass_threshold_false"
        )
    elif recall_pass is False:
        reason = "recall_below_floor"
    else:
        reason = "gate_failed"

    return {
        "pass": gate_pass,
        "reason": reason,
        "eval_pass_threshold": eval_pass,
        "score_pass": score_pass,
        "score_source": score_source,
        "composite_total": round(composite_total, 1) if composite_total is not None else None,
        "composite_threshold": gate.COMPOSITE_PASS_THRESHOLD,
        "mean_recall_at_k": round(recall, 4) if recall is not None else None,
        "recall_floor": gate.RECALL_FLOOR,
        "recall_pass": recall_pass,
    }


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


def _is_study_error(item) -> bool:
    """후보 평가가 끝나기 전에 study 자체가 실패한 이력인지 구분한다."""
    return "study_error" in item.metadata


def _action_point_label(action_key: str) -> str:
    """`retriever.top_k:increase` → `top_k 확대`. 모르는 축이면 None 대신 원문 축약."""
    path, _, operation = action_key.rpartition(":")
    noun = _AXIS_NOUNS.get(path)
    verb = _OPERATION_VERBS.get(operation)
    if noun and verb:
        return f"{noun} {verb}"
    return action_key[:18]


def _course_point_label(item, prescription_index: int) -> str:
    """차트 폭에 맞는 짧은 변경명. Rx 순번 대신 실제로 무엇을 바꿨는지 보여준다.

    prescription_index는 차트 점 번호가 아니라 optimization history의 처방 순번이다.
    실패한 처방이 실패·복구 두 점을 만들더라도 다음 처방의 순번은 하나만 증가한다.

    action_key 를 먼저 본다. 구버전 이력에는 없으므로 처방 id 로 폴백한다.
    """
    action_key = getattr(item, "action_key", None)
    if action_key:
        return _action_point_label(action_key)
    prescription_id = item.selected_prescription_id or ""
    if prescription_id in _PRESCRIPTION_POINT_LABELS:
        return _PRESCRIPTION_POINT_LABELS[prescription_id]
    if prescription_id:
        return prescription_id.replace("_", " ")[:18]
    return f"처방 {prescription_index}"


def _build_course(history: list, baseline_score: float) -> list[dict[str, Any]]:
    # baseline_score 는 이미 헤드라인 스케일(0~100) — 여기서 다시 _to_100 하지 않는다.
    points = [{
        "label": "기준선",
        "score": baseline_score,
        "kept": True,
        "kind": "baseline",
    }]
    for prescription_index, item in enumerate(history, start=1):
        # study 오류는 전후 점수가 관측되지 않았다. baseline 폴백으로 가짜 하락·복구
        # 점을 만들지 말고, 오류 상세는 처방 이력 카드에만 남긴다.
        if _is_study_error(item):
            continue

        kept = item.status == "applied" and not item.metadata.get("pending")
        rolled_back = item.status == "failed"
        before = _course_point_score(item, "before", baseline_score)
        after = _course_point_score(item, "after", before)
        # 같은 진단 라벨에 여러 처방을 시도해도 Rx 순번만으로 뭉뚱그리지 않고,
        # 차트에서는 실제 처방 이름을 점 이름으로 쓴다.
        label = _course_point_label(item, prescription_index)

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
        study_error = _is_study_error(item)
        kept = item.status == "applied" and not item.metadata.get("pending")
        rolled_back = item.status == "failed" and not study_error
        state_key = (
            "error" if study_error else
            ("kept" if kept else ("rolled" if rolled_back else "pending"))
        )

        changed = _changed_keys(item.before_config, item.after_config)
        if changed:
            key = changed[0]
            change = [key, str(item.before_config.get(key, "")), str(item.after_config.get(key, ""))]
        else:
            # 변경 흔적이 없으면(롤백 등) 무엇을 시도했는지라도 보여준다.
            change = [
                _course_point_label(item, idx) or "설정 변경",
                "",
                "",
            ]

        # 헤드라인(composite)과 일관되게 처방 카드 점수도 종합점수로 표시.
        # (유지/롤백 판정 자체는 overall 탐색 신호 기준이므로 direction 과 verdict 이
        #  드물게 어긋날 수 있으나, 표시 점수는 사용자가 보는 종합점수로 통일한다.)
        score = None
        if not study_error:
            before_head = _course_point_score(item, "before", 0.0)
            after_head = _course_point_score(item, "after", before_head)
            direction = "up" if after_head >= before_head else "down"
            score = [str(before_head), str(after_head), direction]

        # 이 변경을 지지한 라벨 전체를 보여준다. 대표 하나로 좁히면 "왜 이걸
        # 골랐나"의 핵심(여러 문제가 같은 변경을 원했다)이 사라진다.
        # 구버전 이력에는 supporting_labels 가 없어 failure_labels 로 폴백한다.
        supporting = list(getattr(item, "supporting_labels", None) or item.failure_labels)
        resolved, remaining = _attribution(item, kept, study_error, supporting)

        out.append({
            "state": state_key,
            "num": f"{idx:02d}",
            "change": change,
            "action": getattr(item, "action_key", None) or "",
            "target": ", ".join(supporting),
            "resolved": resolved,
            "remaining": remaining,
            "reason": ["처방 근거", item.reason or ""],
            "score": score,
            "verdict": (
                ["error", "실험 오류 · 설정 원복"] if study_error else
                ["keep", "유지"] if kept else
                ["roll", "롤백"] if rolled_back else
                ["pending", "판정 대기"]
            ),
            "drill": {
                "label": "오류 원인" if study_error else "판정 근거",
                # rows 는 sweep 후보 실측 막대그래프 전용이다(report.html rxCard).
                # 선택 근거는 모양이 달라 notes 로 따로 싣는다 — 섞으면 렌더가 깨진다.
                "rows": [],
                "notes": _selection_notes(item, supporting, resolved, remaining),
                "caption": (
                    str(item.metadata.get("study_error", ""))
                    if study_error else
                    (item.rollback_reason or item.reason or "")
                ),
            },
        })
    return out


def _attribution(
    item,
    kept: bool,
    study_error: bool,
    supporting: list[str],
) -> tuple[list[str], list[str]]:
    """이 시도가 실제로 해결한 라벨과 남은 라벨.

    ⚠️ **유지된 처방만 해결을 주장할 수 있다.** 롤백은 config 를 되돌렸으므로 그
    개선이 지금 설정에 남아 있지 않다. `history.finalize_item` 이 저장 시점에 이미
    같은 판정을 적용하지만, 여기서 한 번 더 막는다 — 이 함수는 `OptimizationReport`
    가 아니라 raw 이력 metadata 를 직접 읽으므로 reporter 의 가드가 걸리지 않는
    경로다(웹 리포트만 "되돌렸는데 해결됨"으로 보이던 원인).

    study 오류는 해결도 잔존도 주장하지 않는다 — 전후 측정 자체가 성립하지 않았다.
    """
    if study_error:
        return [], []
    if not kept:
        return [], list(supporting)
    return (
        list(item.metadata.get("resolved_labels") or []),
        list(item.metadata.get("remaining_labels") or []),
    )


def _selection_notes(
    item,
    supporting: list[str],
    resolved: list[str],
    remaining: list[str],
) -> list[list[str]]:
    """선택 근거 drill-down: 무엇이 지지했고 무엇이 실제로 해결됐는지.

    "지지받았다"와 "해결됐다"를 나란히 보여주는 것이 핵심이다. 지지 라벨을 그대로
    성과로 읽으면 리포트가 실제보다 좋게 보인다. resolved/remaining 은 호출부가
    _attribution 으로 이미 판정한 값을 받는다(카드 본문과 어긋나지 않도록).

    구버전 이력에는 이 정보가 없어 빈 목록이 되고, 그때는 카드가 caption 만
    보여주던 이전 모습 그대로다.
    """
    notes: list[list[str]] = []
    if len(supporting) > 1:
        notes.append(["지지 라벨", f"{len(supporting)}개 · {', '.join(supporting)}"])
    probes = list(getattr(item, "supporting_probes", None) or [])
    if probes:
        notes.append(["영향 질문", f"{len(probes)}건"])
    if resolved:
        notes.append(["해결됨", ", ".join(resolved)])
    if remaining:
        notes.append(["남음", ", ".join(remaining)])
    return notes


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
                # Eval이 missing_gold_ids를 실었으면 '전체 gold 중 몇 개가 없는지'를 정밀 표기.
                missing = f.metadata.get("missing_gold_ids") if f.metadata else None
                total = len(getattr(probe, "gold_chunk_ids", None) or [])
                if missing and total:
                    where += f" (gold {total}개 중 {len(missing)}개 누락)"
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
    """실패한 검증 질문을 실제 Eval 답변과 함께 UI 데이터로 변환한다.

    한 probe에 Finding이 여러 개여도 질문 카드는 하나만 만들고 진단 설명과 처방을
    합친다. 예비 Finding도 평가상 실패한 질문이므로 숨기지 않는다. 이 섹션은 최신
    Eval의 실제 실패 질문만 다루므로, 질문·답변 근거가 없는 과거 처방 이력에서
    "해결됨" 카드를 추정하지 않는다. 실패 질문은 DiagnosticReport에 보존된 만큼
    모두 전달하며 표시 상한을 임의로 두지 않는다.
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
        reasons = list(dict.fromkeys(
            str(finding.metadata.get("reason") or "").strip() or finding.description
            for finding in related
        ))
        prescriptions = list(dict.fromkeys(
            finding.prescription or "미처방" for finding in related
        ))
        out.append({
            "label": " / ".join(labels) or "검증 실패",
            "q": result.get("question", ""),
            "gold": result.get("expected_answer", ""),
            "actual": result.get("actual_answer", ""),
            "diagnosis": " / ".join(reasons) or "실패 원인을 확인하지 못했습니다.",
            "fix": " / ".join(prescriptions) or "미처방",
        })

    return out
