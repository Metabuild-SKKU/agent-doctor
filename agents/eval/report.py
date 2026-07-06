"""
agents/eval/report.py
STEP5: 진단 리포트 생성

설계 문서 'STEP 5: 진단 리포트' 구현.
모든 Probe 판정 결과(EvalRecord)를 집계해 DiagnosticReport 를 만든다.
    - overall_score : RAGAS 가중 평균(있으면) / 없으면 규칙 지표 폴백
    - pass_threshold: overall_score >= PASS_SCORE_THRESHOLD
    - ragas_scores  : RAGAS 평균 + 규칙 지표 평균 + 브랜치 분포(관측용)
    - findings      : 전 record 의 Finding 합침

graph.route_after_eval() 이 report.pass_threshold 로 Serve/Optimize 분기를 결정한다.
"""
from __future__ import annotations

import uuid
from collections import Counter

from core.schema import DiagnosticReport
from agents.eval.types import (
    EvalRecord, Branch, RAGAS_WEIGHTS, PASS_SCORE_THRESHOLD, F1_PASS_THRESHOLD,
)

_RAGAS_KEYS = ("faithfulness", "context_precision", "context_recall", "response_relevancy")


def build_report(records: list[EvalRecord], iteration: int) -> DiagnosticReport:
    """records 를 집계해 DiagnosticReport 반환."""
    findings = [f for r in records for f in r.findings]

    ragas_means = _ragas_means(records)
    rule_means = _rule_means(records)
    overall = _overall_score(ragas_means, rule_means)
    oracle_acc = _oracle_accuracy(records)

    scores = {**rule_means}
    scores.update(ragas_means)                          # RAGAS 평균(있으면)
    scores["branch_distribution"] = dict(Counter(r.branch for r in records))

    # 평가 신호(GT 규칙지표/RAGAS)가 전혀 없으면 진단 불가 →
    # eval 한계로 파이프라인을 막지 않도록 통과 처리(overall_score=None).
    if overall is None:
        overall_val, pass_thr = None, True
        print("[Eval] 경고: 평가 신호 없음(GT·RAGAS 부재) → 통과 처리")
    else:
        overall_val, pass_thr = round(overall, 4), overall >= PASS_SCORE_THRESHOLD

    report = DiagnosticReport(
        report_id=f"report_{uuid.uuid4().hex[:8]}",
        findings=findings,
        ragas_scores=scores,
        oracle_accuracy=oracle_acc,
        overall_score=overall_val,
        pass_threshold=pass_thr,
        iteration=iteration,
    )

    _print_summary(records, report)
    return report


# ── 집계 헬퍼 ─────────────────────────────────────────────────────

def _ragas_means(records: list[EvalRecord]) -> dict:
    """실제 트랙 RAGAS 지표별 평균 (측정된 것만)."""
    means = {}
    for key in _RAGAS_KEYS:
        vals = [r.ragas[key] for r in records
                if r.ragas.get(key) is not None]
        if vals:
            means[key] = sum(vals) / len(vals)
    return means


def _rule_means(records: list[EvalRecord]) -> dict:
    """
    규칙 지표 평균 (관측·폴백 점수용).
      - recall : gold_chunk_ids 있는 record 만 (-1 제외)
      - f1/oracle : ground_truth 있는 record 만 (정답 없으면 무의미)
    """
    recalls = [max(0.0, r.recall_at_k) for r in records if r.recall_at_k >= 0]
    gt = [r for r in records if r.probe.ground_truth]
    f1s = [r.f1_score for r in gt]
    oracles = [r.oracle_f1 for r in gt]
    out = {}
    if recalls:
        out["mean_recall_at_k"] = sum(recalls) / len(recalls)
    if f1s:
        out["mean_f1"] = sum(f1s) / len(f1s)
    if oracles:
        out["mean_oracle_f1"] = sum(oracles) / len(oracles)
    return out


def _overall_score(ragas_means: dict, rule_means: dict) -> float | None:
    """
    RAGAS 4지표가 하나라도 있으면 설계 §7 가중 평균(있는 것만 재정규화).
    없으면 규칙 지표 폴백: recall 과 F1 의 평균.
    평가 신호가 전혀 없으면 None(진단 불가).
    """
    present = {k: v for k, v in ragas_means.items() if k in RAGAS_WEIGHTS}
    if present:
        wsum = sum(RAGAS_WEIGHTS[k] for k in present)
        return sum(present[k] * RAGAS_WEIGHTS[k] for k in present) / wsum

    # 폴백: 규칙 지표
    recall = rule_means.get("mean_recall_at_k")
    f1 = rule_means.get("mean_f1")
    parts = [v for v in (recall, f1) if v is not None]
    return sum(parts) / len(parts) if parts else None


def _oracle_accuracy(records: list[EvalRecord]) -> float | None:
    """Oracle 트랙 통과율 (ground_truth 보유 record 중 oracle_f1 >= 임계값 비율)."""
    gt = [r for r in records if r.probe.ground_truth]
    if not gt:
        return None
    passed = sum(1 for r in gt if r.oracle_f1 >= F1_PASS_THRESHOLD)
    return passed / len(gt)


# ── 로그 요약 ─────────────────────────────────────────────────────

def _print_summary(records: list[EvalRecord], report: DiagnosticReport) -> None:
    n = len(records)
    fail = sum(1 for r in records if r.branch not in (Branch.SUCCESS, Branch.NO_ANSWER_OK))
    print(f"[Eval] STEP5: 리포트 생성 - probe {n}개, 실패 {fail}개, "
          f"overall={report.overall_score}, pass={report.pass_threshold}")
    if report.findings:
        by_type = Counter(f.type for f in report.findings)
        print(f"[Eval]        Finding {len(report.findings)}개: {dict(by_type)}")
