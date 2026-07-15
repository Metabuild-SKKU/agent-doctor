"""
agents/eval/report.py
STEP5: 진단 리포트 생성

설계 문서 'STEP 5: 진단 리포트' 구현.
모든 Probe 판정 결과(EvalRecord)를 집계해 DiagnosticReport 를 만든다.
    - overall_score : RAGAS 가중 평균(있으면) / 없으면 규칙 지표 폴백
    - pass_threshold: overall_score >= PASS_SCORE_THRESHOLD
    - ragas_scores  : RAGAS 평균 + 규칙 지표 평균 + 결과 분포(관측용)
    - findings      : 전 record 의 Finding 합침(확정 우선 정렬)
    - findings_summary: 확정/예비·라벨 집계(+진단 모드). Optimize 가 확정건 우선 처리하도록 요약

  주의: overall_score/pass_threshold 는 '지표' 기반이라 예비 Finding 이 pass 를 뒤집지 않는다.
        예비는 '더 깊은 모드에서 확정할 수 있는 의심 원인'으로만 싣는다(정보 제공).

graph.route_after_eval() 이 report.pass_threshold 로 Serve/Optimize 분기를 결정한다.
"""
from __future__ import annotations

import uuid

from core.schema import DiagnosticReport
from agents.eval.types import (
    EvalRecord, RAGAS_WEIGHTS, PASS_SCORE_THRESHOLD, F1_PASS_THRESHOLD,
    resolve_mode,
)

_RAGAS_KEYS = ("faithfulness", "context_precision", "context_recall", "response_relevancy")


def build_report(records: list[EvalRecord], iteration: int, mode: int | None = None) -> DiagnosticReport:
    """records 를 집계해 DiagnosticReport 반환.

    mode(진단 모드)는 findings_summary 에 기록해 '이 findings 가 어느 깊이에서 나왔는지'를 남긴다.
    (예비 findings 는 더 깊은 모드에서 확정 가능하다는 맥락.) 미지정 시 EVAL_MODE/기본값.
    """
    if mode is None:
        mode = resolve_mode()

    findings = [f for r in records for f in r.findings]
    # Optimize 소비 편의: 확정 우선. 동률은 원래 순서 유지(stable sort — probe별 D→A→C→B).
    findings.sort(key=lambda f: not f.confirmed)

    ragas_means = _ragas_means(records)
    rule_means = _rule_means(records)
    overall = _overall_score(ragas_means, rule_means)
    oracle_acc = _oracle_accuracy(records)

    scores = {**rule_means}
    scores.update(ragas_means)                          # RAGAS 평균(있으면)
    # 브랜치 제거 → findings 유무로 결과 분포(진단됨/정상)
    n_diag = sum(1 for r in records if r.findings)
    scores["outcome_distribution"] = {"diagnosed": n_diag, "ok": len(records) - n_diag}

    # 평가 신호(GT 규칙지표/RAGAS)가 전혀 없으면 진단 불가 →
    # eval 한계로 파이프라인을 막지 않도록 통과 처리(overall_score=None).
    # [설계 결정] 이건 "판정 보류"이지 "품질 확인"이 아니다 — ground_truth 없는 probe만 있거나
    # (예: user_log 소스) RAGAS 미실행(EVAL_MODE<deep)이면 애초에 점수를 낼 근거 자체가 없다.
    # 근거 없이 fail 로 강제하면 Optimize 가 존재하지도 않는 문제를 잡으러 무한 루프를 돌 수
    # 있으므로, "판단 불가 → 막지 않음"을 택했다(Serve 이후에도 report.overall_score is None
    # 으로 이 상태는 추적 가능하다).
    if overall is None:
        overall_val, pass_thr = None, True
        print("[Eval] 경고: 평가 신호 없음(GT·RAGAS 부재) → 통과 처리")
    else:
        overall_val, pass_thr = round(overall, 4), overall >= PASS_SCORE_THRESHOLD

    report = DiagnosticReport(
        report_id=f"report_{uuid.uuid4().hex[:8]}",
        findings=findings,
        findings_summary=_findings_summary(records, mode),
        ragas_scores=scores,
        oracle_accuracy=oracle_acc,
        overall_score=overall_val,
        pass_threshold=pass_thr,
        iteration=iteration,
    )

    _print_summary(records, report)
    return report


# ── 집계 헬퍼 ─────────────────────────────────────────────────────

def _findings_summary(records: list[EvalRecord], mode: int) -> dict:
    """Finding 들을 확정/예비·라벨로 집계. Optimize 가 확정건 우선 처리하도록 요약 제공.

    라벨 분포는 **테스트셋(probe)당 1로 정규화**해 가중 집계한다: 한 probe 에서 finding 이
    N개 나오면 각 finding 은 1/N 로 계산된다(예: 3개 → 각 0.333). 특정 probe 가 여러 원인을
    동시에 내도, 원인별 분포·우선순위가 그 probe 하나에 의해 과대 계상되지 않게 하기 위함.

      mode              : 이 진단이 실행된 모드(1~4). 예비가 왜 예비인지의 맥락.
      total/confirmed/preliminary : finding **원시 개수**(정수).
      weighted_total    : probe당 정규화 합 = finding 이 1개 이상 나온 probe 수(라벨 가중합의 총계).
      confirmed_labels / preliminary_labels : 라벨별 **가중 개수**(probe당 1/N, 소수 3자리).
    """
    all_findings = [f for r in records for f in r.findings]
    confirmed = [f for f in all_findings if f.confirmed]
    preliminary = [f for f in all_findings if not f.confirmed]

    # probe당 정규화 가중치 w=1/N 로 라벨 분포를 가중 집계 (N = 그 probe 의 finding 수)
    conf_w: dict[str, float] = {}
    prelim_w: dict[str, float] = {}
    for r in records:
        n = len(r.findings)
        if not n:
            continue
        w = 1.0 / n
        for f in r.findings:
            bucket = conf_w if f.confirmed else prelim_w
            bucket[f.label] = bucket.get(f.label, 0.0) + w

    return {
        "mode": mode,
        "total": len(all_findings),
        "confirmed": len(confirmed),
        "preliminary": len(preliminary),
        "weighted_total": round(sum(conf_w.values()) + sum(prelim_w.values()), 3),
        "confirmed_labels": {k: round(v, 3) for k, v in conf_w.items()},
        "preliminary_labels": {k: round(v, 3) for k, v in prelim_w.items()},
    }


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
    fail = sum(1 for r in records if r.findings)
    fs = report.findings_summary
    print(f"[Eval] STEP5: 리포트 생성 - probe {n}개, 실패 {fail}개, "
          f"overall={report.overall_score}, pass={report.pass_threshold} (모드 {fs.get('mode')})")
    if report.findings:
        # 타입 분포도 probe당 1로 정규화(가중): 한 probe 의 N개 finding → 각 1/N
        by_type: dict[str, float] = {}
        for r in records:
            k = len(r.findings)
            if not k:
                continue
            for f in r.findings:
                by_type[f.type] = round(by_type.get(f.type, 0.0) + 1.0 / k, 3)
        print(f"[Eval]        Finding {len(report.findings)}개 "
              f"(확정 {fs.get('confirmed', 0)} / 예비 {fs.get('preliminary', 0)}), "
              f"가중 타입분포 {by_type}")
        if fs.get("preliminary"):
            print(f"[Eval]        예비 {fs['preliminary']}개는 더 깊은 모드(EVAL_MODE=deep/full)에서 확정 가능")
