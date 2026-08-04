"""
agents/optimize/gate.py
serve/optimize 게이트 정책 — "이제 충분히 좋은가"의 단일 기준.

[분업] Eval 은 점수 판정(report.pass_threshold = overall >= PASS_SCORE_THRESHOLD)과
숫자(mean_recall_at_k)를 계산해 넘긴다. Optimize 는 그 위에 운영 정책(검색 바닥선 등)을
얹어 최종 serve/optimize 를 판단한다. 이 모듈이 그 단일 진실원(single source of truth)이며,
아래 소비처가 모두 이 함수를 부른다:
  - graph.route_after_eval           : serve vs optimize 라우팅
  - planner._decide                  : already_optimal 조기 종료
  - _report_metrics → internal_adapter._trial_passed : 후보 sweep 종료
  - report_view._gate_summary        : 리포트 통과 배지·실패 사유(explain_report)
같은 함수라 "1차 처방 후 종합점수가 좋으면 처방/탐색을 종료" 같은 판단에도 그대로 재사용된다.

게이트 = 종합점수 판정 통과 AND 검색 바닥선 통과:
  - score_pass 가 False 면 통과 아님(최적화로). score_pass 의 기준은 설계 종합점수
    composite_score(품질×신뢰도, 0~100) >= COMPOSITE_PASS_THRESHOLD 다. overall(품질 단일축)
    은 통과율이 낮아도 높게 나와 시스템을 과대평가하므로, 게이트도 표시와 같은 composite 를 쓴다.
    (composite 미측정(None, '평가 신호 없음') 이면 Eval 의 pass_threshold 정책을 승계 —
     Eval 이 그 경우 True 로 둬 파이프라인을 막지 않는다.)
  - mean_recall_at_k 미측정(None, gold_chunk 없음) → floor 미적용(근거 없으면 막지 않음).
  - 그 외: recall >= RECALL_FLOOR 여야 통과.
recall floor 는 종합점수 평균이 가리는 "검색이 새는" 케이스(점수는 넘지만 gold 청크를 절반밖에
못 가져옴)를 잡는다. 이건 Optimize 가 top_k/chunk 처방으로 실제 고칠 수 있는 축이라, "고칠 수
있는 문제만 최적화로 보낸다"는 라우팅 정책과 맞물린다.

이 모듈은 qdrant 의존이 없어 단독 테스트가 가능하다.
"""
from __future__ import annotations

# 검색 바닥선(Optimize 전용 정책). 종합점수가 넘어도 이 밑이면 최적화로 보낸다.
# 실제 분포를 본 뒤 튜닝 예정.
RECALL_FLOOR = 0.6

# 종합점수(composite, 0~100) 통과 문턱. 처방을 더 공격적으로 적용·시연하도록 90 으로
# 올린다(품질·신뢰도 두 축이 대략 0.9 수준). 이 값이 안 넘으면 라우팅은 예산(max_iterations)
# 소진까지 처방을 계속 시도한 뒤 그때의 best 를 Serve 한다 — 즉 문턱을 높일수록 "통과" 대신
# "예산 소진 후 Serve"로 끝나는 실행이 늘고, 그만큼 처방이 더 많이 발동한다.
COMPOSITE_PASS_THRESHOLD = 90.0


def passes(score_pass: bool, mean_recall_at_k: float | None = None) -> bool:
    """점수 판정(score_pass)에 검색 바닥선을 AND 로 얹는다.
    True 면 '충분히 좋다'(serve/종료), False 면 '최적화로'."""
    if not score_pass:
        return False
    return mean_recall_at_k is None or mean_recall_at_k >= RECALL_FLOOR


def _as_float(value) -> float | None:
    """숫자로 못 읽는 값(None·문자열 등)은 '미측정'으로 떨어뜨린다."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def explain_report(report) -> dict:
    """passes_report() 와 같은 판정을, 각 축의 값과 실패 사유까지 붙여 돌려준다.
    표시 계층(report_view 등)이 판정 규칙을 다시 구현하지 않게 하려는 단일 창구."""
    if report is None:
        return {
            "pass": False,
            "reason": "report_missing",
            "score_pass": False,
            "score_source": "missing",
            "composite_total": None,
            "composite_threshold": COMPOSITE_PASS_THRESHOLD,
            "mean_recall_at_k": None,
            "recall_floor": RECALL_FLOOR,
            "recall_pass": None,
        }

    composite_total = _as_float((getattr(report, "composite_score", None) or {}).get("total"))
    if composite_total is None:
        score_pass = bool(report.pass_threshold)
        score_source = "report.pass_threshold"
        score_fail_reason = "eval_pass_threshold_false"
    else:
        score_pass = composite_total >= COMPOSITE_PASS_THRESHOLD
        score_source = "composite_score.total"
        score_fail_reason = "composite_below_threshold"

    recall = _as_float(report.ragas_scores.get("mean_recall_at_k")) if report.ragas_scores else None
    recall_pass = None if recall is None else recall >= RECALL_FLOOR

    if not score_pass:
        reason = score_fail_reason
    elif recall_pass is False:
        reason = "recall_below_floor"
    else:
        reason = "passed"

    return {
        "pass": passes(score_pass, recall),
        "reason": reason,
        "score_pass": score_pass,
        "score_source": score_source,
        "composite_total": composite_total,
        "composite_threshold": COMPOSITE_PASS_THRESHOLD,
        "mean_recall_at_k": recall,
        "recall_floor": RECALL_FLOOR,
        "recall_pass": recall_pass,
    }


def passes_report(report) -> bool:
    """DiagnosticReport 에서 종합점수 판정 + recall 을 뽑아 passes() 판정. report 없으면 통과 아님."""
    return explain_report(report)["pass"]
