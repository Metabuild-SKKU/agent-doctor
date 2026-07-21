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
같은 함수라 "1차 처방 후 종합점수가 좋으면 처방/탐색을 종료" 같은 판단에도 그대로 재사용된다.

게이트 = 점수 판정 통과 AND 검색 바닥선 통과:
  - score_pass(=report.pass_threshold) 가 False 면 통과 아님(최적화로).
    (overall 이 None 인 '평가 신호 없음' 경우 Eval 이 이미 pass_threshold=True 로 둬 막지 않는다.)
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


def passes(score_pass: bool, mean_recall_at_k: float | None = None) -> bool:
    """Eval 점수 판정(score_pass)에 검색 바닥선을 AND 로 얹는다.
    True 면 '충분히 좋다'(serve/종료), False 면 '최적화로'."""
    if not score_pass:
        return False
    return mean_recall_at_k is None or mean_recall_at_k >= RECALL_FLOOR


def passes_report(report) -> bool:
    """DiagnosticReport 에서 점수 판정 + recall 을 뽑아 passes() 판정. report 없으면 통과 아님."""
    if report is None:
        return False
    recall = report.ragas_scores.get("mean_recall_at_k") if report.ragas_scores else None
    return passes(report.pass_threshold, recall)
