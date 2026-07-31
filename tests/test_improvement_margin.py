"""
tests/test_improvement_margin.py
개선 판정 최소 마진 검증 — "노이즈로 오른 점수"를 개선으로 보지 않는가.

세 축을 다룬다.
  1. judge(유지/롤백): 상승폭이 마진 미만이면 롤백하고, 하한선은 마진보다 우선한다.
  2. sweep(best 후보 선정): planner 가 min_delta 를 실제로 넘겨 마진이 활성화된다.
  3. 두 모듈의 임계 일관성: 같은 (before, after) 쌍에서 sweep 이 "개선"이라 하면
     judge 도 반드시 keep=True 여야 한다. 이게 이 기능의 핵심 계약이다 —
     어긋나면 "sweep 이 고른 최선이 judge 에서 탈락"해 예산 한 번을 통째로 날린다.

스케일 주의: 두 지점 모두 정규화 composite(0~1)를 쓴다. 표시 점수 2점 = 0.02.
production 배선을 그대로 흉내 내야 부동소수 경계가 실제와 같아진다 —
judge 는 composite_score.total/100 을, sweep 은 관측값에 실린 composite_score
(= 같은 total/100)를 읽으므로 두 값이 비트 단위로 같다.
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.optimize import agent, history, planner
from agents.optimize.adapters.internal_adapter import run
from agents.optimize.history import MIN_IMPROVEMENT_MARGIN, judge
from agents.optimize.schemas import OptimizationRequest, OptimizationResult
from core.schema import DiagnosticReport, Finding
from core.state import AgentDoctorState


MARGIN = MIN_IMPROVEMENT_MARGIN


def _report(
    *,
    composite_total: float | None = None,
    overall: float | None = None,
    ragas: dict[str, float] | None = None,
) -> DiagnosticReport:
    """judge 입력용 Eval 리포트. composite 가 없으면 overall 로 폴백된다."""
    return DiagnosticReport(
        report_id="r",
        overall_score=overall,
        composite_score=({"total": composite_total} if composite_total is not None else None),
        ragas_scores=dict(ragas or {}),
    )


def _judge_totals(before_total: float, after_total: float, **kwargs):
    """표시 스케일(0~100) 종합점수 두 개로 판정한다."""
    return judge(
        _report(composite_total=before_total),
        _report(composite_total=after_total, **kwargs),
    )


# ── 1. judge: 유지/롤백에 마진 적용 ───────────────────────────────

class JudgeMarginTest(unittest.TestCase):
    """티끌만큼 오른 점수는 노이즈와 구분되지 않으므로 개선이 아니다."""

    def test_gain_below_margin_is_rolled_back(self):
        # 0.800 → 0.818 (+0.018) — 마진 0.02 미만
        verdict = _judge_totals(80.0, 81.8)

        self.assertFalse(verdict.keep)
        self.assertTrue(verdict.margin_rejected)
        self.assertIn("상승폭 부족", verdict.reason)
        # 사용자가 "얼마나 모자랐는지" 알 수 있어야 사후 조정이 가능하다.
        self.assertIn(f"필요 +{MARGIN:.3f}", verdict.reason)

    def test_gain_exactly_at_margin_is_kept(self):
        # 경계 포함. internal_adapter 의 math.isclose 판정과 같아야 한다.
        verdict = _judge_totals(80.0, 80.0 + MARGIN * 100)

        self.assertTrue(verdict.keep)
        self.assertFalse(verdict.margin_rejected)

    def test_gain_above_margin_is_kept(self):
        verdict = _judge_totals(80.0, 85.0)

        self.assertTrue(verdict.keep)
        self.assertIn("유지", verdict.reason)

    def test_floor_violation_wins_over_sufficient_gain(self):
        # 상승폭은 충분(+0.05)하지만 하한선을 깼다 → 점수와 무관하게 롤백.
        verdict = _judge_totals(80.0, 85.0, ragas={"faithfulness": 0.10})

        self.assertFalse(verdict.keep)
        self.assertEqual(verdict.floor_violations, ["faithfulness"])
        self.assertIn("하한선 위반", verdict.reason)
        # 하한선 롤백은 '마진 탈락'이 아니다(사후 마진 조정 통계를 오염시키면 안 됨).
        self.assertFalse(verdict.margin_rejected)

    def test_score_drop_is_rolled_back_as_before(self):
        verdict = _judge_totals(80.0, 75.0)

        self.assertFalse(verdict.keep)
        self.assertFalse(verdict.margin_rejected)
        self.assertIn("미상승", verdict.reason)

    def test_equal_score_is_rolled_back(self):
        verdict = _judge_totals(80.0, 80.0)

        self.assertFalse(verdict.keep)
        self.assertFalse(verdict.margin_rejected)

    def test_overall_fallback_uses_the_same_margin(self):
        # composite 미측정이면 overall(0~1)로 폴백하는데, 마진은 그대로 적용된다.
        below = judge(_report(overall=0.80), _report(overall=0.81))
        at_or_above = judge(_report(overall=0.80), _report(overall=0.83))

        self.assertFalse(below.keep)
        self.assertTrue(below.margin_rejected)
        self.assertTrue(at_or_above.keep)


class FinalizeMarginRecordTest(unittest.TestCase):
    """마진 값이 노이즈보다 과도한지 사후 검증할 근거를 이력에 남긴다."""

    def _pending(self):
        state = AgentDoctorState(index_config={"top_k": 5})
        request = OptimizationRequest(
            request_id="req",
            iteration=0,
            baseline_config={"top_k": 5},
            failure_label="retrieval_missing_gold",
            search_space={"retriever.top_k": [7]},
            target_metrics=["context_recall"],
            optimizer="internal",
            max_trials=1,
        )
        return history.create_pending_item(
            state, request, "increase_top_k", {"top_k": 5},
            _report(composite_total=80.0),
        )

    def test_margin_rejection_is_recorded_with_actual_delta(self):
        item = self._pending()
        verdict = _judge_totals(80.0, 81.8)

        history.finalize_item(item, verdict, {"top_k": 7}, _report(composite_total=81.8))

        self.assertTrue(item.metadata["margin_rejected"])
        self.assertAlmostEqual(item.metadata["score_delta"], 0.018, places=9)
        self.assertEqual(item.metadata["improvement_margin"], MARGIN)
        # 사용자 리포트가 읽는 rollback_reason 에도 사유가 실린다.
        self.assertIn("상승폭 부족", item.rollback_reason)

    def test_kept_trial_records_delta_without_margin_flag(self):
        item = self._pending()
        verdict = _judge_totals(80.0, 85.0)

        history.finalize_item(item, verdict, {"top_k": 7}, _report(composite_total=85.0))

        self.assertFalse(item.metadata["margin_rejected"])
        self.assertAlmostEqual(item.metadata["score_delta"], 0.05, places=9)
        self.assertNotIn("improvement_margin", item.metadata)


# ── 2. sweep: planner 가 마진을 실제로 넘기는가 ───────────────────

def _make_finding(label, candidates):
    return Finding(
        finding_id=f"p1:{label}",
        type="retrieval_failure",
        severity="warning",
        description=label,
        label=label,
        confirmed=True,
        affected_probes=["p1"],
        metadata={"parameter_candidates": candidates},
    )


class PlannerRelaysMarginTest(unittest.TestCase):
    """internal_adapter 는 history 를 모른다 — planner 가 중계해야 마진이 켜진다."""

    def test_internal_request_carries_min_delta(self):
        state = AgentDoctorState(
            report=DiagnosticReport(
                report_id="r",
                findings=[_make_finding("retrieval_missing_gold", {"top_k": [3, 7, 9]})],
                overall_score=60.0,
                ragas_scores={"context_recall": 0.6},
                pass_threshold=False,
            ),
            index_config={"chunk_size": 512, "chunk_overlap": 50, "top_k": 5},
            iteration=0,
            max_iterations=3,
        )

        request, _decision = planner.plan(state)

        self.assertEqual(request.optimizer, "internal")
        self.assertEqual(request.metadata["min_delta"], MIN_IMPROVEMENT_MARGIN)
        # objective 와 마진이 같은 스케일(0~1)이어야 100배 어긋나지 않는다.
        self.assertEqual(request.metadata["primary_metric"], "composite_score")


class SweepMinDeltaDirectionTest(unittest.TestCase):
    """낮을수록 좋은 지표에서도 마진의 부호가 맞는가."""

    def _minimize_request(self, min_delta):
        return OptimizationRequest(
            request_id="minimize-request",
            iteration=0,
            baseline_config={"chunk_size": 512, "chunk_overlap": 50},
            failure_label="retrieval_missing_gold",
            search_space={"chunker.chunk_size": [600]},
            target_metrics=[],
            optimizer="internal",
            max_trials=1,
            metadata={
                "primary_metric": "noise_sensitivity",
                "min_delta": min_delta,
                "baseline_metrics": {"noise_sensitivity": 0.30},
            },
        )

    def test_small_decrease_below_margin_keeps_baseline(self):
        # 0.30 → 0.29 는 '좋아졌지만' 폭이 0.01 로 마진 미만이다.
        result = run(
            self._minimize_request(0.02),
            evaluator=lambda _config, _request: {"noise_sensitivity": 0.29},
        )

        self.assertEqual(result.status, "completed")
        self.assertTrue(result.metadata["best_is_baseline"])
        self.assertFalse(result.metadata["improved"])

    def test_decrease_beyond_margin_selects_candidate(self):
        result = run(
            self._minimize_request(0.02),
            evaluator=lambda _config, _request: {"noise_sensitivity": 0.26},
        )

        self.assertEqual(result.best_config, {"chunker.chunk_size": 600})
        self.assertTrue(result.metadata["improved"])

    def test_increase_never_counts_as_improvement(self):
        # 부호가 뒤집혀 있으면 '나빠졌는데 개선'이 된다 — 그걸 막는 테스트.
        result = run(
            self._minimize_request(0.02),
            evaluator=lambda _config, _request: {"noise_sensitivity": 0.40},
        )

        self.assertTrue(result.metadata["best_is_baseline"])
        self.assertFalse(result.metadata["improved"])


class MissingScorableBaselineTest(unittest.TestCase):
    """min_delta > 0 이 되면서 새로 열리는 실패 경로의 분류 검증.

    지금까지 min_delta 가 0 이라 죽어 있던 경로다. baseline trial 이 평가되지 않으면
    sweep 이 비교 자체를 못 해 study 가 failed 로 끝나는데, 이는 '처방이 나빴다'가
    아니라 '측정이 성립하지 않았다'이므로 품질 blacklist 에 넣으면 안 된다.
    """

    @staticmethod
    def _sweep_report(score):
        finding = Finding(
            finding_id="sweep",
            type="retrieval_failure",
            severity="warning",
            description="gold가 검색 결과에 없음",
            label="retrieval_missing_gold",
            affected_probes=["p1"],
            metadata={"parameter_candidates": {"retriever.top_k": [7, 9, 11]}},
        )
        return DiagnosticReport(
            report_id="sweep-report",
            findings=[finding],
            overall_score=score,
            ragas_scores={
                "context_recall": 0.7,
                "faithfulness": 0.7,
                "noise_sensitivity": 0.2,
            },
            pass_threshold=False,
        )

    def _missing_baseline_result(self):
        return OptimizationResult(
            request_id="failed-study",
            status="failed",
            optimizer="internal",
            error="min_delta 비교에 필요한 scorable baseline trial이 없습니다.",
            metadata={
                "adapter_status": "failed",
                "stop_reason": "missing_scorable_baseline",
            },
        )

    def _run_until_missing_baseline(self):
        state = agent.run(
            AgentDoctorState(
                report=self._sweep_report(60.0),
                index_config={"top_k": 5, "chunk_size": 512, "chunk_overlap": 50},
                iteration=0,
                max_iterations=1,
            )
        )
        with patch(
            "agents.optimize.agent.optimizer.run",
            return_value=self._missing_baseline_result(),
        ):
            return agent.run(state)

    def test_classified_as_unjudgeable_not_quality_failure(self):
        state = self._run_until_missing_baseline()

        self.assertEqual(state.index_config["top_k"], 5)   # baseline 복원
        self.assertEqual(state.status, "rolled_back")
        self.assertTrue(state.optimization_history[0].metadata["unjudgeable"])
        self.assertNotIn(
            ("retrieval_missing_gold", "increase_top_k"), state.blacklist
        )

    def test_repeated_measurement_failure_still_avoids_blacklist(self):
        # 일시적 adapter 오류와 달리, 반복돼도 '품질이 나쁘다'는 증거는 생기지 않는다.
        # 무한 루프는 _unjudgeable_exclusions 가 실행 범위 안에서 막는다.
        state = self._run_until_missing_baseline()
        state.report = self._sweep_report(60.0)
        state = agent.run(state)

        self.assertNotIn(
            ("retrieval_missing_gold", "increase_top_k"), state.blacklist
        )
        self.assertIn(
            ("retrieval_missing_gold", "increase_top_k"),
            agent._unjudgeable_exclusions(state.optimization_history),
        )


# ── 3. 두 모듈의 임계 일관성 (핵심) ───────────────────────────────

class ThresholdConsistencyTest(unittest.TestCase):
    """sweep 이 "개선"이라 고른 후보는 judge 에서도 반드시 유지돼야 한다.

    두 모듈이 다른 임계를 쓰면 sweep 이 모든 후보를 평가해 고른 최선이 다음 방문의
    judge 에서 롤백된다 — 예산 한 번을 통째로 날리고 blacklist 까지 채운다.
    """

    def _sweep_improved(self, before_total: float, after_total: float) -> bool:
        request = OptimizationRequest(
            request_id="consistency",
            iteration=0,
            baseline_config={"chunk_size": 512, "chunk_overlap": 50},
            failure_label="retrieval_missing_gold",
            search_space={"chunker.chunk_size": [600]},
            target_metrics=[],
            optimizer="internal",
            max_trials=1,
            metadata={
                "primary_metric": "composite_score",
                "min_delta": MIN_IMPROVEMENT_MARGIN,
                # planner._report_metrics 와 같은 배선: composite_total/100.
                "baseline_metrics": {"composite_score": before_total / 100.0},
            },
        )
        result = run(
            request,
            evaluator=lambda _config, _request: {
                "composite_score": after_total / 100.0
            },
        )
        self.assertEqual(result.status, "completed")
        return bool(result.metadata["improved"])

    def test_verdicts_agree_across_and_around_the_boundary(self):
        before_total = 80.0
        margin_total = MIN_IMPROVEMENT_MARGIN * 100   # 표시 스케일 2점
        after_totals = [
            before_total - 5.0,                    # 하락
            before_total,                          # 동일
            before_total + margin_total / 2,       # 마진 절반
            before_total + margin_total - 1e-10,   # 경계 바로 아래
            before_total + margin_total,           # 경계
            before_total + margin_total + 1e-10,   # 경계 바로 위
            before_total + 5.0,                    # 충분한 상승
        ]

        for after_total in after_totals:
            with self.subTest(after_total=after_total):
                sweep_improved = self._sweep_improved(before_total, after_total)
                keep = _judge_totals(before_total, after_total).keep
                self.assertEqual(
                    sweep_improved,
                    keep,
                    f"sweep(improved={sweep_improved})과 judge(keep={keep})의 판정이 "
                    f"{before_total}→{after_total}에서 갈렸다.",
                )

    def test_sweep_winner_is_never_rejected_by_judge(self):
        # 계약을 한 방향으로 다시 못 박는다: sweep 이 이겼다면 judge 는 유지한다.
        before_total = 62.5
        for after_total in (62.5, 63.5, 64.5, 64.5001, 70.0):
            with self.subTest(after_total=after_total):
                if self._sweep_improved(before_total, after_total):
                    self.assertTrue(_judge_totals(before_total, after_total).keep)


if __name__ == "__main__":
    unittest.main()
