"""
tests/test_improvement_margin.py
개선 판정 최소 마진 검증 — "노이즈로 오른 점수"를 개선으로 보지 않는가.

세 축을 다룬다.
  1. judge(유지/롤백): 상승폭이 마진 미만이면 롤백하고, 하한선은 마진보다 우선한다.
  2. sweep(best 후보 선정): planner 가 min_delta 를 실제로 넘겨 마진이 활성화된다.
  3. 두 모듈의 임계 일관성: 같은 (before, after) 쌍에서 sweep 이 "개선"이라 하면
     judge 도 반드시 keep=True 여야 한다. 이게 이 기능의 핵심 계약이다.
     (sweep 승자는 _finish_internal_study 가 그 자리에서 확정해 judge 를 거치지
     않는다. 그래도 임계를 맞춰야 하는 이유는 두 경로가 각각 독립적으로 "개선했는가"를
     판정하고 둘 다 사용자 리포트로 나가기 때문이다 — 임계가 다르면 같은 점수 변화가
     경로에 따라 다르게 보고된다.)

스케일 주의: 두 지점 모두 정규화 composite(0~1)를 쓴다. 표시 점수 3점 = 0.03.
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
from agents.optimize.schemas import (
    OptimizationHistoryItem,
    OptimizationRequest,
    OptimizationResult,
)
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
        # 0.800 → 0.818 (+0.018) — 마진(0.03) 미만. 실측 σ_Δ(0.0107) 안에 묻히는 폭이다.
        verdict = _judge_totals(80.0, 81.8)

        self.assertFalse(verdict.keep)
        self.assertTrue(verdict.margin_rejected)
        self.assertIn("상승폭 부족", verdict.reason)
        # 사용자가 "얼마나 모자랐는지" 알 수 있어야 사후 조정이 가능하다.
        # reason 은 사용자에게 노출되므로 리포트 헤드라인과 같은 0~100 스케일로 쓴다
        # (판정 자체는 아래처럼 0~1 탐색 신호로 한다).
        self.assertIn(f"필요 +{MARGIN * 100:.1f}", verdict.reason)
        self.assertAlmostEqual(verdict.before_score, 0.80)

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
            supporting_labels=["retrieval_missing_gold"],
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

    def test_kept_trial_also_records_delta_and_margin(self):
        # 마진값이 적정한지 보려면 탈락 분포만으로는 부족하다 — 유지된 상승폭도 있어야
        # "마진을 얼마나 여유 있게 넘었는지"를 알 수 있다.
        item = self._pending()
        verdict = _judge_totals(80.0, 85.0)

        history.finalize_item(item, verdict, {"top_k": 7}, _report(composite_total=85.0))

        self.assertFalse(item.metadata["margin_rejected"])
        self.assertAlmostEqual(item.metadata["score_delta"], 0.05, places=9)
        self.assertEqual(item.metadata["improvement_margin"], MARGIN)


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
            supporting_labels=["retrieval_missing_gold"],
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


class RerankerFloorRelaxationMarginTest(unittest.TestCase):
    """floor 완화 예외도 judge 와 같은 기준으로 "올랐다"를 재야 한다.

    하한선 위반이 있으면 judge 가 floor 판정으로 먼저 반환하므로 마진이 개입하지
    않는다. 이 함수의 점수 관문이 그 경로의 유일한 방어선이고, 통과하면
    floor_violations 를 비운 keep=True 로 판정을 완전히 뒤집는다.
    """

    def _item(self):
        state = AgentDoctorState(index_config={"use_reranker": False})
        request = OptimizationRequest(
            request_id="req",
            iteration=0,
            baseline_config={"use_reranker": False},
            supporting_labels=["retrieval_low_rank"],
            search_space={"retriever.use_reranker": [True]},
            target_metrics=["context_precision"],
            optimizer="rules",
            max_trials=1,
        )
        return history.create_pending_item(
            state, request, "enable_reranker", {"use_reranker": False}, None
        )

    @staticmethod
    def _low_rank_report(count):
        return DiagnosticReport(
            report_id=f"r{count}",
            findings=[
                Finding(
                    finding_id=f"p{i}:retrieval_low_rank",
                    type="retrieval_failure",
                    severity="warning",
                    description="정답 청크의 순위가 낮음",
                    label="retrieval_low_rank",
                    confirmed=True,
                    affected_probes=[f"p{i}"],
                )
                for i in range(count)
            ],
        )

    def _relax(self, before_score, after_score):
        verdict = history.Verdict(
            keep=False,
            before_score=before_score,
            after_score=after_score,
            floor_violations=["context_precision"],
            reason="하한선 위반 ['context_precision'] → 무조건 롤백",
        )
        return agent._relax_reranker_precision_floor(
            self._item(),
            self._low_rank_report(3),   # before: low_rank 3건
            self._low_rank_report(1),   # after:  1건으로 감소 (완화 조건 충족)
            verdict,
        )

    def test_noise_level_gain_no_longer_overrides_the_floor(self):
        # +0.001 — 마진 미만. 예전 기준(after > before)이면 뒤집혔다.
        relaxed = self._relax(0.800, 0.801)

        self.assertFalse(relaxed.keep)
        self.assertEqual(relaxed.floor_violations, ["context_precision"])

    def test_gain_at_margin_still_relaxes(self):
        # 완화 자체는 살아 있어야 한다 — 마진을 넘으면 기존대로 유지로 뒤집힌다.
        relaxed = self._relax(0.800, 0.800 + MARGIN)

        self.assertTrue(relaxed.keep)
        self.assertEqual(relaxed.floor_violations, [])
        self.assertIn("retrieval_low_rank 감소 3→1", relaxed.reason)

    def test_relaxed_reason_uses_display_scale(self):
        """이 reason 도 "종합점수"라고 이름을 대므로 judge 와 같은 0~100 이어야 한다.

        keep=True 판정이라 이력·로그에 그대로 남는다 — 여기만 0~1 이면 같은 실행의
        다른 판정 사유와 스케일이 갈린다.
        """
        relaxed = self._relax(0.800, 0.800 + MARGIN)

        # 마진 상수를 재보정해도(#102) 스케일 계약은 그대로여야 하므로 상수에서 끌어온다.
        self.assertIn(f"80.0→{(0.800 + MARGIN) * 100:.1f}", relaxed.reason)
        self.assertIn(f"마진 {MARGIN * 100:.1f}", relaxed.reason)
        self.assertNotIn("0.800", relaxed.reason)
        # 판정에 쓰는 값 자체는 그대로 0~1 탐색 신호다.
        self.assertAlmostEqual(relaxed.before_score, 0.800)


class PassThresholdMarginTest(unittest.TestCase):
    """게이트를 넘었다는 사실만으로는 부족하다 — 노이즈로 넘었을 수 있다."""

    def _request(self, *, with_baseline=True):
        metadata = {
            "primary_metric": "composite_score",
            "min_delta": MIN_IMPROVEMENT_MARGIN,
        }
        if with_baseline:
            # baseline 89.9점: 게이트 미통과.
            metadata["baseline_metrics"] = {
                "composite_score": 0.899,
                "pass_threshold": False,
            }
        else:
            # baseline trial 자체가 없는 상태. 통과한 후보만 관측돼 있다.
            metadata["trial_results"] = [
                {
                    "trial_id": "candidate-only",
                    "config": {"chunker.chunk_size": 600},
                    "metrics": {"composite_score": 0.901, "pass_threshold": True},
                    "status": "completed",
                }
            ]
        return OptimizationRequest(
            request_id="pass-threshold",
            iteration=0,
            baseline_config={"chunk_size": 512, "chunk_overlap": 50},
            supporting_labels=["retrieval_missing_gold"],
            search_space={"chunker.chunk_size": [600]},
            target_metrics=[],
            optimizer="internal",
            max_trials=1,
            metadata=metadata,
        )

    @staticmethod
    def _passing(score):
        return lambda _config, _request: {
            "composite_score": score,
            "pass_threshold": True,
        }

    def test_passing_candidate_below_margin_falls_back_to_baseline(self):
        # 89.9 → 90.1. 게이트는 넘었지만 상승폭 0.002 는 마진의 1/10이다.
        result = run(self._request(), evaluator=self._passing(0.901))

        self.assertEqual(result.metadata["stop_reason"], "pass_threshold_reached")
        self.assertTrue(result.metadata["best_is_baseline"])
        self.assertFalse(result.metadata["improved"])
        self.assertEqual(result.best_config, {"chunker.chunk_size": 512})
        # judge 였다면 내렸을 판정과 일치한다.
        self.assertFalse(history.meets_improvement_margin(0.901 - 0.899))

    def test_passing_candidate_above_margin_is_selected(self):
        result = run(self._request(), evaluator=self._passing(0.95))

        self.assertEqual(result.metadata["stop_reason"], "pass_threshold_reached")
        self.assertFalse(result.metadata["best_is_baseline"])
        self.assertTrue(result.metadata["improved"])
        self.assertEqual(result.best_config, {"chunker.chunk_size": 600})

    def test_without_scorable_baseline_the_pass_is_kept(self):
        # 비교 대상이 없으면 마진을 적용할 근거가 없다 → 기존 동작(통과 채택) 유지.
        # (통과 조기 반환이 missing_scorable_baseline 검사보다 앞이라는 뜻이기도 하다.)
        result = run(self._request(with_baseline=False))

        self.assertEqual(result.metadata["stop_reason"], "pass_threshold_reached")
        self.assertFalse(result.metadata["best_is_baseline"])
        self.assertEqual(result.best_config, {"chunker.chunk_size": 600})


class UnjudgeableAttemptBudgetTest(unittest.TestCase):
    """측정 실패 사유마다 재시도 여지가 다르다."""

    @staticmethod
    def _item(*, study_error: bool):
        item = OptimizationHistoryItem(
            trial_id="t",
            request_id="r",
            iteration=0,
            failure_labels=["retrieval_missing_gold"],
            optimizer="internal",
            status="failed",
            selected_prescription_id="increase_top_k",
            action_key="retriever.top_k:increase",
        )
        item.metadata["unjudgeable"] = True
        if study_error:
            item.metadata["study_error"] = "scorable baseline 없음"
        return item

    # 실행 제어 단위가 (label, prescription_id) 에서 action key 로 옮겨졌다.
    # 측정 불가는 정확한 전이가 아니라 축 전체를 방문 범위에서 제한한다.
    KEY = "retriever.top_k:increase"

    def test_report_absent_is_excluded_after_one_attempt(self):
        excluded = agent._unjudgeable_exclusions([self._item(study_error=False)])

        self.assertIn(self.KEY, excluded)

    def test_measurement_failure_gets_a_second_attempt(self):
        # min_delta 도입 전 retryable 경로가 허용하던 2회를 그대로 유지한다.
        once = agent._unjudgeable_exclusions([self._item(study_error=True)])
        twice = agent._unjudgeable_exclusions(
            [self._item(study_error=True), self._item(study_error=True)]
        )

        self.assertNotIn(self.KEY, once)
        self.assertIn(self.KEY, twice)


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

    def test_same_prescription_is_retried_once_before_exclusion(self):
        # min_delta 도입 전 retryable 경로가 주던 2회 예산을 유지한다. 측정 실패는
        # 처방이 나쁘다는 증거가 아니고, baseline 관측값은 방문마다 갱신되므로
        # 다음 Eval 에서 성립할 수 있다.
        state = self._run_until_missing_baseline()
        self.assertNotIn(
            ("retrieval_missing_gold", "increase_top_k"),
            agent._unjudgeable_exclusions(state.optimization_history),
        )

        state.report = self._sweep_report(60.0)
        state = agent.run(state)

        # 2회차에서 같은 처방을 다시 골랐다(제외되지 않았다).
        self.assertEqual(
            state.optimization_history[-1].selected_prescription_id,
            "increase_top_k",
        )
        self.assertNotIn(
            ("retrieval_missing_gold", "increase_top_k"), state.blacklist
        )


# ── 3. 두 모듈의 임계 일관성 (핵심) ───────────────────────────────

class ThresholdConsistencyTest(unittest.TestCase):
    """sweep 이 "개선"이라 고른 후보는 judge 기준으로도 유지 판정이어야 한다.

    두 모듈은 각각 독립적으로 "개선했는가"를 판정하고(judge 는 rules 처방을,
    sweep 은 후보 묶음을) 둘 다 사용자 리포트로 나간다. 임계가 다르면 같은 점수
    변화가 어느 경로를 탔는지에 따라 다르게 보고된다.
    """

    def _sweep_improved(self, before_total: float, after_total: float) -> bool:
        request = OptimizationRequest(
            request_id="consistency",
            iteration=0,
            baseline_config={"chunk_size": 512, "chunk_overlap": 50},
            supporting_labels=["retrieval_missing_gold"],
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
        margin_total = MIN_IMPROVEMENT_MARGIN * 100   # 표시 스케일 3점
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
