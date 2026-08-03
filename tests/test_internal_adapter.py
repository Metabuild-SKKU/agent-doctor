import os
import sys
import unittest
from copy import deepcopy


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.optimize.adapters.internal_adapter import InternalAdapter, run
from agents.optimize.schemas import OptimizationRequest
from core.schema import DiagnosticReport


class InternalAdapterTest(unittest.TestCase):
    def make_request(self, **overrides):
        values = {
            "request_id": "internal-request",
            "iteration": 0,
            "baseline_config": {"chunk_size": 512, "chunk_overlap": 50},
            "failure_label": "retrieval_missing_gold",
            "search_space": {"chunker.chunk_size": [600, 800]},
            "target_metrics": ["context_recall"],
            "optimizer": "internal",
            "max_trials": 2,
            "metadata": {"baseline_metrics": {"mean_recall_at_k": 0.5}},
        }
        values.update(overrides)
        return OptimizationRequest(**values)

    def test_evaluator_mode_returns_best_config(self):
        scores = {512: 0.5, 600: 0.62, 800: 0.81}

        def evaluator(config, _request):
            return {"mean_recall_at_k": scores[config["chunk_size"]]}

        result = run(self.make_request(), evaluator=evaluator)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.best_config, {"chunker.chunk_size": 800})
        self.assertEqual(result.best_score, 0.81)
        self.assertEqual(len([t for t in result.trial_results if not t.is_baseline]), 2)

    def test_inconclusive_supplied_baseline_is_re_evaluated(self):
        request = self.make_request(metadata={"baseline_metrics": {}})
        seen = []

        def evaluator(config, _request):
            value = config["chunk_size"]
            seen.append(value)
            return {"mean_recall_at_k": {512: 0.5, 600: 0.7, 800: 0.6}[value]}

        result = run(request, evaluator=evaluator)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.best_config, {"chunker.chunk_size": 600})
        self.assertEqual(seen, [512, 600, 800])

    def test_optimization_control_fields_are_excluded_from_effective_config(self):
        request = self.make_request(
            metadata={
                "study_baseline_config": {
                    "chunk_size": 512,
                    "chunk_overlap": 50,
                    "_optimization": {"trial_results": [{"score": 0.1}]},
                },
                "baseline_metrics": {"mean_recall_at_k": 0.5},
            }
        )

        result = run(request)

        baseline = next(trial for trial in result.trial_results if trial.is_baseline)
        self.assertNotIn("_optimization", baseline.metadata["effective_config"])
        self.assertIn("_optimization", request.metadata["study_baseline_config"])

    def test_without_evaluator_returns_first_unseen_candidate(self):
        result = run(self.make_request())

        self.assertEqual(result.status, "needs_evaluation")
        self.assertEqual(result.next_config, {"chunker.chunk_size": 600})
        self.assertEqual(result.best_config, {"chunker.chunk_size": 512})
        self.assertEqual(result.best_score, 0.5)

    def test_observation_advances_to_next_candidate(self):
        request = self.make_request(
            metadata={
                "baseline_metrics": {"mean_recall_at_k": 0.5},
                "trial_results": [
                    {
                        "trial_id": "trial-600",
                        "config": {"chunk_size": 600},
                        "metrics": {"mean_recall_at_k": 0.61},
                    }
                ],
            }
        )

        result = run(request)

        self.assertEqual(result.status, "needs_evaluation")
        self.assertEqual(result.next_config, {"chunker.chunk_size": 800})
        self.assertEqual(result.best_config, {"chunker.chunk_size": 600})

    def test_frozen_study_baseline_keeps_cross_round_observation_valid(self):
        request = self.make_request(
            baseline_config={"chunk_size": 600, "chunk_overlap": 50},
            metadata={
                "study_baseline_config": {
                    "chunk_size": 512,
                    "chunk_overlap": 50,
                },
                "baseline_metrics": {"mean_recall_at_k": 0.5},
                "trial_results": [
                    {
                        "config": {"chunker.chunk_size": 600},
                        "score": 0.61,
                    }
                ],
            },
        )

        result = run(request)

        self.assertEqual(result.status, "needs_evaluation")
        self.assertEqual(result.next_config, {"chunker.chunk_size": 800})
        self.assertEqual(result.best_config, {"chunker.chunk_size": 600})
        self.assertEqual(result.metadata["budget_used"], 1)

    def test_completed_observations_select_best(self):
        request = self.make_request(
            metadata={
                "baseline_metrics": {"mean_recall_at_k": 0.5},
                "trial_results": [
                    {
                        "config": {"chunker.chunk_size": 600},
                        "score": 0.65,
                    },
                    {
                        "config": {"chunker.chunk_size": 800},
                        "score": 0.72,
                    },
                ],
            }
        )

        result = run(request)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.best_config, {"chunker.chunk_size": 800})
        self.assertTrue(result.metadata["improved"])

    def test_min_delta_keeps_baseline(self):
        request = self.make_request(
            search_space={"chunker.chunk_size": [600]},
            max_trials=1,
            metadata={
                "baseline_metrics": {"mean_recall_at_k": 0.5},
                "min_delta": 0.02,
            },
        )

        result = run(
            request,
            evaluator=lambda _config, _request: {"mean_recall_at_k": 0.51},
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.best_config, {"chunker.chunk_size": 512})
        self.assertTrue(result.metadata["best_is_baseline"])
        self.assertFalse(result.metadata["improved"])

    def test_exact_min_delta_is_accepted(self):
        request = self.make_request(
            search_space={"chunker.chunk_size": [600]},
            max_trials=1,
            metadata={
                "baseline_metrics": {"mean_recall_at_k": 0.5},
                "min_delta": 0.1,
            },
        )

        result = run(
            request,
            evaluator=lambda _config, _request: {"mean_recall_at_k": 0.6},
        )

        self.assertEqual(result.best_config, {"chunker.chunk_size": 600})
        self.assertTrue(result.metadata["improved"])

    def test_minimize_direction_selects_lowest_score(self):
        request = self.make_request(
            target_metrics=[],
            metadata={
                "baseline_metrics": {"overall_score": 10.0},
                "optimization_direction": "minimize",
            },
        )
        scores = {600: 8.0, 800: 6.0}

        result = run(
            request,
            evaluator=lambda config, _request: scores[config["chunk_size"]],
        )

        self.assertEqual(result.best_config, {"chunker.chunk_size": 800})
        self.assertEqual(result.best_score, 6.0)

    def test_lower_is_better_metric_defaults_to_minimize(self):
        request = self.make_request(
            target_metrics=["noise_sensitivity"],
            metadata={"baseline_metrics": {"noise_sensitivity": 0.5}},
        )
        scores = {600: 0.8, 800: 0.2}

        result = run(
            request,
            evaluator=lambda config, _request: {
                "noise_sensitivity": scores[config["chunk_size"]]
            },
        )

        self.assertEqual(result.direction, "minimize")
        self.assertEqual(result.best_config, {"chunker.chunk_size": 800})

    def test_explicit_direction_rejects_incompatible_fallback_metric(self):
        request = self.make_request(
            target_metrics=["noise_sensitivity"],
            metadata={
                "baseline_metrics": {"overall_score": 0.8},
                "optimization_direction": "minimize",
            },
        )

        result = run(
            request,
            evaluator=lambda config, _request: {
                "overall_score": {600: 0.6, 800: 0.9}[config["chunk_size"]]
            },
        )

        self.assertEqual(result.status, "failed")
        self.assertIn("fallback 지표", result.error)

    def test_explicit_direction_still_rejects_mixed_comparison_metrics(self):
        request = self.make_request(
            search_space={"chunker.chunk_size": [600]},
            max_trials=1,
            metadata={
                "baseline_metrics": {"mean_recall_at_k": 0.5},
                "optimization_direction": "maximize",
                "trial_results": [
                    {
                        "config": {"chunker.chunk_size": 600},
                        "metrics": {"overall_score": 0.9},
                    }
                ],
            },
        )

        result = run(request)

        self.assertEqual(result.status, "failed")
        self.assertIn("서로 다른 목적 지표", result.error)

    def test_failed_trial_isolated_and_consumes_budget(self):
        calls = []

        def evaluator(config, _request):
            value = config["chunk_size"]
            calls.append(value)
            if value == 600:
                raise RuntimeError("의도한 trial 실패")
            return {"mean_recall_at_k": 0.7}

        result = run(self.make_request(), evaluator=evaluator)

        self.assertEqual(calls, [600, 800])
        self.assertEqual(result.best_config, {"chunker.chunk_size": 800})
        failed = [trial for trial in result.trial_results if trial.status == "failed"]
        self.assertEqual(len(failed), 1)

    def test_missing_target_metric_falls_back_with_warning(self):
        request = self.make_request(
            search_space={"chunker.chunk_size": [600]},
            max_trials=1,
            metadata={"baseline_metrics": {"overall_score": 0.4}},
        )

        result = run(
            request,
            evaluator=lambda _config, _request: {"overall_score": 0.7},
        )

        self.assertEqual(result.best_config, {"chunker.chunk_size": 600})
        self.assertTrue(any("overall_score" in warning for warning in result.warnings))

    def test_missing_target_metric_without_fallback_is_not_scorable(self):
        request = self.make_request(
            search_space={"chunker.chunk_size": [600]},
            max_trials=1,
            metadata={"allow_overall_fallback": False},
        )

        result = run(
            request,
            evaluator=lambda _config, _request: {"overall_score": 0.7},
        )

        self.assertEqual(result.status, "failed")
        self.assertTrue(
            all(trial.status == "inconclusive" for trial in result.trial_results)
        )

    def test_pass_threshold_stops_remaining_candidates(self):
        calls = []

        def evaluator(config, _request):
            value = config["chunk_size"]
            calls.append(value)
            return {
                "mean_recall_at_k": 0.9,
                "pass_threshold": value == 600,
            }

        result = run(self.make_request(), evaluator=evaluator)

        self.assertEqual(calls, [600])
        self.assertEqual(result.best_config, {"chunker.chunk_size": 600})
        self.assertEqual(result.metadata["stop_reason"], "pass_threshold_reached")

    def test_passing_trial_is_selected_over_higher_nonpassing_score(self):
        scores = {
            600: {"mean_recall_at_k": 0.9, "pass_threshold": False},
            800: {"mean_recall_at_k": 0.8, "pass_threshold": True},
        }

        result = run(
            self.make_request(),
            evaluator=lambda config, _request: scores[config["chunk_size"]],
        )

        self.assertEqual(result.best_config, {"chunker.chunk_size": 800})
        self.assertEqual(result.metadata["stop_reason"], "pass_threshold_reached")

    def test_unscored_eval_pass_keeps_baseline(self):
        request = self.make_request(metadata={})

        result = run(
            request,
            evaluator=lambda _config, _request: DiagnosticReport(
                report_id="unscored-pass",
                overall_score=None,
                pass_threshold=True,
            ),
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.best_config, {"chunker.chunk_size": 512})
        self.assertIsNone(result.best_score)
        self.assertTrue(result.metadata["unscored_pass"])

    def test_diagnostic_report_is_accepted_as_evaluator_result(self):
        request = self.make_request(
            search_space={"chunker.chunk_size": [600]},
            max_trials=1,
        )

        def evaluator(_config, _request):
            return DiagnosticReport(
                report_id="eval-report",
                ragas_scores={"context_recall": 0.75},
                overall_score=0.7,
                pass_threshold=False,
            )

        result = run(request, evaluator=evaluator)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.best_config, {"chunker.chunk_size": 600})
        self.assertEqual(result.best_score, 0.75)

    def test_outside_observation_is_rejected_without_using_budget(self):
        request = self.make_request(
            search_space={"chunker.chunk_size": [600]},
            max_trials=1,
            metadata={
                "baseline_metrics": {"mean_recall_at_k": 0.5},
                "trial_results": [
                    {
                        "config": {"unknown.path": "bad"},
                        "score": 1.0,
                    }
                ],
            },
        )

        result = run(request)

        self.assertEqual(result.status, "needs_evaluation")
        self.assertEqual(result.next_config, {"chunker.chunk_size": 600})
        rejected = [trial for trial in result.trial_results if trial.status == "rejected"]
        self.assertEqual(len(rejected), 1)

    def test_executed_rejected_trial_consumes_budget_and_is_not_repeated(self):
        request = self.make_request(
            metadata={
                "baseline_metrics": {"mean_recall_at_k": 0.5},
                "trial_results": [
                    {
                        "config": {"chunker.chunk_size": 600},
                        "status": "rejected",
                        "error": "guardrail 위반",
                    }
                ],
            }
        )

        result = run(request)

        self.assertEqual(result.status, "needs_evaluation")
        self.assertEqual(result.next_config, {"chunker.chunk_size": 800})
        self.assertEqual(result.metadata["budget_used"], 1)

    def test_over_budget_observations_do_not_affect_best(self):
        request = self.make_request(
            max_trials=1,
            metadata={
                "baseline_metrics": {"mean_recall_at_k": 0.5},
                "trial_results": [
                    {"config": {"chunker.chunk_size": 600}, "score": 0.6},
                    {"config": {"chunker.chunk_size": 800}, "score": 0.9},
                ],
            },
        )

        result = run(request)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.best_config, {"chunker.chunk_size": 600})
        self.assertTrue(any("max_trials" in warning for warning in result.warnings))

    def test_explicit_baseline_trial_precedes_inconclusive_summary(self):
        request = self.make_request(
            search_space={"chunker.chunk_size": [600]},
            max_trials=1,
            metadata={
                "baseline_trial": {
                    "trial_id": "explicit-baseline",
                    "config": {},
                    "score": 0.5,
                },
                "baseline_metrics": {"unrelated": 1.0},
                "trial_results": [
                    {"config": {"chunker.chunk_size": 600}, "score": 0.6}
                ],
            },
        )

        result = run(request)

        self.assertEqual(result.best_config, {"chunker.chunk_size": 600})
        baseline = next(trial for trial in result.trial_results if trial.is_baseline)
        self.assertEqual(baseline.trial_id, "explicit-baseline")

    def test_min_delta_requires_scorable_baseline(self):
        request = self.make_request(
            search_space={"chunker.chunk_size": [600]},
            max_trials=1,
            metadata={
                "min_delta": 0.1,
                "trial_results": [
                    {"config": {"chunker.chunk_size": 600}, "score": 0.6}
                ],
            },
        )

        result = run(request)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metadata["stop_reason"], "missing_scorable_baseline")

    def test_multi_axis_search_space_is_rejected(self):
        request = self.make_request(
            search_space={
                "chunker.chunk_size": [600],
                "chunker.chunk_overlap": [75],
            }
        )

        result = run(request)

        self.assertEqual(result.status, "failed")
        self.assertIn("config 축 하나", result.error)

    def test_request_is_not_mutated(self):
        request = self.make_request()
        before = deepcopy(request)

        InternalAdapter().run(request)

        self.assertEqual(request, before)


class InternalAdapterCategoricalAxisTest(unittest.TestCase):
    """범주형 축(chunker.strategy)도 숫자 축과 같은 계약으로 sweep된다."""

    def make_request(self, **overrides):
        values = {
            "request_id": "strategy-request",
            "iteration": 0,
            "baseline_config": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "chunk_strategy": "fixed",
            },
            "failure_label": "chunking_context_mismatch",
            "search_space": {
                "chunker.strategy": ["recursive_sentence", "markdown_recursive"]
            },
            "target_metrics": ["context_recall"],
            "optimizer": "internal",
            "max_trials": 2,
            "metadata": {"baseline_metrics": {"mean_recall_at_k": 0.5}},
        }
        values.update(overrides)
        return OptimizationRequest(**values)

    def _evaluator(self, scores):
        def evaluator(config, _request):
            return {"mean_recall_at_k": scores[config["chunk_strategy"]]}

        return evaluator

    def test_categorical_sweep_evaluates_every_candidate(self):
        seen = []

        def evaluator(config, _request):
            seen.append(config["chunk_strategy"])
            return {
                "mean_recall_at_k": {
                    "fixed": 0.5,
                    "recursive_sentence": 0.62,
                    "markdown_recursive": 0.81,
                }[config["chunk_strategy"]]
            }

        result = run(self.make_request(), evaluator=evaluator)

        self.assertEqual(result.status, "completed")
        # baseline은 metadata.baseline_metrics로 이미 채점돼 재평가되지 않는다.
        self.assertEqual(seen, ["recursive_sentence", "markdown_recursive"])
        self.assertEqual(result.best_config, {"chunker.strategy": "markdown_recursive"})
        self.assertEqual(result.best_score, 0.81)

    def test_candidate_equal_to_current_value_is_dropped(self):
        # 현재 전략이 후보에 포함되면 no-op이라 정규화 단계에서 제외된다.
        # "2후보 스윕"이라는 등록 표현과 실제 trial 수가 달라질 수 있다.
        request = self.make_request(
            baseline_config={
                "chunk_size": 512,
                "chunk_overlap": 50,
                "chunk_strategy": "markdown_recursive",
            }
        )

        result = run(request, evaluator=self._evaluator({
            "markdown_recursive": 0.5,
            "recursive_sentence": 0.7,
        }))

        self.assertEqual(result.search_space, {"chunker.strategy": ["recursive_sentence"]})
        self.assertEqual(
            [trial.config for trial in result.trial_results if not trial.is_baseline],
            [{"chunker.strategy": "recursive_sentence"}],
        )
        self.assertEqual(result.best_config, {"chunker.strategy": "recursive_sentence"})

    def test_string_fingerprints_are_stable_and_distinct(self):
        scores = {
            "recursive_sentence": 0.62,
            "markdown_recursive": 0.81,
        }

        first = run(self.make_request(), evaluator=self._evaluator(scores))
        second = run(self.make_request(), evaluator=self._evaluator(scores))

        fingerprints = [trial.fingerprint for trial in first.trial_results]
        self.assertEqual(len(set(fingerprints)), len(fingerprints))
        self.assertTrue(all(fingerprints))
        self.assertEqual(
            fingerprints,
            [trial.fingerprint for trial in second.trial_results],
        )

    def test_baseline_trial_uses_study_baseline_strategy(self):
        result = run(self.make_request())

        baseline = next(trial for trial in result.trial_results if trial.is_baseline)
        self.assertEqual(
            baseline.metadata["effective_config"]["chunk_strategy"],
            "fixed",
        )
        self.assertEqual(result.best_config, {"chunker.strategy": "fixed"})
        self.assertEqual(result.next_config, {"chunker.strategy": "recursive_sentence"})

    def test_baseline_is_kept_when_no_candidate_improves(self):
        result = run(
            self.make_request(
                metadata={"baseline_metrics": {"mean_recall_at_k": 0.8}},
            ),
            evaluator=self._evaluator({
                "recursive_sentence": 0.62,
                "markdown_recursive": 0.55,
            }),
        )

        self.assertEqual(result.status, "completed")
        self.assertTrue(result.metadata["best_is_baseline"])
        self.assertEqual(result.best_config, {"chunker.strategy": "fixed"})

    def test_direction_and_objective_follow_the_metric_not_the_axis_type(self):
        request = self.make_request(
            target_metrics=["noise_sensitivity"],
            metadata={
                "primary_metric": "noise_sensitivity",
                "baseline_metrics": {"noise_sensitivity": 0.4},
            },
        )

        def evaluator(config, _request):
            return {
                "noise_sensitivity": {
                    "recursive_sentence": 0.2,
                    "markdown_recursive": 0.6,
                }[config["chunk_strategy"]]
            }

        result = run(request, evaluator=evaluator)

        self.assertEqual(result.direction, "minimize")
        self.assertEqual(result.objective_metric, "noise_sensitivity")
        self.assertEqual(result.best_config, {"chunker.strategy": "recursive_sentence"})


class InternalStudyIdentityTest(unittest.TestCase):
    """sweep 하나 = ActionStudy 하나. 결과가 스스로 어느 study 인지 말해야 한다.

    이력·리포트가 대표 라벨을 되짚어 "아마 이 처방이었을 것"이라고 추측하지 않도록
    관측 결과에 action 정체성을 싣는다.
    """

    def _request(self, **overrides):
        values = {
            "request_id": "study-identity",
            "iteration": 0,
            "baseline_config": {"top_k": 5},
            "failure_label": "retrieval_missing_gold",
            "search_space": {"retriever.top_k": [7, 9]},
            "target_metrics": ["context_recall"],
            "optimizer": "internal",
            "max_trials": 2,
            "metadata": {"baseline_metrics": {"mean_recall_at_k": 0.5}},
            "action_key": "retriever.top_k:increase",
            "supporting_labels": [
                "retrieval_missing_gold",
                "retrieval_incomplete_enumeration",
            ],
        }
        values.update(overrides)
        return OptimizationRequest(**values)

    def test_completed_result_carries_action_identity(self):
        scores = {5: 0.5, 7: 0.7, 9: 0.9}

        def evaluator(config, _request):
            return {"mean_recall_at_k": scores[config["top_k"]]}

        result = run(self._request(), evaluator=evaluator)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.metadata["action_key"], "retriever.top_k:increase")
        self.assertEqual(
            result.metadata["supporting_labels"],
            ["retrieval_missing_gold", "retrieval_incomplete_enumeration"],
        )

    def test_pending_result_carries_action_identity_too(self):
        """evaluator 없이 다음 후보만 내놓는 중간 결과에도 실려야 한다.

        study 는 여러 방문에 걸쳐 이어지므로 중간 결과가 오히려 더 자주 기록된다.
        """
        result = run(self._request())

        self.assertEqual(result.status, "needs_evaluation")
        self.assertEqual(result.metadata["action_key"], "retriever.top_k:increase")

    def test_legacy_request_without_action_is_unchanged(self):
        result = run(self._request(action_key=None, supporting_labels=[]))

        self.assertNotIn("action_key", result.metadata)
        self.assertNotIn("supporting_labels", result.metadata)


if __name__ == "__main__":
    unittest.main()
