import os
import sys
import unittest
from copy import deepcopy


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.optimize.optimizer import (
    filter_candidate_values,
    is_capability_supported,
    merge_constraints,
    run,
)
from agents.optimize.schemas import (
    ConfigPatch,
    InternalAdapterResult,
    OptimizationRequest,
    PrescriptionCandidate,
    RAGBuilderResult,
)


class OptimizerPolicyTest(unittest.TestCase):
    def test_filter_candidate_values_uses_constraints(self):
        values = filter_candidate_values(
            "retriever.top_k",
            [6, 8, 10],
            {"top_k": 4},
            constraints={"top_k": {"max": 8}},
        )

        self.assertEqual(values, [6, 8])

    def test_chunk_overlap_is_limited_by_chunk_size_ratio(self):
        values = filter_candidate_values(
            "chunker.chunk_overlap",
            [100, 250, 300],
            {"chunk_size": 500},
        )

        self.assertEqual(values, [100])

    def test_capability_defaults_are_conservative(self):
        supported, reason = is_capability_supported("reranker")

        self.assertFalse(supported)
        self.assertEqual(reason, "unsupported_capability")

    def test_top_k_is_enabled_because_eval_consumes_it(self):
        # 보수적 기본값 원칙 자체는 위 reranker 케이스가 계속 지킨다.
        supported, reason = is_capability_supported("retriever.top_k")

        self.assertTrue(supported)
        self.assertIsNone(reason)

    def test_merge_constraints_accepts_flat_alias(self):
        constraints = merge_constraints({"top_k": {"max": 8}})

        self.assertEqual(constraints["retriever.top_k"]["max"], 8)

    def test_numeric_constraint_rejects_bool_and_string(self):
        values = filter_candidate_values(
            "chunker.chunk_size",
            [True, "600", 600],
            {"chunk_size": 512},
        )

        self.assertEqual(values, [600])


class OptimizerExecutionTest(unittest.TestCase):
    def make_candidate(
        self,
        *,
        prescription_id="resize_chunks",
        search_space=None,
        reindex=True,
    ):
        return PrescriptionCandidate(
            id=prescription_id,
            failure_label="retrieval_missing_gold",
            group="A",
            status="ready",
            patch=ConfigPatch(
                changes={"chunk_size": "increase"},
                reindex_required=reindex,
                description="청크 크기 조정",
            ),
            search_space=search_space or {},
        )

    def make_request(self, **overrides):
        values = {
            "request_id": "request-1",
            "iteration": 0,
            "baseline_config": {"chunk_size": 512, "chunk_overlap": 50},
            "failure_label": "retrieval_missing_gold",
            "candidates": [
                self.make_candidate(search_space={"chunker.chunk_size": [600, 800]})
            ],
            "optimizer": "rules",
        }
        values.update(overrides)
        return OptimizationRequest(**values)

    def test_rules_selects_first_valid_value_without_applying_config(self):
        request = self.make_request()

        result = run(request)

        self.assertEqual(result.status, "proposed")
        self.assertEqual(result.optimizer, "rules")
        self.assertEqual(result.config_patch.changes, {"chunker.chunk_size": 600})
        self.assertTrue(result.needs_reindex)
        self.assertIsNone(result.improved)
        self.assertEqual(request.baseline_config["chunk_size"], 512)

    def test_single_candidate_can_use_request_level_search_space(self):
        request = self.make_request(
            candidates=[self.make_candidate(search_space={})],
            search_space={"chunk_size": [700]},
        )

        result = run(request)

        self.assertEqual(result.config_patch.changes, {"chunker.chunk_size": 700})

    def test_candidate_search_space_has_priority_over_request_search_space(self):
        request = self.make_request(search_space={"chunk_size": [900]})

        result = run(request)

        self.assertEqual(result.config_patch.changes, {"chunker.chunk_size": 600})

    def test_constraints_remove_invalid_values_before_rules_selection(self):
        request = self.make_request(
            candidates=[
                self.make_candidate(
                    search_space={"chunker.chunk_size": [100, 400, 1600]}
                )
            ]
        )

        result = run(request)

        self.assertEqual(result.config_patch.changes, {"chunker.chunk_size": 400})

    def test_unsupported_pipeline_capability_is_skipped(self):
        # embedding_model 은 아직 소비처가 확인되지 않아 기본 비허용이다.
        # (top_k 는 소비처가 확인돼 허용으로 바뀌었으므로 이 케이스의 예시로 쓰지 않는다.)
        candidate = self.make_candidate(
            prescription_id="swap_embedding_model",
            search_space={"embedding.model": ["openai://text-embedding-3-large"]},
            reindex=True,
        )
        request = self.make_request(
            baseline_config={"embedding_model": "openai://text-embedding-3-small"},
            candidates=[candidate],
            metadata={"capabilities": {"retriever.top_k": False}},
        )

        result = run(request)

        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.metadata["error_code"], "unsupported_capability")
        self.assertEqual(
            result.metadata["skipped_candidates"][0]["prescription_id"],
            "swap_embedding_model",
        )

    def test_capability_can_be_explicitly_enabled(self):
        candidate = self.make_candidate(
            prescription_id="increase_top_k",
            search_space={"retriever.top_k": [5, 7]},
            reindex=False,
        )
        request = self.make_request(
            baseline_config={"top_k": 3},
            candidates=[candidate],
            metadata={"capabilities": {"retriever.top_k": True}},
        )

        result = run(request)

        self.assertEqual(result.status, "proposed")
        self.assertEqual(result.config_patch.changes, {"retriever.top_k": 5})
        self.assertFalse(result.needs_reindex)

    def test_capability_override_cannot_bypass_state_mapper_support(self):
        candidate = self.make_candidate(
            prescription_id="enable_reranker",
            search_space={"reranker.enabled": [True]},
            reindex=False,
        )
        request = self.make_request(
            candidates=[candidate],
            metadata={"capabilities": {"reranker": True}},
        )

        result = run(request)

        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.metadata["error_code"], "unsupported_backend_path")

    def test_missing_search_space_is_skipped_without_symbolic_interpretation(self):
        request = self.make_request(
            candidates=[self.make_candidate(search_space={})],
            search_space={},
        )

        result = run(request)

        self.assertEqual(result.status, "skipped")
        self.assertIsNone(result.config_patch)
        self.assertEqual(result.metadata["error_code"], "missing_search_space")

    def test_multi_axis_search_space_is_skipped(self):
        request = self.make_request(
            candidates=[
                self.make_candidate(
                    search_space={
                        "chunker.chunk_size": [600],
                        "chunker.chunk_overlap": [100],
                    }
                )
            ]
        )

        result = run(request)

        self.assertEqual(result.status, "skipped")

    def test_candidates_are_checked_in_planner_order(self):
        unsupported = self.make_candidate(
            prescription_id="enable_reranker",
            search_space={"reranker.enabled": [True]},
            reindex=False,
        )
        supported = self.make_candidate(
            prescription_id="resize_chunks",
            search_space={"chunker.chunk_size": [700]},
        )
        request = self.make_request(candidates=[unsupported, supported])

        result = run(request)

        self.assertEqual(result.selected_candidate.id, "resize_chunks")
        self.assertEqual(result.config_patch.changes, {"chunker.chunk_size": 700})
        self.assertEqual(
            result.metadata["skipped_candidates"][0]["prescription_id"],
            "enable_reranker",
        )

    def test_internal_next_candidate_is_normalized_to_patch(self):
        candidate = self.make_candidate(
            prescription_id="increase_top_k",
            search_space={"retriever.top_k": [8, 12]},
            reindex=False,
        )
        request = self.make_request(
            optimizer="internal",
            baseline_config={"top_k": 5},
            candidates=[candidate],
            max_trials=2,
        )

        def runner(_request):
            return InternalAdapterResult(
                request_id="request-1",
                status="needs_evaluation",
                next_config={"retriever.top_k": 8},
                search_space={"retriever.top_k": [8, 12]},
                metadata={"stop_reason": "candidate_requires_evaluation"},
            )

        result = run(request, backend_runners={"internal": runner})

        self.assertEqual(result.status, "proposed")
        self.assertEqual(result.optimizer, "internal")
        self.assertEqual(result.config_patch.changes, {"retriever.top_k": 8})
        self.assertFalse(result.needs_reindex)
        self.assertEqual(result.metadata["parameter_path"], "retriever.top_k")

    def test_internal_completed_baseline_keeps_current_config(self):
        candidate = self.make_candidate(
            prescription_id="increase_top_k",
            search_space={"retriever.top_k": [8, 12]},
            reindex=False,
        )
        request = self.make_request(
            optimizer="internal",
            baseline_config={"top_k": 5},
            candidates=[candidate],
            max_trials=2,
        )

        def runner(_request):
            return InternalAdapterResult(
                request_id="request-1",
                status="completed",
                best_config={"retriever.top_k": 5},
                best_score=0.7,
                search_space={"retriever.top_k": [8, 12]},
                metadata={"best_is_baseline": True},
            )

        result = run(request, backend_runners={"internal": runner})

        self.assertEqual(result.status, "skipped")
        self.assertIsNone(result.config_patch)
        self.assertEqual(result.metadata["error_code"], "baseline_selected")

    def test_internal_rejects_config_outside_filtered_candidates(self):
        candidate = self.make_candidate(
            prescription_id="increase_top_k",
            search_space={"retriever.top_k": [8, 12]},
            reindex=False,
        )
        request = self.make_request(
            optimizer="internal",
            baseline_config={"top_k": 5},
            candidates=[candidate],
            max_trials=2,
        )

        def runner(_request):
            return InternalAdapterResult(
                request_id="request-1",
                status="needs_evaluation",
                next_config={"retriever.top_k": 20},
            )

        result = run(request, backend_runners={"internal": runner})

        self.assertEqual(result.status, "failed")
        self.assertEqual(
            result.metadata["error_code"],
            "invalid_internal_next_config",
        )

    def test_ragbuilder_result_is_normalized(self):
        request = self.make_request(optimizer="ragbuilder")

        def runner(_request):
            return RAGBuilderResult(
                request_id="request-1",
                best_config={"chunker.chunk_size": 800},
                best_score=0.82,
                status="completed",
            )

        result = run(request, backend_runners={"ragbuilder": runner})

        self.assertEqual(result.status, "proposed")
        self.assertEqual(result.optimizer, "ragbuilder")
        self.assertEqual(result.best_config, {"chunker.chunk_size": 800})
        self.assertEqual(result.metadata["best_score"], 0.82)
        self.assertTrue(result.needs_reindex)

    def test_ragbuilder_outside_search_space_falls_back_to_rules(self):
        request = self.make_request(optimizer="ragbuilder")

        def runner(_request):
            return RAGBuilderResult(
                request_id="request-1",
                best_config={"chunker.chunk_size": 1000},
                best_score=0.9,
                status="completed",
            )

        result = run(request, backend_runners={"ragbuilder": runner})

        self.assertEqual(result.optimizer, "rules")
        self.assertEqual(result.config_patch.changes, {"chunker.chunk_size": 600})
        self.assertEqual(
            result.metadata["fallback_reason"],
            "best_config_outside_search_space",
        )

    def test_ragbuilder_failure_falls_back_to_verified_rules_candidate(self):
        request = self.make_request(optimizer="ragbuilder")

        def runner(_request):
            return RAGBuilderResult(
                request_id="request-1",
                best_config=None,
                best_score=None,
                status="failed",
                error="external_failure",
            )

        result = run(request, backend_runners={"ragbuilder": runner})

        self.assertEqual(result.status, "proposed")
        self.assertEqual(result.optimizer, "rules")
        self.assertEqual(result.metadata["fallback_reason"], "external_failure")

    def test_request_and_search_space_are_not_mutated(self):
        request = self.make_request(optimizer="ragbuilder")
        before = deepcopy(request)

        def runner(prepared_request):
            prepared_request.search_space["chunker.chunk_size"].append(1000)
            return RAGBuilderResult(
                request_id="request-1",
                best_config={"chunker.chunk_size": 600},
                best_score=0.8,
                status="completed",
            )

        run(request, backend_runners={"ragbuilder": runner})

        self.assertEqual(request, before)


if __name__ == "__main__":
    unittest.main()
