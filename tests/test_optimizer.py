import os
import sys
import unittest
from copy import deepcopy


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.optimize import action_catalog
from agents.optimize.config_mapper import canonicalize_path
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

    def test_both_chunking_strategies_survive_constraint(self):
        # 리뷰 blocker: rules 가 2-후보 스윕(recursive_sentence·markdown_recursive)을 등록하는데
        # allowed 가 recursive_sentence 만이면 markdown_recursive 가 필터돼 스윕이 1개가 된다.
        values = filter_candidate_values(
            "chunker.strategy",
            ["recursive_sentence", "markdown_recursive"],
            {"chunk_strategy": "fixed"},
        )
        self.assertEqual(values, ["recursive_sentence", "markdown_recursive"])

    def test_reranker_is_enabled_because_common_retriever_consumes_it(self):
        supported, reason = is_capability_supported("reranker")

        self.assertTrue(supported)
        self.assertIsNone(reason)

    def test_top_k_is_enabled_because_eval_consumes_it(self):
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
    # 선택 단위가 action 하나가 된 뒤 optimizer 는 후보 목록을 순회하지 않는다.
    # 요청이 이미 고른 변경 하나를 들고 오고, optimizer 는 그것을 재검증할 뿐이다.
    # 이 helper 는 그 "고른 변경"을 request kwargs 로 돌려준다.
    @staticmethod
    def make_action(
        *,
        prescription_id="resize_chunks",
        action_key=None,
        search_space=None,
        **_ignored,
    ):
        space = dict(search_space or {})
        if action_key is None and space:
            # 축과 후보값에서 유도한다. 숫자 축은 이 테스트들에서 늘 '증가' 방향이라
            # 기본값으로 두고, 필요하면 호출부가 action_key 를 직접 준다.
            path, values = next(iter(space.items()))
            first = values[0] if values else None
            operation = action_catalog.derive_operation(first)
            if operation == "replace" and isinstance(first, (int, float)):
                operation = "increase"
            action_key = f"{canonicalize_path(path)}:{operation}"
        return {
            "search_space": space,
            "action_key": action_key,
            "prescription_id": prescription_id,
        }

    def make_request(self, **overrides):
        values = {
            "request_id": "request-1",
            "iteration": 0,
            "baseline_config": {"chunk_size": 512, "chunk_overlap": 50},
            "supporting_labels": ["retrieval_missing_gold"],
            **self.make_action(search_space={"chunker.chunk_size": [600, 800]}),
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

    def test_request_search_space_accepts_flat_keys(self):
        """요청의 search_space 가 유일한 탐색 범위 계약이다.

        전환 전에는 후보별 search_space 와 요청 수준 search_space 가 둘 다 있어
        어느 쪽이 이기는지 규칙이 필요했다. 선택이 planner 로 모이면서 그 이중
        계약이 사라졌다 — 요청에 실린 것 하나뿐이다.
        """
        request = self.make_request(search_space={"chunk_size": [700]})

        result = run(request)

        self.assertEqual(result.config_patch.changes, {"chunker.chunk_size": 700})

    def test_constraints_remove_invalid_values_before_rules_selection(self):
        request = self.make_request(
            **self.make_action(
                    search_space={"chunker.chunk_size": [100, 400, 1600]}
                )
        )

        result = run(request)

        self.assertEqual(result.config_patch.changes, {"chunker.chunk_size": 400})

    def test_unsupported_pipeline_capability_is_skipped(self):
        # embedding_model 은 아직 소비처가 확인되지 않아 기본 비허용이다.
        # (top_k 는 소비처가 확인돼 허용으로 바뀌었으므로 이 케이스의 예시로 쓰지 않는다.)
        candidate = self.make_action(
            prescription_id="swap_embedding_model",
            search_space={"embedding.model": ["openai://text-embedding-3-large"]},
            reindex=True,
        )
        request = self.make_request(
            baseline_config={"embedding_model": "openai://text-embedding-3-small"},
            **candidate,
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
        candidate = self.make_action(
            prescription_id="increase_top_k",
            search_space={"retriever.top_k": [5, 7]},
            reindex=False,
        )
        request = self.make_request(
            baseline_config={"top_k": 3},
            **candidate,
            metadata={"capabilities": {"retriever.top_k": True}},
        )

        result = run(request)

        self.assertEqual(result.status, "proposed")
        self.assertEqual(result.config_patch.changes, {"retriever.top_k": 5})
        self.assertFalse(result.needs_reindex)

    def test_reranker_toggle_is_mappable_and_does_not_reindex(self):
        candidate = self.make_action(
            prescription_id="enable_reranker",
            search_space={"reranker.enabled": [True]},
            reindex=False,
        )
        request = self.make_request(
            baseline_config={"use_reranker": False},
            **candidate,
        )

        result = run(request)

        self.assertEqual(result.status, "proposed")
        self.assertEqual(
            result.config_patch.changes,
            {"reranker.enabled": True},
        )
        self.assertFalse(result.needs_reindex)

    def test_context_compression_is_mappable_and_does_not_reindex(self):
        candidate = self.make_action(
            prescription_id="context_compression",
            search_space={"context.compression.enabled": [True]},
            reindex=False,
        )
        request = self.make_request(
            baseline_config={"context_compression": False},
            **candidate,
        )

        result = run(request)

        self.assertEqual(result.status, "proposed")
        self.assertEqual(
            result.config_patch.changes,
            {"context.compression.enabled": True},
        )
        self.assertFalse(result.needs_reindex)

    def test_unavailable_runtime_reranker_is_skipped(self):
        candidate = self.make_action(
            prescription_id="enable_reranker",
            search_space={"reranker.enabled": [True]},
            reindex=False,
        )
        request = self.make_request(
            baseline_config={"use_reranker": False},
            **candidate,
            metadata={
                "runtime_capabilities": {
                    "reranker": {
                        "status": "unavailable",
                        "reason": "model_load_failed",
                        "retryable": True,
                    }
                }
            },
        )

        result = run(request)

        self.assertEqual(result.status, "skipped")
        self.assertEqual(
            result.metadata["error_code"],
            "runtime_capability_unavailable",
        )
        self.assertEqual(
            result.metadata["skipped_candidates"][0]["prescription_id"],
            "enable_reranker",
        )

    def test_unknown_runtime_reranker_is_skipped(self):
        candidate = self.make_action(
            prescription_id="enable_reranker",
            search_space={"reranker.enabled": [True]},
            reindex=False,
        )
        request = self.make_request(
            baseline_config={"use_reranker": False},
            **candidate,
            metadata={"runtime_capabilities": {}},
        )

        result = run(request)

        self.assertEqual(result.status, "skipped")
        self.assertEqual(
            result.metadata["error_code"],
            "runtime_capability_unavailable",
        )

    def test_reranker_candidate_count_requires_enabled_reranker(self):
        candidate = self.make_action(
            prescription_id="widen_rerank_candidates",
            search_space={"reranker.candidate_count": [40]},
            reindex=False,
        )
        request = self.make_request(
            baseline_config={
                "use_reranker": False,
                "rerank_candidates": 20,
            },
            **candidate,
            metadata={
                "runtime_capabilities": {
                    "reranker": {
                        "status": "verified",
                        "reason": None,
                        "retryable": False,
                    }
                }
            },
        )

        result = run(request)

        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.metadata["error_code"], "reranker_disabled")
        self.assertEqual(
            result.metadata["skipped_candidates"],
            [
                {
                    "action_key": "reranker.candidate_count:increase",
                    "prescription_id": "widen_rerank_candidates",
                    "reason": "reranker_disabled",
                }
            ],
        )

    def test_missing_search_space_is_skipped_without_symbolic_interpretation(self):
        request = self.make_request(search_space={})

        result = run(request)

        self.assertEqual(result.status, "skipped")
        self.assertIsNone(result.config_patch)
        self.assertEqual(result.metadata["error_code"], "missing_search_space")

    def test_multi_axis_search_space_is_skipped(self):
        request = self.make_request(
            **self.make_action(
                    search_space={
                        "chunker.chunk_size": [600],
                        "chunker.chunk_overlap": [100],
                    }
                )
        )

        result = run(request)

        self.assertEqual(result.status, "skipped")

    def test_optimizer_does_not_reorder_the_selection(self):
        """optimizer 는 "어떤 변경을 먼저 시도할지"를 다시 정하지 않는다.

        전환 전에는 planner 가 후보 목록을 넘기고 optimizer 가 선언 순서대로 훑어
        첫 실행 가능 후보를 골랐다 — 선택 책임이 두 계층에 흩어져 있었다. 이제 경쟁은
        planner 안에서 끝나고, optimizer 는 넘어온 변경 하나를 재검증만 한다.
        실행 불가면 대신 고르는 게 아니라 skip 을 돌려주고, 다음 action 선택은
        agent 가 planner 를 다시 불러서 한다.
        """
        request = self.make_request(
            **self.make_action(
                prescription_id="swap_embedding_model",
                action_key="embedding.model:replace",
                search_space={"embedding.model": ["unsupported/model"]},
            )
        )

        result = run(request)

        self.assertEqual(result.status, "skipped")
        self.assertIsNone(result.config_patch)
        self.assertEqual(
            result.metadata["skipped_candidates"][0]["action_key"],
            "embedding.model:replace",
        )

    def test_internal_next_candidate_is_normalized_to_patch(self):
        candidate = self.make_action(
            prescription_id="increase_top_k",
            search_space={"retriever.top_k": [8, 12]},
            reindex=False,
        )
        request = self.make_request(
            optimizer="internal",
            baseline_config={"top_k": 5},
            **candidate,
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
        candidate = self.make_action(
            prescription_id="increase_top_k",
            search_space={"retriever.top_k": [8, 12]},
            reindex=False,
        )
        request = self.make_request(
            optimizer="internal",
            baseline_config={"top_k": 5},
            **candidate,
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
        candidate = self.make_action(
            prescription_id="increase_top_k",
            search_space={"retriever.top_k": [8, 12]},
            reindex=False,
        )
        request = self.make_request(
            optimizer="internal",
            baseline_config={"top_k": 5},
            **candidate,
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

    def test_chunk_prescreener_recoverable_skip_falls_back_to_rules(self):
        candidate = self.make_action(
            search_space={"chunker.chunk_size": [400]},
        )
        request = self.make_request(
            optimizer="internal",
            **candidate,
            max_trials=1,
        )

        for error_code in (
            "missing_chunk_precheck_context",
            "chunk_precheck_unavailable",
        ):
            with self.subTest(error_code=error_code):
                def runner(_request):
                    return InternalAdapterResult(
                        request_id="request-1",
                        status="skipped",
                        error="사전검사를 실행할 수 없음",
                        metadata={"error_code": error_code},
                    )

                result = run(request, backend_runners={"internal": runner})

                self.assertEqual(result.status, "proposed")
                self.assertEqual(result.optimizer, "rules")
                self.assertEqual(
                    result.config_patch.changes,
                    {"chunker.chunk_size": 400},
                )
                self.assertEqual(result.metadata["fallback_reason"], error_code)

    def test_chunk_prescreener_nonrecoverable_skip_does_not_fall_back(self):
        candidate = self.make_action(
            search_space={"chunker.chunk_size": [400]},
        )
        request = self.make_request(
            optimizer="internal",
            **candidate,
            max_trials=1,
        )

        def runner(_request):
            return InternalAdapterResult(
                request_id="request-1",
                status="skipped",
                error="사전검사 실패",
                metadata={"error_code": "chunk_precheck_failed"},
            )

        result = run(request, backend_runners={"internal": runner})

        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.optimizer, "internal")
        self.assertEqual(result.metadata["error_code"], "chunk_precheck_failed")

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

    # internal × chunker.strategy (범주형 축) ------------------------------
    def make_strategy_request(self, *, values=None, **overrides):
        candidate = self.make_action(
            prescription_id="switch_to_recursive_sentence",
            search_space={
                "chunker.strategy": list(
                    values or ["recursive_sentence", "markdown_recursive"]
                )
            },
            reindex=True,
        )
        values_ = {
            "optimizer": "internal",
            "baseline_config": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "chunk_strategy": "fixed",
            },
            "max_trials": 2,
            **candidate,
        }
        values_.update(overrides)
        return self.make_request(**values_)

    def test_internal_accepts_chunker_strategy(self):
        # 회귀 테스트: 범주형 축이 실행 전에 unsupported_backend_path 로 걸리면
        # rules 가 등록한 2-후보 스윕이 실제 비교 없이 통째로 스킵된다.
        result = run(self.make_strategy_request())

        self.assertEqual(result.status, "proposed")
        self.assertEqual(result.optimizer, "internal")
        self.assertNotEqual(result.metadata.get("error_code"), "unsupported_backend_path")
        self.assertEqual(result.metadata["parameter_path"], "chunker.strategy")
        self.assertEqual(
            result.metadata["filtered_search_space"],
            {"chunker.strategy": ["recursive_sentence", "markdown_recursive"]},
        )
        self.assertEqual(
            result.config_patch.changes,
            {"chunker.strategy": "recursive_sentence"},
        )

    def test_internal_strategy_requires_reindex(self):
        result = run(self.make_strategy_request())

        self.assertTrue(result.needs_reindex)
        self.assertTrue(result.config_patch.reindex_required)

    def test_internal_strategy_drops_values_outside_allowed(self):
        result = run(
            self.make_strategy_request(
                values=["semantic_split", "markdown_recursive"],
            )
        )

        self.assertEqual(
            result.metadata["filtered_search_space"],
            {"chunker.strategy": ["markdown_recursive"]},
        )
        self.assertEqual(
            result.config_patch.changes,
            {"chunker.strategy": "markdown_recursive"},
        )

    def test_internal_strategy_is_blocked_when_capability_is_off(self):
        result = run(
            self.make_strategy_request(
                metadata={"capabilities": {"chunking_strategy": False}},
            )
        )

        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.metadata["error_code"], "unsupported_capability")

    def test_internal_strategy_keeps_single_axis_guarantee(self):
        candidate = self.make_action(
            prescription_id="switch_to_recursive_sentence",
            search_space={
                "chunker.strategy": ["recursive_sentence"],
                "chunker.chunk_size": [600],
            },
        )
        request = self.make_request(
            optimizer="internal",
            baseline_config={"chunk_size": 512, "chunk_strategy": "fixed"},
            **candidate,
            max_trials=2,
        )

        result = run(request)

        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.metadata["error_code"], "multi_axis_search_space")

    def test_internal_strategy_skips_candidate_equal_to_current_value(self):
        # 현재값 재적용은 no-op 이라 정규화 단계에서 제외된다. 후보가 그것뿐이면
        # 실행 가능한 후보가 없다는 뜻이다.
        result = run(
            self.make_strategy_request(
                values=["markdown_recursive"],
                baseline_config={"chunk_strategy": "markdown_recursive"},
            )
        )

        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.metadata["error_code"], "no_valid_candidate_values")

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
