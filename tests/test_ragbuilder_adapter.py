import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.optimize.adapters.ragbuilder_adapter import RAGBuilderAdapter
from agents.optimize.schemas import ConfigPatch, OptimizationRequest, PrescriptionCandidate


class RAGBuilderAdapterTest(unittest.TestCase):
    def make_request(self, **overrides):
        values = {
            "request_id": "req-ragbuilder",
            "iteration": 1,
            "baseline_config": {
                "top_k": 3,
                "chunk_size": 512,
                "chunk_overlap": 50,
                "input_source": "sample_data.txt",
            },
            "failure_label": "retrieval_missing_gold",
            "search_space": {"retriever.top_k": [5, 7]},
            "target_metrics": ["context_recall"],
            "metadata": {"use_mock": True},
        }
        values.update(overrides)
        return OptimizationRequest(**values)

    def test_mock_run_returns_normalized_result(self):
        request = OptimizationRequest(
            request_id="req-ragbuilder-mock",
            iteration=1,
            baseline_config={
                "top_k": 3,
                "chunk_size": 512,
                "chunk_overlap": 50,
                "input_source": "sample_data.txt",
            },
            failure_label="retrieval_missing_gold",
            candidates=[
                PrescriptionCandidate(
                    id="increase_top_k",
                    failure_label="retrieval_missing_gold",
                    group="A",
                    status="ready",
                )
            ],
            search_space={"retriever.top_k": [5, 7, 9]},
            target_metrics=["context_recall"],
            max_trials=3,
            metadata={"use_mock": True},
        )

        result = RAGBuilderAdapter().run(request)

        self.assertEqual(result.status, "completed")
        self.assertTrue(result.metadata["is_mock"])
        self.assertEqual(result.best_config, {"retriever.top_k": 5})
        self.assertEqual(len(result.trial_results), 3)
        self.assertEqual(result.search_space["retriever.top_k"], [5, 7, 9])

    def test_client_result_is_converted_to_agentdoctor_paths(self):
        def fake_client(payload):
            return {
                "best_config": {"retrieval.top_k": 8},
                "best_score": 0.81,
                "trial_results": [
                    {
                        "trial_id": "trial-1",
                        "config": {"retrieval.top_k": 8},
                        "score": 0.81,
                    }
                ],
            }

        request = OptimizationRequest(
            request_id="req-ragbuilder-client",
            iteration=1,
            baseline_config={"top_k": 4, "input_source": "sample_data.txt"},
            failure_label="retrieval_missing_gold",
            candidates=[
                PrescriptionCandidate(
                    id="increase_top_k",
                    failure_label="retrieval_missing_gold",
                    group="A",
                    status="ready",
                )
            ],
            search_space={"retriever.top_k": [6, 8]},
            metadata={"ragbuilder_client": fake_client},
        )

        result = RAGBuilderAdapter().run(request)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.best_config, {"retriever.top_k": 8})
        self.assertEqual(result.best_score, 0.81)
        self.assertEqual(result.trial_results[0].config, {"retriever.top_k": 8})

    def test_empty_search_space_is_skipped_without_reading_candidates(self):
        request = self.make_request(
            search_space={},
            candidates=[
                PrescriptionCandidate(
                    id="increase_top_k",
                    failure_label="retrieval_missing_gold",
                    group="A",
                    status="ready",
                    patch=ConfigPatch(changes={"retriever.top_k": 8}),
                )
            ],
        )

        result = RAGBuilderAdapter().run(request)

        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.metadata["error_code"], "empty_search_space")

    def test_unsupported_path_fails_before_execution(self):
        request = self.make_request(search_space={"generation.temperature": [0.0]})

        result = RAGBuilderAdapter().run(request)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metadata["error_code"], "unsupported_config_path")

    def test_mixed_stage_search_space_is_rejected(self):
        request = self.make_request(
            search_space={
                "retriever.top_k": [5],
                "chunker.chunk_size": [400],
            }
        )

        result = RAGBuilderAdapter().run(request)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metadata["error_code"], "mixed_optimization_stage")

    def test_missing_input_source_fails_preflight(self):
        request = self.make_request(baseline_config={"top_k": 3})

        result = RAGBuilderAdapter().run(request)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metadata["error_code"], "missing_input_source")

    def test_zero_score_and_nested_config_are_preserved(self):
        def fake_client(payload):
            return {
                "best_config": {"retrieval": {"top_k": 5}},
                "best_score": 0.0,
                "trial_results": [
                    {
                        "id": 0,
                        "params": {"retrieval": {"top_k": 5}},
                        "value": 0.0,
                    }
                ],
            }

        request = self.make_request(
            search_space={"retriever.top_k": [5]},
            metadata={"ragbuilder_client": fake_client},
        )

        result = RAGBuilderAdapter().run(request)

        self.assertEqual(result.best_score, 0.0)
        self.assertEqual(result.best_config, {"retriever.top_k": 5})
        self.assertEqual(result.trial_results[0].score, 0.0)
        self.assertEqual(result.trial_results[0].trial_id, "0")

    def test_missing_optimized_path_marks_trial_unsupported(self):
        def fake_client(payload):
            return {
                "trial_results": [
                    {
                        "trial_id": "trial-invalid",
                        "config": {"unrelated": "value"},
                        "score": 0.9,
                    }
                ]
            }

        request = self.make_request(metadata={"ragbuilder_client": fake_client})

        result = RAGBuilderAdapter().run(request)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.trial_results[0].status, "rejected")
        self.assertIn(
            "missing_optimized_path:retriever.top_k",
            result.trial_results[0].unsupported_reasons,
        )

    def test_client_exception_returns_standard_failure(self):
        def fake_client(payload):
            raise RuntimeError("client exploded")

        request = self.make_request(metadata={"ragbuilder_client": fake_client})

        result = RAGBuilderAdapter().run(request)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metadata["error_code"], "ragbuilder_execution_failed")
        self.assertEqual(result.error, "client exploded")

    def test_empty_external_result_is_failed(self):
        request = self.make_request(metadata={"ragbuilder_client": lambda payload: {}})

        result = RAGBuilderAdapter().run(request)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metadata["error_code"], "invalid_result_shape")

    def test_minimize_direction_uses_lowest_trial_when_best_is_missing(self):
        def fake_client(payload):
            return {
                "trial_results": [
                    {"config": {"retrieval.top_k": 5}, "score": 0.4},
                    {"config": {"retrieval.top_k": 7}, "score": 0.2},
                ]
            }

        request = self.make_request(
            metadata={
                "ragbuilder_client": fake_client,
                "optimization_direction": "minimize",
            }
        )

        result = RAGBuilderAdapter().run(request)

        self.assertEqual(result.best_config, {"retriever.top_k": 7})
        self.assertEqual(result.best_score, 0.2)

    def test_failed_trial_score_is_not_used_as_best_score(self):
        def fake_client(payload):
            return {
                "trial_results": [
                    {
                        "config": {"retrieval.top_k": 5},
                        "score": 0.5,
                        "status": "completed",
                    },
                    {
                        "config": {"retrieval.top_k": 7},
                        "score": 0.9,
                        "status": "failed",
                    },
                ]
            }

        request = self.make_request(metadata={"ragbuilder_client": fake_client})

        result = RAGBuilderAdapter().run(request)

        self.assertEqual(result.best_config, {"retriever.top_k": 5})
        self.assertEqual(result.best_score, 0.5)

    def test_unsupported_objective_metric_is_rejected(self):
        request = self.make_request(target_metrics=["faithfulness"])

        result = RAGBuilderAdapter().run(request)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metadata["error_code"], "unsupported_objective_metric")

    def test_data_ingest_config_matches_ragbuilder_016_shapes(self):
        class CapturingConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class ContractAdapter(RAGBuilderAdapter):
            def _load_ragbuilder_config_class(self, name):
                if name == "DataIngestOptionsConfig":
                    return CapturingConfig
                return None

        adapter = ContractAdapter()
        request = self.make_request(
            baseline_config={
                "chunk_size": 512,
                "chunk_overlap": 50,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "input_source": "sample_data.txt",
            },
            search_space={"chunker.chunk_size": [400, 600, 800]},
            metadata={"eval_dataset": "eval.csv"},
        )
        mapping = adapter.build_mapping(request)
        payload = adapter.build_payload(request, mapping)

        config = adapter._build_data_ingest_config(payload, request)

        self.assertEqual(
            config.kwargs["chunk_size"],
            {"min": 400, "max": 800, "stepsize": 200},
        )
        self.assertEqual(config.kwargs["chunk_overlap"], [50])
        self.assertEqual(
            config.kwargs["embedding_models"],
            [
                {
                    "type": "huggingface",
                    "model_kwargs": {
                        "model_name": "sentence-transformers/all-MiniLM-L6-v2"
                    },
                }
            ],
        )
        self.assertEqual(
            config.kwargs["evaluation_config"]["test_dataset"],
            "eval.csv",
        )
        self.assertEqual(config.kwargs["optimization"]["n_trials"], 1)

    def test_agentdoctor_embedding_uri_maps_to_same_ragbuilder_model(self):
        adapter = RAGBuilderAdapter()
        request = self.make_request()

        options = adapter._embedding_configs_for_model(
            "openai://text-embedding-3-small",
            request,
        )

        self.assertEqual(
            options,
            [
                {
                    "type": "openai",
                    "model_kwargs": {"model": "text-embedding-3-small"},
                }
            ],
        )

    def test_retrieval_runs_ingest_then_reranker_off_and_on(self):
        class NativeResult:
            def __init__(self, enabled):
                self.best_config = {
                    "retrievers": [{"type": "vector_similarity"}],
                    "rerankers": ([{"type": "BAAI/bge-reranker-base"}] if enabled else []),
                    "top_k": 5,
                }
                self.best_score = 0.9 if enabled else 0.8
                self.avg_latency = 10.0
                self.error_rate = 0.0

        class FakeBuilder:
            def __init__(self):
                self.ingest_calls = 0
                self.retrieval_calls = []

            def optimize_data_ingest(self):
                self.ingest_calls += 1

            def optimize_retrieval(self, config):
                self.retrieval_calls.append(config)
                return NativeResult(config["reranker_enabled"])

        class ContractAdapter(RAGBuilderAdapter):
            def _build_retrieval_config(
                self,
                payload,
                reranker_enabled,
                trial_budget=None,
            ):
                return {
                    "reranker_enabled": reranker_enabled,
                    "trial_budget": trial_budget,
                }

        adapter = ContractAdapter()
        request = self.make_request(
            search_space={"reranker.enabled": [False, True]},
            max_trials=4,
            metadata={"eval_dataset": "eval.csv"},
        )
        mapping = adapter.build_mapping(request)
        payload = adapter.build_payload(request, mapping)
        builder = FakeBuilder()

        raw_result = adapter._run_builder(builder, payload, request)

        self.assertEqual(builder.ingest_calls, 1)
        self.assertEqual(
            builder.retrieval_calls,
            [
                {"reranker_enabled": False, "trial_budget": 2},
                {"reranker_enabled": True, "trial_budget": 2},
            ],
        )
        self.assertEqual(raw_result["best_score"], 0.9)
        self.assertEqual(len(raw_result["trial_results"]), 2)

    def test_reranker_split_requires_two_trials(self):
        request = self.make_request(
            search_space={"reranker.enabled": [False, True]},
            max_trials=1,
        )

        result = RAGBuilderAdapter().run(request)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metadata["error_code"], "insufficient_trial_budget")

    def test_native_retriever_list_is_converted_to_hybrid(self):
        def fake_client(payload):
            return {
                "best_config": {
                    "retrievers": [
                        {"type": "vector_similarity"},
                        {"type": "bm25"},
                    ]
                },
                "best_score": 0.7,
            }

        request = self.make_request(
            search_space={"retriever.search_type": ["hybrid"]},
            metadata={"ragbuilder_client": fake_client},
        )

        result = RAGBuilderAdapter().run(request)

        self.assertEqual(
            result.best_config,
            {"retriever.search_type": "hybrid"},
        )


if __name__ == "__main__":
    unittest.main()
