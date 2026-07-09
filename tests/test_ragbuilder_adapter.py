import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.optimize.adapters.ragbuilder_adapter import RAGBuilderAdapter
from agents.optimize.schemas import OptimizationRequest, PrescriptionCandidate


class RAGBuilderAdapterTest(unittest.TestCase):
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
            metadata={"ragbuilder_client": fake_client},
        )

        result = RAGBuilderAdapter().run(request)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.best_config, {"retriever.top_k": 8})
        self.assertEqual(result.best_score, 0.81)
        self.assertEqual(result.trial_results[0].config, {"retriever.top_k": 8})


if __name__ == "__main__":
    unittest.main()
