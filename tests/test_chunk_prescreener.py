import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import Document
from agents.optimize.adapters.chunk_prescreener import run
from agents.optimize.schemas import OptimizationRequest


def fixed_previewer(document, chunk_size, chunk_overlap, _strategy):
    spans = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(document.content), step):
        end = min(start + chunk_size, len(document.content))
        spans.append((start, end))
        if end >= len(document.content):
            break
    return spans


class ChunkPrescreenerTest(unittest.TestCase):
    def make_request(self, *, path, values, baseline, spans):
        document = Document(
            doc_id="doc-1",
            content="가" * 1000,
            source="memory",
            format="text",
        )
        return OptimizationRequest(
            request_id="chunk-precheck",
            iteration=0,
            baseline_config=baseline,
            failure_label="chunk_test",
            search_space={path: values},
            optimizer="internal",
            max_trials=len(values),
            metadata={
                "chunk_precheck_context": {
                    "documents": [document],
                    "gold_spans": spans,
                    "chunk_strategy": "fixed",
                }
            },
        )

    def test_chunk_size_selects_containing_candidate_with_less_waste(self):
        request = self.make_request(
            path="chunker.chunk_size",
            values=[300, 400, 600],
            baseline={"chunk_size": 512, "chunk_overlap": 0},
            spans=[
                {"doc_id": "doc-1", "start": 100, "end": 350},
                {"doc_id": "doc-1", "start": 600, "end": 750},
            ],
        )

        result = run(request, previewer=fixed_previewer)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.best_config, {"chunker.chunk_size": 400})
        self.assertTrue(result.metadata["proxy_only"])
        self.assertFalse(result.metadata["best_is_baseline"])

    def test_chunk_overlap_selects_smallest_boundary_recovery(self):
        request = self.make_request(
            path="chunker.chunk_overlap",
            values=[100, 150],
            baseline={"chunk_size": 400, "chunk_overlap": 0},
            spans=[
                {"doc_id": "doc-1", "start": 350, "end": 450},
            ],
        )

        result = run(request, previewer=fixed_previewer)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.best_config, {"chunker.chunk_overlap": 100})
        metrics = result.metadata["candidate_metrics"]
        overlap_100 = next(item for item in metrics if item["value"] == 100)
        self.assertEqual(overlap_100["full_span_containment"], 1.0)
        self.assertEqual(overlap_100["boundary_recovery_rate"], 1.0)
        self.assertEqual(overlap_100["unrecovered_cut_rate"], 0.0)

    def test_missing_gold_spans_skips_automatic_precheck(self):
        request = self.make_request(
            path="chunker.chunk_size",
            values=[400, 600],
            baseline={"chunk_size": 512, "chunk_overlap": 50},
            spans=[],
        )

        result = run(request, previewer=fixed_previewer)

        self.assertEqual(result.status, "skipped")
        self.assertEqual(
            result.metadata["error_code"],
            "missing_chunk_precheck_context",
        )


if __name__ == "__main__":
    unittest.main()
