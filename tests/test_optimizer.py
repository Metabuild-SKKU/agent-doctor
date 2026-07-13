import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.optimize.optimizer import (
    filter_candidate_values,
    is_capability_supported,
    merge_constraints,
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

    def test_merge_constraints_accepts_flat_alias(self):
        constraints = merge_constraints({"top_k": {"max": 8}})

        self.assertEqual(constraints["retriever.top_k"]["max"], 8)


if __name__ == "__main__":
    unittest.main()
