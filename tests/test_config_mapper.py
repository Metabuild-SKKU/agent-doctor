import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.optimize.config_mapper import map_prescriptions_to_config


class ConfigMapperTest(unittest.TestCase):
    def test_increase_top_k_uses_current_value(self):
        result = map_prescriptions_to_config(
            ["increase_top_k"],
            {"top_k": 4},
        )

        self.assertEqual(result.search_space["retriever.top_k"], [6, 8, 10])
        self.assertEqual(
            [patch.changes for patch in result.patches],
            [
                {"retriever.top_k": 6},
                {"retriever.top_k": 8},
                {"retriever.top_k": 10},
            ],
        )
        self.assertEqual(
            [patch.metadata["prescription_ids"] for patch in result.patches],
            [["increase_top_k"], ["increase_top_k"], ["increase_top_k"]],
        )

    def test_rules_ids_are_primary_inputs(self):
        result = map_prescriptions_to_config(
            ["enable_hybrid", "context_compression", "shrink_chunk_size"],
            {"use_hybrid": False, "context_compression": False, "chunk_size": 500},
        )

        self.assertEqual(result.search_space["retriever.search_type"], ["hybrid"])
        self.assertEqual(result.search_space["context.compression.enabled"], [True])
        self.assertEqual(result.search_space["chunker.chunk_size"], [350, 250])

    def test_capability_skip(self):
        result = map_prescriptions_to_config(
            ["enable_reranker"],
            {"use_reranker": False},
            capabilities={"reranker": False},
        )

        self.assertEqual(result.patches, [])
        self.assertEqual(result.skipped[0].reason, "unsupported_capability")

    def test_constraints_filter_candidates(self):
        result = map_prescriptions_to_config(
            ["increase_top_k"],
            {"top_k": 4},
            constraints={"retriever.top_k": {"min": 1, "max": 8}},
        )

        self.assertEqual(result.search_space["retriever.top_k"], [6, 8])

    def test_conflicting_directions_are_skipped(self):
        result = map_prescriptions_to_config(
            ["increase_top_k", "decrease_top_k"],
            {"top_k": 4},
        )

        self.assertEqual(result.patches, [])
        self.assertEqual(
            sorted(item.prescription_id for item in result.skipped),
            ["decrease_top_k", "increase_top_k"],
        )
        self.assertTrue(result.warnings)

    def test_same_direction_top_k_prescriptions_are_not_conflicts(self):
        result = map_prescriptions_to_config(
            ["increase_top_k", "dynamic_top_k"],
            {"top_k": 4},
        )

        self.assertEqual(result.search_space["retriever.top_k"], [6, 8, 10])
        self.assertEqual(len(result.patches), 3)
        self.assertEqual(
            [patch.metadata["prescription_ids"] for patch in result.patches],
            [
                ["increase_top_k", "dynamic_top_k"],
                ["increase_top_k", "dynamic_top_k"],
                ["increase_top_k", "dynamic_top_k"],
            ],
        )

    def test_unknown_prescription_is_skipped_without_aliasing(self):
        result = map_prescriptions_to_config(
            ["enable_hybrid_search"],
            {"use_hybrid": False},
        )

        self.assertEqual(result.patches, [])
        self.assertEqual(result.skipped[0].prescription_id, "enable_hybrid_search")
        self.assertEqual(result.skipped[0].reason, "unsupported_prescription")


if __name__ == "__main__":
    unittest.main()
