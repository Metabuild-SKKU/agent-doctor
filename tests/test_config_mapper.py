import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.optimize.config_mapper import (
    apply_best_config,
    apply_config_patch,
    get_current_value,
    map_changes_to_index_config,
)
from agents.optimize.schemas import ConfigPatch


class ConfigMapperTest(unittest.TestCase):
    def test_get_current_value_reads_flat_aliases(self):
        current = {
            "top_k": 4,
            "use_hybrid": True,
            "use_reranker": False,
            "rerank_candidates": 20,
            "chunk_size": 512,
        }

        self.assertEqual(get_current_value(current, "retriever.top_k"), 4)
        self.assertEqual(get_current_value(current, "retriever.search_type"), "hybrid")
        self.assertFalse(get_current_value(current, "reranker.enabled"))
        self.assertEqual(
            get_current_value(current, "reranker.candidate_count"),
            20,
        )
        self.assertEqual(get_current_value(current, "chunker.chunk_size"), 512)

    def test_map_changes_to_index_config_translates_canonical_paths(self):
        mapped, ignored, warnings = map_changes_to_index_config(
            {
                "retriever.top_k": 8,
                "retriever.search_type": "hybrid",
                "reranker.enabled": True,
                "reranker.candidate_count": 40,
                "chunker.chunk_size": 400,
            }
        )

        self.assertEqual(
            mapped,
            {
                "top_k": 8,
                "use_hybrid": True,
                "use_reranker": True,
                "rerank_candidates": 40,
                "chunk_size": 400,
            },
        )
        self.assertEqual(ignored, [])
        self.assertEqual(warnings, [])

    def test_apply_config_patch_mutates_index_config_and_returns_diff(self):
        index_config = {"top_k": 4, "chunk_size": 512, "use_hybrid": False}
        patch = ConfigPatch(
            changes={
                "retriever.top_k": 8,
                "retriever.search_type": "hybrid",
                "reranker.enabled": True,
            },
            metadata={"source": "test"},
        )

        diff = apply_config_patch(index_config, patch)

        self.assertEqual(index_config["top_k"], 8)
        self.assertEqual(index_config["use_hybrid"], True)
        self.assertEqual(index_config["use_reranker"], True)
        self.assertNotIn("reranker.enabled", index_config)
        self.assertEqual(
            diff.changed_keys,
            ["top_k", "use_hybrid"],
        )
        self.assertEqual(diff.added_keys, ["use_reranker"])
        self.assertEqual(diff.ignored_keys, [])
        self.assertFalse(diff.warnings)
        self.assertEqual(diff.metadata["source"], "test")

    def test_apply_best_config_can_run_without_mutation(self):
        index_config = {"chunk_overlap": 50}

        diff = apply_best_config(
            index_config,
            {"chunker.chunk_overlap": 100},
            mutate=False,
        )

        self.assertEqual(index_config, {"chunk_overlap": 50})
        self.assertEqual(diff.after_config, {"chunk_overlap": 100})
        self.assertEqual(diff.changed_keys, ["chunk_overlap"])

    def test_non_index_config_target_is_ignored(self):
        index_config = {"chunk_size": 512}
        patch = ConfigPatch(
            changes={"temperature": 0.0},
            target="generation_config",
        )

        diff = apply_config_patch(index_config, patch)

        self.assertEqual(index_config, {"chunk_size": 512})
        self.assertEqual(diff.ignored_keys, ["temperature"])
        self.assertTrue(diff.warnings)


if __name__ == "__main__":
    unittest.main()
