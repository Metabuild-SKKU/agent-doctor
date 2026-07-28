import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import Probe
from agents.eval import diagnose
from agents.eval.metrics_basic import _compute_metrics, best_answer_match, gold_answer_variants
from agents.eval.scoring import reliability_score
from agents.eval.types import EvalRecord


class GoldAnswerCalibrationTest(unittest.TestCase):
    def test_best_answer_match_observes_safe_gold_variants(self):
        reference = "asset_total 49,157,964,024"
        prediction = "asset_total is 49157964024."

        best, variant, raw, count = best_answer_match(prediction, reference)

        self.assertGreaterEqual(best, raw)
        self.assertTrue(variant)
        self.assertGreaterEqual(count, 2)

    def test_sentence_boilerplate_variant_is_not_added(self):
        variants = gold_answer_variants("asset_total 49,157,964,024")

        self.assertFalse(any(item.endswith("입니다.") for item in variants))
        self.assertFalse(any(item.endswith("is.") for item in variants))

    def test_compute_metrics_keeps_gate_f1_raw(self):
        record = EvalRecord(
            probe=Probe(
                probe_id="p2",
                question="What is the asset total?",
                source="unit",
                gold_chunk_ids=["g_a"],
                ground_truth="asset_total 49,157,964,024",
            ),
            retrieved_chunk_ids=["g_a"],
            generated_answer="asset_total is 49157964024.",
            oracle_answer="asset_total is 49157964024.",
        )

        _compute_metrics(record)

        self.assertEqual(record.f1_score, record.raw_f1_score)
        self.assertGreaterEqual(record.best_gold_answer_f1, record.raw_f1_score)
        self.assertEqual(record.oracle_f1, record.raw_oracle_f1)
        self.assertGreaterEqual(record.best_oracle_gold_answer_f1, record.raw_oracle_f1)

    def test_low_f1_answer_does_not_pass_gate_by_numeric_overlap(self):
        record = EvalRecord(
            probe=Probe(
                probe_id="p3",
                question="What is the asset total?",
                source="unit",
                gold_chunk_ids=["g_a"],
                ground_truth="asset_total 49,157,964,024",
            ),
            retrieved_chunk_ids=["g_a"],
            generated_answer="liability_total is 49,157,964,024.",
        )
        record.recall_at_k = 1.0
        record.f1_score = 0.29
        record.ragas = {"faithfulness": 1.0, "response_relevancy": 0.97}
        record.ragas_done = True

        self.assertFalse(diagnose._f1_ok(record))
        self.assertLess(reliability_score([record]), 0.5)


if __name__ == "__main__":
    unittest.main()
