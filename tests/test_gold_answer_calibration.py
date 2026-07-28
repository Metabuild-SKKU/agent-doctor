import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import Probe
from agents.eval.gold_answer import (
    calibrated_answer_score,
    gold_answer_calibrated_match,
    gold_answer_overlap,
)
from agents.eval.metrics_basic import _compute_metrics, best_answer_match, gold_answer_variants
from agents.eval.scoring import reliability_score
from agents.eval.types import EvalRecord, F1_PASS_THRESHOLD


def _record(*, recall=1.0, f1=0.29, faith=1.0, rel=0.97):
    probe = Probe(
        probe_id="p1",
        question="What are the asset and liability totals?",
        source="unit",
        gold_chunk_ids=["g_a"],
        ground_truth="asset_total 49,157,964,024 and liability_total 32,695,236,480",
    )
    record = EvalRecord(
        probe=probe,
        retrieved_chunk_ids=["g_a"],
        generated_answer=(
            "The report says asset_total is 49,157,964,024 and "
            "liability_total is 32,695,236,480."
        ),
    )
    record.recall_at_k = recall
    record.f1_score = f1
    record.ragas = {"faithfulness": faith, "response_relevancy": rel}
    record.ragas_done = True
    return record


class GoldAnswerCalibrationTest(unittest.TestCase):
    def test_numeric_match_lifts_low_f1_grounded_answer(self):
        record = _record()
        self.assertTrue(gold_answer_calibrated_match(record))
        self.assertEqual(calibrated_answer_score(record), F1_PASS_THRESHOLD)
        self.assertEqual(reliability_score([record]), F1_PASS_THRESHOLD)

    def test_complete_retrieval_is_required(self):
        record = _record(recall=0.5)
        self.assertFalse(gold_answer_calibrated_match(record))
        self.assertEqual(calibrated_answer_score(record), 0.29)

    def test_overlap_reports_numeric_and_keyword_recall(self):
        overlap = gold_answer_overlap(
            "asset_total 49,157,964,024 and liability_total 32,695,236,480",
            "liability_total is 32,695,236,480; asset_total is 49,157,964,024.",
        )
        self.assertEqual(overlap["numeric_recall"], 1.0)
        self.assertGreaterEqual(overlap["keyword_recall"], 0.65)

    def test_best_answer_match_uses_safe_gold_variants(self):
        reference = "자산총계 49,157,964,024"
        prediction = "총자산은 49157964024입니다."
        best, variant, raw, count = best_answer_match(prediction, reference)
        self.assertGreater(best, raw)
        self.assertNotEqual(variant, reference)
        self.assertGreaterEqual(count, 2)

    def test_compute_metrics_keeps_raw_and_best_variant_f1(self):
        record = EvalRecord(
            probe=Probe(
                probe_id="p2",
                question="자산총계는 얼마인가요?",
                source="unit",
                gold_chunk_ids=["g_a"],
                ground_truth="자산총계 49,157,964,024",
            ),
            retrieved_chunk_ids=["g_a"],
            generated_answer="총자산은 49157964024입니다.",
            oracle_answer="자산총계는 49,157,964,024입니다.",
        )
        _compute_metrics(record)
        self.assertGreater(record.f1_score, record.raw_f1_score)
        self.assertNotEqual(record.best_gold_answer, record.probe.ground_truth)
        self.assertGreaterEqual(record.gold_answer_variant_count, 2)

    def test_gold_variants_stay_specific(self):
        variants = gold_answer_variants("자산총계 49,157,964,024")
        self.assertNotIn("보고서에 금액이 있습니다.", variants)
        self.assertTrue(any("49,157,964,024" in item or "49157964024" in item for item in variants))


if __name__ == "__main__":
    unittest.main()
