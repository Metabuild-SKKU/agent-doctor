import csv
import tempfile
import unittest
from pathlib import Path

from tools.eval_ablation_report import (
    PRESCRIPTION_ACTIONS,
    _parse_config_assignments,
    build_markdown,
    parse_log_text,
    write_csv,
)


SAMPLE_LOG = """
  종합점수 65/100 (품질 92 / 신뢰도 50) · overall 0.6501 · pass ✗
  가중 라벨분포 {'retrieval_low_rank': 6.0, 'bad_gold_answer': 2.0}
search: mode=hybrid, search_fallback=false, reranker=applied
recall@k=0.00  f1=0.44  oracle_f1=0.66
! retrieval_low_rank: missed_gold_ranks=[chunk_041:40] > top_k=5, recall@k=0.00
[Optimize] 처방 적용: id=increase_chunk_size, label=chunking_context_mismatch
[Optimize] 변경 전 config: chunk_size=512
[Optimize] 변경 후 config: chunk_size=1024
[Optimize] 이전 처방 판정: KEEP, prescription=increase_chunk_size, before=0.65, after=0.79
[Optimize] 이전 처방 판정: ROLLBACK, prescription=enable_reranker, before=0.41, after=0.46
[Optimize] 변경 전 config: use_reranker=True
[Optimize] 변경 후 config: use_reranker=False
  종합점수 79/100 (품질 94 / 신뢰도 65) · overall 0.7901 · pass ✓
  가중 라벨분포 {'retrieval_low_rank': 4.0, 'context_noise_interference': 2.0}
"""

MULTI_KEY_CONFIG_LOG = """
[Optimize] 처방 적용: id=custom_rerank_tuning, label=reranker_low_precision
[Optimize] 변경 전 config: reranker_threshold=0.2, rerank_candidates=30
[Optimize] 변경 후 config: reranker_threshold=0.5, rerank_candidates=60
"""

SWEEP_LOG = """
[Optimize] 처방 적용: id=dynamic_top_k, label=retrieval_low_rank
[Optimize] 변경 전 config: top_k=5
[Optimize] 변경 후 config: top_k=8
[Optimize] 처방 적용: id=dynamic_top_k, label=retrieval_low_rank
[Optimize] 변경 전 config: top_k=8
[Optimize] 변경 후 config: top_k=12
[Optimize] 처방 적용: id=dynamic_top_k, label=retrieval_low_rank
[Optimize] 변경 전 config: top_k=12
[Optimize] 변경 후 config: top_k=16
[Optimize] 선택한 라벨: retrieval_low_rank
[Optimize] 선택한 처방: dynamic_top_k
[Optimize] 변경 전 config: top_k=16
[Optimize] 변경 후 config: top_k=12
"""

ZERO_OVERALL_LOG = """
  종합점수 0/100 (품질 0 / 신뢰도 0) · overall 0.0 · pass ✗
"""

KORQUAD_STYLE_LOG = """
  종합점수 65/100 (품질 92 / 신뢰도 50) · overall 0.9201 · pass ✓
  가중 라벨분포 {'retrieval_low_rank': 10.5, 'generation_misinterpretation': 3.0}
[Optimize] Eval 결과: overall=0.92, composite=65.0, pass=false
[Optimize] 처방 적용: id=shrink_chunk_size, label=retrieval_semantic_mismatch
[Optimize] 변경 전 config: chunk_size=512
[Optimize] 변경 후 config: chunk_size=500
  종합점수 59/100 (품질 88 / 신뢰도 44) · overall 0.8807 · pass ✓
  가중 라벨분포 {'retrieval_low_rank': 10.5, 'bad_gold_answer': 3.5}
[Optimize] Eval 결과: overall=0.88, composite=59.0, pass=false
[Optimize] 이전 처방 판정: ROLLBACK, prescription=shrink_chunk_size, before=0.65, after=0.59
[Optimize] 처방 적용: id=increase_chunk_size, label=chunking_context_mismatch
[Optimize] 변경 전 config: chunk_size=512
[Optimize] 변경 후 config: chunk_size=1024
  종합점수 80/100 (품질 91 / 신뢰도 71) · overall 0.9148 · pass ✓
  가중 라벨분포 {'retrieval_low_rank': 5.0, 'bad_gold_answer': 5.0}
[Optimize] 이전 처방 판정: KEEP, prescription=increase_chunk_size, before=0.65, after=0.80
overall_score  : 0.9148  gate_pass=True
"""


class EvalAblationReportTest(unittest.TestCase):
    def test_parse_log_extracts_eval_timeline_and_bottlenecks(self):
        parsed = parse_log_text(SAMPLE_LOG, source="sample.log")

        self.assertEqual(len(parsed.evals), 2)
        self.assertEqual(parsed.evals[0].composite, 65)
        self.assertEqual(parsed.evals[0].overall, 0.6501)
        self.assertFalse(parsed.evals[0].passed)
        self.assertEqual(parsed.evals[0].label_distribution["retrieval_low_rank"], 6.0)
        self.assertEqual(parsed.evals[1].label_distribution["context_noise_interference"], 2.0)
        self.assertEqual(parsed.counts["recall_zero"], 2)
        self.assertEqual(parsed.counts["retrieval_low_rank"], 1)
        self.assertEqual(parsed.counts["missed_gold_ranks"], 1)
        self.assertEqual(parsed.counts["reranker_applied"], 1)

    def test_parse_log_extracts_optimize_actions_and_canonical_action(self):
        parsed = parse_log_text(SAMPLE_LOG, source="sample.log")

        self.assertEqual(parsed.actions[0].prescription, "increase_chunk_size")
        self.assertEqual(parsed.actions[0].label, "chunking_context_mismatch")
        self.assertEqual(parsed.actions[0].before_config, "chunk_size=512")
        self.assertEqual(parsed.actions[0].after_config, "chunk_size=1024")
        self.assertEqual(parsed.actions[0].canonical_action, "chunk_size:increase")

    def test_parse_log_extracts_verdicts(self):
        parsed = parse_log_text(SAMPLE_LOG, source="sample.log")

        self.assertEqual(parsed.actions[0].verdict, "KEEP")
        self.assertEqual(parsed.actions[0].verdict_before, 0.65)
        self.assertEqual(parsed.actions[0].verdict_after, 0.79)
        self.assertEqual(parsed.actions[1].verdict, "ROLLBACK")
        self.assertEqual(parsed.actions[1].canonical_action, "use_reranker:enable")

    def test_multi_key_config_matches_before_after_by_key(self):
        parsed = parse_log_text(MULTI_KEY_CONFIG_LOG, source="multi.log")

        self.assertEqual(
            parsed.actions[0].canonical_action,
            "reranker_threshold:increase",
        )

    def test_config_assignment_strips_repr_quotes(self):
        self.assertEqual(
            _parse_config_assignments(
                "reranker_model='bge-m3', chunk_strategy=\"fixed\""
            ),
            [("reranker_model", "bge-m3"), ("chunk_strategy", "fixed")],
        )

    def test_prescription_actions_cover_rules_prescription_ids(self):
        from agents.optimize.rules import LABEL_TO_PRESCRIPTIONS

        actual = {
            prescription["id"]
            for rule in LABEL_TO_PRESCRIPTIONS.values()
            for prescription in rule.get("prescriptions", [])
            if prescription.get("id") and prescription.get("patch")
        }

        self.assertEqual(sorted(actual - set(PRESCRIPTION_ACTIONS)), [])
        self.assertNotIn("disable_reranker", PRESCRIPTION_ACTIONS)
        self.assertNotIn("relax_reranker_threshold", PRESCRIPTION_ACTIONS)

    def test_sweep_candidates_are_counted_as_one_action(self):
        parsed = parse_log_text(SWEEP_LOG, source="sweep.log")

        self.assertEqual(len(parsed.actions), 1)
        self.assertEqual(parsed.actions[0].prescription, "dynamic_top_k")
        self.assertEqual(parsed.actions[0].label, "retrieval_low_rank")
        self.assertEqual(parsed.actions[0].before_config, "top_k=16")
        self.assertEqual(parsed.actions[0].after_config, "top_k=12")
        self.assertEqual(parsed.actions[0].verdict, "KEEP")

    def test_sweep_summary_uses_final_selected_config(self):
        parsed = parse_log_text(SWEEP_LOG, source="sweep.log")

        markdown = build_markdown([parsed])

        self.assertIn("| top_k:increase | 1 | 1 | 0 | retrieval_low_rank=1 | - |", markdown)
        self.assertIn(
            "| sweep.log | 0 | retrieval_low_rank | dynamic_top_k | "
            "top_k:increase | top_k=16 | top_k=12 | KEEP |",
            markdown,
        )

    def test_zero_overall_is_not_treated_as_missing(self):
        parsed = parse_log_text(ZERO_OVERALL_LOG, source="zero.log")

        self.assertEqual(parsed.evals[0].overall, 0.0)

    def test_parse_log_overrides_step5_pass_with_optimize_gate(self):
        parsed = parse_log_text(KORQUAD_STYLE_LOG, source="korquad.log")

        self.assertEqual(len(parsed.evals), 3)
        self.assertFalse(parsed.evals[0].passed)
        self.assertFalse(parsed.evals[1].passed)
        self.assertTrue(parsed.evals[2].passed)

    def test_parse_log_attaches_verdict_to_existing_action(self):
        parsed = parse_log_text(KORQUAD_STYLE_LOG, source="korquad.log")

        self.assertEqual(len(parsed.actions), 2)
        self.assertEqual(parsed.actions[0].prescription, "shrink_chunk_size")
        self.assertEqual(parsed.actions[0].verdict, "ROLLBACK")
        self.assertEqual(parsed.actions[0].before_config, "chunk_size=512")
        self.assertEqual(parsed.actions[0].after_config, "chunk_size=500")
        self.assertEqual(parsed.actions[1].prescription, "increase_chunk_size")
        self.assertEqual(parsed.actions[1].verdict, "KEEP")

    def test_build_markdown_summarizes_korquad_actions_once(self):
        parsed = parse_log_text(KORQUAD_STYLE_LOG, source="korquad.log")

        markdown = build_markdown([parsed])

        self.assertIn("| korquad.log | 1 | 65 | 92 | 50 | 0.9201 | false |", markdown)
        self.assertIn("| korquad.log | 3 | 80 | 91 | 71 | 0.9148 | true |", markdown)
        self.assertIn("| chunk_size:decrease | 1 | 0 | 1 | retrieval_semantic_mismatch=1 | -0.06 |", markdown)
        self.assertIn("| chunk_size:increase | 1 | 1 | 0 | chunking_context_mismatch=1 | 0.15 |", markdown)

    def test_build_markdown_mentions_action_centered_summary(self):
        parsed = parse_log_text(SAMPLE_LOG, source="sample.log")

        markdown = build_markdown([parsed])

        self.assertIn("65 -> 79", markdown)
        self.assertIn("Action-Centered Summary", markdown)
        self.assertIn("chunk_size:increase", markdown)
        self.assertIn("Issue #67", markdown)

    def test_write_csv_exports_eval_rows(self):
        parsed = parse_log_text(SAMPLE_LOG, source="sample.log")
        parsed.evals[0].label_distribution["future_label"] = 1.25

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "summary.csv"
            write_csv([parsed], output)
            with output.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 2)
        self.assertIn("future_label_weight", rows[0])
        self.assertEqual(rows[0]["score"], "65")
        self.assertEqual(rows[0]["gate_pass"], "false")
        self.assertEqual(rows[0]["retrieval_low_rank_weight"], "6.0")
        self.assertEqual(rows[0]["future_label_weight"], "1.25")
        self.assertEqual(rows[1]["context_noise_interference_weight"], "2.0")


if __name__ == "__main__":
    unittest.main()
