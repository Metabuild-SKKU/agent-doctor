import csv
import tempfile
import unittest
from pathlib import Path

from tools.eval_ablation_report import build_markdown, parse_log_text, write_csv


SAMPLE_LOG = """
[Eval] STEP5 report - probe 30, failed 10, overall=0.6501, pass=False
[Eval] score 65/100 (quality 92 / reliability 50) - overall 0.6501 - pass false
[Eval] weighted label distribution {'retrieval_low_rank': 6.0, 'bad_gold_answer': 2.0}
search: mode=hybrid, search_fallback=false, reranker=applied
recall@k=0.00  f1=0.44  oracle_f1=0.66
! retrieval_low_rank: missed_gold_ranks=[chunk_041:40] > top_k=5, recall@k=0.00
[Optimize] prescription applied: id=increase_chunk_size, label=chunking_context_mismatch
[Optimize] before config: chunk_size=512
[Optimize] after config: chunk_size=1024
[Optimize] previous verdict: KEEP prescription=increase_chunk_size, before=0.65, after=0.79
[Optimize] previous verdict: ROLLBACK prescription=enable_reranker, before=0.41, after=0.46
[Optimize] before config: use_reranker=True
[Optimize] after config: use_reranker=False
[Eval] STEP5 report - probe 30, failed 8, overall=0.7901, pass=True
[Eval] score 79/100 (quality 94 / reliability 65) - overall 0.7901 - pass true
[Eval] weighted label distribution {'retrieval_low_rank': 4.0, 'context_noise_interference': 2.0}
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

        self.assertEqual(parsed.actions[1].verdict, "KEEP")
        self.assertEqual(parsed.actions[1].verdict_before, 0.65)
        self.assertEqual(parsed.actions[1].verdict_after, 0.79)
        self.assertEqual(parsed.actions[2].verdict, "ROLLBACK")
        self.assertEqual(parsed.actions[2].canonical_action, "use_reranker:enable")

    def test_build_markdown_mentions_action_centered_summary(self):
        parsed = parse_log_text(SAMPLE_LOG, source="sample.log")

        markdown = build_markdown([parsed])

        self.assertIn("65 -> 79", markdown)
        self.assertIn("Action-Centered Summary", markdown)
        self.assertIn("chunk_size:increase", markdown)
        self.assertIn("Issue #67", markdown)

    def test_write_csv_exports_eval_rows(self):
        parsed = parse_log_text(SAMPLE_LOG, source="sample.log")

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "summary.csv"
            write_csv([parsed], output)
            with output.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["score"], "65")
        self.assertEqual(rows[0]["retrieval_low_rank_weight"], "6.0")
        self.assertEqual(rows[1]["context_noise_interference_weight"], "2.0")


if __name__ == "__main__":
    unittest.main()
