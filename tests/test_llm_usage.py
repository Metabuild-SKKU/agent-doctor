import os
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import llm_usage
from core.llm_usage import _estimate_cost_usd


class EstimateCostUsdTest(unittest.TestCase):
    def test_known_model_prefix_match(self):
        cost = _estimate_cost_usd("gpt-4o-mini", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 0.15 + 0.60)

    def test_longest_prefix_wins_over_shorter_alias(self):
        cost = _estimate_cost_usd("gemini-3.1-flash-lite", 1_000_000, 0)
        self.assertAlmostEqual(cost, 0.25)

    def test_github_models_publisher_slash_model_is_free(self):
        self.assertEqual(_estimate_cost_usd("openai/gpt-4o-mini", 1_000_000, 1_000_000), 0.0)

    def test_unregistered_model_returns_none(self):
        self.assertIsNone(_estimate_cost_usd("some-unlisted-model", 1_000, 1_000))

    def test_zero_tokens_known_model_is_zero_cost(self):
        self.assertEqual(_estimate_cost_usd("gpt-4o", 0, 0), 0.0)


class StageSummaryTest(unittest.TestCase):
    def setUp(self):
        with llm_usage._lock:
            llm_usage._totals.update(
                {"calls": 0, "prompt": 0, "output": 0, "cost": 0.0}
            )

    def tearDown(self):
        with llm_usage._lock:
            llm_usage._totals.update(
                {"calls": 0, "prompt": 0, "output": 0, "cost": 0.0}
            )

    def test_individual_calls_are_silent_and_stage_summary_is_one_line(self):
        started = llm_usage.snapshot_usage()
        buf = StringIO()
        with redirect_stdout(buf):
            llm_usage.log_usage("gpt-4o-mini", 1_000, 100, tag="Eval")
            llm_usage.log_usage("gpt-4o-mini", 2_000, 200, tag="Eval")
            llm_usage.print_summary(
                tag="Eval",
                stage="STEP2 답변 생성",
                since=started,
            )

        lines = buf.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn("STEP2 답변 생성: 호출 2회", lines[0])
        self.assertIn("누적 호출 2회", lines[0])
        self.assertIn("비용 ≈", lines[0])
        self.assertNotIn("토큰", lines[0])

    def test_summary_reports_stage_delta_and_process_total(self):
        llm_usage.log_usage("gpt-4o-mini", 1_000, 100)
        started = llm_usage.snapshot_usage()
        llm_usage.log_usage("gpt-4o-mini", 2_000, 200)

        buf = StringIO()
        with redirect_stdout(buf):
            llm_usage.print_summary(tag="Eval", stage="RAGAS", since=started)

        line = buf.getvalue()
        self.assertIn("RAGAS: 호출 1회", line)
        self.assertIn("누적 호출 2회", line)


if __name__ == "__main__":
    unittest.main()
