import os
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch


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

    def test_graph_extraction_default_model_is_priced(self):
        # Index 그래프 기본 모델. 미등록이면 비용이 통째로 집계에서 빠진다.
        self.assertAlmostEqual(_estimate_cost_usd("gpt-4.1-mini", 1_000_000, 1_000_000),
                               0.40 + 1.60)

    def test_reasoning_models_are_priced(self):
        self.assertAlmostEqual(_estimate_cost_usd("gpt-5-mini", 1_000_000, 1_000_000),
                               0.25 + 2.00)
        self.assertAlmostEqual(_estimate_cost_usd("o3-mini", 1_000_000, 0), 1.10)

    def test_pro_variants_do_not_inherit_base_price(self):
        # "o3-pro" 가 "o3" 접두에 걸리면 10배 과소집계된다.
        self.assertAlmostEqual(_estimate_cost_usd("o3-pro", 1_000_000, 0), 20.00)
        self.assertAlmostEqual(_estimate_cost_usd("gpt-5-pro", 0, 1_000_000), 120.00)

    def test_gpt5_chat_is_priced_separately_from_gpt5(self):
        self.assertAlmostEqual(_estimate_cost_usd("gpt-5-chat-latest", 1_000_000, 0), 5.00)
        self.assertAlmostEqual(_estimate_cost_usd("gpt-5", 1_000_000, 0), 1.25)

    def test_zero_tokens_known_model_is_zero_cost(self):
        self.assertEqual(_estimate_cost_usd("gpt-4o", 0, 0), 0.0)


class StageSummaryTest(unittest.TestCase):
    def setUp(self):
        with llm_usage._lock:
            llm_usage._totals.clear()
            llm_usage._totals.update(
                {
                    "calls": 0,
                    "prompt": 0,
                    "output": 0,
                    "cost": 0.0,
                    "unpriced_calls": 0,
                }
            )
            llm_usage._by_agent.clear()

    def tearDown(self):
        with llm_usage._lock:
            llm_usage._totals.clear()
            llm_usage._totals.update(
                {
                    "calls": 0,
                    "prompt": 0,
                    "output": 0,
                    "cost": 0.0,
                    "unpriced_calls": 0,
                }
            )
            llm_usage._by_agent.clear()

    def test_individual_calls_are_silent_and_stage_summary_is_one_line(self):
        started = llm_usage.snapshot_usage()
        buf = StringIO()
        with redirect_stdout(buf):
            llm_usage.log_usage("gpt-4o-mini", 1_000, 100)
            llm_usage.log_usage("gpt-4o-mini", 2_000, 200)
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

    def test_unregistered_model_is_distinguished_from_free_model(self):
        started = llm_usage.snapshot_usage()
        llm_usage.log_usage("some-unlisted-model", 1_000, 100)
        llm_usage.log_usage("openai/gpt-4o-mini", 1_000, 100)

        buf = StringIO()
        with redirect_stdout(buf):
            llm_usage.print_summary(tag="Eval", stage="STEP2 답변 생성", since=started)

        line = buf.getvalue()
        self.assertIn("호출 2회", line)
        self.assertIn("단가 미등록 1회", line)
        self.assertIn("누적 단가 미등록 1회", line)
        self.assertEqual(llm_usage.snapshot_usage()["prompt"], 2_000)
        self.assertEqual(llm_usage.snapshot_usage()["output"], 200)

    def test_step_reports_timing_separately_from_single_llm_summary(self):
        buf = StringIO()
        with (
            patch(
                "core.llm_usage.time.monotonic",
                side_effect=[10.0, 12.3],
            ),
            redirect_stdout(buf),
        ):
            with llm_usage.step("Eval", 2, "검색 + 답변 생성"):
                llm_usage.log_usage("gpt-4o-mini", 1_000, 100)

        lines = buf.getvalue().splitlines()
        usage_lines = [line for line in lines if "LLM 사용 |" in line]
        self.assertEqual(len(usage_lines), 1)
        self.assertNotIn("소요", usage_lines[0])
        self.assertIn("[Eval] STEP2 소요: 2.3s", lines)
        self.assertFalse(hasattr(llm_usage, "_step_log"))

if __name__ == "__main__":
    unittest.main()
