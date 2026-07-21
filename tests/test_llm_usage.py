import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


if __name__ == "__main__":
    unittest.main()
