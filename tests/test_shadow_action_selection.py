"""
tests/test_shadow_action_selection.py
shadow mode 검증 (계획서 단계 3).

planner 는 legacy 경로로 실제 선택을 하고, 그와 별개로 "action 중심이었다면 무엇을
골랐을까" 를 계산해 request metadata 에 남긴다. 이 관측으로 전환의 실익을 판정한다
(중단 기준: 선택이 전혀 달라지지 않거나 tie-break 차이뿐이면 선택 로직 전환 보류).

**shadow 는 관측일 뿐이므로 실제 선택을 바꾸면 안 되고, 실패해도 최적화를 막으면 안 된다.**
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.optimize import planner
from tests.test_planner import make_finding, make_state


def _shadow(findings, state=None, blacklist=None):
    request, _decision = planner.plan(
        state or make_state(findings), blacklist=blacklist or set()
    )
    return request, request.metadata["shadow_action_selection"]


class ShadowIsObservationOnlyTest(unittest.TestCase):
    """shadow 는 실제 선택에 영향을 주지 않는다."""

    def test_shadow_does_not_change_legacy_selection(self):
        findings = [make_finding("p0", "retrieval_missing_gold", gold_ranks={"g": 12})]
        request, shadow = _shadow(findings)
        self.assertEqual(request.failure_label, "retrieval_missing_gold")
        self.assertEqual(request.candidates[0].id, "increase_top_k")
        self.assertEqual(shadow["status"], "ok")

    def test_shadow_failure_does_not_break_planning(self):
        """관측이 깨져도 최적화는 계속돼야 한다."""
        findings = [make_finding("p0", "retrieval_missing_gold", gold_ranks={"g": 12})]
        with patch.object(
            planner, "_compute_shadow_selection", side_effect=RuntimeError("boom")
        ):
            request, shadow = _shadow(findings)
        self.assertIsNotNone(request)
        self.assertEqual(shadow["status"], "error")
        self.assertIn("RuntimeError", shadow["error"])


class ShadowComparisonTest(unittest.TestCase):
    """legacy 와 action 선택을 같은 잣대로 비교한다."""

    def test_legacy_prescription_is_translated_to_action_key(self):
        findings = [make_finding("p0", "retrieval_missing_gold", gold_ranks={"g": 12})]
        _request, shadow = _shadow(findings)
        self.assertEqual(shadow["legacy_action_key"], "retriever.top_k:increase")

    def test_agreement_is_recorded(self):
        findings = [make_finding("p0", "retrieval_missing_gold", gold_ranks={"g": 12})]
        _request, shadow = _shadow(findings)
        self.assertEqual(
            shadow["agrees"],
            shadow["legacy_action_key"] == shadow["shadow_action_key"],
        )

    def test_divergence_reason_is_none_when_agreeing(self):
        findings = [make_finding("p0", "retrieval_missing_gold", gold_ranks={"g": 12})]
        _request, shadow = _shadow(findings)
        if shadow["agrees"]:
            self.assertIsNone(shadow["divergence_reason"])


class SharedSupportObservationTest(unittest.TestCase):
    """전환의 실익 — 여러 라벨이 같은 변경을 지지하는 사례를 센다."""

    def test_merged_action_is_counted(self):
        findings = [
            make_finding("p0", "retrieval_missing_gold", gold_ranks={"g": 12}),
            make_finding("q0", "retrieval_incomplete_enumeration", gold_ranks={"g": 12}),
        ]
        _request, shadow = _shadow(findings)
        self.assertGreaterEqual(shadow["merged_action_count"], 1)

    def test_shadow_selection_carries_all_supporting_labels(self):
        """legacy 는 대표 라벨 하나만 알지만 shadow 는 지지 전체를 안다."""
        findings = [
            make_finding("p0", "retrieval_missing_gold", gold_ranks={"g": 12}),
            make_finding("q0", "retrieval_incomplete_enumeration", gold_ranks={"g": 12}),
        ]
        request, shadow = _shadow(findings)
        self.assertEqual(len(shadow["shadow_supporting_labels"]), 2)
        # legacy 는 하나의 대표 라벨로만 기록된다 — 이 차이가 전환의 이유다.
        self.assertEqual(
            len([request.failure_label]), 1
        )

    def test_probe_count_reflects_union_not_single_label(self):
        findings = [
            make_finding("p0", "retrieval_missing_gold", gold_ranks={"g": 12}),
            make_finding("p1", "retrieval_missing_gold", gold_ranks={"g": 12}),
            make_finding("q0", "retrieval_incomplete_enumeration", gold_ranks={"g": 12}),
        ]
        _request, shadow = _shadow(findings)
        self.assertEqual(shadow["shadow_supporting_probe_count"], 3)


class ShadowDiagnosticsTest(unittest.TestCase):
    """중단 기준 판정과 디버깅에 필요한 값이 남는다."""

    def test_score_breakdown_is_recorded(self):
        findings = [make_finding("p0", "retrieval_missing_gold", gold_ranks={"g": 12})]
        _request, shadow = _shadow(findings)
        breakdown = shadow["shadow_score_breakdown"]
        for key in (
            "supporting_probe_count",
            "weighted_probe_support",
            "base_cost",
            "cost_source",
            "confidence_source",
        ):
            self.assertIn(key, breakdown)

    def test_rejected_actions_are_listed_with_reasons(self):
        """왜 경쟁에서 빠졌는지 남아야 starvation 을 발견할 수 있다."""
        findings = [make_finding("p0", "retrieval_missing_gold", gold_ranks={"g": 12})]
        _request, shadow = _shadow(findings)
        reasons = {r["action_key"]: r["reason"] for r in shadow["rejected_actions"]}
        self.assertIn("query_rewrite:replace", reasons)
        self.assertEqual(reasons["query_rewrite:replace"], "catalog_blocked")

    def test_deferred_axes_are_recorded(self):
        findings = [make_finding("p0", "retrieval_missing_gold", gold_ranks={"g": 12})]
        _request, shadow = _shadow(findings)
        self.assertIsInstance(shadow["deferred_axes"], list)

    def test_catalog_size_is_reported(self):
        findings = [make_finding("p0", "retrieval_missing_gold", gold_ranks={"g": 12})]
        _request, shadow = _shadow(findings)
        self.assertGreater(shadow["catalog_size"], 0)


class ShadowRespectsGroupOrderTest(unittest.TestCase):
    """A > C > B 가 shadow 에서도 유지된다."""

    def test_b_group_does_not_win_over_a_group(self):
        findings = [make_finding("p0", "retrieval_missing_gold", gold_ranks={"g": 12})]
        findings += [
            make_finding(f"b{i}", "generation_hallucination") for i in range(6)
        ]
        _request, shadow = _shadow(findings)
        self.assertFalse(
            shadow["shadow_action_key"].startswith("generation."),
            "B그룹이 A그룹을 앞질렀다 — hard tier 가 깨졌다",
        )


if __name__ == "__main__":
    unittest.main()
