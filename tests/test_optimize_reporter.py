"""
tests/test_optimize_reporter.py
사용자 리포트가 **action 합의**를 설명하는지 검증한다 (구현계획 §8.4).

전환 전 리포트는 "어떤 라벨의 어떤 처방을 골랐나"를 말했다. 이제는
"어떤 config 변경을 왜 골랐나"를 말해야 한다 — 라벨은 그 변경을 지지한 근거이고,
여러 라벨이 같은 변경을 지지했다는 사실 자체가 선택 근거이기 때문이다.

함께 검증하는 것: 지지와 해결의 구분, 반대편·보류 축 노출, 구버전 이력 fallback.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.optimize import reporter
from agents.optimize.schemas import (
    OptimizationHistoryItem,
    OptimizationRequest,
    OptimizeDecision,
    Verdict,
)


def _decision(mode="apply_optimize", status="proposed", **overrides):
    values = {
        "mode": mode,
        "status": status,
        "requires_user_confirmation": False,
        "next_route": "index",
        "reason": "처방 가능한 finding 존재 → 최적화 진행",
    }
    values.update(overrides)
    return OptimizeDecision(**values)


def _request(**overrides):
    """planner._build_action_request 가 만드는 모양의 요청."""
    values = {
        "request_id": "req-1",
        "iteration": 0,
        "baseline_config": {"top_k": 5},
        "search_space": {"retriever.top_k": [10]},
        "action_key": "retriever.top_k:increase",
        "prescription_id": "increase_top_k",
        "supporting_labels": [
            "retrieval_missing_gold",
            "retrieval_incomplete_enumeration",
        ],
        "supporting_probes": ["p1", "p2", "p3"],
        "opposing_labels": ["too_long_context"],
        "action_score": 3.0,
        "action_score_breakdown": {
            "supporting_probe_count": 3,
            "weighted_probe_support": 3.0,
            "cost_source": "reindex_flag",
            "confidence_source": "default",
        },
        "metadata": {
            "deferred_axes": [
                {"axis": "retriever.top_k", "reason": "conflict_margin_unmet"}
            ]
        },
    }
    values.update(overrides)
    return OptimizationRequest(**values)


class ApplyReportTest(unittest.TestCase):
    def test_summary_names_the_action_not_a_single_label(self):
        report = reporter.build_report(_decision(), _request())

        self.assertEqual(report.action_key, "retriever.top_k:increase")
        self.assertIn("retriever.top_k:increase", report.summary)
        # 여러 라벨이 함께 지지했다는 사실이 선택 근거다 — 하나로 좁히지 않는다.
        self.assertIn("retrieval_missing_gold", report.summary)
        self.assertIn("retrieval_incomplete_enumeration", report.summary)

    def test_report_carries_the_full_selection_rationale(self):
        report = reporter.build_report(_decision(), _request())

        self.assertEqual(
            report.supporting_labels,
            ["retrieval_missing_gold", "retrieval_incomplete_enumeration"],
        )
        self.assertEqual(report.opposing_labels, ["too_long_context"])
        self.assertEqual(report.score_breakdown["supporting_probe_count"], 3)
        self.assertEqual(report.score_breakdown["cost_source"], "reindex_flag")
        self.assertEqual(
            [axis["reason"] for axis in report.deferred_axes],
            ["conflict_margin_unmet"],
        )

    def test_selected_prescription_is_read_not_guessed(self):
        """실제 선택된 action 의 출처를 그대로 읽는다.

        전환 전 reporter 는 `request.candidates[0].id` 로 "가장 가벼운 처방"을
        추측했다. 후보 순서가 선택 순서라는 가정이 깨지면 적용되지 않은 처방
        이름을 보고했다. 이제 후보 목록 자체가 없어 추측할 자리가 사라졌다 —
        요청이 들고 있는 값이 곧 보고되는 값이다.
        """
        request = _request(prescription_id="switch_to_recursive_sentence")

        report = reporter.build_report(_decision(), request)

        self.assertEqual(report.selected_prescription, "switch_to_recursive_sentence")
        self.assertIn("switch_to_recursive_sentence", report.summary)

    def test_request_without_an_action_key_still_reports(self):
        """action 이 비어도 리포트는 만들어진다 — 처방 id 로 설명한다."""
        request = _request(
            action_key=None,
            opposing_labels=[],
            action_score_breakdown={},
        )

        report = reporter.build_report(_decision(), request)

        self.assertIsNone(report.action_key)
        self.assertIn("increase_top_k", report.summary)   # 처방 id 로 설명한다
        self.assertIn("retrieval_missing_gold", report.summary)


class TrialReportTest(unittest.TestCase):
    """판정이 끝난 처방의 리포트. 입력은 이력 항목에서만 뽑는다."""

    @staticmethod
    def _item(**overrides):
        values = {
            "trial_id": "t1",
            "request_id": "r1",
            "iteration": 1,
            "failure_labels": ["retrieval_missing_gold"],
            "optimizer": "rules",
            "status": "applied",
            "selected_prescription_id": "increase_top_k",
            "before_config": {"top_k": 5},
            "after_config": {"top_k": 10},
            "action_key": "retriever.top_k:increase",
            "supporting_labels": [
                "retrieval_missing_gold",
                "retrieval_incomplete_enumeration",
            ],
        }
        values.update(overrides)
        item = OptimizationHistoryItem(**values)
        item.metadata.update(
            {
                "resolved_labels": ["retrieval_missing_gold"],
                "remaining_labels": ["retrieval_incomplete_enumeration"],
            }
        )
        return item

    def test_kept_report_separates_supported_from_resolved(self):
        """지지받은 라벨을 그대로 성과로 읽으면 리포트가 실제보다 좋아 보인다."""
        verdict = Verdict(keep=True, before_score=0.6, after_score=0.8)

        report = reporter.build_trial_report(self._item(), verdict)

        self.assertEqual(report.status, "applied")
        self.assertEqual(report.action_key, "retriever.top_k:increase")
        self.assertEqual(report.resolved_labels, ["retrieval_missing_gold"])
        self.assertEqual(
            report.remaining_labels, ["retrieval_incomplete_enumeration"]
        )
        self.assertIn("해결된 문제: retrieval_missing_gold", report.summary)

    def test_rolled_back_report_claims_nothing_resolved(self):
        """롤백은 설정을 되돌렸다 — 그 개선은 지금 config 에 남아 있지 않다."""
        verdict = Verdict(
            keep=False, before_score=0.8, after_score=0.6, reason="종합점수 미상승"
        )

        report = reporter.build_trial_report(
            self._item(status="failed"), verdict
        )

        self.assertEqual(report.status, "failed")
        self.assertEqual(report.resolved_labels, [])
        self.assertEqual(
            report.remaining_labels,
            ["retrieval_missing_gold", "retrieval_incomplete_enumeration"],
        )
        self.assertNotIn("해결된 문제", report.summary)

    def test_legacy_item_falls_back_to_the_prescription_name(self):
        item = OptimizationHistoryItem(
            trial_id="old",
            request_id="old",
            iteration=1,
            failure_labels=["retrieval_missing_gold"],
            optimizer="rules",
            status="applied",
            selected_prescription_id="increase_top_k",
            before_config={"top_k": 5},
            after_config={"top_k": 10},
        )
        verdict = Verdict(keep=True, before_score=0.6, after_score=0.8)

        report = reporter.build_trial_report(item, verdict)

        self.assertIsNone(report.action_key)
        self.assertIn("increase_top_k", report.summary)
        self.assertEqual(report.supporting_labels, ["retrieval_missing_gold"])


class ScoreDisplayTest(unittest.TestCase):
    """요약의 점수는 표시용 composite(0~100)여야 한다.

    Verdict.before_score/after_score 는 마진 판정용 탐색 신호(0~1)다. 이를
    그대로 :.1f 로 찍으면 72.4→78.1 개선이 "0.7→0.8"로, 그보다 작은 개선은
    "0.7→0.7"(올랐다면서 같은 숫자)로 표시되는 회귀가 있었다.
    """

    @staticmethod
    def _item(**overrides):
        return TrialReportTest._item(**overrides)

    def test_kept_summary_shows_composite_scale(self):
        verdict = Verdict(
            keep=True,
            before_score=0.724,
            after_score=0.781,
            before_composite=72.4,
            after_composite=78.1,
        )

        report = reporter.build_trial_report(self._item(), verdict)

        self.assertIn("72.4→78.1", report.summary)
        self.assertNotIn("0.7→0.8", report.summary)

    def test_small_improvement_never_renders_identical_numbers(self):
        """composite 한 자리 차이(72.4→74.0)도 서로 다른 숫자로 보여야 한다."""
        verdict = Verdict(
            keep=True,
            before_score=0.724,
            after_score=0.740,
            before_composite=72.4,
            after_composite=74.0,
        )

        report = reporter.build_trial_report(self._item(), verdict)

        self.assertIn("72.4→74.0", report.summary)

    def test_missing_composite_falls_back_to_scaled_search_score(self):
        """composite 미측정(구버전 Verdict)이면 탐색 신호×100 으로 표시한다."""
        verdict = Verdict(keep=True, before_score=0.6, after_score=0.8)

        report = reporter.build_trial_report(self._item(), verdict)

        self.assertIn("60.0→80.0", report.summary)

    def test_apply_report_summary_uses_composite_too(self):
        """build_report 의 apply 경로도 같은 표시 규약을 쓴다."""
        verdict = Verdict(
            keep=True,
            before_score=0.724,
            after_score=0.781,
            before_composite=72.4,
            after_composite=78.1,
        )

        report = reporter.build_report(_decision(), _request(), verdict)

        self.assertIn("72.4→78.1", report.summary)

    def test_metadata_carries_display_composites(self):
        """하류 UI 가 0~1 값을 스케일 오인하지 않도록 표시 값도 함께 싣는다."""
        verdict = Verdict(
            keep=True,
            before_score=0.724,
            after_score=0.781,
            before_composite=72.4,
            after_composite=78.1,
        )

        report = reporter.build_trial_report(self._item(), verdict)

        self.assertEqual(report.metadata["before_score"], 0.724)
        self.assertEqual(report.metadata["after_score"], 0.781)
        self.assertEqual(report.metadata["before_composite"], 72.4)
        self.assertEqual(report.metadata["after_composite"], 78.1)

    def test_partial_composite_omits_the_number_instead_of_mixing_axes(self):
        """한쪽만 composite 이면 숫자를 빼고 미측정이라고 말한다.

        prescreener sweep 이 이 경우다 — after_score 는 종합점수가 아니라 정답 span
        포함률이라, ×100 으로 채우면 "점수가 72.4→92.0로 올라"처럼 축이 다른 두 값이
        한 문장에 들어간다. 눈에 보이는 오류를 그럴듯한 오류로 바꾸는 셈이라 표시를 뺀다.
        """
        verdict = Verdict(
            keep=True,
            before_score=0.724,
            after_score=0.92,
            before_composite=72.4,
            after_composite=None,
        )

        report = reporter.build_trial_report(self._item(), verdict)

        self.assertNotIn("92.0", report.summary)
        self.assertNotIn("72.4→", report.summary)
        self.assertIn("유지", report.summary)

    def test_proxy_only_verdict_never_shows_a_composite_number(self):
        """proxy_only 판정은 값이 갖춰져 있어도 종합점수로 표시하지 않는다."""
        verdict = Verdict(
            keep=True,
            before_score=0.724,
            after_score=0.92,
            before_composite=72.4,
            after_composite=92.0,
            proxy_only=True,
        )

        report = reporter.build_trial_report(self._item(), verdict)

        self.assertNotIn("92.0", report.summary)

    def test_rollback_summary_carries_the_reason_on_display_scale(self):
        """롤백 요약은 history 가 만든 reason 을 그대로 싣는다(그쪽이 0~100 으로 쓴다)."""
        verdict = Verdict(
            keep=False,
            before_score=0.780,
            after_score=0.750,
            before_composite=78.0,
            after_composite=75.0,
            reason="종합점수 미상승 78.0→75.0 → 롤백",
        )

        report = reporter.build_trial_report(self._item(), verdict)

        self.assertIn("78.0→75.0", report.summary)
        self.assertNotIn("0.780", report.summary)


class NoActionReportTest(unittest.TestCase):
    """실행 가능한 action 이 하나도 없으면 request 가 없다 — 그래도 이유는 남아야 한다."""

    def test_rejection_reasons_survive_without_a_request(self):
        decision = _decision(
            mode="use_current",
            status="skipped",
            next_route="serve",
            reason="적용 가능한 action 이 없음",
            metadata={
                "rejected_actions": [
                    {
                        "action_key": "chunker.chunk_size:decrease",
                        "reason": "no_candidate_value",
                    }
                ],
                "deferred_axes": [],
            },
        )

        report = reporter.build_report(decision)

        self.assertEqual(report.status, "skipped")
        self.assertIn("실행할 수 없던 변경 1건", report.summary)
        self.assertEqual(
            report.metadata["rejected_actions"][0]["reason"],
            "no_candidate_value",
        )

    def test_plain_skip_is_unchanged(self):
        report = reporter.build_report(
            _decision(mode="use_current", status="already_optimal", next_route="serve")
        )

        self.assertEqual(report.status, "already_optimal")
        self.assertNotIn("실행할 수 없던", report.summary)
        self.assertEqual(report.metadata, {})


if __name__ == "__main__":
    unittest.main()
