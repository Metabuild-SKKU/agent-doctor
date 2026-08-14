"""
tests/test_score_display.py
표시 점수 변환 규약(agents/optimize/score_display)의 단위 고정.

소비처(reporter 요약 · web_api 티커 · report_view 카드)는 각자의 문자열을 검사하지만,
스케일 규약 자체는 여기 한 곳에서 못 박는다. 새 소비처가 생겨도 이 규약만 지키면
같은 종류의 버그(0~1 을 종합점수로 표시 / 축이 다른 두 값을 화살표로 연결)가 나지 않는다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.optimize import history
from agents.optimize.schemas import Verdict
from agents.optimize.score_display import (
    display_scores_from_metadata,
    display_scores_from_verdict,
    resolve_display_scores,
)
from core.schema import DiagnosticReport


class ResolveDisplayScoresTest(unittest.TestCase):
    def test_composite_pair_is_used_as_is(self):
        """composite 는 이미 0~100 이라 그대로 쓴다."""
        scores = resolve_display_scores(0.724, 0.781, 72.4, 78.1)

        self.assertTrue(scores.available)
        self.assertEqual((scores.before, scores.after), (72.4, 78.1))

    def test_legacy_pair_without_composite_is_restored(self):
        """composite 를 기록하지 않던 구버전 이력만 ×100 복원을 탄다.

        그때의 탐색 신호는 composite÷100 이라 이 복원은 정확하다.
        """
        scores = resolve_display_scores(0.724, 0.781)

        self.assertTrue(scores.available)
        self.assertEqual((scores.before, scores.after), (72.4, 78.1))

    def test_small_improvement_stays_two_distinct_numbers(self):
        """0~1 을 :.1f 로 찍던 회귀: 작은 개선이 같은 숫자로 뭉개지면 안 된다."""
        scores = resolve_display_scores(0.724, 0.740, 72.4, 74.0)

        self.assertNotEqual(scores.before, scores.after)

    def test_only_before_composite_refuses_to_display(self):
        """한쪽만 composite 이면 다른 축의 값으로 채우지 않는다.

        이것이 prescreener 경로다 — after_score 는 종합점수가 아니라 정답 span
        포함률이라, ×100 으로 채우면 '종합 72→92' 같은 틀린 숫자가 나온다.
        """
        scores = resolve_display_scores(0.724, 0.92, 72.4, None)

        self.assertFalse(scores.available)
        self.assertIsNone(scores.before)
        self.assertIsNone(scores.after)
        self.assertIsNotNone(scores.unavailable_reason)

    def test_only_after_composite_refuses_to_display(self):
        """반대 방향도 같다 — 규칙은 쌍 단위다."""
        scores = resolve_display_scores(0.724, 0.781, None, 78.1)

        self.assertFalse(scores.available)

    def test_proxy_only_never_displays_a_number(self):
        """프록시 지표는 값이 갖춰져 있어도 숫자 자체를 보여주지 않는다.

        다른 이름으로라도 숫자를 보여주면 사용자는 결국 개선폭처럼 읽는다.
        """
        scores = resolve_display_scores(0.7, 0.9, 70.0, 90.0, proxy_only=True)

        self.assertFalse(scores.available)
        self.assertIsNone(scores.before)
        self.assertIsNone(scores.after)
        self.assertIsNotNone(scores.unavailable_reason)

    def test_no_scores_at_all_is_unavailable(self):
        """판정 전(pending) 이력처럼 점수 자체가 없으면 문장을 만들지 않는다."""
        scores = resolve_display_scores(None, None)

        self.assertFalse(scores.available)


class WrapperTest(unittest.TestCase):
    def test_from_verdict_reads_composite_pair(self):
        verdict = Verdict(
            keep=True,
            before_score=0.724,
            after_score=0.781,
            before_composite=72.4,
            after_composite=78.1,
        )

        scores = display_scores_from_verdict(verdict)

        self.assertEqual((scores.before, scores.after), (72.4, 78.1))

    def test_from_verdict_honours_proxy_only(self):
        """sweep prescreener 경로의 Verdict 는 proxy_only 를 싣고 온다."""
        verdict = Verdict(
            keep=True,
            before_score=0.724,
            after_score=0.92,
            proxy_only=True,
        )

        self.assertFalse(display_scores_from_verdict(verdict).available)

    def test_from_metadata_reads_history_item(self):
        scores = display_scores_from_metadata(
            {"before_composite": 60.0, "after_composite": 80.0}
        )

        self.assertEqual((scores.before, scores.after), (60.0, 80.0))

    def test_from_metadata_tolerates_empty(self):
        self.assertFalse(display_scores_from_metadata({}).available)


def _report(composite_total: float) -> DiagnosticReport:
    report = DiagnosticReport(report_id="r")
    report.composite_score = {"total": composite_total}
    report.ragas_scores = {}
    return report


class JudgeReasonScaleTest(unittest.TestCase):
    """`verdict.reason` 은 사용자에게 그대로 노출된다(처방 카드의 "판정 근거" 캡션).

    "종합점수"라고 이름을 대면서 0.780 을 보여주면 리포트 상단의 78.0 과 앞뒤가
    안 맞는다. 판정은 계속 0~1 탐색 신호로 하되, 문장의 숫자만 0~100 이어야 한다.
    """

    def test_rollback_reason_uses_display_scale(self):
        verdict = history.judge(_report(78.0), _report(75.0))

        self.assertIn("78.0→75.0", verdict.reason)
        self.assertNotIn("0.780", verdict.reason)
        # 판정 자체는 여전히 0~1 탐색 신호로 한다.
        self.assertAlmostEqual(verdict.before_score, 0.78)

    def test_margin_rejected_reason_scales_the_margin_too(self):
        """상승폭과 마진도 같은 스케일이어야 문장 안에서 앞뒤가 맞는다."""
        verdict = history.judge(_report(78.0), _report(79.0))

        self.assertIn("78.0→79.0", verdict.reason)
        self.assertIn("+1.0", verdict.reason)
        # 마진도 ×100 되어야 문장 안에서 상승폭과 같은 축이 된다. 상수를 재보정해도
        # (#102) 이 계약은 그대로여야 하므로 리터럴 대신 상수에서 끌어온다.
        self.assertIn(f"+{history.MIN_IMPROVEMENT_MARGIN * 100:.1f}", verdict.reason)
        self.assertTrue(verdict.margin_rejected)

    def test_keep_reason_uses_display_scale(self):
        verdict = history.judge(_report(75.0), _report(78.0))

        self.assertIn("75.0→78.0", verdict.reason)
        self.assertTrue(verdict.keep)


if __name__ == "__main__":
    unittest.main()
