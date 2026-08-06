"""
tests/test_web_api_ticker.py
웹 진행 티커(_summarize_stage_event)의 점수 표시 검증.

이력 metadata 의 before_score/after_score 는 마진 판정용 탐색 신호(0~1)다.
이를 :.0f 로 찍으면 어떤 개선이든 "종합 0→1" 또는 "종합 0→0"으로 표시되는
회귀가 있었다 — 표시용 composite(0~100, finalize_item 이 함께 기록)를 읽어야 한다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.serve.web_api import _summarize_stage_event
from agents.optimize.schemas import OptimizationHistoryItem
from core.state import AgentDoctorState


def _state_with_history(metadata: dict) -> AgentDoctorState:
    item = OptimizationHistoryItem(
        trial_id="t1",
        request_id="r1",
        iteration=1,
        failure_labels=["retrieval_missing_gold"],
        optimizer="rules",
        status="applied",
        selected_prescription_id="increase_top_k",
        before_config={"top_k": 5},
        after_config={"top_k": 10},
        action_key="retriever.top_k:increase",
    )
    item.metadata.update({"pending": False, **metadata})
    state = AgentDoctorState()
    state.optimization_history = [item]
    return state


class OptimizeTickerScoreTest(unittest.TestCase):
    def test_ticker_shows_composite_scale(self):
        state = _state_with_history(
            {
                "before_score": 0.724,
                "after_score": 0.781,
                "before_composite": 72.4,
                "after_composite": 78.1,
            }
        )

        _tag, text, _kind = _summarize_stage_event("optimize", state)

        self.assertIn("종합 72→78", text)
        self.assertIn("유지", text)

    def test_legacy_history_without_composite_falls_back_scaled(self):
        """구버전 이력(composite 미기록)은 탐색 신호×100 으로 표시한다."""
        state = _state_with_history({"before_score": 0.6, "after_score": 0.8})

        _tag, text, _kind = _summarize_stage_event("optimize", state)

        self.assertIn("종합 60→80", text)

    def test_ticker_never_collapses_to_zero_or_one(self):
        """0~1 값을 :.0f 로 찍던 회귀: 어떤 정상 점수도 '0→1'/'0→0'이면 안 된다."""
        state = _state_with_history(
            {
                "before_score": 0.724,
                "after_score": 0.781,
                "before_composite": 72.4,
                "after_composite": 78.1,
            }
        )

        _tag, text, _kind = _summarize_stage_event("optimize", state)

        self.assertNotIn("종합 0→1", text)
        self.assertNotIn("종합 0→0", text)
        self.assertNotIn("종합 1→1", text)

    def test_no_scores_keeps_generic_message(self):
        """점수가 아예 없으면(pending 등) 기존의 일반 문구를 유지한다."""
        state = _state_with_history({})

        _tag, text, _kind = _summarize_stage_event("optimize", state)

        self.assertEqual(text, "설정 조정 시도")


if __name__ == "__main__":
    unittest.main()
