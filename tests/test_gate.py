"""
tests/test_gate.py
Optimize serve/optimize 게이트 정책 검증 — 점수 판정 + 검색 바닥선(RECALL_FLOOR).

핵심 계약(serve/optimize·처방 종료의 단일 기준):
  - Eval 점수 판정(score_pass=report.pass_threshold)이 False 면 통과 아님.
  - score_pass 여도 측정된 recall 이 floor 미만이면 통과 아님
    (평균이 가리는 "검색이 새는" 케이스 → 최적화로 보냄).
  - recall 미측정(None)이면 근거가 없어 floor 를 적용하지 않는다.
  - report 없음 → 통과 아님(예산 로직으로 넘어감).

gate 는 qdrant 의존이 없어 단독 테스트가 가능하다.
"""
import os
import sys
import unittest
from dataclasses import dataclass, field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.optimize import gate
from agents.optimize.gate import RECALL_FLOOR


class PassesTest(unittest.TestCase):
    def test_score_pass_and_high_recall_passes(self):
        self.assertTrue(gate.passes(True, 0.8))

    def test_score_pass_but_low_recall_fails(self):
        self.assertFalse(gate.passes(True, 0.4))

    def test_recall_at_floor_boundary_passes(self):
        self.assertTrue(gate.passes(True, RECALL_FLOOR))

    def test_score_fail_fails_regardless_of_recall(self):
        self.assertFalse(gate.passes(False, 0.9))

    def test_unmeasured_recall_does_not_apply_floor(self):
        self.assertTrue(gate.passes(True, None))
        self.assertTrue(gate.passes(True))


@dataclass
class _FakeReport:
    pass_threshold: bool
    ragas_scores: dict = field(default_factory=dict)


class PassesReportTest(unittest.TestCase):
    def test_none_report_not_passing(self):
        self.assertFalse(gate.passes_report(None))

    def test_score_pass_but_low_recall_fails(self):
        r = _FakeReport(True, {"mean_recall_at_k": 0.4})
        self.assertFalse(gate.passes_report(r))

    def test_score_pass_and_high_recall_passes(self):
        r = _FakeReport(True, {"mean_recall_at_k": 0.8})
        self.assertTrue(gate.passes_report(r))

    def test_missing_recall_key_treated_as_unmeasured(self):
        r = _FakeReport(True, {"mean_f1": 0.7})  # recall 키 없음 → floor 미적용
        self.assertTrue(gate.passes_report(r))

    def test_score_fail_report_not_passing(self):
        r = _FakeReport(False, {"mean_recall_at_k": 0.9})
        self.assertFalse(gate.passes_report(r))


if __name__ == "__main__":
    unittest.main()
