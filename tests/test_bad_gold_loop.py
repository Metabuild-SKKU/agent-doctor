"""
tests/test_bad_gold_loop.py
D그룹 bad_gold_answer 처리 루프 검증.

Phase 2: 정답셋 오류(bad_gold_answer)로 판정된 probe 는 '거짓 실패'이므로 점수 집계
(composite/overall)에서 제외하되, 진단·리포트(findings)에는 남긴다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import Probe, Finding
from agents.eval.types import EvalRecord
from agents.eval.report import build_report, _is_bad_gold_probe


def _record(pid: str, f1: float, bad_gold: bool = False) -> EvalRecord:
    probe = Probe(probe_id=pid, question="q", source="taxonomy", ground_truth="gt")
    rec = EvalRecord(probe=probe)
    rec.f1_score = f1
    rec.recall_at_k = 1.0
    if bad_gold:
        rec.findings = [Finding(
            finding_id=pid, type="gap", severity="warning", description="bad",
            label="bad_gold_answer", confirmed=True, affected_probes=[pid],
        )]
    return rec


class BadGoldScoreExclusionTest(unittest.TestCase):
    def test_bad_gold_excluded_from_composite(self):
        good = [_record(f"g{i}", 0.9) for i in range(3)]
        bad = [_record("b1", 0.0, bad_gold=True)]

        only_good = build_report(good, 0, mode=1).composite_score["total"]
        with_bad = build_report(good + bad, 0, mode=1).composite_score["total"]
        # 거짓 실패(bad_gold)는 점수에서 빠지므로 두 종합점수가 같아야 한다.
        self.assertEqual(only_good, with_bad)

    def test_bad_gold_still_reported_in_findings(self):
        report = build_report([_record("g1", 0.9), _record("b1", 0.0, bad_gold=True)], 0, mode=1)
        labels = {f.label for f in report.findings}
        self.assertIn("bad_gold_answer", labels)  # 점수엔 빠져도 진단엔 남는다

    def test_only_confirmed_bad_gold_is_excluded(self):
        probe = Probe(probe_id="p", question="q", source="taxonomy", ground_truth="gt")
        rec = EvalRecord(probe=probe)
        rec.findings = [Finding(finding_id="p", type="gap", severity="warning", description="d",
                                label="bad_gold_answer", confirmed=False, affected_probes=["p"])]
        self.assertFalse(_is_bad_gold_probe(rec))  # 예비면 제외 대상 아님


if __name__ == "__main__":
    unittest.main()
