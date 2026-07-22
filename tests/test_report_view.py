"""
tests/test_report_view.py
agents/serve/report_view.build_report_view 가 리포트 헤드라인 '종합 점수'를
설계 종합점수(composite_score)로 노출하는지 검증한다.

핵심 계약: 웹이 보여주는 '종합 점수'는 overall_score(품질 단일축)가 아니라
composite_score(품질×신뢰도) 여야 한다. composite 가 없으면 overall×100 로 폴백.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import DiagnosticReport, Finding
from core.state import AgentDoctorState
from agents.serve.report_view import build_report_view


def make_report(overall=0.90, composite_total=12, pass_threshold=False):
    """overall(0~1)과 composite(0~100)를 일부러 크게 벌려, 어느 쪽이 노출되는지 구분."""
    composite = None
    if composite_total is not None:
        composite = {
            "total": composite_total,
            "components": [
                {"key": "quality", "label": "품질", "score": 89},
                {"key": "reliability", "label": "신뢰도", "score": 7},
            ],
        }
    return DiagnosticReport(
        report_id="r",
        findings=[Finding(finding_id="1", type="retrieval_failure", severity="warning",
                          description="d", label="too_long_context", affected_probes=["p1"])],
        overall_score=overall,
        ragas_scores={"context_recall": 0.7, "faithfulness": 0.7},
        composite_score=composite,
        pass_threshold=pass_threshold,
    )


def make_state(report):
    return AgentDoctorState(report=report, source_url="uploaded.pdf")


class ReportViewCompositeTest(unittest.TestCase):
    def test_headline_score_uses_composite_not_overall(self):
        # overall 0.90(→90점)과 composite 12 를 크게 벌려둠. 헤드라인은 composite(12)여야.
        view = build_report_view(make_state(make_report(overall=0.90, composite_total=12)))
        self.assertEqual(view["score"]["after"], 12)
        self.assertNotEqual(view["score"]["after"], 90.0)  # overall×100 이면 버그

    def test_falls_back_to_overall_when_composite_missing(self):
        # 구버전 리포트(composite 없음) → overall×100 로 폴백.
        view = build_report_view(make_state(make_report(overall=0.90, composite_total=None)))
        self.assertEqual(view["score"]["after"], 90.0)

    def test_no_optimization_before_equals_after(self):
        # 최적화 이력이 없으면 before==after, delta 0.
        view = build_report_view(make_state(make_report(composite_total=88)))
        self.assertEqual(view["score"]["before"], view["score"]["after"])
        self.assertEqual(view["score"]["delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
