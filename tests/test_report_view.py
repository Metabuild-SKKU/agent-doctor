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

from core.schema import DiagnosticReport, Finding, Probe
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


class ReportViewRecommendationsTest(unittest.TestCase):
    """남은 권고: D그룹 매뉴얼 처방 + 예비(의심) finding 을 웹 뷰로 노출하는지 고정."""

    def _state(self, findings, probes):
        st = AgentDoctorState(report=DiagnosticReport(report_id="r", findings=findings))
        st.probes = probes
        return st

    def test_corpus_gap_surfaces_manual_steps_and_where(self):
        probe = Probe(probe_id="u12", question="재택근무 상한은?", source="user_log",
                      gold_chunk_ids=["c1"], gold_doc_id="policy_2024")
        finding = Finding(finding_id="u12:corpus_gap", type="gap", severity="critical",
                          description="[D그룹] corpus_gap", label="corpus_gap",
                          confirmed=True, affected_probes=["u12"], metadata={"group": "D"})
        recs = build_report_view(self._state([finding], [probe]))["recommendations"]
        self.assertEqual(len(recs), 1)
        rec = recs[0]
        self.assertEqual(rec["kind"], "manual")
        self.assertEqual(rec["badge"], ["data", "D · 데이터"])
        self.assertTrue(rec["steps"])                       # 매뉴얼 스텝 실림
        self.assertEqual(rec["items"][0]["q"], "재택근무 상한은?")
        self.assertIn("policy_2024", rec["items"][0]["where"])  # 어디가 문제인지

    def test_preliminary_finding_becomes_prelim_recommendation(self):
        finding = Finding(finding_id="p1:too_long_context", type="context_failure",
                          severity="warning", description="[예비] [C그룹] too_long_context",
                          label="too_long_context", confirmed=False, affected_probes=["p1"],
                          metadata={"group": "C"})
        recs = build_report_view(self._state([finding], []))["recommendations"]
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["kind"], "prelim")
        self.assertEqual(recs[0]["badge"], ["prelim", "의심"])

    def test_confirmed_actionable_excluded_from_recommendations(self):
        # 확정 자동처방 대상(retrieval_low_rank)은 dxs/rxs 몫 — 남은 권고에서 제외.
        finding = Finding(finding_id="p1:retrieval_low_rank", type="retrieval_failure",
                          severity="warning", description="[A그룹] retrieval_low_rank",
                          label="retrieval_low_rank", confirmed=True, affected_probes=["p1"],
                          metadata={"group": "A"})
        recs = build_report_view(self._state([finding], []))["recommendations"]
        self.assertEqual(recs, [])


if __name__ == "__main__":
    unittest.main()
