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
from agents.eval.report import build_report
from agents.eval.types import EvalRecord
from agents.optimize.schemas import OptimizationHistoryItem
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


class TreatmentCourseViewTest(unittest.TestCase):
    @staticmethod
    def _history_item(index, status, before, after, prescription):
        return OptimizationHistoryItem(
            trial_id=f"trial-{index}",
            request_id=f"request-{index}",
            iteration=index,
            failure_labels=["test_failure"],
            optimizer="rules",
            status=status,
            selected_prescription_id=prescription,
            metadata={
                "pending": False,
                "before_composite": before,
                "after_composite": after,
            },
        )

    def test_rollback_adds_failed_score_and_restored_score_as_separate_points(self):
        state = make_state(make_report(composite_total=84))
        state.optimization_history = [
            self._history_item(1, "applied", 62, 70, "increase_top_k"),
            self._history_item(2, "failed", 70, 66, "decrease_chunk_size"),
            self._history_item(3, "applied", 70, 84, "enable_reranker"),
        ]

        course = build_report_view(state)["course"]

        self.assertEqual(
            [(point["kind"], point["score"]) for point in course],
            [
                ("baseline", 62),
                ("kept", 70),
                ("failed", 66),
                ("rollback", 70),
                ("kept", 84),
            ],
        )
        self.assertEqual(course[1]["label"], "top_k 확대")
        self.assertEqual(course[2]["label"], "청크 축소 실패")
        self.assertEqual(course[3]["label"], "원상 복구")
        self.assertEqual(course[4]["label"], "리랭커 활성화")


class FailedQuestionViewTest(unittest.TestCase):
    def test_report_keeps_question_expected_and_actual_answer(self):
        probe = Probe(
            probe_id="p1",
            question="무료 체험 기간은 며칠인가요?",
            source="llm_generated",
            ground_truth="14일",
        )
        finding = Finding(
            finding_id="f1",
            type="generation_failure",
            severity="warning",
            description="답변이 기대 정답과 다릅니다.",
            label="generation_wrong_answer",
            affected_probes=["p1"],
            prescription="생성 프롬프트 조정",
        )
        record = EvalRecord(
            probe=probe,
            generated_answer="30일입니다.",
            findings=[finding],
        )

        report = build_report([record], iteration=1, mode=1)
        view = build_report_view(AgentDoctorState(report=report, probes=[probe]))

        self.assertEqual(report.failed_questions, [{
            "probe_id": "p1",
            "question": "무료 체험 기간은 며칠인가요?",
            "expected_answer": "14일",
            "actual_answer": "30일입니다.",
        }])
        self.assertEqual(len(view["qas"]), 1)
        self.assertEqual(view["qas"][0]["q"], probe.question)
        self.assertEqual(view["qas"][0]["gold"], probe.ground_truth)
        self.assertEqual(view["qas"][0]["actual"], "30일입니다.")
        self.assertEqual(view["qas"][0]["diagnosis"], finding.description)

    def test_failed_question_without_answer_has_explicit_empty_value(self):
        report = make_report()
        report.failed_questions = [{
            "probe_id": "p1",
            "question": "답을 찾을 수 있나요?",
            "expected_answer": "예",
            "actual_answer": "",
        }]
        view = build_report_view(make_state(report))

        self.assertIn("actual", view["qas"][0])
        self.assertEqual(view["qas"][0]["actual"], "")


if __name__ == "__main__":
    unittest.main()
