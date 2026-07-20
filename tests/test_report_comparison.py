import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import DiagnosticReport, Finding
from core.state import AgentDoctorState
from agents.optimize.schemas import OptimizationHistoryItem
from agents.report import build_comparison_payload, save_pipeline_report


def make_report(score: float, recall: float, f1: float) -> DiagnosticReport:
    finding = Finding(
        finding_id="f1",
        type="retrieval_failure",
        severity="warning",
        description="검색 누락",
        label="retrieval_missing_gold",
    )
    return DiagnosticReport(
        report_id=f"r{score}",
        findings=[finding],
        ragas_scores={"mean_recall_at_k": recall, "mean_f1": f1},
        oracle_accuracy=0.8,
        overall_score=score,
        pass_threshold=score >= 0.8,
        iteration=1,
    )


class ReportComparisonTest(unittest.TestCase):
    def test_builds_before_after_payload_from_pending_trial_and_final_report(self):
        before_report = make_report(0.62, 0.5, 0.55)
        after_report = make_report(0.81, 0.75, 0.7)
        trial = OptimizationHistoryItem(
            trial_id="trial-123456",
            request_id="request-123456",
            iteration=1,
            failure_labels=["retrieval_missing_gold"],
            optimizer="rules",
            status="applied",
            selected_prescription_id="increase_top_k",
            before_config={"top_k": 3, "use_hybrid": False},
            after_config={},
            before_metrics=dict(before_report.ragas_scores),
            after_metrics={},
            target_metrics=["mean_recall_at_k"],
            reason="recall 개선 필요",
            metadata={
                "pending": True,
                "before_report": before_report,
                "before_score": before_report.overall_score,
            },
        )
        state = AgentDoctorState(
            report=after_report,
            index_config={"top_k": 5, "use_hybrid": True},
            optimization_history=[trial],
            iteration=2,
            max_iterations=3,
        )

        payload = build_comparison_payload(state)

        self.assertTrue(payload["comparison"]["available"])
        rows = {row["metric"]: row for row in payload["comparison"]["metric_rows"]}
        self.assertAlmostEqual(rows["overall_score"]["delta"], 0.19)
        self.assertAlmostEqual(rows["mean_recall_at_k"]["delta"], 0.25)
        config_rows = {row["key"]: row for row in payload["comparison"]["config_rows"]}
        self.assertEqual(config_rows["top_k"]["before"], 3)
        self.assertEqual(config_rows["top_k"]["after"], 5)

    def test_saves_markdown_and_json_report_files(self):
        state = AgentDoctorState(
            report=make_report(0.75, 0.7, 0.6),
            index_config={"top_k": 5},
            iteration=1,
        )

        with tempfile.TemporaryDirectory() as directory:
            artifacts = save_pipeline_report(state, output_dir=directory)

            self.assertTrue(os.path.exists(artifacts["markdown"]))
            self.assertTrue(os.path.exists(artifacts["json"]))
            with open(artifacts["markdown"], encoding="utf-8") as markdown_file:
                markdown = markdown_file.read()
            self.assertIn("RAG Pipeline Before/After Report", markdown)
            with open(artifacts["json"], encoding="utf-8") as json_file:
                payload = json.load(json_file)
            self.assertEqual(payload["final_eval"]["overall_score"], 0.75)
            self.assertFalse(payload["comparison"]["available"])


if __name__ == "__main__":
    unittest.main()
