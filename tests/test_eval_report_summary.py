import io
import os
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import DiagnosticReport
from agents.eval.report import _print_summary


def make_report(*, overall=0.9, composite_total=79.0, pass_threshold=True,
                recall=0.9):
    composite = {
        "total": composite_total,
        "components": [
            {"key": "quality", "label": "품질", "score": 89},
            {"key": "reliability", "label": "신뢰도", "score": 71},
        ],
    }
    ragas_scores = {}
    if recall is not None:
        ragas_scores["mean_recall_at_k"] = recall
    return DiagnosticReport(
        report_id="r",
        overall_score=overall,
        composite_score=composite,
        pass_threshold=pass_threshold,
        ragas_scores=ragas_scores,
    )


class EvalStep5GateSummaryTest(unittest.TestCase):
    def _summary(self, report):
        out = io.StringIO()
        with redirect_stdout(out):
            _print_summary([], report)
        return out.getvalue()

    def test_step5_summary_uses_final_gate_when_eval_score_passes_but_composite_fails(self):
        report = make_report(overall=0.8869, composite_total=79.0,
                             pass_threshold=True, recall=0.9375)

        text = self._summary(report)

        self.assertIn("gate_pass ✗", text)
        self.assertIn("gate reason: composite_below_threshold", text)
        self.assertIn("composite=79.0/90.0", text)
        self.assertIn("eval_pass ✓", text)

    def test_step5_summary_can_pass_gate_even_when_eval_threshold_is_false(self):
        report = make_report(overall=0.40, composite_total=92.0,
                             pass_threshold=False, recall=0.9)

        text = self._summary(report)

        self.assertIn("gate_pass ✓", text)
        self.assertNotIn("gate reason:", text)
        self.assertIn("eval_pass ✗", text)


if __name__ == "__main__":
    unittest.main()
