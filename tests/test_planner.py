"""Planner의 후보 리스트와 internal 요청 연결 계약을 검증한다."""
import unittest

from agents.optimize import planner
from core.schema import DiagnosticReport, Document, Finding, Probe
from core.state import AgentDoctorState


def _report(finding: Finding) -> DiagnosticReport:
    return DiagnosticReport(
        report_id="report",
        findings=[finding],
        overall_score=60.0,
        ragas_scores={"context_recall": 0.6},
        pass_threshold=False,
    )


class PlannerCandidateListTest(unittest.TestCase):
    def test_preliminary_finding_is_not_auto_applied(self):
        finding = Finding(
            finding_id="f",
            type="retrieval_failure",
            severity="warning",
            description="예비 진단",
            label="retrieval_missing_gold",
            confirmed=False,
        )
        request, decision = planner.plan(AgentDoctorState(report=_report(finding)))
        self.assertIsNone(request)
        self.assertEqual(decision.status, "skipped")

    def test_top_k_candidates_make_one_internal_request(self):
        finding = Finding(
            finding_id="f",
            type="retrieval_failure",
            severity="warning",
            description="gold가 검색 결과에 없음",
            label="retrieval_missing_gold",
            affected_probes=["p1"],
            metadata={"parameter_candidates": {"top_k": [3, 7, 9]}},
        )
        state = AgentDoctorState(
            report=_report(finding),
            index_config={"top_k": 5, "chunk_size": 512, "chunk_overlap": 50},
        )
        request, decision = planner.plan(state)
        self.assertEqual(decision.mode, "apply_optimize")
        self.assertEqual(request.optimizer, "internal")
        self.assertEqual(request.search_space, {"retriever.top_k": [3, 7, 9]})
        self.assertEqual(request.max_trials, 3)

    def test_chunk_candidates_include_preview_inputs(self):
        finding = Finding(
            finding_id="f",
            type="retrieval_failure",
            severity="warning",
            description="context가 너무 김",
            label="too_long_context",
            affected_probes=["p1"],
            metadata={"parameter_candidates": {"chunker.chunk_size": [400, 600]}},
        )
        state = AgentDoctorState(
            report=_report(finding),
            documents=[Document("d1", "memory", "txt", "가" * 1000)],
            probes=[
                Probe(
                    probe_id="p1",
                    question="질문",
                    source="taxonomy",
                    gold_spans=[{"doc_id": "d1", "start": 100, "end": 180}],
                )
            ],
            index_config={"top_k": 5, "chunk_size": 512, "chunk_overlap": 50},
        )
        request, _decision = planner.plan(
            state,
            blacklist={
                ("too_long_context", "decrease_top_k"),
                ("too_long_context", "context_compression"),
            },
        )
        self.assertEqual(request.optimizer, "internal")
        self.assertEqual(request.search_space, {"chunker.chunk_size": [400, 600]})
        context = request.metadata["chunk_precheck_context"]
        self.assertEqual(context["documents"][0].doc_id, "d1")
        self.assertEqual(context["gold_spans"][0]["start"], 100)


if __name__ == "__main__":
    unittest.main()
