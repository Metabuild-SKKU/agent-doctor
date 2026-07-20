"""Eval이 라벨 단위 iteration을 소비하지 않는지 검증한다."""
import unittest
from unittest.mock import patch

from agents.eval import agent
from core.schema import Chunk, DiagnosticReport, Document, Probe
from core.state import AgentDoctorState


class EvalIterationContractTest(unittest.TestCase):
    @patch("agents.eval.agent.generate_probes", return_value=[])
    def test_eval_keeps_iteration_for_candidate_measurement(self, _generate_probes):
        state = AgentDoctorState(
            chunks=[Chunk(chunk_id="c1", doc_id="d1", text="충분한 길이의 테스트 청크")],
            user_questions=["질문"],
            iteration=2,
            max_iterations=3,
        )

        result = agent.run(state)

        self.assertIs(result, state)
        self.assertEqual(result.iteration, 2)
        self.assertEqual(result.status, "evaluated")

    @patch("agents.eval.agent.build_report")
    @patch("agents.eval.agent._evaluate_probe", return_value=object())
    @patch("agents.eval.agent.build_eval_index", return_value=object())
    @patch("agents.eval.agent.generate_probes")
    @patch("agents.eval.agent.load_probes")
    def test_cached_probe_is_resynced_after_rechunking(
        self,
        load_probes,
        generate_probes,
        _build_index,
        _evaluate,
        build_report,
    ):
        content = "가" * 100 + "정답" + "나" * 100
        start = content.index("정답")
        cached = Probe(
            probe_id="p1",
            question="질문",
            source="llm_generated",
            answer_exists=True,
            gold_chunk_ids=["old_chunk"],
            gold_spans=[{"doc_id": "d1", "start": start, "end": start + 2}],
        )
        load_probes.return_value = [cached]
        build_report.return_value = DiagnosticReport(
            report_id="r1",
            findings=[],
            overall_score=100.0,
            pass_threshold=True,
        )
        state = AgentDoctorState(
            documents=[Document("d1", "memory", "txt", content)],
            chunks=[
                Chunk("new_0", "d1", content[:100], char_span=(0, 100)),
                Chunk("new_1", "d1", content[100:], char_span=(100, len(content))),
            ],
        )

        result = agent.run(state)

        generate_probes.assert_not_called()
        self.assertEqual(result.probes[0].gold_chunk_ids, ["new_1"])


if __name__ == "__main__":
    unittest.main()
