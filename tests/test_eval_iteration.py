"""Eval이 라벨 단위 iteration을 소비하지 않는지 검증한다."""
import unittest
from unittest.mock import patch

from agents.eval import agent
from core.schema import Chunk
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


if __name__ == "__main__":
    unittest.main()
