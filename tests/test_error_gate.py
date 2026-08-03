"""노드 간 에러 게이트 계약 검증.

상위 노드(Ingest/Index/Eval)가 status="error" 로 실패하면:
  - 하위 노드는 자체 "데이터 없음" 메시지로 덮지 않고 실제 원인을 유지한 채 건너뛴다.
  - 오케스트레이터 라우터는 Optimize/재색인 루프로 헛돌지 않고 Serve 로 종료시킨다.
이 계약이 없으면 잘못된 소스 URL/미구현 소스 같은 실패가 "청크가 없습니다" 같은
일반 메시지로 둔갑해, 최종 상태를 읽는 web_api 가 진짜 원인을 표시하지 못한다.
"""
import io
import unittest
from contextlib import redirect_stdout

import graph
from agents.index.agent import run as index_run
from agents.eval.agent import run as eval_run
from agents.serve.agent import run as serve_run
from core.state import AgentDoctorState

_REAL_ERROR = "gdrive 수집은 아직 미구현입니다."


def _silent(fn, *args):
    with redirect_stdout(io.StringIO()):
        return fn(*args)


class ErrorGateNodePassthroughTest(unittest.TestCase):
    """실패 상태로 들어온 노드는 실제 error 를 유지한 채 건너뛴다(자체 메시지로 덮지 않음)."""

    def _errored_state(self):
        return AgentDoctorState(
            source_url="x", source_type="gdrive",
            status="error", error=_REAL_ERROR,
        )

    def test_index_preserves_upstream_error(self):
        result = _silent(index_run, self._errored_state())
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error, _REAL_ERROR)  # "문서가 없습니다" 로 덮이지 않음

    def test_eval_preserves_upstream_error(self):
        result = _silent(eval_run, self._errored_state())
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error, _REAL_ERROR)  # "청크가 없습니다" 로 덮이지 않음

    def test_serve_preserves_upstream_error(self):
        result = _silent(serve_run, self._errored_state())
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error, _REAL_ERROR)

    def test_error_survives_full_downstream_chain(self):
        state = self._errored_state()
        for node in (index_run, eval_run, serve_run):
            state = _silent(node, state)
            self.assertEqual(state.status, "error")
            self.assertEqual(state.error, _REAL_ERROR)


class ErrorGateRoutingTest(unittest.TestCase):
    """라우터는 실패 상태를 Serve 로 종료시킨다(정상 흐름은 그대로)."""

    def test_route_after_eval_error_goes_serve(self):
        state = AgentDoctorState(status="error", error=_REAL_ERROR)
        self.assertEqual(_silent(graph.route_after_eval, state), "serve")

    def test_route_after_optimize_error_goes_serve(self):
        state = AgentDoctorState(status="error", error=_REAL_ERROR)
        self.assertEqual(_silent(graph.route_after_optimize, state), "serve")

    def test_normal_reindex_routing_unaffected(self):
        # 에러 게이트가 정상 재색인 흐름(applied/rolled_back → index)을 막지 않는다.
        for status in ("applied", "rolled_back"):
            state = AgentDoctorState(status=status)
            self.assertEqual(_silent(graph.route_after_optimize, state), "index")


if __name__ == "__main__":
    unittest.main()
