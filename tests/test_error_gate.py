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
from unittest.mock import patch

import graph
from agents.index.agent import run as index_run
from agents.eval.agent import run as eval_run
from agents.serve import agent as serve_agent
from agents.serve.agent import run as serve_run
from core.schema import Chunk
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


class ServeErrorGateChunksTest(unittest.TestCase):
    """Serve 가드는 '서빙할 청크 유무'로 갈린다 — error 라도 청크가 있으면 서빙(High).

    색인·평가는 성공했는데 sweep 하나를 판정 못 해(_fail_active_study no-change) status 가
    error 로 남는 정상 종료 경로가 있다. 그 경우 진단서·인덱스는 멀쩡하므로 서빙해야
    web_api 가 진단서를 보여준다 — error 를 이유로 통째로 건너뛰면 안 된다.
    """

    def _served_error_state(self):
        # 상위가 비치명적 error 인데 서빙할 청크는 보유한 상태.
        return AgentDoctorState(
            status="error", error="sweep 판정 불가(measurement)",
            chunks=[Chunk(chunk_id="c1", doc_id="d1", text="본문")],
        )

    def _run_serve(self, state):
        # 실서버·실파일·Claude 등록 없이 상태 전이만 검증한다.
        with patch.object(serve_agent.Path, "write_text"), \
             patch.object(serve_agent, "write_serve_config", return_value={}), \
             patch.object(serve_agent, "_start_api_server", return_value=True), \
             patch.object(serve_agent, "_register_to_claude_desktop"):
            return _silent(serve_run, state)

    def test_serves_when_error_but_chunks_present(self):
        result = self._run_serve(self._served_error_state())
        self.assertEqual(result.status, "done")          # error 로 막히지 않고 서빙됨
        self.assertIsNone(result.error)                  # 정상 종료로 error 정리
        self.assertIsNotNone(result.mcp_endpoint)        # 엔드포인트가 뜬다

    def test_skips_when_error_and_no_chunks(self):
        # 서빙할 게 없는 진짜 상위 실패는 종전대로 error 유지한 채 건너뛴다.
        state = AgentDoctorState(status="error", error=_REAL_ERROR)  # chunks 기본 []
        result = self._run_serve(state)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error, _REAL_ERROR)
        self.assertIsNone(result.mcp_endpoint)

    def test_router_then_serve_serves_abort_with_chunks(self):
        # 라우터(→serve)와 Serve 가드가 함께 동작하는지 — sweep 판정 불가(error)라도
        # 청크가 있으면 route_after_optimize 가 serve 로 보내고 그 serve 가 서빙한다.
        state = self._served_error_state()
        self.assertEqual(_silent(graph.route_after_optimize, state), "serve")
        result = self._run_serve(state)
        self.assertEqual(result.status, "done")
        self.assertIsNotNone(result.mcp_endpoint)


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
