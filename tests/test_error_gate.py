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
from agents.eval import agent as eval_agent
from agents.eval.agent import run as eval_run
from agents.serve import agent as serve_agent
from agents.serve.agent import run as serve_run
from core.schema import Chunk, DiagnosticReport
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
    """Serve 가드는 '청크 + 진단서' 유무로 갈린다.

    비치명(sweep 판정 불가, _fail_active_study no-change)이면 청크·진단서가 둘 다 남아
    있으니 서빙한다. Eval 이 실제로 죽었으면 인덱스가 남아 있어도 진단서가 없으므로
    서빙하지 않고 error 를 유지한다 — 여기서 done 으로 덮으면 실패가 "완료" 로 보고된다.
    """

    def _served_error_state(self):
        # 비치명적 error — 청크와 진단서를 모두 보유한 상태.
        return AgentDoctorState(
            status="error", error="sweep 판정 불가(measurement)",
            chunks=[Chunk(chunk_id="c1", doc_id="d1", text="본문")],
            report=DiagnosticReport(report_id="r1", overall_score=80.0),
        )

    def _run_serve(self, state):
        # 실서버·실파일·Claude 등록 없이 상태 전이만 검증한다.
        with patch.object(serve_agent.Path, "write_text"), \
             patch.object(serve_agent, "write_serve_config", return_value={}), \
             patch.object(serve_agent, "_start_api_server", return_value=True), \
             patch.object(serve_agent, "_register_to_claude_desktop"):
            return _silent(serve_run, state)

    def test_serves_when_error_but_chunks_and_report_present(self):
        result = self._run_serve(self._served_error_state())
        self.assertEqual(result.status, "done")          # error 로 막히지 않고 서빙됨
        self.assertIsNotNone(result.mcp_endpoint)        # 엔드포인트가 뜬다
        # 사유는 지우지 않는다 — web_api 는 status 로만 분기하므로 남겨도 500 이 아니고,
        # 지우면 왜 error 였는지 볼 방법이 사라진다.
        self.assertEqual(result.error, "sweep 판정 불가(measurement)")

    def test_skips_when_error_and_no_chunks(self):
        # 서빙할 게 없는 진짜 상위 실패는 종전대로 error 유지한 채 건너뛴다.
        state = AgentDoctorState(status="error", error=_REAL_ERROR)  # chunks 기본 []
        result = self._run_serve(state)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error, _REAL_ERROR)
        self.assertIsNone(result.mcp_endpoint)

    def test_skips_when_eval_crashed_even_with_chunks(self):
        # Eval 크래시(진단서 없음) — 인덱스가 남아 있어도 서빙하지 않고 실패로 보고한다.
        state = AgentDoctorState(
            status="error", error="평가 실패: GEMINI API 인증 실패",
            chunks=[Chunk(chunk_id="c1", doc_id="d1", text="본문")],
        )
        result = self._run_serve(state)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error, "평가 실패: GEMINI API 인증 실패")
        self.assertIsNone(result.mcp_endpoint)

    def test_eval_crash_clears_stale_report(self):
        # 재색인 뒤 2회차 Eval 크래시 — 1회차 report 가 남아 있으면 가드가 "서빙할 진단서
        # 있음"으로 오판한다. Eval 이 실패하면서 report 를 비우므로 그 경로가 막힌다.
        state = AgentDoctorState(
            source_url="x", source_type="gdrive", iteration=2,
            chunks=[Chunk(chunk_id="c1", doc_id="d1", text="본문")],
            report=DiagnosticReport(report_id="r1", overall_score=80.0, iteration=1),
        )
        crash = RuntimeError("API 인증 실패")
        with patch.object(eval_agent, "load_probes", side_effect=crash), \
             patch.object(eval_agent, "generate_probes", side_effect=crash):
            state = _silent(eval_run, state)
        self.assertEqual(state.status, "error")
        self.assertIsNone(state.report)          # 옛 회차 성적표를 들고 가지 않는다

        result = self._run_serve(state)
        self.assertEqual(result.status, "error")  # "완료"로 뒤집히지 않는다
        self.assertIn("API 인증 실패", result.error)
        self.assertIsNone(result.mcp_endpoint)

    def test_serve_start_failure_keeps_prior_cause(self):
        # 비치명 error 를 안고 서빙하다 API 서버가 안 뜨면, 선행 사유도 함께 남는다.
        with patch.object(serve_agent.Path, "write_text"), \
             patch.object(serve_agent, "write_serve_config", return_value={}), \
             patch.object(serve_agent, "_start_api_server", return_value=False):
            result = _silent(serve_run, self._served_error_state())
        self.assertEqual(result.status, "error")
        self.assertIn("Serve 실패", result.error)
        self.assertIn("sweep 판정 불가", result.error)

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
