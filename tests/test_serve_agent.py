"""
tests/test_serve_agent.py
Serve 노드의 API 서버 기동 실패 전파 검증 (mock 기반, 실서버·실포트 없음).

_start_api_server 가 실패(프로세스 조기 종료/타임아웃)를 bool 로 반환하고,
run() 이 이를 받아 status="error" 로 종료하며 Claude Desktop MCP 등록을
건너뛰는지(죽은 엔드포인트 등록 방지) 확인한다.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import Chunk
from core.state import AgentDoctorState
from agents.serve import agent as serve_agent


def _state_with_chunks() -> AgentDoctorState:
    state = AgentDoctorState()
    state.chunks = [Chunk(chunk_id="c1", doc_id="d1", text="테스트 청크")]
    return state


class StartApiServerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        log_patch = patch.object(
            serve_agent, "API_LOG_FILE", Path(self._tmp.name) / "api_server.log"
        )
        log_patch.start()
        self.addCleanup(log_patch.stop)

    def test_already_running_returns_true_without_spawn(self):
        with patch.object(serve_agent, "requests") as mock_requests, \
             patch.object(serve_agent.subprocess, "Popen") as mock_popen:
            mock_requests.get.return_value = MagicMock(status_code=200)
            self.assertTrue(serve_agent._start_api_server())
            mock_popen.assert_not_called()

    def test_running_server_with_matching_fingerprint_skips_reload(self):
        # 실행 중인 서버의 코퍼스 지문이 기대값과 같으면 spawn·reload 없이 통과.
        health = MagicMock(ok=True)
        health.json.return_value = {"fingerprint": "abc123"}
        with patch.object(serve_agent, "requests") as mock_requests, \
             patch.object(serve_agent.subprocess, "Popen") as mock_popen, \
             patch.object(serve_agent, "_reload_api_server") as mock_reload:
            mock_requests.get.return_value = health
            self.assertTrue(serve_agent._start_api_server("abc123"))
            mock_popen.assert_not_called()
            mock_reload.assert_not_called()

    def test_running_server_with_stale_fingerprint_triggers_reload(self):
        # 지문이 다르면(이전 파이프라인의 낡은 코퍼스) /reload 로 갱신한다.
        health = MagicMock(ok=True)
        health.json.return_value = {"fingerprint": "OLD"}
        with patch.object(serve_agent, "requests") as mock_requests, \
             patch.object(serve_agent.subprocess, "Popen") as mock_popen, \
             patch.object(serve_agent, "_reload_api_server", return_value=True) as mock_reload:
            mock_requests.get.return_value = health
            self.assertTrue(serve_agent._start_api_server("NEW"))
            mock_popen.assert_not_called()
            mock_reload.assert_called_once_with("NEW")

    def test_reload_verifies_fingerprint_after_reload(self):
        # /reload 응답 지문이 기대값과 일치해야 성공. 불일치면 실패로 본다.
        ok_resp = MagicMock(ok=True)
        ok_resp.json.return_value = {"fingerprint": "NEW"}
        bad_resp = MagicMock(ok=True)
        bad_resp.json.return_value = {"fingerprint": "STILL_OLD"}
        with patch.object(serve_agent, "requests") as mock_requests:
            mock_requests.post.return_value = ok_resp
            self.assertTrue(serve_agent._reload_api_server("NEW"))
            mock_requests.post.return_value = bad_resp
            self.assertFalse(serve_agent._reload_api_server("NEW"))

    def test_process_early_exit_returns_false(self):
        proc = MagicMock()
        proc.poll.return_value = 1  # 프로세스가 바로 죽음 (예: 포트 바인드 실패)
        proc.returncode = 1
        with patch.object(serve_agent, "requests") as mock_requests, \
             patch.object(serve_agent.subprocess, "Popen", return_value=proc), \
             patch.object(serve_agent, "API_START_TIMEOUT", 5):
            mock_requests.get.side_effect = ConnectionError("no server")
            self.assertFalse(serve_agent._start_api_server())

    def test_health_timeout_returns_false(self):
        proc = MagicMock()
        proc.poll.return_value = None  # 살아 있지만 응답 없음
        with patch.object(serve_agent, "requests") as mock_requests, \
             patch.object(serve_agent.subprocess, "Popen", return_value=proc), \
             patch.object(serve_agent, "API_START_TIMEOUT", 0):
            mock_requests.get.side_effect = ConnectionError("no server")
            self.assertFalse(serve_agent._start_api_server())


class RunTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        chunks_patch = patch.object(
            serve_agent, "CHUNKS_FILE", Path(self._tmp.name) / "chunks.json"
        )
        chunks_patch.start()
        self.addCleanup(chunks_patch.stop)

    def test_server_start_failure_sets_error_and_skips_registration(self):
        state = _state_with_chunks()
        with patch.object(serve_agent, "_start_api_server", return_value=False), \
             patch.object(serve_agent, "_register_to_claude_desktop") as mock_register:
            result = serve_agent.run(state)
        self.assertEqual(result.status, "error")
        self.assertIn("API 서버", result.error)
        self.assertIsNone(result.mcp_endpoint)
        mock_register.assert_not_called()

    def test_server_start_success_completes(self):
        state = _state_with_chunks()
        with patch.object(serve_agent, "_start_api_server", return_value=True), \
             patch.object(serve_agent, "_register_to_claude_desktop") as mock_register:
            result = serve_agent.run(state)
        self.assertEqual(result.status, "done")
        self.assertEqual(result.mcp_endpoint, f"http://localhost:{serve_agent.API_PORT}")
        mock_register.assert_called_once()

    def test_no_chunks_still_errors(self):
        state = AgentDoctorState()  # chunks 비어 있음
        result = serve_agent.run(state)
        self.assertEqual(result.status, "error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
