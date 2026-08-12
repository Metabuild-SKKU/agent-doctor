"""
tests/test_web_replay.py
웹 "기존 RAG 로그 진단" 경로(POST /runs mode=replay)의 계약 검증.

리뷰 지적: 리플레이 경로만 테스트가 비어 있었다 — create_run(mode="replay") ·
_run_replay_background · run_report 의 리플레이 분기 전부. 실제 진단(RAGAS·LLM)은
부르지 않고 diagnose_external_log 를 대역으로 바꿔, 웹 계층이 지키는 것만 본다:
파일 게이트 · 크기 상한 · 상태 전이 · 뷰 선택.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from agents.eval.replay import GOLDEN_HARD_CAP, LOG_HARD_CAP
from agents.serve import web_api
from agents.serve.web_api import app
from core import run_registry


def _log_lines(n: int, *, inline_golden: bool = True) -> str:
    """유효한 triad 로그 n줄.

    기본으로 정답을 인라인에 넣는다 - 웹 경로는 골든셋 면제가 없어서(옵트아웃 없음),
    정답 없는 로그는 업로드 게이트에서 막힌다. 게이트 자체를 보는 테스트만 False 로 준다.
    """
    def line(i):
        obj = {
            "question": f"질문 {i}",
            "contexts": [f"근거 문단 {i} — 연금저축 세액공제 한도는 연 700만원이다."],
            "answer": f"답변 {i}",
        }
        if inline_golden:
            obj["ground_truth"] = f"정답 {i}"
        return json.dumps(obj, ensure_ascii=False)
    return "\n".join(line(i) for i in range(n))


class _ReplayClient(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        # 업로드는 레포 밖 임시 디렉터리로 — 테스트가 uploads/ 를 더럽히지 않게.
        patcher = patch.object(web_api, "UPLOAD_DIR", Path(self._tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(app)

    def _post(self, log_body: str, *, filename="log.jsonl", golden=None):
        files = {"logfile": (filename, log_body.encode("utf-8"), "application/json")}
        if golden is not None:
            gname, gbody = golden
            files["goldenfile"] = (gname, gbody.encode("utf-8"), "application/json")
        return self.client.post("/runs", data={"mode": "replay"}, files=files)


class UploadGateTests(_ReplayClient):
    def test_rejects_non_jsonl_extension(self):
        res = self._post(_log_lines(1), filename="log.pdf")
        self.assertEqual(res.status_code, 400)
        self.assertIn("JSONL", res.json()["detail"])

    def test_rejects_log_over_hard_cap(self):
        """CLI 는 --limit 으로 표본을 줄일 수 있지만 웹에는 그 손잡이가 없다.
        _PIPELINE_LOCK 을 쥔 채 전 레코드 RAGAS 를 도는 구조라 상한이 필요하다."""
        res = self._post(_log_lines(LOG_HARD_CAP + 1))
        self.assertEqual(res.status_code, 400)
        self.assertIn(str(LOG_HARD_CAP), res.json()["detail"])

    def test_accepts_log_at_hard_cap(self):
        """상한은 초과부터 막는다 — 딱 맞는 크기를 거부하면 off-by-one 이다."""
        with patch.object(web_api, "_run_replay_background"):
            res = self._post(_log_lines(LOG_HARD_CAP))
        self.assertEqual(res.status_code, 200)

    def test_rejects_golden_over_hard_cap(self):
        golden = "\n".join(json.dumps({"question": f"질문 {i}", "ground_truth": f"정답 {i}"},
                                      ensure_ascii=False)
                           for i in range(GOLDEN_HARD_CAP + 1))
        res = self._post(_log_lines(2), golden=("golden.jsonl", golden))
        self.assertEqual(res.status_code, 400)
        self.assertIn(str(GOLDEN_HARD_CAP), res.json()["detail"])

    def test_rejects_unsupported_golden_extension(self):
        res = self._post(_log_lines(1), golden=("golden.txt", "질문,정답"))
        self.assertEqual(res.status_code, 400)
        self.assertIn("골든셋", res.json()["detail"])

    def test_no_run_is_registered_when_gate_rejects(self):
        """게이트에 걸린 업로드가 run 을 남기면 프론트가 영원히 폴링한다."""
        before = len(run_registry.list_runs()) if hasattr(run_registry, "list_runs") else None
        res = self._post(_log_lines(LOG_HARD_CAP + 1))
        self.assertEqual(res.status_code, 400)
        self.assertIsNone(res.json().get("run_id"))
        if before is not None:
            self.assertEqual(len(run_registry.list_runs()), before)


class GoldenGateTests(_ReplayClient):
    """웹 경로에는 골든셋 면제가 없다.

    정답지가 없으면 신뢰도 축을 못 재 종합점수 자체가 안 나오는데, 원인 7종 중 3종만
    담긴 '점수 없는 진단서'를 받아가는 건 오해만 만든다. 정답지를 아직 못 만든 경우는
    CLI 의 --no-golden 이 개발용 통로다(그쪽은 그대로 둔다)."""

    def test_rejects_when_no_golden_anywhere(self):
        res = self._post(_log_lines(2, inline_golden=False))
        self.assertEqual(res.status_code, 400)
        self.assertIn("골든셋", res.json()["detail"])

    def test_no_opt_out_parameter_is_honored(self):
        """예전에는 no_golden=1 로 빠져나갈 수 있었다. 그 통로를 없앤 것이 이 변경이다."""
        files = {"logfile": ("log.jsonl",
                             _log_lines(2, inline_golden=False).encode("utf-8"),
                             "application/json")}
        res = self.client.post("/runs", data={"mode": "replay", "no_golden": "1"},
                               files=files)
        self.assertEqual(res.status_code, 400)

    def test_inline_ground_truth_counts_as_golden(self):
        """로그에 정답이 인라인으로 있으면 골든셋이 없는 게 아니다(CLI 와 같은 판정)."""
        with patch.object(web_api, "_run_replay_background"):
            res = self._post(_log_lines(2))
        self.assertEqual(res.status_code, 200)

    def test_golden_file_satisfies_the_gate(self):
        golden = json.dumps({"question": "질문 0", "ground_truth": "정답"}, ensure_ascii=False)
        with patch.object(web_api, "_run_replay_background"):
            res = self._post(_log_lines(2, inline_golden=False),
                             golden=("golden.jsonl", golden))
        self.assertEqual(res.status_code, 200)

    def test_rejects_golden_that_matches_nothing(self):
        """표기가 달라 한 건도 안 붙는 골든셋. 그대로 두면 전량 RAGAS 를 돌린 뒤에야
        '정답 0건' 리포트가 나온다 - 비싸고, 사용자는 대조된 줄 안다."""
        golden = json.dumps({"question": "전혀 다른 질문입니다", "ground_truth": "정답"},
                            ensure_ascii=False)
        res = self._post(_log_lines(2, inline_golden=False),
                         golden=("golden.jsonl", golden))
        self.assertEqual(res.status_code, 400)
        self.assertIn("매칭", res.json()["detail"])

    def test_partial_match_is_allowed(self):
        """부분 매칭은 막지 않는다 - 0건만 막는다. 매칭률은 진단서가 밝힌다."""
        golden = json.dumps({"question": "질문 0", "ground_truth": "정답"}, ensure_ascii=False)
        with patch.object(web_api, "_run_replay_background"):
            res = self._post(_log_lines(5, inline_golden=False),
                             golden=("golden.jsonl", golden))
        self.assertEqual(res.status_code, 200)


class BackgroundRunTests(_ReplayClient):
    def _run_background(self, diagnose_return):
        """_run_replay_background 를 동기로 한 번 돌리고 run 상태를 돌려준다."""
        run_id = "test-replay-run"
        run_registry.create(run_id, depth="full", upload_path="x.jsonl",
                            created_at=0.0, mode="replay")
        log_path = Path(self._tmp.name) / "log.jsonl"
        log_path.write_text(_log_lines(2), encoding="utf-8")

        with patch.object(web_api, "diagnose_external_log", return_value=diagnose_return), \
             patch("core.run_logger.setup_run_logging"):
            web_api._run_replay_background(run_id, log_path, None)
        return run_registry.get(run_id)

    def test_missing_context_tier_becomes_error(self):
        """qa_only(컨텍스트 없음)는 리포트가 안 나온다 — 빈 진단서를 내보내지 않고
        run 을 error 로 남겨 프론트가 사유를 표시하게 한다."""
        run = self._run_background((None, {"tier": "qa_only"}, []))
        self.assertEqual(run.status, "error")
        self.assertIn("contexts", run.error)

    def test_no_valid_record_becomes_error(self):
        run = self._run_background((None, {"tier": "none"}, []))
        self.assertEqual(run.status, "error")
        self.assertIn("유효한 로그 레코드", run.error)

    def test_report_is_stored_on_success(self):
        report = SimpleNamespace(findings=[], findings_summary={"confirmed": 2})
        cap = {"tier": "triad", "records": 2}
        run = self._run_background((report, cap, []))
        self.assertEqual(run.status, "done")
        self.assertEqual(run.percent, 100)
        self.assertIs(run.ext_report, report)
        self.assertEqual(run.ext_cap, cap)

    def test_preloaded_parse_is_reused(self):
        """create_run 이 게이트 검사로 이미 파싱한 것을 백그라운드가 다시 읽지 않는다 —
        diagnose_external_log 의 logs=/qa= 파라미터가 그 용도다."""
        seen = {}

        def fake_diagnose(path, **kwargs):
            seen.update(kwargs)
            return (None, {"tier": "none"}, [])

        with patch.object(web_api, "diagnose_external_log", side_effect=fake_diagnose), \
             patch.object(web_api, "_run_replay_background",
                          wraps=web_api._run_replay_background), \
             patch("core.run_logger.setup_run_logging"):
            res = self._post(_log_lines(3))

        self.assertEqual(res.status_code, 200)
        self.assertIsNotNone(seen.get("logs"))
        self.assertEqual(len(seen["logs"]), 3)

    def test_forces_deep_llm_regardless_of_form_depth(self):
        """리플레이는 깊이 선택이 없다 — LLM 을 끄면 생성축 라벨 4종이 통째로 죽어
        '점수는 낮은데 소견 0건' 인 진단서가 나간다."""
        with patch.dict(os.environ, {"EVAL_MODE": "fast", "EVAL_ENABLE_LLM": "0"}):
            self._run_background((None, {"tier": "none"}, []))
            self.assertEqual(os.environ["EVAL_MODE"], "deep")
            self.assertEqual(os.environ["EVAL_ENABLE_LLM"], "1")


class ReportRoutingTests(_ReplayClient):
    def _make_run(self, **fields):
        run_id = "test-replay-report"
        run_registry.create(run_id, depth="full", upload_path="x.jsonl",
                            created_at=0.0, mode="replay")
        run_registry.update(run_id, **fields)
        return run_id

    def test_replay_run_renders_with_ext_view(self):
        """state 가 없는 리플레이 모드라 build_report_view 와 경로가 갈린다."""
        run_id = self._make_run(status="done", ext_report=object(),
                                ext_cap={"tier": "triad"})
        with patch.object(web_api, "build_ext_report_view",
                          return_value={"mode": {"kind": "external"}}) as ext_view, \
             patch.object(web_api, "build_report_view") as internal_view:
            res = self.client.get(f"/runs/{run_id}/report")

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["mode"]["kind"], "external")
        ext_view.assert_called_once()
        internal_view.assert_not_called()

    def test_unfinished_replay_run_is_409(self):
        run_id = self._make_run(status="running")
        self.assertEqual(self.client.get(f"/runs/{run_id}/report").status_code, 409)

    def test_errored_replay_run_is_500_with_reason(self):
        run_id = self._make_run(status="error", error="컨텍스트가 없습니다")
        res = self.client.get(f"/runs/{run_id}/report")
        self.assertEqual(res.status_code, 500)
        self.assertIn("컨텍스트", res.json()["detail"])


if __name__ == "__main__":
    unittest.main()
