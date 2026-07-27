"""
tests/test_run_corpus.py
run_corpus.write_report 가 만든 진단서가 "서버 없이 실제 데이터로" 렌더되는지 검증한다.

이 주입은 web/prototype/report.html 의 내부 구조(데이터 로딩 분기, 스크립트 순서)에
의존한다. 템플릿이 바뀌면 조용히 빈 리포트/더미가 뜰 수 있어서 계약을 못박아 둔다.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import DiagnosticReport, Finding
from core.state import AgentDoctorState
from tests.run_corpus import find_source_doc, write_report


def make_state():
    report = DiagnosticReport(
        report_id="r",
        findings=[Finding(finding_id="1", type="retrieval_failure", severity="warning",
                          description="d", label="too_long_context", affected_probes=["p1"])],
        overall_score=0.9,
        ragas_scores={"context_recall": 0.7},
        composite_score={"total": 73.5, "components": []},
        pass_threshold=False,
    )
    return AgentDoctorState(report=report, source_url="sample.pdf")


class WriteReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.corpus = Path(self.tmp.name)
        self.html_path, self.view = write_report(make_state(), self.corpus)
        self.html = self.html_path.read_text(encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_html_and_json(self):
        self.assertTrue(self.html_path.exists())
        json_path = self.corpus / "out" / "report.json"
        self.assertTrue(json_path.exists())
        self.assertEqual(
            json.loads(json_path.read_text(encoding="utf-8"))["score"]["after"], 73.5
        )

    def test_data_defined_before_it_is_consumed(self):
        """데이터 블록이 렌더 호출보다 뒤에 오면 렌더 시점엔 undefined → 빈 리포트가 된다."""
        definition = self.html.find("window.__AGENT_DOCTOR_REPORT__ =")
        consumer = self.html.find("renderReport(window.__AGENT_DOCTOR_REPORT__")
        self.assertNotEqual(definition, -1, "데이터 블록이 주입되지 않았다")
        self.assertNotEqual(consumer, -1, "렌더 호출이 주입되지 않았다")
        self.assertLess(definition, consumer, "데이터가 소비 지점보다 뒤에 정의됐다")

    def test_server_fetch_branch_is_removed(self):
        """서버가 없으므로 fetch 가 남아 있으면 실패 배너가 뜬다. 더미 렌더도 남으면 안 된다."""
        self.assertNotIn("fetch(WEB_API_BASE", self.html)
        self.assertNotIn("renderReport({}, false)", self.html)

    def test_injected_payload_matches_view(self):
        line = next(
            l for l in self.html.splitlines()
            if l.startswith("window.__AGENT_DOCTOR_REPORT__ =")
        )
        payload = json.loads(
            line[len("window.__AGENT_DOCTOR_REPORT__ = "):].rstrip().rstrip(";")
        )
        self.assertEqual(payload["score"]["after"], self.view["score"]["after"])
        self.assertEqual(payload["dxs"][0]["code"], "too_long_context")

    def test_payload_does_not_break_script_parsing(self):
        """페이로드에 </script> 가 섞여도 스크립트 블록이 조기 종료되면 안 된다."""
        state = make_state()
        state.source_url = "</script><b>x</b>.pdf"
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = write_report(state, Path(tmp))
            html = path.read_text(encoding="utf-8")
        data_start = html.find("window.__AGENT_DOCTOR_REPORT__ =")
        next_close = html.find("</script>", data_start)
        # 데이터 라인 안의 </ 는 이스케이프되어야 하므로, 줄 끝 이전에 </script> 가 없어야 한다.
        line_end = html.find("\n", data_start)
        self.assertGreater(next_close, line_end, "페이로드가 script 블록을 조기 종료시킨다")


class FindSourceDocTest(unittest.TestCase):
    """평평한 tests/corpus/ 에서 원본 문서를 고르는 규칙."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_picks_the_only_document(self):
        (self.dir / "설명회.pdf").write_text("x", encoding="utf-8")
        self.assertEqual(find_source_doc(self.dir).name, "설명회.pdf")

    def test_readme_is_not_treated_as_corpus(self):
        """README.md 는 .md 라 이름순 첫 번째로 뽑히기 쉽다 — 원본으로 쓰면 안 된다."""
        (self.dir / "README.md").write_text("문서 설명", encoding="utf-8")
        (self.dir / "설명회.pdf").write_text("x", encoding="utf-8")
        self.assertEqual(find_source_doc(self.dir).name, "설명회.pdf")

    def test_ignores_qa_json_and_unsupported_files(self):
        (self.dir / "qa.json").write_text("{}", encoding="utf-8")
        (self.dir / "메모.docx").write_text("x", encoding="utf-8")
        (self.dir / "본문.txt").write_text("x", encoding="utf-8")
        self.assertEqual(find_source_doc(self.dir).name, "본문.txt")

    def test_no_document_exits_with_guidance(self):
        (self.dir / "README.md").write_text("설명뿐", encoding="utf-8")
        with self.assertRaises(SystemExit) as ctx:
            find_source_doc(self.dir)
        self.assertIn("원본 문서가 없습니다", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
