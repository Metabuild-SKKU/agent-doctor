"""
tests/test_ingest_progress.py
Ingest PDF 두 패스(본문·표)의 진행률 배선을 고정한다.

왜 필요한가:
    이 경로가 이 기능의 주요 동기 중 하나인데(878페이지 본문 추출 실측 2m36s)
    parallel_map·임베딩에 비해 테스트가 비어 있었다(리뷰 지적 #4).

    실제 PDF 를 열지 않는다 — pdfplumber 대역으로 페이지 수만 흉내내면 여기서
    확인할 것(패스마다 page_count 만큼 tick 되는가, 짧은 문서는 조용한가,
    PROGRESS_LOG=0 이면 침묵하는가)은 전부 잡힌다. 878페이지 실측은 수동으로 이미
    돌렸고, 그걸 매 테스트마다 반복할 이유는 없다.
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.ingest import agent as ingest_agent


class _FakePage:
    """페이지마다 서로 다른 여러 줄을 돌려준다.

    한 줄짜리 페이지를 쓰면 preprocess_pages 의 머리말/꼬리말 제거가 그 줄을
    반복 요소로 보고 지워, 문서 전체가 is_empty 로 판정돼 ValueError 가 난다."""

    def __init__(self, index: int):
        self._index = index

    def extract_text(self):
        return "\n".join(
            f"{self._index}쪽 {n}번째 문단. 진행률 배선을 확인하기 위한 더미 본문이며 "
            f"머리말 제거에 걸리지 않도록 페이지마다 다른 내용을 담는다."
            for n in range(4)
        )


class _FakePdf:
    """pdfplumber.open(...) 의 컨텍스트 매니저 대역."""

    def __init__(self, page_count: int):
        self.pages = [_FakePage(i) for i in range(page_count)]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _ProgressSpy:
    """progress.start 가 만든 리포터를 가로채 tick 수를 센다."""

    def __init__(self, label, total):
        self.label = label
        self.total = total
        self.ticks = 0
        self.finished = False
        self.aborted = False

    def tick(self, count=1):
        self.ticks += count

    def finish(self):
        self.finished = True

    def abort(self, reason=""):
        self.aborted = True


class _SpyingProgress:
    """core.progress 모듈 대역 — start 로 만들어진 리포터를 전부 모아 둔다."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.reporters: list[_ProgressSpy] = []

    def start(self, label, total):
        if not self.enabled or not label or total <= 0:
            return None
        spy = _ProgressSpy(label, total)
        self.reporters.append(spy)
        return spy

    @staticmethod
    def tick(reporter, count=1):
        if reporter is not None:
            reporter.tick(count)

    @staticmethod
    def finish(reporter):
        if reporter is not None:
            reporter.finish()

    @staticmethod
    def abort(reporter, reason=""):
        if reporter is not None:
            reporter.abort(reason)


class _ExtractPagesTest(unittest.TestCase):
    """_extract_pages 는 두 패스가 공유하는 진행률 배선이다."""

    def test_페이지마다_한_번씩_tick_한다(self):
        spy = _SpyingProgress()
        pdf = _FakePdf(50)
        with mock.patch.object(ingest_agent, "progress", spy):
            result = ingest_agent._extract_pages(
                pdf, len(pdf.pages), "[테스트] 본문", lambda p: p.extract_text())

        self.assertEqual(len(result), 50)
        self.assertEqual(len(spy.reporters), 1)
        self.assertEqual(spy.reporters[0].ticks, 50)
        self.assertEqual(spy.reporters[0].total, 50)
        self.assertTrue(spy.reporters[0].finished)

    def test_결과는_페이지_순서를_보존한다(self):
        spy = _SpyingProgress()
        pdf = _FakePdf(10)
        with mock.patch.object(ingest_agent, "progress", spy):
            result = ingest_agent._extract_pages(
                pdf, len(pdf.pages), "[테스트] 본문", lambda p: p.extract_text())
        self.assertEqual(result, [page.extract_text() for page in pdf.pages])
        # 순서가 섞이면 청크→페이지 역산(Index 의 _page_of_span)이 통째로 어긋난다.
        for i, text in enumerate(result):
            self.assertTrue(text.startswith(f"{i}쪽 "), f"{i}번째 자리에 {text[:10]!r}")

    def test_예외가_나면_중단으로_닫고_전파한다(self):
        spy = _SpyingProgress()
        pdf = _FakePdf(10)

        def boom(page):
            if page._index == 4:
                raise ValueError("손상된 페이지")
            return page.extract_text()

        with mock.patch.object(ingest_agent, "progress", spy):
            with self.assertRaises(ValueError):
                ingest_agent._extract_pages(pdf, len(pdf.pages), "[테스트] 본문", boom)

        self.assertTrue(spy.reporters[0].aborted)
        self.assertFalse(spy.reporters[0].finished)

    def test_progress_가_꺼져_있으면_리포터_없이_돈다(self):
        spy = _SpyingProgress(enabled=False)
        pdf = _FakePdf(10)
        with mock.patch.object(ingest_agent, "progress", spy):
            result = ingest_agent._extract_pages(
                pdf, len(pdf.pages), "[테스트] 본문", lambda p: p.extract_text())
        self.assertEqual(len(result), 10)
        self.assertEqual(spy.reporters, [])


class PdfTwoPassProgressTest(unittest.TestCase):
    """_ingest_file 의 PDF 분기 — 본문/표 두 패스가 각각 세는지."""

    def _run(self, page_count: int, spy: _SpyingProgress, tmp_name="sample.pdf"):
        fake_pdfplumber = types.ModuleType("pdfplumber")
        fake_pdfplumber.open = lambda path: _FakePdf(page_count)

        # 표는 없는 문서로 둔다 — 여기서 보는 건 표 내용이 아니라 tick 횟수다.
        with mock.patch.dict(sys.modules, {"pdfplumber": fake_pdfplumber}), \
             mock.patch.object(ingest_agent, "progress", spy), \
             mock.patch("agents.ingest.tables.extract_page_tables", return_value=[]), \
             mock.patch("pathlib.Path.exists", return_value=True), \
             mock.patch("builtins.print"):
            return ingest_agent._ingest_file(tmp_name)

    def test_두_패스가_각각_page_count_만큼_tick_한다(self):
        spy = _SpyingProgress()
        self._run(120, spy)

        self.assertEqual(len(spy.reporters), 2, "본문·표 두 패스가 각각 리포터를 연다")
        본문, 표 = spy.reporters
        self.assertIn("본문 추출", 본문.label)
        self.assertIn("표 추출", 표.label)
        self.assertEqual(본문.ticks, 120)
        self.assertEqual(표.ticks, 120)
        self.assertTrue(본문.finished and 표.finished)

    def test_라벨에_파일명이_들어간다(self):
        spy = _SpyingProgress()
        self._run(5, spy, tmp_name="보고서.pdf")
        for reporter in spy.reporters:
            self.assertIn("보고서.pdf", reporter.label)

    def test_progress_가_꺼져_있으면_추출은_그대로_된다(self):
        """진행률은 부가 기능이라 꺼도 문서 내용이 달라지면 안 된다."""
        on, off = _SpyingProgress(), _SpyingProgress(enabled=False)
        docs_on = self._run(30, on)
        docs_off = self._run(30, off)

        self.assertEqual(off.reporters, [])
        self.assertEqual(len(docs_on), len(docs_off))
        self.assertEqual(docs_on[0].content, docs_off[0].content)


class PdfProgressSilenceTest(unittest.TestCase):
    """짧은 PDF 는 완료줄조차 안 찍는다 — 진짜 progress 모듈로 확인한다."""

    def test_짧은_pdf_는_완료줄만_덜렁_찍지_않는다(self):
        fake_pdfplumber = types.ModuleType("pdfplumber")
        fake_pdfplumber.open = lambda path: _FakePdf(3)
        lines: list[str] = []

        with mock.patch.dict(os.environ, {"PROGRESS_LOG": "1",
                                          "PROGRESS_MIN_INTERVAL_SEC": "3600"}), \
             mock.patch.dict(sys.modules, {"pdfplumber": fake_pdfplumber}), \
             mock.patch("agents.ingest.tables.extract_page_tables", return_value=[]), \
             mock.patch("pathlib.Path.exists", return_value=True), \
             mock.patch("builtins.print",
                        side_effect=lambda *a, **k: lines.append(" ".join(map(str, a)))):
            ingest_agent._ingest_file("sample.pdf")

        진행줄 = [line for line in lines if "추출" in line and "%" in line]
        self.assertEqual(진행줄, [], f"짧은 PDF 가 조용하지 않다: {진행줄}")

    def test_PROGRESS_LOG_0_이면_진행줄이_없다(self):
        fake_pdfplumber = types.ModuleType("pdfplumber")
        fake_pdfplumber.open = lambda path: _FakePdf(500)
        lines: list[str] = []

        with mock.patch.dict(os.environ, {"PROGRESS_LOG": "0",
                                          "PROGRESS_MIN_INTERVAL_SEC": "0.001"}), \
             mock.patch.dict(sys.modules, {"pdfplumber": fake_pdfplumber}), \
             mock.patch("agents.ingest.tables.extract_page_tables", return_value=[]), \
             mock.patch("pathlib.Path.exists", return_value=True), \
             mock.patch("builtins.print",
                        side_effect=lambda *a, **k: lines.append(" ".join(map(str, a)))):
            ingest_agent._ingest_file("sample.pdf")

        진행줄 = [line for line in lines if "%" in line and "추출" in line]
        self.assertEqual(진행줄, [], f"꺼져 있는데 진행줄이 나왔다: {진행줄}")


if __name__ == "__main__":
    unittest.main()
