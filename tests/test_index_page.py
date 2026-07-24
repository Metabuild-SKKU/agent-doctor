import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.index.agent import _page_of_span
from core.schema import Document


def _doc(page_spans):
    return Document(
        doc_id="d1",
        source="x.pdf",
        format="pdf",
        content="",
        metadata={"filename": "x.pdf", "page_spans": page_spans},
    )


class PageOfSpanTest(unittest.TestCase):
    def test_maps_span_to_page_number(self):
        doc = _doc([[0, 34], [36, 81], [83, 107]])

        self.assertEqual(_page_of_span(doc, (0, 10)), 1)
        self.assertEqual(_page_of_span(doc, (40, 60)), 2)
        self.assertEqual(_page_of_span(doc, (90, 100)), 3)

    def test_chunk_crossing_pages_reports_starting_page(self):
        # 인용 표기 목적이라 사람이 찾아갈 첫 페이지를 준다.
        doc = _doc([[0, 34], [36, 81]])
        self.assertEqual(_page_of_span(doc, (30, 50)), 1)

    def test_offset_in_page_separator_falls_back_to_previous_page(self):
        # 35 는 1페이지 끝(34)과 2페이지 시작(36) 사이의 구분자 위치.
        doc = _doc([[0, 34], [36, 81]])
        self.assertEqual(_page_of_span(doc, (35, 50)), 1)

    def test_missing_page_spans_returns_none(self):
        doc = Document(doc_id="d", source="a.txt", format="txt", content="", metadata={})
        self.assertIsNone(_page_of_span(doc, (0, 10)))

    def test_malformed_page_spans_returns_none(self):
        doc = _doc("not-a-list")
        self.assertIsNone(_page_of_span(doc, (0, 10)))

    def test_malformed_entry_keeps_positional_numbering(self):
        # 깨진 항목도 페이지 한 자리를 차지한다. 건너뛰면서 번호까지 당기면
        # 그 뒤 페이지가 전부 하나씩 밀려 잘못된 출처를 표기하게 된다.
        doc = _doc([[0, 34], "junk", [36, 81]])
        self.assertEqual(_page_of_span(doc, (40, 50)), 3)

    def test_offset_past_last_page_returns_last_page(self):
        doc = _doc([[0, 34], [36, 81]])
        self.assertEqual(_page_of_span(doc, (500, 510)), 2)

    def test_empty_page_spans_returns_none(self):
        doc = _doc([])
        self.assertIsNone(_page_of_span(doc, (0, 10)))


if __name__ == "__main__":
    unittest.main()
