import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.ingest.preprocess import preprocess_pages
from agents.ingest.tables import extract_page_tables, serialize_table


class SerializeTableTest(unittest.TestCase):
    def test_emits_markdown_and_row_sentences(self):
        rows = [["연도", "실업률"], ["2024", "3.5"], ["2025", "4.1"]]
        out = serialize_table(rows)

        self.assertIn("| 연도 | 실업률 |", out)
        self.assertIn("| --- | --- |", out)
        # 행 문장 — 청크가 잘려도 자립하도록 헤더를 매 행에 반복한다.
        self.assertIn("연도: 2024, 실업률: 3.5", out)
        self.assertIn("연도: 2025, 실업률: 4.1", out)

    def test_none_cells_become_empty(self):
        rows = [["연도", "실업률"], ["2024", None]]
        out = serialize_table(rows)

        self.assertIn("| 2024 |  |", out)
        # 빈 값은 행 문장에서 빠진다 — "실업률: " 같은 노이즈 방지
        self.assertIn("연도: 2024", out)
        self.assertNotIn("실업률: ,", out)

    def test_newline_inside_cell_is_flattened(self):
        rows = [["항목", "설명"], ["가", "여러 줄로\n조판된 셀"]]
        out = serialize_table(rows)

        self.assertIn("여러 줄로 조판된 셀", out)
        # 셀 안 줄바꿈이 남으면 Markdown 표가 깨진다
        self.assertNotIn("여러 줄로\n조판", out)

    def test_pipe_in_cell_is_escaped(self):
        rows = [["a", "b"], ["x|y", "z"]]
        out = serialize_table(rows)
        self.assertIn("x\\|y", out)

    def test_ragged_rows_are_padded(self):
        # 병합 셀 때문에 행마다 길이가 다를 수 있다.
        rows = [["a", "b", "c"], ["1", "2"]]
        out = serialize_table(rows)
        self.assertIn("| 1 | 2 |  |", out)

    def test_caption_is_prepended(self):
        rows = [["a", "b"], ["1", "2"]]
        out = serialize_table(rows, caption="표 1. 연도별 지표")
        self.assertTrue(out.startswith("표 1. 연도별 지표"))

    def test_headerless_table_uses_values_only(self):
        rows = [["", ""], ["2024", "3.5"]]
        out = serialize_table(rows)
        self.assertIn("2024, 3.5", out)


class TableRejectionTest(unittest.TestCase):
    def test_single_row_is_rejected(self):
        self.assertEqual(serialize_table([["a", "b"]]), "")

    def test_single_column_is_rejected(self):
        self.assertEqual(serialize_table([["a"], ["b"], ["c"]]), "")

    def test_all_empty_is_rejected(self):
        self.assertEqual(serialize_table([["", ""], ["", ""]]), "")

    def test_long_cell_rejected_as_prose(self):
        # 본문 문단이 선 안에 들어간 경우 — 표로 오인하면 본문이 망가진다.
        prose = "가" * 250
        self.assertEqual(serialize_table([["a", "b"], [prose, "x"]]), "")

    def test_empty_input_is_rejected(self):
        self.assertEqual(serialize_table([]), "")


class _FakePage:
    def __init__(self, tables=None, raise_exc=False):
        self._tables = tables or []
        self._raise = raise_exc

    def extract_tables(self):
        if self._raise:
            raise RuntimeError("pdfplumber 내부 오류")
        return self._tables


class ExtractPageTablesTest(unittest.TestCase):
    def test_extracts_valid_tables_only(self):
        page = _FakePage([
            [["연도", "값"], ["2024", "3.5"]],   # 유효
            [["단독행"]],                          # 무효 — 버려짐
        ])
        out = extract_page_tables(page)

        self.assertEqual(len(out), 1)
        self.assertIn("연도: 2024", out[0])

    def test_extraction_failure_does_not_raise(self):
        # 표는 부가 정보 — 실패해도 본문 수집을 죽이면 안 된다.
        out = extract_page_tables(_FakePage(raise_exc=True))
        self.assertEqual(out, [])

    def test_no_tables_returns_empty(self):
        self.assertEqual(extract_page_tables(_FakePage([])), [])


class TableInPreprocessTest(unittest.TestCase):
    def test_table_appended_within_its_page_span(self):
        pages = ["1페이지 본문입니다.", "2페이지 본문입니다."]
        tables = [[], ["| 연도 | 값 |\n| --- | --- |\n| 2024 | 3.5 |"]]
        result = preprocess_pages(pages, page_tables=tables)

        self.assertEqual(result.table_count, 1)
        # 표가 2페이지 span 안에 들어가야 청크→페이지 역산이 맞는다
        start, end = result.page_spans[1]
        self.assertIn("2024", result.content[start:end])
        self.assertNotIn("2024", result.content[result.page_spans[0][0]:result.page_spans[0][1]])

    def test_table_survives_body_cleaning(self):
        # _clean_body 가 표를 건드리면 파이프·줄구조가 깨진다.
        table = "| 연도 | 값 |\n| --- | --- |\n| 2024 | 3.5 |"
        result = preprocess_pages(["본문"], page_tables=[[table]])

        self.assertIn("| 연도 | 값 |", result.content)
        self.assertIn("| --- | --- |", result.content)

    def test_table_only_page_is_not_counted_empty(self):
        result = preprocess_pages([None], page_tables=[["| a | b |\n| --- | --- |\n| 1 | 2 |"]])

        self.assertEqual(result.empty_page_count, 0)
        self.assertFalse(result.is_empty)

    def test_no_page_tables_argument_still_works(self):
        result = preprocess_pages(["본문만 있는 문서"])
        self.assertEqual(result.table_count, 0)
        self.assertIn("본문만", result.content)


if __name__ == "__main__":
    unittest.main()
