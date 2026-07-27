import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.ingest.preprocess import PAGE_SEPARATOR, preprocess_pages


class HeaderFooterStripTest(unittest.TestCase):
    def test_removes_running_header_repeated_across_pages(self):
        pages = [
            "고용동향 보고서\n1장 개요입니다.\n1",
            "고용동향 보고서\n2장 내용입니다.\n2",
            "고용동향 보고서\n3장 결론입니다.\n3",
        ]
        result = preprocess_pages(pages)

        self.assertNotIn("고용동향 보고서", result.content)
        self.assertIn("1장 개요입니다.", result.content)
        self.assertIn("3장 결론입니다.", result.content)

    def test_removes_page_numbers_in_various_shapes(self):
        pages = [
            "본문 가나다\n- 12 -",
            "본문 라마바\nPage 13",
            "본문 사아자\n14 / 30",
        ]
        result = preprocess_pages(pages)

        self.assertNotIn("- 12 -", result.content)
        self.assertNotIn("Page 13", result.content)
        self.assertNotIn("14 / 30", result.content)
        self.assertIn("본문 라마바", result.content)

    def test_header_with_varying_page_number_is_still_detected(self):
        # "3장 서론 · 12" 처럼 숫자만 바뀌는 머리말도 같은 머리말로 봐야 한다.
        pages = [
            f"3장 서론 · {i}\n본문 {i} 입니다.\n이어지는 설명 문장입니다."
            for i in range(1, 5)
        ]
        result = preprocess_pages(pages)

        self.assertNotIn("3장 서론", result.content)
        self.assertIn("본문 3 입니다.", result.content)

    def test_short_document_keeps_everything(self):
        # 2페이지짜리는 "반복"을 신뢰할 수 없으므로 머리말을 지우지 않는다.
        pages = ["제목\n본문 하나", "제목\n본문 둘"]
        result = preprocess_pages(pages)

        self.assertEqual(result.content.count("제목"), 2)

    def test_repeated_body_text_is_not_stripped(self):
        # 가장자리가 아닌 본문 중간에 반복되는 문장은 살아남아야 한다.
        long_line = "이 문장은 본문 한가운데에서 반복되는 충분히 긴 서술형 문장입니다."
        pages = [f"머리말\n{long_line}\n페이지 {i} 고유 내용\n{i}" for i in range(1, 5)]
        result = preprocess_pages(pages)

        self.assertEqual(result.content.count(long_line), 4)
        self.assertNotIn("머리말", result.content)


class LineBreakRepairTest(unittest.TestCase):
    def test_joins_hyphenated_english_word(self):
        result = preprocess_pages(["The infor-\nmation is here."])
        self.assertIn("information is here.", result.content)

    def test_does_not_join_list_bullets(self):
        result = preprocess_pages(["항목 목록:\n- 첫째 항목\n- 둘째 항목"])
        self.assertIn("- 첫째 항목", result.content)

    def test_hangul_line_break_becomes_space_not_fusion(self):
        # 줄바꿈이 단어 사이인지 중간인지 알 수 없으므로 공백으로 편다.
        # 붙여버리면 "없이줄이" 같은 없는 단어가 생겨 임베딩·F1 이 망가진다.
        result = preprocess_pages(["한국어는 하이픈 없이\n줄이 바뀝니다."])

        self.assertIn("없이 줄이", result.content)
        self.assertNotIn("없이줄이", result.content)
        self.assertNotIn("\n", result.content)

    def test_strips_invisible_characters(self):
        result = preprocess_pages(["soft­hyphen and zero​width"])
        self.assertIn("softhyphen", result.content)
        self.assertIn("zerowidth", result.content)

    def test_collapses_excess_blank_lines(self):
        result = preprocess_pages(["가나다\n\n\n\n\n라마바"])
        self.assertNotIn("\n\n\n", result.content)


class PageSpanTest(unittest.TestCase):
    def test_page_spans_point_at_correct_slices(self):
        pages = ["첫째 페이지 본문", "둘째 페이지 본문", "셋째 페이지 본문"]
        result = preprocess_pages(pages)

        self.assertEqual(len(result.page_spans), 3)
        for idx, (start, end) in enumerate(result.page_spans):
            self.assertEqual(result.content[start:end], pages[idx])

    def test_page_of_maps_offset_back_to_page_number(self):
        pages = ["첫째 페이지 본문", "둘째 페이지 본문", "셋째 페이지 본문"]
        result = preprocess_pages(pages)

        second_start = result.content.index("둘째")
        self.assertEqual(result.page_of(second_start), 2)
        self.assertEqual(result.page_of(0), 1)

    def test_empty_pages_keep_page_numbering_aligned(self):
        # 2페이지가 비어도 3페이지는 여전히 3페이지여야 한다.
        pages = ["첫째 페이지 본문", None, "셋째 페이지 본문"]
        result = preprocess_pages(pages)

        self.assertEqual(len(result.page_spans), 3)
        self.assertEqual(result.empty_page_count, 1)
        third_start = result.content.index("셋째")
        self.assertEqual(result.page_of(third_start), 3)

    def test_pages_are_separated_by_blank_line(self):
        result = preprocess_pages(["가나다", "라마바"])
        self.assertEqual(result.content, f"가나다{PAGE_SEPARATOR}라마바")

    def test_leading_empty_page_does_not_shift_spans(self):
        # 첫 페이지가 비면 content 앞쪽이 strip 으로 잘리는데, span 을 함께 당기지
        # 않으면 이후 모든 페이지가 PAGE_SEPARATOR 길이만큼 밀린다. 예외 없이
        # 조용히 틀린 페이지 번호가 되므로 회귀 테스트로 고정한다.
        pages = [None, "둘째 페이지 본문", "셋째 페이지 본문"]
        result = preprocess_pages(pages)

        self.assertEqual(len(result.page_spans), 3)
        second_start, second_end = result.page_spans[1]
        self.assertEqual(result.content[second_start:second_end], "둘째 페이지 본문")
        third_start, third_end = result.page_spans[2]
        self.assertEqual(result.content[third_start:third_end], "셋째 페이지 본문")
        self.assertEqual(result.page_of(result.content.index("셋째")), 3)

    def test_cover_page_emptied_by_header_strip_keeps_numbering(self):
        # 표지처럼 머리말·페이지번호만 있던 첫 페이지는 전처리 후 비어버린다.
        # 실제 PDF 에서 흔한 모양이고, 여기서 span 이 밀리면 Chunk.page 가 전부 틀어진다.
        pages = [
            "회사 보고서\n- 1 -",
            "회사 보고서\n본문 A 입니다.\n- 2 -",
            "회사 보고서\n본문 B 입니다.\n- 3 -",
            "회사 보고서\n본문 C 입니다.\n- 4 -",
        ]
        result = preprocess_pages(pages)

        self.assertEqual(len(result.page_spans), 4)
        for page_no, marker in ((2, "본문 A 입니다."), (3, "본문 B 입니다."), (4, "본문 C 입니다.")):
            start, end = result.page_spans[page_no - 1]
            self.assertEqual(result.content[start:end], marker)
            self.assertEqual(result.page_of(result.content.index(marker)), page_no)


class ScannedDetectionTest(unittest.TestCase):
    def test_all_none_pages_is_empty(self):
        result = preprocess_pages([None, None, None])

        self.assertTrue(result.is_empty)
        self.assertEqual(result.page_count, 3)
        self.assertEqual(result.empty_page_count, 3)

    def test_sparse_text_flagged_as_probably_scanned(self):
        result = preprocess_pages(["쪽", "쪽", "쪽", "쪽"])
        self.assertTrue(result.is_probably_scanned)

    def test_normal_document_not_flagged(self):
        page = "본문이 충분히 긴 문서입니다. " * 10
        result = preprocess_pages([page, page, page])

        self.assertFalse(result.is_empty)
        self.assertFalse(result.is_probably_scanned)

    def test_empty_input_does_not_crash(self):
        result = preprocess_pages([])

        self.assertTrue(result.is_empty)
        self.assertFalse(result.is_probably_scanned)
        self.assertEqual(result.page_spans, [])


if __name__ == "__main__":
    unittest.main()
