"""구조 기반 evidence window 계산 단위 테스트."""
import unittest
from unittest.mock import patch

from agents.optimize import evidence_window
from agents.optimize.evidence_window import build_evidence_windows
from core.schema import Document


POLICY = {
    "min_chars": 40,
    "max_chars": 300,
    "heading_max_distance": 80,
    "adjacent_context_blocks": 1,
}


def _span(content: str, text: str, doc_id: str = "d1") -> dict:
    start = content.index(text)
    return {"doc_id": doc_id, "start": start, "end": start + len(text)}


class EvidenceWindowTest(unittest.TestCase):
    def test_prose_expands_answer_to_neighboring_sentences(self):
        content = (
            "회사의 배당 정책을 설명합니다. "
            "배당가능이익의 90% 이상을 배당하면 세금 혜택을 받습니다. "
            "이 기준은 매년 적용됩니다."
        )
        document = Document("d1", "memory", "txt", content)

        windows = build_evidence_windows(
            [document], [_span(content, "90%")], POLICY
        )

        window_text = content[windows[0]["start"]:windows[0]["end"]]
        self.assertIn("배당 정책", window_text)
        self.assertIn("90% 이상", window_text)
        self.assertIn("매년 적용", window_text)
        self.assertEqual(windows[0]["kind"], "prose")

    def test_table_uses_contiguous_table_context(self):
        content = (
            "자산 현황\n"
            "| 구분 | 전기말 | 당분기말 |\n"
            "| --- | --- | --- |\n"
            "| 자산총계 | 49,181 | 49,158 |\n"
            "표 아래 설명입니다.\n"
        )
        document = Document("d1", "memory", "md", content)

        windows = build_evidence_windows(
            [document], [_span(content, "49,158")], POLICY
        )

        window_text = content[windows[0]["start"]:windows[0]["end"]]
        self.assertIn("구분", window_text)
        self.assertIn("자산총계", window_text)
        self.assertEqual(windows[0]["kind"], "table")

    def test_list_includes_intro_and_target_item(self):
        content = (
            "필요한 서류는 다음과 같습니다.\n"
            "- 신분증 사본\n"
            "- 소득 증빙 자료\n"
            "- 신청서\n"
        )
        document = Document("d1", "memory", "md", content)

        windows = build_evidence_windows(
            [document], [_span(content, "소득 증빙")], POLICY
        )

        window_text = content[windows[0]["start"]:windows[0]["end"]]
        self.assertIn("필요한 서류", window_text)
        self.assertIn("소득 증빙", window_text)
        self.assertEqual(windows[0]["kind"], "list")

    def test_distant_multihop_spans_remain_separate(self):
        content = (
            "첫 번째 근거는 서울에 있습니다. "
            + ("중간 내용입니다. " * 30)
            + "두 번째 근거는 부산에 있습니다."
        )
        document = Document("d1", "memory", "txt", content)

        windows = build_evidence_windows(
            [document],
            [_span(content, "서울"), _span(content, "부산")],
            POLICY,
        )

        self.assertEqual(len(windows), 2)
        self.assertLessEqual(max(window["length"] for window in windows), 300)
        self.assertNotEqual(
            (windows[0]["start"], windows[0]["end"]),
            (windows[1]["start"], windows[1]["end"]),
        )

    def test_window_keeps_gold_when_paragraph_exceeds_maximum(self):
        content = "가" * 500 + "정답" + "나" * 500
        document = Document("d1", "memory", "txt", content)

        windows = build_evidence_windows(
            [document], [_span(content, "정답")], POLICY
        )

        window = windows[0]
        answer_start = content.index("정답")
        self.assertLessEqual(window["length"], POLICY["max_chars"])
        self.assertLessEqual(window["start"], answer_start)
        self.assertGreaterEqual(window["end"], answer_start + len("정답"))

    def test_multiline_table_gold_is_never_replaced_with_document_prefix(self):
        prefix = ("문서 앞부분입니다.\n" * 40) + "\n"
        table = (
            "| 항목 | 값 |\n"
            "| --- | --- |\n"
            "| A | 첫째 |\n"
            "| B | 둘째 |\n"
        )
        content = prefix + table
        start = content.index("| A")
        end = content.index("둘째") + len("둘째")

        windows = build_evidence_windows(
            [Document("d1", "memory", "md", content)],
            [{"doc_id": "d1", "start": start, "end": end}],
            POLICY,
        )

        window = windows[0]
        self.assertEqual(window["kind"], "table")
        self.assertLessEqual(window["start"], start)
        self.assertGreaterEqual(window["end"], end)
        self.assertNotEqual(window["start"], 0)

    def test_multiline_list_gold_keeps_all_target_items(self):
        content = (
            "처리 순서는 다음과 같습니다.\n"
            "- 첫 번째 작업\n"
            "- 두 번째 작업\n"
            "- 세 번째 작업\n"
        )
        start = content.index("- 첫")
        end = content.index("두 번째") + len("두 번째")

        windows = build_evidence_windows(
            [Document("d1", "memory", "md", content)],
            [{"doc_id": "d1", "start": start, "end": end}],
            POLICY,
        )

        window_text = content[windows[0]["start"]:windows[0]["end"]]
        self.assertEqual(windows[0]["kind"], "list")
        self.assertIn("첫 번째", window_text)
        self.assertIn("두 번째", window_text)

    def test_same_structural_window_keeps_one_sample_per_gold_span(self):
        content = "구분자 없는 짧은 문단에 서로 다른 정답이 함께 있습니다"
        spans = [
            _span(content, "짧은"),
            _span(content, "서로"),
            _span(content, "정답"),
        ]

        windows = build_evidence_windows(
            [Document("d1", "memory", "txt", content)],
            spans,
            POLICY,
        )

        self.assertEqual(len(windows), len(spans))
        self.assertEqual(
            {(window["start"], window["end"]) for window in windows},
            {(0, len(content))},
        )
        self.assertEqual(
            [window["sample_index"] for window in windows],
            [0, 1, 2],
        )

    def test_korean_closing_quote_and_fullwidth_punctuation_split_sentences(self):
        content = "첫째입니다.” 둘째입니다。 셋째입니다！"
        policy = {
            **POLICY,
            "min_chars": 1,
            "adjacent_context_blocks": 0,
        }

        windows = build_evidence_windows(
            [Document("d1", "memory", "txt", content)],
            [_span(content, "둘째")],
            policy,
        )

        window_text = content[windows[0]["start"]:windows[0]["end"]]
        self.assertIn("둘째입니다。", window_text)
        self.assertNotIn("첫째", window_text)
        self.assertNotIn("셋째", window_text)

    def test_single_pipe_expression_is_not_misclassified_as_table(self):
        content = "선택지는 A | B | C입니다."

        windows = build_evidence_windows(
            [Document("d1", "memory", "txt", content)],
            [_span(content, "B")],
            POLICY,
        )

        self.assertEqual(windows[0]["kind"], "prose")

    def test_pipe_prose_immediately_before_table_is_not_absorbed(self):
        content = (
            "선택지는 A | B 중 하나다\n"
            "| 항목 | 값 |\n"
            "|---|---|\n"
            "| a | 1 |\n"
        )
        policy = {
            **POLICY,
            "min_chars": 1,
            "heading_max_distance": 0,
            "adjacent_context_blocks": 0,
        }

        windows = build_evidence_windows(
            [Document("d1", "memory", "md", content)],
            [_span(content, "A | B")],
            policy,
        )

        window_text = content[windows[0]["start"]:windows[0]["end"]]
        self.assertEqual(windows[0]["kind"], "prose")
        self.assertEqual(window_text, "선택지는 A | B 중 하나다")
        self.assertNotIn("| 항목 |", window_text)

    def test_adjacent_markdown_tables_remain_separate_blocks(self):
        content = (
            "| 첫째 | 값 |\n"
            "|---|---|\n"
            "| a | 1 |\n"
            "| 둘째 | 값 |\n"
            "|---|---|\n"
            "| b | 2 |\n"
        )
        policy = {
            **POLICY,
            "min_chars": 1,
            "heading_max_distance": 0,
            "adjacent_context_blocks": 0,
        }

        windows = build_evidence_windows(
            [Document("d1", "memory", "md", content)],
            [_span(content, "a | 1")],
            policy,
        )

        window_text = content[windows[0]["start"]:windows[0]["end"]]
        self.assertEqual(windows[0]["kind"], "table")
        self.assertIn("| 첫째 | 값 |", window_text)
        self.assertNotIn("| 둘째 | 값 |", window_text)

    def test_gold_longer_than_policy_maximum_is_preserved(self):
        content = ("앞" * 50) + ("정답" * 200) + ("뒤" * 50)
        start = content.index("정답")
        end = start + len("정답" * 200)

        windows = build_evidence_windows(
            [Document("d1", "memory", "txt", content)],
            [{"doc_id": "d1", "start": start, "end": end}],
            POLICY,
        )

        self.assertEqual(windows[0]["start"], start)
        self.assertEqual(windows[0]["end"], end)
        self.assertGreater(windows[0]["length"], POLICY["max_chars"])

    def test_document_layout_is_built_once_for_many_spans(self):
        content = ("문장입니다. " * 1000)
        spans = [
            {"doc_id": "d1", "start": index * 20, "end": index * 20 + 2}
            for index in range(200)
        ]

        with patch.object(
            evidence_window,
            "_line_ranges",
            wraps=evidence_window._line_ranges,
        ) as line_ranges:
            windows = build_evidence_windows(
                [Document("d1", "memory", "txt", content)],
                spans,
                POLICY,
            )

        self.assertEqual(len(windows), 200)
        line_ranges.assert_called_once()


if __name__ == "__main__":
    unittest.main()
