"""구조 기반 evidence window 계산 단위 테스트."""
import unittest

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


if __name__ == "__main__":
    unittest.main()
