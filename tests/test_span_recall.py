import unittest

from agents.eval.metrics_basic import span_recall_at_k
from core.schema import Chunk


class SpanRecallAtKTest(unittest.TestCase):
    def test_one_containing_chunk_is_enough_even_if_gold_ids_had_two_chunks(self):
        chunks = [
            Chunk("c0", "d1", "a" * 400, char_span=(0, 400)),
            Chunk("c1", "d1", "a" * 400, char_span=(275, 675)),
        ]

        recall = span_recall_at_k(
            [{"doc_id": "d1", "start": 325, "end": 450}],
            ["c1"],
            chunks,
        )

        self.assertEqual(recall, 1.0)

    def test_multiple_retrieved_chunks_may_cover_one_span_together(self):
        chunks = [
            Chunk("c0", "d1", "a" * 400, char_span=(0, 400)),
            Chunk("c1", "d1", "b" * 400, char_span=(400, 800)),
        ]
        spans = [{"doc_id": "d1", "start": 350, "end": 450}]

        self.assertEqual(span_recall_at_k(spans, ["c0", "c1"], chunks), 1.0)
        self.assertEqual(span_recall_at_k(spans, ["c0"], chunks), 0.0)

    def test_missing_chunk_coordinates_uses_legacy_fallback_signal(self):
        chunks = [Chunk("c0", "d1", "legacy", char_span=None)]

        recall = span_recall_at_k(
            [{"doc_id": "d1", "start": 0, "end": 6}],
            ["c0"],
            chunks,
        )

        self.assertIsNone(recall)

    def test_retrieved_legacy_chunk_in_mixed_document_uses_fallback(self):
        chunks = [
            Chunk("legacy", "d1", "정답", char_span=None),
            Chunk("positioned", "d1", "다른 내용", char_span=(10, 20)),
        ]

        recall = span_recall_at_k(
            [{"doc_id": "d1", "start": 0, "end": 2}],
            ["legacy"],
            chunks,
        )

        self.assertIsNone(recall)


class SectionGapCoverageTest(unittest.TestCase):
    """issue #100 — 트림이 만든 좌표 틈이 gold span 을 갈라 0점이 되던 문제.

    청크 텍스트에서 앞뒤 공백을 떼면 char_span 도 같이 당겨져, 섹션 경계마다
    좌표에 틈이 남는다. original_char_span(트림 전 경계)으로 판정해 그 틈만 닫는다.
    """

    def _gapped(self, original=True):
        # 청크0 [0,147) / 청크1 [149,406) — 147~149 는 섹션 사이 빈 줄.
        return [
            Chunk("c0", "d1", "a" * 147, char_span=(0, 147),
                  original_char_span=(0, 149) if original else None),
            Chunk("c1", "d1", "b" * 257, char_span=(149, 406),
                  original_char_span=(149, 406) if original else None),
        ]

    def test_gold_span_across_section_gap_is_covered(self):
        span = [{"doc_id": "d1", "start": 120, "end": 200}]

        self.assertEqual(span_recall_at_k(span, ["c0", "c1"], self._gapped()), 1.0)

    def test_same_span_without_original_span_still_reproduces_the_bug(self):
        """대조군 — 필드가 없으면 종전 동작(0점) 그대로다. legacy 인덱스 호환 고정."""
        span = [{"doc_id": "d1", "start": 120, "end": 200}]

        self.assertEqual(span_recall_at_k(span, ["c0", "c1"], self._gapped(original=False)), 0.0)

    def test_span_inside_one_chunk_is_unaffected(self):
        span = [{"doc_id": "d1", "start": 160, "end": 200}]

        self.assertEqual(span_recall_at_k(span, ["c0", "c1"], self._gapped()), 1.0)

    def test_retrieving_only_one_side_of_the_gap_still_fails(self):
        """틈을 닫는 것이지 미검색을 덮는 게 아니다."""
        span = [{"doc_id": "d1", "start": 120, "end": 200}]

        self.assertEqual(span_recall_at_k(span, ["c0"], self._gapped()), 0.0)

    def test_dedup_hole_is_not_bridged(self):
        """청크가 통째로 빠진 자리(dedup)는 실제 누락이라 0점이 맞다."""
        chunks = [
            Chunk("c0", "d1", "a" * 100, char_span=(0, 100), original_char_span=(0, 102)),
            # [102, 300) 을 차지하던 청크가 중복이라 빠졌다 → 원좌표로도 안 닫힌다.
            Chunk("c2", "d1", "c" * 100, char_span=(300, 400), original_char_span=(300, 402)),
        ]
        span = [{"doc_id": "d1", "start": 50, "end": 350}]

        self.assertEqual(span_recall_at_k(span, ["c0", "c2"], chunks), 0.0)

    def test_broken_original_span_falls_back_to_char_span(self):
        """제 char_span 도 못 덮는 값은 신뢰하지 않는다(직렬화 사고 방어)."""
        chunks = [
            Chunk("c0", "d1", "a" * 147, char_span=(0, 147), original_char_span=(0, 10)),
            Chunk("c1", "d1", "b" * 257, char_span=(149, 406), original_char_span=(149, 406)),
        ]
        span = [{"doc_id": "d1", "start": 120, "end": 200}]

        self.assertEqual(span_recall_at_k(span, ["c0", "c1"], chunks), 0.0)

    def test_legacy_metadata_carries_original_span(self):
        """Chunk 필드가 비어도 metadata 로 온 값(payload 왕복)을 읽는다."""
        chunks = [
            Chunk("c0", "d1", "a" * 147, char_span=(0, 147),
                  metadata={"original_char_span": [0, 149]}),
            Chunk("c1", "d1", "b" * 257, char_span=(149, 406),
                  metadata={"original_char_span": [149, 406]}),
        ]
        span = [{"doc_id": "d1", "start": 120, "end": 200}]

        self.assertEqual(span_recall_at_k(span, ["c0", "c1"], chunks), 1.0)


if __name__ == "__main__":
    unittest.main()
