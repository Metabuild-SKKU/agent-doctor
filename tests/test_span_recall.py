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


if __name__ == "__main__":
    unittest.main()
