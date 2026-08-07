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
        # 2026-08-07 이전에는 0.0 이었다(빈틈없이 덮어야 1점인 이진 판정).
        # c0 는 350~400 의 50자, 즉 골드 100자의 절반을 덮는다 → 부분점수 0.5.
        self.assertEqual(span_recall_at_k(spans, ["c0"], chunks), 0.5)

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


class PartialSpanCoverageTest(unittest.TestCase):
    """골드 구간을 부분만 덮었을 때 문자 비율로 점수를 준다(2026-08-07).

    이 클래스가 고정하는 성질은 셋이다.
      ① 0/1 이 아니라 그 사이 값이 나온다 — 실측 47% 의 recall=0 이 여기서 왔다.
      ② 양 끝(전부·전무)은 정확히 1.0·0.0 이다 — `_recall_ok`(>=1)·A슬롯 진입
         (0<=recall<1)의 참거짓이 이 변경으로 뒤집히면 안 된다.
      ③ 겹치는 청크가 문자를 두 번 세지 않는다 — 청킹 overlap 이 켜지면 인접 청크가
         같은 문자를 공유하므로 합집합이 아니면 비율이 1을 넘는다.
    """

    def _doc_chunks(self, *ranges):
        return [
            Chunk(f"c{i}", "d1", "x" * (e - s), char_span=(s, e))
            for i, (s, e) in enumerate(ranges)
        ]

    def test_gap_in_the_middle_still_counts_both_sides(self):
        """틈이 있어도 앞뒤로 덮은 문자를 전부 센다.

        예전 로직은 첫 틈에서 break 해 뒤쪽을 버렸다. top-k 가 흩어져 오는 건 정상이라
        연속성을 요구할 이유가 없다 — 덮은 근거는 덮은 만큼 값을 한다.
        """
        chunks = self._doc_chunks((0, 30), (30, 70), (70, 110))
        spans = [{"doc_id": "d1", "start": 0, "end": 100}]

        # c0(0~30) + c2(70~100) = 60자 / 100자. 가운데 30~70 은 안 가져왔다.
        self.assertAlmostEqual(span_recall_at_k(spans, ["c0", "c2"], chunks), 0.60)

    def test_full_coverage_is_exactly_one(self):
        """전부 덮으면 정확히 1.0 — 부동소수 오차로 0.999… 가 되면 _recall_ok 가 깨진다."""
        chunks = self._doc_chunks((0, 33), (33, 66), (66, 100))
        spans = [{"doc_id": "d1", "start": 0, "end": 100}]

        self.assertEqual(span_recall_at_k(spans, ["c0", "c1", "c2"], chunks), 1.0)

    def test_no_overlap_is_exactly_zero(self):
        chunks = self._doc_chunks((0, 50), (200, 300))
        spans = [{"doc_id": "d1", "start": 100, "end": 150}]

        self.assertEqual(span_recall_at_k(spans, ["c0", "c1"], chunks), 0.0)

    def test_overlapping_chunks_do_not_double_count(self):
        """청킹 overlap 으로 같은 문자를 공유하는 청크들. 합집합이 아니면 1.4 가 나온다."""
        chunks = self._doc_chunks((0, 70), (30, 100))
        spans = [{"doc_id": "d1", "start": 0, "end": 100}]

        self.assertEqual(span_recall_at_k(spans, ["c0", "c1"], chunks), 1.0)

    def test_chunk_contained_in_another_is_absorbed(self):
        """작은 청크가 큰 청크 안에 완전히 들어가도 두 번 세지 않는다."""
        chunks = self._doc_chunks((0, 100), (20, 40))
        spans = [{"doc_id": "d1", "start": 0, "end": 100}]

        self.assertEqual(span_recall_at_k(spans, ["c0", "c1"], chunks), 1.0)

    def test_multiple_spans_average_their_ratios(self):
        """span 이 여러 개면 비율의 평균이다 — 하나만 덮었다고 0 이 되지 않는다."""
        chunks = self._doc_chunks((0, 100), (100, 200))
        spans = [
            {"doc_id": "d1", "start": 0, "end": 100},      # c0 로 전부 덮임 → 1.0
            {"doc_id": "d1", "start": 100, "end": 200},    # 안 가져옴      → 0.0
        ]

        self.assertEqual(span_recall_at_k(spans, ["c0"], chunks), 0.5)

    def test_partial_on_every_span_averages(self):
        chunks = self._doc_chunks((0, 25), (100, 175))
        spans = [
            {"doc_id": "d1", "start": 0, "end": 100},      # 25/100 = 0.25
            {"doc_id": "d1", "start": 100, "end": 200},    # 75/100 = 0.75
        ]

        self.assertAlmostEqual(span_recall_at_k(spans, ["c0", "c1"], chunks), 0.5)

    def test_ratio_never_exceeds_one_when_chunk_is_wider_than_span(self):
        """청크가 골드보다 훨씬 넓어도 비율은 1 을 넘지 않는다(교집합을 span 으로 자른다)."""
        chunks = self._doc_chunks((0, 10_000))
        spans = [{"doc_id": "d1", "start": 500, "end": 510}]

        self.assertEqual(span_recall_at_k(spans, ["c0"], chunks), 1.0)


if __name__ == "__main__":
    unittest.main()
