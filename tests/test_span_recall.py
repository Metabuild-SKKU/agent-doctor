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
        """대조군 — 필드가 없으면 틈이 그대로 남는다(legacy 인덱스 호환 고정).

        ⚠️ 부분점수 도입(2026-08-07) 이후 이 대조가 얇아졌다. 예전엔 1.0 vs 0.0 이었는데
        이제 1.0 vs 0.975 다 — 2자 틈이 80자 골드에서 0.025 밖에 안 깎이기 때문이다.
        원좌표 브리징이 무의미해진 건 아니지만(틈이 크면 여전히 크게 깎인다), "틈 하나가
        점수를 0 으로 만든다" 던 원래 동기는 부분점수가 대부분 흡수한다.
        """
        span = [{"doc_id": "d1", "start": 120, "end": 200}]

        gapped = span_recall_at_k(span, ["c0", "c1"], self._gapped(original=False))
        self.assertEqual(gapped, 0.975)                       # 80자 중 78자(2자 틈)
        self.assertLess(gapped, span_recall_at_k(span, ["c0", "c1"], self._gapped()))

    def test_span_inside_one_chunk_is_unaffected(self):
        span = [{"doc_id": "d1", "start": 160, "end": 200}]

        self.assertEqual(span_recall_at_k(span, ["c0", "c1"], self._gapped()), 1.0)

    def test_retrieving_only_one_side_of_the_gap_still_fails(self):
        """틈을 닫는 것이지 미검색을 덮는 게 아니다."""
        span = [{"doc_id": "d1", "start": 120, "end": 200}]

        # 미검색분(149~200)은 그대로 못 덮는다 — 29/80 = 0.3625.
        self.assertEqual(span_recall_at_k(span, ["c0"], self._gapped()), 0.3625)

    def test_dedup_hole_without_alias_is_not_bridged(self):
        """별칭이 없으면 dedup 구멍은 안 닫힌다 — 원좌표는 제 자리만 늘릴 뿐이다.

        이 필드 이전에 색인된 청크(캐시 재사용 경로)가 그대로 여기 해당한다.
        """
        chunks = [
            Chunk("c0", "d1", "a" * 100, char_span=(0, 100), original_char_span=(0, 102)),
            # [102, 300) 을 차지하던 청크가 중복이라 빠졌다 → 원좌표로도 안 닫힌다.
            Chunk("c2", "d1", "c" * 100, char_span=(300, 400), original_char_span=(300, 402)),
        ]
        span = [{"doc_id": "d1", "start": 50, "end": 350}]

        # 198자 구멍은 그대로 뚫려 있다 — 300자 중 102자만 덮인다.
        self.assertEqual(span_recall_at_k(span, ["c0", "c2"], chunks), 0.34)

    def test_broken_original_span_falls_back_to_char_span(self):
        """제 char_span 도 못 덮는 값은 신뢰하지 않는다(직렬화 사고 방어)."""
        chunks = [
            Chunk("c0", "d1", "a" * 147, char_span=(0, 147), original_char_span=(0, 10)),
            Chunk("c1", "d1", "b" * 257, char_span=(149, 406), original_char_span=(149, 406)),
        ]
        span = [{"doc_id": "d1", "start": 120, "end": 200}]

        # 깨진 원좌표를 안 믿고 char_span 으로 폴백하므로 2자 틈이 남는다(78/80).
        self.assertEqual(span_recall_at_k(span, ["c0", "c1"], chunks), 0.975)

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


class DuplicateSpanAliasTest(unittest.TestCase):
    """dedup 이 버린 쌍둥이의 자리를 생존 청크가 대표한다(별칭 좌표).

    핵심은 '항상 덮인다'가 아니라 판정이 **갈린다**는 것이다 — 대표 청크가 검색되면
    그 내용이 실제로 컨텍스트에 들어갔으니 덮이고, 안 되면 그대로 구멍이다.
    """

    def _chunks(self, alias_doc="d1"):
        # [102, 300) 을 차지하던 쌍둥이가 dedup 으로 빠지고, 같은 본문인 c2 가 대표한다.
        twin = "b" * 190
        return [
            Chunk("c0", "d1", "a" * 100, char_span=(0, 100), original_char_span=(0, 102)),
            Chunk(
                "c2", "d2", twin,
                char_span=(10, 200), original_char_span=(8, 202),
                duplicate_spans=[[alias_doc, 102, 300]],
            ),
            Chunk("c3", "d1", "c" * 100, char_span=(300, 400), original_char_span=(300, 402)),
        ]

    def test_alias_closes_hole_when_representative_is_retrieved(self):
        span = [{"doc_id": "d1", "start": 50, "end": 350}]

        recall = span_recall_at_k(span, ["c0", "c2", "c3"], self._chunks())

        self.assertEqual(recall, 1.0)

    def test_alias_does_not_close_hole_when_representative_is_missed(self):
        """대표 청크가 안 검색되면 그 내용은 컨텍스트에 없다 — 0점이 맞다."""
        span = [{"doc_id": "d1", "start": 50, "end": 350}]

        recall = span_recall_at_k(span, ["c0", "c3"], self._chunks())

        # 별칭 구간(102~300)이 통째로 비어 300자 중 102자만 덮인다.
        self.assertEqual(recall, 0.34)
        # 대표 청크를 검색했을 때(1.0)와 확실히 갈린다 — 그게 이 클래스의 요점이다.
        self.assertLess(recall, span_recall_at_k(span, ["c0", "c2", "c3"], self._chunks()))

    def test_alias_is_filed_under_its_own_doc_not_the_chunks_doc(self):
        """문서 간 dedup — 별칭이 청크 제 doc_id 밑으로 들어가면 엉뚱한 문서를 덮는다."""
        chunks = self._chunks(alias_doc="d9")
        # 별칭이 가리키는 곳은 d9 지 d1 이 아니므로 d1 의 구멍은 그대로다.
        self.assertEqual(
            span_recall_at_k([{"doc_id": "d1", "start": 50, "end": 350}], ["c0", "c2", "c3"], chunks),
            0.34,      # d1 의 구멍은 그대로 — 별칭이 d9 를 가리키므로 못 닫는다
        )
        self.assertEqual(
            span_recall_at_k([{"doc_id": "d9", "start": 110, "end": 290}], ["c2"], chunks),
            1.0,
        )

    def test_alias_shorter_than_chunk_text_is_ignored(self):
        """본문이 같으니 별칭 구간은 최소 텍스트 길이여야 한다(직렬화 사고 방어)."""
        chunks = [
            Chunk("c0", "d1", "a" * 100, char_span=(0, 100), original_char_span=(0, 102)),
            Chunk(
                "c2", "d2", "b" * 190,
                char_span=(10, 200), original_char_span=(8, 202),
                duplicate_spans=[["d1", 102, 120]],      # 190자를 담기엔 너무 짧다
            ),
            Chunk("c3", "d1", "c" * 100, char_span=(300, 400), original_char_span=(300, 402)),
        ]

        recall = span_recall_at_k(
            [{"doc_id": "d1", "start": 50, "end": 350}], ["c0", "c2", "c3"], chunks
        )

        self.assertEqual(recall, 0.34)      # 별칭을 무시하므로 구멍이 그대로 남는다

    def test_malformed_alias_entries_are_skipped(self):
        chunks = self._chunks()
        chunks[1].duplicate_spans = [
            ["d1", 102],                 # 길이 부족
            [None, 102, 300],            # doc_id 가 문자열이 아님
            ["d1", 300, 102],            # end <= start
            "d1",                        # 항목이 리스트가 아님
            ["d1", 102, 300],            # 유일하게 정상
        ]

        recall = span_recall_at_k(
            [{"doc_id": "d1", "start": 50, "end": 350}], ["c0", "c2", "c3"], chunks
        )

        self.assertEqual(recall, 1.0)

    def test_legacy_metadata_carries_alias(self):
        """Chunk 필드가 비어도 metadata 로 온 값(payload 왕복)을 읽는다."""
        chunks = self._chunks()
        chunks[1].duplicate_spans = []
        chunks[1].metadata = {"duplicate_spans": [["d1", 102, 300]]}

        recall = span_recall_at_k(
            [{"doc_id": "d1", "start": 50, "end": 350}], ["c0", "c2", "c3"], chunks
        )

        self.assertEqual(recall, 1.0)


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
        """span 이 여러 개면 하나만 덮었다고 0 이 되지 않는다.

        길이가 같은 두 span 이라 macro·micro 가 같은 값을 낸다 — 둘을 가르는 케이스는
        `MicroAverageTest` 에 있다.
        """
        chunks = self._doc_chunks((0, 100), (100, 200))
        spans = [
            {"doc_id": "d1", "start": 0, "end": 100},      # c0 로 전부 덮임 → 100자
            {"doc_id": "d1", "start": 100, "end": 200},    # 안 가져옴      →   0자
        ]

        self.assertEqual(span_recall_at_k(spans, ["c0"], chunks), 0.5)

    def test_partial_on_every_span_averages(self):
        chunks = self._doc_chunks((0, 25), (100, 175))
        spans = [
            {"doc_id": "d1", "start": 0, "end": 100},      # 25자
            {"doc_id": "d1", "start": 100, "end": 200},    # 75자
        ]

        # 100자 / 200자. 여기서도 span 길이가 같아 macro 와 값이 겹친다.
        self.assertAlmostEqual(span_recall_at_k(spans, ["c0", "c1"], chunks), 0.5)

    def test_ratio_never_exceeds_one_when_chunk_is_wider_than_span(self):
        """청크가 골드보다 훨씬 넓어도 비율은 1 을 넘지 않는다(교집합을 span 으로 자른다)."""
        chunks = self._doc_chunks((0, 10_000))
        spans = [{"doc_id": "d1", "start": 500, "end": 510}]

        self.assertEqual(span_recall_at_k(spans, ["c0"], chunks), 1.0)


class MicroAverageTest(unittest.TestCase):
    """span 이 여러 개일 때 **길이로 가중**해 합친다(micro average, 2026-08-10).

    span 별 비율을 단순 평균(macro)하면 점수가 **청킹 경계에 의존한다**. gold span 은
    원자적 근거 단위가 아니라 positive_chunk_ids 를 좌표로 환산한 값이라, 몇 조각으로
    나뉘는지가 청커 설정의 산물이기 때문이다. optimize 가 처방으로 바꾸는 축이 바로 청크
    전략이므로, macro 는 "검색을 잘하는 설정" 대신 "채점에 유리하게 잘리는 설정" 쪽으로
    탐색을 끌 수 있다.
    """

    def _doc_chunks(self, *ranges):
        return [
            Chunk(f"c{i}", "d1", "x" * (e - s), char_span=(s, e))
            for i, (s, e) in enumerate(ranges)
        ]

    def test_long_span_outweighs_short_one(self):
        """길이가 다르면 macro 와 갈린다 — 짧은 쪽만 덮고 만점 절반을 받을 수 없다."""
        chunks = self._doc_chunks((0, 100), (100, 500))
        spans = [
            {"doc_id": "d1", "start": 0, "end": 100},      # 100자, 전부 덮임
            {"doc_id": "d1", "start": 100, "end": 500},    # 400자, 전혀 못 덮음
        ]

        # micro: 100자 / 500자 = 0.2   (macro 였다면 (1.0+0.0)/2 = 0.5)
        self.assertEqual(span_recall_at_k(spans, ["c0"], chunks), 0.2)

    def test_score_does_not_move_with_how_the_gold_is_split(self):
        """같은 근거·같은 검색이면 골드를 몇 조각으로 쪼개든 점수가 같다.

        이 클래스의 존재 이유다. 아래 셋은 전부 "0~500 의 근거 중 앞 100자만 검색" 이라
        검색 성과가 동일한데, macro 로는 0.2·0.5·0.2 로 갈린다.
        """
        chunks = self._doc_chunks((0, 100), (100, 500))
        retrieved = ["c0"]

        whole = [{"doc_id": "d1", "start": 0, "end": 500}]
        uneven = [{"doc_id": "d1", "start": 0, "end": 100},
                  {"doc_id": "d1", "start": 100, "end": 500}]
        even = [{"doc_id": "d1", "start": 0, "end": 250},
                {"doc_id": "d1", "start": 250, "end": 500}]

        self.assertEqual(span_recall_at_k(whole, retrieved, chunks), 0.2)
        self.assertEqual(span_recall_at_k(uneven, retrieved, chunks), 0.2)
        self.assertEqual(span_recall_at_k(even, retrieved, chunks), 0.2)

    def test_full_coverage_of_uneven_spans_is_exactly_one(self):
        """길이가 달라도 전부 덮으면 정확히 1.0 — 정수로 모아 한 번만 나누기 때문이다.

        비율을 먼저 내고 길이를 다시 곱하는 구현이면 여기서 0.999… 가 나올 수 있고,
        `_recall_ok` 는 정확히 `>= 1` 을 요구하므로 그 오차 하나가 라벨을 뒤집는다.
        """
        chunks = self._doc_chunks((0, 3), (3, 1000))
        spans = [
            {"doc_id": "d1", "start": 0, "end": 3},        # 3자
            {"doc_id": "d1", "start": 3, "end": 1000},     # 997자
        ]

        self.assertEqual(span_recall_at_k(spans, ["c0", "c1"], chunks), 1.0)

    def test_no_coverage_of_uneven_spans_is_exactly_zero(self):
        chunks = self._doc_chunks((0, 10), (2000, 3000))
        spans = [
            {"doc_id": "d1", "start": 100, "end": 103},
            {"doc_id": "d1", "start": 103, "end": 1100},
        ]

        self.assertEqual(span_recall_at_k(spans, ["c0", "c1"], chunks), 0.0)


if __name__ == "__main__":
    unittest.main()
