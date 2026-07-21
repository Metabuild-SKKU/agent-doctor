import unittest
from unittest.mock import patch

from agents.eval.knowledge_graph import KGNode
from agents.eval.probe_gen import (
    _SynthesizedProbe,
    _from_chunks,
    _gold_spans_from_evidence,
    _heuristic_evidence_of,
    _heuristic_synthesize_query,
    _llm_synthesize_query,
    _make_ragas_probe,
    _resync_gold_chunk_ids,
)
from core.schema import Chunk, Document, Probe


class ProbeGoldSpanGroundingTest(unittest.TestCase):
    def test_exact_quote_is_located_inside_selected_repeated_chunk(self):
        first = "반복되는 근거 문장입니다. 첫 번째 위치입니다."
        second = "반복되는 근거 문장입니다. 두 번째 위치입니다."
        content = f"{first}\n구분선\n{second}"
        start = content.index(second)
        document = Document("d1", "memory", "txt", content)
        chunk = Chunk(
            chunk_id="c2",
            doc_id="d1",
            text=second,
            char_span=(start, start + len(second)),
        )
        node = KGNode(chunk_id="c2", doc_id="d1", text=second)
        synthesized = _SynthesizedProbe(
            question="두 번째 위치의 근거는 무엇인가요?",
            ground_truth="두 번째 위치에 같은 근거가 있습니다.",
            evidence=[{"source_index": 0, "quote": "반복되는 근거 문장입니다."}],
        )

        spans, located, exact, fallback = _gold_spans_from_evidence(
            synthesized,
            [node],
            {"c2": chunk},
            {"d1": document},
        )

        self.assertEqual((located, exact, fallback), (1, 1, 0))
        self.assertEqual(spans[0]["start"], start)
        self.assertEqual(
            content[spans[0]["start"]:spans[0]["end"]],
            "반복되는 근거 문장입니다.",
        )

    def test_legacy_repeated_chunk_uses_selected_chunk_order(self):
        chunk_text = "반복되는 근거 문장입니다."
        content = f"{chunk_text}\n구분선\n{chunk_text}"
        second_start = content.rindex(chunk_text)
        document = Document("d1", "memory", "txt", content)
        first = Chunk(
            "d1_chunk_000", "d1", chunk_text,
            char_span=None, metadata={"chunk_index": 0},
        )
        second = Chunk(
            "d1_chunk_001", "d1", chunk_text,
            char_span=None, metadata={"chunk_index": 1},
        )
        node = KGNode(second.chunk_id, "d1", chunk_text)
        synthesized = _SynthesizedProbe(
            question="두 번째 근거는 무엇인가요?",
            ground_truth="두 번째 위치의 근거입니다.",
            evidence=[{"source_index": 0, "quote": chunk_text}],
        )

        spans, located, exact, fallback = _gold_spans_from_evidence(
            synthesized,
            [node],
            {first.chunk_id: first, second.chunk_id: second},
            {"d1": document},
        )

        self.assertEqual((located, exact, fallback), (1, 1, 0))
        self.assertEqual(spans[0]["start"], second_start)

    def test_multihop_missing_evidence_falls_back_per_source(self):
        documents = {
            "d1": Document("d1", "memory", "txt", "첫 번째 문서의 정확한 근거입니다."),
            "d2": Document("d2", "memory", "txt", "두 번째 문서의 보조 근거입니다."),
        }
        chunks = {
            "c1": Chunk(
                "c1", "d1", documents["d1"].content,
                char_span=(0, len(documents["d1"].content)),
            ),
            "c2": Chunk(
                "c2", "d2", documents["d2"].content,
                char_span=(0, len(documents["d2"].content)),
            ),
        }
        nodes = [
            KGNode("c1", "d1", chunks["c1"].text),
            KGNode("c2", "d2", chunks["c2"].text),
        ]
        synthesized = _SynthesizedProbe(
            question="두 문서의 관계는 무엇인가요?",
            ground_truth="두 근거를 함께 사용해야 합니다.",
            evidence=[
                {"source_index": 0, "quote": "정확한 근거"},
                {"source_index": 1, "quote": "원문에 없는 문장"},
            ],
        )

        spans, located, exact, fallback = _gold_spans_from_evidence(
            synthesized,
            nodes,
            chunks,
            documents,
        )

        self.assertEqual((located, exact, fallback), (2, 1, 1))
        self.assertEqual([span["doc_id"] for span in spans], ["d1", "d2"])
        self.assertEqual(spans[1]["doc_id"], "d2")
        self.assertEqual(spans[1]["start"], 0)
        self.assertEqual(spans[1]["end"], len(documents["d2"].content))
        self.assertEqual(
            [span["_grounding_quality"] for span in spans],
            ["exact", "chunk_fallback"],
        )

    @patch("agents.eval.probe_gen._llm_synthesize_query")
    def test_ragas_probe_sets_gold_fields_and_grounding_metadata(self, synthesize):
        content = "정책 신청은 전날 오후 여섯 시까지 제출해야 합니다."
        document = Document("d1", "memory", "txt", content)
        chunk = Chunk("c1", "d1", content, char_span=(0, len(content)))
        node = KGNode("c1", "d1", content)
        synthesize.return_value = _SynthesizedProbe(
            question="정책 신청 마감은 언제인가요?",
            ground_truth="전날 오후 여섯 시까지입니다.",
            evidence=[{"source_index": 0, "quote": "전날 오후 여섯 시까지"}],
        )

        probe = _make_ragas_probe(
            [node],
            "single_specific",
            None,
            0,
            {"c1": chunk},
            {"d1": document},
        )

        self.assertIsNotNone(probe)
        self.assertEqual(len(probe.gold_spans), 1)
        self.assertEqual(probe.gold_doc_id, "d1")
        self.assertEqual(probe.gold_char_span, (
            content.index("전날 오후 여섯 시까지"),
            content.index("전날 오후 여섯 시까지") + len("전날 오후 여섯 시까지"),
        ))
        self.assertEqual(probe.metadata["span_grounding"]["status"], "exact")
        self.assertEqual(
            probe.metadata["span_grounding"]["span_qualities"],
            ["exact"],
        )

    @patch("agents.eval.probe_gen._llm_synthesize_query", return_value=None)
    def test_ragas_heuristic_path_uses_short_exact_evidence(self, _synthesize):
        content = (
            "소개입니다. "
            "정책 신청은 전날 오후 여섯 시까지 제출해야 합니다. "
            "이후 요청은 다음 날 처리됩니다."
        )
        evidence = "정책 신청은 전날 오후 여섯 시까지 제출해야 합니다."
        document = Document("d1", "memory", "txt", content)
        chunk = Chunk("c1", "d1", content, char_span=(0, len(content)))
        node = KGNode("c1", "d1", content)

        probe = _make_ragas_probe(
            [node],
            "single_specific",
            None,
            0,
            {"c1": chunk},
            {"d1": document},
        )

        start = content.index(evidence)
        self.assertEqual(probe.gold_spans, [{
            "doc_id": "d1",
            "start": start,
            "end": start + len(evidence),
        }])
        self.assertEqual(probe.ground_truth, evidence)
        self.assertEqual(probe.metadata["span_grounding"]["status"], "exact")

    def test_heuristic_caps_unpunctuated_long_evidence(self):
        content = "근거내용" * 200
        node = KGNode("c1", "d1", content)

        result = _heuristic_synthesize_query([node])

        quote = result.evidence[0]["quote"]
        self.assertLess(len(quote), len(content))
        self.assertLessEqual(len(quote), 240)
        self.assertEqual(result.ground_truth, quote)
        self.assertIn(quote, content)

    def test_heuristic_evidence_preserves_dots_inside_tokens(self):
        cases = [
            (
                "자세한 내용은 https://example.com/docs에서 확인할 수 있습니다. "
                "다음 안내입니다.",
                "자세한 내용은 https://example.com/docs에서 확인할 수 있습니다.",
            ),
            (
                "버전 2.1은 2026.07.20부터 모든 직원에게 적용됩니다. "
                "다음 안내입니다.",
                "버전 2.1은 2026.07.20부터 모든 직원에게 적용됩니다.",
            ),
            (
                "오류는 support@example.com으로 신고해야 정상적으로 접수됩니다. "
                "다음 안내입니다.",
                "오류는 support@example.com으로 신고해야 정상적으로 접수됩니다.",
            ),
            (
                "금액은 1,234.56달러이며 내일까지 납부해야 합니다. "
                "다음 안내입니다.",
                "금액은 1,234.56달러이며 내일까지 납부해야 합니다.",
            ),
        ]

        for content, expected in cases:
            with self.subTest(content=content):
                self.assertEqual(_heuristic_evidence_of(content), expected)

    @patch("agents.eval.probe_gen._llm_synthesize_query", return_value=None)
    def test_multihop_heuristic_uses_exact_evidence_per_source(self, _synthesize):
        contents = {
            "d1": "안내입니다. 첫 번째 정책의 신청 기한은 매월 마지막 영업일입니다.",
            "d2": "개요입니다. 두 번째 정책의 승인 결과는 다음 달 첫 영업일에 통지됩니다.",
        }
        documents = {
            doc_id: Document(doc_id, "memory", "txt", content)
            for doc_id, content in contents.items()
        }
        chunks = {
            "c1": Chunk("c1", "d1", contents["d1"], char_span=(0, len(contents["d1"]))),
            "c2": Chunk("c2", "d2", contents["d2"], char_span=(0, len(contents["d2"]))),
        }
        nodes = [
            KGNode("c1", "d1", contents["d1"]),
            KGNode("c2", "d2", contents["d2"]),
        ]

        probe = _make_ragas_probe(
            nodes,
            "multi_specific",
            "shared_entity",
            0,
            chunks,
            documents,
        )

        self.assertEqual(len(probe.gold_spans), 2)
        self.assertTrue(all(
            span["end"] - span["start"] < len(contents[span["doc_id"]])
            for span in probe.gold_spans
        ))
        self.assertEqual(probe.metadata["span_grounding"]["status"], "exact")
        self.assertEqual(
            probe.metadata["span_grounding"]["span_qualities"],
            ["exact", "exact"],
        )

    @patch("agents.eval.probe_gen.llm_provider.chat_json")
    @patch("agents.eval.probe_gen.llm_provider.has_key", return_value=True)
    def test_llm_contract_parses_evidence_and_labels_sources(self, _has_key, chat_json):
        chat_json.return_value = {
            "question": "질문",
            "ground_truth": "정답",
            "evidence": [{"source_index": 0, "quote": "정확한 근거"}],
        }
        node = KGNode("c1", "d1", "정확한 근거가 포함된 충분히 긴 문서 조각입니다.")

        result = _llm_synthesize_query(
            [node],
            "single_specific",
            None,
            "직원",
            "formal",
            "short",
            "depth",
        )

        self.assertEqual(result.evidence[0]["quote"], "정확한 근거")
        self.assertIn("[SOURCE 0]", chat_json.call_args.kwargs["user"])

    @patch(
        "agents.eval.probe_gen._llm_generate_single_hop",
        return_value=("레거시 청크 질문", "레거시 청크 정답"),
    )
    def test_chunk_fallback_locates_legacy_chunk_without_char_span(self, _generate):
        chunk_text = "레거시 청크도 원문에서 위치를 다시 찾을 수 있어야 합니다."
        content = f"머리말\n{chunk_text}\n꼬리말"
        document = Document("d1", "memory", "txt", content)
        chunk = Chunk("legacy_0", "d1", chunk_text, char_span=None)

        probes = _from_chunks([chunk], 1, {"d1": document})

        self.assertEqual(len(probes), 1)
        self.assertEqual(probes[0].metadata["span_grounding"]["status"], "chunk_fallback")
        self.assertEqual(
            probes[0].metadata["span_grounding"]["span_qualities"],
            ["chunk_fallback"],
        )
        self.assertEqual(probes[0].gold_spans[0]["start"], content.index(chunk_text))

    @patch("agents.eval.probe_gen._llm_generate_single_hop", return_value=None)
    def test_chunk_heuristic_locates_selected_sentence_exactly(self, _generate):
        chunk_text = "제목입니다. 실제 근거로 사용할 충분히 구체적인 문장입니다. 마무리입니다."
        evidence = "실제 근거로 사용할 충분히 구체적인 문장입니다."
        document = Document("d1", "memory", "txt", chunk_text)
        chunk = Chunk("c1", "d1", chunk_text, char_span=(0, len(chunk_text)))

        probes = _from_chunks([chunk], 1, {"d1": document})

        start = chunk_text.index(evidence)
        self.assertEqual(probes[0].gold_spans, [{
            "doc_id": "d1",
            "start": start,
            "end": start + len(evidence),
        }])
        self.assertEqual(probes[0].metadata["span_grounding"]["status"], "exact")
        self.assertEqual(probes[0].metadata["gen_method"], "heuristic_evidence")

    def test_resync_replaces_gold_chunk_ids_after_rechunking(self):
        content = "가" * 100 + "정답근거" + "나" * 100
        document = Document("d1", "memory", "txt", content)
        start = content.index("정답근거")
        probe = Probe(
            probe_id="p1",
            question="질문",
            source="llm_generated",
            gold_chunk_ids=["old_chunk"],
            gold_spans=[{"doc_id": "d1", "start": start, "end": start + 4}],
        )
        chunks = [
            Chunk("new_0", "d1", content[:100], metadata={"chunk_index": 0}),
            Chunk("new_1", "d1", content[100:160], metadata={"chunk_index": 1}),
            Chunk("new_2", "d1", content[160:], metadata={"chunk_index": 2}),
        ]

        _resync_gold_chunk_ids([probe], chunks, [document])

        self.assertEqual(probe.gold_chunk_ids, ["new_1"])

    def test_resync_prefers_one_chunk_that_fully_contains_the_span(self):
        content = "가" * 1000
        document = Document("d1", "memory", "txt", content)
        probe = Probe(
            probe_id="p1",
            question="질문",
            source="taxonomy",
            gold_chunk_ids=["old_0", "old_1"],
            gold_spans=[{"doc_id": "d1", "start": 325, "end": 450}],
        )
        chunks = [
            Chunk("c0", "d1", content[0:400], char_span=(0, 400)),
            Chunk("c1", "d1", content[275:675], char_span=(275, 675)),
        ]

        _resync_gold_chunk_ids([probe], chunks, [document])

        self.assertEqual(probe.gold_chunk_ids, ["c1"])

    def test_resync_uses_minimum_continuous_cover_when_no_chunk_contains_span(self):
        content = "가" * 1000
        document = Document("d1", "memory", "txt", content)
        probe = Probe(
            probe_id="p1",
            question="질문",
            source="taxonomy",
            gold_spans=[{"doc_id": "d1", "start": 350, "end": 450}],
        )
        chunks = [
            Chunk("c0", "d1", content[0:400], char_span=(0, 400)),
            Chunk("redundant", "d1", content[300:380], char_span=(300, 380)),
            Chunk("c1", "d1", content[400:800], char_span=(400, 800)),
        ]

        _resync_gold_chunk_ids([probe], chunks, [document])

        self.assertEqual(probe.gold_chunk_ids, ["c0", "c1"])

    def test_resync_clears_stale_ids_when_no_current_chunk_matches(self):
        document = Document("d1", "memory", "txt", "짧은 문서")
        probe = Probe(
            probe_id="p1",
            question="질문",
            source="taxonomy",
            gold_chunk_ids=["old_chunk"],
            gold_spans=[{"doc_id": "d1", "start": 0, "end": 2}],
        )

        _resync_gold_chunk_ids([probe], [], [document])

        self.assertEqual(probe.gold_chunk_ids, [])


if __name__ == "__main__":
    unittest.main()
