from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.index.qdrant_store import build_client, build_sparse_vector
from agents.rag.retriever import build_retriever, get_retriever, reset_retriever_cache
from agents.rag.generator import answer_question, answer_text, generate_answer
from core.schema import Chunk


class RetrieverTests(unittest.TestCase):
    def tearDown(self):
        reset_retriever_cache()

    def test_keyword_fallback_works_without_embeddings(self):
        retriever = build_retriever(
            [
                Chunk(
                    chunk_id="remote",
                    doc_id="policy",
                    text="재택근무는 주 2일까지 가능합니다.",
                    metadata={"title": "근무 규정"},
                ),
                Chunk(
                    chunk_id="vacation",
                    doc_id="policy",
                    text="연차는 15일입니다.",
                    metadata={"title": "휴가 규정"},
                ),
            ],
            config={"top_k": 1},
        )

        response = retriever.search_with_details("재택근무", top_k=1)

        self.assertTrue(response["fallback_used"])
        self.assertEqual(response["search_mode"], "keyword")
        self.assertEqual(response["results"][0]["chunk_id"], "remote")

    def test_dense_search_uses_index_embeddings(self):
        retriever = build_retriever(
            [
                Chunk(
                    chunk_id="remote",
                    doc_id="policy",
                    text="재택근무 규정",
                    embedding=[1.0, 0.0],
                ),
                Chunk(
                    chunk_id="vacation",
                    doc_id="policy",
                    text="연차 규정",
                    embedding=[0.0, 1.0],
                ),
            ],
            config={
                "embedding_model": "test-model",
                "embedding_dimension": 2,
                "top_k": 1,
            },
        )

        with patch("agents.rag.retriever.embed", return_value=[1.0, 0.0]):
            response = retriever.search_with_details("재택근무", top_k=1)

        self.assertFalse(response["fallback_used"])
        self.assertEqual(response["search_mode"], "dense")
        self.assertEqual(response["results"][0]["chunk_id"], "remote")

    def test_shared_qdrant_client_is_limited_to_current_corpus(self):
        client = build_client(":memory:")
        build_retriever(
            [
                Chunk(
                    chunk_id="a-remote",
                    doc_id="corpus-a",
                    text="remote policy from another corpus",
                    embedding=[1.0, 0.0],
                )
            ],
            config={"embedding_model": "test-model", "embedding_dimension": 2},
            client=client,
        )
        retriever = build_retriever(
            [
                Chunk(
                    chunk_id="b-vacation",
                    doc_id="corpus-b",
                    text="vacation policy for this corpus",
                    embedding=[0.0, 1.0],
                )
            ],
            config={"embedding_model": "test-model", "embedding_dimension": 2},
            client=client,
        )

        with patch("agents.rag.retriever.embed", return_value=[1.0, 0.0]):
            response = retriever.search_with_details("remote policy", top_k=1)

        self.assertFalse(response["fallback_used"])
        self.assertEqual(response["search_mode"], "dense")
        self.assertEqual([item["doc_id"] for item in response["results"]], ["corpus-b"])

    def test_retriever_cache_reuses_population_when_hybrid_flag_changes(self):
        client = build_client(":memory:")
        chunks = [
            Chunk(
                chunk_id="policy",
                doc_id="doc",
                text="hybrid policy",
                embedding=[1.0, 0.0],
                sparse_vector=build_sparse_vector("hybrid policy"),
            )
        ]

        with patch("agents.rag.retriever.upsert_chunks") as upsert:
            get_retriever(
                chunks,
                config={"embedding_model": "test-model", "embedding_dimension": 2, "use_hybrid": False},
                client=client,
            )
            get_retriever(
                chunks,
                config={"embedding_model": "test-model", "embedding_dimension": 2, "use_hybrid": True},
                client=client,
            )

        self.assertEqual(upsert.call_count, 1)


class RagGeneratorTests(unittest.TestCase):
    def test_generate_answer_falls_back_to_top_context(self):
        with patch("agents.rag.generator._llm_generate", return_value=None):
            answer = generate_answer("재택근무 며칠?", ["재택근무는 주 2일까지 가능합니다."])

        self.assertEqual(answer, "재택근무는 주 2일까지 가능합니다.")

    def test_context_compression_filters_noise_before_generation(self):
        contexts = [
            "Remote work is allowed two days per week. Cafeteria menu changes daily.",
            "Parking registration is handled by the facilities team.",
        ]

        with patch("agents.rag.generator._llm_generate", return_value="ok") as generate:
            answer = generate_answer(
                "How many remote work days are allowed?",
                contexts,
                config={
                    "context_compression": True,
                    "context_compression_max_contexts": 1,
                },
            )

        self.assertEqual(answer, "ok")
        prompt_contexts = generate.call_args.args[1]
        self.assertEqual(
            [context.text for context in prompt_contexts],
            ["Remote work is allowed two days per week."],
        )

    def test_context_compression_disabled_preserves_contexts(self):
        contexts = [
            "Remote work is allowed two days per week. Cafeteria menu changes daily.",
            "Parking registration is handled by the facilities team.",
        ]

        with patch("agents.rag.generator._llm_generate", return_value="ok") as generate:
            generate_answer("How many remote work days are allowed?", contexts)

        prompt_contexts = generate.call_args.args[1]
        self.assertEqual([context.text for context in prompt_contexts], contexts)

    def test_context_compression_handles_korean_particles(self):
        contexts = [
            "식당 메뉴는 매일 바뀝니다.",
            "재택근무는 주 2일까지 가능합니다. 사무실 좌석 예약은 별도입니다.",
        ]

        with patch("agents.rag.generator._llm_generate", return_value="ok") as generate:
            generate_answer(
                "재택근무 가능 일수는?",
                contexts,
                config={"context_compression": True},
            )

        prompt_contexts = generate.call_args.args[1]
        self.assertEqual(len(prompt_contexts), 1)
        self.assertEqual(prompt_contexts[0].citation_index, 2)
        self.assertEqual(prompt_contexts[0].text, "재택근무는 주 2일까지 가능합니다.")

    def test_context_compression_trims_unmatched_contexts(self):
        contexts = [
            "Remote work is allowed two days per week. Cafeteria menu changes daily.",
            "Parking registration is handled by facilities. Lunch menus rotate daily.",
        ]

        with patch("agents.rag.generator._llm_generate", return_value="ok") as generate:
            generate_answer(
                "How many remote work days are allowed?",
                contexts,
                config={
                    "context_compression": True,
                    "context_compression_min_contexts": 2,
                    "context_compression_max_sentences": 1,
                },
            )

        prompt_contexts = generate.call_args.args[1]
        self.assertEqual(len(prompt_contexts), 2)
        self.assertEqual(prompt_contexts[0].text, "Remote work is allowed two days per week.")
        self.assertEqual(prompt_contexts[1].text, "Parking registration is handled by facilities.")

    def test_context_compression_min_contexts_overrides_smaller_max_contexts(self):
        contexts = [
            "Remote work is allowed two days per week.",
            "Parking registration is handled by facilities.",
            "Lunch menus rotate daily.",
        ]

        with patch("agents.rag.generator._llm_generate", return_value="ok") as generate:
            generate_answer(
                "How many remote work days are allowed?",
                contexts,
                config={
                    "context_compression": True,
                    "context_compression_max_contexts": 1,
                    "context_compression_min_contexts": 2,
                },
            )

        prompt_contexts = generate.call_args.args[1]
        self.assertEqual(len(prompt_contexts), 2)
        self.assertEqual(prompt_contexts[0].citation_index, 1)
        self.assertEqual(prompt_contexts[1].citation_index, 2)

    def test_context_compression_fallback_uses_compressed_context(self):
        contexts = [
            "Remote work is allowed two days per week. Cafeteria menu changes daily.",
            "Parking registration is handled by facilities.",
        ]

        with patch("agents.rag.generator._llm_generate", return_value=None):
            answer = generate_answer(
                "How many remote work days are allowed?",
                contexts,
                config={
                    "context_compression": True,
                    "context_compression_max_contexts": 1,
                },
            )

        self.assertEqual(answer, "Remote work is allowed two days per week.")

    def test_answer_question_marks_compressed_fallback_as_extractive(self):
        retriever = build_retriever(
            [
                Chunk(
                    chunk_id="remote",
                    doc_id="policy",
                    text="Remote work is allowed two days per week. Cafeteria menu changes daily.",
                )
            ]
        )

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch("agents.rag.generator._llm_generate", return_value=None),
        ):
            response = answer_question(
                "How many remote work days are allowed?",
                retriever,
                provider="openai",
                top_k=1,
                config={
                    "context_compression": True,
                    "context_compression_max_contexts": 1,
                },
            )

        self.assertEqual(response["answer"], "Remote work is allowed two days per week.")
        self.assertEqual(response["generation_mode"], "extractive")

    def test_context_compression_preserves_original_when_all_scores_are_zero(self):
        contexts = [
            "복리후생 제도는 별도 공지합니다.",
            "사내 교육 일정은 다음 주에 공개됩니다.",
        ]

        with patch("agents.rag.generator._llm_generate", return_value="ok") as generate:
            generate_answer(
                "원격 근무 신청 기준은?",
                contexts,
                config={"context_compression": True},
            )

        prompt_contexts = generate.call_args.args[1]
        self.assertEqual([context.text for context in prompt_contexts], contexts)

    def test_context_compression_preserves_citation_rank_after_filtering(self):
        contexts = [
            "식당 메뉴는 매일 바뀝니다.",
            "주차 등록은 시설팀에서 처리합니다.",
            "재택근무는 주 2일까지 가능합니다. 사무실 좌석 예약은 별도입니다.",
        ]

        with patch("agents.rag.generator._llm_generate", return_value="ok") as generate:
            generate_answer(
                "재택근무 가능 일수는?",
                contexts,
                config={
                    "context_compression": True,
                    "context_compression_max_contexts": 1,
                },
            )

        prompt_contexts = generate.call_args.args[1]
        self.assertEqual(len(prompt_contexts), 1)
        self.assertEqual(prompt_contexts[0].citation_index, 3)
        self.assertEqual(prompt_contexts[0].text, "재택근무는 주 2일까지 가능합니다.")

    def test_answer_question_returns_citations(self):
        retriever = build_retriever(
            [
                Chunk(
                    chunk_id="remote",
                    doc_id="policy",
                    text="재택근무는 주 2일까지 가능합니다.",
                    metadata={"title": "근무 규정"},
                )
            ]
        )

        with patch("agents.rag.generator._llm_generate", return_value=None):
            response = answer_question("재택근무 며칠?", retriever, top_k=1)

        self.assertEqual(response["answer"], "재택근무는 주 2일까지 가능합니다.")
        self.assertEqual(response["citations"][0]["chunk_id"], "remote")
        self.assertEqual(response["generation_mode"], "extractive")

    def test_answer_text_returns_only_answer(self):
        retriever = build_retriever(
            [
                Chunk(
                    chunk_id="remote",
                    doc_id="policy",
                    text="재택근무는 주 2일까지 가능합니다.",
                )
            ]
        )

        with patch("agents.rag.generator._llm_generate", return_value=None):
            answer = answer_text("재택근무 며칠?", retriever, top_k=1)

        self.assertEqual(answer, "재택근무는 주 2일까지 가능합니다.")


if __name__ == "__main__":
    unittest.main()
