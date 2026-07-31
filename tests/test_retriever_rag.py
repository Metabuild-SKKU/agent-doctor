from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.index.qdrant_store import build_client, build_sparse_vector
from agents.rag.retriever import (
    build_retriever,
    get_retriever,
    reset_retriever_cache,
    _mmr_select,
)
from agents.rag.generator import (
    answer_question,
    answer_text,
    generate_answer,
    _build_prompt,
    _generation_temperature,
)
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

    def test_mmr_applies_end_to_end_and_diversifies(self):
        # 리뷰 blocker#1 회귀: 실제 dense 경로(결과 dict 에 embedding 없음)에서도 MMR 이
        # 원본 청크 임베딩을 chunk_id 로 조회해 발동하고 다양화하는지 end-to-end 로 고정.
        client = build_client(":memory:")
        chunks = [
            Chunk(chunk_id="a", doc_id="d", text="재택근무 규정 A", embedding=[1.0, 0.0, 0.0]),
            Chunk(chunk_id="b", doc_id="d", text="재택근무 규정 B", embedding=[0.99, 0.01, 0.0]),
            Chunk(chunk_id="c", doc_id="d", text="재택근무 규정 C", embedding=[0.98, 0.02, 0.0]),
            Chunk(chunk_id="z", doc_id="d", text="연차 규정", embedding=[0.0, 1.0, 0.0]),
        ]
        cfg = {
            "embedding_model": "test-model", "embedding_dimension": 3,
            "use_mmr": True, "mmr_lambda": 0.5, "mmr_candidates": 10, "top_k": 2,
        }
        retriever = build_retriever(chunks, config=cfg, client=client)
        with patch("agents.rag.retriever.embed", return_value=[1.0, 0.0, 0.0]):
            resp = retriever.search_with_details("재택근무", top_k=2)

        self.assertEqual(resp["search_mode"], "dense")
        self.assertTrue(resp["mmr_enabled"])
        self.assertTrue(resp["mmr_applied"], "MMR 이 실제 경로에서 발동해야 한다(no-op 회귀 방지)")
        picked = [r["chunk_id"] for r in resp["results"]]
        # 순수 관련성이면 근접중복 a,b 가 뽑히지만, MMR 은 다양한 z 를 끌어올린다.
        self.assertIn("z", picked)
        # 검색 결과 dict 에 무거운 embedding 을 싣지 않는다(조회는 _chunks_by_id 로).
        self.assertNotIn("embedding", resp["results"][0])

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
        self.assertEqual(len(prompt_contexts), 2)
        self.assertEqual(prompt_contexts[0].citation_index, 1)
        self.assertEqual(prompt_contexts[1].citation_index, 2)
        self.assertEqual(prompt_contexts[1].text, "재택근무는 주 2일까지 가능합니다.")

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
                    "context_compression_max_contexts": 2,
                },
            )

        prompt_contexts = generate.call_args.args[1]
        self.assertEqual(len(prompt_contexts), 2)
        self.assertEqual(prompt_contexts[0].citation_index, 1)
        self.assertEqual(prompt_contexts[1].citation_index, 3)
        self.assertEqual(prompt_contexts[1].text, "재택근무는 주 2일까지 가능합니다.")

    def test_context_compression_keeps_rank_one_when_synonym_has_no_overlap(self):
        contexts = [
            "To use vacation, get manager approval first. Then register it in HR.",
            "PTO questions can be sent to the HR team.",
        ]

        with patch("agents.rag.generator._llm_generate", return_value="ok") as generate:
            generate_answer(
                "What is the PTO request procedure?",
                contexts,
                config={
                    "context_compression": True,
                    "context_compression_max_contexts": 1,
                },
            )

        prompt_contexts = generate.call_args.args[1]
        self.assertEqual(len(prompt_contexts), 2)
        self.assertEqual(prompt_contexts[0].citation_index, 1)
        self.assertIn("manager approval", prompt_contexts[0].text)
        self.assertEqual(prompt_contexts[1].citation_index, 2)


class MmrSelectTests(unittest.TestCase):
    @staticmethod
    def _cand(cid: str, score: float, emb: list[float]) -> dict:
        return {"chunk_id": cid, "score": score, "embedding": emb}

    def _pool(self) -> list[dict]:
        # a,b,c 는 서로 거의 동일(중복), d,e 는 다양. score 내림차순.
        return [
            self._cand("a", 0.95, [1.0, 0.0, 0.0]),
            self._cand("b", 0.94, [0.99, 0.01, 0.0]),
            self._cand("c", 0.93, [0.98, 0.02, 0.0]),
            self._cand("d", 0.60, [0.0, 1.0, 0.0]),
            self._cand("e", 0.50, [0.0, 0.0, 1.0]),
        ]

    @staticmethod
    def _embs(pool: list[dict]) -> list[list[float]]:
        return [r["embedding"] for r in pool]

    def test_balances_relevance_and_diversity(self):
        pool = self._pool()
        picked = [r["chunk_id"] for r in _mmr_select(pool, 3, 0.5, self._embs(pool))]
        # 최상위 a 채택 후 중복(b,c) 대신 다양한 d,e 를 고른다.
        self.assertEqual(picked[0], "a")
        self.assertIn("d", picked)
        self.assertIn("e", picked)
        self.assertNotIn("b", picked)

    def test_lambda_one_is_pure_relevance(self):
        pool = self._pool()
        picked = [r["chunk_id"] for r in _mmr_select(pool, 3, 1.0, self._embs(pool))]
        self.assertEqual(picked, ["a", "b", "c"])

    def test_missing_embedding_returns_none(self):
        pool = [{"chunk_id": "x", "score": 1.0}]  # embedding 없음
        self.assertIsNone(_mmr_select(pool, 1, 0.5, [None]))


class GenerationConfigTests(unittest.TestCase):
    """B그룹 Tier1: generation 플래그가 프롬프트/온도에 반영되는지 고정."""

    def _sys(self, config):
        system, _ = _build_prompt("질문?", ["문서"], max_context_chars=500, config=config)
        return system

    def test_default_reproduces_grounded_prompt(self):
        # 기본값(config 없음/기본 플래그) = 과거 하드코딩 프롬프트 재현.
        system = self._sys(None)
        self.assertIn("제공된 컨텍스트만 근거로", system)
        self.assertIn("근거 번호를 대괄호로", system)
        self.assertNotIn("재진술", system)

    def test_flags_add_clauses(self):
        system = self._sys({
            "restate_question": True,
            "completeness_mode": True,
            "abstention_strict": True,
        })
        self.assertIn("재진술", system)
        self.assertIn("빠짐없이", system)
        self.assertIn("확신이 없으면", system)

    def test_grounding_off_loosens_prompt(self):
        system = self._sys({"grounding_strict": False, "require_citation": False})
        self.assertNotIn("제공된 컨텍스트만 근거로", system)
        self.assertNotIn("근거 번호를 대괄호로", system)

    def test_temperature_read_from_config(self):
        self.assertEqual(_generation_temperature(None), 0.0)
        self.assertEqual(_generation_temperature({"temperature": 0.7}), 0.7)
        self.assertEqual(_generation_temperature({"temperature": "bad"}), 0.0)


if __name__ == "__main__":
    unittest.main()
