from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.ingest.document_type import detect_document_type, has_math_signal
from agents.index.qdrant_store import (
    build_client,
    build_sparse_vector,
    keyword_search,
    normalize_math_text,
)
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

    def _policy_chunks(self):
        return [
            Chunk(chunk_id="remote", doc_id="policy",
                  text="재택근무는 주 2일까지 가능합니다."),
            Chunk(chunk_id="vacation", doc_id="policy", text="연차는 15일입니다."),
        ]

    def test_pre_rerank_ids_expose_the_reranker_input(self):
        """리랭크 직전 후보 순서를 남긴다 — Eval 이 '후보창 밖'과 '강등'을 가르는 유일한 신호."""
        retriever = build_retriever(self._policy_chunks(), config={"top_k": 1})

        response = retriever.search_with_details("재택근무", top_k=1)

        self.assertEqual(response["pre_rerank_ids"], ["remote"])
        self.assertIn("rerank_candidate_count", response)

    def test_apply_rerank_override_skips_the_rerank_stage(self):
        """순위 측정용 wide 재검색은 리랭크를 건너뛴다(프로덕션 순위와 다른 함수가 되지 않게)."""
        retriever = build_retriever(
            self._policy_chunks(),
            config={"top_k": 1, "use_reranker": True},
        )

        response = retriever.search_with_details(
            "재택근무", top_k=2, apply_rerank=False
        )

        self.assertFalse(response["reranked"])
        self.assertFalse(response["reranker_attempted"])
        self.assertEqual(response["reranker_status"], "disabled")
        self.assertEqual(response["results"][0]["chunk_id"], "remote")

    def test_math_expression_normalization_matches_korean_fraction(self):
        chunks = [
            Chunk(
                chunk_id="pi-four",
                doc_id="math",
                text="x=2cos^3(t), y=3sin^3(t), t=pi/4 일 때 접선의 기울기를 구한다.",
            )
        ]

        results = keyword_search(chunks, "t가 4분의 파이일 때 기울기", top_k=1)

        self.assertEqual(results[0]["chunk_id"], "pi-four")
        self.assertIn("pi/4", normalize_math_text("4분의 파이"))

    def test_math_expression_normalization_skips_general_text(self):
        text = "당분기말 자산총계는 전기말 이상으로 증가했다."

        self.assertEqual(normalize_math_text(text), text)

    def test_math_signal_skips_limit_and_user_id_terms(self):
        self.assertFalse(has_math_signal("API rate limit 정책"))
        self.assertFalse(has_math_signal("user_id 조회 정책"))
        self.assertEqual(normalize_math_text("API rate limit 정책"), "API rate limit 정책")

    def test_document_type_detection_uses_explicit_metadata_only(self):
        self.assertEqual(
            detect_document_type(
                "SET 17\n172번 문제\nx=2cos^3(t), y=3sin^3(t)\nt=π/4 일 때 접선의 기울기를 구한다."
            ),
            "general",
        )
        self.assertEqual(
            detect_document_type(
                "SET 17\n172번 문제\nx=2cos^3(t), y=3sin^3(t)\nt=π/4 일 때 접선의 기울기를 구한다.",
                {"document_type": "math"},
            ),
            "math",
        )
        self.assertEqual(
            detect_document_type("API rate limit 정책은 분당 요청 수를 제한한다. user_id별 quota를 기록한다."),
            "general",
        )

    def test_document_type_detection_preserves_explicit_metadata(self):
        self.assertEqual(
            detect_document_type("일반 본문", {"document_type": "finance_table"}),
            "finance_table",
        )
        self.assertEqual(
            detect_document_type("일반 본문", {"retrieval_profile": "math_formula"}),
            "math",
        )

    def test_math_expression_normalization_matches_pi_fraction_forms(self):
        chunks = [
            Chunk(
                chunk_id="pi-symbol",
                doc_id="math",
                text="t=π/4 일 때의 접선 기울기를 계산한다.",
            )
        ]

        results = keyword_search(chunks, "t=pi/4 기울기", top_k=1)

        self.assertEqual(results[0]["chunk_id"], "pi-symbol")
        self.assertIn("pi/4", normalize_math_text("π/4"))
        self.assertIn("π/4", normalize_math_text("pi/4"))

    def test_general_limit_query_does_not_prefer_math_alias(self):
        chunks = [
            Chunk(
                chunk_id="math-limit",
                doc_id="math",
                text="수열의 극한값과 극한을 계산한다.",
            ),
            Chunk(
                chunk_id="api-limit",
                doc_id="api",
                text="API rate limit 정책과 요청 제한을 설명한다.",
            ),
        ]

        results = keyword_search(chunks, "API rate limit 정책", top_k=1)

        self.assertEqual(results[0]["chunk_id"], "api-limit")

    def test_translation_aliases_do_not_leak_into_general_terms(self):
        self.assertNotIn("limit", normalize_math_text("수열의 극한값과 극한을 계산한다."))
        self.assertFalse(has_math_signal("대부분의 API rate limit 정책"))

    def test_reranker_uses_configured_candidate_pool_without_math_expansion(self):
        retriever = build_retriever(
            [
                Chunk(chunk_id=f"c-{i}", doc_id="math", text=f"수열 급수 후보 {i}")
                for i in range(80)
            ],
            config={"use_reranker": True, "rerank_candidates": 20, "top_k": 5},
        )
        seen = {}

        def fake_keyword_search(chunks, query, top_k=5):
            seen["top_k"] = top_k
            return [
                {
                    "chunk_id": chunk.get("chunk_id") if isinstance(chunk, dict) else chunk.chunk_id,
                    "doc_id": chunk.get("doc_id") if isinstance(chunk, dict) else chunk.doc_id,
                    "text": chunk.get("text") if isinstance(chunk, dict) else chunk.text,
                    "score": 1.0,
                }
                for chunk in chunks[:top_k]
            ]

        with patch("agents.rag.retriever.keyword_search", side_effect=fake_keyword_search):
            with patch("agents.rag.retriever.rerank_with_status", side_effect=lambda q, r, model_name, top_k: (r[:top_k], "applied")):
                response = retriever.search_with_details("t=pi/4 일 때 x=2인지 확인하라", top_k=5)

        self.assertEqual(seen["top_k"], 20)
        self.assertEqual(response["reranker_status"], "applied")
        self.assertEqual(len(response["results"]), 5)

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
