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

    def test_undeclared_math_text_does_not_expand_reranker_candidate_pool(self):
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

    def test_math_document_metadata_keeps_configured_candidate_pool(self):
        retriever = build_retriever(
            [
                Chunk(
                    chunk_id=f"c-{i}",
                    doc_id="math",
                    text=f"후보 {i}",
                    metadata={"document_type": "math"},
                )
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
                response = retriever.search_with_details("t=pi/4 기울기", top_k=5)

        self.assertEqual(seen["top_k"], 20)
        self.assertEqual(response["reranker_status"], "applied")

    def test_general_query_does_not_expand_reranker_candidate_pool_in_math_corpus(self):
        retriever = build_retriever(
            [
                Chunk(chunk_id=f"c-{i}", doc_id="math", text=f"x^{i} 수식 후보")
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
                retriever.search_with_details("user_id 조회 정책", top_k=5)

        self.assertEqual(seen["top_k"], 20)

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
