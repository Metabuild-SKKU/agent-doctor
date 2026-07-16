from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.rag.retriever import build_retriever
from agents.rag.generator import answer_question, answer_text, generate_answer
from core.schema import Chunk


class RetrieverTests(unittest.TestCase):
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
