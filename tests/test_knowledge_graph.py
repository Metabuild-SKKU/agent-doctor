"""Eval 질문 생성용 지식그래프의 블록 top-k 회귀 테스트."""
from __future__ import annotations

import unittest

from agents.eval.knowledge_graph import build_graph
from core.schema import Chunk


def _chunk(chunk_id: str, text: str, embedding=None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        text=text,
        embedding=embedding,
    )


class KnowledgeGraphTests(unittest.TestCase):
    def test_embedding_top_k_connects_nearest_chunks(self):
        graph = build_graph(
            [
                _chunk("c1", "원격 근무 정책", [1.0, 0.0]),
                _chunk("c2", "재택 근무 지침", [0.99, 0.01]),
                _chunk("c3", "연차 사용 안내", [0.0, 1.0]),
            ],
            {
                "eval_graph_device": "cpu",
                "eval_graph_top_k": 1,
                "eval_graph_batch_size": 2,
            },
        )

        self.assertIn("c2", {chunk_id for chunk_id, _score in graph.edges["c1"]})
        self.assertNotIn("c3", {chunk_id for chunk_id, _score in graph.edges["c1"]})

    def test_keyword_inverted_index_keeps_keyword_only_edge(self):
        graph = build_graph(
            [
                _chunk("c1", "alpha beta gamma delta"),
                _chunk("c2", "alpha beta gamma epsilon"),
                _chunk("c3", "vacation annual leave policy"),
            ],
            {
                "eval_graph_device": "cpu",
                "eval_graph_top_k": 1,
                "eval_graph_batch_size": 2,
            },
        )

        self.assertIn("c2", {chunk_id for chunk_id, _score in graph.edges["c1"]})

    def test_invalid_performance_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "eval_graph_top_k"):
            build_graph([], {"eval_graph_top_k": 0})


if __name__ == "__main__":
    unittest.main()
