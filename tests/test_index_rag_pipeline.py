from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from agents.index.agent import IndexTools, run as index_run
from agents.rag.generator import answer_question
from agents.rag.retriever import build_retriever, reset_retriever_cache
from core.schema import Document
from core.state import AgentDoctorState
from tests.embedding_env import pin_local_embedding


# Smoke test에서는 실제 embedding 모델 대신 고정 벡터를 써서 API 없이 검색 방향만 고정한다.
def _embedding_for_text(text: str, **_kwargs) -> list[float]:
    normalized = text.lower()
    if "remote work" in normalized:
        return [1.0, 0.0, 0.0]
    if "vacation" in normalized:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


# Index run이 외부 모델/그래프 생성에 기대지 않도록 필요한 도구만 테스트용으로 주입한다.
def _index_tools(get_retriever: Mock) -> IndexTools:
    return IndexTools(
        get_retriever=get_retriever,
        embed=_embedding_for_text,
        count_tokens=lambda text, **_kwargs: len(text.split()),
        build_sparse_vector=lambda _text: {"indices": [], "values": []},
        build_graph_artifacts=lambda _chunks, _config: {},
        embed_batch=lambda texts, **kwargs: [
            _embedding_for_text(text, **kwargs) for text in texts
        ],
        embedding_is_fallback=lambda _model: False,
        probe_reranker_capability=lambda **_kwargs: {
            "available": False,
            "status": "disabled",
        },
    )


class IndexRagPipelineSmokeTest(unittest.TestCase):
    def setUp(self):
        # 질의축이 openrouter 로 열려 있고 키가 없는 머신에서는 search_with_details 의
        # 설정 preflight 가 RuntimeError 로 끊는다 — 이 테스트가 코드가 아니라 .env 를
        # 재게 된다. 질의 임베딩 자체는 아래에서 대역으로 바꾸므로 실제 호출은 없다.
        pin_local_embedding(self)

    def tearDown(self):
        reset_retriever_cache()

    # Index가 만든 chunk가 RAG 검색과 답변 생성까지 그대로 이어지는 최소 사용자 흐름을 확인한다.
    def test_indexed_chunks_can_feed_retrieval_and_answer_generation(self):
        state = AgentDoctorState(
            documents=[
                Document(
                    doc_id="remote-policy",
                    source="memory://remote-policy",
                    format="md",
                    content=(
                        "# Remote Work\n"
                        "Remote work is allowed two days per week. "
                        "Requests must be submitted before Friday."
                    ),
                    metadata={"title": "Remote Work Policy"},
                ),
                Document(
                    doc_id="vacation-policy",
                    source="memory://vacation-policy",
                    format="md",
                    content=(
                        "# Vacation\n"
                        "Vacation requests require manager approval. "
                        "Unused vacation days follow the HR carryover rule."
                    ),
                    metadata={"title": "Vacation Policy"},
                ),
            ]
        )
        state.index_config.update(
            {
                "chunk_size": 500,
                "chunk_overlap": 0,
                "chunk_strategy": "markdown_recursive",
                "embedding_model": "test-embedding",
                "embedding_dimension": 3,
                "top_k": 1,
                "use_hybrid": False,
                "use_reranker": False,
                "reranker_preflight": "disabled",
                "graph_enabled": False,
                "corpus_visualization_enabled": False,
            }
        )
        get_retriever = Mock()

        indexed = index_run(state, tools=_index_tools(get_retriever))

        self.assertEqual(indexed.status, "indexed")
        self.assertEqual(len(indexed.chunks), 2)
        self.assertTrue(all(chunk.embedding for chunk in indexed.chunks))
        get_retriever.assert_called_once()
        self.assertEqual(get_retriever.call_args.args[0], indexed.chunks)

        retriever_config = {
            **indexed.index_config,
            "qdrant_collection_name": "test_index_rag_pipeline",
        }
        with (
            patch("agents.rag.retriever.embed", side_effect=_embedding_for_text),
            patch("agents.rag.generator._llm_generate", return_value=None),
        ):
            retriever = build_retriever(indexed.chunks, config=retriever_config)
            response = answer_question(
                "How many remote work days are allowed?",
                retriever,
                top_k=1,
            )

        self.assertEqual(response["retrieval"]["search_mode"], "dense")
        self.assertFalse(response["retrieval"]["fallback_used"])
        self.assertEqual(
            response["retrieval"]["results"][0]["doc_id"],
            "remote-policy",
        )
        self.assertIn("two days per week", response["answer"])
        self.assertEqual(response["generation_mode"], "extractive")
        self.assertEqual(
            response["citations"][0]["chunk_id"],
            "remote-policy_chunk_000",
        )


if __name__ == "__main__":
    unittest.main()
