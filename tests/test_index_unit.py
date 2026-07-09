"""Index Module의 외부 모델·서버 의존성 없는 단위 테스트."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agents.index.agent import CHUNK_STRATEGIES, _chunk_document, _chunk_text, run
from agents.index.graph_index import build_graph_artifacts
from agents.index.qdrant_store import (
    build_client,
    delete_document_chunks,
    ensure_collection,
    hybrid_search,
    search,
    upsert_chunks,
)
from core.schema import Chunk, Document
from core.state import AgentDoctorState


def _document(doc_id: str, content: str) -> Document:
    return Document(
        doc_id=doc_id,
        source=f"https://example.com/{doc_id}",
        format="md",
        content=content,
        metadata={"title": doc_id},
    )


class ChunkingTests(unittest.TestCase):
    def test_all_chunk_strategies_are_registered(self):
        self.assertEqual(
            set(CHUNK_STRATEGIES),
            {"fixed", "markdown", "recursive", "markdown_recursive"},
        )

    def test_overlap_and_max_size_are_respected(self):
        source = "가나다라마바사 " * 30
        chunks = _chunk_text(source, chunk_size=40, chunk_overlap=8)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(text) <= 40 for text, _, _ in chunks))
        self.assertTrue(all(source[start:end] == text for text, start, end in chunks))

    def test_markdown_section_is_preserved(self):
        document = _document(
            "guide",
            "# 설치\n설치 방법입니다.\n\n## Windows\nPowerShell을 사용합니다.",
        )

        drafts = _chunk_document(document, chunk_size=100, chunk_overlap=10)

        self.assertEqual(drafts[0].section, "설치")
        self.assertEqual(drafts[1].section, "설치 > Windows")
        self.assertEqual(document.content[drafts[1].start : drafts[1].end], drafts[1].text)

    def test_strategies_can_be_swapped_with_one_config_value(self):
        document = _document(
            "guide",
            "# 설치\n" + ("설치 설명 문장입니다. " * 8)
            + "\n## Windows\n" + ("PowerShell 설명입니다. " * 8),
        )

        fixed = _chunk_document(document, 40, 8, strategy="fixed")
        markdown = _chunk_document(document, 40, 8, strategy="markdown")
        recursive = _chunk_document(document, 40, 8, strategy="recursive")
        combined = _chunk_document(
            document,
            40,
            8,
            strategy="markdown_recursive",
        )

        self.assertTrue(all(chunk.section is None for chunk in fixed))
        self.assertEqual([chunk.section for chunk in markdown], ["설치", "설치 > Windows"])
        self.assertTrue(any(len(chunk.text) > 40 for chunk in markdown))
        self.assertTrue(all(chunk.section is None for chunk in recursive))
        self.assertTrue(all(len(chunk.text) <= 40 for chunk in recursive))
        self.assertTrue(all(chunk.section is not None for chunk in combined))
        self.assertTrue(all(len(chunk.text) <= 40 for chunk in combined))


class IndexRunTests(unittest.TestCase):
    def _state(self) -> AgentDoctorState:
        state = AgentDoctorState()
        state.index_config.update(
            {
                "chunk_size": 60,
                "chunk_overlap": 10,
                "embedding_model": "test-model",
                "embedding_dimension": 4,
                "graph_enabled": False,
            }
        )
        return state

    @patch("agents.index.agent.upsert_chunks")
    @patch("agents.index.agent.ensure_collection")
    @patch("agents.index.agent.build_client", return_value=Mock())
    @patch("agents.index.agent.embed", return_value=[1.0, 0.0, 0.0, 0.0])
    def test_run_validates_deduplicates_and_writes_metadata(
        self, mock_embed, _mock_client, _mock_collection, mock_upsert
    ):
        state = self._state()
        content = "# 규정\n재택근무는 주 2일까지 가능합니다."
        state.documents = [_document("doc-1", content), _document("doc-2", content)]
        state.index_config["use_hybrid"] = True

        result = run(state)

        self.assertEqual(result.status, "indexed")
        self.assertIsNone(result.error)
        self.assertEqual(result.index_artifacts["documents"], 1)
        self.assertTrue(result.chunks)
        self.assertEqual(result.chunks[0].section, "규정")
        self.assertEqual(
            content[result.chunks[0].char_span[0] : result.chunks[0].char_span[1]],
            result.chunks[0].text,
        )
        self.assertGreater(result.chunks[0].token_count, 0)
        self.assertEqual(len(result.chunks[0].hash), 16)
        self.assertEqual(result.chunks[0].metadata["embedding_model"], "test-model")
        self.assertIsNotNone(result.chunks[0].sparse_vector)
        self.assertEqual(mock_embed.call_count, len(result.chunks))
        mock_upsert.assert_called_once()

    @patch("agents.index.agent.upsert_chunks")
    @patch("agents.index.agent.ensure_collection")
    @patch("agents.index.agent.build_client", return_value=Mock())
    @patch("agents.index.agent.embed", return_value=[1.0, 0.0, 0.0, 0.0])
    def test_same_signature_reuses_embeddings(
        self, first_embed, _mock_client, _mock_collection, _mock_upsert
    ):
        state = self._state()
        state.documents = [_document("doc-1", "동일한 문서 본문입니다.")]
        first = run(state)
        self.assertEqual(first_embed.call_count, 1)

        with patch("agents.index.agent.embed") as second_embed:
            second = run(first)

        self.assertEqual(second.status, "indexed")
        self.assertEqual(second.index_artifacts["reused_embeddings"], 1)
        second_embed.assert_not_called()

    def test_invalid_overlap_returns_error_state(self):
        state = self._state()
        state.documents = [_document("doc-1", "본문")]
        state.index_config.update({"chunk_size": 10, "chunk_overlap": 10})

        result = run(state)

        self.assertEqual(result.status, "error")
        self.assertIn("chunk_overlap", result.error)

    def test_unknown_chunk_strategy_returns_error_state(self):
        state = self._state()
        state.documents = [_document("doc-1", "본문")]
        state.index_config["chunk_strategy"] = "unknown"

        result = run(state)

        self.assertEqual(result.status, "error")
        self.assertIn("chunk_strategy", result.error)

    def test_blank_document_fails_pydantic_validation(self):
        state = self._state()
        state.documents = [_document("doc-1", "   \n")]

        result = run(state)

        self.assertEqual(result.status, "error")
        self.assertIn("문서 검증 실패", result.error)

    @patch("agents.index.agent.embed", return_value=[1.0, 0.0, 0.0, 0.0])
    def test_same_doc_id_with_different_content_is_rejected(self, _mock_embed):
        state = self._state()
        state.documents = [
            _document("same-id", "첫 번째 본문"),
            _document("same-id", "두 번째 본문"),
        ]

        result = run(state)

        self.assertEqual(result.status, "error")
        self.assertIn("같은 doc_id", result.error)


class SearchAndGraphTests(unittest.TestCase):
    def test_qdrant_dense_search_round_trip(self):
        client = build_client(":memory:")
        ensure_collection(client, vector_dim=2)
        chunks = [
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
        ]
        upsert_chunks(client, chunks)

        results = search(client, [1.0, 0.0], top_k=1)

        self.assertEqual(results[0]["chunk_id"], "remote")

        delete_document_chunks(client, ["policy"])
        self.assertEqual(search(client, [1.0, 0.0], top_k=5), [])

    @patch("agents.index.qdrant_store.search")
    def test_hybrid_search_recovers_exact_keyword(self, dense_search):
        dense_search.return_value = [
            {
                "chunk_id": "dense",
                "doc_id": "d1",
                "text": "근무 제도 안내",
                "metadata": {},
                "score": 0.9,
                "section": None,
            }
        ]
        chunks = [
            {
                "chunk_id": "keyword",
                "doc_id": "d2",
                "text": "RAGAS Oracle Test 설정",
                "metadata": {},
            }
        ]

        results = hybrid_search(
            Mock(),
            query_vector=[1.0],
            query="Oracle Test",
            chunks=chunks,
            top_k=2,
            dense_weight=0.5,
        )

        self.assertEqual({item["chunk_id"] for item in results}, {"dense", "keyword"})

    def test_graph_artifacts_are_written(self):
        chunk = Chunk(
            chunk_id="doc_chunk_000",
            doc_id="doc",
            text="Qdrant는 벡터 검색과 metadata filter를 지원한다.",
            embedding=[1.0, 0.0],
            metadata={"title": "설계"},
        )
        with tempfile.TemporaryDirectory() as directory:
            artifacts = build_graph_artifacts(
                [chunk],
                {
                    "graph_extraction": "keyword",
                    "graph_output_dir": directory,
                    "graph_similarity_threshold": 0.9,
                },
            )

            self.assertTrue(Path(artifacts["graphml"]).exists())
            self.assertTrue(Path(artifacts["mermaid"]).exists())
            self.assertGreater(artifacts["graph_nodes"], 2)


if __name__ == "__main__":
    unittest.main()
