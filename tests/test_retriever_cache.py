# get_retriever 적재 캐시 계약만 확인하는 테스트 (Qdrant 서버·임베딩 모델 없이).
from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from agents.rag import retriever as retriever_mod
from agents.rag.retriever import get_retriever, reset_retriever_cache


def _chunk(chunk_id: str, text: str = "본문입니다.") -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": "doc-1",
        "text": text,
        "hash": chunk_id,
        "embedding": [1.0, 0.0, 0.0, 0.0],
        "metadata": {"embedding_model": "test-model", "embedding_dimension": 4},
    }


class RetrieverCacheTests(unittest.TestCase):
    def setUp(self):
        reset_retriever_cache()
        self.addCleanup(reset_retriever_cache)

        patcher = patch.multiple(
            retriever_mod,
            build_client=Mock(return_value=Mock(name="qdrant")),
            ensure_collection=Mock(),
            upsert_chunks=Mock(),
            delete_document_chunks=Mock(),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_same_chunks_are_upserted_once(self):
        """Index → Eval → Serve 가 같은 청크로 불러도 적재는 한 번."""
        chunks = [_chunk("c1"), _chunk("c2")]

        first = get_retriever(chunks, {"top_k": 3})
        second = get_retriever(chunks, {"top_k": 3})

        retriever_mod.upsert_chunks.assert_called_once()
        retriever_mod.ensure_collection.assert_called_once()
        self.assertIs(first.client, second.client)

    def test_changed_chunks_trigger_repopulation(self):
        """재색인으로 청크가 바뀌면 캐시를 버리고 다시 적재한다."""
        get_retriever([_chunk("c1")], {"top_k": 3})
        get_retriever([_chunk("c1"), _chunk("c2")], {"top_k": 3})

        self.assertEqual(retriever_mod.upsert_chunks.call_count, 2)

    def test_search_only_config_change_reuses_index(self):
        """Optimize 가 top_k 만 바꾼 경우: 재적재 없이 검색 설정만 갈아끼운다."""
        chunks = [_chunk("c1")]

        first = get_retriever(chunks, {"top_k": 3})
        second = get_retriever(chunks, {"top_k": 9, "use_hybrid": True})

        retriever_mod.upsert_chunks.assert_called_once()
        self.assertEqual(first.settings.top_k, 3)
        self.assertEqual(second.settings.top_k, 9)
        self.assertTrue(second.settings.use_hybrid)

    def test_delete_doc_ids_runs_on_populate_only(self):
        """증분 삭제는 실제 적재가 일어날 때만 — 캐시가 맞으면 건너뛴다."""
        chunks = [_chunk("c1")]

        get_retriever(chunks, {}, delete_doc_ids=["doc-1"])
        get_retriever(chunks, {}, delete_doc_ids=["doc-1"])

        retriever_mod.delete_document_chunks.assert_called_once()

    def test_reset_forces_repopulation(self):
        chunks = [_chunk("c1")]

        get_retriever(chunks, {})
        reset_retriever_cache()
        get_retriever(chunks, {})

        self.assertEqual(retriever_mod.upsert_chunks.call_count, 2)

    def test_missing_embeddings_fall_back_to_keyword(self):
        """임베딩이 없으면 Qdrant 를 건드리지 않고 keyword 폴백으로 남는다."""
        plain = [{"chunk_id": "c1", "doc_id": "doc-1", "text": "본문", "metadata": {}}]

        built = get_retriever(plain, {})

        self.assertIsNone(built.client)
        retriever_mod.upsert_chunks.assert_not_called()


if __name__ == "__main__":
    unittest.main()
