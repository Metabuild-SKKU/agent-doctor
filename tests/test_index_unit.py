# 외부 모델이나 Qdrant 서버 없이 Index 계약만 확인하는 테스트.
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agents.index.agent import CHUNK_STRATEGIES, IndexTools, _chunk_document, run
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


def _index_tools() -> IndexTools:
    return IndexTools(
        get_retriever=lambda *_args, **_kwargs: Mock(),
        embed=lambda _text, **_kwargs: [1.0, 0.0, 0.0, 0.0],
        count_tokens=lambda _text, **_kwargs: 3,
        build_sparse_vector=lambda _text: {"indices": [], "values": []},
        build_graph_artifacts=lambda _chunks, _config: {},
    )


class ChunkingTests(unittest.TestCase):
    def test_all_chunk_strategies_are_registered(self):
        self.assertEqual(
            set(CHUNK_STRATEGIES),
            {"fixed", "markdown", "recursive", "markdown_recursive"},
        )

    def test_overlap_and_max_size_are_respected(self):
        source = "가나다라마바사 " * 30
        document = _document("fixed-doc", source)
        drafts = _chunk_document(document, chunk_size=40, chunk_overlap=8, strategy="fixed")

        self.assertGreater(len(drafts), 1)
        self.assertTrue(all(0 < len(d.text) <= 40 for d in drafts))
        self.assertTrue(all(document.content[d.start:d.end] == d.text for d in drafts))

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

    def test_numbered_chunk_stages_are_supported(self):
        document = _document(
            "guide",
            "# 설치\n" + ("설치 설명 문장입니다. " * 8)
            + "\n## Windows\n" + ("PowerShell 설명입니다. " * 8),
        )

        stage_1 = _chunk_document(document, 40, 8, strategy=1)
        stage_2 = _chunk_document(document, 40, 8, strategy=2)
        stage_3 = _chunk_document(document, 40, 8, strategy=3)

        self.assertTrue(all(chunk.section is None for chunk in stage_1))
        self.assertTrue(all(chunk.section is None for chunk in stage_2))
        self.assertTrue(all(chunk.section is not None for chunk in stage_3))
        self.assertTrue(all(len(chunk.text) <= 40 for chunk in stage_3))


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

    @patch("agents.index.agent.get_retriever")
    @patch("agents.index.agent.embed_batch", None)  # 단건 embed 폴백 경로로 강제
    @patch("agents.index.agent.embed", return_value=[1.0, 0.0, 0.0, 0.0])
    def test_run_validates_deduplicates_and_writes_metadata(
        self, mock_embed, mock_get_retriever
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
        mock_get_retriever.assert_called_once()

    @patch("agents.index.agent.get_retriever")
    @patch("agents.index.agent.embed_batch", None)
    @patch("agents.index.agent.embed", return_value=[1.0, 0.0, 0.0, 0.0])
    def test_same_signature_reuses_embeddings(
        self, first_embed, _mock_get_retriever
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

    def test_reused_chunks_refresh_retrieval_metadata(self):
        state = self._state()
        state.documents = [_document("doc-1", "metadata refresh target")]
        first = run(state, tools=_index_tools())

        first.index_config["top_k"] = 9
        first.index_config["use_reranker"] = True
        first.index_config["rerank_candidates"] = 40
        second = run(first, tools=_index_tools())

        self.assertEqual(second.status, "indexed")
        self.assertEqual(second.index_artifacts["reused_embeddings"], 1)
        self.assertEqual(second.chunks[0].metadata["top_k"], 9)
        self.assertTrue(second.chunks[0].metadata["use_reranker"])
        self.assertEqual(
            second.chunks[0].metadata["reranker_model"],
            "BAAI/bge-reranker-v2-m3",
        )
        self.assertEqual(second.chunks[0].metadata["rerank_candidates"], 40)

    def test_model_recovery_reembeds_fallback_chunks(self):
        # 리뷰 회귀: 최초 색인이 fallback(해시 벡터)으로 이뤄진 뒤 모델이 복구되면,
        # 같은 문서·설정으로 재색인해도 fallback 청크를 그대로 재사용하면 안 되고
        # 실제 모델 벡터로 강제 재임베딩해야 한다(문서·질의 벡터 공간 불일치 방지).
        FALLBACK_VEC = [0.5, 0.5, 0.5, 0.5]
        REAL_VEC = [1.0, 0.0, 0.0, 0.0]

        def _tools(is_fallback: bool) -> IndexTools:
            vec = FALLBACK_VEC if is_fallback else REAL_VEC
            return IndexTools(
                get_retriever=lambda *_a, **_k: Mock(),
                embed=lambda _t, **_k: list(vec),
                count_tokens=lambda _t, **_k: 3,
                build_sparse_vector=lambda _t: {"indices": [], "values": []},
                build_graph_artifacts=lambda _c, _cfg: {},
                embed_batch=lambda texts, **_k: [list(vec) for _ in texts],
                embedding_is_fallback=lambda *_a, **_k: is_fallback,
            )

        state = self._state()
        state.documents = [_document("doc-1", "복구 대상 문서 본문입니다.")]

        # 1) 모델 로드 실패 상태로 색인 → fallback 벡터 + provenance 기록
        first = run(state, tools=_tools(is_fallback=True))
        self.assertEqual(first.status, "indexed")
        self.assertTrue(first.chunks[0].metadata["embedding_fallback"])
        self.assertEqual(first.chunks[0].embedding, FALLBACK_VEC)

        # 2) 모델 복구 후 재색인 → fallback 청크는 실제 벡터로 재임베딩, 캐시 reset
        with patch("agents.index.agent.reset_retriever_cache") as mock_reset:
            second = run(first, tools=_tools(is_fallback=False))

        self.assertEqual(second.status, "indexed")
        # fallback 이었으므로 재사용이 아니라 재임베딩돼야 한다.
        self.assertEqual(second.index_artifacts["reused_embeddings"], 0)
        self.assertEqual(second.index_artifacts["reembedded_fallback"], len(second.chunks))
        self.assertEqual(second.chunks[0].embedding, REAL_VEC)
        self.assertFalse(second.chunks[0].metadata["embedding_fallback"])
        mock_reset.assert_called_once()

    def test_fallback_flag_survives_repeated_failures_then_recovery(self):
        # 리뷰 회귀(@SeonUI): 재사용 경로가 embedding_fallback 을 이어주지 않으면,
        # 모델이 두 번 연속 실패하는 사이에 플래그가 기본값 False 로 덮여
        # 이후 모델이 복구돼도 재임베딩 대상으로 잡히지 않는다(해시 벡터 영구 고착).
        # 위 2회 테스트는 이 경로를 지나지 않으므로 3회(실패→실패→복구)로 검증한다.
        FALLBACK_VEC = [0.5, 0.5, 0.5, 0.5]
        REAL_VEC = [1.0, 0.0, 0.0, 0.0]

        def _tools(is_fallback: bool) -> IndexTools:
            vec = FALLBACK_VEC if is_fallback else REAL_VEC
            return IndexTools(
                get_retriever=lambda *_a, **_k: Mock(),
                embed=lambda _t, **_k: list(vec),
                count_tokens=lambda _t, **_k: 3,
                build_sparse_vector=lambda _t: {"indices": [], "values": []},
                build_graph_artifacts=lambda _c, _cfg: {},
                embed_batch=lambda texts, **_k: [list(vec) for _ in texts],
                embedding_is_fallback=lambda *_a, **_k: is_fallback,
            )

        state = self._state()
        state.documents = [_document("doc-1", "두 번 실패 후 복구되는 문서 본문입니다.")]

        # 1) 모델 실패 → fallback 벡터로 색인되고 provenance 기록
        first = run(state, tools=_tools(is_fallback=True))
        self.assertTrue(first.chunks[0].metadata["embedding_fallback"])

        # 2) 모델 여전히 실패 → 임베딩 재사용 경로. 여기서 플래그가 유실되면 안 된다.
        second = run(first, tools=_tools(is_fallback=True))
        self.assertEqual(second.index_artifacts["reused_embeddings"], len(second.chunks))
        self.assertTrue(
            second.chunks[0].metadata["embedding_fallback"],
            "재사용 경로가 embedding_fallback 을 유실하면 복구 후 재임베딩이 동작하지 않는다",
        )
        self.assertEqual(second.chunks[0].embedding, FALLBACK_VEC)

        # 3) 모델 복구 → 2회차를 거쳤어도 여전히 재임베딩 대상이어야 한다
        with patch("agents.index.agent.reset_retriever_cache") as mock_reset:
            third = run(second, tools=_tools(is_fallback=False))

        self.assertEqual(third.status, "indexed")
        self.assertEqual(third.index_artifacts["reused_embeddings"], 0)
        self.assertEqual(third.index_artifacts["reembedded_fallback"], len(third.chunks))
        self.assertEqual(third.chunks[0].embedding, REAL_VEC)
        self.assertFalse(third.chunks[0].metadata["embedding_fallback"])
        mock_reset.assert_called_once()

    def test_recovered_model_still_reuses_non_fallback_chunks(self):
        # 정상(fallback 아님) 벡터로 색인된 청크는 모델이 로드 가능해도 그대로 재사용한다
        # (재임베딩은 fallback provenance 가 있는 청크에만 적용).
        def _tools() -> IndexTools:
            return IndexTools(
                get_retriever=lambda *_a, **_k: Mock(),
                embed=lambda _t, **_k: [1.0, 0.0, 0.0, 0.0],
                count_tokens=lambda _t, **_k: 3,
                build_sparse_vector=lambda _t: {"indices": [], "values": []},
                build_graph_artifacts=lambda _c, _cfg: {},
                embed_batch=lambda texts, **_k: [[1.0, 0.0, 0.0, 0.0] for _ in texts],
                embedding_is_fallback=lambda *_a, **_k: False,
            )

        state = self._state()
        state.documents = [_document("doc-1", "정상 색인 문서 본문입니다.")]
        first = run(state, tools=_tools())
        self.assertFalse(first.chunks[0].metadata["embedding_fallback"])

        with patch("agents.index.agent.reset_retriever_cache") as mock_reset:
            second = run(first, tools=_tools())

        self.assertEqual(second.index_artifacts["reused_embeddings"], len(second.chunks))
        self.assertEqual(second.index_artifacts["reembedded_fallback"], 0)
        mock_reset.assert_not_called()

    def test_runtime_only_config_change_skips_reindex_work(self):
        state = self._state()
        state.documents = [_document("doc-1", "기존 문서")]
        state.chunks = [
            Chunk("c1", "doc-1", "기존 문서", embedding=[1.0, 0.0, 0.0, 0.0])
        ]
        state.reindex_required = False
        tools = IndexTools(
            get_retriever=Mock(),
            embed=Mock(),
            count_tokens=Mock(),
            build_sparse_vector=Mock(),
            build_graph_artifacts=Mock(),
        )

        result = run(state, tools=tools)

        self.assertEqual(result.status, "indexed")
        self.assertTrue(result.index_artifacts["reindex_skipped"])
        self.assertTrue(result.reindex_required)
        tools.get_retriever.assert_not_called()
        tools.embed.assert_not_called()

    def test_reused_chunks_still_seed_chunk_deduplication(self):
        state = self._state()
        state.index_config.update(
            {
                "chunk_size": 6,
                "chunk_overlap": 0,
                "chunk_stage": 1,
            }
        )
        state.documents = [_document("doc-a", "shared")]
        first = run(state, tools=_index_tools())

        first.documents = [
            _document("doc-a", "shared"),
            _document("doc-b", "sharedzz"),
        ]
        second = run(first, tools=_index_tools())

        self.assertEqual(second.status, "indexed")
        self.assertEqual(second.index_artifacts["reused_embeddings"], 1)
        self.assertEqual(
            [(chunk.doc_id, chunk.text) for chunk in second.chunks],
            [("doc-a", "shared"), ("doc-b", "zz")],
        )

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

    @patch("agents.index.agent.get_retriever")
    @patch("agents.index.agent.embed_batch", None)
    @patch("agents.index.agent.embed", return_value=[1.0, 0.0, 0.0, 0.0])
    def test_chunk_stage_config_overrides_default_strategy(
        self, _mock_embed, _mock_get_retriever
    ):
        state = self._state()
        state.documents = [_document("doc-1", "# 제목\n" + ("본문입니다. " * 8))]
        state.index_config["chunk_stage"] = 1

        result = run(state)

        self.assertEqual(result.status, "indexed")
        self.assertEqual(result.index_artifacts["chunk_strategy"], "fixed")
        self.assertTrue(all(chunk.section is None for chunk in result.chunks))

    def test_run_accepts_swapped_index_tools(self):
        state = self._state()
        state.documents = [_document("doc-1", "도구 교체 테스트 본문입니다.")]
        upserted: list[list[Chunk]] = []
        tools = IndexTools(
            get_retriever=lambda chunks, *_args, **_kwargs: upserted.append(chunks),
            embed=lambda _text, **_kwargs: [0.0, 1.0, 0.0, 0.0],
            count_tokens=lambda _text, **_kwargs: 7,
            build_sparse_vector=lambda _text: {"indices": [], "values": []},
            build_graph_artifacts=lambda _chunks, _config: {},
        )

        result = run(state, tools=tools)

        self.assertEqual(result.status, "indexed")
        self.assertEqual(result.chunks[0].embedding, [0.0, 1.0, 0.0, 0.0])
        self.assertEqual(result.chunks[0].token_count, 7)
        self.assertEqual(len(upserted[0]), len(result.chunks))

    def test_blank_document_fails_pydantic_validation(self):
        state = self._state()
        state.documents = [_document("doc-1", "   \n")]

        result = run(state)

        self.assertEqual(result.status, "error")
        self.assertIn("문서 검증 실패", result.error)

    @patch("agents.index.agent.embed_batch", None)
    @patch("agents.index.agent.embed", return_value=[1.0, 0.0, 0.0, 0.0])
    def test_same_doc_id_with_different_content_skips_conflicting_document(self, _mock_embed):
        # 충돌 문서만 건너뛰고 먼저 들어온 문서는 정상 인덱싱되어야 한다.
        state = self._state()
        state.documents = [
            _document("same-id", "첫 번째 본문"),
            _document("same-id", "두 번째 본문"),
        ]

        result = run(state)

        self.assertEqual(result.status, "indexed")
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].text, "첫 번째 본문")
        failed = result.index_artifacts["failed_documents"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["doc_id"], "same-id")
        self.assertIn("같은 doc_id", failed[0]["error"])

    @patch("agents.index.agent.embed_batch", None)
    @patch("agents.index.agent.embed", return_value=[1.0, 0.0, 0.0, 0.0])
    def test_partial_failure_preserves_valid_documents(self, _mock_embed):
        # 불량 문서 1개가 나머지 정상 문서들의 작업을 버리게 만들면 안 된다.
        state = self._state()
        state.documents = [
            _document("doc-ok-1", "정상 문서 본문입니다."),
            _document("doc-bad", "   \n"),  # 공백뿐 → pydantic 검증 실패
            _document("doc-ok-2", "또 다른 정상 문서 본문입니다."),
        ]

        result = run(state)

        self.assertEqual(result.status, "indexed")
        indexed_doc_ids = {chunk.doc_id for chunk in result.chunks}
        self.assertEqual(indexed_doc_ids, {"doc-ok-1", "doc-ok-2"})
        failed = result.index_artifacts["failed_documents"]
        self.assertEqual([f["doc_id"] for f in failed], ["doc-bad"])

    def test_failed_document_does_not_pollute_chunk_dedup(self):
        # 임베딩 도중 실패한 문서의 청크 해시가 dedup 집합에 남으면
        # 뒤에 오는 동일 텍스트 청크가 중복으로 오인되어 누락된다.
        state = self._state()
        shared_text = "실패 후에도 인덱싱되어야 하는 본문입니다."
        state.documents = [
            _document("doc-fails", shared_text),
            _document("doc-succeeds", shared_text),
        ]

        calls = {"n": 0}

        def flaky_embed(text, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("임베딩 일시 실패")
            return [1.0, 0.0, 0.0, 0.0]

        with patch("agents.index.agent.embed_batch", None), \
                patch("agents.index.agent.embed", side_effect=flaky_embed):
            result = run(state)

        self.assertEqual(result.status, "indexed")
        self.assertEqual({c.doc_id for c in result.chunks}, {"doc-succeeds"})
        self.assertEqual(result.chunks[0].text, shared_text)
        failed = result.index_artifacts["failed_documents"]
        self.assertEqual([f["doc_id"] for f in failed], ["doc-fails"])


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


class ModelLoadCooldownTests(unittest.TestCase):
    """로드 실패를 영구 캐시하지 않고 쿨다운 후 재시도하는지 확인(embedding·reranker).

    sentence_transformers import 를 None 으로 막아 로드를 실패시키고, time.monotonic 을
    가짜로 진행시켜 '쿨다운 중에는 재시도 안 함 / 지나면 재시도함'을 검증한다.
    """

    def setUp(self):
        import agents.index.qdrant_store as store
        self.store = store
        # 각 테스트가 깨끗한 실패 캐시에서 시작하도록 초기화한다.
        store._failed_models.clear()
        store._failed_rerankers.clear()
        store._models.pop("m", None)
        store._rerankers.pop("m", None)
        self.addCleanup(store._failed_models.clear)
        self.addCleanup(store._failed_rerankers.clear)

    def test_embedding_model_retries_after_cooldown(self):
        store = self.store
        # import 를 막아 로드를 실패시킨다.
        with patch.dict(sys.modules, {"sentence_transformers": None}), \
             patch.object(store, "_FAILED_MODEL_RETRY_SEC", 300.0), \
             patch.object(store.time, "monotonic") as clock:
            clock.return_value = 1000.0
            self.assertIsNone(store._get_embedding_model("m"))
            self.assertIn("m", store._failed_models)

            # 쿨다운 중(=재시도 안 함): 실패 시각이 그대로여야 한다.
            clock.return_value = 1100.0   # +100s < 300s
            first_failed_at = store._failed_models["m"]
            self.assertIsNone(store._get_embedding_model("m"))
            self.assertEqual(store._failed_models["m"], first_failed_at)

            # 쿨다운 경과 후: 재시도하여 실패 시각이 갱신된다.
            clock.return_value = 1400.0   # +400s > 300s
            self.assertIsNone(store._get_embedding_model("m"))
            self.assertEqual(store._failed_models["m"], 1400.0)

    def test_reranker_retries_after_cooldown(self):
        store = self.store
        results = [{"chunk_id": "c1", "text": "t1", "score": 0.5}]
        with patch.dict(sys.modules, {"sentence_transformers": None}), \
             patch.object(store, "_FAILED_RERANKER_RETRY_SEC", 300.0), \
             patch.object(store.time, "monotonic") as clock:
            clock.return_value = 1000.0
            store.rerank("q", results, model_name="m", top_k=5)
            self.assertIn("m", store._failed_rerankers)
            first_failed_at = store._failed_rerankers["m"]

            # 쿨다운 중: 실패 시각 유지(재시도 안 함).
            clock.return_value = 1100.0
            store.rerank("q", results, model_name="m", top_k=5)
            self.assertEqual(store._failed_rerankers["m"], first_failed_at)

            # 쿨다운 경과 후: 재시도로 실패 시각 갱신.
            clock.return_value = 1400.0
            store.rerank("q", results, model_name="m", top_k=5)
            self.assertEqual(store._failed_rerankers["m"], 1400.0)


if __name__ == "__main__":
    unittest.main()
