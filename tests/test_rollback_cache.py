"""롤백용 인덱스·진단 2-slot 캐시 통합 계약을 검증한다."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from qdrant_client import QdrantClient

from agents.eval import agent as eval_agent
from agents.index.agent import IndexTools, run as index_run
from agents.index.qdrant_store import (
    collection_index_cache_key,
    search as dense_search,
)
from agents.rag import retriever as retriever_module
from core.schema import Chunk, Document, Probe
from core.state import AgentDoctorState


def _document() -> Document:
    return Document(
        doc_id="doc-1",
        source="memory",
        format="txt",
        content="첫 문장입니다. 두 번째 문장입니다. 세 번째 문장입니다.",
    )


def _index_tools() -> tuple[IndexTools, Mock, Mock]:
    embed = Mock(return_value=[1.0, 0.0, 0.0, 0.0])
    get_retriever = Mock()
    tools = IndexTools(
        get_retriever=get_retriever,
        embed=embed,
        count_tokens=Mock(return_value=3),
        build_sparse_vector=Mock(return_value={"indices": [], "values": []}),
        build_graph_artifacts=Mock(return_value={}),
        embed_batch=None,
    )
    return tools, embed, get_retriever


class IndexRollbackCacheTest(unittest.TestCase):
    def test_a_b_a_restores_previous_index_without_embedding(self):
        state = AgentDoctorState(documents=[_document()])
        state.index_config.update(
            {
                "chunk_stage": "fixed",
                "chunk_size": 30,
                "chunk_overlap": 0,
                "embedding_model": "test-model",
                "embedding_dimension": 4,
                "graph_enabled": False,
            }
        )
        tools, embed, get_retriever = _index_tools()

        index_run(state, tools=tools)
        key_a = state.active_index_key
        chunks_a = [chunk.text for chunk in state.chunks]

        state.index_config["chunk_size"] = 20
        state.reindex_required = True
        index_run(state, tools=tools)
        key_b = state.active_index_key

        self.assertNotEqual(key_a, key_b)
        self.assertEqual(len(state.index_cache), 2)

        state.index_config["chunk_size"] = 30
        state.reindex_required = True
        embed.reset_mock()
        get_retriever.reset_mock()
        index_run(state, tools=tools)

        self.assertEqual(state.active_index_key, key_a)
        self.assertEqual([chunk.text for chunk in state.chunks], chunks_a)
        self.assertTrue(state.index_cache_hit)
        self.assertEqual(len(state.index_cache), 2)
        embed.assert_not_called()
        get_retriever.assert_called_once()

    def test_qdrant_slots_restore_a_and_never_exceed_two(self):
        retriever_module.reset_retriever_cache()
        self.addCleanup(retriever_module.reset_retriever_cache)
        client = QdrantClient(":memory:")

        def shared_get_retriever(
            chunks,
            config,
            delete_doc_ids=None,
        ):
            return retriever_module.get_retriever(
                chunks,
                config,
                client=client,
                delete_doc_ids=delete_doc_ids,
            )

        embed = Mock(return_value=[1.0, 0.0, 0.0, 0.0])
        tools = IndexTools(
            get_retriever=shared_get_retriever,
            embed=embed,
            count_tokens=Mock(return_value=3),
            build_sparse_vector=Mock(
                return_value={"indices": [], "values": []}
            ),
            build_graph_artifacts=Mock(return_value={}),
            embed_batch=None,
        )
        state = AgentDoctorState(documents=[_document()])
        state.index_config.update(
            {
                "chunk_stage": "fixed",
                "chunk_size": 30,
                "chunk_overlap": 0,
                "embedding_model": "test-model",
                "embedding_dimension": 4,
                "graph_enabled": False,
            }
        )

        index_run(state, tools=tools)
        key_a = state.active_index_key
        collection_a = state.index_artifacts[
            "qdrant_collection_name"
        ]
        texts_a = {chunk.text for chunk in state.chunks}
        self.assertEqual(
            collection_index_cache_key(client, collection_a),
            key_a,
        )

        state.index_config["chunk_size"] = 20
        state.reindex_required = True
        index_run(state, tools=tools)

        state.index_config["chunk_size"] = 30
        state.reindex_required = True
        index_run(state, tools=tools)

        collections = {
            item.name
            for item in client.get_collections().collections
        }
        restored = dense_search(
            client,
            [1.0, 0.0, 0.0, 0.0],
            top_k=20,
            collection_name=collection_a,
        )
        self.assertEqual(state.active_index_key, key_a)
        self.assertTrue(state.index_cache_hit)
        self.assertEqual(len(collections), 2)
        self.assertEqual(
            {item["text"] for item in restored},
            texts_a,
        )

    def test_pending_baseline_survives_multiple_candidates(self):
        state = AgentDoctorState(documents=[_document()])
        state.index_config.update(
            {
                "chunk_stage": "fixed",
                "chunk_size": 30,
                "chunk_overlap": 0,
                "embedding_model": "test-model",
                "embedding_dimension": 4,
                "graph_enabled": False,
            }
        )
        tools, embed, _ = _index_tools()

        index_run(state, tools=tools)
        key_a = state.active_index_key
        collection_a = state.index_artifacts["qdrant_collection_name"]
        state.optimization_history.append(
            SimpleNamespace(
                metadata={
                    "pending": True,
                    "before_index_key": key_a,
                }
            )
        )

        state.index_config["chunk_size"] = 20
        state.reindex_required = True
        index_run(state, tools=tools)
        key_b = state.active_index_key
        collection_b = state.index_artifacts["qdrant_collection_name"]

        state.index_config["chunk_size"] = 10
        state.reindex_required = True
        index_run(state, tools=tools)
        key_c = state.active_index_key

        self.assertEqual(
            [snapshot.cache_key for snapshot in state.index_cache],
            [key_a, key_c],
        )
        self.assertNotIn(
            key_b,
            [snapshot.cache_key for snapshot in state.index_cache],
        )
        self.assertEqual(
            state.index_artifacts["qdrant_collection_name"],
            collection_b,
        )
        self.assertNotEqual(collection_a, collection_b)

        state.index_config["chunk_size"] = 30
        state.reindex_required = True
        embed.reset_mock()
        index_run(state, tools=tools)

        self.assertEqual(state.active_index_key, key_a)
        self.assertTrue(state.index_cache_hit)
        embed.assert_not_called()

    def test_cache_restore_failure_returns_error_without_partial_commit(self):
        state = AgentDoctorState(documents=[_document()])
        state.index_config.update(
            {
                "chunk_stage": "fixed",
                "chunk_size": 30,
                "chunk_overlap": 0,
                "embedding_model": "test-model",
                "embedding_dimension": 4,
                "graph_enabled": False,
            }
        )
        tools, _, get_retriever = _index_tools()

        index_run(state, tools=tools)
        state.index_config["chunk_size"] = 20
        state.reindex_required = True
        index_run(state, tools=tools)
        key_b = state.active_index_key
        texts_b = [chunk.text for chunk in state.chunks]

        state.index_config["chunk_size"] = 30
        state.reindex_required = True
        get_retriever.side_effect = RuntimeError("재연결 실패")
        out = index_run(state, tools=tools)

        self.assertIs(out, state)
        self.assertEqual(out.status, "error")
        self.assertIn("Index 캐시 복원 실패", out.error)
        self.assertEqual(out.active_index_key, key_b)
        self.assertEqual([chunk.text for chunk in out.chunks], texts_b)
        self.assertFalse(out.index_cache_hit)

    def test_explicit_reindex_rebuilds_same_fingerprint_in_other_slot(self):
        state = AgentDoctorState(documents=[_document()])
        state.index_config.update(
            {
                "chunk_stage": "fixed",
                "chunk_size": 30,
                "chunk_overlap": 0,
                "embedding_model": "test-model",
                "embedding_dimension": 4,
                "graph_enabled": False,
            }
        )
        tools, embed, get_retriever = _index_tools()

        index_run(state, tools=tools)
        key = state.active_index_key
        first_collection = state.index_artifacts[
            "qdrant_collection_name"
        ]

        state.reindex_required = True
        embed.reset_mock()
        get_retriever.reset_mock()
        index_run(state, tools=tools)

        self.assertEqual(state.active_index_key, key)
        self.assertNotEqual(
            state.index_artifacts["qdrant_collection_name"],
            first_collection,
        )
        self.assertFalse(state.index_cache_hit)
        embed.assert_not_called()
        get_retriever.assert_called_once()

    def test_document_order_is_part_of_dedup_cache_key(self):
        first = _document()
        second = Document(
            doc_id="doc-2",
            source="memory-2",
            format="txt",
            content=first.content,
        )
        config = {
            "chunk_stage": "fixed",
            "chunk_size": 30,
            "chunk_overlap": 0,
            "embedding_model": "test-model",
            "embedding_dimension": 4,
            "graph_enabled": False,
        }

        state_ab = AgentDoctorState(documents=[first, second])
        state_ab.index_config.update(config)
        state_ba = AgentDoctorState(documents=[second, first])
        state_ba.index_config.update(config)

        tools_ab, _, _ = _index_tools()
        tools_ba, _, _ = _index_tools()
        index_run(state_ab, tools=tools_ab)
        index_run(state_ba, tools=tools_ba)

        self.assertNotEqual(
            state_ab.active_index_key,
            state_ba.active_index_key,
        )
        self.assertEqual(state_ab.chunks[0].doc_id, "doc-1")
        self.assertEqual(state_ba.chunks[0].doc_id, "doc-2")


class EvalRollbackCacheTest(unittest.TestCase):
    def test_a_b_a_restores_full_eval_without_generation(self):
        state = AgentDoctorState(
            documents=[_document()],
            user_questions=["무엇을 설명하나요?"],
        )
        state.index_config.update(
            {
                "top_k": 5,
                "graph_enabled": False,
            }
        )
        index_tools, _, _ = _index_tools()
        index_run(state, tools=index_tools)

        probe = Probe(
            probe_id="p1",
            question="무엇을 설명하나요?",
            source="user_log",
        )
        retriever = Mock()
        retriever.search.return_value = []

        def diagnose_record(record, _mode):
            record.signals["cached_top_k"] = state.index_config["top_k"]
            return []

        with (
            patch.dict("os.environ", {"EVAL_PROBE_SOURCE": "user_log"}),
            patch("agents.eval.agent.resolve_mode", return_value=3),
            patch(
                "agents.eval.agent.generate_probes",
                return_value=[probe],
            ) as generate_probes,
            patch(
                "agents.eval.agent.get_retriever",
                return_value=retriever,
            ) as get_retriever,
            patch("agents.eval.agent.generate_answer", return_value="답변") as generate,
            patch(
                "agents.eval.agent._ragas_track",
                return_value={},
            ) as ragas_track,
            patch(
                "agents.eval.agent.diagnose",
                side_effect=diagnose_record,
            ) as diagnose,
            patch("agents.eval.agent._log_probe"),
            patch("agents.eval.agent.print_summary"),
        ):
            eval_agent.run(state)
            key_a = state.active_eval_key
            report_a = state.report.report_id
            version_a = state.diagnosis_cache_version
            self.assertEqual(
                state.diagnosis_cache["p1"]["cached_top_k"],
                5,
            )

            state.index_config["top_k"] = 7
            eval_agent.run(state)
            key_b = state.active_eval_key
            self.assertNotEqual(key_a, key_b)
            self.assertEqual(len(state.eval_cache), 2)
            self.assertGreater(ragas_track.call_count, 0)

            state.index_config["top_k"] = 5
            generate_probes.reset_mock()
            get_retriever.reset_mock()
            generate.reset_mock()
            ragas_track.reset_mock()
            diagnose.reset_mock()
            retriever.search.reset_mock()
            eval_agent.run(state)

        self.assertEqual(state.active_eval_key, key_a)
        self.assertEqual(state.report.report_id, report_a)
        self.assertEqual(state.diagnosis_cache_version, version_a)
        self.assertEqual(
            state.diagnosis_cache["p1"]["cached_top_k"],
            5,
        )
        self.assertTrue(state.eval_cache_hit)
        self.assertEqual(len(state.eval_cache), 2)
        generate_probes.assert_not_called()
        get_retriever.assert_not_called()
        generate.assert_not_called()
        ragas_track.assert_not_called()
        diagnose.assert_not_called()
        retriever.search.assert_not_called()

    def test_auto_probe_file_created_on_first_run_still_hits_on_rollback(self):
        state = AgentDoctorState(documents=[_document()])
        state.index_config.update(
            {
                "top_k": 5,
                "graph_enabled": False,
            }
        )
        index_tools, _, _ = _index_tools()
        index_run(state, tools=index_tools)

        probe = Probe(
            probe_id="auto-p1",
            question="자동 질문",
            source="llm_generated",
        )
        retriever = Mock()
        retriever.search.return_value = []

        with TemporaryDirectory() as temp_dir:
            probe_path = Path(temp_dir) / "eval_probes.json"

            def load_probe_file(_version, *, ignore_version=False):
                del ignore_version
                return [probe] if probe_path.exists() else None

            def save_probe_file(_probes, version):
                probe_path.write_text(
                    f'{{"version": "{version}"}}',
                    encoding="utf-8",
                )

            with (
                patch.dict("os.environ", {"EVAL_PROBE_SOURCE": "auto"}),
                patch(
                    "agents.eval.agent.DEFAULT_STORE_PATH",
                    str(probe_path),
                ),
                patch("agents.eval.agent.resolve_mode", return_value=1),
                patch(
                    "agents.eval.agent.load_probes",
                    side_effect=load_probe_file,
                ) as load_probes,
                patch(
                    "agents.eval.agent.save_probes",
                    side_effect=save_probe_file,
                ),
                patch(
                    "agents.eval.agent.generate_probes",
                    return_value=[probe],
                ) as generate_probes,
                patch(
                    "agents.eval.agent.get_retriever",
                    return_value=retriever,
                ),
                patch(
                    "agents.eval.agent.generate_answer",
                    return_value="답변",
                ) as generate,
                patch("agents.eval.agent.diagnose", return_value=[]),
                patch("agents.eval.agent._log_probe"),
                patch("agents.eval.agent.print_summary"),
            ):
                eval_agent.run(state)
                key_a = state.active_eval_key
                self.assertTrue(probe_path.exists())

                state.index_config["top_k"] = 7
                eval_agent.run(state)

                state.index_config["top_k"] = 5
                load_probes.reset_mock()
                generate_probes.reset_mock()
                generate.reset_mock()
                retriever.search.reset_mock()
                eval_agent.run(state)

        self.assertEqual(state.active_eval_key, key_a)
        self.assertTrue(state.eval_cache_hit)
        load_probes.assert_not_called()
        generate_probes.assert_not_called()
        generate.assert_not_called()
        retriever.search.assert_not_called()

    def test_pending_baseline_survives_multiple_eval_candidates(self):
        state = AgentDoctorState(
            documents=[_document()],
            user_questions=["무엇을 설명하나요?"],
        )
        state.index_config.update(
            {
                "top_k": 5,
                "graph_enabled": False,
            }
        )
        index_tools, _, _ = _index_tools()
        index_run(state, tools=index_tools)

        probe = Probe(
            probe_id="p1",
            question="무엇을 설명하나요?",
            source="user_log",
        )
        retriever = Mock()
        retriever.search.return_value = []

        with (
            patch.dict("os.environ", {"EVAL_PROBE_SOURCE": "user_log"}),
            patch("agents.eval.agent.resolve_mode", return_value=1),
            patch("agents.eval.agent.generate_probes", return_value=[probe]),
            patch("agents.eval.agent.get_retriever", return_value=retriever),
            patch("agents.eval.agent.generate_answer", return_value="답변") as generate,
            patch("agents.eval.agent.diagnose", return_value=[]),
            patch("agents.eval.agent._log_probe"),
            patch("agents.eval.agent.print_summary"),
        ):
            eval_agent.run(state)
            key_a = state.active_eval_key
            state.optimization_history.append(
                SimpleNamespace(
                    metadata={
                        "pending": True,
                        "before_eval_key": key_a,
                    }
                )
            )

            state.index_config["top_k"] = 7
            eval_agent.run(state)
            key_b = state.active_eval_key

            state.index_config["top_k"] = 9
            eval_agent.run(state)
            key_c = state.active_eval_key

            self.assertEqual(
                [snapshot.cache_key for snapshot in state.eval_cache],
                [key_a, key_c],
            )
            self.assertNotIn(
                key_b,
                [snapshot.cache_key for snapshot in state.eval_cache],
            )

            state.index_config["top_k"] = 5
            generate.reset_mock()
            retriever.search.reset_mock()
            eval_agent.run(state)

        self.assertEqual(state.active_eval_key, key_a)
        self.assertTrue(state.eval_cache_hit)
        generate.assert_not_called()
        retriever.search.assert_not_called()


class RetrieverTwoSlotCacheTest(unittest.TestCase):
    def setUp(self):
        retriever_module.reset_retriever_cache()
        self.addCleanup(retriever_module.reset_retriever_cache)

    @staticmethod
    def _chunk(chunk_id: str) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            doc_id="doc-1",
            text=chunk_id,
            hash=chunk_id,
            embedding=[1.0, 0.0],
        )

    def test_replacing_one_slot_keeps_other_slot_cached(self):
        def cacheable_population(
            raw_chunks,
            _scope_id,
            _settings,
            client,
            _delete_doc_ids,
        ):
            return raw_chunks, client, True

        config_a = {"qdrant_collection_name": "agent_doctor_slot_0"}
        config_b = {"qdrant_collection_name": "agent_doctor_slot_1"}

        with patch(
            "agents.rag.retriever._populate",
            side_effect=cacheable_population,
        ) as populate:
            retriever_module.get_retriever([self._chunk("a")], config_a)
            retriever_module.get_retriever([self._chunk("b")], config_b)
            retriever_module.get_retriever([self._chunk("c")], config_b)
            retriever_module.get_retriever([self._chunk("a")], config_a)

        self.assertEqual(populate.call_count, 3)
        self.assertEqual(len(retriever_module._cached_entries), 2)

    def test_new_logical_index_never_reuses_old_slot_payload(self):
        def cacheable_population(
            raw_chunks,
            _scope_id,
            _settings,
            client,
            _delete_doc_ids,
        ):
            return raw_chunks, client, True

        old_chunk = self._chunk("same")
        old_chunk.metadata["payload_version"] = "old"
        new_chunk = self._chunk("same")
        new_chunk.metadata["payload_version"] = "new"

        with patch(
            "agents.rag.retriever._populate",
            side_effect=cacheable_population,
        ) as populate:
            retriever_module.get_retriever(
                [old_chunk],
                {
                    "qdrant_collection_name": "slot",
                    "index_cache_key": "index-old",
                },
            )
            restored = retriever_module.get_retriever(
                [new_chunk],
                {
                    "qdrant_collection_name": "slot",
                    "index_cache_key": "index-new",
                },
            )

        self.assertEqual(populate.call_count, 2)
        self.assertEqual(
            restored.chunks[0]["metadata"]["payload_version"],
            "new",
        )

    def test_runtime_metadata_refreshes_without_repopulation(self):
        def cacheable_population(
            raw_chunks,
            _scope_id,
            _settings,
            client,
            _delete_doc_ids,
        ):
            return raw_chunks, client, True

        old_chunk = self._chunk("same")
        old_chunk.metadata["top_k"] = 5
        new_chunk = self._chunk("same")
        new_chunk.metadata["top_k"] = 7
        config = {
            "qdrant_collection_name": "slot",
            "index_cache_key": "same-index",
        }

        with patch(
            "agents.rag.retriever._populate",
            side_effect=cacheable_population,
        ) as populate:
            retriever_module.get_retriever([old_chunk], config)
            refreshed = retriever_module.get_retriever(
                [new_chunk],
                config,
            )

        self.assertEqual(populate.call_count, 1)
        self.assertEqual(
            refreshed.chunks[0]["metadata"]["top_k"],
            7,
        )

    def test_existing_persistent_slot_reconnects_without_upsert(self):
        client = Mock()
        client.get_collections.return_value = SimpleNamespace(
            collections=[SimpleNamespace(name="persistent-slot")]
        )

        with (
            patch("agents.rag.retriever.ensure_collection") as ensure,
            patch("agents.rag.retriever.upsert_chunks") as upsert,
            patch(
                "agents.rag.retriever.collection_index_cache_key",
                return_value="persisted-index",
            ),
        ):
            restored = retriever_module.get_retriever(
                [self._chunk("persisted")],
                {
                    "qdrant_collection_name": "persistent-slot",
                    "index_cache_key": "persisted-index",
                    "reuse_existing_collection": True,
                },
                client=client,
            )

        self.assertIs(restored.client, client)
        ensure.assert_called_once()
        upsert.assert_not_called()

    def test_mismatched_persistent_slot_is_rebuilt(self):
        client = Mock()
        client.get_collections.return_value = SimpleNamespace(
            collections=[SimpleNamespace(name="persistent-slot")]
        )

        with (
            patch("agents.rag.retriever.ensure_collection") as ensure,
            patch("agents.rag.retriever.upsert_chunks") as upsert,
            patch(
                "agents.rag.retriever.collection_index_cache_key",
                return_value="unexpected-index",
            ),
        ):
            restored = retriever_module.get_retriever(
                [self._chunk("persisted")],
                {
                    "qdrant_collection_name": "persistent-slot",
                    "index_cache_key": "expected-index",
                    "reuse_existing_collection": True,
                },
                client=client,
            )

        self.assertIs(restored.client, client)
        client.delete_collection.assert_called_once_with(
            collection_name="persistent-slot"
        )
        ensure.assert_called_once()
        upsert.assert_called_once()


if __name__ == "__main__":
    unittest.main()
