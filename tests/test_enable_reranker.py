from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from agents.index import qdrant_store
from agents.index.agent import IndexTools, run as run_index
from agents.optimize.agent import run as run_optimize
from agents.rag.retriever import (
    RetrievalSettings,
    Retriever,
    resolve_retrieval_settings,
)
from core.schema import Chunk, DiagnosticReport, Document, Finding
from core.state import AgentDoctorState


class _FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        return list(self.scores)


class RerankerExecutionTest(unittest.TestCase):
    def tearDown(self):
        qdrant_store._rerankers.clear()
        qdrant_store._failed_rerankers.clear()

    def test_none_metadata_uses_default_reranker_model(self):
        settings = resolve_retrieval_settings(
            [
                {
                    "chunk_id": "c1",
                    "doc_id": "d1",
                    "text": "본문",
                    "metadata": {"reranker_model": None},
                }
            ],
            {"use_reranker": True},
        )

        self.assertEqual(
            settings.reranker_model,
            qdrant_store.DEFAULT_RERANKER_MODEL,
        )

    def test_enabled_reranker_scores_configured_candidate_count(self):
        chunks = [
            {
                "chunk_id": f"c{i}",
                "doc_id": "d1",
                "text": f"alpha 문서 {i}",
                "metadata": {},
            }
            for i in range(6)
        ]
        model_name = "test/fake-reranker"
        model = _FakeCrossEncoder([0.1, 0.2, 0.3, 0.4])
        qdrant_store._rerankers[model_name] = model
        retriever = Retriever(
            chunks,
            RetrievalSettings(
                use_reranker=True,
                reranker_model=model_name,
                rerank_candidates=4,
            ),
            client=None,
        )

        result = retriever.search_with_details("alpha", top_k=2)

        self.assertTrue(result["reranked"])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(len(model.calls[0]), 4)
        self.assertEqual(
            [item["chunk_id"] for item in result["results"]],
            ["c3", "c2"],
        )

    def test_inference_failure_keeps_original_order_and_reports_not_reranked(self):
        chunks = [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "text": "alpha 첫째",
                "metadata": {},
            },
            {
                "chunk_id": "c2",
                "doc_id": "d1",
                "text": "alpha 둘째",
                "metadata": {},
            },
        ]
        model_name = "test/broken-reranker"
        model = _FakeCrossEncoder([])
        qdrant_store._rerankers[model_name] = model
        retriever = Retriever(
            chunks,
            RetrievalSettings(
                use_reranker=True,
                reranker_model=model_name,
                rerank_candidates=2,
            ),
            client=None,
        )

        with patch("agents.index.qdrant_store.time.monotonic", return_value=10.0):
            result = retriever.search_with_details("alpha", top_k=2)

        self.assertFalse(result["reranked"])
        self.assertEqual(
            [item["chunk_id"] for item in result["results"]],
            ["c1", "c2"],
        )
        self.assertNotIn(model_name, qdrant_store._rerankers)
        self.assertEqual(qdrant_store._failed_rerankers[model_name], 10.0)


class EnableRerankerPipelineTest(unittest.TestCase):
    def test_low_rank_prescription_reaches_serve_metadata_without_reindex(self):
        finding = Finding(
            finding_id="p1:retrieval_low_rank",
            type="retrieval_failure",
            severity="warning",
            description="정답 청크의 순위가 낮음",
            label="retrieval_low_rank",
            confirmed=True,
            affected_probes=["p1"],
        )
        state = AgentDoctorState(
            documents=[
                Document(
                    doc_id="d1",
                    source="memory://d1",
                    format="md",
                    content="연차는 매년 15일 부여됩니다.",
                )
            ],
            chunks=[
                Chunk(
                    chunk_id="c1",
                    doc_id="d1",
                    text="연차는 매년 15일 부여됩니다.",
                    embedding=[1.0, 0.0],
                )
            ],
            report=DiagnosticReport(
                report_id="r1",
                findings=[finding],
                overall_score=30.0,
                ragas_scores={"context_precision": 0.2},
                pass_threshold=False,
            ),
        )

        optimized = run_optimize(state)

        self.assertEqual(optimized.status, "applied")
        self.assertTrue(optimized.index_config["use_reranker"])
        self.assertFalse(optimized.reindex_required)

        tools = IndexTools(
            get_retriever=Mock(),
            embed=Mock(),
            count_tokens=Mock(),
            build_sparse_vector=Mock(),
            build_graph_artifacts=Mock(),
        )
        indexed = run_index(optimized, tools=tools)
        serve_settings = resolve_retrieval_settings(indexed.chunks)

        self.assertEqual(indexed.status, "indexed")
        self.assertTrue(indexed.index_artifacts["reindex_skipped"])
        self.assertTrue(serve_settings.use_reranker)
        self.assertEqual(
            serve_settings.reranker_model,
            qdrant_store.DEFAULT_RERANKER_MODEL,
        )
        self.assertEqual(serve_settings.rerank_candidates, 20)
        tools.get_retriever.assert_not_called()
        tools.embed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
