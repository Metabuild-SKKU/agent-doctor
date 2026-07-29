from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from agents.eval.agent import _store_eval_snapshot
from agents.eval.report import build_report
from agents.eval.types import EvalRecord
from agents.index import qdrant_store
from agents.index.agent import (
    IndexTools,
    _refresh_runtime_metadata,
    run as run_index,
)
from agents.optimize.agent import run as run_optimize
from agents.rag.retriever import (
    RetrievalSettings,
    Retriever,
    resolve_retrieval_settings,
)
from core.schema import Chunk, DiagnosticReport, Document, Finding, Probe
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
        self.assertTrue(result["reranker_enabled"])
        self.assertTrue(result["reranker_attempted"])
        self.assertEqual(result["reranker_status"], "applied")
        self.assertFalse(result["reranker_fallback_used"])
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
        self.assertEqual(result["reranker_status"], "inference_failed")
        self.assertTrue(result["reranker_fallback_used"])
        self.assertEqual(
            [item["chunk_id"] for item in result["results"]],
            ["c1", "c2"],
        )
        self.assertNotIn(model_name, qdrant_store._rerankers)
        self.assertEqual(qdrant_store._failed_rerankers[model_name], 10.0)

    def test_concurrent_requests_load_cross_encoder_once(self):
        """동시 첫 요청도 같은 대형 모델을 한 번만 생성한다."""
        model_name = "test/concurrent-reranker"
        load_count = 0
        count_lock = threading.Lock()

        class _ConcurrentCrossEncoder:
            def __init__(self, _model_name):
                nonlocal load_count
                with count_lock:
                    load_count += 1
                time.sleep(0.05)

            def predict(self, pairs):
                return [1.0] * len(pairs)

        fake_module = types.ModuleType("sentence_transformers")
        fake_module.CrossEncoder = _ConcurrentCrossEncoder
        results = [{"chunk_id": "c1", "text": "본문", "score": 0.5}]

        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            with ThreadPoolExecutor(max_workers=6) as pool:
                outputs = list(
                    pool.map(
                        lambda _: qdrant_store.rerank(
                            "질문",
                            results,
                            model_name=model_name,
                            top_k=1,
                        ),
                        range(6),
                    )
                )

        self.assertEqual(load_count, 1)
        self.assertTrue(all(output[0]["score"] == 1.0 for output in outputs))


class RerankerCapabilityTest(unittest.TestCase):
    def tearDown(self):
        qdrant_store._rerankers.clear()
        qdrant_store._failed_rerankers.clear()

    def test_smoke_inference_marks_model_verified(self):
        model_name = "test/verified-reranker"
        fake_module = types.ModuleType("sentence_transformers")
        fake_module.CrossEncoder = lambda _name: _FakeCrossEncoder([0.7])

        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            capability = qdrant_store.probe_reranker_capability(model_name)

        self.assertEqual(capability["status"], "verified")
        self.assertIsNone(capability["reason"])
        self.assertIn(model_name, qdrant_store._rerankers)

    def test_missing_dependency_marks_model_unavailable(self):
        model_name = "test/missing-dependency"

        with patch.dict(sys.modules, {"sentence_transformers": None}):
            capability = qdrant_store.probe_reranker_capability(model_name)

        self.assertEqual(capability["status"], "unavailable")
        self.assertEqual(capability["reason"], "dependency_missing")
        self.assertFalse(capability["retryable"])

    def test_model_load_failure_marks_model_unavailable(self):
        model_name = "test/load-failure"

        class _BrokenCrossEncoder:
            def __init__(self, _name):
                raise OSError("download failed")

        fake_module = types.ModuleType("sentence_transformers")
        fake_module.CrossEncoder = _BrokenCrossEncoder
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            capability = qdrant_store.probe_reranker_capability(model_name)

        self.assertEqual(capability["status"], "unavailable")
        self.assertEqual(capability["reason"], "model_load_failed")
        self.assertTrue(capability["retryable"])

    def test_smoke_failure_marks_model_unavailable(self):
        model_name = "test/smoke-failure"
        fake_module = types.ModuleType("sentence_transformers")
        fake_module.CrossEncoder = lambda _name: _FakeCrossEncoder([])

        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            capability = qdrant_store.probe_reranker_capability(model_name)

        self.assertEqual(capability["status"], "unavailable")
        self.assertEqual(capability["reason"], "inference_failed")
        self.assertNotIn(model_name, qdrant_store._rerankers)

    def test_failure_cooldown_is_reported_without_reloading(self):
        model_name = "test/cooldown"
        qdrant_store._failed_rerankers[model_name] = 90.0

        with patch("agents.index.qdrant_store.time.monotonic", return_value=100.0):
            capability = qdrant_store.probe_reranker_capability(model_name)

        self.assertEqual(capability["status"], "unavailable")
        self.assertEqual(capability["reason"], "cooldown")
        self.assertTrue(capability["retryable"])

    def test_index_publishes_capability_without_failing_pipeline(self):
        capability = {
            "status": "unavailable",
            "model": qdrant_store.DEFAULT_RERANKER_MODEL,
            "checked_at": 1.0,
            "retryable": True,
            "reason": "model_load_failed",
        }
        probe = Mock(return_value=capability)
        tools = IndexTools(
            get_retriever=Mock(),
            embed=Mock(),
            count_tokens=Mock(),
            build_sparse_vector=Mock(),
            build_graph_artifacts=Mock(),
            probe_reranker_capability=probe,
        )
        state = AgentDoctorState(
            documents=[
                Document("d1", "memory://d1", "md", "본문"),
            ],
            chunks=[
                Chunk("c1", "d1", "본문", embedding=[1.0, 0.0]),
            ],
            reindex_required=False,
        )

        indexed = run_index(state, tools=tools)

        self.assertEqual(indexed.status, "indexed")
        self.assertEqual(
            indexed.runtime_capabilities["reranker"],
            capability,
        )
        self.assertEqual(
            indexed.index_artifacts["runtime_capabilities"]["reranker"],
            capability,
        )
        probe.assert_called_once_with(
            qdrant_store.DEFAULT_RERANKER_MODEL,
            smoke_test=True,
        )


class RerankerEvaluationSafetyTest(unittest.TestCase):
    def tearDown(self):
        qdrant_store._rerankers.clear()
        qdrant_store._failed_rerankers.clear()

    @staticmethod
    def _finding():
        return Finding(
            finding_id="p1:retrieval_low_rank",
            type="retrieval_failure",
            severity="warning",
            description="정답 청크의 순위가 낮음",
            label="retrieval_low_rank",
            confirmed=True,
            affected_probes=["p1"],
        )

    def test_report_counts_actual_reranker_execution(self):
        records = [
            EvalRecord(
                probe=Probe("p1", "질문1", "test"),
                retrieval_details={
                    "reranker_enabled": True,
                    "reranker_attempted": True,
                    "reranked": False,
                    "reranker_status": "load_failed",
                },
            ),
            EvalRecord(
                probe=Probe("p2", "질문2", "test"),
                retrieval_details={
                    "reranker_enabled": True,
                    "reranker_attempted": True,
                    "reranked": True,
                    "reranker_status": "applied",
                },
            ),
        ]

        report = build_report(records, iteration=1)

        self.assertEqual(
            report.runtime_summary["reranker"],
            {
                "enabled": True,
                "enabled_probes": 2,
                "attempted": 2,
                "applied": 1,
                "failed": 1,
                "status_counts": {
                    "load_failed": 1,
                    "applied": 1,
                },
            },
        )

    def test_incomplete_reranker_eval_is_not_cached(self):
        report = DiagnosticReport(
            report_id="failed-reranker",
            runtime_summary={
                "reranker": {
                    "enabled": True,
                    "attempted": 3,
                    "applied": 0,
                }
            },
        )
        state = AgentDoctorState(
            report=report,
            diagnosis_cache={"p1": {"stale": True}},
            diagnosis_cache_version="old",
        )

        _store_eval_snapshot(state, "same-config")

        self.assertEqual(state.eval_cache, [])
        self.assertEqual(state.diagnosis_cache, {})
        self.assertEqual(state.diagnosis_cache_version, "")

    def test_runtime_failure_rolls_back_and_tries_next_label(self):
        missing_gold = Finding(
            finding_id="p2:retrieval_missing_gold",
            type="retrieval_failure",
            severity="warning",
            description="정답 청크를 찾지 못함",
            label="retrieval_missing_gold",
            confirmed=True,
            affected_probes=["p2"],
        )
        state = AgentDoctorState(
            report=DiagnosticReport(
                report_id="before",
                findings=[self._finding(), missing_gold],
                overall_score=0.3,
                ragas_scores={"context_precision": 0.2},
                pass_threshold=False,
            ),
            runtime_capabilities={
                "reranker": {
                    "status": "verified",
                    "model": qdrant_store.DEFAULT_RERANKER_MODEL,
                    "retryable": False,
                    "reason": None,
                }
            },
        )
        state = run_optimize(state)
        self.assertTrue(state.index_config["use_reranker"])

        failed_report = DiagnosticReport(
            report_id="after-failed",
            findings=[self._finding(), missing_gold],
            overall_score=0.3,
            ragas_scores={"context_precision": 0.2},
            runtime_summary={
                "reranker": {
                    "enabled": True,
                    "attempted": 3,
                    "applied": 0,
                }
            },
            pass_threshold=False,
        )
        state.runtime_capabilities["reranker"] = {
            "status": "unavailable",
            "model": qdrant_store.DEFAULT_RERANKER_MODEL,
            "retryable": True,
            "reason": "inference_failed",
        }
        state.report = failed_report
        state = run_optimize(state)

        self.assertEqual(state.status, "applied")
        self.assertFalse(state.index_config["use_reranker"])
        self.assertGreater(state.index_config["top_k"], 5)
        self.assertNotIn(
            ("retrieval_low_rank", "enable_reranker"),
            state.blacklist,
        )
        self.assertEqual(len(state.optimization_history), 2)
        deferred = state.optimization_report.metadata[
            "runtime_deferred_prescriptions"
        ]
        self.assertTrue(
            any(
                item["prescription_id"] == "enable_reranker"
                for item in deferred
            )
        )

    def test_unavailable_reranker_is_deferred_without_quality_blacklist(self):
        state = AgentDoctorState(
            report=DiagnosticReport(
                report_id="unavailable",
                findings=[self._finding()],
                overall_score=0.3,
                ragas_scores={"context_precision": 0.2},
                pass_threshold=False,
            ),
            runtime_capabilities={
                "reranker": {
                    "status": "unavailable",
                    "model": qdrant_store.DEFAULT_RERANKER_MODEL,
                    "retryable": True,
                    "reason": "model_load_failed",
                }
            },
        )

        optimized = run_optimize(state)

        self.assertEqual(optimized.status, "skipped")
        self.assertFalse(optimized.index_config["use_reranker"])
        self.assertEqual(optimized.optimization_history, [])
        self.assertNotIn(
            ("retrieval_low_rank", "enable_reranker"),
            optimized.blacklist,
        )
        self.assertNotIn(
            ("retrieval_low_rank", "widen_rerank_candidates"),
            optimized.blacklist,
        )
        deferred = optimized.optimization_report.metadata[
            "runtime_deferred_prescriptions"
        ]
        self.assertEqual(
            {item["prescription_id"] for item in deferred},
            {"enable_reranker", "widen_rerank_candidates"},
        )
        widen = next(
            item
            for item in deferred
            if item["prescription_id"] == "widen_rerank_candidates"
        )
        self.assertEqual(widen["reason"], "reranker_disabled")
        self.assertFalse(widen["retryable"])

    def test_incomplete_reranker_execution_does_not_reopen_same_prescription(self):
        state = AgentDoctorState(
            report=DiagnosticReport(
                report_id="before",
                findings=[self._finding()],
                overall_score=0.3,
                ragas_scores={"context_precision": 0.2},
                pass_threshold=False,
            ),
            runtime_capabilities={
                "reranker": {
                    "status": "verified",
                    "model": qdrant_store.DEFAULT_RERANKER_MODEL,
                    "retryable": True,
                    "reason": None,
                }
            },
        )

        state = run_optimize(state)
        self.assertEqual(
            state.optimization_history[-1].selected_prescription_id,
            "enable_reranker",
        )
        self.assertEqual(state.iteration, 1)

        state.report = DiagnosticReport(
            report_id="after-incomplete",
            findings=[self._finding()],
            overall_score=0.3,
            ragas_scores={"context_precision": 0.2},
            runtime_summary={
                "reranker": {
                    "enabled": True,
                    "attempted": 1,
                    "applied": 0,
                }
            },
            pass_threshold=False,
        )
        state = run_optimize(state)

        self.assertEqual(state.status, "rolled_back")
        self.assertFalse(state.index_config["use_reranker"])
        self.assertEqual(len(state.optimization_history), 1)
        self.assertTrue(
            state.optimization_history[0].metadata[
                "reranker_execution_incomplete"
            ]
        )

        # Index/Eval을 한 번 거쳐 같은 finding이 유지된 다음 Optimize 방문을
        # 재현한다. 이전 처방과 무의미한 후보 수 확대 모두 다시 적용되지 않아야 한다.
        state.report = DiagnosticReport(
            report_id="baseline-restored",
            findings=[self._finding()],
            overall_score=0.3,
            ragas_scores={"context_precision": 0.2},
            pass_threshold=False,
        )
        state = run_optimize(state)

        self.assertEqual(state.status, "skipped")
        self.assertEqual(state.iteration, 1)
        self.assertEqual(len(state.optimization_history), 1)
        self.assertFalse(state.index_config["use_reranker"])
        self.assertEqual(state.index_config["rerank_candidates"], 20)
        self.assertNotIn(
            ("retrieval_low_rank", "enable_reranker"),
            state.blacklist,
        )
        self.assertNotIn(
            ("retrieval_low_rank", "widen_rerank_candidates"),
            state.blacklist,
        )


class RerankerMetadataValidationTest(unittest.TestCase):
    def test_invalid_runtime_values_fall_back_to_safe_defaults(self):
        chunks = [
            Chunk(
                chunk_id="c1",
                doc_id="d1",
                text="본문",
                metadata={},
            )
        ]

        refreshed = _refresh_runtime_metadata(
            chunks,
            {"rerank_candidates": None, "top_k": "not-a-number"},
        )

        self.assertEqual(refreshed[0].metadata["rerank_candidates"], 20)
        self.assertEqual(refreshed[0].metadata["top_k"], 5)

    def test_large_candidate_count_is_preserved_without_arbitrary_cap(self):
        settings = resolve_retrieval_settings(
            [],
            {"use_reranker": True, "rerank_candidates": 1_000_000},
        )
        refreshed = _refresh_runtime_metadata(
            [Chunk("c1", "d1", "본문")],
            {"rerank_candidates": 1_000_000},
        )

        self.assertEqual(settings.rerank_candidates, 1_000_000)
        self.assertEqual(
            refreshed[0].metadata["rerank_candidates"],
            1_000_000,
        )

    def test_index_normalizes_invalid_candidate_counts(self):
        cases = [
            (None, 20),
            ("", 20),
            ("invalid", 20),
            (0, 20),
            (-1, 20),
            (True, 20),
            ("40", 40),
        ]
        tools = IndexTools(
            get_retriever=Mock(),
            embed=Mock(),
            count_tokens=Mock(),
            build_sparse_vector=Mock(),
            build_graph_artifacts=Mock(),
        )

        for raw_value, expected in cases:
            with self.subTest(raw_value=raw_value):
                state = AgentDoctorState(
                    documents=[
                        Document("d1", "memory://d1", "md", "본문"),
                    ],
                    chunks=[
                        Chunk(
                            "c1",
                            "d1",
                            "본문",
                            embedding=[1.0, 0.0],
                        ),
                    ],
                    index_config={
                        "rerank_candidates": raw_value,
                        "reranker_preflight": "disabled",
                    },
                    reindex_required=False,
                )

                indexed = run_index(state, tools=tools)

                self.assertEqual(indexed.status, "indexed")
                self.assertEqual(
                    indexed.index_config["rerank_candidates"],
                    expected,
                )
                self.assertEqual(
                    indexed.chunks[0].metadata["rerank_candidates"],
                    expected,
                )


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
            runtime_capabilities={
                "reranker": {
                    "status": "verified",
                    "model": qdrant_store.DEFAULT_RERANKER_MODEL,
                    "retryable": False,
                    "reason": None,
                }
            },
        )

        optimized = run_optimize(state)

        self.assertEqual(optimized.status, "applied")
        self.assertTrue(optimized.index_config["use_reranker"])
        self.assertFalse(optimized.reindex_required)
        self.assertIn(
            "use_reranker: False → True",
            optimized.optimization_report.config_changes,
        )

        capability = {
            "status": "verified",
            "model": qdrant_store.DEFAULT_RERANKER_MODEL,
            "checked_at": 1.0,
            "retryable": False,
            "reason": None,
        }
        tools = IndexTools(
            get_retriever=Mock(),
            embed=Mock(),
            count_tokens=Mock(),
            build_sparse_vector=Mock(),
            build_graph_artifacts=Mock(),
            probe_reranker_capability=Mock(return_value=capability),
        )
        indexed = run_index(optimized, tools=tools)
        serve_settings = resolve_retrieval_settings(indexed.chunks)

        self.assertEqual(indexed.status, "indexed")
        self.assertTrue(indexed.index_artifacts["reindex_skipped"])
        self.assertEqual(
            indexed.runtime_capabilities["reranker"],
            capability,
        )
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
