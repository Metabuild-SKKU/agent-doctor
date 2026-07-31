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
    # 실제 CrossEncoder 처럼 max_length 등 키워드를 받는다 — 못 받으면 프로덕션이
    # 상한 없이 로드하는 폴백 경로로 새서, 테스트가 실제 경로를 검증하지 못한다.
    def __init__(self, scores, **kwargs):
        self.scores = scores
        self.kwargs = kwargs
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

    def test_loader_caps_reranker_input_length(self):
        """리랭크 1쌍의 비용은 청크 길이에 비례한다 — 상한이 없으면 chunk_size 처방 하나로
        검색 시간이 몇 배가 된다(모델 tokenizer 기본 상한은 8192)."""
        model_name = "test/max-length-reranker"
        fake_module = types.ModuleType("sentence_transformers")
        fake_module.CrossEncoder = lambda _name, **kw: _FakeCrossEncoder([0.1], **kw)

        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            model, status = qdrant_store._load_reranker(model_name)

        self.assertEqual(status, "ready")
        self.assertEqual(model.kwargs["max_length"], qdrant_store._RERANKER_MAX_LENGTH)

    def test_loader_falls_back_when_max_length_unsupported(self):
        """상한 인자를 못 받는 구현이어도 리랭킹 자체는 죽지 않는다(상한만 빠진다)."""
        model_name = "test/no-max-length-reranker"
        fake_module = types.ModuleType("sentence_transformers")
        fake_module.CrossEncoder = lambda _name: _FakeCrossEncoder([0.1])

        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            model, status = qdrant_store._load_reranker(model_name)

        self.assertEqual(status, "ready")
        self.assertEqual(model.kwargs, {})

    def test_search_reports_rerank_cost(self):
        """리랭크 소요 시간·쌍 수를 검색 결과에 실어 리포트가 비용을 집계할 수 있게 한다."""
        chunks = [
            {"chunk_id": f"c{i}", "doc_id": "d1", "text": f"alpha 문서 {i}", "metadata": {}}
            for i in range(4)
        ]
        model_name = "test/cost-reranker"
        qdrant_store._rerankers[model_name] = _FakeCrossEncoder([0.1, 0.2, 0.3])
        retriever = Retriever(
            chunks,
            RetrievalSettings(
                use_reranker=True, reranker_model=model_name, rerank_candidates=3
            ),
            client=None,
        )

        result = retriever.search_with_details("alpha", top_k=2)

        self.assertEqual(result["rerank_pairs"], 3)
        self.assertGreaterEqual(result["rerank_seconds"], 0.0)

    def test_search_reports_zero_rerank_cost_when_disabled(self):
        chunks = [{"chunk_id": "c1", "doc_id": "d1", "text": "alpha", "metadata": {}}]
        retriever = Retriever(chunks, RetrievalSettings(use_reranker=False), client=None)

        result = retriever.search_with_details("alpha", top_k=1)

        self.assertEqual(result["rerank_pairs"], 0)
        self.assertEqual(result["rerank_seconds"], 0.0)

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
            def __init__(self, _model_name, **_kwargs):
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
        fake_module.CrossEncoder = lambda _name, **kw: _FakeCrossEncoder([0.7], **kw)

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
            def __init__(self, _name, **_kwargs):
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
        fake_module.CrossEncoder = lambda _name, **kw: _FakeCrossEncoder([], **kw)

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
                "seconds": 0.0,
                "pairs": 0,
                "ms_per_pair": None,
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
        deferred = optimized.optimization_report.metadata[
            "runtime_deferred_prescriptions"
        ]
        # low_rank 의 처방은 enable_reranker 하나다 — 후보창 확대는
        # retrieval_rerank_candidate_miss 로 분리됐다(아래 테스트가 그쪽을 덮는다).
        self.assertEqual(
            {item["prescription_id"] for item in deferred},
            {"enable_reranker"},
        )

    def test_candidate_widening_is_deferred_while_reranker_is_off(self):
        """후보창 확대는 리랭커가 꺼져 있으면 의미가 없다 — 재시도 불가로 미룬다."""
        finding = Finding(
            finding_id="p1:retrieval_rerank_candidate_miss",
            type="retrieval_failure",
            severity="warning",
            description="정답 청크가 리랭커 후보창 밖",
            label="retrieval_rerank_candidate_miss",
            confirmed=True,
            affected_probes=["p1"],
        )
        state = AgentDoctorState(
            report=DiagnosticReport(
                report_id="unavailable",
                findings=[finding],
                overall_score=0.3,
                ragas_scores={"context_precision": 0.2},
                pass_threshold=False,
            ),
        )

        optimized = run_optimize(state)

        deferred = optimized.optimization_report.metadata[
            "runtime_deferred_prescriptions"
        ]
        widen = next(
            item
            for item in deferred
            if item["prescription_id"] == "widen_rerank_candidates"
        )
        self.assertEqual(widen["reason"], "reranker_disabled")
        self.assertFalse(widen["retryable"])
        self.assertNotIn(
            ("retrieval_rerank_candidate_miss", "widen_rerank_candidates"),
            optimized.blacklist,
        )

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

    def test_reranker_precision_floor_is_relaxed_when_low_rank_improves(self):
        before_findings = [
            self._finding(),
            Finding(
                finding_id="p2:retrieval_low_rank",
                type="retrieval_failure",
                severity="warning",
                description="정답 청크의 순위가 낮음",
                label="retrieval_low_rank",
                confirmed=True,
                affected_probes=["p2"],
            ),
        ]
        state = AgentDoctorState(
            report=DiagnosticReport(
                report_id="before",
                findings=before_findings,
                overall_score=0.41,
                ragas_scores={"context_precision": 0.50},
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

        state.report = DiagnosticReport(
            report_id="after",
            findings=[self._finding()],
            overall_score=0.46,
            ragas_scores={"context_precision": 0.20},
            runtime_summary={
                "reranker": {
                    "enabled": True,
                    "attempted": 2,
                    "applied": 2,
                }
            },
            pass_threshold=False,
        )
        state = run_optimize(state)

        # 완화 판정의 핵심: 처방은 유지되고(롤백 없음) precision 위반이 지워진다.
        self.assertTrue(state.index_config["use_reranker"])
        self.assertEqual(state.optimization_history[0].status, "applied")
        self.assertEqual(state.optimization_history[0].metadata["floor_violations"], [])

        # 순위 원인 분할 이후: low_rank 의 처방은 enable_reranker 하나뿐이라, 리랭커가 이미
        # 켜진 상태에서 같은 라벨이 남아 있으면 그 쌍은 더 시도할 게 없어(no_valid_candidate_values)
        # 소진 처리된다. 품질 때문에 롤백된 게 아니므로 위 세 단언(유지·applied·위반 없음)이
        # 완화 판정의 검증이고, 이 소진은 그와 별개다.
        #   분할 전에는 low_rank 에 widen_rerank_candidates 가 2순위로 달려 있어 다음 후보로
        #   넘어갔다. 지금은 창 확대가 retrieval_rerank_candidate_miss 로 옮겨갔고, 실제
        #   파이프라인에서도 리랭커가 켜진 뒤의 순위 실패는 그 라벨(또는 reranker_demotion)로
        #   잡히므로 low_rank 쪽이 소진돼도 처방이 막히지 않는다.
        self.assertIn(("retrieval_low_rank", "enable_reranker"), state.blacklist)

    def test_reranker_precision_floor_still_rolls_back_without_low_rank_improvement(self):
        state = AgentDoctorState(
            report=DiagnosticReport(
                report_id="before",
                findings=[self._finding()],
                overall_score=0.41,
                ragas_scores={"context_precision": 0.50},
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

        state.report = DiagnosticReport(
            report_id="after",
            findings=[self._finding()],
            overall_score=0.46,
            ragas_scores={"context_precision": 0.20},
            runtime_summary={
                "reranker": {
                    "enabled": True,
                    "attempted": 2,
                    "applied": 2,
                }
            },
            pass_threshold=False,
        )
        state = run_optimize(state)

        self.assertFalse(state.index_config["use_reranker"])
        self.assertEqual(state.optimization_history[0].status, "failed")
        self.assertEqual(
            state.optimization_history[0].metadata["floor_violations"],
            ["context_precision"],
        )
        self.assertIn(
            ("retrieval_low_rank", "enable_reranker"),
            state.blacklist,
        )

    def test_reranker_precision_floor_still_rolls_back_with_multiple_floor_violations(self):
        before_findings = [
            self._finding(),
            Finding(
                finding_id="p2:retrieval_low_rank",
                type="retrieval_failure",
                severity="warning",
                description="gold chunk is ranked too low",
                label="retrieval_low_rank",
                confirmed=True,
                affected_probes=["p2"],
            ),
        ]
        state = AgentDoctorState(
            report=DiagnosticReport(
                report_id="before",
                findings=before_findings,
                overall_score=0.41,
                ragas_scores={"context_precision": 0.50, "faithfulness": 0.80},
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

        state.report = DiagnosticReport(
            report_id="after",
            findings=[self._finding()],
            overall_score=0.46,
            ragas_scores={"context_precision": 0.20, "faithfulness": 0.40},
            runtime_summary={
                "reranker": {
                    "enabled": True,
                    "attempted": 2,
                    "applied": 2,
                }
            },
            pass_threshold=False,
        )
        state = run_optimize(state)

        self.assertFalse(state.index_config["use_reranker"])
        self.assertEqual(state.optimization_history[0].status, "failed")
        self.assertEqual(
            state.optimization_history[0].metadata["floor_violations"],
            ["context_precision", "faithfulness"],
        )
        self.assertIn(
            ("retrieval_low_rank", "enable_reranker"),
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

    def test_context_compression_runtime_metadata_is_preserved(self):
        refreshed = _refresh_runtime_metadata(
            [Chunk("c1", "d1", "body")],
            {
                "context_compression": True,
                "context_compression_max_contexts": 2,
                "context_compression_min_contexts": 1,
                "context_compression_max_sentences": 3,
            },
        )

        metadata = refreshed[0].metadata
        self.assertTrue(metadata["context_compression"])
        self.assertTrue(metadata["context.compression.enabled"])
        self.assertEqual(metadata["context_compression_max_contexts"], 2)
        self.assertEqual(metadata["context_filter_max_contexts"], 2)
        self.assertEqual(metadata["context_compression_min_contexts"], 1)
        self.assertEqual(metadata["context_filter_min_contexts"], 1)
        self.assertEqual(metadata["context_compression_max_sentences"], 3)
        self.assertEqual(metadata["context_filter_max_sentences"], 3)

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
