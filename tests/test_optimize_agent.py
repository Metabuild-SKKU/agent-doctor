"""
tests/test_optimize_agent.py
Optimize 노드(agent.py)의 방문 간 판정·롤백 + graph 라우팅 통합 검증.

전체 파이프라인(Ingest/Index/Eval)은 외부 의존성(qdrant 등) 때문에 이 환경에서
end-to-end 로 돌릴 수 없으므로, 여기서는 Optimize 노드를 여러 번 호출하며
Eval 이 report 를 갱신하는 것을 손으로 흉내 내 검증한다.
"""
import os
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# graph.py 는 Index 팀 의존성(qdrant_client)까지 import 한다. 미설치 환경에서는
# 라우팅 함수 검증을 위해 최소 스텁을 주입한다(설치돼 있으면 그대로 사용).
try:  # pragma: no cover
    import qdrant_client  # noqa: F401
except ImportError:  # pragma: no cover
    import types

    class _AnyModule(types.ModuleType):
        def __getattr__(self, name):
            return type(name, (), {})

    for _n in (
        "qdrant_client",
        "qdrant_client.models",
        "qdrant_client.http",
        "qdrant_client.http.models",
        "requests",
        "sentence_transformers",
    ):
        sys.modules.setdefault(_n, _AnyModule(_n))

try:  # pragma: no cover
    import requests  # noqa: F401
except ImportError:  # pragma: no cover
    import types
    sys.modules.setdefault("requests", types.ModuleType("requests"))

try:  # pragma: no cover
    import langgraph.graph  # noqa: F401
except ImportError:  # pragma: no cover
    import types

    _langgraph = types.ModuleType("langgraph")
    _langgraph_graph = types.ModuleType("langgraph.graph")

    class _StateGraph:
        def __init__(self, *args, **kwargs):
            pass

        def add_node(self, *args, **kwargs):
            pass

        def set_entry_point(self, *args, **kwargs):
            pass

        def add_edge(self, *args, **kwargs):
            pass

        def add_conditional_edges(self, *args, **kwargs):
            pass

        def compile(self):
            return self

    _langgraph_graph.StateGraph = _StateGraph
    _langgraph_graph.END = "__end__"
    sys.modules.setdefault("langgraph", _langgraph)
    sys.modules.setdefault("langgraph.graph", _langgraph_graph)

import graph
from core.schema import DiagnosticReport, Finding
from core.state import AgentDoctorState
from agents.optimize import agent, history
from agents.optimize.schemas import ConfigPatch, OptimizationHistoryItem, OptimizationResult


def make_report(overall, pass_threshold=False, label="too_long_context"):
    """floor 는 통과하도록 넉넉한 ragas 를 주고, overall_score 로만 유지/롤백을 가른다."""
    finding = Finding(
        finding_id="1", type="retrieval_failure", severity="warning",
        description="desc", label=label, affected_probes=["p1"],
    )
    return DiagnosticReport(
        report_id="r", findings=[finding], overall_score=overall,
        ragas_scores={"context_recall": 0.7, "faithfulness": 0.7, "noise_sensitivity": 0.2},
        pass_threshold=pass_threshold,
    )


def make_state(overall=60.0, chunk_size=512, iteration=0, max_iterations=3,
               label="too_long_context"):
    return AgentDoctorState(
        report=make_report(overall, label=label),
        index_config={"top_k": 4, "chunk_size": chunk_size, "chunk_overlap": 50},
        iteration=iteration, max_iterations=max_iterations,
    )


class OptimizeAgentForwardTest(unittest.TestCase):
    def test_apply_creates_pending_and_increments_iteration(self):
        state = agent.run(make_state())
        self.assertEqual(state.status, "applied")
        self.assertEqual(state.iteration, 1)
        self.assertEqual(state.current_agent, "optimize")
        self.assertEqual(state.index_config["top_k"], 2)  # 가장 싼 top-k 처방이 먼저 적용됨
        self.assertFalse(state.reindex_required)
        self.assertIsNotNone(history.find_pending(state.optimization_history))

    def test_manual_label_makes_no_change(self):
        state = make_state(label="corpus_gap")
        before = dict(state.index_config)
        buf = StringIO()
        with redirect_stdout(buf):
            state = agent.run(state)
        self.assertEqual(state.status, "manual_required")
        self.assertEqual(state.iteration, 0)  # 수동 경로는 iteration 미소비
        self.assertEqual(state.index_config, before)
        self.assertIn("[Optimize] 반복 횟수: 0/3", buf.getvalue())

    def test_apply_log_matches_index_eval_route_without_physical_reindex(self):
        state = make_state(overall=0.42)
        state.report.composite_score = {"total": 40.0}
        state.report.findings_summary = {
            "confirmed_labels": {"too_long_context": 0.333}
        }

        buf = StringIO()
        with redirect_stdout(buf):
            state = agent.run(state)

        out = buf.getvalue()
        self.assertIn("[Optimize] 반복 횟수: 1/3", out)
        self.assertIn("Eval 결과: overall=0.42, composite=40.0, pass=false", out)
        self.assertIn("발견된 문제: too_long_context 1건", out)
        self.assertIn("다음 단계: Index 경유(물리 재색인 생략) 후 Eval 재실행", out)
        self.assertEqual(graph.route_after_optimize(state), "index")

    def test_apply_log_includes_added_config_keys(self):
        state = make_state(overall=0.42)
        request, decision = agent.planner.plan(state)
        result = OptimizationResult(
            request_id=request.request_id,
            status="proposed",
            optimizer="rules",
            selected_candidate=request.candidates[0],
            config_patch=ConfigPatch(
                changes={"embedding.model": "BAAI/bge-m3"},
                reindex_required=True,
            ),
            needs_reindex=True,
        )

        buf = StringIO()
        with patch("agents.optimize.agent.planner.plan", return_value=(request, decision)), patch(
            "agents.optimize.agent.optimizer.run", return_value=result
        ), redirect_stdout(buf):
            state = agent.run(state)

        out = buf.getvalue()
        self.assertEqual(state.index_config["embedding_model"], "BAAI/bge-m3")
        self.assertIn("변경 전 config: embedding_model=None", out)
        self.assertIn("변경 후 config: embedding_model='BAAI/bge-m3'", out)

    def test_prescreener_baseline_selection_tries_the_next_prescription(self):
        state = make_state(label="chunking_context_mismatch", chunk_size=400)
        state.report.findings[0].metadata["parameter_candidates"] = {
            "chunker.chunk_overlap": [50, 75]
        }
        real_optimizer_run = agent.optimizer.run
        calls = []

        def select_baseline_once(request):
            calls.append(request)
            if len(calls) == 1:
                return OptimizationResult(
                    request_id=request.request_id,
                    status="skipped",
                    optimizer="internal",
                    selected_candidate=request.candidates[0],
                    metadata={"error_code": "baseline_selected"},
                )
            return real_optimizer_run(request)

        with patch("agents.optimize.agent.optimizer.run", side_effect=select_baseline_once):
            result_state = agent.run(state)

        self.assertEqual(len(calls), 2)
        self.assertIn(
            ("chunking_context_mismatch", "increase_chunk_overlap"),
            result_state.blacklist,
        )
        self.assertEqual(result_state.index_config["chunk_overlap"], 50)
        self.assertEqual(result_state.index_config["chunk_size"], 800)
        self.assertEqual(result_state.status, "applied")
        self.assertEqual(result_state.iteration, 1)

    def test_inapplicable_prescription_falls_through_to_next(self):
        """issue #26: 최우선 라벨의 유일 처방이 적용 불가(enable_reranker 는 reranker
        미연동으로 unsupported_backend_path)여도, optimize 가 포기하지 않고 그 처방을
        블랙리스트에 넣은 뒤 다음 actionable finding 을 실제로 처방한다."""
        def _finding(pid, label):
            return Finding(
                finding_id=f"{pid}:{label}", type="retrieval_failure",
                severity="warning", description=label, label=label,
                confirmed=True, affected_probes=[pid],
            )
        # low_rank(적용 불가) 를 더 흔하게 → 최우선. semantic_mismatch(적용 가능) 는 차선.
        findings = (
            [_finding(f"lr{i}", "retrieval_low_rank") for i in range(6)]
            + [_finding(f"sm{i}", "retrieval_semantic_mismatch") for i in range(3)]
        )
        state = AgentDoctorState(
            report=DiagnosticReport(
                report_id="r", findings=findings, overall_score=30.0,
                ragas_scores={"context_recall": 0.4}, pass_threshold=False,
            ),
            index_config={"top_k": 5, "chunk_size": 512, "chunk_overlap": 50},
            iteration=0, max_iterations=3,
        )
        out = agent.run(state)
        self.assertEqual(out.status, "applied")
        # 적용 불가 처방은 블랙리스트에 남고
        self.assertIn(("retrieval_low_rank", "enable_reranker"), out.blacklist)
        # 실제로는 다음 우선순위(적용 가능) finding 이 처방됐다
        self.assertEqual(len(out.optimization_history), 1)
        self.assertEqual(
            out.optimization_history[-1].failure_labels,
            ["retrieval_semantic_mismatch"],
        )

    def test_always_returns_state_even_without_report(self):
        result = agent.run(AgentDoctorState(report=None, index_config={}, iteration=0))
        self.assertIsInstance(result, AgentDoctorState)


class OptimizeAgentRollbackTest(unittest.TestCase):
    def test_improved_keeps_config(self):
        state = agent.run(make_state(overall=60.0))         # 방문1: 적용
        applied = state.index_config["chunk_size"]
        state.report = make_report(75.0)                    # Eval: 개선
        state = agent.run(state)                            # 방문2: 판정 → 유지
        self.assertEqual(state.index_config["chunk_size"], applied)  # 유지됨
        self.assertEqual(state.optimization_history[0].status, "applied")
        self.assertEqual(len(state.blacklist), 0)

    def test_worse_rolls_back_and_blacklists(self):
        state = agent.run(make_state(overall=60.0))         # 방문1: top_k 4 -> 2
        self.assertEqual(state.index_config["top_k"], 2)
        state.report = make_report(50.0)                    # Eval: 악화
        state = agent.run(state)                            # 방문2: 판정 → 롤백
        self.assertEqual(state.status, "applied")           # 같은 라벨의 다음 처방은 계속 진행
        self.assertEqual(state.index_config["top_k"], 4)    # 첫 후보는 baseline으로 복원
        self.assertEqual(state.index_config["chunk_size"], 256)
        self.assertEqual(len(state.blacklist), 1)
        self.assertEqual(state.optimization_history[0].status, "failed")
        self.assertIsNotNone(state.optimization_history[0].rollback_reason)

    def test_budget_exhausted_allows_same_label_without_increment(self):
        state = agent.run(make_state(overall=60.0, iteration=2, max_iterations=3))
        self.assertEqual(state.iteration, 3)                # 방문1: 2 -> 3
        state.report = make_report(50.0)                    # 악화
        state = agent.run(state)                            # 방문2: 같은 라벨의 다음 처방
        self.assertEqual(state.iteration, 3)                # 후보/처방 전환은 증가 없음
        self.assertEqual(state.index_config["chunk_size"], 256)
        self.assertEqual(state.status, "applied")

    def test_baseline_dead_end_preserves_same_visit_rollback(self):
        state = agent.run(make_state(overall=60.0))
        state.report = make_report(50.0)
        request, decision = agent.planner.plan(make_state(overall=60.0))
        baseline_result = OptimizationResult(
            request_id=request.request_id,
            status="skipped",
            optimizer="internal",
            selected_candidate=request.candidates[0],
            metadata={"error_code": "baseline_selected"},
        )

        with patch("agents.optimize.agent.planner.plan", return_value=(request, decision)), patch(
            "agents.optimize.agent.optimizer.run", return_value=baseline_result
        ):
            state = agent.run(state)

        self.assertEqual(state.status, "rolled_back")
        self.assertEqual(state.index_config["top_k"], 4)
        self.assertEqual(state.optimization_report.status, "failed")


class OptimizeTopKSweepTest(unittest.TestCase):
    @staticmethod
    def _report(score):
        finding = Finding(
            finding_id="sweep", type="retrieval_failure", severity="warning",
            description="gold가 검색 결과에 없음", label="retrieval_missing_gold",
            affected_probes=["p1"],
            metadata={"parameter_candidates": {"retriever.top_k": [7, 9, 11]}},
        )
        return DiagnosticReport(
            report_id="sweep-report", findings=[finding], overall_score=score,
            ragas_scores={
                "context_recall": 0.7,
                "faithfulness": 0.7,
                "noise_sensitivity": 0.2,
            },
            pass_threshold=False,
        )

    def _state(self):
        return AgentDoctorState(
            report=self._report(60.0),
            index_config={"top_k": 5, "chunk_size": 512, "chunk_overlap": 50},
            iteration=0,
            max_iterations=1,
        )

    def test_candidates_share_one_iteration_and_best_is_selected(self):
        state = agent.run(self._state())
        self.assertEqual((state.index_config["top_k"], state.iteration), (7, 1))

        state.report = self._report(55.0)
        buf = StringIO()
        with redirect_stdout(buf):
            state = agent.run(state)
        self.assertEqual((state.index_config["top_k"], state.iteration), (9, 1))
        self.assertIn("선택한 처방: increase_top_k", buf.getvalue())

        state.report = self._report(70.0)
        state = agent.run(state)
        self.assertEqual((state.index_config["top_k"], state.iteration), (11, 1))

        state.report = self._report(65.0)
        buf = StringIO()
        with redirect_stdout(buf):
            state = agent.run(state)
        self.assertEqual(state.index_config["top_k"], 9)
        self.assertEqual(state.iteration, 1)
        self.assertIsNone(history.find_pending(state.optimization_history))
        self.assertEqual(len(state.optimization_history[0].metadata["trial_results"]), 4)
        self.assertIn("다음 단계: Index 경유(물리 재색인 생략) 후 Eval 재실행", buf.getvalue())

    def test_baseline_is_restored_only_after_all_candidates(self):
        state = self._state()
        state.report.findings[0].metadata["parameter_candidates"] = {
            "retriever.top_k": [7, 9]
        }
        state = agent.run(state)

        state.report = self._report(55.0)
        state.report.findings[0].metadata["parameter_candidates"] = {
            "retriever.top_k": [7, 9]
        }
        state = agent.run(state)
        self.assertEqual(state.index_config["top_k"], 9)
        self.assertFalse(state.blacklist)

        state.report = self._report(58.0)
        state.report.findings[0].metadata["parameter_candidates"] = {
            "retriever.top_k": [7, 9]
        }
        state = agent.run(state)
        self.assertEqual(state.index_config["top_k"], 5)
        self.assertEqual(state.status, "rolled_back")
        self.assertIn(
            ("retrieval_missing_gold", "increase_top_k"),
            state.blacklist,
        )

    def test_broken_active_study_restores_without_fake_error(self):
        state = agent.run(self._state())
        state.optimization_history[0].metadata.pop("study_request")

        state = agent.run(state)

        self.assertEqual(state.index_config["top_k"], 5)
        self.assertEqual(state.status, "rolled_back")
        self.assertIsNone(state.error)
        self.assertIn(
            ("retrieval_missing_gold", "increase_top_k"), state.blacklist
        )

    def test_transient_adapter_failure_does_not_immediately_blacklist(self):
        state = agent.run(self._state())
        failed = OptimizationResult(
            request_id="failed-study",
            status="failed",
            optimizer="internal",
            error="adapter 일시 실패",
        )

        with patch("agents.optimize.agent.optimizer.run", return_value=failed):
            state = agent.run(state)

        self.assertEqual(state.index_config["top_k"], 5)
        self.assertEqual(state.status, "rolled_back")
        self.assertIsNone(state.error)
        self.assertNotIn(
            ("retrieval_missing_gold", "increase_top_k"), state.blacklist
        )

    def test_floor_violating_sweep_winner_restores_baseline(self):
        state = self._state()
        state.report.findings[0].metadata["parameter_candidates"] = {
            "retriever.top_k": [7, 9]
        }
        state = agent.run(state)

        state.report = self._report(90.0)
        state.report.ragas_scores["context_recall"] = 0.2
        state = agent.run(state)

        state.report = self._report(80.0)
        state = agent.run(state)

        self.assertEqual(state.index_config["top_k"], 5)
        self.assertEqual(state.status, "rolled_back")
        self.assertIn("context_recall", state.optimization_history[0].rollback_reason)


class OptimizeReportWiringTest(unittest.TestCase):
    """agent.py 가 매 방문마다 state.optimization_report 를 알맞게 채우는지 검증."""

    def test_apply_stores_pending_report(self):
        state = agent.run(make_state())                     # 방문1: 적용
        report = state.optimization_report
        self.assertIsNotNone(report)
        self.assertEqual(report.status, "proposed")          # 검증 대기
        self.assertTrue(report.config_changes)               # 변경 내역 담김

    def test_keep_stores_applied_trial_report(self):
        # 방문2가 '판정만' 하도록 예산을 소진시킨다(예산이 남으면 새 처방 적용이 headline).
        state = agent.run(make_state(overall=60.0, iteration=2, max_iterations=3))
        state.report = make_report(75.0, label="retrieval_missing_gold")  # 개선 + 다음 라벨
        state = agent.run(state)                             # 방문2: 예산소진 → 유지 판정만
        report = state.optimization_report
        self.assertEqual(report.status, "applied")
        self.assertIn("유지", report.summary)

    def test_rollback_stores_failed_trial_report(self):
        state = agent.run(make_state(overall=60.0, iteration=2, max_iterations=3))
        state.report = make_report(50.0, label="retrieval_missing_gold")  # 악화 + 다음 라벨
        buf = StringIO()
        with redirect_stdout(buf):
            state = agent.run(state)                         # 방문2: 예산소진 → 판정만(롤백)
        report = state.optimization_report
        self.assertEqual(report.status, "failed")
        self.assertIn("되돌렸", report.summary)
        self.assertGreater(len(report.metadata.get("floor_violations", [])) +
                           int("점수" in report.summary), 0)  # 롤백 사유가 실림
        out = buf.getvalue()
        self.assertIn("선택한 처방:", out)
        self.assertIn("다음 단계: Index 경유(물리 재색인 생략) 후 Eval 재실행", out)

    def test_rollback_baseline_carries_restored_score_not_degraded(self):
        """#2 회귀: 롤백 후 같은 방문에서 이어 제안되는 처방의 비교 기준(before_report)은
        복원된 baseline 점수여야 한다. 롤백 직전의 열화된 Eval 을 baseline 으로 쓰면
        원래보다 나쁜 처방도 '개선'으로 오판해 유지된다."""
        # 방문1: Rx1 적용 (baseline=60)
        state = agent.run(make_state(overall=60.0, label="too_long_context"))
        self.assertEqual(state.status, "applied")
        rx1 = history.find_pending(state.optimization_history)
        self.assertEqual(rx1.metadata["before_report"].overall_score, 60.0)

        # 방문2 진입 전 Eval 이 Rx1 을 열화(40)로 측정 + 새 라벨의 finding 제시.
        # → Rx1 롤백(40<60) 후, 새 라벨 처방(Rx2)이 같은 방문에서 제안된다.
        state.report = make_report(40.0, label="retrieval_semantic_mismatch")
        state = agent.run(state)

        pending = history.find_pending(state.optimization_history)
        self.assertIsNotNone(pending, "롤백 후 다음 처방이 제안돼야 이 회귀를 검증할 수 있다")
        # 핵심: Rx2 의 baseline 은 복원된 60 이어야 한다 (열화값 40 이면 버그).
        self.assertEqual(pending.metadata["before_report"].overall_score, 60.0)

    def test_manual_stores_decision_report(self):
        state = agent.run(make_state(label="corpus_gap"))    # 수동 라벨
        report = state.optimization_report
        self.assertEqual(report.status, "manual_required")
        self.assertTrue(report.manual_actions)


class GraphRoutingTest(unittest.TestCase):
    @staticmethod
    def _pending():
        item = OptimizationHistoryItem(
            trial_id="t", request_id="r", iteration=1,
            failure_labels=["x"], optimizer="rules", status="applied",
        )
        item.metadata["pending"] = True
        return item

    @staticmethod
    def _route(fn, state):
        with redirect_stdout(StringIO()):
            return fn(state)

    def test_route_after_optimize(self):
        self.assertEqual(self._route(graph.route_after_optimize, AgentDoctorState(status="applied")), "index")
        self.assertEqual(self._route(graph.route_after_optimize, AgentDoctorState(status="rolled_back")), "index")
        self.assertEqual(self._route(graph.route_after_optimize, AgentDoctorState(status="skipped")), "serve")
        self.assertEqual(self._route(graph.route_after_optimize, AgentDoctorState(status="manual_required")), "serve")
        self.assertEqual(self._route(graph.route_after_optimize, AgentDoctorState(status="verified")), "serve")

    def test_route_after_eval_pass_goes_serve(self):
        state = AgentDoctorState(report=make_report(90.0, pass_threshold=True),
                                 iteration=1, max_iterations=3)
        self.assertEqual(self._route(graph.route_after_eval, state), "serve")

    def test_route_after_eval_budget_left_goes_optimize(self):
        state = AgentDoctorState(report=make_report(50.0), iteration=1, max_iterations=3)
        self.assertEqual(self._route(graph.route_after_eval, state), "optimize")

    def test_route_after_eval_exhausted_with_pending_goes_optimize(self):
        state = AgentDoctorState(report=make_report(50.0), iteration=3, max_iterations=3,
                                 optimization_history=[self._pending()])
        self.assertEqual(self._route(graph.route_after_eval, state), "optimize")

    def test_route_after_eval_exhausted_without_pending_goes_serve(self):
        state = AgentDoctorState(report=make_report(50.0), iteration=3, max_iterations=3,
                                 optimization_history=[])
        self.assertEqual(self._route(graph.route_after_eval, state), "serve")


if __name__ == "__main__":
    unittest.main()
