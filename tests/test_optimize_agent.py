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
        "sentence_transformers",
    ):
        sys.modules.setdefault(_n, _AnyModule(_n))

import graph
from core.schema import DiagnosticReport, Finding
from core.state import AgentDoctorState
from agents.optimize import agent, history
from agents.optimize.schemas import OptimizationHistoryItem


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


def make_state(overall=60.0, chunk_size=512, top_k=5, iteration=0, max_iterations=3,
               label="too_long_context"):
    return AgentDoctorState(
        report=make_report(overall, label=label),
        index_config={"chunk_size": chunk_size, "chunk_overlap": 50, "top_k": top_k},
        iteration=iteration, max_iterations=max_iterations,
    )


class OptimizeAgentForwardTest(unittest.TestCase):
    def test_apply_creates_pending_and_increments_iteration(self):
        state = agent.run(make_state())
        self.assertEqual(state.status, "applied")
        self.assertEqual(state.iteration, 1)
        self.assertEqual(state.current_agent, "optimize")
        # too_long_context 의 첫(가장 가벼운) 처방 decrease_top_k 가 실제로 적용된다.
        self.assertNotEqual(state.index_config["top_k"], 5)
        self.assertIsNotNone(history.find_pending(state.optimization_history))

    def test_manual_label_makes_no_change(self):
        state = make_state(label="corpus_gap")
        before = dict(state.index_config)
        state = agent.run(state)
        self.assertEqual(state.status, "manual_required")
        self.assertEqual(state.iteration, 0)  # 수동 경로는 iteration 미소비
        self.assertEqual(state.index_config, before)

    def test_always_returns_state_even_without_report(self):
        result = agent.run(AgentDoctorState(report=None, index_config={}, iteration=0))
        self.assertIsInstance(result, AgentDoctorState)


class OptimizeAgentRollbackTest(unittest.TestCase):
    def test_improved_keeps_config(self):
        # 방문2가 '판정만' 하도록 예산을 소진시킨다(예산이 남으면 새 처방을 또 적용해
        # top_k 가 다시 줄어들어 '유지됐는지'를 볼 수 없다).
        state = agent.run(make_state(overall=60.0, iteration=2, max_iterations=3))
        applied = state.index_config["top_k"]
        state.report = make_report(75.0)                    # Eval: 개선
        state = agent.run(state)                            # 방문2: 판정 → 유지
        self.assertEqual(state.index_config["top_k"], applied)  # 유지됨
        self.assertEqual(state.optimization_history[0].status, "applied")
        self.assertEqual(len(state.blacklist), 0)

    def test_worse_rolls_back_and_blacklists(self):
        # 예산이 남은 방문2는 '롤백 + 다음 처방 적용'을 한 번에 한다.
        state = agent.run(make_state(overall=60.0))         # 방문1: top_k 5 -> 2
        self.assertEqual(state.index_config["top_k"], 2)
        state.report = make_report(50.0)                    # Eval: 악화
        state = agent.run(state)                            # 방문2: 롤백 후 다음 처방

        # 롤백: top_k 가 되돌려지고, 실패 이력·블랙리스트가 남는다.
        self.assertEqual(state.index_config["top_k"], 5)
        self.assertEqual(state.optimization_history[0].status, "failed")
        self.assertIsNotNone(state.optimization_history[0].rollback_reason)
        self.assertIn(("too_long_context", "decrease_top_k"), state.blacklist)

        # 블랙리스트가 같은 처방 재시도를 막아 다음 후보로 넘어간다.
        self.assertEqual(state.status, "applied")
        self.assertEqual(
            state.optimization_history[1].selected_prescription_id, "shrink_chunk_size"
        )
        self.assertEqual(state.index_config["chunk_size"], 256)

    def test_budget_exhausted_judges_only(self):
        state = agent.run(make_state(overall=60.0, iteration=2, max_iterations=3))
        self.assertEqual(state.iteration, 3)                # 방문1: 2 -> 3
        self.assertEqual(state.index_config["top_k"], 2)    # 적용됨
        state.report = make_report(50.0)                    # 악화
        state = agent.run(state)                            # 방문2: 예산소진 → 판정만
        self.assertEqual(state.iteration, 3)                # iteration 증가 안 함
        self.assertEqual(state.index_config["top_k"], 5)    # 롤백까지 정상
        self.assertEqual(state.status, "rolled_back")


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
        state.report = make_report(75.0)                     # 개선
        state = agent.run(state)                             # 방문2: 예산소진 → 유지 판정만
        report = state.optimization_report
        self.assertEqual(report.status, "applied")
        self.assertIn("유지", report.summary)

    def test_rollback_stores_failed_trial_report(self):
        state = agent.run(make_state(overall=60.0, iteration=2, max_iterations=3))
        state.report = make_report(50.0)                     # 악화
        state = agent.run(state)                             # 방문2: 예산소진 → 판정만(롤백)
        report = state.optimization_report
        self.assertEqual(report.status, "failed")
        self.assertIn("되돌렸", report.summary)
        self.assertGreater(len(report.metadata.get("floor_violations", [])) +
                           int("점수" in report.summary), 0)  # 롤백 사유가 실림

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

    def test_route_after_optimize(self):
        self.assertEqual(graph.route_after_optimize(AgentDoctorState(status="applied")), "index")
        self.assertEqual(graph.route_after_optimize(AgentDoctorState(status="rolled_back")), "index")
        self.assertEqual(graph.route_after_optimize(AgentDoctorState(status="skipped")), "serve")
        self.assertEqual(graph.route_after_optimize(AgentDoctorState(status="manual_required")), "serve")
        self.assertEqual(graph.route_after_optimize(AgentDoctorState(status="verified")), "serve")

    def test_route_after_eval_pass_goes_serve(self):
        state = AgentDoctorState(report=make_report(90.0, pass_threshold=True),
                                 iteration=1, max_iterations=3)
        self.assertEqual(graph.route_after_eval(state), "serve")

    def test_route_after_eval_budget_left_goes_optimize(self):
        state = AgentDoctorState(report=make_report(50.0), iteration=1, max_iterations=3)
        self.assertEqual(graph.route_after_eval(state), "optimize")

    def test_route_after_eval_exhausted_with_pending_goes_optimize(self):
        state = AgentDoctorState(report=make_report(50.0), iteration=3, max_iterations=3,
                                 optimization_history=[self._pending()])
        self.assertEqual(graph.route_after_eval(state), "optimize")

    def test_route_after_eval_exhausted_without_pending_goes_serve(self):
        state = AgentDoctorState(report=make_report(50.0), iteration=3, max_iterations=3,
                                 optimization_history=[])
        self.assertEqual(graph.route_after_eval(state), "serve")


if __name__ == "__main__":
    unittest.main()
