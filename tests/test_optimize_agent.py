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
from agents.optimize import agent, history, rules
from agents.optimize.schemas import (
    ConfigPatch,
    OptimizationHistoryItem,
    OptimizationRequest,
    OptimizationResult,
    OptimizeDecision,
    SelectedAction,
    Verdict,
)


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


class OptimizeManualLogTest(unittest.TestCase):
    def test_manual_prescriptions_are_logged(self):
        """D그룹(manual) 결정이면 라벨 헤드라인 + 매뉴얼 스텝이 출력로그에 남는다."""
        from agents.optimize.schemas import OptimizeDecision
        dec = OptimizeDecision(
            mode="manual_required", status="manual_required",
            requires_user_confirmation=True, next_route="serve",
            reason="사람 개입 필요(D그룹)", manual_labels=["corpus_gap"],
        )
        buf = StringIO()
        with redirect_stdout(buf):
            agent._log_manual_prescriptions(dec)
        out = buf.getvalue()
        self.assertIn("수동 조치 필요: corpus_gap", out)
        self.assertIn("코퍼스에서 빠진 근거를 특정", out)  # 매뉴얼 스텝
        self.assertIn("재색인", out)


class OptimizeActionLogTest(unittest.TestCase):
    """로그가 "후보 목록 중 하나"가 아니라 **정해진 변경 하나**를 보고한다."""

    def test_request_skip_does_not_reassign_reason_to_the_skipped_entry(self):
        result = OptimizationResult(
            request_id="req-1",
            status="skipped",
            optimizer="rules",
            metadata={
                "error_code": "missing_search_space",
                "skipped_candidates": [
                    {
                        "action_key": "retriever.top_k:increase",
                        "prescription_id": "increase_top_k",
                        "reason": "no_valid_candidate_values",
                    },
                ],
            },
        )

        buf = StringIO()
        with redirect_stdout(buf):
            agent._log_action_review(result)

        self.assertEqual(
            buf.getvalue().splitlines(),
            [
                "[Optimize] action SKIP: retriever.top_k:increase, "
                "reason=no_valid_candidate_values",
                "[Optimize] 요청 SKIP: reason=missing_search_space",
            ],
        )

    def test_failed_action_is_not_logged_as_selected(self):
        result = OptimizationResult(
            request_id="req-1",
            status="failed",
            optimizer="internal",
            selected_action=SelectedAction(
                action_key="retriever.top_k:increase",
                prescription_id="increase_top_k",
            ),
            message="internal 평가에 실패했습니다.",
            error="internal 평가에 실패했습니다.",
            metadata={"error_code": "internal_failed"},
        )

        buf = StringIO()
        with redirect_stdout(buf):
            agent._log_action_review(result)

        self.assertEqual(
            buf.getvalue().strip(),
            "[Optimize] action FAIL: retriever.top_k:increase, reason=internal_failed",
        )
        self.assertNotIn("SELECT", buf.getvalue())

    def test_failed_request_without_action_is_logged(self):
        result = OptimizationResult(
            request_id="req-1",
            status="failed",
            optimizer="internal",
            message="후보 범위가 잘못됐습니다.",
            error="후보 범위가 잘못됐습니다.",
            metadata={"error_code": "invalid_internal_next_config"},
        )

        buf = StringIO()
        with redirect_stdout(buf):
            agent._log_action_review(result)

        self.assertEqual(
            buf.getvalue().strip(),
            "[Optimize] 요청 FAIL: reason=invalid_internal_next_config",
        )

    def test_selection_log_shows_the_action_and_what_it_beat(self):
        """선택 로그는 고른 변경과 **밀린 것들**을 함께 보여준다.

        전환 전에는 "후보 N개" 목록을 나열했는데, 그 목록은 이제 planner 안에서
        경쟁이 끝난 뒤라 존재하지 않는다. 대신 runner-up 과 보류 축을 남긴다.
        """
        request = OptimizationRequest(
            request_id="req-1",
            iteration=1,
            baseline_config={"top_k": 4},
            search_space={"retriever.top_k": [8]},
            supporting_labels=["too_long_context", "retrieval_missing_gold"],
            supporting_probes=["p1", "p2"],
            action_key="retriever.top_k:increase",
            prescription_id="increase_top_k",
            action_score=2.0,
            action_score_breakdown={
                "weighted_probe_support": 2.0,
                "base_cost": 1.0,
                "cost_source": "reindex_flag",
                "confidence_source": "default",
            },
            metadata={
                "runner_up_actions": [
                    {"action_key": "chunker.chunk_size:decrease", "score": 0.67}
                ],
                "deferred_axes": [
                    {"axis": "retriever.top_k", "reason": "conflict_margin_unmet"}
                ],
            },
        )

        buf = StringIO()
        with redirect_stdout(buf):
            agent._log_selected_action(request)

        output = buf.getvalue()
        self.assertIn("선택된 action: retriever.top_k:increase", output)
        self.assertIn("지지 라벨 2개", output)
        self.assertIn("probe 2개", output)
        self.assertIn("밀린 action: chunker.chunk_size:decrease", output)
        self.assertIn("보류된 축: retriever.top_k (conflict_margin_unmet)", output)

    def test_selected_reason_is_logged_once_at_application(self):
        request = OptimizationRequest(
            request_id="req-1",
            iteration=1,
            baseline_config={"top_k": 4},
            search_space={"retriever.top_k": [2]},
            supporting_labels=["too_long_context"],
            action_key="retriever.top_k:decrease",
            prescription_id="decrease_top_k",
        )
        result = OptimizationResult(
            request_id="req-1",
            status="applied",
            optimizer="rules",
            selected_action=SelectedAction(
                action_key="retriever.top_k:decrease",
                prescription_id="decrease_top_k",
            ),
            message="top_k를 줄입니다.",
        )
        state = AgentDoctorState(
            index_config={"top_k": 2},
            iteration=1,
            max_iterations=3,
        )

        buf = StringIO()
        with redirect_stdout(buf):
            agent._log_action_review(result)
            agent._log_optimize_application(
                state,
                request,
                result,
                {"top_k": 4},
                {"top_k": 2},
                ["top_k"],
                "decrease_top_k",
            )

        output = buf.getvalue()
        self.assertIn("[Optimize] action SELECT: retriever.top_k:decrease", output)
        self.assertEqual(output.count("top_k를 줄입니다."), 1)


def make_state(overall=60.0, chunk_size=512, iteration=0, max_iterations=3,
               label="too_long_context"):
    return AgentDoctorState(
        report=make_report(overall, label=label),
        index_config={"top_k": 4, "chunk_size": chunk_size, "chunk_overlap": 50},
        iteration=iteration, max_iterations=max_iterations,
    )


class OptimizeAgentForwardTest(unittest.TestCase):
    def test_unjudgeable_exclusions_only_returns_recorded_failures(self):
        """측정 불가는 action key 단위로만 막는다.

        정확한 전이(attempt)가 아니라 축 전체를 막는 이유는, 못 잰 것이 그 후보값의
        문제가 아니라 이 방문의 문제라서다 — 같은 축의 다른 값으로 바꿔도 똑같이
        못 잰다. 대신 품질 blacklist 와 달리 방문 예산으로만 제한한다.
        """
        unjudgeable = OptimizationHistoryItem(
            trial_id="u1",
            request_id="r1",
            iteration=1,
            failure_labels=["retrieval_low_rank"],
            optimizer="rules",
            status="failed",
            selected_prescription_id="enable_reranker",
            action_key="reranker.enabled:enable",
            metadata={"unjudgeable": True},
        )
        judged = OptimizationHistoryItem(
            trial_id="u2",
            request_id="r2",
            iteration=1,
            failure_labels=["retrieval_missing_gold"],
            optimizer="rules",
            status="failed",
            selected_prescription_id="increase_top_k",
            action_key="retriever.top_k:increase",
            metadata={"unjudgeable": False},
        )

        self.assertEqual(
            agent._unjudgeable_exclusions([unjudgeable, judged]),
            {"reranker.enabled:enable"},
        )

    def test_unjudgeable_exclusions_ignore_legacy_items_without_action(self):
        """구버전 이력(action_key 없음)은 실행 제어에 쓰이지 않는다."""
        legacy = OptimizationHistoryItem(
            trial_id="u1",
            request_id="r1",
            iteration=1,
            failure_labels=["retrieval_low_rank"],
            optimizer="rules",
            status="failed",
            selected_prescription_id="enable_reranker",
            metadata={"unjudgeable": True},
        )

        self.assertEqual(agent._unjudgeable_exclusions([legacy]), set())

    def test_absolute_visit_limit_stops_before_new_prescription(self):
        state = make_state()
        state.optimize_visit_count = 19
        before = dict(state.index_config)

        state = agent.run(state)

        self.assertEqual(state.optimize_visit_count, 20)
        self.assertEqual(state.status, "verified")
        self.assertEqual(state.index_config, before)
        self.assertEqual(state.optimization_history, [])
        self.assertEqual(state.optimization_report.status, "skipped")
        self.assertIn("절대 방문 상한", state.optimization_report.summary)

    def test_apply_creates_pending_and_increments_iteration(self):
        """too_long_context 가 지지하는 action 중 하나가 적용되고 예산 1회를 쓴다.

        셋 다 같은 C tier 이고 probe 도 하나뿐이라 점수가 동률이다. 재색인이 필요한
        chunker.chunk_size:decrease 는 비용 3 으로 밀리고, 남은 둘은 action_key
        사전순으로 갈린다(context.compression < retriever.top_k). rules.py 의 선언
        순서('가벼운 것 먼저')가 실행 순서를 소유하지 않게 된 결과다 — 최종 판정은
        Eval 실측이 하고, 밀린 후보도 예산 안에서 결국 시도된다(구현계획 §8.1).
        """
        state = agent.run(make_state())
        self.assertEqual(state.status, "applied")
        self.assertEqual(state.iteration, 1)
        self.assertEqual(state.current_agent, "optimize")
        self.assertTrue(state.index_config["context_compression"])
        self.assertEqual(state.index_config["top_k"], 4)   # 다른 축은 건드리지 않는다
        self.assertFalse(state.reindex_required)
        pending = history.find_pending(state.optimization_history)
        self.assertIsNotNone(pending)
        self.assertEqual(pending.action_key, "context.compression.enabled:enable")
        self.assertEqual(pending.supporting_labels, ["too_long_context"])
        self.assertIsNotNone(pending.action_attempt_key)
        self.assertIsNotNone(pending.action_study_key)

    def test_manual_label_makes_no_change(self):
        state = make_state(label="corpus_gap")
        before = dict(state.index_config)
        buf = StringIO()
        with redirect_stdout(buf):
            state = agent.run(state)
        self.assertEqual(state.status, "manual_required")
        self.assertEqual(state.iteration, 0)  # 수동 경로는 iteration 미소비
        self.assertEqual(state.index_config, before)
        out = buf.getvalue()
        self.assertIn("[Optimize] 반복 횟수: 0/3", out)
        self.assertIn("다음 단계: Serve 이동 (manual_required)", out)
        self.assertNotIn("reindex_required=", out)

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
        self.assertIn("[Optimize] 선택된 action:", out)
        self.assertIn("[Optimize] action SELECT:", out)
        self.assertIn("[Optimize] action 적용:", out)
        self.assertIn("다음 단계: Index 경유(물리 재색인 생략) 후 Eval 재실행", out)
        self.assertEqual(graph.route_after_optimize(state), "index")

    def test_apply_log_includes_added_config_keys(self):
        state = make_state(overall=0.42)
        request, decision = agent.planner.plan(state)
        result = OptimizationResult(
            request_id=request.request_id,
            status="proposed",
            optimizer="rules",
            selected_action=SelectedAction(action_key=request.action_key),
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
                    selected_action=SelectedAction(
                        action_key=request.action_key,
                        prescription_id=request.prescription_id,
                    ),
                    metadata={"error_code": "baseline_selected"},
                )
            return real_optimizer_run(request)

        buf = StringIO()
        with patch(
            "agents.optimize.agent.optimizer.run",
            side_effect=select_baseline_once,
        ), redirect_stdout(buf):
            result_state = agent.run(state)

        self.assertEqual(len(calls), 2)
        self.assertIn(
            "[Optimize] action SKIP: chunker.chunk_overlap:increase, "
            "reason=baseline_selected",
            buf.getvalue(),
        )
        self.assertIn(
            ("chunking_context_mismatch", "increase_chunk_overlap"),
            result_state.blacklist,
        )
        self.assertEqual(result_state.index_config["chunk_overlap"], 50)
        self.assertEqual(result_state.index_config["chunk_size"], 800)
        self.assertEqual(result_state.status, "applied")
        self.assertEqual(result_state.iteration, 1)

    def test_retrieval_low_rank_enables_reranker(self):
        """retrieval_low_rank의 최우선 처방이 실제 runtime config에 반영된다."""
        def _finding(pid, label):
            return Finding(
                finding_id=f"{pid}:{label}", type="retrieval_failure",
                severity="warning", description=label, label=label,
                confirmed=True, affected_probes=[pid],
            )
        # low_rank를 더 흔하게 만들어 최우선 처방으로 선택한다.
        findings = (
            [_finding(f"lr{i}", "retrieval_low_rank") for i in range(6)]
            + [_finding(f"sm{i}", "retrieval_semantic_mismatch") for i in range(3)]
        )
        state = AgentDoctorState(
            report=DiagnosticReport(
                report_id="r", findings=findings, overall_score=30.0,
                ragas_scores={"context_recall": 0.4}, pass_threshold=False,
            ),
            index_config={
                "top_k": 5,
                "chunk_size": 512,
                "chunk_overlap": 50,
                "use_reranker": False,
                "reranker_model": "BAAI/bge-reranker-v2-m3",
                "rerank_candidates": 20,
            },
            iteration=0, max_iterations=3,
            runtime_capabilities={
                "reranker": {
                    "status": "verified",
                    "model": "BAAI/bge-reranker-v2-m3",
                    "retryable": False,
                    "reason": None,
                }
            },
        )
        out = agent.run(state)
        self.assertEqual(out.status, "applied")
        self.assertTrue(out.index_config["use_reranker"])
        self.assertFalse(out.reindex_required)
        self.assertNotIn(("retrieval_low_rank", "enable_reranker"), out.blacklist)
        self.assertEqual(len(out.optimization_history), 1)
        self.assertEqual(
            out.optimization_history[-1].failure_labels,
            ["retrieval_low_rank"],
        )

    def _reranker_on_state(self, label):
        state = make_state(overall=30.0, label=label)
        state.index_config.update(
            {
                "use_reranker": True,
                "reranker_model": "BAAI/bge-reranker-v2-m3",
                "rerank_candidates": 20,
            }
        )
        state.runtime_capabilities = {
            "reranker": {
                "status": "verified",
                "model": "BAAI/bge-reranker-v2-m3",
                "retryable": False,
                "reason": None,
            }
        }
        return state

    def test_candidate_miss_widens_candidate_count(self):
        """후보창 밖 gold(candidate_miss)는 후보 수를 실제 config에 반영한다."""
        state = self._reranker_on_state("retrieval_rerank_candidate_miss")

        out = agent.run(state)

        self.assertEqual(out.status, "applied")
        self.assertEqual(out.index_config["rerank_candidates"], 40)
        self.assertFalse(out.reindex_required)
        self.assertEqual(
            out.optimization_history[-1].selected_prescription_id,
            "widen_rerank_candidates",
        )

    def _rank_cause_state(self, label, metadata=None, config=None):
        finding = Finding(
            finding_id=f"p1:{label}", type="retrieval_failure", severity="warning",
            description=label, label=label, confirmed=True,
            affected_probes=["p1"], metadata=dict(metadata or {}),
        )
        index_config = {
            "top_k": 5, "chunk_size": 512, "chunk_overlap": 50,
            "use_hybrid": True, "hybrid_dense_weight": 0.7,
            "use_reranker": False, "rerank_candidates": 20,
            "rerank_candidate_policy": {"max_candidates": 50},
        }
        index_config.update(config or {})
        state = make_state(overall=30.0, label=label)
        state.report.findings = [finding]
        state.index_config = index_config
        state.runtime_capabilities = {
            "reranker": {
                "status": "verified",
                "model": "BAAI/bge-reranker-v2-m3",
                "retryable": False,
                "reason": None,
            }
        }
        return state

    def test_rank_cause_prescriptions_reach_index_config(self):
        """순위 원인 4형제의 처방이 실제 config 변경까지 도달하는지 end-to-end 고정.

        rules.py 에 처방을 적어도 optimizer 의 경로 레지스트리
        (STATE_MAPPABLE_PATHS / BACKEND_SUPPORTED_PATHS / PATH_CAPABILITIES)에 빠져 있으면
        unsupported_backend_path 로 조용히 건너뛰고 다음 후보가 대신 적용된다. 라벨별
        '무엇이 바뀌어야 하는가'를 여기서 못박아 그 누락을 드러낸다.
        """
        cases = [
            # (라벨, finding metadata, 시작 config, 기대 처방, 기대 config 변화)
            ("retrieval_low_rank", {}, {},
             "enable_reranker", ("use_reranker", True)),
            ("retrieval_rank_fusion_loss", {"favored_channel": "lexical"}, {},
             "rebalance_hybrid_weight", ("hybrid_dense_weight", 0.6)),
            ("retrieval_rank_fusion_loss", {"favored_channel": "dense"}, {},
             "rebalance_hybrid_weight", ("hybrid_dense_weight", 0.8)),
            ("retrieval_rerank_candidate_miss", {"gold_ranks": {"g": 34}},
             {"use_reranker": True},
             "widen_rerank_candidates", ("rerank_candidates", 34)),
            ("retrieval_reranker_demotion", {"pre_rerank_ranks": {"g": 12}},
             {"use_reranker": True},
             "disable_reranker", ("use_reranker", False)),
        ]
        for label, metadata, config, expected_id, (key, value) in cases:
            with self.subTest(label=label, metadata=metadata):
                out = agent.run(self._rank_cause_state(label, metadata, config))

                self.assertEqual(out.status, "applied")
                self.assertEqual(
                    out.optimization_history[-1].selected_prescription_id,
                    expected_id,
                )
                self.assertEqual(out.index_config[key], value)

    def test_candidate_widening_never_exceeds_policy_ceiling(self):
        """정책 상한은 근거값 계산뿐 아니라 방향 폴백(현재값×2)에도 걸려야 한다.

        근거값이 없으면 폴백이 30×2=60 을 내는데, 이를 거르는 게 optimizer 의 정적 제약
        (max=100)뿐이면 정책 상한 50 을 넘는 값이 실제 config 에 박힌다.
        """
        state = self._rank_cause_state(
            "retrieval_rerank_candidate_miss",
            {},                                   # gold_ranks 없음 → 방향 폴백 경로
            {"use_reranker": True, "rerank_candidates": 30,
             "rerank_candidate_policy": {"max_candidates": 50}},
        )

        out = agent.run(state)

        self.assertLessEqual(out.index_config["rerank_candidates"], 50)

    def test_duplicate_crowding_prescribes_mmr(self):
        """중복 밀림의 레버는 MMR 다. 리랭커가 아니다.

        예전엔 "config 에 레버가 없다"며 미뤘는데, PR #51 이 Retriever 에 MMR 을 구현하고
        core/state.py 에 use_mmr 을 넣으면서 그 전제가 사라졌다(같은 enable_mmr 처방이
        retrieval_incomplete_enumeration·context_noise_interference 에서는 이미 실행 중).

        리랭커를 처방하면 안 된다는 조건도 함께 고정한다 — cross-encoder 는 중복 청크를
        상위에 그대로 둔다(각각이 질문과 실제로 관련 있어 점수가 높다). 처방했다면 실패 후
        blacklist 만 쌓인다.
        """
        state = self._rank_cause_state(
            "retrieval_duplicate_crowding",
            {"crowding_analysis": {"g": {"rank": 4, "redundant": 2,
                                         "projected_rank": 2}}},
        )

        out = agent.run(state)

        self.assertTrue(rules.is_actionable("retrieval_duplicate_crowding"))
        self.assertTrue(out.index_config["use_mmr"])
        self.assertFalse(out.index_config["use_reranker"])   # 리랭커는 이 라벨의 레버가 아니다
        self.assertNotEqual(out.status, "skipped")

    def test_low_rank_does_not_widen_candidate_count(self):
        """low_rank 는 'gold 가 후보창 안'이라는 신호라 창 확대가 처방이 아니다.

        예전에는 같은 라벨에 enable_reranker → widen_rerank_candidates 가 순차로 달려 있어,
        리랭커가 이미 켜진 상태에서 신호 없이 창을 넓혔다. 창을 넓혀야 하는 케이스는
        retrieval_rerank_candidate_miss 로 분리됐다.
        """
        state = self._reranker_on_state("retrieval_low_rank")

        out = agent.run(state)

        self.assertEqual(out.index_config["rerank_candidates"], 20)   # 그대로
        self.assertNotEqual(
            getattr(out.optimization_history[-1], "selected_prescription_id", None)
            if out.optimization_history else None,
            "widen_rerank_candidates",
        )

    def test_inapplicable_prescription_falls_through_to_next(self):
        """issue #26: 적용 불가 action 이 최다 지지를 받아도 다음 action 이 적용된다.

        lexical_mismatch(12 probe)가 최다 지지지만 baseline 이 이미 hybrid 라
        retriever.search_type:replace 는 유효 후보가 없다(no-op). 전환 전에는 그
        처방이 선택된 **뒤** optimizer 에서 탈락해 blacklist 에 올랐다 — 그 방문이
        통째로 낭비됐다. 이제는 **점수 경쟁 전** eligibility 에서 제외되므로
        blacklist 에 오르지 않고, 같은 방문에서 곧바로 missing_gold 의 top_k 확대가
        적용된다(구현계획 §4.6 / 안전 원칙 7).
        """
        def _finding(pid, label, **metadata):
            return Finding(
                finding_id=f"{pid}:{label}",
                type="retrieval_failure",
                severity="warning",
                description=label,
                label=label,
                confirmed=True,
                affected_probes=[pid],
                metadata=metadata,
            )

        findings = (
            [
                _finding(
                    f"lm{i}",
                    "retrieval_lexical_mismatch",
                )
                for i in range(12)
            ]
            + [
                _finding(f"mg{i}", "retrieval_missing_gold")
                for i in range(3)
            ]
        )
        state = AgentDoctorState(
            report=DiagnosticReport(
                report_id="r",
                findings=findings,
                overall_score=30.0,
                ragas_scores={"context_recall": 0.4},
                pass_threshold=False,
            ),
            index_config={
                "top_k": 5,
                "chunk_size": 512,
                "chunk_overlap": 50,
                "embedding_model": "BAAI/bge-m3",
                "use_hybrid": True,   # 이미 hybrid → enable_hybrid는 no-op(적용 불가)
            },
            iteration=0,
            max_iterations=3,
        )

        request, _decision = agent.planner.plan(state)
        rejected = {
            entry["action_key"]: entry["reason"]
            for entry in request.metadata["rejected_actions"]
        }
        self.assertEqual(
            rejected.get("retriever.search_type:replace"),
            "no_candidate_value",
        )

        out = agent.run(state)

        self.assertEqual(out.status, "applied")
        self.assertEqual(out.index_config["top_k"], 10)
        self.assertTrue(out.index_config["use_hybrid"])   # no-op 축은 손대지 않는다
        self.assertEqual(
            out.optimization_history[-1].action_key,
            "retriever.top_k:increase",
        )
        # 실행 전에 걸러졌으므로 품질 실패로 기록되지 않는다.
        self.assertEqual(
            [key.action_key for key in out.blocked_action_attempts],
            [],
        )

    def test_always_returns_state_even_without_report(self):
        result = agent.run(AgentDoctorState(report=None, index_config={}, iteration=0))
        self.assertIsInstance(result, AgentDoctorState)


class OptimizeCGroupUnblockTest(unittest.TestCase):
    """C그룹 언블록: lost_in_the_middle / context_noise_interference 가 실행 가능한
    처방(top_k 축소 / MMR)을 맨 앞 후보로 내는지 고정. 나머지(정렬·필터 프롬프트)는
    소비 경로가 없어 뒤쪽 fallback 으로만 남고 optimizer 가 걸러낸다."""

    def _first_candidate_space(self, label):
        from agents.optimize.planner import plan
        state = AgentDoctorState(
            report=DiagnosticReport(
                report_id="r",
                findings=[Finding(
                    finding_id="f1", type="context_failure", severity="critical",
                    description="c", label=label, affected_probes=["p1"],
                )],
                composite_score={"total": 40.0},
            ),
        )
        request, decision = plan(state)
        self.assertEqual(decision.mode, "apply_optimize")
        return request.search_space

    def test_lost_in_the_middle_leads_with_top_k(self):
        space = self._first_candidate_space("lost_in_the_middle")
        self.assertIn("retriever.top_k", space)

    def test_context_noise_interference_leads_with_context_compression(self):
        space = self._first_candidate_space("context_noise_interference")
        self.assertEqual(space, {"context.compression.enabled": [True]})


class OptimizeAgentRollbackTest(unittest.TestCase):
    def test_improved_keeps_config(self):
        state = agent.run(make_state(overall=60.0))         # 방문1: 적용
        applied = state.index_config["chunk_size"]
        state.report = make_report(75.0)                    # Eval: 개선
        state = agent.run(state)                            # 방문2: 판정 → 유지
        self.assertEqual(state.index_config["chunk_size"], applied)  # 유지됨
        self.assertEqual(state.optimization_history[0].status, "applied")
        self.assertEqual(len(state.blacklist), 0)

    def test_worse_rolls_back_and_blocks_the_exact_transition(self):
        state = agent.run(make_state(overall=60.0))         # 방문1: 압축 켜기
        self.assertTrue(state.index_config["context_compression"])
        state.report = make_report(50.0)                    # Eval: 악화
        state = agent.run(state)                            # 방문2: 판정 → 롤백
        self.assertEqual(state.status, "applied")           # 다음 action 은 계속 진행
        # 첫 시도는 baseline 으로 복원됐다(적용 전 config 에 없던 키라 사라진다).
        self.assertNotIn("context_compression", state.index_config)
        self.assertEqual(state.index_config["top_k"], 2)
        self.assertEqual(
            state.optimization_history[-1].action_key,
            "retriever.top_k:decrease",
        )
        # 차단 단위는 (action, baseline, 후보값) 전이 하나다.
        blocked = list(state.blocked_action_attempts)
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0].action_key, "context.compression.enabled:enable")
        self.assertEqual(state.optimization_history[0].status, "failed")
        self.assertIsNotNone(state.optimization_history[0].rollback_reason)

    def test_visit_excluded_request_is_not_sent_to_optimizer(self):
        """planner 가 제외된 request 를 돌려줘도 agent 가 optimizer 직전에 막는다."""
        state = make_state(overall=60.0)
        blocked_request = OptimizationRequest(
            request_id="blocked-request",
            iteration=state.iteration,
            baseline_config=dict(state.index_config),
            search_space={"context.compression.enabled": [True]},
            action_key="context.compression.enabled:enable",
            supporting_labels=["too_long_context"],
            supporting_probes=["p1"],
            prescription_id="context_compression",
        )
        state.optimization_history.append(
            OptimizationHistoryItem(
                trial_id="unjudgeable-1",
                request_id="req-unjudgeable-1",
                iteration=state.iteration,
                failure_labels=["too_long_context"],
                optimizer="rules",
                status="skipped",
                selected_prescription_id="context_compression",
                action_key=blocked_request.action_key,
                supporting_labels=["too_long_context"],
                supporting_probes=["p1"],
                metadata={"unjudgeable": True},
            )
        )
        apply_decision = OptimizeDecision(
            mode="apply_optimize",
            status="proposed",
            requires_user_confirmation=False,
            next_route="index",
        )
        with patch.object(
            agent.planner,
            "plan",
            return_value=(blocked_request, apply_decision),
        ) as plan, patch.object(agent.optimizer, "run") as run_optimizer:
            out = agent.run(state)

        self.assertEqual(plan.call_count, 1)
        run_optimizer.assert_not_called()
        self.assertEqual(out.status, "skipped")

    def test_blocked_attempt_is_released_on_a_new_baseline(self):
        """같은 action 이라도 baseline 이 달라지면 다시 시도할 수 있다(구현계획 §5.1).

        기존 (label, prescription_id) blacklist 는 baseline 무관 영구 차단이라
        "검색을 고친 뒤 하류를 다시 본다"는 A>C>B 설계 의도와 충돌했다.
        """
        state = agent.run(make_state(overall=60.0))
        state.report = make_report(50.0)
        state = agent.run(state)                             # 롤백 → 전이 차단
        blocked = set(state.blocked_action_attempts)
        self.assertEqual(
            {key.action_key for key in blocked},
            {"context.compression.enabled:enable"},
        )

        # 차단이 걸린 그 baseline(=시도 직전 config)에서는 다시 뽑히지 않는다.
        same_baseline = make_state(overall=60.0)
        request, _decision = agent.planner.plan(same_baseline, blacklist=blocked)
        self.assertNotEqual(request.action_key, "context.compression.enabled:enable")

        # 다른 축이 baseline 을 바꾸면 같은 action 이 후보로 돌아온다.
        moved = make_state(overall=60.0)
        moved.index_config["chunk_size"] = 256
        request, _decision = agent.planner.plan(moved, blacklist=blocked)
        self.assertEqual(request.action_key, "context.compression.enabled:enable")

    def _reranker_baseline(self, rerank_candidates=20):
        return {
            "top_k": 5,
            "chunk_size": 512,
            "chunk_overlap": 50,
            "use_reranker": True,
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "rerank_candidates": rerank_candidates,
        }

    def _disable_reranker_rollback(self, trial_id, iteration, before_config, caps):
        item = OptimizationHistoryItem(
            trial_id=trial_id,
            request_id=f"req-{trial_id}",
            iteration=iteration,
            failure_labels=["retrieval_reranker_demotion"],
            optimizer="rules",
            status="failed",
            selected_prescription_id="disable_reranker",
            before_config=before_config,
            after_config={**before_config, "use_reranker": False},
            rollback_reason="종합점수 미상승 0.780→0.750 → 롤백",
            action_key="reranker.enabled:disable",
            supporting_labels=["retrieval_reranker_demotion"],
            supporting_probes=["p1"],
            metadata={
                # ⚠️ 이 파일의 점수 규약은 make_report 가 쓰는 0~100 이다(overall_score
                # 를 그대로 _read_score 가 돌려준다). 이력 점수만 0~1 로 적으면 리포트
                # 기준값과 스케일이 어긋나, 도착 config 기억이 "측정한 모든 config 가
                # 현재보다 한참 나쁘다"고 오판해 멀쩡한 재시도까지 막는다.
                "pending": False,
                "before_score": 78.0,
                "after_score": 75.0,
                "unjudgeable": False,
            },
        )
        item.action_attempt_key = history.build_attempt_key(
            item.action_key,
            before_config,
            {"reranker.enabled": False},
            caps,
        )
        return item

    def _reranker_state(self, rerank_candidates):
        return AgentDoctorState(
            report=make_report(73.0, label="retrieval_reranker_demotion"),
            index_config=self._reranker_baseline(rerank_candidates),
            iteration=2,
            max_iterations=5,
            runtime_capabilities={
                "reranker": {
                    "status": "verified",
                    "model": "BAAI/bge-reranker-v2-m3",
                    "retryable": False,
                    "reason": None,
                }
            },
        )

    def test_single_rollback_still_retries_on_a_moved_baseline(self):
        """한 번 롤백했다고 축을 닫지 않는다 — 옮겨간 baseline 에서 한 번은 더 본다(§5.1)."""
        state = self._reranker_state(22)
        state.optimization_history.append(
            self._disable_reranker_rollback(
                "disable-1", 1, self._reranker_baseline(20), state.runtime_capabilities
            )
        )
        state.blocked_action_attempts.add(
            state.optimization_history[0].action_attempt_key
        )

        self.assertEqual(
            agent._rollback_action_cooldown_exclusions(state.optimization_history),
            set(),
        )
        buf = StringIO()
        with redirect_stdout(buf):
            out = agent.run(state)
        self.assertIn("선택한 action: reranker.enabled:disable", buf.getvalue())
        self.assertFalse(out.index_config["use_reranker"])

    def test_confirmed_rollback_action_enters_run_cooldown_after_second_rollback(self):
        """두 번째 롤백부터는 baseline 이 또 움직여도 같은 action 을 다시 고르지 않는다.

        실제 로그에서 reranker.enabled:disable 이 롤백된 뒤 rerank_candidates 만
        20→22로 바뀌자 같은 action 이 곧바로 다시 선택되어 한 번 더 롤백됐다. exact
        attempt 차단의 완화(§5.1)는 그대로 두고, 확인된 롤백이 쌓인 뒤에만 닫는다.
        """
        state = self._reranker_state(24)
        for index, candidates in enumerate((20, 22), start=1):
            item = self._disable_reranker_rollback(
                f"disable-{index}",
                index,
                self._reranker_baseline(candidates),
                state.runtime_capabilities,
            )
            state.optimization_history.append(item)
            state.blocked_action_attempts.add(item.action_attempt_key)

        # exact attempt 차단만 넘기면 baseline 이 달라져 disable 이 다시 열린다.
        request, _decision = agent.planner.plan(
            state,
            blacklist=set(state.blocked_action_attempts),
        )
        self.assertEqual(request.action_key, "reranker.enabled:disable")

        buf = StringIO()
        with redirect_stdout(buf):
            out = agent.run(state)

        self.assertEqual(out.status, "skipped")
        self.assertTrue(out.index_config["use_reranker"])
        self.assertEqual(out.index_config["rerank_candidates"], 24)
        self.assertEqual(len(out.optimization_history), 2)
        self.assertIn("제외된 action: [reranker.enabled:disable]", buf.getvalue())
        self.assertNotIn("선택한 action: reranker.enabled:disable", buf.getvalue())

    def test_rollback_cooldown_ignores_unjudgeable_rollbacks(self):
        item = OptimizationHistoryItem(
            trial_id="u1",
            request_id="r1",
            iteration=1,
            failure_labels=["retrieval_low_rank"],
            optimizer="rules",
            status="failed",
            selected_prescription_id="enable_reranker",
            rollback_reason="판정 불가 — 롤백",
            action_key="reranker.enabled:enable",
            metadata={"unjudgeable": True},
        )

        self.assertEqual(agent._rollback_action_cooldown_exclusions([item, item]), set())

    def test_rollback_cooldown_ignores_margin_rejected_rollbacks(self):
        """마진 미달 롤백은 점수가 오른 시도다 — 그 축이 나쁘다는 증거가 아니다."""
        item = OptimizationHistoryItem(
            trial_id="m1",
            request_id="r1",
            iteration=1,
            failure_labels=["retrieval_low_rank"],
            optimizer="rules",
            status="failed",
            selected_prescription_id="increase_top_k",
            rollback_reason="종합점수 상승폭 부족 0.780→0.782 → 롤백",
            action_key="retriever.top_k:increase",
            metadata={"unjudgeable": False, "margin_rejected": True},
        )

        self.assertEqual(agent._rollback_action_cooldown_exclusions([item, item]), set())

    def test_excluded_action_log_survives_a_rollback_in_the_same_visit(self):
        """롤백 방문에서도 로그가 살아 있는 차단을 빠뜨리지 않는다.

        롤백은 index_config 를 복원한 뒤 planner 를 부른다. 제외 목록을 그 전에
        찍으면 복원될 baseline 에 걸린 차단이 로그에서 사라져, 이번 방문에 어떤
        action 이 닫혀 있었는지 사후에 읽을 수 없다.
        """
        baseline = self._reranker_baseline(20)
        state = self._reranker_state(20)
        state.index_config = {**baseline, "use_reranker": False}  # 롤백 전(적용 상태)
        state.report = make_report(40.0, label="retrieval_reranker_demotion")
        pending = OptimizationHistoryItem(
            trial_id="p1",
            request_id="req-p1",
            iteration=1,
            failure_labels=["retrieval_reranker_demotion"],
            optimizer="rules",
            status="pending",
            selected_prescription_id="disable_reranker",
            before_config=baseline,
            after_config={},
            action_key="reranker.enabled:disable",
            metadata={"pending": True, "before_report": make_report(75.0)},
        )
        state.optimization_history.append(pending)
        # 복원될 baseline 에서만 걸려 있는 다른 축의 차단
        state.blocked_action_attempts.add(
            history.build_attempt_key(
                "retriever.top_k:increase",
                baseline,
                {"retriever.top_k": 7},
                state.runtime_capabilities,
            )
        )

        buf = StringIO()
        with redirect_stdout(buf):
            out = agent.run(state)

        self.assertEqual(out.optimization_history[0].status, "failed")  # 롤백 완료
        self.assertIn("제외된 action: [retriever.top_k:increase]", buf.getvalue())

    def test_rollback_then_followup_application_logs_both_prescriptions(self):
        # 검증 대상은 "한 방문에 롤백과 새 적용이 **둘 다** 로그에 남는가"다.
        #
        # ⚠️ 후속 처방이 무엇인지는 바뀌었다. 롤백이 config 와 함께 진단서도 되돌리게
        # 되면서(구멍 3), 후속은 여기서 손으로 넣은 열화 리포트(retrieval_semantic_
        # mismatch)가 아니라 **복원된 리포트**(방문1 의 too_long_context)에서 나온다.
        # 열화 리포트는 롤백으로 사라진 config 를 측정한 것이라 복원된 config 의 근거가
        # 못 된다 — Eval 도 다음 방문에 롤백 진단 캐시로 같은 리포트를 돌려준다.
        # 그래서 too_long_context 의 남은 처방(decrease_top_k)이 뽑힌다.
        state = agent.run(make_state(overall=60.0))
        state.report = make_report(40.0, label="retrieval_semantic_mismatch")

        buf = StringIO()
        with redirect_stdout(buf):
            state = agent.run(state)

        out = buf.getvalue()
        self.assertEqual(state.optimization_history[0].status, "failed")
        self.assertEqual(
            state.optimization_history[-1].action_key,
            "retriever.top_k:decrease",
        )
        self.assertIn("context.compression.enabled:enable", out)   # 롤백된 action
        self.assertIn("retriever.top_k:decrease", out)             # 새로 적용된 action
        self.assertIn("판정 결과: keep=false, before=60.00, after=40.00", out)
        verdict_block = out[:out.index("판정 결과: keep=false")]
        self.assertIn("Eval 결과:", verdict_block)
        self.assertIn("발견된 문제:", verdict_block)
        self.assertIn("이전 처방 판정: ROLLBACK", verdict_block)
        self.assertNotIn("reindex_required=", verdict_block)
        self.assertNotIn("다음 단계:", verdict_block)
        self.assertLess(
            out.index("이전 처방 판정: ROLLBACK"),
            out.index("action 적용: retriever.top_k:decrease"),
        )

    def test_keep_then_followup_application_logs_previous_verdict(self):
        state = agent.run(make_state(overall=60.0))
        state.report = make_report(75.0, label="retrieval_semantic_mismatch")

        buf = StringIO()
        with redirect_stdout(buf):
            state = agent.run(state)

        out = buf.getvalue()
        self.assertEqual(state.optimization_history[0].status, "applied")
        self.assertEqual(
            state.optimization_history[-1].action_key,
            "chunker.chunk_size:decrease",
        )
        self.assertIn(
            "선택한 action: context.compression.enabled:enable "
            "(처방 context_compression)",
            out,
        )
        self.assertIn("판정 결과: keep=true, before=60.00, after=75.00", out)
        self.assertIn(
            "선택한 action: chunker.chunk_size:decrease (처방 shrink_chunk_size)",
            out,
        )
        verdict_block = out[:out.index("판정 결과: keep=true")]
        self.assertIn("Eval 결과:", verdict_block)
        self.assertIn("발견된 문제:", verdict_block)
        self.assertIn("이전 처방 판정: KEEP", verdict_block)
        self.assertNotIn("다음 단계:", verdict_block)
        self.assertLess(
            out.index("이전 처방 판정: KEEP"),
            out.index("action 적용: chunker.chunk_size:decrease"),
        )

    def test_unjudgeable_rollback_does_not_blacklist(self):
        # 측정이 없어(before_report None) 판정 불가한 경우: config 는 안전하게 복원하되,
        # '나빴다는 증거'가 아니므로 블랙리스트엔 넣지 않는다(리뷰 #36). 같은 시나리오라도
        # 실제 판정으로 악화가 확인되면 블랙리스트에 들어가는 test_worse_rolls_back 과 대비.
        state = agent.run(make_state(overall=60.0))          # 방문1: 처방 적용 → pending
        self.assertEqual(len(state.blacklist), 0)
        pending = history.find_pending(state.optimization_history)
        pending.metadata["before_report"] = None             # 측정 없음 상황 유발
        state.report = make_report(50.0)
        state = agent.run(state)                             # 방문2: 판정 불가 → 롤백(차단 X)
        self.assertEqual(len(state.blacklist), 0)             # 처방이 소진/차단되지 않음
        self.assertEqual(state.optimization_history[0].status, "failed")
        self.assertTrue(state.optimization_history[0].metadata["unjudgeable"])

    def test_rollback_reindex_survives_followup_search_time_rx(self):
        # index-time 처방(shrink_chunk_size, reindex=True)이 롤백된 뒤 같은 방문에서
        # 검색시점 처방(dynamic_top_k, needs_reindex=False)이 적용될 때, 롤백이 요구한
        # 재색인이 검색시점 needs_reindex=False 에 덮여 사라지면 안 된다. reindex_required
        # 가 True 로 유지돼야 실제 인덱스가 baseline 청킹으로 복원된다(config/인덱스
        # 불일치 방지, 버그 A). 버그가 있으면 이 값이 False 가 된다.
        # ⚠️ 방문1을 agent.run 으로 만들지 않고 손으로 세운다. 롤백이 config 와 함께
        # 진단서도 되돌리게 되면서(구멍 3), 방문2의 처방은 **복원된 리포트**에서
        # 나온다. 그래서 이 시나리오가 성립하려면 복원 대상 리포트가 검색시점 레버를
        # 갖고 있어야 하는데, retrieval_semantic_mismatch 는 처방이 전부 index-time 이라
        # 그 조합을 자연 선택으로는 만들 수 없다. 검증 대상은 "index-time 롤백의
        # 재색인 요구가 검색시점 후속 처방에 덮이지 않는가"이지 어떤 라벨이 뽑히는지가
        # 아니므로, 그 조합을 setup 으로 직접 세운다.
        baseline = {"top_k": 4, "chunk_size": 512, "chunk_overlap": 50}
        restored_report = make_report(60.0, label="retrieval_incomplete_enumeration")
        state = AgentDoctorState(
            report=make_report(50.0, label="retrieval_incomplete_enumeration"),
            index_config={**baseline, "chunk_size": 256},
            iteration=1,
            max_iterations=3,
        )
        state.optimization_history.append(
            OptimizationHistoryItem(
                trial_id="t1",
                request_id="r1",
                iteration=1,
                failure_labels=["retrieval_semantic_mismatch"],
                optimizer="rules",
                status="pending",
                selected_prescription_id="shrink_chunk_size",
                action_key="chunker.chunk_size:decrease",
                supporting_labels=["retrieval_semantic_mismatch"],
                supporting_probes=["p1"],
                before_config=baseline,
                after_config={**baseline, "chunk_size": 256},
                metadata={
                    "pending": True,
                    "before_report": restored_report,
                    "reindex_required": True,      # 방문1 = index-time
                },
            )
        )

        # 방문2: 악화 → 롤백(재색인 요구) 후 검색시점 처방(재색인 불필요) 적용
        buf = StringIO()
        with redirect_stdout(buf):
            state = agent.run(state)
        self.assertEqual(state.status, "applied")
        self.assertTrue(state.reindex_required)   # 롤백 재색인 요구가 보존됨
        out = buf.getvalue()
        self.assertIn("선택한 action: chunker.chunk_size:decrease", out)
        self.assertIn("판정 결과: keep=false, before=60.00, after=50.00", out)
        self.assertIn("선택한 action: retriever.mmr:enable", out)
        verdict_block = out[:out.index("판정 결과: keep=false")]
        self.assertNotIn("다음 단계:", verdict_block)
        tail = out[out.rindex("retriever.mmr:enable"):]
        self.assertIn("reindex_required=true", tail)
        self.assertIn("다음 단계: Index 재색인 후 Eval 재실행", tail)

    def test_budget_exhausted_stops_before_a_new_action_study(self):
        """예산이 바닥나면 같은 라벨이어도 새 ActionStudy 를 시작하지 않는다.

        ⚠️ 전환으로 뒤집힌 동작이다. 예전 판정 기준은 '대표 라벨이 바뀌었는가'라,
        한 라벨이 처방을 갈아 끼우는 동안은 예산을 소비하지 않았다 — 라벨 하나가
        예산 밖에서 무한히 config 를 바꿀 수 있었다는 뜻이다. 이제 iteration 은
        '새 ActionStudy 를 적용한 횟수'이므로(구현계획 §5.3) 다른 action 은 곧
        새 iteration 이고, 예산이 없으면 롤백만 확정하고 종료한다.

        롤백 자체는 예산과 무관하게 먼저 처리된다 — 나빠진 config 를 예산이 없다는
        이유로 남겨 두면 안전망이 무의미해진다.
        """
        state = agent.run(make_state(overall=60.0, iteration=2, max_iterations=3))
        self.assertEqual(state.iteration, 3)                # 방문1: 2 -> 3
        self.assertTrue(state.index_config["context_compression"])
        state.report = make_report(50.0)                    # 악화
        state = agent.run(state)                            # 방문2: 롤백 후 종료
        self.assertEqual(state.iteration, 3)                # 새 study 를 시작하지 않음
        self.assertEqual(state.status, "rolled_back")
        self.assertNotIn("context_compression", state.index_config)
        self.assertIsNone(history.find_pending(state.optimization_history))
        self.assertEqual(len(state.optimization_history), 1)

    def test_sweep_candidates_do_not_consume_extra_iterations(self):
        """같은 action 안의 후보 전환은 예산을 소비하지 않는다(구현계획 §5.3).

        OptimizeTopKSweepTest 가 sweep 전체를 검증하지만, 예산 규칙 자체를 여기에
        고정해 두어 iteration 의미가 바뀌면 바로 드러나게 한다.
        """
        finding = Finding(
            finding_id="sweep", type="retrieval_failure", severity="warning",
            description="gold가 검색 결과에 없음", label="retrieval_missing_gold",
            affected_probes=["p1"],
            metadata={"parameter_candidates": {"retriever.top_k": [7, 9]}},
        )
        state = AgentDoctorState(
            report=DiagnosticReport(
                report_id="r", findings=[finding], overall_score=60.0,
                ragas_scores={"context_recall": 0.7, "faithfulness": 0.7,
                              "noise_sensitivity": 0.2},
                pass_threshold=False,
            ),
            index_config={"top_k": 5, "chunk_size": 512, "chunk_overlap": 50},
            iteration=0, max_iterations=1,
        )

        state = agent.run(state)                             # 후보 1
        self.assertEqual((state.index_config["top_k"], state.iteration), (7, 1))
        state.report.overall_score = 55.0
        state = agent.run(state)                             # 후보 2 — 같은 study
        self.assertEqual((state.index_config["top_k"], state.iteration), (9, 1))

    def test_baseline_dead_end_preserves_same_visit_rollback(self):
        state = agent.run(make_state(overall=60.0))
        state.report = make_report(50.0)
        request, decision = agent.planner.plan(make_state(overall=60.0))
        baseline_result = OptimizationResult(
            request_id=request.request_id,
            status="skipped",
            optimizer="internal",
            selected_action=SelectedAction(action_key=request.action_key),
            metadata={"error_code": "baseline_selected"},
        )

        with patch("agents.optimize.agent.planner.plan", return_value=(request, decision)), patch(
            "agents.optimize.agent.optimizer.run", return_value=baseline_result
        ):
            state = agent.run(state)

        self.assertEqual(state.status, "rolled_back")
        self.assertEqual(state.index_config["top_k"], 4)
        self.assertEqual(state.optimization_report.status, "failed")

    def test_reindex_rollback_requirement_survives_runtime_followup(self):
        """재색인형 B 롤백 직후 런타임형 C가 복원 작업을 지우지 않는다."""
        state = make_state(overall=50.0)
        state.active_index_key = "index-b"
        state.active_eval_key = "eval-b"
        request, decision = agent.planner.plan(state)
        runtime_result = agent.optimizer.run(request)
        self.assertFalse(runtime_result.needs_reindex)

        judged = OptimizationHistoryItem(
            trial_id="trial-b",
            request_id="request-b",
            iteration=1,
            failure_labels=["too_long_context"],
            optimizer="rules",
            status="failed",
            before_config={
                "top_k": 4,
                "chunk_size": 512,
                "chunk_overlap": 50,
            },
        )
        judged.metadata.update(
            {
                "before_index_key": "index-a",
                "before_eval_key": "eval-a",
                "reindex_required": True,
            }
        )
        verdict = Verdict(
            keep=False,
            before_score=60.0,
            after_score=50.0,
            reason="성능 하락",
        )
        baseline_report = make_report(60.0)

        def rollback_first(current):
            current.index_config = dict(judged.before_config)
            current.reindex_required = True
            return judged, verdict, baseline_report

        with (
            patch(
                "agents.optimize.agent._judge_pending_trial",
                side_effect=rollback_first,
            ),
            patch(
                "agents.optimize.agent.planner.plan",
                return_value=(request, decision),
            ),
            patch(
                "agents.optimize.agent.optimizer.run",
                return_value=runtime_result,
            ),
        ):
            out = agent.run(state)

        pending = history.find_pending(out.optimization_history)
        self.assertTrue(out.reindex_required)
        self.assertEqual(pending.metadata["before_index_key"], "index-a")
        self.assertEqual(pending.metadata["before_eval_key"], "eval-a")


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
        self.assertIn("선택한 action: retriever.top_k:increase", buf.getvalue())

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
        self.assertIn(
            ("retrieval_missing_gold", "increase_top_k"),
            state.completed_prescriptions,
        )
        self.assertIn("다음 단계: Index 경유(물리 재색인 생략) 후 Eval 재실행", buf.getvalue())

        # 끝난 탐색은 다시 시작하지 않는다. sweep 승자가 baseline 을 움직이므로
        # study fingerprint 만으로는 부족하고, 측정한 후보값이 결과 baseline 에서도
        # 소진 처리돼야 같은 축을 곧바로 재탐색하지 않는다.
        finished = state.optimization_history[-1]
        state.report = self._report(65.0)
        request, _decision = agent.planner.plan(
            state,
            blacklist=set(state.blocked_action_attempts)
            | set(state.completed_action_studies),
        )
        self.assertNotEqual(
            getattr(request, "action_key", None),
            "retriever.top_k:increase",
        )

        state = agent.run(state)
        self.assertIsNone(history.find_active_study(state.optimization_history))
        self.assertIs(state.optimization_history[-1], finished)

    def test_absolute_visit_limit_restores_active_sweep_baseline(self):
        state = agent.run(self._state())
        self.assertEqual(state.index_config["top_k"], 7)
        state.optimize_visit_count = 19

        state = agent.run(state)

        self.assertEqual(state.optimize_visit_count, 20)
        self.assertEqual(state.index_config["top_k"], 5)
        self.assertEqual(state.status, "rolled_back")
        self.assertIsNone(history.find_active_study(state.optimization_history))
        self.assertTrue(
            state.optimization_history[0].metadata["visit_limit_reached"]
        )
        self.assertEqual(state.optimization_report.status, "failed")
        self.assertIn("절대 방문 상한", state.optimization_report.summary)

        rollback_report = state.optimization_report
        state.report = self._report(60.0)
        state = agent.run(state)

        self.assertEqual(state.optimize_visit_count, 20)
        self.assertEqual(state.status, "verified")
        self.assertIsNot(state.optimization_report, rollback_report)
        self.assertEqual(state.optimization_report.status, "skipped")
        self.assertIn("절대 방문 상한", state.optimization_report.summary)
        self.assertEqual(graph.route_after_optimize(state), "serve")

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
        self.assertEqual(
            state.optimization_history[-1].action_key,
            "context.compression.enabled:enable",
        )
        state.report = make_report(75.0, label="retrieval_missing_gold")  # 개선 + 다음 라벨
        buf = StringIO()
        with redirect_stdout(buf):
            state = agent.run(state)                         # 방문2: 예산소진 → 유지 판정만
        report = state.optimization_report
        self.assertEqual(report.status, "applied")
        self.assertIn("유지", report.summary)
        out = buf.getvalue()
        self.assertIn("지지 라벨: too_long_context", out)
        self.assertIn("선택한 action: context.compression.enabled:enable", out)
        self.assertIn("판정 결과: keep=true, before=60.00, after=75.00", out)
        self.assertIn("다음 단계: Serve 이동 (verified, 반복 예산 소진)", out)
        self.assertNotIn("선택한 action: -", out)

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
        self.assertIn("선택한 action:", out)
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

    def test_route_after_eval_pass_with_pending_goes_optimize_to_finalize(self):
        # 방금 적용한 처방으로 품질이 통과했는데 아직 판정(마감) 안 됐으면,
        # Optimize 를 한 번 더 태워 pending 을 확정한 뒤 Serve 로 보낸다.
        state = AgentDoctorState(report=make_report(90.0, pass_threshold=True),
                                 iteration=1, max_iterations=3,
                                 optimization_history=[self._pending()])
        self.assertEqual(self._route(graph.route_after_eval, state), "optimize")

    def test_route_after_eval_pass_with_active_study_pending_goes_serve(self):
        # 진행 중 sweep(active_study)은 통과 시 기존대로 그대로 Serve.
        item = self._pending()
        item.metadata["active_study"] = True
        state = AgentDoctorState(report=make_report(90.0, pass_threshold=True),
                                 iteration=1, max_iterations=3,
                                 optimization_history=[item])
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


class OptimizeVisitBudgetInvariantTest(unittest.TestCase):
    """종료 보장은 iteration 이 아니라 optimize_visit_count 가 한다.

    iteration 은 "새 ActionStudy 를 적용한 횟수"라 소비하지 않는 경로가 여럿이다
    (sweep 중간 후보, 판정만 한 방문, 적용 불가 소진, study 오류 복원…). graph.py 는
    iteration 만 읽고 visit 은 모른 채 라우팅하므로, **미소비 경로가 반복돼도 닫히는
    이유는 오직 visit 이 무조건 증가하기 때문**이다. 그 불변조건을 여기 고정한다.
    """

    _TERMINAL = {"verified", "skipped", "manual_required", "already_optimal", "error"}

    def _drive(self, state, max_steps=64):
        """graph 라우팅을 흉내 내 config 가 안 바뀔 때까지 Optimize 를 반복 호출한다."""
        seen_counts = []
        for _ in range(max_steps):
            before = state.optimize_visit_count
            with redirect_stdout(StringIO()):
                state = agent.run(state)
            seen_counts.append((before, state.optimize_visit_count))
            if graph.route_after_optimize(state) != "index":
                return state, seen_counts
        self.fail(f"Optimize 방문이 {max_steps}회 안에 끝나지 않았다")

    def _assert_visit_consumed(self, transitions):
        """상한 도달 방문을 빼면 매 방문이 visit 을 정확히 1 소비한다."""
        for before, after in transitions:
            self.assertIn(
                after - before,
                (0, 1),
                f"visit 이 1을 넘게 움직였다: {before} → {after}",
            )
            if after == before:
                # 증가하지 않는 유일한 경로는 상한 도달이며 그 방문은 종료 경로다.
                self.assertGreaterEqual(before, 20)

    def test_every_visit_consumes_the_visit_budget(self):
        """처방 적용 → 롤백 → 다음 action 이 반복돼도 방문 예산이 계속 줄어든다."""
        state = make_state(overall=60.0, max_iterations=5)
        transitions = []
        for _ in range(12):
            before = state.optimize_visit_count
            with redirect_stdout(StringIO()):
                state = agent.run(state)
            transitions.append((before, state.optimize_visit_count))
            if graph.route_after_optimize(state) != "index":
                break
            # Eval 이 매번 악화를 보고한다 = 최악 시나리오(적용·롤백 왕복)
            state.report = make_report(50.0)
        self._assert_visit_consumed(transitions)
        self.assertTrue(all(after > before for before, after in transitions))

    def test_iteration_free_paths_still_close(self):
        """iteration 을 소비하지 않는 경로들도 결국 Serve 로 닫힌다.

        sweep 중간 후보(경로 1)와 sweep 종료(경로 2)는 iteration 이 1에 묶여 있어
        graph.py 의 예산 가드가 발동하지 않는다. 그래도 방문은 유한하다.
        """
        finding = Finding(
            finding_id="sweep", type="retrieval_failure", severity="warning",
            description="gold가 검색 결과에 없음", label="retrieval_missing_gold",
            affected_probes=["p1"],
            metadata={"parameter_candidates": {"retriever.top_k": [7, 9, 11]}},
        )
        state = AgentDoctorState(
            report=DiagnosticReport(
                report_id="r", findings=[finding], overall_score=60.0,
                ragas_scores={"context_recall": 0.7, "faithfulness": 0.7,
                              "noise_sensitivity": 0.2},
                pass_threshold=False,
            ),
            index_config={"top_k": 5, "chunk_size": 512, "chunk_overlap": 50},
            iteration=0, max_iterations=1,
        )

        state, transitions = self._drive(state)

        self._assert_visit_consumed(transitions)
        self.assertEqual(state.iteration, 1)            # sweep 전체가 iteration 하나
        self.assertGreater(state.optimize_visit_count, 1)
        self.assertIn(state.status, self._TERMINAL)

    def test_visit_limit_converges_in_one_extra_round(self):
        """상한 도달 방문은 visit 을 늘리지 않는다 — 그래도 한 바퀴 안에 닫힌다.

        복원이 config 를 바꾸면 status=rolled_back 이라 Index 를 한 번 더 돌지만,
        그 다음 방문에는 복원할 pending 이 없어 verified 로 Serve 에 도달한다.
        """
        state = make_state(overall=60.0)
        with redirect_stdout(StringIO()):
            state = agent.run(state)
        state.optimize_visit_count = 19
        state.report = make_report(50.0)

        state, transitions = self._drive(state)

        self.assertEqual(state.optimize_visit_count, 20)
        self.assertIn(state.status, self._TERMINAL)
        self.assertEqual(graph.route_after_optimize(state), "serve")
        # 상한 처리에 쓴 방문은 두 번(rolled_back 1 + verified 1)을 넘지 않는다.
        self.assertLessEqual(len(transitions), 3)


if __name__ == "__main__":
    unittest.main()
