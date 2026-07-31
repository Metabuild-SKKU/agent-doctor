"""
tests/test_planner.py
Planner 검증 — 후보값을 어떻게 정하고, 그걸 optimizer 요청으로 어떻게 넘기는지.

두 축을 다룬다.
  1. 후보 산출: 진단 측정값에서 후보를 계산한다(라벨 묶음 + 무릎 분석).
  2. 후보 전달: 후보 수에 따라 rules/internal 요청을 만들고 sweep 입력을 싣는다.

핵심 전제: Eval 은 Finding 을 probe 마다 따로 만든다(affected_probes 는 항상 1개).
같은 원인이 probe N개에서 터지면 Finding 도 N개다. Planner 는 이를 라벨로 묶어
점수(빈도)와 근거값(측정 기반 목표값)을 계산해야 한다.
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import DiagnosticReport, Document, Finding, Probe
from core.state import AgentDoctorState
from agents.optimize import planner
from agents.optimize.planner import _knee, _knee_candidates


def make_finding(probe_id, label, gold_n=0, confirmed=True, candidates=None,
                 gold_ranks=None):
    """probe 1개에서 나온 Finding 하나. gold_n = 그 probe 가 필요로 하는 gold 청크 수.
    gold_ranks = {gold_id: 순위} (Eval tier2 실측). 있으면 개수보다 우선한다."""
    metadata = {}
    if candidates:
        metadata["parameter_candidates"] = candidates
    if gold_ranks is not None:
        metadata["gold_ranks"] = gold_ranks
    return Finding(
        finding_id=f"{probe_id}:{label}",
        type="retrieval_failure",
        severity="warning",
        description=label,
        label=label,
        confirmed=confirmed,
        affected_chunks=[f"c{i}" for i in range(gold_n)],
        affected_probes=[probe_id],
        metadata=metadata,
    )


def _report(findings) -> DiagnosticReport:
    if isinstance(findings, Finding):
        findings = [findings]
    return DiagnosticReport(
        report_id="report",
        findings=findings,
        overall_score=60.0,
        ragas_scores={"context_recall": 0.6},
        pass_threshold=False,
    )


def make_state(findings, top_k=5):
    return AgentDoctorState(
        report=_report(findings),
        index_config={"chunk_size": 512, "chunk_overlap": 50, "top_k": top_k},
        iteration=0, max_iterations=3,
    )


# ── 1. 후보 산출 ──────────────────────────────────────────────────

class KneeTest(unittest.TestCase):
    """한계비용 무릎 분석: 'probe 1개 더 커버하는 비용'이 급등하면 멈춘다."""

    def test_outlier_does_not_drag_result(self):
        # 100 하나 때문에 top_k=100 이 되면 노이즈·비용 폭발. 평균(16.4)도 끌려간다.
        self.assertEqual(_knee([3, 4, 4, 5, 6, 7, 8, 12, 15, 100]), 8)

    def test_covers_all_when_values_are_dense(self):
        # 한 칸씩 올릴 때마다 probe 하나씩 회수 → 끝까지 가는 게 이득
        self.assertEqual(_knee([3, 4, 5]), 5)
        self.assertEqual(_knee([5, 6, 7, 8, 9, 10, 11, 12]), 12)

    def test_single_value(self):
        self.assertEqual(_knee([7]), 7)

    def test_stops_before_expensive_jump(self):
        # 2 → 50 은 probe 1개에 48을 쓰는 셈이라 멈춘다(4/5 커버).
        self.assertEqual(_knee([2, 2, 2, 2, 50]), 2)

    def test_empty_input_has_clear_contract_error(self):
        with self.assertRaisesRegex(ValueError, "하나 이상의 필요값"):
            _knee([])


class KneeCandidatesTest(unittest.TestCase):
    """sweep 후보: 무릎 위 구간은 '추측이 밑진다고 본' 곳이라 실측으로 확인한다."""

    def test_candidates_start_at_knee_and_go_up(self):
        # 무릎 아래(6,7)는 무릎(8)이 지배하므로 후보에 없다.
        self.assertEqual(_knee_candidates([3, 4, 4, 5, 6, 7, 8, 12, 15, 100]),
                         [8, 12, 15])

    def test_single_candidate_when_knee_covers_everything(self):
        # 더 올릴 이유가 없으면 후보 1개 → sweep 불필요(rules 로 1회 검증)
        self.assertEqual(_knee_candidates([3, 4, 5]), [5])
        self.assertEqual(_knee_candidates([7]), [7])

    def test_candidate_count_is_capped(self):
        # 후보 1개당 파이프라인 전체 재평가가 드므로 상한을 넘지 않는다.
        self.assertLessEqual(len(_knee_candidates(list(range(1, 30)))),
                             planner._MAX_SWEEP_CANDIDATES)


class GroundedValueTest(unittest.TestCase):
    """근거값: 방향 키워드(×2 추측) 대신 진단 측정값에서 계산한다."""

    def test_enumeration_top_k_computed_from_gold_counts(self):
        golds = [3, 4, 4, 5, 6, 7, 8, 12, 15, 100]
        findings = [
            make_finding(f"p{i}", "retrieval_incomplete_enumeration", gold_n=n)
            for i, n in enumerate(golds)
        ]
        request, decision = planner.plan(make_state(findings, top_k=5))

        self.assertEqual(decision.mode, "apply_optimize")
        first = request.candidates[0]
        self.assertEqual(first.id, "dynamic_top_k")
        # ×2 추측이면 [10] 하나가 나왔을 것. 측정 기반은 무릎(8)부터 그 위로.
        self.assertEqual(first.search_space, {"retriever.top_k": [8, 12, 15]})
        # 후보가 여러 개 → internal 이 방문에 걸쳐 sweep 하고 실측으로 승자를 고른다.
        self.assertEqual(request.optimizer, "internal")
        self.assertEqual(request.max_trials, 3)

    def test_gold_rank_beats_count_for_top_k(self):
        # gold 3개지만 가장 늦은 놈이 20위 → top_k 는 개수(3)가 아니라 순위(20) 기준.
        # multi-hop/나열형에서 개수 ≪ 순위인 경우를 실측으로 반영한다.
        findings = [
            make_finding("p1", "retrieval_incomplete_enumeration", gold_n=3,
                         gold_ranks={"g_a": 2, "g_b": 9, "g_c": 20}),
        ]
        request, _decision = planner.plan(make_state(findings, top_k=5))

        self.assertEqual(request.search_space, {"retriever.top_k": [20]})

    def test_missing_gold_also_grounds_top_k_from_rank(self):
        # missing_gold 도 top_k 처방(increase_top_k)이 있어 순위 근거가 먹힌다.
        findings = [
            make_finding("p1", "retrieval_missing_gold", gold_n=2,
                         gold_ranks={"g_a": 7, "g_b": 14}),
        ]
        request, _decision = planner.plan(make_state(findings, top_k=5))

        self.assertEqual(request.search_space, {"retriever.top_k": [14]})

    def test_candidate_miss_grounds_window_from_gold_rank_knee(self):
        # 후보창은 '가장 늦게 나오는 gold 의 순위'가 근거다 — ×2 추측(40)이 아니라 무릎(28).
        findings = [
            make_finding(f"p{i}", "retrieval_rerank_candidate_miss",
                         gold_ranks={"g": rank})
            for i, rank in enumerate([24, 25, 26, 27, 28])
        ]
        state = make_state(findings)
        state.index_config.update({"use_reranker": True, "rerank_candidates": 20})

        request, _decision = planner.plan(state)

        self.assertEqual(request.search_space, {"reranker.candidate_count": [28]})

    def test_candidate_miss_keeps_reachable_ranks_from_mixed_probe(self):
        """한 probe 에 도달 가능/불가 순위가 섞이면 도달 가능한 쪽 근거는 살아남아야 한다.

        probe 단위(그 probe 의 최대 순위)로 상한을 걸면 90 때문에 30 이라는 멀쩡한 근거까지
        같이 버려지고 방향 키워드 추측으로 내려간다.
        """
        findings = [
            make_finding("p1", "retrieval_rerank_candidate_miss",
                         gold_ranks={"g_near": 30, "g_far": 90}),
        ]
        state = make_state(findings)
        state.index_config.update({
            "use_reranker": True,
            "rerank_candidates": 20,
            "rerank_candidate_policy": {"max_candidates": 50},
        })

        request, _decision = planner.plan(state)

        self.assertEqual(request.search_space, {"reranker.candidate_count": [30]})

    def test_candidate_miss_respects_policy_ceiling(self):
        # 후보 1개 = cross-encoder 추론 1쌍이라 상한을 넘는 후보는 내지 않는다.
        findings = [
            make_finding("p1", "retrieval_rerank_candidate_miss",
                         gold_ranks={"g": 90}),
        ]
        state = make_state(findings)
        state.index_config.update({
            "use_reranker": True,
            "rerank_candidates": 20,
            "rerank_candidate_policy": {"max_candidates": 50},
        })

        request, _decision = planner.plan(state)

        # 근거값이 상한 밖 → 방향 키워드 폴백(20×2=40)으로 내려간다. 90 은 나오지 않는다.
        self.assertEqual(request.search_space, {"reranker.candidate_count": [40]})

    def test_fusion_loss_shifts_weight_toward_favored_channel(self):
        findings = [
            make_finding("p1", "retrieval_rank_fusion_loss"),
            make_finding("p2", "retrieval_rank_fusion_loss"),
        ]
        for finding in findings:
            finding.metadata["favored_channel"] = "lexical"
        state = make_state(findings)
        state.index_config.update({"use_hybrid": True, "hybrid_dense_weight": 0.7})

        request, _decision = planner.plan(state)

        # lexical 우세 → dense 가중치를 내린다(정책 폭 0.1/0.2).
        self.assertEqual(
            request.search_space, {"retriever.hybrid_dense_weight": [0.6, 0.5]}
        )

    def test_fusion_loss_drops_key_without_direction_evidence(self):
        """우세 채널이 동수면 근거가 없다 — 자리표시자 문자열이 config 에 새지 않아야 한다."""
        findings = [
            make_finding("p1", "retrieval_rank_fusion_loss"),
            make_finding("p2", "retrieval_rank_fusion_loss"),
        ]
        findings[0].metadata["favored_channel"] = "dense"
        findings[1].metadata["favored_channel"] = "lexical"
        state = make_state(findings)
        state.index_config.update({"use_hybrid": True, "hybrid_dense_weight": 0.7})

        request, _decision = planner.plan(state)

        self.assertNotIn("retriever.hybrid_dense_weight", request.search_space)

    def test_low_rank_does_not_ground_top_k(self):
        # low_rank 처방은 리랭커(use_reranker)뿐 — top_k 를 처방하지 않으므로
        # 순위가 있어도 top_k search_space 가 생기지 않는다(옵션 1).
        findings = [
            make_finding("p1", "retrieval_low_rank",
                         gold_ranks={"g_a": 15}),
        ]
        request, _decision = planner.plan(make_state(findings, top_k=5))

        self.assertNotIn("retriever.top_k", request.search_space)

    def test_gold_beyond_wide_n_is_excluded_from_top_k(self):
        # wide 밖 gold(None)는 top_k 로 도달 불가라 제외 — 도달 가능한 최대 순위(11)만 쓴다.
        findings = [
            make_finding("p1", "retrieval_incomplete_enumeration", gold_n=3,
                         gold_ranks={"g_a": 4, "g_b": 11, "g_far": None}),
        ]
        request, _decision = planner.plan(make_state(findings, top_k=5))

        self.assertEqual(request.search_space, {"retriever.top_k": [11]})

    def test_falls_back_to_direction_keyword_without_evidence(self):
        # gold 개수가 없으면(affected_chunks 비어있음) 계산 불가 → ×2 폴백
        findings = [make_finding("p1", "retrieval_incomplete_enumeration", gold_n=0)]
        request, _decision = planner.plan(make_state(findings, top_k=5))

        self.assertEqual(request.candidates[0].search_space, {"retriever.top_k": [10]})

    def test_eval_supplied_candidates_win_over_computed(self):
        # Eval 이 직접 후보를 주면 planner 계산보다 우선한다(후보 산출을 Eval 로
        # 옮기더라도 planner 를 고치지 않게 하는 확장점).
        findings = [
            make_finding("p1", "retrieval_incomplete_enumeration", gold_n=4,
                         candidates={"top_k": [6, 9]})
        ]
        request, _decision = planner.plan(make_state(findings, top_k=5))

        self.assertEqual(request.search_space, {"retriever.top_k": [6, 9]})

    def test_top_k_candidates_are_clamped_to_prescription_direction(self):
        findings = [
            make_finding(
                "p1",
                "retrieval_missing_gold",
                candidates={"top_k": [3, 15]},
            )
        ]

        request, _decision = planner.plan(make_state(findings, top_k=10))

        self.assertEqual(request.search_space, {"retriever.top_k": [15]})


class GroupingTest(unittest.TestCase):
    """같은 라벨의 finding 을 묶어야 빈도가 제대로 계산된다."""

    def test_frequency_counts_all_affected_probes(self):
        # 같은 라벨이 probe 3개에서 터짐 → 빈도 3 (묶기 전에는 항상 1이었다)
        findings = [
            make_finding(f"p{i}", "retrieval_incomplete_enumeration", gold_n=4)
            for i in range(3)
        ]
        request, _decision = planner.plan(make_state(findings))

        self.assertIn("probe 3개 영향", request.reason)

    def test_more_frequent_label_wins_within_same_group(self):
        # 두 라벨 모두 A그룹. probe 수가 많은 쪽이 먼저 처방된다.
        findings = [
            make_finding("p1", "retrieval_missing_gold"),
            make_finding("p2", "retrieval_incomplete_enumeration", gold_n=4),
            make_finding("p3", "retrieval_incomplete_enumeration", gold_n=4),
            make_finding("p4", "retrieval_incomplete_enumeration", gold_n=4),
        ]
        request, _decision = planner.plan(make_state(findings))

        self.assertEqual(request.failure_label, "retrieval_incomplete_enumeration")

    def test_context_noise_can_precede_top_k_expansion_when_dominant(self):
        findings = [
            make_finding("noise1", "context_noise_interference"),
            make_finding("noise2", "context_noise_interference"),
            make_finding("noise3", "context_noise_interference"),
            make_finding("enum1", "retrieval_incomplete_enumeration", gold_n=4),
        ]

        request, decision = planner.plan(make_state(findings, top_k=5))

        self.assertEqual(decision.mode, "apply_optimize")
        self.assertEqual(request.failure_label, "context_noise_interference")
        first = request.candidates[0]
        self.assertEqual(first.id, "context_compression")
        self.assertEqual(
            first.search_space,
            {"context.compression.enabled": [True]},
        )


class ConfirmedGatingTest(unittest.TestCase):
    """예비(confirmed=False) 진단에는 비싼 처방 trial 을 쓰지 않는다."""

    def test_preliminary_findings_are_not_prescribed(self):
        findings = [
            make_finding("p1", "retrieval_incomplete_enumeration",
                         gold_n=4, confirmed=False)
        ]
        request, decision = planner.plan(make_state(findings))

        self.assertIsNone(request)
        self.assertEqual(decision.mode, "use_current")
        self.assertEqual(decision.status, "skipped")

    def test_confirmed_finding_wins_over_preliminary(self):
        findings = [
            make_finding("p1", "retrieval_missing_gold", confirmed=False),
            make_finding("p2", "retrieval_incomplete_enumeration",
                         gold_n=4, confirmed=True),
        ]
        request, _decision = planner.plan(make_state(findings))

        self.assertEqual(request.failure_label, "retrieval_incomplete_enumeration")


# ── 2. 후보 전달 (optimizer 요청 계약) ────────────────────────────

class PlannerCandidateListTest(unittest.TestCase):
    """후보 수에 따라 rules/internal 을 고르고 sweep 입력을 싣는다."""

    def test_preliminary_finding_is_not_auto_applied(self):
        finding = make_finding("p1", "retrieval_missing_gold", confirmed=False)
        request, decision = planner.plan(AgentDoctorState(report=_report(finding)))

        self.assertIsNone(request)
        self.assertEqual(decision.status, "skipped")

    def test_single_non_chunk_candidate_uses_rules_backend(self):
        # chunk가 아닌 후보는 하나뿐이면 sweep 할 게 없어 rules 로 1회 검증한다.
        findings = [make_finding("p1", "retrieval_incomplete_enumeration", gold_n=4)]
        request, _decision = planner.plan(make_state(findings))

        self.assertEqual(request.optimizer, "rules")
        self.assertEqual(request.max_trials, 1)

    def test_top_k_candidates_make_one_internal_request(self):
        finding = make_finding(
            "p1", "retrieval_missing_gold",
            candidates={"top_k": [3, 7, 9]},
        )
        state = make_state([finding])
        request, decision = planner.plan(state)

        self.assertEqual(decision.mode, "apply_optimize")
        self.assertEqual(request.optimizer, "internal")
        self.assertEqual(request.search_space, {"retriever.top_k": [7, 9]})
        self.assertEqual(request.max_trials, 2)

    def test_chunk_candidates_include_preview_inputs(self):
        finding = make_finding(
            "p1", "too_long_context",
            candidates={"chunker.chunk_size": [400, 600]},
        )
        state = AgentDoctorState(
            report=_report(finding),
            documents=[Document("d1", "memory", "txt", "가" * 1000)],
            probes=[
                Probe(
                    probe_id="p1",
                    question="질문",
                    source="taxonomy",
                    gold_spans=[{"doc_id": "d1", "start": 100, "end": 180}],
                )
            ],
            index_config={"top_k": 5, "chunk_size": 512, "chunk_overlap": 50},
        )
        request, _decision = planner.plan(
            state,
            blacklist={
                ("too_long_context", "decrease_top_k"),
                ("too_long_context", "context_compression"),
            },
        )

        self.assertEqual(request.optimizer, "internal")
        self.assertEqual(request.search_space, {"chunker.chunk_size": [400, 600]})
        context = request.metadata["chunk_precheck_context"]
        self.assertEqual(context["documents"][0].doc_id, "d1")
        self.assertEqual(context["span_source"], "structural_evidence_windows")
        self.assertEqual(context["evidence_spans"][0]["start"], 0)
        self.assertEqual(context["evidence_spans"][0]["end"], 1000)

    def test_single_explicit_chunk_candidate_uses_prescreener(self):
        finding = make_finding(
            "p1", "too_long_context",
            candidates={"chunker.chunk_size": [400]},
        )
        state = AgentDoctorState(
            report=_report(finding),
            documents=[Document("d1", "memory", "txt", "가" * 1000)],
            probes=[
                Probe(
                    probe_id="p1",
                    question="질문",
                    source="taxonomy",
                    gold_spans=[{"doc_id": "d1", "start": 100, "end": 180}],
                )
            ],
            index_config={"top_k": 5, "chunk_size": 512, "chunk_overlap": 50},
        )

        request, _decision = planner.plan(
            state,
            blacklist=self._chunk_blacklist(),
        )

        self.assertEqual(request.search_space, {"chunker.chunk_size": [400]})
        self.assertEqual(request.optimizer, "internal")
        self.assertEqual(request.max_trials, 1)
        self.assertIn("chunk_precheck_context", request.metadata)

    def test_single_explicit_chunk_candidate_without_measurements_uses_rules(self):
        finding = make_finding(
            "p1", "too_long_context",
            candidates={"chunker.chunk_size": [400]},
        )
        state = AgentDoctorState(
            report=_report(finding),
            documents=[Document("d1", "memory", "txt", "가" * 1000)],
            index_config={"top_k": 5, "chunk_size": 512, "chunk_overlap": 50},
        )

        request, _decision = planner.plan(
            state,
            blacklist=self._chunk_blacklist(),
        )

        self.assertEqual(request.search_space, {"chunker.chunk_size": [400]})
        self.assertEqual(request.optimizer, "rules")
        self.assertEqual(request.max_trials, 1)
        self.assertNotIn("chunk_precheck_context", request.metadata)

    def test_multiple_chunk_candidates_without_measurements_use_rules(self):
        finding = make_finding(
            "p1", "too_long_context",
            candidates={"chunker.chunk_size": [400, 600]},
        )
        state = AgentDoctorState(
            report=_report(finding),
            index_config={"top_k": 5, "chunk_size": 512, "chunk_overlap": 50},
        )

        request, _decision = planner.plan(
            state,
            blacklist=self._chunk_blacklist(),
        )

        self.assertEqual(
            request.search_space,
            {"chunker.chunk_size": [400, 600]},
        )
        self.assertEqual(request.optimizer, "rules")
        self.assertNotIn("chunk_precheck_context", request.metadata)

    def test_chunk_strategy_candidates_make_one_internal_request(self):
        # 전략 교체는 청크 사전검증(prescreener) 대상이 아니라 후보 수만 보고
        # internal 로 간다 — 경계 생성 규칙 자체가 바뀌어 기존 경계 기하가 무의미하다.
        finding = make_finding(
            "p1", "chunking_context_mismatch",
            candidates={
                "chunker.strategy": ["recursive_sentence", "markdown_recursive"]
            },
        )
        state = AgentDoctorState(
            report=_report(finding),
            index_config={
                "top_k": 5,
                "chunk_size": 512,
                "chunk_overlap": 50,
                "chunk_strategy": "fixed",
            },
        )

        request, decision = planner.plan(
            state,
            blacklist=self._chunk_strategy_blacklist(),
        )

        self.assertEqual(decision.mode, "apply_optimize")
        self.assertEqual(request.candidates[0].id, "switch_to_recursive_sentence")
        self.assertEqual(
            request.search_space,
            {"chunker.strategy": ["recursive_sentence", "markdown_recursive"]},
        )
        self.assertEqual(request.optimizer, "internal")
        self.assertEqual(request.max_trials, 2)
        self.assertNotIn("chunk_precheck_context", request.metadata)

    def test_single_chunk_strategy_candidate_uses_rules_backend(self):
        # rules 처방 하나만 있으면 후보값도 하나라 sweep 할 게 없다(rules 1회 검증).
        finding = make_finding("p1", "chunking_context_mismatch")
        state = AgentDoctorState(
            report=_report(finding),
            index_config={
                "top_k": 5,
                "chunk_size": 512,
                "chunk_overlap": 50,
                "chunk_strategy": "fixed",
            },
        )

        request, _decision = planner.plan(
            state,
            blacklist=self._chunk_strategy_blacklist(),
        )

        self.assertEqual(
            request.search_space,
            {"chunker.strategy": ["recursive_sentence"]},
        )
        self.assertEqual(request.optimizer, "rules")
        self.assertEqual(request.max_trials, 1)
        self.assertNotIn("chunk_precheck_context", request.metadata)

    @staticmethod
    def _chunk_blacklist():
        """too_long_context의 앞선 두 처방을 건너뛰고 chunk 처방을 선택한다."""

        return {
            ("too_long_context", "decrease_top_k"),
            ("too_long_context", "context_compression"),
        }

    @staticmethod
    def _chunk_strategy_blacklist():
        """chunking_context_mismatch의 크기·중첩 처방을 건너뛰고 전략 교체를 고른다."""

        return {
            ("chunking_context_mismatch", "increase_chunk_overlap"),
            ("chunking_context_mismatch", "increase_chunk_size"),
        }

    @staticmethod
    def _chunk_policy():
        """chunk_size 후보 경계 테스트용 기본 정책."""

        return {
            "target_quantile": 0.85,
            "margin_ratio": 0.20,
            "rounding_step": 50,
            "path_fractions": [0.33, 0.66, 1.0],
            "candidate_count": 3,
            "min_span_count": 3,
            "max_step_ratio": 0.25,
            "min_chunk_size": 200,
            "max_chunk_size": 1500,
        }

    @staticmethod
    def _evidence_analysis(length: int = 100):
        """동일 길이 evidence window 세 개와 provenance를 만든다."""

        return (
            [
                {
                    "doc_id": "d1",
                    "start": index * (length + 10),
                    "end": index * (length + 10) + length,
                }
                for index in range(3)
            ],
            {
                "status": "grounded",
                "source": "structural_evidence_windows",
            },
        )

    def test_chunk_size_does_not_shrink_below_undelimited_evidence(self):
        finding = make_finding("p1", "too_long_context")
        state = AgentDoctorState(
            report=_report(finding),
            documents=[Document("d1", "memory", "txt", "가" * 3000)],
            probes=[
                Probe(
                    probe_id="p1",
                    question="질문 1",
                    source="taxonomy",
                    answer_exists=True,
                    gold_spans=[
                        {"doc_id": "d1", "start": 0, "end": 100},
                        {"doc_id": "d1", "start": 200, "end": 500},
                        {"doc_id": "d1", "start": 600, "end": 1000},
                    ],
                ),
                Probe(
                    probe_id="p2",
                    question="관련 없는 질문",
                    source="taxonomy",
                    answer_exists=True,
                    gold_spans=[
                        {"doc_id": "d1", "start": 1200, "end": 2400},
                    ],
                ),
            ],
            index_config={
                "top_k": 5,
                "chunk_size": 800,
                "chunk_overlap": 50,
                "chunk_candidate_policy": {
                    "target_quantile": 0.85,
                    "margin_ratio": 0.20,
                    "rounding_step": 50,
                    "path_fractions": [0.33, 0.66, 1.0],
                    "candidate_count": 3,
                    "min_span_count": 3,
                },
            },
        )

        request, _decision = planner.plan(state, blacklist=self._chunk_blacklist())

        self.assertEqual(request.search_space, {})
        grounding = request.metadata["candidate_grounding"]
        self.assertEqual(grounding["status"], "direction_conflict")
        self.assertEqual(grounding["span_count"], 3)
        self.assertEqual(grounding["source"], "structural_evidence_windows")

    def test_exact_spans_expand_before_candidate_calculation(self):
        finding = make_finding("p1", "too_long_context")
        finding.affected_probes = ["p1", "p2"]
        state = AgentDoctorState(
            report=_report(finding),
            documents=[Document("d1", "memory", "txt", "가" * 3000)],
            probes=[
                Probe(
                    probe_id="p1",
                    question="정확한 질문",
                    source="llm_generated",
                    answer_exists=True,
                    gold_spans=[
                        {"doc_id": "d1", "start": 0, "end": 100},
                        {"doc_id": "d1", "start": 200, "end": 400},
                        {"doc_id": "d1", "start": 500, "end": 800},
                    ],
                    metadata={"span_grounding": {"status": "exact"}},
                ),
                Probe(
                    probe_id="p2",
                    question="폴백 질문",
                    source="llm_generated",
                    answer_exists=True,
                    gold_spans=[
                        {"doc_id": "d1", "start": 1000, "end": 2000},
                    ],
                    metadata={"span_grounding": {"status": "chunk_fallback"}},
                ),
            ],
            index_config={
                "top_k": 5,
                "chunk_size": 800,
                "chunk_overlap": 50,
                "chunk_candidate_policy": {
                    "target_quantile": 0.85,
                    "margin_ratio": 0.20,
                    "rounding_step": 50,
                    "path_fractions": [0.33, 0.66, 1.0],
                    "candidate_count": 3,
                    "min_span_count": 3,
                },
            },
        )

        request, _decision = planner.plan(state, blacklist=self._chunk_blacklist())

        self.assertEqual(request.search_space, {})
        grounding = request.metadata["candidate_grounding"]
        self.assertEqual(grounding["status"], "direction_conflict")
        self.assertEqual(grounding["span_count"], 3)
        self.assertEqual(grounding["source"], "structural_evidence_windows")

    def test_too_few_evidence_windows_do_not_trigger_blind_halving(self):
        finding = make_finding("p1", "too_long_context")
        state = AgentDoctorState(
            report=_report(finding),
            documents=[Document("d1", "memory", "txt", "가" * 1000)],
            probes=[
                Probe(
                    probe_id="p1",
                    question="질문",
                    source="taxonomy",
                    answer_exists=True,
                    gold_spans=[
                        {"doc_id": "d1", "start": 0, "end": 100},
                        {"doc_id": "d1", "start": 200, "end": 400},
                    ],
                )
            ],
            index_config={
                "top_k": 5,
                "chunk_size": 800,
                "chunk_overlap": 50,
                "chunk_candidate_policy": {
                    "target_quantile": 0.85,
                    "margin_ratio": 0.20,
                    "rounding_step": 50,
                    "path_fractions": [0.33, 0.66, 1.0],
                    "candidate_count": 3,
                    "min_span_count": 3,
                },
            },
        )

        request, _decision = planner.plan(state, blacklist=self._chunk_blacklist())

        self.assertEqual(request.search_space, {})
        self.assertEqual(request.optimizer, "rules")
        grounding = request.metadata["candidate_grounding"]
        self.assertEqual(grounding["status"], "insufficient_spans")
        self.assertEqual(grounding["source"], "structural_evidence_windows")

    def test_invalid_chunk_policy_falls_back_to_single_rule_value(self):
        finding = make_finding("p1", "too_long_context")
        state = AgentDoctorState(
            report=_report(finding),
            documents=[Document("d1", "memory", "txt", "가" * 1000)],
            probes=[
                Probe(
                    probe_id="p1",
                    question="질문",
                    source="taxonomy",
                    answer_exists=True,
                    gold_spans=[{"doc_id": "d1", "start": 0, "end": 300}],
                )
            ],
            index_config={
                "top_k": 5,
                "chunk_size": 800,
                "chunk_overlap": 50,
                "chunk_candidate_policy": {"target_quantile": 2.0},
            },
        )

        request, _decision = planner.plan(state, blacklist=self._chunk_blacklist())

        self.assertEqual(request.search_space, {"chunker.chunk_size": [400]})
        self.assertEqual(request.optimizer, "rules")
        self.assertNotIn("chunk_precheck_context", request.metadata)
        self.assertEqual(
            request.metadata["candidate_grounding"]["status"],
            "invalid_policy",
        )

    def test_chunk_candidate_limits_apply_ratio_and_absolute_bounds(self):
        policy = {
            "max_step_ratio": 0.25,
            "min_chunk_size": 200,
            "max_chunk_size": 1500,
        }

        self.assertEqual(
            planner._chunk_candidate_limits(800, 50, policy),
            (600, 1000),
        )
        self.assertEqual(
            planner._chunk_candidate_limits(300, 50, policy),
            (250, 350),
        )
        self.assertEqual(
            planner._chunk_candidate_limits(1400, 50, policy),
            (1050, 1500),
        )
        self.assertIsNone(
            planner._chunk_candidate_limits(10000, 50, policy),
        )

    def test_chunk_size_direction_at_absolute_limit_has_dedicated_status(self):
        cases = [
            (200, "decrease"),
            (1500, "increase"),
            (100, "decrease"),
            (10000, "increase"),
        ]

        for current, direction in cases:
            with self.subTest(current=current, direction=direction):
                state = AgentDoctorState(index_config={
                    "chunk_size": current,
                    "chunk_candidate_policy": self._chunk_policy(),
                })

                values, metadata = planner._ground_chunk_size_candidates(
                    [],
                    state,
                    direction,
                    self._evidence_analysis(),
                )

                self.assertIsNone(values)
                self.assertEqual(metadata["status"], "at_safe_limit")
                self.assertFalse(planner._allows_symbolic_fallback(
                    "chunker.chunk_size",
                    metadata,
                ))

    def test_single_safe_chunk_candidate_at_boundary_is_kept(self):
        state = AgentDoctorState(index_config={
            "chunk_size": 250,
            "chunk_candidate_policy": self._chunk_policy(),
        })

        values, metadata = planner._ground_chunk_size_candidates(
            [],
            state,
            "decrease",
            self._evidence_analysis(),
        )

        self.assertEqual(values, [200])
        self.assertEqual(metadata["status"], "grounded")
        self.assertEqual(metadata["generated_candidates"], [200])

    def test_out_of_range_current_clamps_to_nearest_absolute_bound(self):
        cases = [
            (128, "increase", 160, 200, "min_chunk_size"),
            (2048, "decrease", 100, 1500, "max_chunk_size"),
        ]

        for current, direction, length, expected, bound_name in cases:
            with self.subTest(
                current=current,
                direction=direction,
                bound_name=bound_name,
            ):
                state = AgentDoctorState(index_config={
                    "chunk_size": current,
                    "chunk_candidate_policy": self._chunk_policy(),
                })

                values, metadata = planner._ground_chunk_size_candidates(
                    [],
                    state,
                    direction,
                    self._evidence_analysis(length=length),
                )

                self.assertEqual(values, [expected])
                self.assertEqual(metadata["status"], "grounded")
                self.assertEqual(metadata["safety_bound_clamp"], bound_name)
                self.assertFalse(metadata["max_step_ratio_applied"])
                self.assertEqual(metadata["generated_candidates"], [expected])

    def test_near_out_of_range_current_keeps_normal_candidate_path(self):
        cases = [
            (160, "increase", 160, [200]),
            (1550, "decrease", 100, [1450, 1300, 1200]),
            (1600, "decrease", 100, [1450, 1350, 1200]),
            (1750, "decrease", 100, [1500, 1350]),
            (2000, "decrease", 100, [1500]),
        ]

        for current, direction, length, expected in cases:
            with self.subTest(current=current, direction=direction):
                state = AgentDoctorState(index_config={
                    "chunk_size": current,
                    "chunk_candidate_policy": self._chunk_policy(),
                })

                values, metadata = planner._ground_chunk_size_candidates(
                    [],
                    state,
                    direction,
                    self._evidence_analysis(length=length),
                )

                self.assertEqual(values, expected)
                self.assertTrue(metadata["max_step_ratio_applied"])
                self.assertNotIn("safety_bound_clamp", metadata)

    def test_out_of_range_clamp_does_not_override_evidence_direction_conflict(self):
        cases = [
            (128, "increase", 50),
            (2048, "decrease", 2000),
        ]

        for current, direction, length in cases:
            with self.subTest(current=current, direction=direction):
                state = AgentDoctorState(index_config={
                    "chunk_size": current,
                    "chunk_candidate_policy": self._chunk_policy(),
                })

                values, metadata = planner._ground_chunk_size_candidates(
                    [],
                    state,
                    direction,
                    self._evidence_analysis(length=length),
                )

                self.assertIsNone(values)
                self.assertEqual(metadata["status"], "direction_conflict")
                self.assertNotIn("safety_bound_clamp", metadata)

    def test_single_grounded_chunk_candidate_uses_prescreener(self):
        finding = make_finding("p1", "too_long_context")
        state = AgentDoctorState(
            report=_report(finding),
            documents=[Document("d1", "memory", "txt", "가" * 1000)],
            index_config={
                "top_k": 5,
                "chunk_size": 250,
                "chunk_overlap": 50,
                "chunk_candidate_policy": self._chunk_policy(),
            },
        )

        with patch.object(
            planner,
            "_evidence_windows",
            return_value=self._evidence_analysis(),
        ):
            request, _decision = planner.plan(
                state,
                blacklist=self._chunk_blacklist(),
            )

        self.assertEqual(request.search_space, {"chunker.chunk_size": [200]})
        self.assertEqual(request.optimizer, "internal")
        self.assertEqual(request.max_trials, 1)
        context = request.metadata["chunk_precheck_context"]
        self.assertEqual(context["span_source"], "structural_evidence_windows")
        self.assertEqual(len(context["evidence_spans"]), 3)

    def test_semantic_mismatch_chunk_prescription_uses_evidence_grounding(self):
        finding = make_finding("p1", "retrieval_semantic_mismatch")
        state = AgentDoctorState(index_config={
            "chunk_size": 800,
            "chunk_candidate_policy": self._chunk_policy(),
        })

        space, metadata = planner._grounded_search_space(
            finding.label,
            [finding],
            state,
            {"chunk_size": "decrease"},
            self._evidence_analysis(),
        )

        self.assertEqual(space, {"chunk_size": [750, 650, 600]})
        self.assertEqual(metadata["status"], "grounded")
        self.assertEqual(metadata["source"], "structural_evidence_windows")

    def test_chunk_candidate_policy_rejects_invalid_safety_bounds(self):
        base_policy = {
            "target_quantile": 0.85,
            "margin_ratio": 0.20,
            "rounding_step": 50,
            "path_fractions": [0.33, 0.66, 1.0],
            "candidate_count": 3,
            "min_span_count": 3,
            "max_step_ratio": 0.25,
            "min_chunk_size": 200,
            "max_chunk_size": 1500,
        }
        invalid_overrides = [
            {"max_step_ratio": 0.75},
            {"max_step_ratio": True},
            {"min_chunk_size": 0},
            {"max_chunk_size": 100},
        ]

        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                state = AgentDoctorState(index_config={
                    "chunk_candidate_policy": {
                        **base_policy,
                        **overrides,
                    }
                })
                policy, error = planner._chunk_candidate_policy(state)
                self.assertIsNone(policy)
                self.assertIn("유효하지 않음", error)

    def test_chunk_candidate_policy_allows_single_candidate_budget(self):
        state = AgentDoctorState(index_config={
            "chunk_candidate_policy": {
                **self._chunk_policy(),
                "path_fractions": [1.0],
                "candidate_count": 1,
            }
        })

        policy, error = planner._chunk_candidate_policy(state)

        self.assertIsNone(error)
        self.assertEqual(policy["candidate_count"], 1)
        state.index_config["chunk_size"] = 800
        values, metadata = planner._ground_chunk_size_candidates(
            [],
            state,
            "decrease",
            self._evidence_analysis(),
        )
        self.assertEqual(values, [600])
        self.assertEqual(metadata["status"], "grounded")

    def test_state_evidence_window_policy_controls_candidate_measurements(self):
        finding = make_finding("p1", "too_long_context")
        state = AgentDoctorState(
            report=_report(finding),
            documents=[Document("d1", "memory", "txt", "가" * 3000)],
            probes=[
                Probe(
                    probe_id="p1",
                    question="질문",
                    source="taxonomy",
                    answer_exists=True,
                    gold_spans=[
                        {"doc_id": "d1", "start": 100, "end": 110},
                        {"doc_id": "d1", "start": 1000, "end": 1010},
                        {"doc_id": "d1", "start": 2000, "end": 2010},
                    ],
                )
            ],
            index_config={
                "chunk_size": 800,
                "chunk_overlap": 50,
                "evidence_window_policy": {
                    "min_chars": 100,
                    "max_chars": 120,
                    "heading_max_distance": 0,
                    "adjacent_context_blocks": 0,
                },
                "chunk_candidate_policy": {
                    "target_quantile": 0.85,
                    "margin_ratio": 0.20,
                    "rounding_step": 50,
                    "path_fractions": [0.33, 0.66, 1.0],
                    "candidate_count": 3,
                    "min_span_count": 3,
                },
            },
        )

        request, _decision = planner.plan(
            state,
            blacklist=self._chunk_blacklist(),
        )

        grounding = request.metadata["candidate_grounding"]
        self.assertEqual(grounding["status"], "grounded")
        self.assertEqual(grounding["max"], 120)
        self.assertEqual(grounding["span_count"], 3)
        self.assertTrue(
            all(value >= 600 for value in request.search_space["chunker.chunk_size"])
        )

    def test_invalid_evidence_window_policy_has_explicit_status(self):
        finding = make_finding("p1", "too_long_context")
        state = AgentDoctorState(
            report=_report(finding),
            documents=[Document("d1", "memory", "txt", "가" * 1000)],
            probes=[
                Probe(
                    probe_id="p1",
                    question="질문",
                    source="taxonomy",
                    answer_exists=True,
                    gold_spans=[{"doc_id": "d1", "start": 10, "end": 20}],
                )
            ],
            index_config={
                "chunk_size": 800,
                "chunk_overlap": 50,
                "evidence_window_policy": {
                    "min_chars": 200,
                    "max_chars": 100,
                },
            },
        )

        request, _decision = planner.plan(
            state,
            blacklist=self._chunk_blacklist(),
        )

        self.assertEqual(request.search_space, {"chunker.chunk_size": [400]})
        self.assertEqual(
            request.metadata["candidate_grounding"]["status"],
            "invalid_evidence_window_policy",
        )

    def test_non_dict_evidence_window_policy_is_rejected(self):
        finding = make_finding("p1", "too_long_context")
        state = AgentDoctorState(
            report=_report(finding),
            documents=[Document("d1", "memory", "txt", "가" * 1000)],
            probes=[
                Probe(
                    probe_id="p1",
                    question="질문",
                    source="taxonomy",
                    answer_exists=True,
                    gold_spans=[{"doc_id": "d1", "start": 10, "end": 20}],
                )
            ],
            index_config={
                "chunk_size": 800,
                "chunk_overlap": 50,
                "evidence_window_policy": "invalid",
            },
        )

        request, _decision = planner.plan(
            state,
            blacklist=self._chunk_blacklist(),
        )

        self.assertEqual(
            request.metadata["candidate_grounding"]["status"],
            "invalid_evidence_window_policy",
        )

    def test_plan_reuses_evidence_analysis_for_candidates_and_precheck(self):
        finding = make_finding(
            "p1",
            "too_long_context",
            candidates={"chunker.chunk_size": [400, 600]},
        )
        state = AgentDoctorState(
            report=_report(finding),
            documents=[Document("d1", "memory", "txt", "가" * 1000)],
            probes=[
                Probe(
                    probe_id="p1",
                    question="질문",
                    source="taxonomy",
                    answer_exists=True,
                    gold_spans=[{"doc_id": "d1", "start": 100, "end": 180}],
                )
            ],
            index_config={"chunk_size": 800, "chunk_overlap": 50},
        )

        with patch.object(
            planner,
            "build_evidence_windows",
            wraps=planner.build_evidence_windows,
        ) as build_windows:
            planner.plan(state, blacklist=self._chunk_blacklist())

        build_windows.assert_called_once()

    def test_chunk_symbolic_fallback_uses_status_allowlist(self):
        self.assertTrue(planner._allows_symbolic_fallback(
            "chunker.chunk_size",
            {"status": "missing_gold_spans", "source": "anything"},
        ))
        self.assertFalse(planner._allows_symbolic_fallback(
            "chunker.chunk_size",
            {"status": "insufficient_spans"},
        ))
        self.assertFalse(planner._allows_symbolic_fallback(
            "chunker.chunk_size",
            {"status": "direction_conflict"},
        ))


class TopicClusterSignalDeferredTest(unittest.TestCase):
    """topic_cluster 신호 소비가 꺼진 현재 동작 — 관측용 신호로만 유지.

    planner._CONSUME_TOPIC_CLUSTER_SIGNAL=False 인 동안 _available_prescriptions 는
    findings 를 줘도 applies_when 을 보지 않고, 신호값과 무관하게 전 처방을 순서대로
    돌려줘야 한다(신호 배선 이전 = 순차 fallback 과 동작 동일). 소비를 켰을 때의 대조
    로직 계약은 아래 TopicClusterAppliesWhenConsumeOnTest 가 별도로 고정한다.
    """

    LABEL = "retrieval_semantic_mismatch"

    def _rule(self):
        from agents.optimize.rules import LABEL_TO_PRESCRIPTIONS
        return LABEL_TO_PRESCRIPTIONS[self.LABEL]

    def _finding_with_signal(self, signal):
        f = make_finding("p0", self.LABEL)
        if signal is not None:
            f.metadata["topic_cluster"] = signal
        return f

    def _ids(self, signal):
        rule = self._rule()
        avail = planner._available_prescriptions(
            rule, self.LABEL, set(), [self._finding_with_signal(signal)]
        )
        return [p["id"] for p in avail]

    def test_consume_flag_is_off(self):
        # 이 PR 이 착지시키는 상태 = 소비 OFF. 켜지면 아래 회귀들이 의미를 잃으므로
        # 플래그 자체를 명시적으로 고정한다(소비를 켤 때 이 테스트가 먼저 걸린다).
        self.assertFalse(planner._CONSUME_TOPIC_CLUSTER_SIGNAL)

    def test_all_signals_keep_all_prescriptions(self):
        # 신호값이 무엇이든(소비 OFF) 전 처방이 순서대로 통과해야 한다.
        rule = self._rule()
        all_ids = [p["id"] for p in rule["prescriptions"]]
        for signal in ("concentrated", "spread", "none", "unmeasured", None):
            with self.subTest(signal=signal):
                self.assertEqual(self._ids(signal), all_ids)

    def test_no_findings_arg_is_legacy_blacklist_only(self):
        # findings 를 안 주는 레거시 호출도 블랙리스트만 본다(소비 OFF 와 결과 동일).
        rule = self._rule()
        ids = [p["id"] for p in planner._available_prescriptions(rule, self.LABEL, set())]
        self.assertEqual(ids, [p["id"] for p in rule["prescriptions"]])

    def test_blacklist_still_filters_under_deferred_consume(self):
        # 소비가 꺼져 있어도 블랙리스트는 계속 유효하다(신호와 무관한 기존 경로).
        rule = self._rule()
        blacklist = {(self.LABEL, "swap_embedding_model")}
        finding = self._finding_with_signal("spread")
        ids = [
            p["id"]
            for p in planner._available_prescriptions(
                rule, self.LABEL, blacklist, [finding]
            )
        ]
        self.assertNotIn("swap_embedding_model", ids)
        self.assertIn("shrink_chunk_size", ids)


class TopicClusterAppliesWhenConsumeOnTest(unittest.TestCase):
    """소비를 켰을 때(_CONSUME_TOPIC_CLUSTER_SIGNAL=True)의 applies_when 대조 계약.

    소비는 현재 꺼져 있지만 대조 로직(_prescription_applies)과 완화 경로는 그대로 배선돼
    있다. 향후 캘리브레이션·임베딩 교체가 준비돼 소비를 켤 때 이 계약이 깨지지 않도록,
    플래그를 켠 상태로 고정해 회귀를 잡아둔다.

    rules.py 계약: spread/concentrated → swap_embedding_model 만, none → 청킹 처방만,
    신호 미측정/unmeasured → 셋 다(순차 fallback).
    """

    LABEL = "retrieval_semantic_mismatch"

    def setUp(self):
        self._saved = planner._CONSUME_TOPIC_CLUSTER_SIGNAL
        planner._CONSUME_TOPIC_CLUSTER_SIGNAL = True

    def tearDown(self):
        planner._CONSUME_TOPIC_CLUSTER_SIGNAL = self._saved

    def _rule(self):
        from agents.optimize.rules import LABEL_TO_PRESCRIPTIONS
        return LABEL_TO_PRESCRIPTIONS[self.LABEL]

    def _finding_with_signal(self, signal):
        f = make_finding("p0", self.LABEL)
        if signal is not None:
            f.metadata["topic_cluster"] = signal
        return f

    def _ids(self, signal):
        rule = self._rule()
        avail = planner._available_prescriptions(
            rule, self.LABEL, set(), [self._finding_with_signal(signal)]
        )
        return [p["id"] for p in avail]

    def test_concentrated_keeps_only_embedding_swap(self):
        self.assertEqual(self._ids("concentrated"), ["swap_embedding_model"])

    def test_spread_keeps_only_embedding_swap(self):
        self.assertEqual(self._ids("spread"), ["swap_embedding_model"])

    def test_none_keeps_only_chunking(self):
        # none → 청킹 처방만 남고 임베딩 교체는 빠진다. 청킹 전략 교체는 main 에서
        # 2-후보 스윕(switch_to_recursive_sentence / switch_to_markdown_recursive)으로
        # 분화됐으므로, 특정 id 를 박지 말고 "swap 을 뺀 나머지 = 청킹 처방 전부"인지 본다.
        ids = self._ids("none")
        self.assertNotIn("swap_embedding_model", ids)
        self.assertIn("shrink_chunk_size", ids)
        rule = self._rule()
        chunking_ids = [
            p["id"] for p in rule["prescriptions"] if p["id"] != "swap_embedding_model"
        ]
        self.assertEqual(ids, chunking_ids)

    def test_missing_signal_keeps_all_prescriptions(self):
        # 신호 미측정 → 전부 통과 = 기존 순차 fallback (동작 불변).
        rule = self._rule()
        all_ids = [p["id"] for p in rule["prescriptions"]]
        self.assertEqual(self._ids(None), all_ids)

    def test_prescription_without_applies_when_always_passes(self):
        # applies_when 이 없는 라벨(예: retrieval_lexical_mismatch)은 신호와 무관하게 통과.
        from agents.optimize.rules import LABEL_TO_PRESCRIPTIONS
        label = "retrieval_lexical_mismatch"
        rule = LABEL_TO_PRESCRIPTIONS[label]
        f = make_finding("p0", label)
        f.metadata["topic_cluster"] = "concentrated"   # 무관 신호
        avail = planner._available_prescriptions(rule, label, set(), [f])
        self.assertEqual(len(avail), len(rule["prescriptions"]))

    def test_unmeasured_signal_keeps_all_prescriptions(self):
        # 판정 불가(unmeasured)는 어느 허용 리스트에도 없지만, 완화 경로로 전부 통과해야
        # 한다 — 근거가 없을 때는 신호 배선 이전의 순차 fallback 과 같아야 하기 때문.
        rule = self._rule()
        self.assertEqual(self._ids("unmeasured"), [p["id"] for p in rule["prescriptions"]])

    def test_signal_relaxes_when_blacklist_exhausts_preferred(self):
        """신호가 고른 처방이 블랙리스트에 걸리면 나머지로 완화된다(라벨 스킵 금지).

        spread + swap_embedding_model 블랙리스트 조합. 완화가 없으면 후보가 []가 되어
        _pick_top 이 라벨을 통째로 건너뛴다 — 신호 배선 이전에는 청킹 처방으로 넘어가던
        경로라 회귀다.
        """
        rule = self._rule()
        blacklist = {(self.LABEL, "swap_embedding_model")}
        finding = self._finding_with_signal("spread")

        avail = planner._available_prescriptions(rule, self.LABEL, blacklist, [finding])
        ids = [p["id"] for p in avail]
        self.assertNotIn("swap_embedding_model", ids)      # 블랙리스트는 계속 유효
        self.assertIn("shrink_chunk_size", ids)            # 신호 조건은 완화됨

        ranked = [(self.LABEL, [finding], rule, 1.0)]
        self.assertIsNotNone(planner._pick_top(ranked, blacklist))

    def test_blacklist_still_skips_label_when_fully_exhausted(self):
        # 완화는 신호에만 적용된다 — 블랙리스트로 전부 소진되면 라벨은 그대로 스킵.
        rule = self._rule()
        blacklist = {(self.LABEL, p["id"]) for p in rule["prescriptions"]}
        finding = self._finding_with_signal("spread")

        self.assertEqual(
            planner._available_prescriptions(rule, self.LABEL, blacklist, [finding]), []
        )
        ranked = [(self.LABEL, [finding], rule, 1.0)]
        self.assertIsNone(planner._pick_top(ranked, blacklist))


if __name__ == "__main__":
    unittest.main()
