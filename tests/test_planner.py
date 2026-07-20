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

    def test_single_candidate_uses_rules_backend(self):
        # 후보가 하나뿐이면 sweep 할 게 없어 rules 로 1회 검증한다.
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
        self.assertEqual(request.search_space, {"retriever.top_k": [3, 7, 9]})
        self.assertEqual(request.max_trials, 3)

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
        self.assertEqual(context["gold_spans"][0]["start"], 100)


if __name__ == "__main__":
    unittest.main()
