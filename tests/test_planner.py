"""
tests/test_planner.py
planner 의 라벨 묶음 처리 + 근거값 계산 검증.

핵심 전제: Eval 은 Finding 을 probe 마다 따로 만든다(affected_probes 는 항상 1개).
같은 원인이 probe N개에서 터지면 Finding 도 N개다. planner 는 이를 라벨로 묶어
점수(빈도)와 근거값(측정 기반 목표값)을 계산해야 한다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import DiagnosticReport, Finding
from core.state import AgentDoctorState
from agents.optimize import planner
from agents.optimize.planner import _knee


def make_finding(probe_id, label, gold_n=0, confirmed=True):
    """probe 1개에서 나온 Finding 하나. gold_n = 그 probe 가 필요로 하는 gold 청크 수."""
    return Finding(
        finding_id=f"{probe_id}:{label}",
        type="retrieval_failure",
        severity="warning",
        description=label,
        label=label,
        confirmed=confirmed,
        affected_chunks=[f"c{i}" for i in range(gold_n)],
        affected_probes=[probe_id],
    )


def make_state(findings, top_k=5):
    return AgentDoctorState(
        report=DiagnosticReport(
            report_id="r", findings=findings, overall_score=0.5,
            ragas_scores={}, pass_threshold=False,
        ),
        index_config={"chunk_size": 512, "chunk_overlap": 50, "top_k": top_k},
        iteration=0, max_iterations=3,
    )


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
        # ×2 추측이면 10 이 나왔을 것. 측정 기반 무릎은 8.
        self.assertEqual(first.search_space, {"top_k": [8]})

    def test_falls_back_to_direction_keyword_without_evidence(self):
        # gold 개수가 없으면(affected_chunks 비어있음) 계산 불가 → ×2 폴백
        findings = [
            make_finding("p1", "retrieval_incomplete_enumeration", gold_n=0)
        ]
        request, _decision = planner.plan(make_state(findings, top_k=5))

        self.assertEqual(request.candidates[0].search_space, {"top_k": [10]})


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


if __name__ == "__main__":
    unittest.main()
