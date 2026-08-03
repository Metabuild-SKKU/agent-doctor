"""
tests/test_action_aggregator.py
Finding → action 후보 집계·점수·충돌·정렬 검증 (계획서 단계 3).

전환의 핵심 계층이라 세 가지를 집중해서 본다.
  1. 같은 실제 config 변경이 하나로 통합되는가 (전환의 목적)
  2. 고유 probe 단위로만 투표되는가 (라벨을 쪼개도 점수가 부풀지 않아야 한다)
  3. 실행 불가 action 이 점수 경쟁에서 밀려나지 않는가 (starvation 방지)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.optimize import action_aggregator as aggregator
from agents.optimize import action_catalog, candidate_values, history
from agents.optimize.schemas import ActionCandidate, ActionSupport
from tests.test_planner import make_finding, make_state


def _grouped(findings):
    grouped = {}
    for finding in findings:
        grouped.setdefault(finding.label, []).append(finding)
    return grouped


def _pipeline(findings, state=None, backend="rules"):
    """Finding 부터 정렬까지 한 번에 돌려 중간 결과를 모두 돌려준다."""
    state = state or make_state(findings)

    def space_for(label, group, changes):
        return candidate_values._finding_search_space(group, changes, state, None)

    supports = aggregator.build_action_supports(
        _grouped(findings), state, search_space_for=space_for
    )
    candidates = aggregator.aggregate_action_candidates(supports, state)
    eligible, rejected = aggregator.filter_ineligible_actions(
        candidates, state, backend=backend
    )
    kept, deferred = aggregator.resolve_action_conflicts(eligible)
    return {
        "supports": supports,
        "candidates": candidates,
        "eligible": eligible,
        "rejected": rejected,
        "kept": kept,
        "deferred": deferred,
        "ranked": aggregator.rank_action_candidates(kept),
        "selected": aggregator.select_action(kept),
    }


def _support(action_key, label, probes, *, group="A", confidence=1.0, grounded=False):
    return ActionSupport(
        action_key=action_key,
        label=label,
        group=group,
        affected_probes=set(probes),
        confidence=confidence,
        grounding_metadata={"status": "grounded"} if grounded else {},
    )


def _candidate(action_key, supports):
    definition = action_catalog.get_action(action_key)
    candidate = ActionCandidate(
        action_key=action_key,
        definition=definition,
        supports=list(supports),
        supporting_labels=sorted({s.label for s in supports}),
        supporting_probes={p for s in supports for p in s.affected_probes},
    )
    candidate.score, candidate.score_breakdown = aggregator.score_candidate(candidate)
    return candidate


class ActionMergingTest(unittest.TestCase):
    """같은 실제 config 변경은 이름이 달라도 하나로 모인다 — 전환의 목적."""

    def test_two_labels_supporting_same_change_merge_into_one_candidate(self):
        findings = [
            make_finding("p0", "retrieval_missing_gold", gold_ranks={"g": 12}),
            make_finding("q0", "retrieval_incomplete_enumeration", gold_ranks={"g": 12}),
        ]
        result = _pipeline(findings)
        merged = [
            c for c in result["candidates"]
            if c.action_key == "retriever.top_k:increase"
        ]
        self.assertEqual(len(merged), 1, "같은 변경이 두 후보로 남았다")
        self.assertEqual(
            sorted(merged[0].supporting_labels),
            ["retrieval_incomplete_enumeration", "retrieval_missing_gold"],
        )

    def test_merged_candidate_unions_probes_and_metrics(self):
        findings = [
            make_finding("p0", "retrieval_missing_gold", gold_ranks={"g": 12}),
            make_finding("q0", "retrieval_incomplete_enumeration", gold_ranks={"g": 12}),
        ]
        result = _pipeline(findings)
        merged = next(
            c for c in result["candidates"]
            if c.action_key == "retriever.top_k:increase"
        )
        self.assertEqual(merged.supporting_probes, {"p0", "q0"})
        self.assertIn("context_recall", merged.target_metrics)

    def test_candidate_records_value_provenance(self):
        """어떤 라벨이 어떤 값을 제안했는지 남는다(리포트·디버깅용)."""
        findings = [make_finding("p0", "retrieval_missing_gold", gold_ranks={"g": 12})]
        result = _pipeline(findings)
        merged = next(
            c for c in result["candidates"]
            if c.action_key == "retriever.top_k:increase"
        )
        self.assertIn("candidate_support", merged.metadata)


class ProbeVotingTest(unittest.TestCase):
    """고유 probe 단위 투표 (계획서 §4.1)."""

    def test_same_probe_counted_once_regardless_of_finding_count(self):
        """같은 probe 에서 Finding 이 여러 개 나와도 한 표다."""
        one_probe = _candidate(
            "retriever.top_k:increase",
            [_support("retriever.top_k:increase", f"label{i}", {"p1"}) for i in range(5)],
        )
        two_probes = _candidate(
            "retriever.top_k:increase",
            [_support("retriever.top_k:increase", "label", {"p1", "p2"})],
        )
        self.assertLess(one_probe.score, two_probes.score)

    def test_probe_contributes_max_confidence_not_sum(self):
        """한 probe 가 여러 support 로 지지해도 최대 confidence 한 번만 센다."""
        candidate = _candidate(
            "retriever.top_k:increase",
            [
                _support("retriever.top_k:increase", "a", {"p1"}, confidence=0.4),
                _support("retriever.top_k:increase", "b", {"p1"}, confidence=0.9),
            ],
        )
        self.assertAlmostEqual(candidate.score_breakdown["weighted_probe_support"], 0.9)
        self.assertEqual(candidate.score_breakdown["supporting_probe_count"], 1)

    def test_score_divides_by_action_own_cost(self):
        """비용은 action 자신의 것이다 — 라벨의 첫 처방이 아니다."""
        cheap = _candidate(
            "retriever.top_k:increase",
            [_support("retriever.top_k:increase", "a", {"p1", "p2", "p3"})],
        )
        costly = _candidate(
            "chunker.chunk_size:increase",
            [_support("chunker.chunk_size:increase", "a", {"p1", "p2", "p3"})],
        )
        self.assertEqual(cheap.score, 3.0)      # 3 probe / cost 1
        self.assertEqual(costly.score, 1.0)     # 3 probe / cost 3

    def test_breakdown_records_cost_and_confidence_source(self):
        """나중에 실측 confidence 가 들어올 때 이력에서 구분할 수 있어야 한다."""
        candidate = _candidate(
            "retriever.top_k:increase",
            [_support("retriever.top_k:increase", "a", {"p1"})],
        )
        self.assertEqual(candidate.score_breakdown["cost_source"], "reindex_flag")
        self.assertEqual(candidate.score_breakdown["confidence_source"], "default")


class CausalTierTest(unittest.TestCase):
    """A > C > B 는 점수보다 먼저다 (계획서 §4.3 불변조건)."""

    def test_a_group_beats_b_group_with_higher_score(self):
        a = _candidate(
            "retriever.top_k:increase",
            [_support("retriever.top_k:increase", "a", {"p1"}, group="A")],
        )
        b = _candidate(
            "generation.require_citation:enable",
            [_support("generation.require_citation:enable", "b",
                      {f"q{i}" for i in range(9)}, group="B")],
        )
        self.assertGreater(b.score, a.score, "전제: B 가 점수는 더 높다")
        self.assertEqual(aggregator.rank_action_candidates([b, a])[0].action_key,
                         a.action_key)

    def test_a_group_beats_c_group(self):
        a = _candidate(
            "retriever.top_k:increase",
            [_support("retriever.top_k:increase", "a", {"p1"}, group="A")],
        )
        c = _candidate(
            "retriever.top_k:decrease",
            [_support("retriever.top_k:decrease", "c",
                      {f"q{i}" for i in range(5)}, group="C")],
        )
        self.assertEqual(aggregator.rank_action_candidates([c, a])[0].action_key,
                         a.action_key)

    def test_causal_rank_uses_highest_tier_among_supporters(self):
        """A·C 가 함께 지지하면 더 높은 A 로 판정한다."""
        candidate = _candidate(
            "chunker.chunk_size:decrease",
            [
                _support("chunker.chunk_size:decrease", "c1", {"p1"}, group="C"),
                _support("chunker.chunk_size:decrease", "a1", {"p2"}, group="A"),
            ],
        )
        self.assertEqual(candidate.causal_rank_group, "A")


class DeterministicRankingTest(unittest.TestCase):
    """동일 입력은 항상 같은 선택을 낸다 (계획서 §7.2 invariant 6)."""

    def _tied_pair(self):
        return [
            _candidate("retriever.top_k:increase",
                       [_support("retriever.top_k:increase", "a", {"p1"})]),
            _candidate("retriever.mmr:enable",
                       [_support("retriever.mmr:enable", "b", {"p2"})]),
        ]

    def test_tie_broken_by_action_key(self):
        first = aggregator.rank_action_candidates(self._tied_pair())[0].action_key
        reversed_order = list(reversed(self._tied_pair()))
        self.assertEqual(
            aggregator.rank_action_candidates(reversed_order)[0].action_key, first
        )

    def test_repeated_ranking_is_stable(self):
        first = aggregator.rank_action_candidates(self._tied_pair())[0].action_key
        for _ in range(5):
            self.assertEqual(
                aggregator.rank_action_candidates(self._tied_pair())[0].action_key,
                first,
            )


class EligibilityFilterTest(unittest.TestCase):
    """실행 불가 action 은 경쟁에서 빠진다 (starvation 방지)."""

    def test_catalog_blocked_action_is_rejected(self):
        findings = [make_finding("p0", "retrieval_missing_gold", gold_ranks={"g": 12})]
        result = _pipeline(findings)
        rejected = {c.action_key: c.reason for c in result["rejected"]}
        self.assertEqual(rejected.get("query_rewrite:replace"), "catalog_blocked")

    def test_action_without_usable_candidate_is_rejected(self):
        """근거값이 방향과 맞지 않아 후보가 비면 제외된다."""
        findings = [make_finding("p0", "retrieval_missing_gold", gold_n=3)]
        state = make_state(findings, top_k=5)   # 필요 top_k(3) < 현재(5) → 늘릴 근거 없음
        result = _pipeline(findings, state=state)
        rejected = {c.action_key: c.reason for c in result["rejected"]}
        self.assertEqual(rejected.get("retriever.top_k:increase"), "no_candidate_value")

    def test_eligible_candidates_carry_filtered_search_space(self):
        findings = [make_finding("p0", "retrieval_missing_gold", gold_ranks={"g": 12})]
        result = _pipeline(findings)
        for candidate in result["eligible"]:
            with self.subTest(action=candidate.action_key):
                self.assertEqual(len(candidate.search_space), 1)
                self.assertTrue(next(iter(candidate.search_space.values())))

    def test_selection_never_returns_blocked_action(self):
        findings = [make_finding("p0", "retrieval_missing_gold", gold_ranks={"g": 12})]
        result = _pipeline(findings)
        if result["selected"]:
            self.assertEqual(result["selected"].status, "ready")


class ConflictResolutionTest(unittest.TestCase):
    """반대 방향 충돌 (계획서 §4.4)."""

    def _opposing(self, top_probes, other_probes, *, top_group="A", other_group="A"):
        increase = _candidate(
            "chunker.chunk_size:increase",
            [_support("chunker.chunk_size:increase", "inc",
                      top_probes, group=top_group)],
        )
        decrease = _candidate(
            "chunker.chunk_size:decrease",
            [_support("chunker.chunk_size:decrease", "dec",
                      other_probes, group=other_group)],
        )
        return [increase, decrease]

    def test_close_conflict_defers_the_axis(self):
        """근소한 차이면 그 축을 보류한다 — 왕복 방지."""
        peers = self._opposing({"p1", "p2", "p3"}, {"q1", "q2"})
        kept, deferred = aggregator.resolve_action_conflicts(peers)
        self.assertEqual(kept, [])
        self.assertEqual(len(deferred), 1)
        self.assertEqual(deferred[0]["reason"], "conflict_margin_unmet")

    def test_clear_margin_picks_the_winner(self):
        """상대 20% + 절대 probe 2 를 모두 넘으면 우세 쪽을 고른다."""
        peers = self._opposing({f"p{i}" for i in range(6)}, {"q1", "q2"})
        kept, deferred = aggregator.resolve_action_conflicts(peers)
        self.assertEqual([c.action_key for c in kept],
                         ["chunker.chunk_size:increase"])
        self.assertEqual(deferred, [])

    def test_different_tier_resolves_without_margin(self):
        """tier 가 다르면 점수를 보지 않는다."""
        peers = self._opposing({"p1"}, {f"q{i}" for i in range(9)},
                               top_group="A", other_group="C")
        kept, _deferred = aggregator.resolve_action_conflicts(peers)
        self.assertEqual([c.action_key for c in kept],
                         ["chunker.chunk_size:increase"])

    def test_grounded_side_wins_at_same_tier(self):
        """같은 tier 면 실측 근거가 있는 쪽이 이긴다."""
        increase = _candidate(
            "chunker.chunk_size:increase",
            [_support("chunker.chunk_size:increase", "inc", {"p1"}, grounded=True)],
        )
        decrease = _candidate(
            "chunker.chunk_size:decrease",
            [_support("chunker.chunk_size:decrease", "dec", {"q1", "q2"})],
        )
        kept, _deferred = aggregator.resolve_action_conflicts([increase, decrease])
        self.assertEqual([c.action_key for c in kept],
                         ["chunker.chunk_size:increase"])

    def test_deferred_records_both_sides_for_report(self):
        peers = self._opposing({"p1", "p2", "p3"}, {"q1", "q2"})
        _kept, deferred = aggregator.resolve_action_conflicts(peers)
        keys = {c["action_key"] for c in deferred[0]["candidates"]}
        self.assertEqual(
            keys, {"chunker.chunk_size:increase", "chunker.chunk_size:decrease"}
        )

    def test_single_action_on_axis_is_not_a_conflict(self):
        single = [
            _candidate("retriever.top_k:increase",
                       [_support("retriever.top_k:increase", "a", {"p1"})])
        ]
        kept, deferred = aggregator.resolve_action_conflicts(single)
        self.assertEqual(len(kept), 1)
        self.assertEqual(deferred, [])


class BooleanAxisExclusionTest(unittest.TestCase):
    """boolean 축은 현재 상태가 한쪽을 no-op 으로 만들어 자동 배타된다 (§4.4).

    두 action 이 동시에 링에 오르지 않으므로 충돌 정책이 발동하지 않아야 한다.
    발동하면 2:1 지지에서 절대 조건(probe 2)에 걸려 축이 영구히 닫힌다.
    """

    def test_only_one_direction_survives_eligibility(self):
        findings = [
            make_finding("p0", "retrieval_low_rank"),
            make_finding("p1", "retrieval_reranker_demotion"),
        ]
        state = make_state(findings)
        state.index_config["use_reranker"] = False   # 꺼짐 → enable 만 유효
        state.runtime_capabilities = {"reranker": {"status": "verified"}}
        result = _pipeline(findings, state=state)

        keys = {c.action_key for c in result["eligible"]}
        self.assertNotIn("reranker.enabled:disable", keys,
                         "꺼진 상태에서 disable 은 no-op 이라 남으면 안 된다")

    def test_boolean_axis_does_not_reach_conflict_policy(self):
        findings = [
            make_finding("p0", "retrieval_low_rank"),
            make_finding("p1", "retrieval_reranker_demotion"),
        ]
        state = make_state(findings)
        state.index_config["use_reranker"] = False
        state.runtime_capabilities = {"reranker": {"status": "verified"}}
        result = _pipeline(findings, state=state)

        axes = {d["axis"] for d in result["deferred"]}
        self.assertNotIn("reranker.enabled", axes,
                         "boolean 축이 충돌 보류에 걸리면 영구 교착이 된다")


class SupportConstructionTest(unittest.TestCase):
    """support 생성 규칙."""

    def test_draft_label_produces_no_support(self):
        findings = [make_finding("p0", "reranker_low_recall")]
        supports = aggregator.build_action_supports(_grouped(findings),
                                                    make_state(findings))
        self.assertEqual(supports, [])

    def test_manual_label_produces_no_support(self):
        findings = [make_finding("p0", "corpus_gap")]
        supports = aggregator.build_action_supports(_grouped(findings),
                                                    make_state(findings))
        self.assertEqual(supports, [])

    def test_support_carries_probe_set_not_finding_count(self):
        findings = [
            make_finding("p0", "retrieval_missing_gold", gold_ranks={"g": 12}),
            make_finding("p0", "retrieval_missing_gold", gold_ranks={"g": 13}),
        ]
        supports = aggregator.build_action_supports(_grouped(findings),
                                                    make_state(findings))
        for support in supports:
            with self.subTest(action=support.action_key):
                self.assertEqual(support.affected_probes, {"p0"})


class AppliesWhenSignalDeferredTest(unittest.TestCase):
    """applies_when(topic_cluster) 신호 소비가 꺼진 현재 동작 — 관측용으로만 유지.

    planner 에서 이관했다(계획서 §3.3: "applies_when 을 만족하지 않는 support 는
    생성하지 않는다"는 이 계층의 책임). 소비가 꺼진 동안에는 신호값과 무관하게 전
    처방이 support 를 만들어야 한다 — 신호 배선 이전과 동작이 같아야 한다.
    """

    LABEL = "retrieval_semantic_mismatch"

    def _rule(self):
        from agents.optimize.rules import LABEL_TO_PRESCRIPTIONS
        return LABEL_TO_PRESCRIPTIONS[self.LABEL]

    def _finding_with_signal(self, signal):
        finding = make_finding("p0", self.LABEL)
        if signal is not None:
            finding.metadata["topic_cluster"] = signal
        return finding

    def _ids(self, signal):
        return [
            p["id"]
            for p in aggregator._applicable_prescriptions(
                self._rule(), [self._finding_with_signal(signal)]
            )
        ]

    def test_consume_flag_is_off(self):
        # 이 전환이 착지시키는 상태 = 소비 OFF. 켜지면 아래 회귀들이 의미를 잃으므로
        # 플래그 자체를 고정한다(소비를 켤 때 이 테스트가 먼저 걸린다).
        self.assertFalse(aggregator.CONSUME_APPLIES_WHEN_SIGNAL)

    def test_all_signals_keep_all_prescriptions(self):
        all_ids = [p["id"] for p in self._rule()["prescriptions"]]
        for signal in ("concentrated", "spread", "none", "unmeasured", None):
            with self.subTest(signal=signal):
                self.assertEqual(self._ids(signal), all_ids)

    def test_supports_are_built_for_every_prescription(self):
        """소비 OFF 에서는 신호가 support 생성을 막지 않는다."""
        findings = [self._finding_with_signal("spread")]
        state = make_state(findings)

        supports = aggregator.build_action_supports(_grouped(findings), state)

        # swap_embedding_model 은 신호가 '선호'하는 처방이지만, 신호를 보지 않으므로
        # 청킹 처방들도 그대로 support 를 만든다.
        self.assertIn(
            "chunker.chunk_size:decrease",
            {support.action_key for support in supports},
        )


class AppliesWhenSignalConsumeOnTest(unittest.TestCase):
    """소비를 켰을 때의 applies_when 대조 계약.

    소비는 꺼져 있지만 대조 로직과 완화 경로는 그대로 배선돼 있다. 캘리브레이션이
    끝나 켤 때 이 계약이 깨지지 않도록 플래그를 켠 상태로 고정해 회귀를 잡는다.

    rules.py 계약: spread/concentrated → swap_embedding_model 만,
    none → 청킹 처방만, 신호 미측정/unmeasured → 셋 다(순차 fallback).
    """

    LABEL = "retrieval_semantic_mismatch"

    def setUp(self):
        self._saved = aggregator.CONSUME_APPLIES_WHEN_SIGNAL
        aggregator.CONSUME_APPLIES_WHEN_SIGNAL = True

    def tearDown(self):
        aggregator.CONSUME_APPLIES_WHEN_SIGNAL = self._saved

    def _rule(self, label=None):
        from agents.optimize.rules import LABEL_TO_PRESCRIPTIONS
        return LABEL_TO_PRESCRIPTIONS[label or self.LABEL]

    def _finding_with_signal(self, signal, label=None):
        finding = make_finding("p0", label or self.LABEL)
        if signal is not None:
            finding.metadata["topic_cluster"] = signal
        return finding

    def _ids(self, signal):
        return [
            p["id"]
            for p in aggregator._applicable_prescriptions(
                self._rule(), [self._finding_with_signal(signal)]
            )
        ]

    def test_concentrated_keeps_only_embedding_swap(self):
        self.assertEqual(self._ids("concentrated"), ["swap_embedding_model"])

    def test_spread_keeps_only_embedding_swap(self):
        self.assertEqual(self._ids("spread"), ["swap_embedding_model"])

    def test_none_keeps_only_chunking(self):
        ids = self._ids("none")
        self.assertNotIn("swap_embedding_model", ids)
        self.assertIn("shrink_chunk_size", ids)
        chunking_ids = [
            p["id"]
            for p in self._rule()["prescriptions"]
            if p["id"] != "swap_embedding_model"
        ]
        self.assertEqual(ids, chunking_ids)

    def test_missing_signal_keeps_all_prescriptions(self):
        # 신호 미측정 → 전부 통과 = 순차 fallback (동작 불변).
        self.assertEqual(
            self._ids(None), [p["id"] for p in self._rule()["prescriptions"]]
        )

    def test_unmeasured_signal_keeps_all_prescriptions(self):
        # 판정 불가는 어느 허용 리스트에도 없지만 완화 경로로 전부 통과해야 한다 —
        # 근거가 없을 때는 신호 배선 이전과 같아야 하기 때문.
        self.assertEqual(
            self._ids("unmeasured"), [p["id"] for p in self._rule()["prescriptions"]]
        )

    def test_prescription_without_applies_when_always_passes(self):
        label = "retrieval_lexical_mismatch"
        rule = self._rule(label)
        finding = self._finding_with_signal("concentrated", label)   # 무관 신호

        self.assertEqual(
            len(aggregator._applicable_prescriptions(rule, [finding])),
            len(rule["prescriptions"]),
        )

    def test_signal_never_empties_a_label(self):
        """신호가 전부 걸러내면 조건을 완화한다 — 라벨을 통째로 막아선 안 된다.

        신호는 처방 순서를 '선호'하게 만들 뿐이다. 걸러서 0개가 되면 그 라벨이 어떤
        action 도 지지하지 못하고 사라지는데, 임계값이 아직 임의값이라 더욱 위험하다.
        """
        rule = {
            "prescriptions": [
                {"id": "only", "patch": {"top_k": "increase"},
                 "applies_when": {"topic_cluster": ["spread"]}},
            ]
        }
        finding = self._finding_with_signal("none")

        ids = [p["id"] for p in aggregator._applicable_prescriptions(rule, [finding])]

        self.assertEqual(ids, ["only"])


class BlockedAttemptFilterTest(unittest.TestCase):
    """품질 실패는 **정확한 전이 하나**만 막는다 (계획서 §5.1).

    action 통째로 막으면 "top_k 7 이 나빴다"가 "top_k 축을 영원히 닫는다"가 되고,
    상류를 고친 뒤 하류를 재평가한다는 A>C>B 설계 의도와 충돌한다.
    """

    def _candidates(self, findings, state):
        def space_for(label, group, changes):
            return candidate_values._finding_search_space(group, changes, state, None)

        supports = aggregator.build_action_supports(
            _grouped(findings), state, search_space_for=space_for
        )
        return aggregator.aggregate_action_candidates(supports, state)

    def _top_k_action(self, candidates):
        return next(
            candidate
            for candidate in candidates
            if candidate.action_key == "retriever.top_k:increase"
        )

    def test_only_the_blocked_value_is_removed(self):
        findings = [
            make_finding(
                "p1",
                "retrieval_missing_gold",
                candidates={"top_k": [7, 9]},
            )
        ]
        state = make_state(findings, top_k=5)
        blocked = {
            history.build_attempt_key(
                "retriever.top_k:increase", state.index_config, {"retriever.top_k": 7}
            )
        }

        eligible, _rejected = aggregator.filter_ineligible_actions(
            self._candidates(findings, state), state, blocked_attempts=blocked
        )

        self.assertEqual(
            self._top_k_action(eligible).search_space,
            {"retriever.top_k": [9]},
        )

    def test_action_is_dropped_only_when_every_value_is_blocked(self):
        findings = [
            make_finding(
                "p1",
                "retrieval_missing_gold",
                candidates={"top_k": [7, 9]},
            )
        ]
        state = make_state(findings, top_k=5)
        blocked = {
            history.build_attempt_key(
                "retriever.top_k:increase", state.index_config, {"retriever.top_k": value}
            )
            for value in (7, 9)
        }

        eligible, rejected = aggregator.filter_ineligible_actions(
            self._candidates(findings, state), state, blocked_attempts=blocked
        )

        self.assertNotIn(
            "retriever.top_k:increase",
            {candidate.action_key for candidate in eligible},
        )
        self.assertEqual(
            self._top_k_action(rejected).reason,
            aggregator.REASON_BLOCKED_ATTEMPT,
        )

    def test_a_block_from_another_baseline_does_not_apply(self):
        findings = [
            make_finding(
                "p1",
                "retrieval_missing_gold",
                candidates={"top_k": [7, 9]},
            )
        ]
        state = make_state(findings, top_k=5)
        blocked = {
            history.build_attempt_key(
                "retriever.top_k:increase",
                {"top_k": 5, "chunk_size": 256, "chunk_overlap": 50},   # 다른 baseline
                {"retriever.top_k": 7},
            )
        }

        eligible, _rejected = aggregator.filter_ineligible_actions(
            self._candidates(findings, state), state, blocked_attempts=blocked
        )

        self.assertEqual(
            self._top_k_action(eligible).search_space,
            {"retriever.top_k": [7, 9]},
        )


if __name__ == "__main__":
    unittest.main()
