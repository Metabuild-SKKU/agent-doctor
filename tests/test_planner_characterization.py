"""
tests/test_planner_characterization.py
전환 전 planner 선택 결과를 박제한다 (계획서 단계 0).

[이 파일이 하는 일]
  "지금 planner가 어떤 입력에 어떤 선택을 하는가"를 코드로 고정한다. Action-Centered
  전환은 선택 로직을 통째로 바꾸므로, 전환 후 선택이 달라졌을 때 **의도된 변화인지
  회귀인지** 구분할 근거가 필요하다. 그 근거가 이 파일이다.

[⚠️ 동점 입력은 여기에 넣지 않는다 — 가장 중요한 규칙]
  현재 `_group_by_label` 은 dict 삽입 순서에 의존한다. 그런데 계획서 §4.2 는 결정적
  tie-break(`action_key` 사전순 등)를 도입한다. 따라서 **점수가 같은 입력에서는 선택이
  바뀌는 것이 정상**이다. 그런 입력을 여기 박제하면 단계 1·2 의 "기존 동작 변화 없음"
  완료 조건과 충돌해 헛수고를 하게 된다.

    이 파일          : 점수 차이가 분명한 입력만. 전환 후에도 같은 선택이어야 한다.
    TieBreakTest     : 점수가 같은 입력. **선택 내용이 아니라 '결정적인가'만** 본다.

[baseline 측정 시점]
  origin/main 머지 후(#70·#72·#73·#76·#78 반영).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.optimize import planner
from tests.test_planner import make_finding, make_state


def _select(findings, blacklist=None, **state_kwargs):
    """planner.plan 을 돌려 (대표 지지 라벨, 처방 id, search_space) 를 뽑는다.

    전환 후 "대표 라벨"은 실행 의미가 없다 — 여기서는 박제 비교를 위해 지지 라벨의
    첫 원소를 쓴다(인과 등급이 가장 높은 support 가 앞에 온다).
    """
    state = make_state(findings, **state_kwargs)
    request, decision = planner.plan(state, blacklist=blacklist or set())
    if request is None:
        return None, None, None, decision
    return (
        next(iter(request.supporting_labels), None),
        request.prescription_id,
        dict(request.search_space),
        decision,
    )


class GroupPriorityCharacterizationTest(unittest.TestCase):
    """그룹 인과(A > C > B)가 점수보다 먼저 적용된다.

    이것은 전환 후에도 **반드시 유지**되어야 하는 성질이다(계획서 §4.3 불변조건).
    B그룹이 ready 로 승격돼 실제로 A 와 경쟁하게 됐으므로, 점수만으로 정렬하면
    검색이 새는 상태에서 생성 프롬프트를 먼저 손대는 순서 역전이 일어난다.
    """

    def test_a_group_wins_over_c_group_despite_lower_score(self):
        """A그룹은 probe 수가 적어도 C그룹을 이긴다."""
        findings = [
            make_finding("p1", "retrieval_missing_gold", gold_n=3),          # A, probe 1개
            make_finding("p2", "too_long_context"),                          # C, probe 3개
            make_finding("p3", "too_long_context"),
            make_finding("p4", "too_long_context"),
        ]
        label, _prescription, _space, _decision = _select(findings)
        self.assertEqual(label, "retrieval_missing_gold")

    def test_a_group_wins_over_b_group_despite_lower_score(self):
        """A그룹은 B그룹보다 지지가 적어도 먼저 선택된다.

        전환 후 이 성질이 깨지면 §4.3 hard tier 가 무력화된 것이다.
        """
        findings = [
            make_finding("p1", "retrieval_missing_gold", gold_n=3),          # A, probe 1개
            make_finding("p2", "generation_hallucination"),                  # B, probe 3개
            make_finding("p3", "generation_hallucination"),
            make_finding("p4", "generation_hallucination"),
        ]
        label, _prescription, _space, _decision = _select(findings)
        self.assertEqual(label, "retrieval_missing_gold")


class ScoreOrderCharacterizationTest(unittest.TestCase):
    """같은 그룹 안에서는 영향 probe 수가 많은 라벨이 이긴다.

    점수 = 빈도 × 신뢰도 ÷ 비용 이며 `confidence` 가 전부 None(=1.0) 이라 사실상
    probe 수 ÷ 비용이다. 아래 입력은 **점수 차이가 분명해** tie-break 도입에
    영향받지 않는다.
    """

    def test_more_probes_wins_within_same_group(self):
        findings = [
            make_finding("p1", "retrieval_missing_gold", gold_n=3),
            make_finding("p2", "retrieval_incomplete_enumeration", gold_n=3),
            make_finding("p3", "retrieval_incomplete_enumeration", gold_n=3),
            make_finding("p4", "retrieval_incomplete_enumeration", gold_n=3),
        ]
        label, _prescription, _space, _decision = _select(findings)
        self.assertEqual(label, "retrieval_incomplete_enumeration")

    def test_same_label_probes_are_deduplicated_by_probe_id(self):
        """같은 probe 에서 나온 Finding 은 빈도를 부풀리지 않는다.

        전환 후에도 유지되어야 한다(계획서 §4.1 고유 probe 단위 투표).
        """
        many_findings_one_probe = [
            make_finding("p1", "retrieval_missing_gold", gold_n=3) for _ in range(5)
        ]
        two_probes = [
            make_finding("p1", "retrieval_incomplete_enumeration", gold_n=3),
            make_finding("p2", "retrieval_incomplete_enumeration", gold_n=3),
        ]
        label, _prescription, _space, _decision = _select(
            many_findings_one_probe + two_probes
        )
        self.assertEqual(label, "retrieval_incomplete_enumeration")


class LabelNoLongerOwnsExecutionOrderTest(unittest.TestCase):
    """라벨이 처방 실행 순서를 소유하지 않는다 — 전환의 목표 자체다.

    이 클래스는 `PrescriptionOrderCharacterizationTest` 를 대체한다. 그쪽은
    "라벨이 선택되면 그 라벨의 처방을 **선언 순서대로** 시도한다"를 박제했는데,
    그 성질이 사라지는 것이 이 작업의 목적이었다(계획서 §1). 박제를 그대로 두면
    "전환이 일어났다"와 "테스트가 깨졌다"를 구분할 수 없으므로, 같은 입력에 대해
    **새 규칙**을 못박아 둔다.

    새 규칙(계획서 §4.2): causal tier → 점수 → grounded → 비용 → action_key 사전순.
    rules.py 의 선언 순서는 어디에도 들어가지 않는다.
    """

    def test_declaration_order_does_not_decide_the_winner(self):
        """`increase_top_k` 가 먼저 선언됐다는 사실은 선택에 관여하지 않는다.

        같은 라벨의 세 축(top_k·chunk_overlap·chunk_size)이 모두 probe 1개 지지다.
        재색인 두 축은 비용 3 으로 밀리고 남은 top_k 가 이긴다 — 선언 순서가 아니라
        **비용**이 갈랐다. (gold 근거가 있으면 top_k 축은 방향 충돌로 아예 제외돼
        비용 비교까지 가지도 않는다. 여기서는 비용 규칙만 보려고 근거를 비운다.)
        """
        findings = [make_finding("p1", "retrieval_missing_gold")]
        request, _decision = planner.plan(make_state(findings))

        self.assertEqual(request.action_key, "retriever.top_k:increase")
        self.assertEqual(request.action_score_breakdown["base_cost"], 1.0)
        runner_ups = {
            entry["action_key"] for entry in request.metadata["runner_up_actions"]
        }
        self.assertTrue(
            runner_ups <= {
                "chunker.chunk_overlap:increase",
                "chunker.chunk_size:increase",
            },
            f"예상 밖의 경쟁 action: {runner_ups}",
        )

    def test_same_cost_ties_break_on_action_key_not_declaration(self):
        """비용까지 같으면 action_key 사전순이다(결정성 보장).

        `retrieval_incomplete_enumeration` 은 top_k 확대(1번 선언)와 MMR(2번 선언)을
        같은 비용으로 지지한다. 선언 순서를 따르면 top_k 인데, 사전순은 MMR 이다.
        """
        findings = [make_finding("p1", "retrieval_incomplete_enumeration", gold_n=3)]
        request, _decision = planner.plan(make_state(findings))

        self.assertEqual(request.action_key, "retriever.mmr:enable")

    def test_selection_is_deterministic_across_finding_order(self):
        """입력 순서가 바뀌어도 같은 선택이 나온다.

        예전 `_group_by_label` 은 dict 삽입 순서에 의존해 동점에서 결과가 흔들렸다.
        """
        findings = [
            make_finding("p1", "retrieval_missing_gold", gold_n=3),
            make_finding("p2", "retrieval_incomplete_enumeration", gold_n=3),
        ]
        forward, _ = planner.plan(make_state(findings))
        backward, _ = planner.plan(make_state(list(reversed(findings))))

        self.assertEqual(forward.action_key, backward.action_key)

    def test_blocked_action_falls_through_to_the_next_axis(self):
        """차단된 축이 있으면 다음 순위 축으로 넘어간다(라벨을 통째로 건너뛰지 않는다)."""
        findings = [make_finding("p1", "retrieval_missing_gold", gold_n=3)]
        request, _decision = planner.plan(
            make_state(findings), blacklist={"retriever.top_k:increase"}
        )

        self.assertIsNotNone(request)
        self.assertNotEqual(request.action_key, "retriever.top_k:increase")
        self.assertEqual(request.supporting_labels, ["retrieval_missing_gold"])


class SearchSpaceCharacterizationTest(unittest.TestCase):
    """근거값 계산 결과를 박제한다.

    후보값 수학은 전환 중 **변경하지 않는다**(계획서 §11 범위 밖). 단계 2 에서
    candidate_values.py 로 이동할 때 결과가 같아야 한다.
    """

    def test_grounded_top_k_from_gold_ranks(self):
        """gold 순위 실측이 있으면 무릎 분석으로 후보를 만든다."""
        findings = [
            make_finding("p1", "retrieval_missing_gold", gold_ranks={"g1": 7}),
            make_finding("p2", "retrieval_missing_gold", gold_ranks={"g1": 8}),
            make_finding("p3", "retrieval_missing_gold", gold_ranks={"g1": 9}),
        ]
        _label, _prescription, space, _decision = _select(findings)
        self.assertIn("retriever.top_k", space)
        for value in space["retriever.top_k"]:
            self.assertGreater(value, 5, "increase 방향인데 baseline 이하 후보가 있다")

    def test_direction_keyword_fallback_without_evidence(self):
        """근거가 없으면 방향 키워드(현재값 ×2)로 폴백한다."""
        findings = [make_finding("p1", "retrieval_missing_gold")]
        _label, _prescription, space, _decision = _select(findings)
        self.assertEqual(space.get("retriever.top_k"), [10])


class DecisionModeCharacterizationTest(unittest.TestCase):
    """흐름 결정(3-way)은 전환 후에도 그대로 유지된다(계획서 §4 유지 항목)."""

    def test_no_actionable_finding_skips_optimize(self):
        findings = [make_finding("p1", "corpus_gap")]           # D그룹 manual
        label, _prescription, _space, decision = _select(findings)
        self.assertIsNone(label)
        self.assertEqual(decision.mode, "manual_required")
        self.assertEqual(decision.manual_labels, ["corpus_gap"])

    def test_preliminary_finding_is_excluded(self):
        """확정되지 않은(preliminary) Finding 은 자동 처방 대상이 아니다."""
        findings = [
            make_finding("p1", "retrieval_missing_gold", gold_n=3, confirmed=False)
        ]
        label, _prescription, _space, decision = _select(findings)
        self.assertIsNone(label)
        self.assertNotEqual(decision.mode, "apply_optimize")

    def test_all_candidates_blacklisted_skips_optimize(self):
        findings = [make_finding("p1", "chunking_overchunking", gold_n=3)]
        blacklist = {
            ("chunking_overchunking", p["id"])
            for p in planner.rules.get_rule("chunking_overchunking")["prescriptions"]
        }
        label, _prescription, _space, decision = _select(findings, blacklist=blacklist)
        self.assertIsNone(label)
        self.assertEqual(decision.mode, "use_current")


class TieBreakDeterminismTest(unittest.TestCase):
    """⚠️ 동점 입력 — 선택 '내용'을 박제하지 않는다.

    현재는 dict 삽입 순서로 갈리고, 전환 후에는 결정적 tie-break 로 갈린다.
    즉 **선택이 바뀌는 것이 정상**이다. 여기서는 "같은 입력이면 항상 같은 선택"
    이라는 성질만 확인한다. 이 성질은 전환 전후 모두 유지되어야 한다
    (계획서 §7.2 invariant 6).
    """

    def _tied_findings(self):
        """같은 그룹·같은 probe 수 → 점수가 같은 두 라벨."""
        return [
            make_finding("p1", "retrieval_missing_gold", gold_n=3),
            make_finding("p2", "retrieval_incomplete_enumeration", gold_n=3),
        ]

    def test_same_input_yields_same_selection(self):
        first = _select(self._tied_findings())[0]
        for _ in range(5):
            self.assertEqual(_select(self._tied_findings())[0], first)

    def test_tied_selection_is_one_of_the_tied_labels(self):
        """어느 쪽이 뽑히든 동점 후보 안에서 나와야 한다."""
        label = _select(self._tied_findings())[0]
        self.assertIn(
            label, {"retrieval_missing_gold", "retrieval_incomplete_enumeration"}
        )


if __name__ == "__main__":
    unittest.main()
