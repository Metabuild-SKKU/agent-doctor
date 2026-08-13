"""
tests/test_ext_advisor.py
외부 RAG 권고 카드(agents/optimize/ext_advisor.py) 회귀 테스트.

이 계층이 지켜야 할 것:
  - 내부 라벨을 주우면 안 된다(자동 적용 루프와 섞이는 경로가 열린다)
  - 확정/예비를 뭉뚱그리면 안 된다(증거 수준을 속이게 된다)
  - rules.py 의 처방과 어긋나면 안 된다(문구 테이블이 따로 늙는 것을 막는다)
"""
import unittest

from core.schema import Finding

from agents.optimize.ext_advisor import (
    EXT_LABEL_TO_RULE,
    _PRESCRIPTION_TEXT,
    build_ext_recommendations,
)
from agents.optimize.rules import LABEL_TO_PRESCRIPTIONS


def _f(label, confirmed=True, severity="warning", desc="사유"):
    return Finding(
        finding_id=f"t_{label}_{confirmed}", type="generation",
        severity=severity, description=desc, label=label, confirmed=confirmed)


class ExtAdvisorTests(unittest.TestCase):

    def test_internal_labels_are_ignored(self):
        """내부 라벨은 카드가 되지 않는다 — 이 경로는 ext_ 전용이다."""
        cards = build_ext_recommendations([
            _f("retrieval_low_rank"), _f("generation_hallucination"),
        ])
        self.assertEqual(cards, [])

    def test_same_label_groups_into_one_card(self):
        """라벨이 처방의 단위이므로 probe 별 소견은 카드 하나로 묶인다."""
        cards = build_ext_recommendations([
            _f("ext_grounded_but_wrong"), _f("ext_grounded_but_wrong"),
            _f("ext_grounded_but_wrong", confirmed=False),
        ])
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["confirmed"], 2)
        self.assertEqual(cards[0]["tentative"], 1)

    def test_tentative_findings_survive(self):
        """예비 소견도 카드가 된다 — gold_contexts 가 없는 로그에서 환각 소견은
        전부 예비로 강등되므로, 여기서 버리면 카드가 통째로 사라진다."""
        cards = build_ext_recommendations([
            _f("ext_generation_hallucination", confirmed=False)])
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["confirmed"], 0)
        self.assertEqual(cards[0]["tentative"], 1)

    def test_severity_then_confirmed_ordering(self):
        cards = build_ext_recommendations([
            _f("ext_grounded_but_wrong", severity="warning"),
            _f("ext_answer_off_topic", severity="critical"),
        ])
        self.assertEqual(cards[0]["label"], "ext_answer_off_topic")

    def test_current_value_is_injected_from_config(self):
        """로그 config 가 있으면 '현재값 → 방향'이 실린다."""
        cards = build_ext_recommendations(
            [_f("ext_context_overflow")], {"top_k": 5, "chunk_size": 512})
        steps = {s["prescription_id"]: s for s in cards[0]["steps"]}
        self.assertEqual(steps["decrease_top_k"]["current"], "현재 top_k=5 → 낮추기")
        self.assertEqual(steps["shrink_chunk_size"]["current"], "현재 chunk_size=512 → 낮추기")

    def test_missing_config_yields_no_current_hint(self):
        """config 가 없으면 숫자를 지어내지 않고 비운다."""
        cards = build_ext_recommendations([_f("ext_context_overflow")])
        self.assertTrue(all(s["current"] is None for s in cards[0]["steps"]))

    def test_dotted_patch_key_matches_flat_config(self):
        """상대 로그의 config 는 자유 키다. 'generation.temperature' 를
        'temperature' 로만 준 로그에서도 현재값을 찾아야 한다."""
        cards = build_ext_recommendations(
            [_f("ext_generation_hallucination")], {"temperature": 0.9})
        steps = {s["prescription_id"]: s for s in cards[0]["steps"]}
        self.assertEqual(steps["lower_temperature"]["current"], "현재 temperature=0.9 → 낮추기")

    def test_reindex_flag_is_carried(self):
        """재색인이 필요한 처방은 표시돼야 한다 — 상대가 부담을 알고 골라야 한다."""
        cards = build_ext_recommendations([_f("ext_context_overflow")])
        steps = {s["prescription_id"]: s for s in cards[0]["steps"]}
        self.assertTrue(steps["shrink_chunk_size"]["needs_reindex"])
        self.assertFalse(steps["decrease_top_k"]["needs_reindex"])

    def test_manual_prescription_text_comes_from_rules(self):
        """manual 처방은 rules.py 가 이미 사람용 문구를 갖고 있다 — 재작성하지 않는다."""
        cards = build_ext_recommendations([_f("ext_grounded_but_wrong")])
        steps = {s["prescription_id"]: s for s in cards[0]["steps"]}
        rules_action = LABEL_TO_PRESCRIPTIONS["corpus_gap"]["prescriptions"][0]["action"]
        self.assertEqual(steps["locate_missing_evidence"]["action"], rules_action)
        self.assertTrue(steps["locate_missing_evidence"]["manual"])

    def test_every_mapped_rule_label_exists(self):
        """rules.py 에서 라벨이 사라지면 KeyError 가 아니라 여기서 잡힌다."""
        for ext, rule in EXT_LABEL_TO_RULE.items():
            self.assertIn(rule, LABEL_TO_PRESCRIPTIONS, f"{ext} → {rule} 가 rules 에 없음")

    def test_every_card_step_has_action_text(self):
        """카드에 빈 문구가 실리지 않는다 — 처방 id 폴백이라도 있어야 한다."""
        for ext in EXT_LABEL_TO_RULE:
            cards = build_ext_recommendations([_f(ext)])
            self.assertEqual(len(cards), 1, ext)
            self.assertTrue(cards[0]["steps"], f"{ext} 처방 0개")
            for s in cards[0]["steps"]:
                self.assertTrue(s["action"].strip(), f"{ext}/{s['prescription_id']} 문구 없음")

    def test_prescription_text_table_matches_rules(self):
        """문구 테이블이 rules 와 어긋나 늙는 것을 막는다 — 여기 있는 id 는
        전부 ext_ 라벨이 실제로 참조하는 처방이어야 한다."""
        referenced = {
            p["id"]
            for rule in EXT_LABEL_TO_RULE.values()
            for p in LABEL_TO_PRESCRIPTIONS[rule]["prescriptions"]
        }
        self.assertTrue(
            set(_PRESCRIPTION_TEXT) <= referenced,
            f"쓰이지 않는 문구: {set(_PRESCRIPTION_TEXT) - referenced}")

    def test_evidence_is_capped(self):
        """근거 문구가 카드를 넘치게 하지 않는다."""
        cards = build_ext_recommendations(
            [_f("ext_answer_off_topic", desc=f"사유{i}") for i in range(10)])
        self.assertLessEqual(len(cards[0]["evidence"]), 3)

    def test_empty_input(self):
        self.assertEqual(build_ext_recommendations([]), [])
        self.assertEqual(build_ext_recommendations(None), [])


class RepresentativeConfigTests(unittest.TestCase):
    """assess_capability["config"] — 권고에 실을 '현재값'의 출처."""

    def _rec(self, config):
        from agents.eval.log_intake import ExternalLogRecord
        return ExternalLogRecord(question="q", answer="a", config=config)

    def test_uniform_config_is_representative(self):
        from agents.eval.log_intake import assess_capability
        cfg = {"top_k": 5, "chunk_size": 512}
        cap = assess_capability([self._rec(dict(cfg)) for _ in range(3)])
        self.assertEqual(cap["config"], cfg)

    def test_mixed_config_is_dropped(self):
        """레코드마다 다르면 대표값이 성립하지 않는다 — 사실이 아닌 현재값을
        '현재 top_k=5' 라고 단정하면 권고가 거짓말이 된다."""
        from agents.eval.log_intake import assess_capability
        cap = assess_capability([
            self._rec({"top_k": 5}), self._rec({"top_k": 10})])
        self.assertEqual(cap["config"], {})

    def test_absent_config(self):
        from agents.eval.log_intake import assess_capability
        self.assertEqual(assess_capability([self._rec({})])["config"], {})


class OneAtATimeTests(unittest.TestCase):
    """처방은 하나씩 적용해야 효과 귀속이 된다(CONTEXT.md §2 방식2).

    이 원칙이 내부 루프에만 있고 외부 권고에는 없으면, 상대는 처방 4개를 한꺼번에
    적용하고 무엇이 들었는지 알 수 없게 된다 — 방식1 후퇴다."""

    def _steps(self, label="ext_generation_hallucination"):
        cards = build_ext_recommendations([_f(label)])
        return cards[0]["steps"]

    def test_steps_are_ordered(self):
        """rules.py 의 처방 리스트는 우선순위 순서다 — 그 순서를 번호로 넘긴다."""
        steps = self._steps()
        self.assertEqual([s["order"] for s in steps], list(range(1, len(steps) + 1)))

    def test_only_first_is_primary(self):
        steps = self._steps()
        self.assertTrue(steps[0]["primary"])
        self.assertTrue(all(not s["primary"] for s in steps[1:]))

    def test_manual_steps_are_all_primary(self):
        """manual 처방은 순차 시도가 아니라 다 해야 하는 절차다(근거 특정 → 재색인)."""
        steps = self._steps("ext_grounded_but_wrong")
        self.assertTrue(all(s["manual"] for s in steps))
        self.assertTrue(all(s["primary"] for s in steps))

    def test_card_carries_apply_guide(self):
        """처방 목록만 주면 한꺼번에 적용한다 — 적용 방법을 카드가 직접 말해야 한다."""
        card = build_ext_recommendations([_f("ext_generation_hallucination")])[0]
        self.assertIn("하나만", card["how_to_apply"])
        self.assertIn("다시 로그", card["how_to_apply"])


if __name__ == "__main__":
    unittest.main()
