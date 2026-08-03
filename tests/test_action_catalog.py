"""
tests/test_action_catalog.py
action catalog 계약 검증 (계획서 단계 1).

catalog는 config 변경의 단일 진실 원천이다. 여기가 틀리면 이후 모든 선택이 틀리므로,
계약을 테스트로 고정한다.

[test_action_inventory.py 와의 차이]
  inventory : rules + optimizer 정책에서 "무엇이 실행 가능한가"를 집계하고, 그 결과가
              baseline과 같은지 본다(= 실행 가능 범위가 바뀌었는지 감시).
  이 파일   : catalog가 그 집계와 **일치하는지**, 그리고 catalog 자체의 계약
              (key 형식·비용 유도·starvation 방지 등)을 지키는지 본다.
  두 파일이 동시에 깨지면 rules/optimizer가 바뀐 것이고, 이 파일만 깨지면 catalog
  구현이 어긋난 것이다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.optimize import action_catalog as catalog
from agents.optimize import optimizer as optimizer_module
from agents.optimize import rules
from tools.action_inventory import collect


class CatalogIntegrityTest(unittest.TestCase):
    """catalog 자체의 계약."""

    def test_validate_catalog_reports_no_problems(self):
        problems = catalog.validate_catalog()
        self.assertEqual(problems, [], "\n".join(problems))

    def test_action_keys_are_unique_and_sorted_lookup(self):
        keys = [a.key for a in catalog.all_actions()]
        self.assertEqual(len(keys), len(set(keys)), "중복 action key")
        self.assertEqual(keys, sorted(keys), "all_actions()는 사전순이어야 한다")

    def test_key_format_is_path_and_operation(self):
        """key에 label·prescription id·고정값이 들어가면 안 된다."""
        for action in catalog.all_actions():
            with self.subTest(action=action.key):
                self.assertEqual(
                    action.key, f"{action.canonical_path}:{action.operation}"
                )
                self.assertEqual(action.key.count(":"), 1)

    def test_each_action_owns_single_axis(self):
        for action in catalog.all_actions():
            with self.subTest(action=action.key):
                self.assertEqual(action.conflict_family, action.canonical_path)

    def test_lookup_returns_none_for_unknown_key(self):
        self.assertIsNone(catalog.get_action("nope.does_not_exist:increase"))


class CostDerivationTest(unittest.TestCase):
    """비용은 재색인 여부에서만 유도된다 (계획서 §3.1).

    rules.py의 cost가 전부 None이라 실측 근거가 없다. 근거 없는 숫자를 새로 만들지
    않는 것이 전환의 전제이며, 세분화는 confidence 생산과 함께 별도 작업이다.
    """

    def test_reindex_actions_cost_more(self):
        for action in catalog.all_actions():
            with self.subTest(action=action.key):
                expected = (
                    catalog.COST_REINDEX
                    if action.reindex_required
                    else catalog.COST_RUNTIME
                )
                self.assertEqual(action.base_cost, expected)

    def test_reindex_flag_matches_optimizer_source_of_truth(self):
        """재색인 판정은 optimizer.REINDEX_PATHS를 정본으로 삼는다."""
        for action in catalog.all_actions():
            if action.canonical_path in optimizer_module.REINDEX_PATHS:
                with self.subTest(action=action.key):
                    self.assertTrue(action.reindex_required)

    def test_cost_uses_action_own_reindex_not_label_first_prescription(self):
        """action 자신의 재색인 여부를 쓴다 (초판이 지적한 왜곡의 수정).

        기존 planner는 라벨의 '첫 처방' 비용을 라벨 전체 점수에 썼다. 그래서
        retrieval_missing_gold는 첫 처방이 top_k 증가(런타임)라 점수가 3배 뜨는데,
        같은 라벨이 지지하는 chunk_size 증가(재색인)에도 그 싼 비용이 적용됐다.
        """
        top_k = catalog.get_action("retriever.top_k:increase")
        chunk_size = catalog.get_action("chunker.chunk_size:increase")
        self.assertEqual(top_k.base_cost, catalog.COST_RUNTIME)
        self.assertEqual(chunk_size.base_cost, catalog.COST_REINDEX)


class BlockedActionTest(unittest.TestCase):
    """차단 action도 등록하되 사유를 구분한다 (계획서 §2)."""

    def test_blocked_actions_are_registered(self):
        """차단됐다고 빼지 않는다. 리포트가 '왜 못 썼는지' 설명해야 한다."""
        self.assertEqual(len(catalog.blocked_actions()), 9)

    def test_blocked_reason_kinds(self):
        """해제 조건이 다르므로 사유를 구분한다."""
        for action in catalog.blocked_actions():
            with self.subTest(action=action.key):
                self.assertIn(
                    action.blocked_reason,
                    {"not_state_mappable", "capability_off", "runtime_unavailable"},
                )
                self.assertTrue(action.blocked_detail, "차단 사유 상세가 비어 있다")

    def test_capability_off_names_the_capability(self):
        """capability_off는 어떤 capability인지 밝혀야 해제 조건을 알 수 있다."""
        for action in catalog.blocked_actions():
            if action.blocked_reason == "capability_off":
                with self.subTest(action=action.key):
                    self.assertIn(
                        action.blocked_detail,
                        optimizer_module.DEFAULT_CAPABILITIES,
                    )

    def test_executable_actions_pass_both_gates(self):
        for action in catalog.executable_actions():
            with self.subTest(action=action.key):
                self.assertIn(
                    action.canonical_path, optimizer_module.STATE_MAPPABLE_PATHS
                )
                if action.capability:
                    self.assertTrue(
                        optimizer_module.DEFAULT_CAPABILITIES.get(action.capability)
                    )


class StarvationPreventionTest(unittest.TestCase):
    """같은 축의 여러 값이 서로 경쟁하지 않아야 한다 (계획서 §3.2).

    고정값을 action key에 넣으면 지지 label 집합이 같은 값들이 경쟁자가 되어 점수가
    영원히 동률이 되고, 충돌 보류 규칙에 걸려 그 축이 한 번도 선택되지 않는다.
    chunker.strategy가 실제로 이 경우였다.
    """

    def test_single_replace_action_per_axis(self):
        per_axis: dict[str, list[str]] = {}
        for action in catalog.all_actions():
            if action.operation == "replace":
                per_axis.setdefault(action.canonical_path, []).append(action.key)
        for axis, keys in per_axis.items():
            with self.subTest(axis=axis):
                self.assertEqual(len(keys), 1, f"{axis}에 replace가 {keys}")

    def test_chunker_strategy_is_single_action_with_candidates(self):
        """두 청킹 전략은 경쟁자가 아니라 한 action의 두 후보다."""
        strategy_actions = catalog.actions_on_axis("chunker.strategy")
        self.assertEqual(len(strategy_actions), 1)
        self.assertEqual(strategy_actions[0].key, "chunker.strategy:replace")

    def test_multi_candidate_axis_is_sweepable(self):
        """후보가 여럿인 축은 sweep으로 한 study에서 비교할 수 있어야 한다."""
        self.assertTrue(catalog.is_sweepable("chunker.strategy:replace"))


class BooleanAxisTest(unittest.TestCase):
    """boolean 축은 enable/disable 쌍이며 자동 배타된다 (계획서 §4.4)."""

    def test_reranker_enabled_has_enable_and_disable(self):
        ops = {a.operation for a in catalog.actions_on_axis("reranker.enabled")}
        self.assertEqual(ops, {"enable", "disable"})

    def test_boolean_actions_carry_no_fixed_value_in_key(self):
        """set:true/set:false 형태가 아니어야 한다.

        값을 key에 넣으면 두 action이 동시에 경쟁 대상이 되어, 2:1 지지에서
        충돌 보류 규칙에 걸려 축이 영구히 닫힌다.
        """
        for action in catalog.all_actions():
            if action.operation in {"enable", "disable"}:
                with self.subTest(action=action.key):
                    self.assertNotIn("true", action.key.lower())
                    self.assertNotIn("false", action.key.lower())


class PrerequisiteTest(unittest.TestCase):
    """실행 전 조건을 catalog가 드러낸다."""

    def test_candidate_count_requires_reranker_enabled(self):
        action = catalog.get_action("reranker.candidate_count:increase")
        self.assertIn("reranker.enabled", action.prerequisites)


class CatalogMatchesInventoryTest(unittest.TestCase):
    """catalog가 inventory 집계와 일치해야 한다.

    두 모듈이 같은 정책(optimizer)에서 파생하므로 결과가 같아야 한다. 어긋나면
    한쪽이 정책을 잘못 읽고 있다는 뜻이다.
    """

    @classmethod
    def setUpClass(cls):
        cls.snapshot = collect()

    def test_executable_sets_match(self):
        self.assertEqual(
            {a.key for a in catalog.executable_actions()},
            {a["action_key"] for a in self.snapshot["executable"]},
        )

    def test_blocked_sets_match(self):
        self.assertEqual(
            {a.key for a in catalog.blocked_actions()},
            {a["action_key"] for a in self.snapshot["blocked"]},
        )

    def test_sweepability_matches(self):
        for action in catalog.executable_actions():
            expected = action.canonical_path in optimizer_module.BACKEND_SUPPORTED_PATHS[
                "internal"
            ]
            with self.subTest(action=action.key):
                self.assertEqual(catalog.is_sweepable(action.key), expected)


class RulesReferenceTest(unittest.TestCase):
    """rules가 참조하는 모든 변경이 catalog에 있어야 한다."""

    def test_every_ready_prescription_maps_to_catalog(self):
        for label, rule in rules.LABEL_TO_PRESCRIPTIONS.items():
            if rule.get("status") != "ready":
                continue
            for prescription in rule.get("prescriptions") or []:
                for raw_path, value in (prescription.get("patch") or {}).items():
                    key = catalog.build_action_key(raw_path, value)
                    with self.subTest(label=label, action=key):
                        self.assertIsNotNone(catalog.get_action(key))

    def test_flat_and_canonical_keys_map_to_same_action(self):
        """flat 선언과 canonical 선언이 같은 action으로 모여야 한다.

        rules.py는 두 표기가 섞여 있다. canonicalize_path가 흡수하므로 전면
        재작성이 전환의 전제가 아니라는 것이 이 테스트의 요지다.
        """
        self.assertEqual(
            catalog.build_action_key("top_k", "increase"),
            catalog.build_action_key("retriever.top_k", "increase"),
        )
        self.assertEqual(
            catalog.build_action_key("chunk_size", "increase"),
            catalog.build_action_key("chunker.chunk_size", "increase"),
        )


class OperationDerivationTest(unittest.TestCase):
    """patch 값 → operation 유도 규칙."""

    def test_direction_keywords(self):
        self.assertEqual(catalog.derive_operation("increase"), "increase")
        self.assertEqual(catalog.derive_operation("decrease"), "decrease")

    def test_booleans(self):
        self.assertEqual(catalog.derive_operation(True), "enable")
        self.assertEqual(catalog.derive_operation(False), "disable")

    def test_symbolic_adjust(self):
        """방향이 진단 실측으로 정해지는 지시어."""
        self.assertEqual(
            catalog.derive_operation("shift_to_favored_channel"), "adjust"
        )

    def test_fixed_value_becomes_replace(self):
        self.assertEqual(catalog.derive_operation("recursive_sentence"), "replace")
        self.assertEqual(catalog.derive_operation("hybrid"), "replace")


if __name__ == "__main__":
    unittest.main()
