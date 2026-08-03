"""
tests/test_action_inventory.py
Action-Centered Optimizer 전환의 baseline 고정 (계획서 단계 0).

[이 파일이 하는 일]
  전환 전 "지금 rules.py가 선언한 것 중 실제로 실행 가능한 canonical action이
  무엇인가"를 박제한다. 단계 1에서 action catalog를 만들 때 이 스냅샷이 등록 목록의
  정답지가 되고, 전환 도중 실행 가능 범위가 조용히 넓어지거나 좁아지면 여기서 깨진다.

[왜 목록을 하드코딩하는가]
  tools/action_inventory.py 는 rules.py + optimizer 정책에서 목록을 계산한다. 그
  계산을 그대로 다시 호출해 비교하면 아무것도 검증하지 못한다(같은 코드가 같은 답을
  낸다). 그래서 사람이 검토한 시점의 결과를 상수로 박아 두고 비교한다.

  따라서 이 테스트가 깨지는 것은 "버그"가 아니라 "실행 가능 범위가 바뀌었다"는 신호다.
  변경이 의도된 것이면 아래 상수를 갱신하고, 계획서 §2 현황표도 같이 갱신한다.

[baseline 측정 시점]
  origin/main 머지 후(#70·#72·#73·#76·#78 반영). 전체 스위트 936 tests / 약 11초,
  환경 사유 수집 에러 4건(test_eval·test_oauth·test_pipeline·test_ragas_eval —
  모두 실데이터/실키가 필요한 실행 스크립트라 unittest 가 오수집한 것).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.action_inventory import collect


# ── baseline 스냅샷 ────────────────────────────────────────────────
# (action_key, 지지 라벨 수, tier, 재색인 여부, backend)
EXPECTED_EXECUTABLE = {
    "chunker.chunk_overlap:increase":        (2, ("A",),      True,  "internal"),
    "chunker.chunk_size:decrease":           (3, ("A", "C"),  True,  "internal"),
    "chunker.chunk_size:increase":           (3, ("A",),      True,  "internal"),
    "chunker.strategy:replace":              (2, ("A",),      True,  "internal"),
    "context.compression.enabled:enable":    (2, ("C",),      False, "rules"),
    # #79: generation_wrongful_abstention(과다 기권)의 relax_abstention. abstention_strict 의
    # 반대 방향 레버라 같은 축의 경쟁 action 으로 함께 선다.
    "generation.abstention_relaxed:enable":  (1, ("B",),      False, "rules"),
    "generation.abstention_strict:enable":   (3, ("B",),      False, "rules"),
    "generation.completeness_mode:enable":   (1, ("B",),      False, "rules"),
    "generation.require_citation:enable":    (3, ("B",),      False, "rules"),
    "generation.restate_question:enable":    (1, ("B",),      False, "rules"),
    "generation.temperature:decrease":       (2, ("B",),      False, "rules"),
    "reranker.candidate_count:increase":     (1, ("A",),      False, "rules"),
    "reranker.enabled:disable":              (1, ("A",),      False, "rules"),
    "reranker.enabled:enable":               (2, ("A",),      False, "rules"),
    "retriever.hybrid_dense_weight:adjust":  (1, ("A",),      False, "internal"),
    # 지지 라벨 2 → 3: retrieval_duplicate_crowding 이 ready 로 올라오며 합류했다.
    "retriever.mmr:enable":                  (3, ("A", "C"),  False, "rules"),
    "retriever.search_type:replace":         (1, ("A",),      False, "rules"),
    "retriever.top_k:decrease":              (2, ("C",),      False, "internal"),
    "retriever.top_k:increase":              (2, ("A",),      False, "internal"),
}

# 차단 사유는 성격이 다르다. 계획서 §2가 둘을 구분해 기록하도록 정했다.
#   not_state_mappable : mapper 계약 부재 → 영구적. mapper + 소비 노드가 함께 필요
#   capability_off     : 소비 경로는 있으나 검증된 후보 부재 → capability 값만 바꾸면 열림
EXPECTED_BLOCKED = {
    "adaptive_retrieval:enable":            "not_state_mappable",
    "answer_checklist_review:enable":       "not_state_mappable",
    "conflict_resolution_prompt:replace":   "not_state_mappable",
    "context_ordering:replace":             "not_state_mappable",
    "embedding.model:replace":              "capability_off(embedding_model)",
    "generation.model:replace":             "capability_off(generation_model)",
    "noise_filter:enable":                  "not_state_mappable",
    "query_rewrite:replace":                "not_state_mappable",
    "reranker_model:replace":               "not_state_mappable",
}

EXPECTED_SWEEP_AXES = {
    "chunker.chunk_overlap",
    "chunker.chunk_size",
    "chunker.strategy",
    "retriever.hybrid_dense_weight",
    "retriever.top_k",
}


class ActionInventoryBaselineTest(unittest.TestCase):
    """전환 전 실행 가능 action 목록을 고정한다."""

    @classmethod
    def setUpClass(cls):
        cls.snapshot = collect()
        cls.executable = {a["action_key"]: a for a in cls.snapshot["executable"]}
        cls.blocked = {a["action_key"]: a for a in cls.snapshot["blocked"]}

    def test_executable_action_set_unchanged(self):
        """실행 가능 action 집합이 baseline과 같아야 한다."""
        self.assertEqual(
            set(self.executable),
            set(EXPECTED_EXECUTABLE),
            "실행 가능 action 집합이 바뀌었다. 의도된 변경이면 이 파일의 "
            "EXPECTED_EXECUTABLE 과 계획서 §2 현황표를 함께 갱신할 것.",
        )

    def test_executable_action_attributes_unchanged(self):
        """각 action의 지지 수·tier·재색인·backend가 baseline과 같아야 한다."""
        for key, expected in EXPECTED_EXECUTABLE.items():
            with self.subTest(action=key):
                action = self.executable[key]
                actual = (
                    action["support_count"],
                    tuple(action["tiers"]),
                    action["reindex_required"],
                    action["backend"],
                )
                self.assertEqual(actual, expected)

    def test_blocked_actions_and_reasons_unchanged(self):
        """차단 action과 그 사유가 baseline과 같아야 한다."""
        actual = {k: a["blocked_reason"] for k, a in self.blocked.items()}
        self.assertEqual(actual, EXPECTED_BLOCKED)

    def test_blocked_reason_kinds_are_distinguishable(self):
        """차단 사유가 두 종류로만 나뉘어야 한다(계획서 §2).

        catalog가 해제 조건을 구분해 기록하려면 사유가 섞이면 안 된다.
        """
        for key, reason in EXPECTED_BLOCKED.items():
            with self.subTest(action=key):
                self.assertTrue(
                    reason == "not_state_mappable"
                    or reason.startswith("capability_off("),
                    f"알 수 없는 차단 사유: {reason}",
                )

    def test_sweep_axes_unchanged(self):
        """internal sweep 지원 축이 baseline과 같아야 한다.

        #73으로 chunker.strategy가 추가됐다. 이 집합이 줄면 후보가 여러 개인 action이
        sweep 없이 1회 적용으로 떨어진다.
        """
        self.assertEqual(set(self.snapshot["sweep_axes"]), EXPECTED_SWEEP_AXES)

    def test_counts_match_plan_document(self):
        """계획서 §2에 적힌 수치와 일치해야 한다."""
        # #79 머지로 실행 가능 action 과 ready 라벨이 각각 하나씩 늘었다
        # (generation.abstention_relaxed:enable / generation_wrongful_abstention).
        # retrieval_duplicate_crowding 이 ready 로 올라오며 ready 가 하나 더 늘었지만
        # action 은 안 늘었다 — 이미 있는 retriever.mmr:enable 의 지지 라벨이 될 뿐이다.
        self.assertEqual(self.snapshot["executable_count"], 19)
        self.assertEqual(self.snapshot["shared_count"], 12)
        self.assertEqual(self.snapshot["blocked_count"], 9)
        self.assertEqual(self.snapshot["label_status"]["ready"], 21)


class CompetingAxisBaselineTest(unittest.TestCase):
    """같은 축에 여러 action이 있는 경우를 고정한다 (계획서 §4.4).

    전환 후 이 축들이 "경쟁"으로 처리되는지 "eligibility가 먼저 걸러내는지"가
    starvation 여부를 가른다. 축 목록이 바뀌면 §4.4 정책을 재검토해야 한다.
    """

    @classmethod
    def setUpClass(cls):
        snapshot = collect()
        by_axis: dict[str, set[str]] = {}
        for action in snapshot["executable"]:
            by_axis.setdefault(action["canonical_path"], set()).add(action["operation"])
        cls.competing = {a: ops for a, ops in by_axis.items() if len(ops) > 1}

    def test_competing_axes_unchanged(self):
        self.assertEqual(
            self.competing,
            {
                "chunker.chunk_size": {"increase", "decrease"},
                "retriever.top_k": {"increase", "decrease"},
                "reranker.enabled": {"enable", "disable"},
            },
            "같은 축의 action 구성이 바뀌었다. 계획서 §4.4(충돌 정책)를 재검토할 것.",
        )

    def test_boolean_axis_is_enable_disable_pair(self):
        """boolean 축은 enable/disable 쌍이어야 한다.

        §4.4는 boolean 축을 충돌 정책에서 제외한다 — no-op 필터가 현재 상태로
        한쪽을 자동 제거하기 때문이다. 이 전제가 깨지면(예: set:true/set:false로
        선언되면) 두 action이 동시에 경쟁해 2:1 지지에서 영구 보류가 발생한다.
        """
        self.assertEqual(self.competing["reranker.enabled"], {"enable", "disable"})

    def test_no_axis_has_multiple_replace_actions(self):
        """같은 축에 replace action이 둘 이상이면 안 된다 (starvation 방지).

        §3.2: 고정값을 action key에 넣으면 지지 label 집합이 같은 값들이 서로
        경쟁자가 되어 점수가 영원히 같아진다(chunker.strategy 사례). 값은 후보로
        넘겨 하나의 replace action이 되어야 한다.
        """
        snapshot = collect()
        replace_per_axis: dict[str, int] = {}
        for action in snapshot["executable"]:
            if action["operation"] == "replace":
                axis = action["canonical_path"]
                replace_per_axis[axis] = replace_per_axis.get(axis, 0) + 1
        for axis, count in replace_per_axis.items():
            with self.subTest(axis=axis):
                self.assertEqual(
                    count, 1,
                    f"{axis}에 replace action이 {count}개다. 값을 후보로 통합할 것.",
                )

    def test_multi_value_replace_is_sweepable(self):
        """후보값이 여러 개인 action은 sweep 대상이어야 한다.

        sweep이 안 되면 후보를 한 번에 하나씩만 적용하게 되어 예산을 더 쓴다.
        chunker.strategy가 이 경우였고 #73으로 해소됐다.
        """
        snapshot = collect()
        for action in snapshot["executable"]:
            if action["candidate_value_count"] > 1:
                with self.subTest(action=action["action_key"]):
                    self.assertEqual(
                        action["backend"], "internal",
                        f"{action['action_key']}는 후보값이 "
                        f"{action['candidate_value_count']}개인데 sweep 대상이 아니다.",
                    )


class FlatKeyNormalizationTest(unittest.TestCase):
    """flat/canonical 혼재가 무해함을 고정한다 (계획서 §6).

    rules.py의 patch 키는 flat과 canonical이 섞여 있다. catalog가 canonicalize_path를
    거치면 문제가 없다는 것이 전환의 전제이며, 이 테스트가 그 전제를 지킨다.
    """

    def test_executable_actions_use_canonical_paths(self):
        """실행 가능 action의 경로는 모두 canonical(점 표기)이어야 한다."""
        for action in collect()["executable"]:
            with self.subTest(action=action["action_key"]):
                self.assertIn(
                    ".", action["canonical_path"],
                    f"{action['canonical_path']}가 정규화되지 않았다.",
                )

    def test_flat_declared_keys_are_absorbed(self):
        """flat으로 선언된 키가 canonical로 흡수됨을 확인한다.

        top_k / chunk_size / chunk_overlap / embedding_model 이 여기 해당한다.
        canonicalize_path가 이들을 흡수하지 못하면 rules.py 전면 재작성이 전환의
        전제가 되어 작업 범위가 크게 늘어난다.
        """
        absorbed = {
            action["canonical_path"]
            for action in collect()["executable"]
            if action["flat_keys"]
        }
        self.assertTrue(
            {"retriever.top_k", "chunker.chunk_size", "chunker.chunk_overlap"}
            <= absorbed,
            f"flat 키 흡수가 깨졌다. 흡수된 경로: {sorted(absorbed)}",
        )


class InventoryFollowsCatalogTest(unittest.TestCase):
    """집계 스크립트가 catalog 와 어긋날 자리를 만들지 않는지.

    표가 낡는 문제를 고치려고 만든 스크립트가 action key 조립·operation 유도·차단
    사유를 따로 구현하면, 정확히 그 문제를 다른 곳에 옮겨 놓는 셈이다
    (PR #75 리뷰 지적). catalog 를 정본으로 읽는지 여기서 고정한다.
    """

    def test_every_catalog_action_appears_exactly_once(self):
        from agents.optimize import action_catalog

        snapshot = collect()
        listed = [a["action_key"] for a in snapshot["executable"] + snapshot["blocked"]]

        self.assertEqual(sorted(listed), sorted(a.key for a in action_catalog.all_actions()))
        self.assertEqual(len(listed), len(set(listed)))

    def test_definitional_facts_come_from_the_catalog(self):
        from agents.optimize import action_catalog

        snapshot = collect()
        for record in snapshot["executable"] + snapshot["blocked"]:
            action = action_catalog.get_action(record["action_key"])
            with self.subTest(action=record["action_key"]):
                self.assertEqual(record["operation"], action.operation)
                self.assertEqual(record["canonical_path"], action.canonical_path)
                self.assertEqual(record["reindex_required"], action.reindex_required)
                self.assertEqual("blocked_reason" in record, action.is_blocked)

    def test_backend_matches_catalog_sweepability(self):
        from agents.optimize import action_catalog

        for record in collect()["executable"]:
            with self.subTest(action=record["action_key"]):
                expected = (
                    "internal"
                    if action_catalog.is_sweepable(record["action_key"])
                    else "rules"
                )
                self.assertEqual(record["backend"], expected)


if __name__ == "__main__":
    unittest.main()
