"""
tests/test_action_history.py
이력·차단 identity 가 action 기반으로 옮겨졌는지 검증한다 (구현계획 §5.1 / §5.4 / §7.1).

세 가지를 본다.
  1. fingerprint  — 같은 의미의 config 는 같은 지문, 다른 의미는 다른 지문
  2. 결과 귀속    — "지지받았다"와 "해결됐다"를 구분한다
  3. 구버전 fallback — action 필드가 없는 저장 이력을 읽어도 깨지지 않는다
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import DiagnosticReport, Finding
from core.state import AgentDoctorState
from agents.optimize import history
from agents.optimize.schemas import (
    ActionAttemptKey,
    ActionStudyKey,
    OptimizationHistoryItem,
    OptimizationRequest,
    Verdict,
)


def _finding(label, probe="p1", confirmed=True):
    return Finding(
        finding_id=f"{probe}:{label}",
        type="retrieval_failure",
        severity="warning",
        description=label,
        label=label,
        confirmed=confirmed,
        affected_probes=[probe],
    )


def _report(labels):
    return DiagnosticReport(
        report_id="r",
        findings=[_finding(label) for label in labels],
        overall_score=60.0,
        ragas_scores={"context_recall": 0.7},
        pass_threshold=False,
    )


class FingerprintTest(unittest.TestCase):
    """지문이 흔들리면 차단이 무력화되고, 너무 굳으면 재시도가 막힌다."""

    def test_flat_and_canonical_configs_share_one_baseline(self):
        """같은 설정을 flat key 로 담든 표준 경로로 담든 같은 baseline 이다.

        이게 깨지면 config 표현이 바뀌는 것만으로 모든 차단이 풀린다.
        """
        flat = {"top_k": 5, "chunk_size": 512, "use_reranker": False}
        canonical = {
            "retriever.top_k": 5,
            "chunker.chunk_size": 512,
            "reranker.enabled": False,
        }

        self.assertEqual(
            history.baseline_fingerprint(flat),
            history.baseline_fingerprint(canonical),
        )

    def test_key_order_does_not_change_the_fingerprint(self):
        self.assertEqual(
            history.baseline_fingerprint({"top_k": 5, "chunk_size": 512}),
            history.baseline_fingerprint({"chunk_size": 512, "top_k": 5}),
        )

    def test_changing_any_optimizable_axis_moves_the_baseline(self):
        """축 하나만 바뀌어도 새 baseline 이다 — 그래야 재평가가 열린다(§5.1)."""
        before = history.baseline_fingerprint({"top_k": 5, "chunk_size": 512})
        after = history.baseline_fingerprint({"top_k": 5, "chunk_size": 256})

        self.assertNotEqual(before, after)

    def test_unrelated_config_noise_does_not_move_the_baseline(self):
        """Optimize 가 건드리지 않는 키는 baseline 정체성이 아니다.

        출력 경로 같은 값까지 지문에 넣으면 관계없는 변화로 차단이 전부 풀린다.
        """
        base = {"top_k": 5, "chunk_size": 512}
        noisy = {**base, "graph_output_dir": "output/other", "deduplicate": True}

        self.assertEqual(
            history.baseline_fingerprint(base),
            history.baseline_fingerprint(noisy),
        )

    def test_reranker_runtime_identity_is_part_of_the_baseline(self):
        """reranker 는 Index 가 실제 로드해 봐야 아는 축이다.

        모델이 바뀌거나 verified 로 승격되면 같은 config 전이라도 결과가 달라지므로
        새 baseline 으로 봐야 한다. 무관한 action 에는 영향이 없어야 한다.
        """
        config = {"top_k": 5, "use_reranker": False}
        unverified = {"reranker": {"status": "unavailable", "model": "m1"}}
        verified = {"reranker": {"status": "verified", "model": "m1"}}

        self.assertNotEqual(
            history.baseline_fingerprint(config, "reranker.enabled:enable", unverified),
            history.baseline_fingerprint(config, "reranker.enabled:enable", verified),
        )
        self.assertEqual(
            history.baseline_fingerprint(config, "retriever.top_k:increase", unverified),
            history.baseline_fingerprint(config, "retriever.top_k:increase", verified),
        )

    def test_candidate_fingerprint_ignores_path_spelling(self):
        self.assertEqual(
            history.candidate_fingerprint({"top_k": 9}),
            history.candidate_fingerprint({"retriever.top_k": 9}),
        )
        self.assertNotEqual(
            history.candidate_fingerprint({"retriever.top_k": 9}),
            history.candidate_fingerprint({"retriever.top_k": 7}),
        )

    def test_search_space_fingerprint_distinguishes_candidate_sets(self):
        self.assertEqual(
            history.search_space_fingerprint({"retriever.top_k": [7, 9]}),
            history.search_space_fingerprint({"top_k": [7, 9]}),
        )
        self.assertNotEqual(
            history.search_space_fingerprint({"retriever.top_k": [7, 9]}),
            history.search_space_fingerprint({"retriever.top_k": [7, 11]}),
        )

    def test_keys_are_hashable_and_compare_by_value(self):
        """set 에 담아 차단하므로 값 동등성이 곧 차단 동등성이다."""
        first = history.build_attempt_key(
            "retriever.top_k:increase", {"top_k": 5}, {"retriever.top_k": 9}
        )
        second = history.build_attempt_key(
            "retriever.top_k:increase", {"top_k": 5}, {"retriever.top_k": 9}
        )

        self.assertEqual(first, second)
        self.assertEqual(len({first, second}), 1)
        self.assertIsInstance(first, ActionAttemptKey)

    def test_study_key_for_request_needs_an_action(self):
        """action 이 없는 요청(구버전 파생)은 study 식별자를 만들지 않는다."""
        request = OptimizationRequest(
            request_id="r",
            iteration=0,
            baseline_config={"top_k": 5},
            supporting_labels=["retrieval_missing_gold"],
        )
        self.assertIsNone(history.study_key_for_request(request))

        request.action_key = "retriever.top_k:increase"
        request.search_space = {"retriever.top_k": [7, 9]}
        study = history.study_key_for_request(request)
        self.assertIsInstance(study, ActionStudyKey)
        self.assertEqual(study.action_key, "retriever.top_k:increase")


class PendingItemSnapshotTest(unittest.TestCase):
    def _request(self):
        return OptimizationRequest(
            request_id="req-1",
            iteration=0,
            baseline_config={"top_k": 5},
            search_space={"retriever.top_k": [7, 9]},
            action_key="retriever.top_k:increase",
            supporting_labels=[
                "retrieval_missing_gold",
                "retrieval_incomplete_enumeration",
            ],
            supporting_probes=["p1", "p2"],
        )

    def test_pending_item_carries_the_support_snapshot(self):
        """적용 당시 지지 집합이 없으면 §5.4 결과 귀속을 계산할 수 없다."""
        state = AgentDoctorState(index_config={"top_k": 5})
        item = history.create_pending_item(
            state,
            self._request(),
            "increase_top_k",
            {"top_k": 5},
            _report(["retrieval_missing_gold"]),
            applied_changes={"retriever.top_k": 7},
        )

        self.assertEqual(item.action_key, "retriever.top_k:increase")
        self.assertEqual(len(item.supporting_labels), 2)
        self.assertEqual(item.supporting_probes, ["p1", "p2"])
        self.assertEqual(item.metadata["applied_changes"], {"retriever.top_k": 7})

    def test_attempt_key_uses_the_applied_value_not_the_search_space(self):
        """탐색 범위 전체로 지문을 만들면 sweep 중간 후보가 서로를 차단한다."""
        state = AgentDoctorState(index_config={"top_k": 5})
        first = history.create_pending_item(
            state, self._request(), "increase_top_k", {"top_k": 5},
            None, applied_changes={"retriever.top_k": 7},
        )
        second = history.create_pending_item(
            state, self._request(), "increase_top_k", {"top_k": 5},
            None, applied_changes={"retriever.top_k": 9},
        )

        self.assertNotEqual(
            first.action_attempt_key.candidate_fingerprint,
            second.action_attempt_key.candidate_fingerprint,
        )
        # 같은 탐색이므로 study 는 하나다.
        self.assertEqual(first.action_study_key, second.action_study_key)

    def test_legacy_request_without_action_yields_no_keys(self):
        state = AgentDoctorState(index_config={"top_k": 5})
        request = self._request()
        request.action_key = None

        item = history.create_pending_item(
            state, request, "increase_top_k", {"top_k": 5}, None,
        )

        self.assertIsNone(item.action_attempt_key)
        self.assertIsNone(item.action_study_key)
        # failure_labels 의 의미가 "대표 라벨 하나"에서 "대상 라벨 전체"로 넓어졌다.
        # 값이 supporting_labels 와 같아지므로 리포트의 두 경로가 어긋나지 않는다.
        self.assertEqual(
            item.failure_labels,
            ["retrieval_missing_gold", "retrieval_incomplete_enumeration"],
        )
        self.assertEqual(item.selected_prescription_id, "increase_top_k")


class ResultAttributionTest(unittest.TestCase):
    """"지지받았다"와 "해결됐다"는 다른 사실이다 (구현계획 §5.4)."""

    @staticmethod
    def _item(supporting):
        return OptimizationHistoryItem(
            trial_id="t",
            request_id="r",
            iteration=1,
            failure_labels=supporting[:1],
            optimizer="rules",
            status="applied",
            action_key="retriever.top_k:increase",
            supporting_labels=list(supporting),
        )

    def test_resolved_labels_are_the_confirmed_finding_difference(self):
        item = self._item(["retrieval_missing_gold", "retrieval_low_rank"])

        result = history.attribute_result(
            item,
            _report(["retrieval_missing_gold", "retrieval_low_rank"]),
            _report(["retrieval_low_rank"]),
        )

        self.assertEqual(result["resolved_labels"], ["retrieval_missing_gold"])
        self.assertEqual(result["remaining_labels"], ["retrieval_low_rank"])

    def test_a_label_that_was_never_confirmed_is_not_counted_as_resolved(self):
        """before 에 없던 라벨이 after 에도 없다고 '해결'은 아니다."""
        item = self._item(["retrieval_missing_gold", "retrieval_low_rank"])

        result = history.attribute_result(
            item,
            _report(["retrieval_missing_gold"]),
            _report([]),
        )

        self.assertEqual(result["resolved_labels"], ["retrieval_missing_gold"])
        self.assertEqual(result["remaining_labels"], ["retrieval_low_rank"])

    def test_missing_measurement_claims_nothing(self):
        """리포트가 없으면 해결을 주장하지 않는다 — 전부 remaining 이다."""
        item = self._item(["retrieval_missing_gold"])

        result = history.attribute_result(item, None, _report([]))

        self.assertEqual(result["resolved_labels"], [])
        self.assertEqual(result["remaining_labels"], ["retrieval_missing_gold"])

    def test_finalize_item_records_attribution(self):
        item = self._item(["retrieval_missing_gold", "retrieval_low_rank"])
        item.metadata["before_report"] = _report(
            ["retrieval_missing_gold", "retrieval_low_rank"]
        )
        after = _report(["retrieval_low_rank"])
        verdict = Verdict(keep=True, before_score=0.6, after_score=0.8)

        history.finalize_item(item, verdict, {"top_k": 9}, after)

        self.assertEqual(item.metadata["resolved_labels"], ["retrieval_missing_gold"])
        self.assertEqual(item.metadata["remaining_labels"], ["retrieval_low_rank"])
        self.assertNotIn("before_report", item.metadata)   # 무거운 참조는 정리된다


class LegacyHistoryFallbackTest(unittest.TestCase):
    """이전 실행이 남긴 이력에는 action 필드가 없다."""

    @staticmethod
    def _legacy():
        return OptimizationHistoryItem(
            trial_id="old",
            request_id="old",
            iteration=1,
            failure_labels=["retrieval_missing_gold"],
            optimizer="rules",
            status="applied",
            selected_prescription_id="increase_top_k",
        )

    def test_last_action_key_ignores_legacy_items(self):
        self.assertIsNone(history.last_action_key([self._legacy()]))
        self.assertIsNone(history.last_action_study_key([self._legacy()]))

    def test_legacy_label_reader_still_works(self):
        self.assertEqual(
            history.last_failure_label([self._legacy()]),
            "retrieval_missing_gold",
        )

    def test_newest_action_wins_over_older_entries(self):
        older = self._legacy()
        newer = OptimizationHistoryItem(
            trial_id="new",
            request_id="new",
            iteration=2,
            failure_labels=["retrieval_low_rank"],
            optimizer="rules",
            status="applied",
            action_key="reranker.enabled:enable",
            action_study_key=ActionStudyKey("reranker.enabled:enable", "b", "s"),
        )

        self.assertEqual(
            history.last_action_key([older, newer]),
            "reranker.enabled:enable",
        )
        self.assertEqual(
            history.last_action_study_key([older, newer]).action_key,
            "reranker.enabled:enable",
        )

    def test_attribution_on_a_legacy_item_claims_nothing(self):
        """supporting_labels 가 비어 있으면 해결 라벨을 만들어내지 않는다."""
        result = history.attribute_result(
            self._legacy(),
            _report(["retrieval_missing_gold"]),
            _report([]),
        )

        self.assertEqual(result["resolved_labels"], [])
        self.assertEqual(result["remaining_labels"], [])


if __name__ == "__main__":
    unittest.main()
