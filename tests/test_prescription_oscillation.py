"""
tests/test_prescription_oscillation.py
"켰다 껐다"를 막는 두 견제 장치 검증.

실행 로그(corpus_20260804_103059)에서 reranker 를 켜서 종합점수가 0.75→0.78 로
오르고 유지 판정까지 받은 직후, 같은 방문에서 끄는 처방이 선택됐다. 되돌린 결과는
이미 측정해 둔 0.75 였고, 예산 5회 중 3회가 이 왕복에 소모됐다.

원인은 서로 다른 두 구멍이다.
  1. 차단 단위가 **전이**(action, baseline, 후보값)라 도착 지점이 같은 다른 전이를
     막지 못한다. 리랭커를 끄면 후보창 값과 무관하게 같은 동작으로 수렴하는데,
     baseline 지문이 달라 차단이 풀렸다.
  2. 유지 판정 결과를 처방 선택이 읽지 않는다. probe 7개를 살린 변경을 probe 1개짜리
     부작용 라벨이 그대로 되돌렸다.

그래서 두 장치를 **독립적으로** 검증한다. 하나가 꺼져도 다른 하나가 같은 왕복을
막아야 하므로, 각 테스트는 나머지 하나가 발동하지 않는 조건에서 돌린다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.optimize import action_aggregator, history, planner
from agents.optimize.schemas import OptimizationHistoryItem
from core.schema import DiagnosticReport, Finding
from core.state import AgentDoctorState


DISABLE = "reranker.enabled:disable"
ENABLE = "reranker.enabled:enable"
WIDEN = "reranker.candidate_count:increase"


def _finding(probe_id, label):
    return Finding(
        finding_id=f"{probe_id}:{label}",
        type="retrieval_failure",
        severity="warning",
        description=label,
        label=label,
        confirmed=True,
        affected_probes=[probe_id],
    )


def _report(findings, composite=78.0):
    return DiagnosticReport(
        report_id="r",
        findings=list(findings),
        overall_score=0.89,
        composite_score={"total": composite},
        ragas_scores={"context_recall": 0.7},
        pass_threshold=False,
    )


def _config(*, reranker=True, candidates=20):
    return {
        "top_k": 5,
        "chunk_size": 512,
        "chunk_overlap": 50,
        "use_reranker": reranker,
        "rerank_candidates": candidates,
    }


def _kept_item(
    action_key,
    before_config,
    after_config,
    *,
    support=7.0,
    before_score=None,
    after_score=None,
    status="applied",
):
    """확정된 이력 항목 하나.

    점수를 넣지 않으면 measured_config_scores 가 아무것도 기록하지 않는다 —
    되돌리기 견제(장치 2)만 켠 채 무진전 기억(장치 1)을 끄는 데 쓴다.
    """
    metadata = {"pending": False, "weighted_probe_support": support}
    if before_score is not None:
        metadata["before_score"] = before_score
    if after_score is not None:
        metadata["after_score"] = after_score
    return OptimizationHistoryItem(
        trial_id="t1",
        request_id="q1",
        iteration=1,
        failure_labels=["retrieval_low_rank"],
        optimizer="rules",
        status=status,
        selected_prescription_id="enable_reranker",
        before_config=dict(before_config),
        after_config=dict(after_config),
        action_key=action_key,
        supporting_labels=["retrieval_low_rank"],
        supporting_probes=[f"p{i}" for i in range(7)],
        metadata=metadata,
    )


def _state(findings, index_config, optimization_history, composite=78.0):
    return AgentDoctorState(
        report=_report(findings, composite=composite),
        index_config=dict(index_config),
        optimization_history=list(optimization_history),
        iteration=1,
        max_iterations=5,
    )


def _plan(state):
    request, decision = planner.plan(state)
    metadata = request.metadata if request is not None else decision.metadata
    rejected = {
        entry["action_key"]: entry for entry in metadata.get("rejected_actions") or []
    }
    return request, decision, rejected


# ── 0. 도착 config 지문 ───────────────────────────────────────────

class EffectiveConfigViewTest(unittest.TestCase):
    """리랭커가 꺼져 있으면 후보창 값은 검색 동작에 영향이 없다."""

    def test_candidate_count_is_inert_while_the_reranker_is_off(self):
        self.assertEqual(
            history.effective_config_fingerprint(_config(reranker=False, candidates=20)),
            history.effective_config_fingerprint(_config(reranker=False, candidates=22)),
        )

    def test_candidate_count_still_counts_while_the_reranker_is_on(self):
        self.assertNotEqual(
            history.effective_config_fingerprint(_config(reranker=True, candidates=20)),
            history.effective_config_fingerprint(_config(reranker=True, candidates=22)),
        )

    def test_every_executable_action_moves_the_effective_fingerprint(self):
        """실행 가능한 모든 action 의 축은 config 뷰에 보여야 한다.

        보이지 않는 축을 바꾸면 도착 지문이 현재와 같아지고, 현재 config 는 (상승폭
        0 이라) 항상 무진전 집합에 들어간다 — 그 action 이 영원히 '이미 재 봤다'로
        막힌다. 지금은 성립하지만 축을 추가하며 조용히 깨질 수 있어 고정한다.
        """
        from agents.optimize import action_catalog
        from agents.optimize.config_mapper import CONFIG_READ_PATHS

        invisible = [
            action.key
            for action in action_catalog.executable_actions()
            if action.canonical_path not in CONFIG_READ_PATHS
        ]

        self.assertEqual(invisible, [])

    def test_missing_gate_axis_does_not_erase_the_candidate_count(self):
        """리랭커 축이 config 에 없으면 기본값을 추측하지 않는다.

        추측해서 비활성으로 단정하면 서로 다른 동작의 config 두 개가 같은 지문을
        갖게 되고, 그 순간 정당한 시도까지 '이미 재 봤다'로 막힌다.
        """
        self.assertNotEqual(
            history.effective_config_fingerprint({"rerank_candidates": 20}),
            history.effective_config_fingerprint({"rerank_candidates": 22}),
        )


# ── 1. 무진전 config 기억 ─────────────────────────────────────────

class NoProgressConfigMemoTest(unittest.TestCase):
    """이미 재 봤고 개선이 아니었던 config 로는 다시 가지 않는다."""

    def test_measured_scores_keep_the_most_favourable_observation(self):
        """같은 config 를 두 번 쟀으면 좋았던 쪽을 남긴다 — 노이즈 한 번으로 축을 닫지 않는다."""
        off = _config(reranker=False)
        items = [
            _kept_item(ENABLE, off, _config(), before_score=0.75, after_score=0.78),
            _kept_item(ENABLE, off, _config(), before_score=0.73, after_score=0.78),
        ]

        scores = history.measured_config_scores(items)

        self.assertEqual(scores[history.effective_config_fingerprint(off)], 0.75)

    def test_unjudgeable_measurements_are_not_remembered(self):
        """측정이 성립하지 않은 항목의 점수는 재적용 차단 근거가 못 된다."""
        item = _kept_item(
            ENABLE, _config(reranker=False), _config(), before_score=0.75, after_score=0.78
        )
        item.metadata["unjudgeable"] = True

        self.assertEqual(history.measured_config_scores([item]), {})

    def test_reverting_a_kept_change_is_blocked(self):
        """로그의 방문 1 — 0.78 에서 0.75 로 되돌아가는 처방은 고르지 않는다."""
        state = _state(
            [_finding("p10", "retrieval_reranker_demotion"),
             _finding("p11", "generation_hallucination")],
            _config(),
            [_kept_item(
                ENABLE,
                _config(reranker=False),
                _config(),
                support=0.0,   # 되돌리기 견제를 끄고 기억만 시험한다
                before_score=0.75,
                after_score=0.78,
            )],
        )

        request, _decision, rejected = _plan(state)

        self.assertNotEqual(request.action_key, DISABLE)
        self.assertEqual(
            rejected[DISABLE]["reason"], action_aggregator.REASON_NO_PROGRESS_CONFIG
        )

    def test_a_different_baseline_does_not_reopen_the_same_destination(self):
        """로그의 방문 4 — 후보창만 22 로 바뀌어도 '끄기'의 도착 지점은 그대로다.

        전이 단위 차단은 baseline 지문이 달라져 풀린다. 도착 config 로 기억해야
        같은 실패를 다시 재지 않는다.
        """
        state = _state(
            [_finding("p10", "retrieval_reranker_demotion"),
             _finding("p11", "generation_hallucination")],
            _config(candidates=22),
            [_kept_item(
                WIDEN,
                _config(candidates=20),
                _config(candidates=22),
                support=0.0,
                before_score=0.78,
                after_score=0.80,
            ),
             _kept_item(
                ENABLE,
                _config(reranker=False, candidates=20),
                _config(candidates=20),
                support=0.0,
                before_score=0.75,
                after_score=0.78,
            )],
            composite=80.0,
        )

        request, _decision, rejected = _plan(state)

        self.assertNotEqual(request.action_key, DISABLE)
        self.assertEqual(
            rejected[DISABLE]["reason"], action_aggregator.REASON_NO_PROGRESS_CONFIG
        )

    def test_returning_to_a_clearly_better_config_stays_open(self):
        """기억은 금지 목록이 아니다 — 더 좋았던 config 로 되돌아가는 길은 열어 둔다."""
        fingerprints = history.no_progress_config_fingerprints(
            [_kept_item(
                ENABLE,
                _config(reranker=False),
                _config(),
                before_score=0.90,
                after_score=0.70,
            )],
            _config(),
        )

        self.assertNotIn(
            history.effective_config_fingerprint(_config(reranker=False)), fingerprints
        )

    def test_the_restored_baseline_sets_the_bar_after_a_rollback(self):
        """롤백 직후엔 열화된 리포트가 아니라 복원된 config 의 관측값이 기준이다.

        열화 점수를 기준으로 삼으면 문턱이 낮아져, 이미 실패한 config 가 다시 통과한다.
        """
        restored = _config()
        item = _kept_item(
            ENABLE, _config(reranker=False), restored, before_score=0.75, after_score=0.78
        )

        fingerprints = history.no_progress_config_fingerprints(
            [item],
            restored,
            _report([], composite=67.0),   # 되돌리기 전의 열화된 Eval
        )

        self.assertIn(
            history.effective_config_fingerprint(_config(reranker=False)), fingerprints
        )


# ── 2. 되돌리기 견제 ──────────────────────────────────────────────

class ReversalGuardTest(unittest.TestCase):
    """유지 판정된 변경은 그보다 넓은 지지가 있을 때만 되돌린다."""

    def test_narrow_support_cannot_undo_a_kept_change(self):
        """probe 1개짜리 부작용이 probe 7개짜리 이득을 되돌리지 못한다."""
        state = _state(
            [_finding("p10", "retrieval_reranker_demotion"),
             _finding("p11", "generation_hallucination")],
            _config(),
            # 점수를 넣지 않아 무진전 기억은 비어 있다 — 견제 장치만 시험한다.
            [_kept_item(ENABLE, _config(reranker=False), _config(), support=7.0)],
        )

        request, _decision, rejected = _plan(state)

        self.assertNotEqual(request.action_key, DISABLE)
        self.assertEqual(
            rejected[DISABLE]["reason"], action_aggregator.REASON_KEEP_PROTECTED_AXIS
        )
        # "얼마가 필요했는데 얼마였다"가 남아야 사용자가 판단을 되짚을 수 있다.
        self.assertEqual(rejected[DISABLE]["keep_protection"]["required_support"], 7.0)

    def test_broader_support_is_allowed_to_undo_a_kept_change(self):
        """진단이 실제로 뒤집힌 경우까지 막지는 않는다 — 문턱이지 금지가 아니다."""
        state = _state(
            [_finding(f"p{i}", "retrieval_reranker_demotion") for i in range(4)],
            _config(),
            [_kept_item(ENABLE, _config(reranker=False), _config(), support=1.0)],
        )

        request, _decision, _rejected = _plan(state)

        self.assertEqual(request.action_key, DISABLE)

    def test_equal_support_does_not_undo_a_kept_change(self):
        """동률이면 실측으로 이득이 확인된 쪽을 남긴다."""
        guards = history.reversal_guard_thresholds(
            [_kept_item(ENABLE, _config(reranker=False), _config(), support=1.0)],
            _config(),
        )
        candidates, rejected = action_aggregator.filter_keep_protected_actions(
            [_stub_candidate(DISABLE, support=1.0)], guards
        )

        self.assertEqual(candidates, [])
        self.assertEqual(rejected[0].reason, action_aggregator.REASON_KEEP_PROTECTED_AXIS)

    def test_an_overwritten_axis_is_no_longer_protected(self):
        """그 뒤 다른 처방이 축을 덮었으면 지킬 이득이 이미 없다."""
        guards = history.reversal_guard_thresholds(
            [_kept_item(ENABLE, _config(reranker=False), _config(reranker=True))],
            _config(reranker=False),   # 현재는 다시 꺼져 있다
        )

        self.assertEqual(guards, {})

    def test_a_rolled_back_change_is_not_protected(self):
        """롤백된 처방은 이득이 확인되지 않았다 — 지킬 대상이 아니다."""
        guards = history.reversal_guard_thresholds(
            [_kept_item(
                ENABLE, _config(reranker=False), _config(), status="failed"
            )],
            _config(),
        )

        self.assertEqual(guards, {})

    def test_same_direction_actions_are_untouched(self):
        """견제 대상은 되돌리기뿐이다 — 같은 방향으로 더 미는 처방은 막지 않는다."""
        guards = history.reversal_guard_thresholds(
            [_kept_item(
                WIDEN, _config(candidates=20), _config(candidates=22), support=5.0
            )],
            _config(candidates=22),
        )

        self.assertEqual(list(guards), ["reranker.candidate_count:decrease"])


# ── 3. 방문을 이어 붙인 재현 ──────────────────────────────────────

class OscillationReplayTest(unittest.TestCase):
    """로그의 방문 순서를 그대로 태워, 축이 왕복하지 않는지 본다.

    단위 테스트는 각 장치를 따로 확인한다. 여기서는 판정(judge)→선택(plan)이
    이어지는 실제 경로에서 두 장치가 함께 도는지를 본다 — 유지 판정이 이력에
    남고, 그 이력이 다음 방문의 선택을 실제로 견제하는지.
    """

    def _report(self, findings, composite):
        return DiagnosticReport(
            report_id="r",
            findings=list(findings),
            overall_score=0.89,
            composite_score={"total": composite},
            ragas_scores={"context_recall": 0.7, "mean_recall_at_k": 0.7},
            # 리랭커 처방은 실제 실행 여부를 확인해야 판정이 성립한다.
            runtime_summary={"reranker": {"enabled": True, "attempted": 30, "applied": 30}},
            pass_threshold=False,
        )

    def test_the_reranker_axis_never_flips_back(self):
        from agents.optimize import agent

        state = AgentDoctorState(
            report=self._report(
                [_finding(f"p{i}", "retrieval_low_rank") for i in range(7)], 75.0
            ),
            index_config=_config(reranker=False),
            iteration=0,
            max_iterations=5,
            runtime_capabilities={
                "reranker": {"status": "verified", "model": "BAAI/bge-reranker-v2-m3"}
            },
        )

        # 방문 0: low_rank 7건 → 리랭커를 켠다.
        agent.run(state)
        self.assertTrue(state.index_config["use_reranker"])

        # 방문 1: 0.75→0.78 로 유지 판정. 같은 방문에서 demotion 1건이 끄기를 지지한다.
        state.report = self._report(
            [_finding("p10", "retrieval_reranker_demotion"),
             _finding("p11", "generation_hallucination")],
            78.0,
        )
        agent.run(state)
        self.assertTrue(state.index_config["use_reranker"])
        self.assertTrue(state.index_config.get("abstention_strict"))

        # 방문 2: 그 처방이 점수를 떨어뜨려 롤백. 롤백 방문에서도 견제는 살아 있다.
        state.report = self._report(
            [_finding("p10", "retrieval_reranker_demotion"),
             _finding("p12", "retrieval_rerank_candidate_miss"),
             _finding("p13", "generation_wrongful_abstention")],
            67.0,
        )
        agent.run(state)
        self.assertTrue(state.index_config["use_reranker"])
        self.assertFalse(state.index_config.get("abstention_strict"))

        # 방문 3: 후보창 확대가 유지된다. baseline 이 움직여도 끄기는 여전히 막힌다 —
        # 로그에서 마지막 iteration 을 통째로 날린 지점이 여기다.
        state.report = self._report(
            [_finding("p10", "retrieval_reranker_demotion"),
             _finding("p14", "generation_misinterpretation"),
             _finding("p15", "generation_misinterpretation")],
            80.0,
        )
        agent.run(state)
        self.assertTrue(state.index_config["use_reranker"])

        applied = [item.action_key for item in state.optimization_history]
        self.assertNotIn(DISABLE, applied)


def _stub_candidate(action_key, *, support):
    """지지 크기만 채운 최소 후보. 견제 판정은 그 값 하나만 본다."""
    from agents.optimize import action_catalog
    from agents.optimize.schemas import ActionCandidate

    definition = action_catalog.get_action(action_key)
    return ActionCandidate(
        action_key=action_key,
        definition=definition,
        score_breakdown={"weighted_probe_support": support},
    )


if __name__ == "__main__":
    unittest.main()
