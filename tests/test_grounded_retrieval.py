"""
tests/test_grounded_retrieval.py
검증된 label-recall miss(16a) 처리 검증 — 재청킹 recall 스윙으로 pass/fail 이 뒤집히는 문제.

라벨 골드는 top-k 에 못 들었지만(recall<1) 답이 정답이고 검색 근거에 붙었고(faithfulness↑)
골드도 유효하면(oracle 통과), 검색은 '다른 유효 근거'로 정답을 뒷받침한 것이다. 이때는 실패가
아니라 성공으로 처리하고 검색축 크레딧(faithfulness)을 record.retrieval_axis 에 남겨 reliability
와 pass/fail 이 같은 판정을 쓰게 한다.

  · parametric(근거 없이 맞힌 답, faithfulness↓)  → 크레딧 없음, 검색 실패로 남는다.
  · 골드 오류(oracle 실패)                        → bad_gold_chunk 가 먼저 가져간다.
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import Probe
from agents.eval import metrics_common, diagnose
from agents.eval.scoring import _probe_reliability
from agents.eval.types import EvalRecord, Mode


def _rec(*, recall, f1, oracle_f1, faith):
    probe = Probe(probe_id="p1", question="질문", source="taxonomy",
                  ground_truth="정답", gold_chunk_ids=["g_a"])
    rec = EvalRecord(probe=probe, generated_answer="정답", oracle_answer="오라클 답")
    rec.recall_at_k = recall
    rec.f1_score = f1
    rec.oracle_f1 = oracle_f1
    rec.ragas = {"faithfulness": faith}
    rec.ragas_done = True
    rec.oracle_ragas = {}
    rec.oracle_ragas_done = True
    return rec


class GroundedCreditDiagnoseTest(unittest.TestCase):
    """diagnose 통합: 지표·RAGAS 재계산은 주입값 보존을 위해 no-op 로 둔다."""

    def setUp(self):
        metrics_common.set_mode(Mode.DEEP)

    def tearDown(self):
        metrics_common.set_mode(Mode.FAST)

    def _diagnose(self, rec):
        with patch.object(diagnose, "_compute_metrics"), \
             patch.object(diagnose, "_compute_ragas_real"), \
             patch.object(diagnose, "_compute_ragas_oracle"):
            return diagnose.diagnose(rec, mode=int(Mode.DEEP))

    def test_grounded_label_recall_miss_is_success(self):
        # 답 정답 + 골드 유효(oracle↑) + 검색 근거에 붙음(faith↑) + recall 낮음 → 성공(findings 없음)
        rec = _rec(recall=0.0, f1=1.0, oracle_f1=1.0, faith=0.9)
        findings = self._diagnose(rec)
        self.assertEqual(findings, [])                       # retrieval_low_rank 안 붙음
        self.assertAlmostEqual(rec.retrieval_axis, 0.9)      # 검색축 크레딧 = faithfulness

    def test_parametric_stays_failure(self):
        # 답은 맞지만 검색 근거에 안 붙음(faith↓) = parametric → 크레딧 없음, 검색 실패로 남는다
        rec = _rec(recall=0.0, f1=1.0, oracle_f1=1.0, faith=0.1)
        findings = self._diagnose(rec)
        self.assertTrue(any(f.label.startswith("retrieval") for f in findings))
        self.assertIsNone(rec.retrieval_axis)                # 크레딧 없음 → recall 그대로

    def test_gold_error_goes_to_bad_gold_chunk_not_credit(self):
        # 골드 오류(oracle 실패)면 bad_gold_chunk 가 먼저 가져간다 — 검색축 크레딧 대상 아님
        rec = _rec(recall=0.0, f1=1.0, oracle_f1=0.0, faith=0.9)
        findings = self._diagnose(rec)
        self.assertEqual([f.label for f in findings], ["bad_gold_chunk"])
        self.assertIsNone(rec.retrieval_axis)


class GroundedCreditReliabilityTest(unittest.TestCase):
    """reliability 가 pass/fail 과 같은 판정(retrieval_axis)을 쓴다."""

    def _rec(self, *, recall, answer_f1, axis=None):
        probe = Probe(probe_id="p1", question="q", source="taxonomy", ground_truth="gt")
        rec = EvalRecord(probe=probe)
        rec.recall_at_k = recall
        rec.f1_score = answer_f1
        rec.retrieval_axis = axis
        return rec

    def test_uses_recall_when_no_axis(self):
        # 크레딧 없으면 기존대로 recall × answer
        rec = self._rec(recall=0.0, answer_f1=1.0)
        self.assertEqual(_probe_reliability(rec), 0.0)       # 0.0 × 1.0

    def test_uses_axis_when_set(self):
        # 크레딧 있으면 검색축이 axis(=faithfulness) → 신뢰도가 recall 0 에 짓눌리지 않는다
        rec = self._rec(recall=0.0, answer_f1=1.0, axis=0.9)
        self.assertAlmostEqual(_probe_reliability(rec), 0.9)  # 0.9 × 1.0

    def test_axis_does_not_exceed_answer(self):
        # 검색축이 살아도 답변축이 낮으면 곱이라 신뢰도도 낮다(두 축 AND 유지)
        rec = self._rec(recall=0.0, answer_f1=0.4, axis=0.9)
        self.assertLess(_probe_reliability(rec), 0.5)


if __name__ == "__main__":
    unittest.main()
