"""
tests/test_correctness_counts.py
answer_correctness 의 TP/FP/FN 카운트 노출 계약 고정.

카운트는 generation_partial_answer 의 판별 근거다(FN = gold 에만 있는 누락 요소).
여기서는 세 가지를 못 박는다:
  1. factual 성분이 측정되면 카운트가 트랙 dict 에 실린다 (LLM 추가 호출 없음 — 이미 계산되던 값).
  2. degraded(factual 실패) 면 카운트가 없고, answer_correctness 점수 자체는 영향받지 않는다.
  3. 새 키가 리포트/스코어링 평균에 섞이지 않는다 (둘 다 키 allowlist 순회라는 보장).
"""
import contextlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.eval import metrics_ragas, report, scoring, metrics_common
from agents.eval.types import EvalRecord, Mode
from core.schema import Probe


def _record(**oracle_ragas):
    rec = EvalRecord(
        probe=Probe(probe_id="p1", question="질문", source="taxonomy", ground_truth="정답"),
        generated_answer="답변",
        oracle_answer="오라클 답변",
    )
    rec.oracle_ragas = dict(oracle_ragas)
    rec.oracle_ragas_done = True
    rec.ragas_done = True
    return rec


class _StubJudge:
    """_chat/_embed 를 대체해 TP/FP/FN 분류 결과만 흉내낸다(실제 LLM 호출 없음)."""

    def __init__(self, tp, fp, fn, *, statements=True, embed=True):
        self.payload = {
            "TP": [{"statement": f"tp{i}", "reason": "r"} for i in range(tp)],
            "FP": [{"statement": f"fp{i}", "reason": "r"} for i in range(fp)],
            "FN": [{"statement": f"fn{i}", "reason": "r"} for i in range(fn)],
        }
        self.statements = statements
        self.embed = embed

    def chat(self, judge, prompt, label="", max_output_tokens=None):
        if "statements" in prompt and "TP" not in prompt:      # StatementGenerator 호출
            return {"statements": ["s1", "s2"]} if self.statements else {}
        return self.payload

    def embed_fn(self, judge, texts):
        if not self.embed:
            raise RuntimeError("embed 실패")
        return [[1.0, 0.0], [1.0, 0.0]]


class _StubbedRagas(unittest.TestCase):
    def _run(self, stub):
        orig_chat, orig_embed = metrics_ragas._chat, metrics_ragas._embed
        metrics_ragas._chat, metrics_ragas._embed = stub.chat, stub.embed_fn
        try:
            return metrics_ragas._answer_correctness(None, "질문", "답변", "정답")
        finally:
            metrics_ragas._chat, metrics_ragas._embed = orig_chat, orig_embed


class CorrectnessCountsTest(_StubbedRagas):
    def test_counts_exposed_when_factual_measured(self):
        out = self._run(_StubJudge(2, 1, 3))
        self.assertEqual(out["answer_correctness_tp"], 2)
        self.assertEqual(out["answer_correctness_fp"], 1)
        self.assertEqual(out["answer_correctness_fn"], 3)
        self.assertNotIn("answer_correctness_degraded", out)

    def test_zero_fn_is_kept_not_dropped(self):
        """FN=0 은 '누락 없음'이라는 정보다 — 0 이라고 사라지면 안 된다."""
        out = self._run(_StubJudge(3, 0, 0))
        self.assertEqual(out["answer_correctness_fn"], 0)

    def test_counts_absent_when_factual_degraded(self):
        """분류가 한 건도 안 나옴(판정기 실패) → 카운트 없음 + degraded 표시."""
        out = self._run(_StubJudge(0, 0, 0))
        self.assertNotIn("answer_correctness_fn", out)
        self.assertTrue(out["answer_correctness_degraded"])
        self.assertIn("answer_correctness", out)          # 유사도 성분으로 점수는 유지

    def test_score_unchanged_by_count_exposure(self):
        """카운트 노출은 순수 추가 — answer_correctness 값 자체는 그대로다."""
        out = self._run(_StubJudge(2, 1, 3))
        w_f, w_s = metrics_ragas._ANSWER_CORRECTNESS_WEIGHTS
        factual = 2 / (2 + 0.5 * (1 + 3))
        self.assertAlmostEqual(out["answer_correctness"],
                               (w_f * factual + w_s * 1.0) / (w_f + w_s))


class DegradeReasonLogTest(_StubbedRagas):
    """degrade·미측정의 '원인'이 로그로 남는지 — 리포트엔 건수만 남아 원인 추적이 안 됐다."""

    def _log(self, stub):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = self._run(stub)
        return out, buf.getvalue()

    def test_classification_failure_reason_logged(self):
        out, log = self._log(_StubJudge(0, 0, 0))
        self.assertTrue(out["answer_correctness_degraded"])
        self.assertIn("TP/FP/FN 분류 무응답", log)

    def test_decomposition_empty_reason_logged_without_calling_it_a_failure(self):
        """기권 답변은 0문장이 정상 분해 결과다 — '실패'로 단정하면 안 된다."""
        out, log = self._log(_StubJudge(1, 0, 0, statements=False))
        self.assertTrue(out["answer_correctness_degraded"])
        self.assertIn("분해 결과 없음", log)
        self.assertNotIn("분해 실패", log)

    def test_both_components_failed_still_logs_cause(self):
        """두 성분 다 죽으면 degraded 플래그가 안 붙어 리포트 집계에도 안 잡힌다 —
        여기서 안 남기면 의미축이 왜 없는지 로그에서 아예 못 찾는다."""
        out, log = self._log(_StubJudge(0, 0, 0, embed=False))
        self.assertEqual(out, {})
        self.assertIn("answer_correctness 미측정", log)
        self.assertIn("임베딩 실패", log)

    def test_measured_run_logs_nothing(self):
        _, log = self._log(_StubJudge(2, 1, 3))
        self.assertEqual(log, "")


class CorrectnessCountsAccessorTest(unittest.TestCase):
    def tearDown(self):
        metrics_common.set_mode(Mode.FAST)

    def test_returns_counts_at_deep(self):
        metrics_common.set_mode(Mode.DEEP)
        rec = _record(answer_correctness_tp=2, answer_correctness_fp=1, answer_correctness_fn=3)
        self.assertEqual(metrics_ragas._correctness_counts_oracle(rec), (2, 1, 3))

    def test_none_below_deep(self):
        metrics_common.set_mode(Mode.STANDARD)
        rec = _record(answer_correctness_tp=2, answer_correctness_fp=1, answer_correctness_fn=3)
        self.assertIsNone(metrics_ragas._correctness_counts_oracle(rec))

    def test_none_when_degraded(self):
        metrics_common.set_mode(Mode.DEEP)
        rec = _record(answer_correctness=0.4, answer_correctness_degraded=True)
        self.assertIsNone(metrics_ragas._correctness_counts_oracle(rec))


class CountsDoNotPolluteAveragesTest(unittest.TestCase):
    """리포트·스코어링은 키 allowlist 로만 순회한다 — 카운트가 RAGAS 평균에 섞이면 안 된다."""

    def _records(self):
        rec = _record()
        rec.ragas = {"faithfulness": 0.8, "response_relevancy": 0.6,
                     "answer_correctness_tp": 2, "answer_correctness_fp": 1,
                     "answer_correctness_fn": 3}
        rec.ragas_done = True
        return [rec]

    def test_ragas_means_ignores_counts(self):
        means = report._ragas_means(self._records())
        self.assertEqual(set(means), {"faithfulness", "response_relevancy"})

    def test_quality_score_ignores_counts(self):
        score = scoring.quality_score(self._records())
        self.assertIsNotNone(score)
        self.assertLessEqual(score, 1.0)          # 카운트(2·1·3)가 섞였다면 1.0 을 넘는다


if __name__ == "__main__":
    unittest.main()
