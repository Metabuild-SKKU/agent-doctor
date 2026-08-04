"""
tests/test_ragas_fused.py
fused RAGAS 경로(트랙당 chat 1회 + embed 1회)의 계약 고정.

fused 는 '판정을 한 번에 받아오기'만 바꾸고 점수 계산식은 legacy 와 같은 함수를 쓴다.
그래서 여기서 못 박는 건 네 가지다:
  1. 호출 수 — 실제 트랙이 chat 1 + embed 1 로 끝난다(legacy 는 12 + 2).
  2. 점수 — 같은 판정을 주면 legacy 계산식과 같은 값이 나온다.
  3. 순위 복원 — context_precision 은 순위 가중이라, 응답이 순서를 흔들어도 index 로 되돌린다.
  4. 결손 격리/보수 — 블록 하나가 빠져도 나머지 지표는 살고, 빠진 지표만 개별 호출로 메운다.
실제 LLM 은 부르지 않는다(_chat/_embed 스텁).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.eval import metrics_ragas
from agents.eval.types import EvalRecord
from core.schema import Probe


# fused 응답 1건에 담기는 판정 전체. context_verdicts 는 일부러 순서를 섞어 둔다.
_FUSED_PAYLOAD = {
    "answer_statements": ["a1", "a2"],
    "faithfulness_verdicts": [{"statement": "a1", "reason": "r", "verdict": 1},
                              {"statement": "a2", "reason": "r", "verdict": 0}],
    "generated_questions": ["q1", "q2", "q3"],
    "noncommittal": 0,
    "context_verdicts": [{"index": 2, "reason": "r", "verdict": 1},
                         {"index": 0, "reason": "r", "verdict": 1},
                         {"index": 1, "reason": "r", "verdict": 0}],
    "reference_statements": ["r1", "r2"],
    "correctness": {"TP": [{"statement": "t1"}, {"statement": "t2"}],
                    "FP": [{"statement": "f1"}],
                    "FN": [{"statement": "n1"}]},
    "recall_classifications": [{"statement": "r1", "reason": "r", "attributed": 1},
                               {"statement": "r2", "reason": "r", "attributed": 0}],
}

# legacy 개별 호출이 돌 때 각 프롬프트가 받을 응답(보수 경로 검증용).
_LEGACY_REPLIES = [
    ("Perform ALL of the numbered sub-tasks", None),           # fused — 아래에서 따로 처리
    ("analyze the complexity of each sentence", {"statements": ["a1", "a2"]}),
    ("judge the faithfulness of a series of statements",
     {"statements": [{"verdict": 1}, {"verdict": 0}]}),
    ("Generate a question for the given answer", {"question": "q", "noncommittal": 0}),
    ("verify if the context was useful", {"verdict": 1}),
    ("classify if the statement can be attributed",
     {"classifications": [{"attributed": 1}, {"attributed": 0}]}),
    ("classify them in one of the following categories", _FUSED_PAYLOAD["correctness"]),
]


class _Stub:
    """_chat/_embed 대체. 프롬프트 내용으로 어느 호출인지 가려 응답한다(fused 우선)."""

    def __init__(self, fused_payload=_FUSED_PAYLOAD):
        self.fused_payload = fused_payload
        self.chat_calls: list[str] = []
        self.embed_calls: list[list[str]] = []

    def chat(self, judge, prompt, max_output_tokens=None, label=""):
        self.chat_calls.append(prompt)
        for marker, reply in _LEGACY_REPLIES:
            if marker in prompt:
                return dict(self.fused_payload) if reply is None else dict(reply)
        return {}

    def embed(self, judge, texts):
        self.embed_calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]      # 전부 같은 벡터 → 코사인 1.0


def _record():
    return EvalRecord(
        probe=Probe(probe_id="p1", question="질문", source="taxonomy", ground_truth="정답"),
        generated_answer="답변",
        retrieved_context=["c0", "c1", "c2"],
        oracle_answer="오라클 답변",
        oracle_context=["g0"],
    )


class _FusedTestBase(unittest.TestCase):
    """_chat/_embed 스텁 + fused 환경변수를 테스트 단위로 격리한다."""

    ENV = {"EVAL_RAGAS_FUSED": "1", "EVAL_RAGAS_FUSED_REPAIR": "1"}

    def setUp(self):
        self.stub = _Stub()
        self._orig = (metrics_ragas._chat, metrics_ragas._embed)
        metrics_ragas._chat, metrics_ragas._embed = self.stub.chat, self.stub.embed
        self._env = {k: os.environ.get(k) for k in
                     ("EVAL_RAGAS_FUSED", "EVAL_RAGAS_FUSED_REPAIR")}
        os.environ.update(self.ENV)

    def tearDown(self):
        metrics_ragas._chat, metrics_ragas._embed = self._orig
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class FusedCallCountTest(_FusedTestBase):
    def test_real_track_is_one_chat_one_embed(self):
        """실제 트랙 5개 지표가 chat 1회 + embed 1회로 끝난다."""
        out = metrics_ragas.evaluate_real_track(_record(), True)
        self.assertEqual(len(self.stub.chat_calls), 1)
        self.assertEqual(len(self.stub.embed_calls), 1)
        # 임베딩도 한 번에: [질문] + 생성질문 3개 + [정답, 답변]
        self.assertEqual(len(self.stub.embed_calls[0]), 6)
        self.assertEqual(set(out) >= {"faithfulness", "response_relevancy", "context_precision",
                                      "context_recall", "answer_correctness"}, True)

    def test_oracle_track_is_one_chat_one_embed(self):
        """오라클 트랙도 1+1. 검색 품질 지표(precision/recall)는 대상이 아니라 빠진다."""
        out = metrics_ragas.evaluate_oracle_track(_record(), True)
        self.assertEqual(len(self.stub.chat_calls), 1)
        self.assertEqual(len(self.stub.embed_calls), 1)
        self.assertNotIn("context_precision", out)
        self.assertNotIn("context_recall", out)

    def test_legacy_switch_restores_per_metric_calls(self):
        """EVAL_RAGAS_FUSED=0 이면 지표별 개별 호출로 되돌아간다(비교 실행용 스위치)."""
        os.environ["EVAL_RAGAS_FUSED"] = "0"
        metrics_ragas.evaluate_real_track(_record(), True)
        # faithfulness 2 + relevancy 3 + precision 3(청크수) + recall 1 + correctness 3
        self.assertEqual(len(self.stub.chat_calls), 12)
        self.assertEqual(len(self.stub.embed_calls), 2)


class FusedScoreTest(_FusedTestBase):
    def test_scores_match_legacy_formulas(self):
        out = metrics_ragas.evaluate_real_track(_record(), True)
        self.assertAlmostEqual(out["faithfulness"], 0.5)            # 2건 중 1건 지지
        self.assertAlmostEqual(out["context_recall"], 0.5)          # 2건 중 1건 귀속
        self.assertAlmostEqual(out["response_relevancy"], 1.0)      # 스텁 벡터가 전부 동일
        # 순위 [1,0,1] 의 average precision — legacy 와 같은 _average_precision 을 쓴다
        self.assertAlmostEqual(out["context_precision"],
                               metrics_ragas._average_precision([1, 0, 1]))
        # TP2/FP1/FN1 → factual 2/3, 유사도 1.0, 가중 (0.75, 0.25)
        self.assertAlmostEqual(out["answer_correctness"], 0.75 * (2 / 3) + 0.25 * 1.0)
        self.assertEqual((out["answer_correctness_tp"], out["answer_correctness_fp"],
                          out["answer_correctness_fn"]), (2, 1, 1))
        self.assertNotIn("answer_correctness_degraded", out)

    def test_context_precision_restores_rank_order(self):
        """응답이 순서를 흔들어도 index 로 원래 순위를 복원한다(순위 가중 지표라 필수)."""
        shuffled = metrics_ragas._fused_context_precision(_FUSED_PAYLOAD, 3)
        self.assertAlmostEqual(shuffled, metrics_ragas._average_precision([1, 0, 1]))
        # index 순서대로 왔을 때와 같은 값이어야 한다
        ordered = {"context_verdicts": [{"index": i, "verdict": v}
                                        for i, v in enumerate([1, 0, 1])]}
        self.assertAlmostEqual(shuffled, metrics_ragas._fused_context_precision(ordered, 3))

    def test_noncommittal_zeroes_relevancy(self):
        """회피형 답변이면 relevancy 0 — legacy 의 all-noncommittal 규칙과 같다."""
        self.stub.fused_payload = dict(_FUSED_PAYLOAD, noncommittal=1)
        out = metrics_ragas.evaluate_real_track(_record(), True)
        self.assertEqual(out["response_relevancy"], 0.0)


class FusedMissingBlockTest(_FusedTestBase):
    ENV = {"EVAL_RAGAS_FUSED": "1", "EVAL_RAGAS_FUSED_REPAIR": "0"}

    def test_broken_block_does_not_kill_the_others(self):
        """한 블록이 깨져도 나머지 지표는 살아남는다(블록별 독립 파싱)."""
        payload = dict(_FUSED_PAYLOAD)
        payload.pop("faithfulness_verdicts")
        payload["context_verdicts"] = "쓰레기"
        self.stub.fused_payload = payload
        out = metrics_ragas.evaluate_real_track(_record(), True)
        self.assertNotIn("faithfulness", out)          # 보수 off → 미측정
        self.assertNotIn("context_precision", out)
        self.assertAlmostEqual(out["response_relevancy"], 1.0)   # 안 깨진 블록은 그대로 산다
        self.assertAlmostEqual(out["context_recall"], 0.5)
        self.assertIn("answer_correctness", out)
        self.assertEqual(len(self.stub.chat_calls), 1)  # 보수 off 면 추가 호출 없음


class FusedRepairTest(_FusedTestBase):
    def test_empty_response_is_repaired_by_per_metric_calls(self):
        """응답이 통째로 날아가도(파싱 실패) 지표별 개별 호출로 메워 미측정을 막는다."""
        self.stub.fused_payload = {}
        out = metrics_ragas.evaluate_real_track(_record(), True)
        self.assertAlmostEqual(out["faithfulness"], 0.5)
        self.assertAlmostEqual(out["context_recall"], 0.5)
        # relevancy 도 0.0(=완전 무관)으로 굳지 않고 되물어 채운다
        self.assertAlmostEqual(out["response_relevancy"], 1.0)
        self.assertIn("context_precision", out)
        # correctness 는 유사도 단독(degraded)으로도 점수가 나오지만, 강등 근거인 FN 카운트가
        # 없으므로 보수 대상이다 — 보수에 성공하면 degraded 표시가 걷힌다.
        self.assertAlmostEqual(out["answer_correctness"], 0.75 * (2 / 3) + 0.25 * 1.0)
        self.assertNotIn("answer_correctness_degraded", out)
        self.assertEqual(out["answer_correctness_fn"], 1)
        self.assertGreater(len(self.stub.chat_calls), 1)   # fused 1 + 보수 호출들

    def test_partial_response_repairs_only_the_missing_metric(self):
        payload = dict(_FUSED_PAYLOAD)
        payload.pop("recall_classifications")
        self.stub.fused_payload = payload
        out = metrics_ragas.evaluate_real_track(_record(), True)
        self.assertAlmostEqual(out["context_recall"], 0.5)          # 보수됨
        # fused 1 + fused 재요청 1 + recall 보수 1. 재요청도 같은 결손 응답을 돌려주는
        # 스텁이라 개별 보수까지 간다 — 실제로는 재요청에서 채워지면 보수가 안 돈다.
        self.assertEqual(len(self.stub.chat_calls), 3)


class FusedMaterialGuardTest(_FusedTestBase):
    def test_no_answer_short_circuits_relevancy(self):
        """답변이 비면 legacy 와 같이 relevancy 0.0, 그리고 답변 의존 지표는 묻지 않는다."""
        rec = _record()
        rec.generated_answer = ""
        rec.probe.ground_truth = None
        out = metrics_ragas.evaluate_real_track(rec, True)
        self.assertEqual(out["response_relevancy"], 0.0)
        self.assertEqual(self.stub.chat_calls, [])                  # 물을 게 없으면 호출도 없다
        self.assertEqual(self.stub.embed_calls, [])


if __name__ == "__main__":
    unittest.main()
