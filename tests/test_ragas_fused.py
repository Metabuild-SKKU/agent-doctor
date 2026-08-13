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
import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.eval import llm_provider, metrics_ragas
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
        self.schemas: list[dict | None] = []
        self.systems: list[str] = []
        self.batchable: list[bool] = []

    def chat(self, judge, prompt, max_output_tokens=None, label="",
             cache_prefix="", json_schema=None, batchable=False):
        # fused 경로는 지시문을 cache_prefix(anthropic 에서 캐시 지점)로, 입력만
        # prompt 로 보낸다. 어느 호출인지 가리는 마커는 지시문 쪽에 있으므로 둘을
        # 합쳐서 본다 — prompt 만 보면 fused 호출을 legacy 로 오인한다.
        self.chat_calls.append(cache_prefix + prompt)
        self.schemas.append(json_schema)
        self.systems.append(cache_prefix)
        self.batchable.append(batchable)
        for marker, reply in _LEGACY_REPLIES:
            if marker in cache_prefix + prompt:
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
    """_chat/_embed 스텁 + fused 환경변수를 테스트 단위로 격리한다.

    EVAL_LLM_PROVIDER 를 고정하는 이유: compact 변형(_compact_enabled)이 provider 로
    갈리므로, 개발 머신 .env 가 anthropic 이면 레거시 계약 테스트가 조용히 compact
    프롬프트를 검증하게 된다."""

    ENV = {"EVAL_RAGAS_FUSED": "1", "EVAL_RAGAS_FUSED_REPAIR": "1",
           "EVAL_LLM_PROVIDER": "openrouter"}

    def setUp(self):
        self.stub = _Stub()
        self._orig = (metrics_ragas._chat, metrics_ragas._embed)
        metrics_ragas._chat, metrics_ragas._embed = self.stub.chat, self.stub.embed
        self._env = {k: os.environ.get(k) for k in self.ENV}
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


_REAL_BLOCKS = ["answer_statements", "faithfulness_verdicts", "relevancy",
                "context_verdicts", "reference_statements", "correctness",
                "recall_classifications"]
_ORACLE_BLOCKS = ["answer_statements", "faithfulness_verdicts", "relevancy",
                  "reference_statements", "correctness"]


class FusedJsonSchemaTest(unittest.TestCase):
    """anthropic output_config.format 에 실리는 스키마의 유효성.

    이 스키마가 틀리면 심판 호출이 400 으로 죽거나(구조 위반), 강제가 헐거워져
    결손 보수 경로가 되살아난다(이 transport 를 만든 이유가 사라진다)."""

    def _walk_objects(self, node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                yield node
            for value in node.values():
                yield from self._walk_objects(value)
        elif isinstance(node, list):
            for value in node:
                yield from self._walk_objects(value)

    def test_block_and_schema_tables_are_in_sync(self):
        # 두 표가 갈라지면 프롬프트가 요구하는 키와 스키마가 강제하는 키가 어긋난다.
        self.assertEqual(set(metrics_ragas._FUSED_BLOCKS),
                         set(metrics_ragas._FUSED_SCHEMA_PROPS))

    def test_required_matches_prompt_required(self):
        for blocks in (_REAL_BLOCKS, _ORACLE_BLOCKS):
            with self.subTest(n=len(blocks)):
                schema = metrics_ragas._fused_json_schema(blocks)
                self.assertEqual(schema["required"],
                                 metrics_ragas._fused_required_keys(blocks))

    def test_every_object_closes_and_requires_all(self):
        # additionalProperties: false 와 required 전량은 Anthropic structured outputs 의
        # 필수 조건이다. 하나라도 빠지면 요청이 400 이다.
        for obj in self._walk_objects(metrics_ragas._fused_json_schema(_REAL_BLOCKS)):
            self.assertIs(obj.get("additionalProperties"), False, obj)
            self.assertEqual(set(obj["required"]), set(obj["properties"]), obj)

    def test_no_unsupported_keywords(self):
        # minimum/maxLength 류 수치·길이 제약과 $ref 는 미지원이다.
        text = json.dumps(metrics_ragas._fused_json_schema(_REAL_BLOCKS))
        for keyword in ("minimum", "maximum", "minLength", "maxLength",
                        "multipleOf", "$ref"):
            self.assertNotIn(keyword, text)

    def test_binary_verdicts_use_enum(self):
        # 0/1 판정은 minimum/maximum 을 못 쓰므로 enum 으로 좁힌다.
        schema = metrics_ragas._fused_json_schema(_REAL_BLOCKS)
        verdict = (schema["properties"]["faithfulness_verdicts"]
                   ["items"]["properties"]["verdict"])
        self.assertEqual(verdict, {"type": "integer", "enum": [0, 1]})
        self.assertEqual(schema["properties"]["noncommittal"]["enum"], [0, 1])

    def test_no_blocks_means_no_schema(self):
        self.assertIsNone(metrics_ragas._fused_json_schema([]))


class FusedPromptSplitTest(unittest.TestCase):
    """캐시 지점 분할 — 프리픽스가 매 호출 동일해야 캐시가 걸린다."""

    def test_split_is_lossless(self):
        # 다른 provider 는 두 부분을 이어붙여 예전과 같은 한 덩어리를 받는다.
        prefix, suffix = metrics_ragas._fused_prompt_parts(
            _REAL_BLOCKS, "질문", "답변", ["컨텍스트"], "정답")
        self.assertEqual(
            prefix + suffix,
            metrics_ragas._fused_prompt(_REAL_BLOCKS, "질문", "답변",
                                        ["컨텍스트"], "정답"))

    def test_variable_input_never_leaks_into_prefix(self):
        # 가변 내용이 한 바이트라도 앞에 끼면 캐시가 probe 마다 무효화된다.
        prefix, _ = metrics_ragas._fused_prompt_parts(
            _REAL_BLOCKS, "질문", "답변", ["컨텍스트"], "정답")
        for leaked in ("질문", "답변", "컨텍스트", "정답", "input:"):
            self.assertNotIn(leaked, prefix)

    def test_prefix_is_stable_across_inputs(self):
        first, _ = metrics_ragas._fused_prompt_parts(
            _REAL_BLOCKS, "q1", "a1", ["c1"], "r1")
        second, _ = metrics_ragas._fused_prompt_parts(
            _REAL_BLOCKS, "완전히 다른 질문", "다른 답변", ["다른 컨텍스트"], "다른 정답")
        self.assertEqual(first, second)

    def test_prefix_clears_sonnet5_cache_minimum(self):
        """캐시 최소치(sonnet-5 는 1024 토큰) 미달이면 에러 없이 캐시가 무시된다.

        토큰 수를 오프라인에서 정확히 알 수는 없으므로 글자수÷4 를 **하한**으로만 쓴다.
        실측(count_tokens, 스키마 포함)은 실제 4,325 / 오라클 3,211 토큰으로 이 어림보다
        1.35 배 크다 — 즉 이 단언을 통과하면 실제로도 통과한다.

        어림값으로 모델별 최소치를 판정하지 말 것: haiku-4-5 / opus-4-6 의 최소 4096 은
        실측 기준으로 실제 트랙만 넘고 오라클 트랙은 미달이라 한쪽만 캐시된다."""
        for blocks in (_REAL_BLOCKS, _ORACLE_BLOCKS):
            prefix, _ = metrics_ragas._fused_prompt_parts(
                blocks, "q", "a", ["c"], "r")
            with self.subTest(n=len(blocks)):
                self.assertGreater(len(prefix) // 4, 1024)

    def test_real_prefix_is_larger_than_oracle(self):
        # 블록이 더 많으니 당연하지만, 이 순서가 뒤집히면 위 실측 표(4,325 > 3,211)와
        # 모델별 캐시 판정이 통째로 어긋난다.
        real, _ = metrics_ragas._fused_prompt_parts(_REAL_BLOCKS, "q", "a", ["c"], "r")
        oracle, _ = metrics_ragas._fused_prompt_parts(_ORACLE_BLOCKS, "q", "a", ["c"], "r")
        self.assertGreater(len(real), len(oracle))


class CachePrefixIsProviderScopedTest(unittest.TestCase):
    """캐시 지점 분리가 anthropic 밖으로 새지 않는지.

    cache_prefix 를 그냥 system 에 실으면 지시문이 user → system 으로 역할이 바뀌어,
    캐싱과 무관한 provider 의 판정 분포까지 건드리게 된다. 여기서 막는다."""

    def _sent(self, provider):
        seen = {}

        def fake(system, user, model, **kw):
            seen.update(system=system, user=user)
            return "{}"

        env = {"EVAL_LLM_PROVIDER": provider,
               "ANTHROPIC_API_KEY": "k", "OPENROUTER_API_KEY": "k",
               "OPENAI_API_KEY": "k", "GEMINI_API_KEY": "k"}
        target = {"anthropic": "anthropic_chat", "openrouter": "openai_chat",
                  "openai": "openai_chat", "gemini": "gemini_chat"}[provider]
        with patch.dict(os.environ, env, clear=False), \
                patch.object(llm_provider, target, fake), \
                redirect_stdout(io.StringIO()):
            llm_provider.chat_json("", "INPUT", cache_prefix="PREFIX")
        return seen

    def test_anthropic_gets_prefix_as_system(self):
        sent = self._sent("anthropic")
        self.assertEqual(sent["system"], "PREFIX")
        self.assertEqual(sent["user"], "INPUT")

    def test_other_providers_get_one_concatenated_user_message(self):
        # 예전(분리 전)과 바이트가 같아야 한다 — system 은 비어 있고 user 는 한 덩어리.
        for provider in ("openrouter", "openai", "gemini"):
            with self.subTest(provider=provider):
                sent = self._sent(provider)
                self.assertEqual(sent["system"], "")
                self.assertEqual(sent["user"], "PREFIXINPUT")


class FusedSchemaReachesTransportTest(_FusedTestBase):
    """스키마가 실제로 호출까지 도달하는지 — 조립만 되고 안 실리면 의미가 없다."""

    def test_fused_call_carries_schema_and_cached_prefix(self):
        metrics_ragas.evaluate_real_track(_record(), object())
        self.assertEqual(len(self.stub.schemas), 1)
        schema = self.stub.schemas[0]
        self.assertIsNotNone(schema)
        self.assertEqual(schema["required"],
                         metrics_ragas._fused_required_keys(_REAL_BLOCKS))
        # 지시문은 cache_prefix(캐시 지점), 입력은 prompt 로 갈려야 한다.
        self.assertIn("evaluation judge", self.stub.systems[0])
        self.assertNotIn("질문", self.stub.systems[0])

    def test_only_the_fanned_out_fused_call_is_batchable(self):
        """배치 opt-in 은 fused 호출 하나뿐이다.

        보수(_fused_repair)로 되살아나는 legacy 개별 호출까지 배치로 보내면, 드물게
        도는 그 호출 하나가 배치 턴어라운드(수 분)를 통째로 기다린다 — 아끼는 돈보다
        잃는 시간이 크다. Phase C 순차 판정(기권·추론모드)도 같은 이유로 제외한다."""
        stub = _Stub(fused_payload={})           # 빈 응답 → 전 지표 보수 호출로 흘러간다
        metrics_ragas._chat = stub.chat
        with redirect_stdout(io.StringIO()):
            metrics_ragas.evaluate_real_track(_record(), object())
        self.assertGreater(len(stub.batchable), 1)          # fused 1 + 보수 n
        self.assertTrue(stub.batchable[0])                  # fused 만 True
        self.assertFalse(any(stub.batchable[1:]))


# compact 응답 — correctness 는 인덱스 배열, faithfulness 는 reason 없음.
_COMPACT_PAYLOAD = dict(
    _FUSED_PAYLOAD,
    faithfulness_verdicts=[{"statement": "a1", "verdict": 1},
                           {"statement": "a2", "verdict": 0}],
    correctness={"TP": [0, 1], "FP": [], "FN": [1]},
)


class CompactVariantTest(_FusedTestBase):
    """anthropic 전용 압축 변형(A-2 correctness 인덱스화 + A-1b faithfulness reason 제거
    + D-2 프롬프트 내 스키마 문자열 생략)의 계약 고정.

    실측 근거(2026-08-11, 20 probe A/B ×2): reason 은 faithfulness·correctness 판정을
    바꾸지 않고, context 지표의 reason 제거는 +0.05 관대화를 만든다 — 그래서 두 블록만
    바꾸고 context_verdicts·recall_classifications 는 그대로 둔다."""

    ENV = {"EVAL_RAGAS_FUSED": "1", "EVAL_RAGAS_FUSED_REPAIR": "1",
           "EVAL_LLM_PROVIDER": "anthropic", "EVAL_RAGAS_COMPACT": "1"}

    def setUp(self):
        super().setUp()
        self.stub.fused_payload = dict(_COMPACT_PAYLOAD)

    def test_compact_gate_is_provider_scoped(self):
        self.assertTrue(metrics_ragas._compact_enabled())
        with patch.dict(os.environ, {"EVAL_LLM_PROVIDER": "openrouter"}):
            self.assertFalse(metrics_ragas._compact_enabled())     # 다른 provider 는 불변
        with patch.dict(os.environ, {"EVAL_RAGAS_COMPACT": "0"}):
            self.assertFalse(metrics_ragas._compact_enabled())     # kill switch

    def test_compact_schema_drops_reason_and_uses_index_arrays(self):
        schema = metrics_ragas._fused_json_schema(_REAL_BLOCKS, compact=True)
        faith_props = schema["properties"]["faithfulness_verdicts"]["items"]["properties"]
        self.assertNotIn("reason", faith_props)                     # A-1(b)
        corr = schema["properties"]["correctness"]["properties"]
        for bucket in ("TP", "FP", "FN"):
            self.assertEqual(corr[bucket], {"items": {"type": "integer"}, "type": "array"})  # A-2
        # 관대화가 실측된 두 블록의 reason 은 남아야 한다.
        ctx = schema["properties"]["context_verdicts"]["items"]["properties"]
        recall = schema["properties"]["recall_classifications"]["items"]["properties"]
        self.assertIn("reason", ctx)
        self.assertIn("reason", recall)
        # required 규칙·구조 유효성은 레거시와 같은 규약을 지켜야 한다(Anthropic 400 방지).
        self.assertEqual(schema["required"],
                         metrics_ragas._fused_required_keys(_REAL_BLOCKS))

    def test_compact_prefix_omits_schema_paragraph(self):
        # D-2: output_config 가 스키마를 강제하므로 프롬프트 속 스키마 문단은 이중 전송이다.
        legacy, _ = metrics_ragas._fused_prompt_parts(_REAL_BLOCKS, "q", "a", ["c"], "r")
        compact, _ = metrics_ragas._fused_prompt_parts(
            _REAL_BLOCKS, "q", "a", ["c"], "r", compact=True)
        self.assertIn("complies with the following schema", legacy)
        self.assertNotIn("complies with the following schema", compact)
        self.assertIn("--------EXAMPLE-----------", compact)        # few-shot 은 유지
        self.assertIn('"TP": [', compact)                           # 인덱스 배열 예시 노출
        self.assertLess(len(compact), len(legacy))

    def test_compact_scores_match_legacy_formulas(self):
        # 형식이 바뀌어도 점수 계산은 같은 함수·같은 식이다(len/verdict 만 소비).
        out = metrics_ragas.evaluate_real_track(_record(), True)
        self.assertAlmostEqual(out["faithfulness"], 0.5)
        self.assertEqual((out["answer_correctness_tp"], out["answer_correctness_fp"],
                          out["answer_correctness_fn"]), (2, 0, 1))
        self.assertAlmostEqual(out["context_recall"], 0.5)          # 비압축 블록은 그대로
        # 호출에 실린 스키마·프리픽스도 compact 여야 한다(조립만 되고 안 실리면 무의미).
        self.assertEqual(len(self.stub.chat_calls), 1)
        sent = self.stub.schemas[0]
        self.assertEqual(sent["properties"]["correctness"]["properties"]["TP"],
                         {"items": {"type": "integer"}, "type": "array"})
        self.assertNotIn("complies with the following schema", self.stub.systems[0])

    def test_compact_counts_survive_index_arrays(self):
        self.assertEqual(
            metrics_ragas._fused_correctness_counts(
                {"answer_statements": ["a0", "a1", "a2"], "reference_statements": ["r0"],
                 "correctness": {"TP": [0, 2], "FP": [1], "FN": []}}),
            (2, 1, 0))

    def test_index_drop_notice_prints_once_with_raw_values(self):
        """중복·범위 밖 제외는 조용하면 안 된다(리뷰 지적) — 모델이 1-based 로 답하는
        상태가 신호 없이 점수만 낮춘다. 원시 배열과 함께 실행당 1회, 2회차부터 침묵."""
        import io as _io
        from contextlib import redirect_stdout
        metrics_ragas._index_drop_notified = False      # 새 전역 — 테스트가 반드시 리셋
        self.addCleanup(setattr, metrics_ragas, "_index_drop_notified", False)
        d = {"answer_statements": ["a0", "a1", "a2"], "reference_statements": [],
             "correctness": {"TP": [1, 2, 3], "FP": [], "FN": []}}
        buf = _io.StringIO()
        with redirect_stdout(buf):
            counts = metrics_ragas._fused_correctness_counts(d)
        self.assertEqual(counts, (2, 0, 0))             # 1-based 오답: 3 이 범위 밖
        self.assertIn("[1, 2, 3]", buf.getvalue())      # 원시 값이 보여야 진단 가능
        self.assertIn("1-based", buf.getvalue())
        buf2 = _io.StringIO()
        with redirect_stdout(buf2):
            metrics_ragas._fused_correctness_counts(d)
        self.assertEqual(buf2.getvalue(), "")           # 실행당 1회

    def test_compact_counts_ignore_duplicate_and_out_of_range_indexes(self):
        """"TP": [0, 0, 9] 같은 값이 문장 전문 중복보다 훨씬 싼 실수라 가드가 필요하다
        (리뷰 지적) — 중복은 1개로, 범위 밖은 0개로 센다. legacy dict 목록은 불변."""
        d = {"answer_statements": ["a0", "a1"], "reference_statements": ["r0"],
             "correctness": {"TP": [0, 0, 9], "FP": [-1], "FN": [0, 0]}}
        self.assertEqual(metrics_ragas._fused_correctness_counts(d), (1, 0, 1))
        legacy = {"correctness": {"TP": [{"statement": "s"}, {"statement": "s"}],
                                  "FP": [], "FN": [{"statement": "n"}]}}
        self.assertEqual(metrics_ragas._fused_correctness_counts(legacy), (2, 0, 1))

    def test_block_and_schema_override_tables_are_in_sync(self):
        self.assertEqual(set(metrics_ragas._COMPACT_BLOCK_OVERRIDES),
                         set(metrics_ragas._COMPACT_SCHEMA_OVERRIDES))


if __name__ == "__main__":
    unittest.main()
