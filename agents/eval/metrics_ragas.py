"""
agents/eval/metrics_ragas.py
STEP3-2: LLM 진단 (RAGAS 지표 측정)

RAGAS 4개 지표 + AspectCritic 을 **LLM-as-Judge** 로 측정한다.
    - 실제 트랙  : Faithfulness, Context Precision/Recall, Response Relevancy
    - 오라클 트랙 : Faithfulness, Response Relevancy (gold context 투입 결과)
    - AspectCritic: contradiction 이진 판정

프롬프트 출처:
    RAGAS 라이브러리는 이 환경(langchain 1.x + langgraph)과 의존성 충돌로 import가 불가하다.
    그래서 라이브러리는 쓰지 않되, **프롬프트·알고리즘은 설치된 ragas 0.4.3 소스와 일치**시킨다.
    (지시문/few-shot 예시/조립 형식/스코어 계산식 모두 아래 소스에서 그대로 옮김)
      - ragas/metrics/collections/faithfulness/util.py   (StatementGenerator + NLI, 2단계)
      - ragas/metrics/collections/answer_relevancy/{util,metric}.py  (strictness=3, noncommittal)
      - ragas/metrics/collections/context_precision/{util,metric}.py (청크별 verdict, avg-precision)
      - ragas/metrics/collections/context_recall/util.py (문장별 attributed)
      - ragas/metrics/_aspect_critic.py                  (Evaluate the Input ... criterial)
      - ragas/prompt/metrics/base_prompt.py (BasePrompt.to_string 조립 형식)
    환경이 ragas 를 지원하면 라이브러리 호출로 교체해도 결과가 동일하다.

비용·재현성:
    - 실행 게이트는 호출부(agent._ragas_track + signals RAGAS 신호)가 담당: `EVAL_ENABLE_LLM=1`
      + `EVAL_MODE≥deep` 일 때만 evaluate_real_track/oracle_track 을 호출한다(기본 비활성).
    - 응답 모델 ≠ 평가 모델(EVAL_JUDGE_MODEL, 기본 gpt-4o), temperature=0.
    - 키 없음·호출/파싱 실패 → 조용히 건너뛰고(폴백) 규칙 지표(STEP3-1)로 진행.
    - 실제 LLM 호출은 agents/eval/llm_provider.py 가 담당 (OpenAI 기본, EVAL_LLM_PROVIDER=gemini로
      Google AI Studio 무료 API 임시 대체 가능 — OpenAI 토큰 승인 전 브릿지).
"""
from __future__ import annotations

import json
import math
import os

from agents.eval import llm_provider
from agents.eval.types import EvalRecord


def _env_int(name: str, default: int) -> int:
    """환경변수 정수 파싱 — 비정수/음수면 기본값(≥1)으로 폴백. import 시점 크래시 방지."""
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


# RAGAS AnswerRelevancy 기본 strictness (생성 질문 개수)
RELEVANCY_STRICTNESS = _env_int("EVAL_RELEVANCY_STRICTNESS", 3)


# ══════════════════════════════════════════════════════════════════
#  RAGAS 프롬프트 (ragas 0.4.3 소스 verbatim)
# ══════════════════════════════════════════════════════════════════

# ── Faithfulness: ① 문장 분해 (StatementGeneratorPrompt) ─────────
_FAITH_STMT_INSTRUCTION = (
    "Given a question and an answer, analyze the complexity of each sentence "
    "in the answer. Break down each sentence into one or more fully understandable "
    "statements. Ensure that no pronouns are used in any statement."
)
_FAITH_STMT_EXAMPLES = [
    (
        {"question": "Who was Albert Einstein and what is he best known for?",
         "answer": "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics."},
        {"statements": [
            "Albert Einstein was a German-born theoretical physicist.",
            "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.",
            "Albert Einstein was best known for developing the theory of relativity.",
            "Albert Einstein made important contributions to the development of the theory of quantum mechanics.",
        ]},
    ),
]

# ── Faithfulness: ② NLI 판정 (NLIStatementPrompt) ────────────────
_FAITH_NLI_INSTRUCTION = (
    "Your task is to judge the faithfulness of a series of statements based on a "
    "given context. For each statement you must return verdict as 1 if the statement "
    "can be directly inferred based on the context or 0 if the statement can not be "
    "directly inferred based on the context."
)
_FAITH_NLI_EXAMPLES = [
    (
        {"context": "John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.",
         "statements": [
             "John is majoring in Biology.",
             "John is taking a course on Artificial Intelligence.",
             "John is a dedicated student.",
             "John has a part-time job.",
         ]},
        {"statements": [
            {"statement": "John is majoring in Biology.",
             "reason": "John's major is explicitly stated as Computer Science, not Biology.", "verdict": 0},
            {"statement": "John is taking a course on Artificial Intelligence.",
             "reason": "The context mentions courses in Data Structures, Algorithms, and Database Management, but does not mention Artificial Intelligence.", "verdict": 0},
            {"statement": "John is a dedicated student.",
             "reason": "The context states that John is a diligent student who spends a significant amount of time studying and completing assignments.", "verdict": 1},
            {"statement": "John has a part-time job.",
             "reason": "There is no information in the context about John having a part-time job.", "verdict": 0},
        ]},
    ),
]

# ── Answer Relevancy (AnswerRelevancePrompt) ─────────────────────
_RELEVANCY_INSTRUCTION = (
    "Generate a question for the given answer and identify if the answer is noncommittal.\n"
    "Give noncommittal as 1 if the answer is noncommittal (evasive, vague, or ambiguous) "
    "and 0 if the answer is substantive.\n"
    'Examples of noncommittal answers: "I don\'t know", "I\'m not sure", "It depends".'
)
_RELEVANCY_EXAMPLES = [
    ({"response": "Albert Einstein was born in Germany."},
     {"question": "Where was Albert Einstein born?", "noncommittal": 0}),
    ({"response": "The capital of France is Paris, a city known for its architecture and culture."},
     {"question": "What is the capital of France?", "noncommittal": 0}),
    ({"response": "I don't know about the groundbreaking feature of the smartphone invented in 2023 as I am unaware of information beyond 2022."},
     {"question": "What was the groundbreaking feature of the smartphone invented in 2023?", "noncommittal": 1}),
]

# ── Context Precision (ContextPrecisionPrompt) ───────────────────
_CTX_PREC_INSTRUCTION = (
    'Given question, answer and context verify if the context was useful in arriving '
    'at the given answer. Give verdict as "1" if useful and "0" if not with json output.'
)
_CTX_PREC_EXAMPLES = [
    ({"question": "What can you tell me about Albert Einstein?",
      "context": "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect', a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.",
      "answer": "Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics."},
     {"reason": "The provided context was indeed useful in arriving at the given answer. The context includes key information about Albert Einstein's life and contributions, which are reflected in the answer.", "verdict": 1}),
    ({"question": "who won 2020 icc world cup?",
      "context": "The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.",
      "answer": "England"},
     {"reason": "the context was useful in clarifying the situation regarding the 2020 ICC World Cup and indicating that England was the winner of the tournament that was intended to be held in 2020 but actually took place in 2022.", "verdict": 1}),
    ({"question": "What is the tallest mountain in the world?",
      "context": "The Andes is the longest continental mountain range in the world, located in South America. It stretches across seven countries and features many of the highest peaks in the Western Hemisphere. The range is known for its diverse ecosystems, including the high-altitude Andean Plateau and the Amazon rainforest.",
      "answer": "Mount Everest."},
     {"reason": "the provided context discusses the Andes mountain range, which, while impressive, does not include Mount Everest or directly relate to the question about the world's tallest mountain.", "verdict": 0}),
]

# ── Context Recall (ContextRecallPrompt) ─────────────────────────
_CTX_RECALL_INSTRUCTION = (
    "Given a context and an answer, analyze each statement in the answer and classify "
    "if the statement can be attributed to the given context or not.\n"
    "Use only binary classification: 1 if the statement can be attributed to the context, "
    "0 if it cannot.\nProvide detailed reasoning for each classification."
)
_CTX_RECALL_EXAMPLES = [
    ({"question": "What can you tell me about Albert Einstein?",
      "context": "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass-energy equivalence formula E = mc2, which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect', a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.",
      "answer": "Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in 1905. Einstein moved to Switzerland in 1895."},
     {"classifications": [
         {"statement": "Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time.",
          "reason": "The date of birth of Einstein is mentioned clearly in the context.", "attributed": 1},
         {"statement": "He received the 1921 Nobel Prize in Physics for his services to theoretical physics.",
          "reason": "The exact sentence is present in the given context.", "attributed": 1},
         {"statement": "He published 4 papers in 1905.",
          "reason": "There is no mention about papers he wrote in the given context.", "attributed": 0},
         {"statement": "Einstein moved to Switzerland in 1895.",
          "reason": "There is no supporting evidence for this in the given context.", "attributed": 0},
     ]}),
    ({"question": "who won 2020 icc world cup?",
      "context": "The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.",
      "answer": "England"},
     {"classifications": [
         {"statement": "England", "reason": "The context clarifies that England won the 2022 edition (which was originally scheduled for 2020).", "attributed": 1},
     ]}),
    ({"question": "What is the tallest mountain in the world?",
      "context": "The Andes is the longest continental mountain range in the world, located in South America. It stretches across seven countries and features many of the highest peaks in the Western Hemisphere. The range is known for its diverse ecosystems, including the high-altitude Andean Plateau and the Amazon rainforest.",
      "answer": "Mount Everest."},
     {"classifications": [
         {"statement": "Mount Everest.", "reason": "The provided context discusses the Andes mountain range, which does not include Mount Everest or directly relate to the world's tallest mountain.", "attributed": 0},
     ]}),
]

# ── AspectCritic instruction 템플릿 (definition 삽입; RAGAS 원문의 'criterial' 오타 그대로) ──
_ASPECT_INSTRUCTION_TMPL = (
    "Evaluate the Input based on the criterial defined. Use only 'Yes' (1) and 'No' (0) "
    "as verdict.\nCriteria Definition: {definition}"
)
# 커스텀 criteria (RAGAS AspectCritic definition 슬롯에 주입)
_ASPECT_CONTRADICTION = ("Does the response contain information that contradicts the "
                         "retrieved context?")

# 출력 JSON 스키마 힌트 (BasePrompt.to_string 의 output_schema 자리)
_SCHEMA_STATEMENTS = '{"properties": {"statements": {"items": {"type": "string"}, "type": "array"}}, "required": ["statements"]}'
_SCHEMA_NLI = '{"properties": {"statements": {"items": {"properties": {"statement": {"type": "string"}, "reason": {"type": "string"}, "verdict": {"type": "integer"}}, "required": ["statement", "reason", "verdict"], "type": "object"}, "type": "array"}}, "required": ["statements"]}'
_SCHEMA_RELEVANCY = '{"properties": {"question": {"type": "string"}, "noncommittal": {"type": "integer"}}, "required": ["question", "noncommittal"]}'
_SCHEMA_VERDICT = '{"properties": {"reason": {"type": "string"}, "verdict": {"type": "integer"}}, "required": ["reason", "verdict"]}'
_SCHEMA_RECALL = '{"properties": {"classifications": {"items": {"properties": {"statement": {"type": "string"}, "reason": {"type": "string"}, "attributed": {"type": "integer"}}, "required": ["statement", "reason", "attributed"], "type": "object"}, "type": "array"}}, "required": ["classifications"]}'


# ══════════════════════════════════════════════════════════════════
#  심판 LLM
# ══════════════════════════════════════════════════════════════════

def _judge():
    """평가(심판) LLM 사용 가능 여부(OpenAI/Gemini/GitHub Models, EVAL_LLM_PROVIDER로 선택). 키 없으면 None."""
    if not llm_provider.has_key():
        return None
    # 설계 원칙: 응답 모델과 다른 모델로 채점 (모델 선택은 llm_provider 내부에서 처리)
    return True


# ══════════════════════════════════════════════════════════════════
#  트랙별 측정
#    diagnose(signals)가 트랙별로 필요한 것만 lazy 호출한다(agent._ragas_track 경유).
#    실제 트랙 = 검색결과 컨텍스트, 오라클 트랙 = gold 컨텍스트.
# ══════════════════════════════════════════════════════════════════

def evaluate_real_track(record: EvalRecord, judge) -> dict:
    """실제 결과 지표. faithfulness, response_relevancy, (+정답 있으면) context_precision, context_recall.

    [TODO 비용] DEEP 트랙 1개당 LLM 호출 ~11회(faithfulness 2 + precision top_k개 + recall 1 +
    relevancy strictness). precision 청크별 호출을 배치/병렬화하거나 strictness↓로 절감 여지."""
    q = record.probe.question
    ans = record.generated_answer
    ctx = record.retrieved_context
    ref = record.probe.ground_truth

    out: dict = {
        "faithfulness": _faithfulness(judge, q, ans, ctx),
        "response_relevancy": _response_relevancy(judge, q, ans),
    }
    if ref:  # reference 있어야 Context Precision/Recall(WithReference) 계산 가능
        out["context_precision"] = _context_precision(judge, q, ref, ctx)
        out["context_recall"] = _context_recall(judge, q, ref, ctx)
    return _drop_none(out)


def evaluate_oracle_track(record: EvalRecord, judge) -> dict:
    """gold context 로 생성한 답에 대한 지표. faithfulness, response_relevancy."""
    q = record.probe.question
    ans = record.oracle_answer or ""
    ctx = record.oracle_context or record.retrieved_context
    return _drop_none({
        "faithfulness": _faithfulness(judge, q, ans, ctx),
        "response_relevancy": _response_relevancy(judge, q, ans),
    })


def answer_similarity(record: EvalRecord, track: str):
    """생성 답변↔gold 정답의 임베딩 코사인 유사도(tier3 의미 게이트용).
    lexical(정규화 F1/recall)이 임계 미달일 때 '표면형은 달라도 의미는 정답'을 구제하는 승급 신호.
        track: 'real'(generated_answer) | 'oracle'(oracle_answer)
    키 없음·재료(정답/답변) 없음·임베딩 실패 → None(미측정)."""
    ref = record.probe.ground_truth
    ans = record.oracle_answer if track == "oracle" else record.generated_answer
    if not (ref or "").strip() or not (ans or "").strip():
        return None
    if _judge() is None:
        return None
    try:
        vecs = _embed(None, [ref, ans])
    except Exception:
        return None
    if not vecs or len(vecs) < 2:
        return None
    return _cosine(vecs[0], vecs[1])


def evaluate_aspect_critics(record: EvalRecord, judge) -> dict:
    """커스텀 AspectCritic(이진): contradiction.
    [예약] 현재 라이브 진단 경로는 record.aspect 를 소비하지 않는다 — diagnose.generation_contradiction
    라벨(주석처리, '나중에 개발')이 이 값을 쓸 예정. 그 라벨을 켤 때 함께 배선한다."""
    q = record.probe.question
    ans = record.generated_answer
    ctx = record.retrieved_context
    return {
        "contradiction": _aspect_critic(judge, _ASPECT_CONTRADICTION, q, ans, ctx),
    }


# ══════════════════════════════════════════════════════════════════
#  RAGAS 지표 알고리즘 (소스와 동일)
# ══════════════════════════════════════════════════════════════════

def _faithfulness(judge, question: str, answer: str, contexts: list[str]):
    """RAGAS Faithfulness (2단계): 답변→문장 분해 → 각 문장 NLI 판정 → 지지 비율."""
    if not (answer or "").strip() or not contexts:
        return None
    # 1. 문장 분해: 검증가능한 주장들로 분해
    d1 = _chat(judge, _ragas_prompt(_FAITH_STMT_INSTRUCTION, _SCHEMA_STATEMENTS,
                                    _FAITH_STMT_EXAMPLES, {"question": question, "answer": answer}))
    statements = [s for s in _as_list(d1, "statements") if isinstance(s, str) and s.strip()]
    if not statements:
        return None
    # 2. NLI 판정: 각 주장이 컨텍스트만으로 추론 가능한지 판단
    context_str = "\n".join(contexts)
    d2 = _chat(judge, _ragas_prompt(_FAITH_NLI_INSTRUCTION, _SCHEMA_NLI,
                                    _FAITH_NLI_EXAMPLES, {"context": context_str, "statements": statements}))
    verdicts = [v for v in _as_list(d2, "statements") if isinstance(v, dict)]
    if not verdicts:
        return None
    supported = sum(1 for v in verdicts if _truthy(v.get("verdict")))
    return supported / len(verdicts)


def _response_relevancy(judge, question: str, answer: str):
    """RAGAS AnswerRelevancy: 답변→질문 strictness(3)회 생성 → 원 질문과 코사인 평균. 모두 회피성이면 0."""
    if not (answer or "").strip():
        return 0.0
    gen_qs, noncommittal = [], []
    # 답변으로부터 질문 n회 생성
    for _ in range(RELEVANCY_STRICTNESS):
        d = _chat(judge, _ragas_prompt(_RELEVANCY_INSTRUCTION, _SCHEMA_RELEVANCY,
                                       _RELEVANCY_EXAMPLES, {"response": answer}))
        q = d.get("question")
        # LLM으로부터 생성된 str이 잘 존재하는지 판별 후 -> 판별에 필요한것 저장
        if isinstance(q, str) and q.strip():
            gen_qs.append(q)
            noncommittal.append(1 if _truthy(d.get("noncommittal")) else 0)
    if not gen_qs:
        return 0.0
    all_noncommittal = all(n == 1 for n in noncommittal) # noncommittal: 답변이 회피형(잘모르겠다.)인지 판별
    vecs = _embed(judge, [question] + gen_qs)  # Embedding
    if not vecs or len(vecs) < 2:
        return None
    sims = [_cosine(vecs[0], v) for v in vecs[1:]] # Cosine Similarity
    return (sum(sims) / len(sims)) * (0 if all_noncommittal else 1) # 모든 답변이 회피형이면 0 출력


def _context_precision(judge, question: str, reference: str, contexts: list[str]):
    """RAGAS ContextPrecision: 청크마다 유용성 판정 → 순위 가중 average precision."""
    if not contexts or not (reference or "").strip():
        return None
    verdicts = []
    for c in contexts:  # RAGAS: 청크 하나씩 판정
        d = _chat(judge, _ragas_prompt(_CTX_PREC_INSTRUCTION, _SCHEMA_VERDICT,
                                       _CTX_PREC_EXAMPLES, {"question": question, "context": c, "answer": reference}))
        verdicts.append(1 if _truthy(d.get("verdict")) else 0)
    return _average_precision(verdicts)

def _context_recall(judge, question: str, reference: str, contexts: list[str]):
    """RAGAS ContextRecall: 정답(reference)을 문장별로 나눠 context 귀속 여부 → 귀속 비율."""
    if not contexts or not (reference or "").strip():
        return None
    context_str = "\n".join(contexts)
    d = _chat(judge, _ragas_prompt(_CTX_RECALL_INSTRUCTION, _SCHEMA_RECALL, _CTX_RECALL_EXAMPLES,
                                   {"question": question, "context": context_str, "answer": reference}))
    cls = _as_list(d, "classifications")
    if not cls:
        return None
    return sum(1 for c in cls if _truthy(c.get("attributed"))) / len(cls)


def _aspect_critic(judge, definition: str, user_input: str, response: str, contexts: list[str]) -> int:
    """RAGAS AspectCritic: definition 기준 이진 판정(strictness=1 → 단일 호출)."""
    instruction = _ASPECT_INSTRUCTION_TMPL.format(definition=definition)
    inp = {"user_input": user_input, "response": response, "retrieved_contexts": contexts}
    d = _chat(judge, _ragas_prompt(instruction, _SCHEMA_VERDICT, [], inp))
    return 1 if _truthy(d.get("verdict")) else 0


# ══════════════════════════════════════════════════════════════════
#  RAGAS 프롬프트 조립 (BasePrompt.to_string 형식과 동일)
# ══════════════════════════════════════════════════════════════════

def _ragas_prompt(instruction: str, output_schema: str, examples: list, input_obj: dict) -> str:
    """
    RAGAS BasePrompt.to_string 과 동일한 형식으로 완성 프롬프트를 만든다.
    (instruction → 출력 스키마 → EXAMPLES → 'Now perform the same...' → input → 'Output: ')
    """
    examples_str = ""
    if examples:
        parts = []
        for i, (inp, out) in enumerate(examples):
            parts.append(
                f"Example {i + 1}\n"
                f"Input: {json.dumps(inp, indent=4, ensure_ascii=False)}\n"
                f"Output: {json.dumps(out, indent=4, ensure_ascii=False)}"
            )
        examples_str = "--------EXAMPLES-----------\n" + "\n\n".join(parts)
    input_json = json.dumps(input_obj, indent=4, ensure_ascii=False)
    return (
        f"{instruction}\n"
        f"Please return the output in a JSON format that complies with the following "
        f"schema as specified in JSON Schema:\n"
        f"{output_schema}Do not use single quotes in your response but double quotes,"
        f"properly escaped with a backslash.\n\n"
        f"{examples_str}\n"
        f"-----------------------------\n\n"
        f"Now perform the same with the following input\n"
        f"input: {input_json}\n"
        f"Output: "
    )


# ══════════════════════════════════════════════════════════════════
#  OpenAI 호출 / 유틸
# ══════════════════════════════════════════════════════════════════


def _average_precision(verdicts: list[int]) -> float:
    """순위 가중 평균. [TODO] 부분합을 매 스텝 재계산해 O(n^2) — top_k(≈5) 작아 무시 가능."""
    denominator = sum(verdicts) + 1e-10
    numerator = sum(
        (sum(verdicts[: i + 1]) / (i + 1)) * verdicts[i]
        for i in range(len(verdicts))
    )
    return numerator / denominator

def _chat(judge, prompt: str) -> dict:
    """RAGAS 형식 단일 프롬프트를 JSON 강제로 호출 → dict. 실패 시 {}."""
    return llm_provider.chat_json("", prompt)


def _embed(judge, texts: list[str]) -> list[list[float]]:
    """텍스트 리스트 → 임베딩 벡터 리스트."""
    return llm_provider.embed_texts(texts)


def _cosine(a: list[float], b: list[float]) -> float:
    """cosine 유사도"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _as_list(data, key: str) -> list:
    """list로 변환"""
    if isinstance(data, dict) and isinstance(data.get(key), list):
        return data[key]
    return []


def _truthy(v) -> bool:
    """LLM의 true 출력 변환"""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v == 1
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "t")
    return False


def _drop_none(d: dict) -> dict:
    """value가 None이면 버리기"""
    return {k: v for k, v in d.items() if v is not None}
