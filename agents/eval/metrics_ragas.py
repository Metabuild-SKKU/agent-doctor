"""
agents/eval/metrics_ragas.py
[tier3] LLM(RAGAS) 호출이 필요한 측정을 모은 파일. (STEP3-2: LLM 진단)

`active_mode() < DEEP` 이면 한 번도 LLM 을 부르지 않고 None 을 돌려준다(비용 게이트).
트랙별 결과는 record.ragas / record.oracle_ragas 에 1회만 채운다(*_done 플래그).
여기는 '측정'만 한다 — 임계값 판정과 라벨 부여는 diagnose 소관이다.

RAGAS 4개 지표 + AspectCritic 을 **LLM-as-Judge** 로 측정한다.
    - 실제 트랙  : Faithfulness, Context Precision/Recall, Response Relevancy, Answer Correctness
    - 오라클 트랙 : Faithfulness, Response Relevancy, Answer Correctness (gold context 투입 결과)
    - AspectCritic: 기권 여부 이진 판정

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

호출 통합(fused, 기본 ON — EVAL_RAGAS_FUSED=0 이면 아래 legacy 경로):
    5개 지표의 입력이 (question, answer, contexts, reference)로 **전부 같다**. RAGAS 가
    호출을 쪼갠 건 지표별 독립 클래스라는 라이브러리 구조 때문이지 정보가 부족해서가 아니다.
    그래서 판정 요청만 한 프롬프트로 묶어 chat 1회 + embed 1회로 받고(_fused_track),
    점수 계산식은 legacy 와 **같은 함수**를 그대로 쓴다(_average_precision, _correctness_score …).
      실제 트랙  : chat 14 + embed 2 → chat 1 + embed 1
      오라클 트랙 : chat  8 + embed 2 → chat 1 + embed 1
    응답 1건이 지표 5개를 지고 있으므로 파싱 사고의 폭발 반경이 크다 → 블록별로 독립 파싱하고,
    그래도 빠진 지표는 _fused_repair 가 그 지표만 legacy 개별 호출로 메운다(보통 0건).

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
from typing import Any, Callable

from agents.eval import llm_provider
from agents.eval.types import Mode, EvalRecord
from agents.eval.metrics_common import _ctx, active_mode
from core.llm_clients import SCHEMA_INT01, SCHEMA_STR, array_of, strict_object
from core.parallel import parallel_map


def _env_int(name: str, default: int) -> int:
    """환경변수 정수 파싱 — 비정수/음수면 기본값(≥1)으로 폴백. import 시점 크래시 방지."""
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


# RAGAS AnswerRelevancy 기본 strictness (생성 질문 개수)
RELEVANCY_STRICTNESS = _env_int("EVAL_RELEVANCY_STRICTNESS", 3)


def _inner_concurrency() -> int:
    """트랙 1개 내부(지표/청크/strictness) 동시성. 기본 1(=off) — 바깥 probe×track
    병렬(EVAL_LLM_CONCURRENCY)과 곱해지면 429 폭풍이 나므로, probe 가 적어 바깥
    병렬이 놀 때만 명시적으로 켠다. 1이면 parallel_map 이 기존 순차와 동일.
    (fused 경로에선 트랙당 호출이 1건이라 이 값이 아무 일도 하지 않는다.)

    이 모듈의 parallel_map 호출에는 진행률 label 을 일부러 안 붙인다 — 여기는 바깥
    트랙 병렬(agents/eval/agent.py 의 STEP3)의 **워커 스레드 안**이라, 라벨을 붙이면
    probe 수만큼의 리포터가 동시에 제 진행률을 찍어 줄이 뒤섞인다. 진행률은 바깥
    한 곳(트랙 단위)에서만 세는 것이 맞다."""
    return _env_int("EVAL_RAGAS_INNER_CONCURRENCY", 1)


def _env_flag(name: str, default: bool) -> bool:
    """불리언 환경변수 — 미설정이면 default. (1/true/yes/on 만 참)"""
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _fused_enabled() -> bool:
    """fused 경로(트랙당 chat 1회) 사용 여부. 기본 켬 — EVAL_RAGAS_FUSED=0 이면 지표별
    개별 호출(legacy, ragas 원본과 1:1)로 되돌아간다. 두 경로의 점수 비교용 스위치."""
    return _env_flag("EVAL_RAGAS_FUSED", True)


def _fused_repair_enabled() -> bool:
    """fused 응답에서 특정 지표가 빠졌을 때 그 지표만 legacy 개별 호출로 보수할지.
    기본 켬 — 파싱/절단 사고 1건이 트랙 지표 전부를 '미측정'으로 날리는 걸 막는다."""
    return _env_flag("EVAL_RAGAS_FUSED_REPAIR", True)


def _fused_max_tokens() -> int:
    """fused 응답 출력 상한. 7개 블록이 한 JSON 에 들어가므로 기본 2048 로는 잘린다
    (잘리면 파싱 실패 → {} → 보수 경로). top_k·답변이 길면 더 올릴 것."""
    return _env_int("EVAL_RAGAS_FUSED_MAX_TOKENS", 4096)


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
_ASPECT_ABSTENTION = ("Does the response decline to answer — stating that it does not know, "
                      "that the information is unavailable, or that the question cannot be "
                      "answered from the given context — instead of asserting a substantive answer?")

# ── 추론 실패 모드 다중분류 (모순/수치/해석/결합/시간축) ─────────────
#   셋 이상이 같은 실패를 두고 경쟁하는 설명이라, 이진 판정 여러 번 대신 단일 분류로 배타성을
#   측정 자체에 넣는다(이진 판정을 나눠 물으면 여러 개가 동시에 참이 되어 순서가 원인을 정한다).
_REASONING_MODE_INSTRUCTION = (
    "The response was written from the given context but does not match the reference answer. "
    "Identify the single most likely failure mode and return it in 'mode'. "
    "Use exactly one of these values:\n"
    "- 'contradiction': the response asserts something that conflicts with the context.\n"
    "- 'numerical_error': a number, unit, date, or calculation is itself wrong — the value, "
    "not the order of events.\n"
    "- 'misinterpretation': the context was read but its meaning, or a condition of the question, "
    "was misunderstood.\n"
    "- 'hop_binding': the individual facts are each correct but were combined or related incorrectly.\n"
    "- 'chronological': every date or event is quoted correctly, but their order, sequence, or the "
    "duration between them is stated wrongly.\n"
    "- 'other': none of the above."
)
_SCHEMA_REASONING_MODE = ('{"properties": {"reason": {"type": "string"}, "mode": {"type": "string"}}, '
                          '"required": ["reason", "mode"]}')
_REASONING_MODE_EXAMPLES = [
    ({"user_input": "How many employees does the company have?",
      "response": "The company has 250 employees.",
      "retrieved_contexts": ["The company employs 150 people across three offices."],
      "reference": "150"},
     {"reason": "The headcount is stated as 150 in the context but the response says 250.",
      "mode": "numerical_error"}),
    ({"user_input": "Who founded the lab that developed the vaccine?",
      "response": "The vaccine was developed by Dr. Kim, who founded the lab.",
      "retrieved_contexts": ["The lab was founded by Dr. Park.", "Dr. Kim led the vaccine project."],
      "reference": "Dr. Park"},
     {"reason": "Both facts are individually correct but the founder and the project lead were merged.",
      "mode": "hop_binding"}),
    ({"user_input": "Is the policy mandatory for part-time staff?",
      "response": "Yes, the policy applies to all part-time staff.",
      "retrieved_contexts": ["The policy is recommended, but not required, for part-time staff."],
      "reference": "No, it is only recommended."},
     {"reason": "The context says recommended, the response asserts it is required.",
      "mode": "contradiction"}),
    # 시간축 예시는 연도를 '맞게' 인용한 상태로 순서만 뒤집는다 — numerical_error 와 갈리는
    # 지점이 값의 정확성이라, 예시가 값까지 틀리면 두 모드가 다시 겹친다.
    ({"user_input": "Which came first, the pilot program or the policy revision?",
      "response": "The policy revision came first in 2021, and the pilot program followed in 2019.",
      "retrieved_contexts": ["The pilot program started in 2019.",
                             "The policy was revised in 2021."],
      "reference": "The pilot program came first."},
     {"reason": "Both years are quoted correctly but the order of the two events is reversed.",
      "mode": "chronological"}),
]
_REASONING_MODES = frozenset(
    {"contradiction", "numerical_error", "misinterpretation", "hop_binding",
     "chronological", "other"}
)

# ── Answer Correctness: TP/FP/FN 분류 (CorrectnessClassifierPrompt) ──
#   ragas/metrics/collections/answer_correctness (0.4.3) 소스와 일치.
#   답변·정답을 각각 문장으로 분해(위 StatementGenerator 재사용) 후, 답변 문장을 정답 기준
#   TP/FP/FN 으로 분류 → factual F1. 최종 answer_correctness = w·factual_F1 + (1-w)·의미유사도.
_CORRECTNESS_INSTRUCTION = (
    "Given a ground truth and an answer statements, analyze each statement and classify them "
    "in one of the following categories: TP (true positive): statements that are present in "
    "answer that are also directly supported by the one or more statements in ground truth, "
    "FP (false positive): statements present in the answer but not directly supported by any "
    "statement in ground truth, FN (false negative): statements found in the ground truth but "
    "not present in answer. Each statement can only belong to one of the categories. Provide a "
    "reason for each classification."
)
_CORRECTNESS_EXAMPLES = [
    (
        {"question": "What powers the sun and what is its primary function?",
         "answer": [
             "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
             "The primary function of the sun is to provide light to the solar system.",
         ],
         "ground_truth": [
             "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
             "This fusion process in the sun's core releases a tremendous amount of energy.",
             "The energy from the sun provides heat and light, which are essential for life on Earth.",
             "The sun's light plays a critical role in Earth's climate system.",
             "Sunlight helps to drive the weather and ocean currents.",
         ]},
        {"TP": [
            {"statement": "The primary function of the sun is to provide light to the solar system.",
             "reason": "This statement is somewhat supported by the ground truth mentioning the sun providing light and its roles, though it focuses more broadly on the sun's energy."},
         ],
         "FP": [
            {"statement": "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
             "reason": "This statement is incorrect and contradicts the ground truth which states that the sun is powered by nuclear fusion."},
         ],
         "FN": [
            {"statement": "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
             "reason": "This accurate statement about the sun's power source is not included in the answer."},
            {"statement": "This fusion process in the sun's core releases a tremendous amount of energy.",
             "reason": "This process and its significance are not mentioned in the answer."},
            {"statement": "The energy from the sun provides heat and light, which are essential for life on Earth.",
             "reason": "The answer only mentions light, omitting the essential aspects of heat and its necessity for life, which the ground truth covers."},
            {"statement": "The sun's light plays a critical role in Earth's climate system.",
             "reason": "This broader impact of the sun's light on Earth's climate system is not addressed in the answer."},
            {"statement": "Sunlight helps to drive the weather and ocean currents.",
             "reason": "The effect of sunlight on weather patterns and ocean currents is omitted in the answer."},
         ]},
    ),
    (
        {"question": "What is the boiling point of water?",
         "answer": ["The boiling point of water is 100 degrees Celsius at sea level"],
         "ground_truth": [
             "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
             "The boiling point of water can change with altitude.",
         ]},
        {"TP": [
            {"statement": "The boiling point of water is 100 degrees Celsius at sea level",
             "reason": "This statement is directly supported by the ground truth which specifies the boiling point of water as 100 degrees Celsius at sea level."},
         ],
         "FP": [],
         "FN": [
            {"statement": "The boiling point of water can change with altitude.",
             "reason": "This additional information about how the boiling point of water changes with altitude is not mentioned in the answer."},
         ]},
    ),
]
# answer_correctness = weights[0]·factual_F1 + weights[1]·의미유사도 (ragas 기본 [0.75, 0.25])
_ANSWER_CORRECTNESS_WEIGHTS = (0.75, 0.25)

# 출력이 입력 크기에 비례하는 심판 호출의 상한. 공용 기본값(2048)보다 넉넉히 잡는다.
#
# 여기 걸리는 호출들은 입력 문장 하나마다 statement+reason 을 되풀이해 뱉는 스키마다
# (NLI·TP/FP/FN 분류·context_recall 분류·문장 분해). 골드가 표나 장문이면 문장이 수십 개라
# 출력이 2048 을 넘고, 잘린 응답은 JSON 파싱에 실패해 chat_json 이 {} 를 돌려준다 →
# answer_correctness 의 factual 성분이 통째로 죽고(degrade) 정답 판정이 lexical 단독으로
# 떨어진다. probe 합성(_SYNTHESIS_MAX_OUTPUT_TOKENS)이 같은 이유로 이미 올려 잡고 있다.
# 출력이 입력과 무관하게 짧은 호출(relevancy·context_precision·aspect_critic·reasoning_mode)은
# 기본값을 그대로 둬 비용 방어선을 유지한다.
# 주의: Gemini 추론 모델은 내부 사고(thoughts)도 이 상한을 함께 소진한다(core/llm_clients.py).
_LARGE_JUDGE_MAX_OUTPUT_TOKENS = _env_int("EVAL_JUDGE_MAX_OUTPUT_TOKENS", 4096)


# 출력 JSON 스키마 힌트 (BasePrompt.to_string 의 output_schema 자리)
_SCHEMA_STATEMENTS = '{"properties": {"statements": {"items": {"type": "string"}, "type": "array"}}, "required": ["statements"]}'
_SCHEMA_NLI = '{"properties": {"statements": {"items": {"properties": {"statement": {"type": "string"}, "reason": {"type": "string"}, "verdict": {"type": "integer"}}, "required": ["statement", "reason", "verdict"], "type": "object"}, "type": "array"}}, "required": ["statements"]}'
_SCHEMA_RELEVANCY = '{"properties": {"question": {"type": "string"}, "noncommittal": {"type": "integer"}}, "required": ["question", "noncommittal"]}'
_SCHEMA_VERDICT = '{"properties": {"reason": {"type": "string"}, "verdict": {"type": "integer"}}, "required": ["reason", "verdict"]}'
_SCHEMA_RECALL = '{"properties": {"classifications": {"items": {"properties": {"statement": {"type": "string"}, "reason": {"type": "string"}, "attributed": {"type": "integer"}}, "required": ["statement", "reason", "attributed"], "type": "object"}, "type": "array"}}, "required": ["classifications"]}'
_TPFPFN_ITEM = '{"items": {"properties": {"statement": {"type": "string"}, "reason": {"type": "string"}}, "required": ["statement", "reason"], "type": "object"}, "type": "array"}'
_SCHEMA_CORRECTNESS = '{"properties": {"TP": ' + _TPFPFN_ITEM + ', "FP": ' + _TPFPFN_ITEM + ', "FN": ' + _TPFPFN_ITEM + '}, "required": ["TP", "FP", "FN"]}'


# ══════════════════════════════════════════════════════════════════
#  Fused 프롬프트 (지표 5개를 chat 1회로 통합)
#    RAGAS 가 호출을 쪼갠 건 지표별 독립 클래스라는 라이브러리 구조 때문이지 정보가
#    부족해서가 아니다 — 5개 지표의 입력이 전부 (question, answer, contexts, reference)
#    로 같다. 그래서 판정 요청만 한 프롬프트에 모으고, 점수 계산식(_average_precision,
#    TP/(TP+0.5(FP+FN)), 지지 비율, 코사인 평균)은 legacy 와 **같은 함수를 그대로** 쓴다.
#    지시문도 위 legacy 상수를 재사용해 문구가 갈라지지 않게 한다.
#
#    비용: 실제 트랙 14 chat + 2 embed → 1 chat + 1 embed. (오라클 8+2 → 1+1)
#    위험: 응답 1건이 지표 5개를 다 지고 있으므로 파싱 사고의 폭발 반경이 크다 →
#          블록별로 독립 파싱하고(한 블록이 깨져도 나머지는 산다), 그래도 빠진 지표는
#          _fused_repair 가 legacy 개별 호출로 메운다.
# ══════════════════════════════════════════════════════════════════

_FUSED_HEADER = (
    "You are an evaluation judge. Perform ALL of the numbered sub-tasks below on the SAME input "
    "in a single pass, and return ONE JSON object that contains every requested key.\n"
    "Judge each sub-task independently — a verdict in one sub-task must never influence another, "
    "and you must not skip a sub-task because another one seemed to answer it.\n"
    "Keep every \"reason\" to one short sentence."
)

# 블록 = (지시문, 스키마 조각, few-shot 출력 조각). 필요한 블록만 골라 조립한다.
_FUSED_BLOCKS: dict[str, tuple[str, str, Any]] = {
    "answer_statements": (
        f"{_FAITH_STMT_INSTRUCTION} Apply this to `answer` and return them in \"answer_statements\".",
        '"answer_statements": {"items": {"type": "string"}, "type": "array"}',
        ["Albert Einstein was a German-born theoretical physicist.",
         "Albert Einstein is best known for the theory of relativity."],
    ),
    "faithfulness_verdicts": (
        f"{_FAITH_NLI_INSTRUCTION} The statements are the \"answer_statements\" you produced above, "
        f"and the context is the concatenation of every entry in `contexts`. Return one entry per "
        f"statement, in the same order, in \"faithfulness_verdicts\".",
        '"faithfulness_verdicts": {"items": {"properties": {"statement": {"type": "string"}, '
        '"reason": {"type": "string"}, "verdict": {"type": "integer"}}, '
        '"required": ["statement", "reason", "verdict"], "type": "object"}, "type": "array"}',
        [{"statement": "Albert Einstein was a German-born theoretical physicist.",
          "reason": "The context states this directly.", "verdict": 1},
         {"statement": "Albert Einstein is best known for the theory of relativity.",
          "reason": "The context says he is best known for developing the theory of relativity.",
          "verdict": 1}],
    ),
    "relevancy": (
        f"{_RELEVANCY_INSTRUCTION}\nGenerate {{n}} distinct questions for `answer` (ignore `question` "
        f"and `contexts` entirely for this sub-task — judge only what `answer` itself asks for) and "
        f"return them in \"generated_questions\", plus a single \"noncommittal\" flag for `answer`.",
        '"generated_questions": {"items": {"type": "string"}, "type": "array"}, '
        '"noncommittal": {"type": "integer"}',
        None,   # 두 키를 동시에 채우므로 예시는 _FUSED_EXAMPLE_OUTPUT 에서 직접 넣는다
    ),
    "context_verdicts": (
        f"{_CTX_PREC_INSTRUCTION} Judge EACH entry of `contexts` separately against `reference` "
        f"(treat `reference` as the answer to arrive at). Return one entry per context, carrying "
        f"its `index`, in \"context_verdicts\".",
        '"context_verdicts": {"items": {"properties": {"index": {"type": "integer"}, '
        '"reason": {"type": "string"}, "verdict": {"type": "integer"}}, '
        '"required": ["index", "reason", "verdict"], "type": "object"}, "type": "array"}',
        [{"index": 0, "reason": "It contains the physicist and relativity facts of the reference.",
          "verdict": 1},
         {"index": 1, "reason": "It is about a mountain range and is unrelated.", "verdict": 0}],
    ),
    "reference_statements": (
        f"{_FAITH_STMT_INSTRUCTION} Apply this to `reference` and return them in "
        f"\"reference_statements\".",
        '"reference_statements": {"items": {"type": "string"}, "type": "array"}',
        ["Albert Einstein was a German-born theoretical physicist.",
         "Albert Einstein is best known for the theory of relativity.",
         "Albert Einstein received the 1921 Nobel Prize in Physics."],
    ),
    "correctness": (
        f"{_CORRECTNESS_INSTRUCTION} The answer statements are your \"answer_statements\" and the "
        f"ground truth statements are your \"reference_statements\". Return the classification in "
        f"\"correctness\".",
        '"correctness": {"properties": {"TP": ' + _TPFPFN_ITEM + ', "FP": ' + _TPFPFN_ITEM
        + ', "FN": ' + _TPFPFN_ITEM + '}, "required": ["TP", "FP", "FN"], "type": "object"}',
        {"TP": [{"statement": "Albert Einstein was a German-born theoretical physicist.",
                 "reason": "Directly supported by the ground truth."},
                {"statement": "Albert Einstein is best known for the theory of relativity.",
                 "reason": "Directly supported by the ground truth."}],
         "FP": [],
         "FN": [{"statement": "Albert Einstein received the 1921 Nobel Prize in Physics.",
                 "reason": "Present in the ground truth but missing from the answer."}]},
    ),
    "recall_classifications": (
        f"{_CTX_RECALL_INSTRUCTION}\nHere the analyzed answer is `reference` and the context is the "
        f"concatenation of every entry in `contexts`. Return it in \"recall_classifications\".",
        '"recall_classifications": {"items": {"properties": {"statement": {"type": "string"}, '
        '"reason": {"type": "string"}, "attributed": {"type": "integer"}}, '
        '"required": ["statement", "reason", "attributed"], "type": "object"}, "type": "array"}',
        [{"statement": "Albert Einstein was a German-born theoretical physicist.",
          "reason": "Stated in context 0.", "attributed": 1},
         {"statement": "Albert Einstein is best known for the theory of relativity.",
          "reason": "Stated in context 0.", "attributed": 1},
         {"statement": "Albert Einstein received the 1921 Nobel Prize in Physics.",
          "reason": "No context mentions the Nobel Prize.", "attributed": 0}],
    ),
}

# ── fused 응답의 JSON Schema (anthropic output_config.format 전용) ─
#
# 위 _FUSED_BLOCKS 의 스키마 조각은 **프롬프트에 글자로 박히는 문자열**이라 모델이 지키든
# 말든 강제력이 없다(그래서 _refetch_fused_if_incomplete·_fused_repair 가 필요했다).
# 여기 dict 는 Anthropic Messages API 의 output_config.format 에 실려 디코딩 단계에서
# **강제**된다 — 키 누락·타입 위반이 원천적으로 불가능해진다.
#
# 두 표는 같은 키 집합을 가져야 한다(어긋나면 스키마가 요구하지 않는 키를 프롬프트만
# 요구하거나 그 반대가 된다). tests/test_ragas_fused.py 가 대칭을 핀으로 잡는다.
#
# 스키마 제약(Anthropic structured outputs):
#   - 모든 object 에 additionalProperties: false 와 required 전량 명시가 필수다.
#   - minimum/maximum/minLength 류 수치·길이 제약은 미지원이다. 그래서 0/1 판정은
#     {"type":"integer"} 가 아니라 enum 으로 좁힌다 — 프롬프트 문자열 스키마보다 강하다.
#   - 재귀 스키마는 미지원(여기선 쓰지 않는다).
# 조립 헬퍼는 core 가 소유한다 — 제약이 Anthropic API 규칙이고 쓰는 곳이 여럿이라
# (probe_gen.py 도 같은 헬퍼를 쓴다) 규칙이 갈라지지 않게 한 곳에 둔다.
_obj = strict_object
_arr = array_of
_STR = SCHEMA_STR
_INT01 = SCHEMA_INT01
_TPFPFN_ITEM_SCHEMA = _arr(_obj({"statement": _STR, "reason": _STR}))

# 블록 이름 → 그 블록이 채우는 최상위 프로퍼티들. relevancy 만 키가 2개다
# (_fused_required_keys 의 예외 규칙과 같은 이유).
_FUSED_SCHEMA_PROPS: dict[str, dict] = {
    "answer_statements": {"answer_statements": _arr(_STR)},
    "faithfulness_verdicts": {
        "faithfulness_verdicts": _arr(_obj(
            {"statement": _STR, "reason": _STR, "verdict": _INT01}))},
    "relevancy": {
        "generated_questions": _arr(_STR),
        "noncommittal": _INT01},
    "context_verdicts": {
        "context_verdicts": _arr(_obj(
            {"index": {"type": "integer"}, "reason": _STR, "verdict": _INT01}))},
    "reference_statements": {"reference_statements": _arr(_STR)},
    "correctness": {
        "correctness": _obj({"TP": _TPFPFN_ITEM_SCHEMA,
                             "FP": _TPFPFN_ITEM_SCHEMA,
                             "FN": _TPFPFN_ITEM_SCHEMA})},
    "recall_classifications": {
        "recall_classifications": _arr(_obj(
            {"statement": _STR, "reason": _STR, "attributed": _INT01}))},
}


def _fused_json_schema(blocks: list[str]) -> dict | None:
    """선택된 블록만으로 응답 JSON Schema 를 만든다. 블록이 없으면 None.

    프로퍼티 순서 = 블록 순서 = 모델이 답을 만들어야 하는 순서(분해 → 판정)다.
    스키마 강제는 순서를 요구하지 않지만, 순서를 맞춰두면 프롬프트와 스키마가 같은
    이야기를 해서 모델이 헷갈릴 여지가 줄어든다."""
    if not blocks:
        return None
    props: dict = {}
    for name in blocks:
        props.update(_FUSED_SCHEMA_PROPS[name])
    return _obj(props)


_FUSED_EXAMPLE_INPUT = {
    "question": "What can you tell me about Albert Einstein?",
    "answer": ("Albert Einstein was a German-born theoretical physicist. "
               "He is best known for the theory of relativity."),
    "contexts": [
        {"index": 0, "text": ("Albert Einstein (1879-1955) was a German-born theoretical physicist, "
                              "best known for developing the theory of relativity.")},
        {"index": 1, "text": ("The Andes is the longest continental mountain range in the world, "
                              "located in South America.")},
    ],
    "reference": ("Albert Einstein was a German-born theoretical physicist best known for the theory "
                  "of relativity. He received the 1921 Nobel Prize in Physics."),
}
# relevancy 블록은 키가 2개라 예시를 따로 둔다(블록 조각 자리는 None).
_FUSED_RELEVANCY_EXAMPLE = {
    "generated_questions": [
        "Who was Albert Einstein and what is he best known for?",
        "What is Albert Einstein best known for?",
        "Which German-born theoretical physicist developed the theory of relativity?",
    ],
    "noncommittal": 0,
}


def _fused_prompt(blocks: list[str], question: str, answer: str,
                  contexts: list[str], reference: str) -> str:
    """선택된 블록만으로 fused 프롬프트를 조립한다(하위호환 — 두 부분을 이어붙인다)."""
    prefix, suffix = _fused_prompt_parts(blocks, question, answer, contexts, reference)
    return prefix + suffix


def _fused_prompt_parts(blocks: list[str], question: str, answer: str,
                        contexts: list[str], reference: str) -> tuple[str, str]:
    """fused 프롬프트를 (안정 프리픽스, 가변 입력) 두 부분으로 조립한다.

    나누는 이유는 **프롬프트 캐싱**이다. 프리픽스(헤더·지시문·스키마·few-shot)는 블록
    구성이 같으면 매 호출 바이트가 동일하고, 뒷부분(질문·답변·컨텍스트)만 probe 마다
    바뀐다. 앞부분을 system 으로 보내 캐시 지점을 걸면 두 번째 호출부터 그 몫이 입력가의
    0.1 배가 된다.

    실측(count_tokens, claude-sonnet-5): 캐시 프리픽스는 실제 트랙 4,325 / 오라클 트랙
    3,211 토큰이다. 주의 — 이 값은 system 블록만이 아니라 **output_config 의 스키마까지
    포함**한다(스키마도 messages 앞에 렌더돼 캐시 프리픽스에 들어간다). 글자수÷4 로
    어림하면 1.35 배 과소평가되므로(9,025자 → 어림 2,256 vs 실측 4,325) 모델별 캐시
    최소치와 비교할 때는 반드시 count_tokens 로 잴 것.

    캐싱은 **접두 일치**라 순서가 곧 설계다 — 가변 내용이 한 바이트라도 앞에 끼면 그
    뒤가 전부 무효화된다. 그래서 질문·컨텍스트는 반드시 프리픽스 뒤에 둔다.

    블록 순서 = 모델이 답을 만들어야 하는 순서(분해 → 판정)이므로 호출부 순서를 지킨다.
    (anthropic 이 아닌 provider 는 두 부분이 이어붙어 예전과 같은 한 덩어리가 된다 —
    _fused_prompt 참고. 캐싱만 못 쓸 뿐 프롬프트 내용은 동일하다.)"""
    instructions, schema_parts, example_out = [], [], {}
    for i, name in enumerate(blocks, start=1):
        instr, schema, example = _FUSED_BLOCKS[name]
        instructions.append(f"({i}) " + instr.replace("{n}", str(RELEVANCY_STRICTNESS)))
        schema_parts.append(schema)
        if name == "relevancy":
            example_out.update(_FUSED_RELEVANCY_EXAMPLE)
        else:
            example_out[name] = example

    required = [k for name in blocks
                for k in (("generated_questions", "noncommittal") if name == "relevancy" else (name,))]
    schema = ('{"properties": {' + ", ".join(schema_parts) + '}, "required": '
              + json.dumps(required) + '}')

    input_obj: dict[str, Any] = {"question": question, "answer": answer}
    if contexts:
        input_obj["contexts"] = [{"index": i, "text": c} for i, c in enumerate(contexts)]
    if reference:
        input_obj["reference"] = reference

    example_str = (
        "--------EXAMPLE-----------\n"
        f"Input: {json.dumps(_FUSED_EXAMPLE_INPUT, indent=4, ensure_ascii=False)}\n"
        f"Output: {json.dumps(example_out, indent=4, ensure_ascii=False)}"
    )
    # 캐시 지점 앞 — 블록 구성이 같으면 매 호출 바이트가 동일해야 한다.
    # 여기에 질문·컨텍스트가 섞여 들어가면 캐시가 probe 마다 무효화된다.
    prefix = (
        f"{_FUSED_HEADER}\n\n"
        + "\n\n".join(instructions)
        + "\n\nPlease return the output in a JSON format that complies with the following schema "
          "as specified in JSON Schema:\n"
        + f"{schema}Do not use single quotes in your response but double quotes, "
          "properly escaped with a backslash.\n\n"
        + f"{example_str}\n"
        + "-----------------------------\n\n"
          "Now perform the same with the following input\n"
    )
    # 캐시 지점 뒤 — probe 마다 바뀐다.
    suffix = (
        f"input: {json.dumps(input_obj, indent=4, ensure_ascii=False)}\n"
        # json_object 모드는 유효한 JSON 만 보장하고 스키마는 강제하지 않는다. 블록이 7개까지
        # 늘면 모델이 뒤쪽 키를 빠뜨린 채 JSON 을 닫아버리는 일이 실제로 있었다(실측: 마지막
        # 블록인 recall_classifications 가 30회 중 7회 누락). 스키마의 required 는 프롬프트
        # 중간에 묻히므로 끝에서 키 목록을 한 번 더 못박는다.
        #
        # anthropic(output_config.format)에서는 디코딩 단계가 키를 강제하므로 이 줄이
        # 없어도 되지만, 나머지 provider 에는 여전히 필요해서 남긴다. 위치가 프리픽스가
        # 아니라 여기인 건 "입력 바로 뒤에서 못박는다"는 원래 의도 때문이다 — required 는
        # 블록 구성에만 의존하므로 프리픽스에 둬도 캐시는 깨지지 않는다(무해한 선택).
        + "IMPORTANT: the JSON object MUST contain ALL of these top-level keys, even if an "
          f"array would be empty: {json.dumps(required)}. Do not omit any of them.\n"
          "Output: "
    )
    return prefix, suffix


# ══════════════════════════════════════════════════════════════════
#  심판 LLM
# ══════════════════════════════════════════════════════════════════

def _judge():
    """평가(심판) LLM 사용 가능 여부(OpenAI/Gemini/GitHub Models/OpenRouter, EVAL_LLM_PROVIDER로 선택). 키 없으면 None."""
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
    """실제 결과 지표. faithfulness, response_relevancy, (+정답 있으면) context_precision,
    context_recall, answer_correctness.

    [비용] fused(기본): chat 1 + embed 1. legacy(EVAL_RAGAS_FUSED=0): chat 14 + embed 2
    (faithfulness 2 + relevancy strictness 3 + precision top_k 5 + recall 1 + correctness 3)."""
    if _fused_enabled():
        return _fused_real_track(judge, record)
    return _evaluate_real_track_legacy(record, judge)


def evaluate_oracle_track(record: EvalRecord, judge) -> dict:
    """gold context 로 생성한 답에 대한 지표. faithfulness, response_relevancy, (+정답 있으면)
    answer_correctness. [비용] fused: chat 1 + embed 1 / legacy: chat 8 + embed 2."""
    if _fused_enabled():
        return _fused_oracle_track(judge, record)
    return _evaluate_oracle_track_legacy(record, judge)


def _evaluate_real_track_legacy(record: EvalRecord, judge) -> dict:
    """지표별 개별 호출(ragas 원본과 1:1). EVAL_RAGAS_FUSED=0 스위치와 fused 결손 보수용.

    EVAL_RAGAS_INNER_CONCURRENCY>1 이면 지표 5개를 동시 실행(지표 내부의 청크별/strictness
    호출도 각자 같은 동시성으로 돈다). 순차 대비 차이: 순차에선 첫 지표의 예외가 나머지
    호출을 생략시키지만, 어차피 호출부(_ragas_track)가 트랙 전체를 {} 로 폴백하므로
    최종 결과는 동일하다."""
    q = record.probe.question
    ans = record.generated_answer
    ctx = record.retrieved_context
    ref = record.probe.ground_truth

    tasks: list[tuple[str, Callable[[], Any]]] = [
        ("faithfulness", lambda: _faithfulness(judge, q, ans, ctx)),
        ("response_relevancy", lambda: _response_relevancy(judge, q, ans)),
    ]
    if ref:  # reference 있어야 Context Precision/Recall(WithReference)·AnswerCorrectness 계산 가능
        tasks.append(("context_precision", lambda: _context_precision(judge, q, ref, ctx)))
        tasks.append(("context_recall", lambda: _context_recall(judge, q, ref, ctx)))
        tasks.append(("answer_correctness", lambda: _answer_correctness(judge, q, ans, ref)))
    values = parallel_map(lambda t: t[1](), tasks, _inner_concurrency())
    return _drop_none(_merge_task_values(tasks, values))


def _evaluate_oracle_track_legacy(record: EvalRecord, judge) -> dict:
    """오라클 트랙의 지표별 개별 호출 경로(ragas 원본과 1:1)."""
    q = record.probe.question
    ans = record.oracle_answer or ""
    ctx = record.oracle_context or record.retrieved_context
    ref = record.probe.ground_truth
    out = {
        "faithfulness": _faithfulness(judge, q, ans, ctx),
        "response_relevancy": _response_relevancy(judge, q, ans),
    }
    if ref:
        out.update(_answer_correctness(judge, q, ans, ref))
    return _drop_none(out)


# ══════════════════════════════════════════════════════════════════
#  Fused 트랙 측정 (chat 1회 + embed 1회)
#    판정만 한 번에 받아오고, 점수 계산은 legacy 와 같은 식·같은 함수를 쓴다.
#    블록별 독립 파싱 → 한 블록이 깨져도 나머지 지표는 살아남고, 빠진 지표만 보수한다.
# ══════════════════════════════════════════════════════════════════

def _fused_real_track(judge, record: EvalRecord) -> dict:
    """실제 트랙 fused. 검색 컨텍스트를 쓰므로 precision/recall 까지 한 응답에 담는다."""
    return _fused_track(judge, record.probe.question, record.generated_answer,
                        record.retrieved_context, record.probe.ground_truth,
                        with_retrieval=True)


def _fused_oracle_track(judge, record: EvalRecord) -> dict:
    """오라클 트랙 fused. gold context 는 검색 품질 지표의 대상이 아니라 precision/recall 은 빼고
    faithfulness·relevancy·correctness 만 받는다(legacy 오라클 트랙과 같은 지표 구성)."""
    return _fused_track(judge, record.probe.question, record.oracle_answer or "",
                        record.oracle_context or record.retrieved_context,
                        record.probe.ground_truth, with_retrieval=False)


def _fused_required_keys(blocks: list[str]) -> list[str]:
    """blocks 가 요구하는 최상위 키 목록(_fused_prompt 의 required 와 같은 규칙)."""
    return [k for name in blocks
            for k in (("generated_questions", "noncommittal") if name == "relevancy" else (name,))]


def _refetch_fused_if_incomplete(judge, d: dict, prompt: str, blocks: list[str],
                                 cache_prefix: str = "", json_schema: dict | None = None) -> dict:
    """필수 키가 빠졌으면 같은 프롬프트로 딱 1회 다시 받아 채운다.

    누락은 모델의 간헐적 비준수다 — json_object 모드가 스키마를 강제하지 않아서, 같은
    입력으로 다시 물으면 대개 온전한 응답이 온다(실측: 30여 회 중 부분 누락 재현 불가,
    실행 로그에서는 30 probe 중 7회).

    anthropic(output_config.format)에서는 디코딩 단계가 키를 강제하므로 이 경로가 애초에
    발화하지 않는다. 그래도 지우지 않는다 — openrouter/openai/gemini 는 여전히 스키마
    강제가 없어서 이 보수가 필요하고, provider 는 실행 시점 env 로 갈린다.

    개별 지표 보수(_fused_repair)보다 먼저 두는 이유는 비용이다. 보수는 지표마다 따로
    호출하고 그중 response_relevancy 는 strictness 만큼(기본 3회) 더 부른다 —
    {faithfulness, relevancy, correctness} 3개가 빠지면 5회다. fused 재요청은 1회로
    전부 되찾을 수 있다. 그래도 남는 지표는 기존 보수 경로가 받는다.

    재시도는 1회 고정이다. 반복 비준수에 계속 돈을 태우지 않는다."""
    if not blocks or not _fused_repair_enabled():
        return d
    absent = [k for k in _fused_required_keys(blocks) if k not in d]
    if not absent:
        return d
    print(f"[Eval] RAGAS fused 응답에 키 누락 {absent} → 같은 프롬프트로 1회 재요청")
    retry = _chat(judge, prompt, max_output_tokens=_fused_max_tokens(),
                  label="fused.refetch", cache_prefix=cache_prefix, json_schema=json_schema)
    if not retry:
        return d
    # 재요청분으로 빈 자리만 메운다 — 처음 받은 값이 더 신뢰할 근거는 없지만, 바꿔 끼우면
    # 같은 트랙 안에서 두 응답이 섞여 판정 근거가 일관되지 않는다.
    merged = dict(retry)
    merged.update(d)
    return merged


def _fused_track(judge, question: str, answer: str, contexts: list[str], reference: str,
                 *, with_retrieval: bool) -> dict:
    """재료가 있는 지표의 판정을 chat 1회로 받아 점수로 환산한다.

    블록 선택 규칙은 legacy 의 재료 가드와 같다 — 답변/컨텍스트/정답이 없으면 그 지표를
    애초에 묻지 않으므로, 없는 재료로 만든 헛 판정이 섞이지 않는다."""
    answer = answer or ""
    reference = reference or ""
    contexts = list(contexts or [])
    has_answer = bool(answer.strip())
    has_ref = bool(reference.strip())

    want_faith = has_answer and bool(contexts)
    want_corr = has_answer and has_ref
    want_prec = with_retrieval and has_ref and bool(contexts)   # precision·recall 은 재료가 같다

    blocks: list[str] = []
    if want_faith or want_corr:
        blocks.append("answer_statements")
    if want_faith:
        blocks.append("faithfulness_verdicts")
    if has_answer:
        blocks.append("relevancy")
    if want_prec:
        blocks.append("context_verdicts")
    if want_corr:
        blocks += ["reference_statements", "correctness"]
    if want_prec:
        blocks.append("recall_classifications")

    # 프리픽스(지시문·스키마·few-shot)와 입력을 나눠 보낸다 — 앞부분이 anthropic 의
    # 캐시 지점이 된다. 블록 구성이 같으면 프리픽스 바이트가 동일하므로, 실제/오라클
    # 두 모양에 대해 각각 1회만 캐시를 쓰고 나머지는 전부 읽기(입력가의 0.1배)다.
    prefix, user_input = _fused_prompt_parts(blocks, question, answer, contexts, reference)
    schema = _fused_json_schema(blocks)
    d = _chat(judge, user_input, max_output_tokens=_fused_max_tokens(),
              label="fused", cache_prefix=prefix, json_schema=schema) if blocks else {}
    d = _refetch_fused_if_incomplete(judge, d, user_input, blocks,
                                     cache_prefix=prefix, json_schema=schema)

    out: dict = {}
    if not has_answer:
        out["response_relevancy"] = 0.0          # legacy _response_relevancy 와 동일한 단락
    if want_faith:
        out["faithfulness"] = _fused_faithfulness(d)
    if want_prec:
        out["context_precision"] = _fused_context_precision(d, len(contexts))
        out["context_recall"] = _fused_context_recall(d)
    out.update(_fused_embedded(judge, d, question, answer, reference,
                               want_relevancy=has_answer, want_correctness=want_corr))
    out = _drop_none(out)

    if _fused_repair_enabled():
        _fused_repair(judge, out, question, answer, contexts, reference,
                      want_faith=want_faith, want_prec=want_prec, want_corr=want_corr)
    return out


def _fused_faithfulness(d: dict):
    """faithfulness_verdicts → 지지 비율. legacy 2단계(_faithfulness)의 최종식과 동일."""
    verdicts = [v for v in _as_list(d, "faithfulness_verdicts") if isinstance(v, dict)]
    if not verdicts:
        return None
    return sum(1 for v in verdicts if _truthy(v.get("verdict"))) / len(verdicts)


def _fused_context_precision(d: dict, n: int):
    """context_verdicts → 순위 가중 average precision.

    _average_precision 은 **청크 순위**에 가중을 주므로 판정을 원래 순서로 되돌리는 게 핵심이다.
    그래서 index 를 함께 받아 그 값으로 재정렬한다(모델이 순서를 흔들어도 안전).
    index 가 깨졌으면 개수가 맞을 때만 응답 순서를 순위로 믿고, 그마저 안 맞으면 미측정."""
    items = [v for v in _as_list(d, "context_verdicts") if isinstance(v, dict)]
    if not items or n <= 0:
        return None
    by_index: dict[int, int] = {}
    for v in items:
        i = v.get("index")
        if isinstance(i, bool) or not isinstance(i, int) or not (0 <= i < n) or i in by_index:
            continue
        by_index[i] = 1 if _truthy(v.get("verdict")) else 0
    if len(by_index) == n:
        verdicts = [by_index[i] for i in range(n)]
    elif len(items) == n:
        verdicts = [1 if _truthy(v.get("verdict")) else 0 for v in items]
    else:
        return None
    return _average_precision(verdicts)


def _fused_context_recall(d: dict):
    """recall_classifications → 귀속 비율. legacy _context_recall 의 최종식과 동일."""
    cls = [c for c in _as_list(d, "recall_classifications") if isinstance(c, dict)]
    if not cls:
        return None
    return sum(1 for c in cls if _truthy(c.get("attributed"))) / len(cls)


def _fused_embedded(judge, d: dict, question: str, answer: str, reference: str,
                    *, want_relevancy: bool, want_correctness: bool) -> dict:
    """임베딩이 필요한 두 지표(response_relevancy·answer_correctness 의 유사도 성분)를
    **embed 1회**로 함께 계산한다. legacy 는 지표마다 따로 불러 2회였다."""
    out: dict = {}
    gen_qs: list[str] = []
    if want_relevancy:
        gen_qs = [q for q in _as_list(d, "generated_questions")
                  if isinstance(q, str) and q.strip()][:RELEVANCY_STRICTNESS]
        # 생성 질문이 없으면 legacy 처럼 0.0 으로 확정하지 않고 '미측정'으로 둔다 — legacy 의
        # 0.0 은 strictness 회 시도가 모두 질문을 못 냈다는 뜻이지만, 여기선 응답 1건이
        # 통째로 날아간 경우와 구분이 안 된다. 미측정으로 두면 _fused_repair 가 되묻는다.

    texts: list[str] = ([question] + gen_qs) if gen_qs else []
    sim_at = None
    if want_correctness:
        sim_at = len(texts)
        texts += [reference, answer]

    vecs = None
    if texts:
        try:
            vecs = _embed(judge, texts)
        except Exception:
            vecs = None
        if vecs is not None and len(vecs) != len(texts):
            vecs = None                          # 길이 불일치면 인덱스를 믿을 수 없다

    if gen_qs:
        if vecs:
            sims = [_cosine(vecs[0], v) for v in vecs[1:1 + len(gen_qs)]]
            # 회피형 판정은 fused 에서 답변당 1개(legacy 는 draft 마다 받아 전부 1일 때만 0).
            # 같은 답변을 두고 나온 판정이라 다수결이 아니라 그 값이 곧 결론이다.
            out["response_relevancy"] = (sum(sims) / len(sims)) * (0 if _truthy(d.get("noncommittal")) else 1)
        else:
            out["response_relevancy"] = None
    if want_correctness:
        sim = max(_cosine(vecs[sim_at], vecs[sim_at + 1]), 0.0) if vecs else None
        out.update(_correctness_score(_fused_correctness_counts(d), sim,
                                      "fused 응답에 correctness 블록 없음"))
    return out


def _fused_correctness_counts(d: dict):
    """correctness 블록 → (TP, FP, FN) 카운트. 블록이 없거나 형식이 깨졌으면 None(=factual 결손)."""
    corr = d.get("correctness")
    if not isinstance(corr, dict):
        return None
    return len(_as_list(corr, "TP")), len(_as_list(corr, "FP")), len(_as_list(corr, "FN"))


def _fused_repair(judge, out: dict, question: str, answer: str, contexts: list[str],
                  reference: str, *, want_faith: bool, want_prec: bool, want_corr: bool) -> None:
    """fused 응답에서 빠진 지표만 legacy 개별 호출로 메운다(out 을 제자리 수정).

    보통 0건이라 비용이 0이다 — 출력 절단·JSON 파싱 실패처럼 응답 하나가 통째로 날아간
    사고에서만 돈다. 이게 없으면 그런 사고 1건이 트랙 지표 5개를 전부 '미측정'으로 만들고,
    diagnose 의 정답 강등(_f1_ok)이 조용히 스킵돼 lexical 오통과가 성공으로 굳는다.
    보수 자체가 실패하면 이미 확보한 fused 값은 그대로 두고 물러난다."""
    missing: list[tuple[str, Callable[[], Any]]] = []
    if want_faith and "faithfulness" not in out:
        missing.append(("faithfulness", lambda: _faithfulness(judge, question, answer, contexts)))
    if "response_relevancy" not in out:
        missing.append(("response_relevancy", lambda: _response_relevancy(judge, question, answer)))
    if want_prec and "context_precision" not in out:
        missing.append(("context_precision", lambda: _context_precision(judge, question, reference, contexts)))
    if want_prec and "context_recall" not in out:
        missing.append(("context_recall", lambda: _context_recall(judge, question, reference, contexts)))
    # degraded = TP/FP/FN 분류가 빠져 유사도 단독으로 낸 점수다. 점수 자체는 있지만 강등
    # 근거(FN 카운트)가 없으므로 결손으로 보고 다시 받는다 — 성공하면 아래 update 가 덮어쓴다.
    if want_corr and ("answer_correctness" not in out or out.get("answer_correctness_degraded")):
        missing.append(("answer_correctness", lambda: _answer_correctness(judge, question, answer, reference)))
    if not missing:
        return
    # 보수 건수 = fused 가 아끼려던 개별 호출이 되살아난 수. 리포트가 집계해(scores.fused_repaired)
    # '통합했는데 실제로는 개별 호출로 되돌아가고 있다'를 로그가 아니라 지표로 보게 한다.
    out["fused_repaired"] = out.get("fused_repaired", 0) + len(missing)
    print(f"[Eval] RAGAS fused 결손 {[k for k, _ in missing]} → 개별 호출로 보수")
    try:
        values = parallel_map(lambda t: t[1](), missing, _inner_concurrency())
    except Exception as e:
        print(f"[Eval] RAGAS fused 보수 실패({e}) → 해당 지표 미측정 유지")
        return
    repaired = _drop_none(_merge_task_values(missing, values))
    # 보수로 factual 을 되찾았으면 degraded 표시는 사실이 아니게 된다(이 키는 legacy 조각이
    # 스스로 실을 때만 남아야 한다). 보수가 또 실패했으면 legacy 가 다시 실어 준다.
    if "answer_correctness" in repaired:
        out.pop("answer_correctness_degraded", None)
    out.update(repaired)


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


def evaluate_reasoning_mode(record: EvalRecord, judge) -> dict:
    """오라클 답변의 추론 실패 모드 단일 분류(모순/수치/해석/결합/기타)."""
    inp = {
        "user_input": record.probe.question,
        "response": record.oracle_answer or "",
        "retrieved_contexts": record.oracle_context or record.retrieved_context,
        "reference": record.probe.ground_truth or "",
    }
    d = _chat(judge, _ragas_prompt(_REASONING_MODE_INSTRUCTION, _SCHEMA_REASONING_MODE,
                                   _REASONING_MODE_EXAMPLES, inp), label="reasoning_mode")
    mode = d.get("mode")
    if not isinstance(mode, str) or mode not in _REASONING_MODES:
        return {}                                    # 미상·파싱 실패 → 미측정
    return {"reasoning_mode": mode}


def evaluate_abstention(record: EvalRecord, judge) -> dict:
    """기권 여부 이진 판정(AspectCritic). generation_abstention_failure 의 DEEP+ 경로."""
    return {
        "abstention": _aspect_critic(
            judge, _ASPECT_ABSTENTION, record.probe.question,
            record.generated_answer, record.retrieved_context,
        ),
    }


# ══════════════════════════════════════════════════════════════════
#  RAGAS 지표 알고리즘 (소스와 동일)
# ══════════════════════════════════════════════════════════════════

def _decompose_statements(judge, question: str, text: str, label: str = "statements") -> list[str]:
    """RAGAS StatementGenerator: 텍스트를 대명사 없는 독립 주장 문장들로 분해.
    faithfulness·answer_correctness 가 공유(답변/정답 모두 이 형식으로 분해).

    label 은 실패 로그 구분용 — 한 함수를 세 자리(faithfulness/정답분해/골드분해)가
    공유해서, 이름이 없으면 어느 쪽이 비었는지 로그로 못 가린다."""
    if not (text or "").strip():
        return []
    d = _chat(judge, _ragas_prompt(_FAITH_STMT_INSTRUCTION, _SCHEMA_STATEMENTS,
                                   _FAITH_STMT_EXAMPLES, {"question": question, "answer": text}),
              label=label, max_output_tokens=_LARGE_JUDGE_MAX_OUTPUT_TOKENS)
    return [s for s in _as_list(d, "statements") if isinstance(s, str) and s.strip()]


def _faithfulness(judge, question: str, answer: str, contexts: list[str]):
    """RAGAS Faithfulness (2단계): 답변→문장 분해 → 각 문장 NLI 판정 → 지지 비율."""
    if not (answer or "").strip() or not contexts:
        return None
    # 1. 문장 분해: 검증가능한 주장들로 분해
    statements = _decompose_statements(judge, question, answer, "faithfulness.statements")
    if not statements:
        return None
    # 2. NLI 판정: 각 주장이 컨텍스트만으로 추론 가능한지 판단
    context_str = "\n".join(contexts)
    d2 = _chat(judge, _ragas_prompt(_FAITH_NLI_INSTRUCTION, _SCHEMA_NLI,
                                    _FAITH_NLI_EXAMPLES, {"context": context_str, "statements": statements}),
               label="faithfulness.nli",
               max_output_tokens=_LARGE_JUDGE_MAX_OUTPUT_TOKENS)
    verdicts = [v for v in _as_list(d2, "statements") if isinstance(v, dict)]
    if not verdicts:
        return None
    supported = sum(1 for v in verdicts if _truthy(v.get("verdict")))
    return supported / len(verdicts)


def _answer_correctness(judge, question: str, answer: str, reference: str):
    """RAGAS AnswerCorrectness: 답변↔정답(gold) 비교 점수(0~1).
    factual F1(답변 문장을 정답 기준 TP/FP/FN 분류) 와 의미유사도(임베딩 코사인)의 가중합.
    lexical answer_match 오통과를 강등하는 gold-비교 신호다.

    반환은 트랙 dict 에 합쳐질 조각: {"answer_correctness": float}
    (+ factual 성분을 못 재면 {"answer_correctness_degraded": True}). 재료가 없거나 두 성분
    모두 실패하면 빈 dict — 즉 '미측정'이라 diagnose 가 lexical 판정을 그대로 확정한다."""
    if not (answer or "").strip() or not (reference or "").strip():
        return {}

    # ① factual F1 — 답변 문장을 정답 기준 TP/FP/FN 으로 분류
    counts = None
    degrade_reason = ""                          # factual 을 못 잰 이유(로그용) — 성공하면 빈 값
    ans_stmts = _decompose_statements(judge, question, answer, "correctness.answer_statements")
    ref_stmts = _decompose_statements(judge, question, reference, "correctness.gold_statements")
    if not ans_stmts or not ref_stmts:
        # '실패' 로 단정하지 않는다 — 기권("모르겠습니다")처럼 주장이 없는 답변은 0문장이
        # 정상 분해 결과다. 어느 쪽이 비었는지는 문장 수로 읽는다.
        degrade_reason = (f"분해 결과 없음(answer={len(ans_stmts)}문장, "
                          f"gold={len(ref_stmts)}문장, gold {len(reference)}자)")
    if ans_stmts and ref_stmts:
        d = _chat(judge, _ragas_prompt(_CORRECTNESS_INSTRUCTION, _SCHEMA_CORRECTNESS, _CORRECTNESS_EXAMPLES,
                                       {"question": question, "answer": ans_stmts, "ground_truth": ref_stmts}),
                  label="correctness.classify",
                  max_output_tokens=_LARGE_JUDGE_MAX_OUTPUT_TOKENS)
        counts = (len(_as_list(d, "TP")), len(_as_list(d, "FP")), len(_as_list(d, "FN")))
        if sum(counts) == 0:
            degrade_reason = (f"TP/FP/FN 분류 무응답(answer={len(ans_stmts)}문장, "
                              f"gold={len(ref_stmts)}문장, gold {len(reference)}자)")

    # ② 의미 유사도 — 답변↔정답 임베딩 코사인
    try:
        vecs = _embed(judge, [reference, answer])
    except Exception:
        vecs = None
    sim = max(_cosine(vecs[0], vecs[1]), 0.0) if vecs and len(vecs) >= 2 else None

    return _correctness_score(counts, sim, degrade_reason)


def _correctness_score(counts, sim, degrade_reason: str = "") -> dict:
    """(TP,FP,FN) 카운트 + 의미유사도 → answer_correctness 조각. legacy·fused 공용 계산식.

    counts=None 또는 denom==0 = 분류가 한 건도 안 나옴. 분해된 문장이 있으면 정상 분류는
    최소 1건이 나오므로 이건 판정기 무응답/파싱 실패다 — 이 성분만 버리고 남은 성분으로
    계속 간다. (예전엔 여기서 함수 전체가 None 이라 '미측정'이 되어, 판정기가 죽으면 강등이
    통째로 스킵되고 lexical 오통과가 그대로 성공 처리됐다.)

    degrade_reason 은 호출부가 아는 '왜 factual 이 없는지' — 리포트엔 건수만 남아서
    원인이 '골드가 길어 분해가 안 됨'인지 '분류 호출이 죽음'인지 구분이 안 된다. 처방이
    갈리는 지점이라(전자는 골드/프롬프트, 후자는 출력 상한) 로그로 드러낸다."""
    w_f, w_s = _ANSWER_CORRECTNESS_WEIGHTS
    components: list[tuple[float, float]] = []   # (가중치, 값) — 측정에 성공한 성분만 담는다
    out_counts: dict = {}                        # TP/FP/FN — factual 성공 시에만 채운다
    if counts is not None:
        tp, fp, fn = counts
        denom = tp + 0.5 * (fp + fn)
        if denom > 0:
            components.append((w_f, tp / denom))
            # TP=맞은 요소 / FP=gold 에 없는 군더더기 / FN=gold 에만 있는 누락 요소.
            # FN 이 generation_partial_answer 의 판별 근거다(임계 판정은 diagnose 소관).
            out_counts = {"answer_correctness_tp": tp,
                          "answer_correctness_fp": fp,
                          "answer_correctness_fn": fn}
    if sim is not None:
        components.append((w_s, sim))

    if not components:
        # 두 성분 다 실패 → 진짜 미측정. degraded 플래그가 안 붙어 리포트 집계에도 안 잡히는
        # 경로라, 여기서 안 남기면 '의미축이 왜 없는지'를 로그에서 아예 못 찾는다.
        print(f"[Eval] answer_correctness 미측정 — {degrade_reason or '원인 미상'} + 임베딩 실패")
        return {}
    # 성분이 하나만 측정돼도 가중 재정규화해 0~1 스케일을 유지한다.
    score = sum(w * v for w, v in components) / sum(w for w, _ in components)
    out = {"answer_correctness": score}
    out.update(out_counts)                       # factual 실패면 빈 dict = 카운트 미측정

    # factual(TP/FP/FN 분류)이 빠지면 유사도 단독 점수다. 이 폴백은 판정을 느슨하게 만들지
    # 않는다 — 이 지표를 보는 diagnose._f1_ok 은 이미 lexical 을 통과한 답에만 도달하므로,
    # 유사도가 할 수 있는 일은 '강등'뿐이고 통과시키는 답은 지표가 없었어도 통과했을 답이다.
    # 다만 부정문·근접 오답은 유사도도 높게 나와 못 거르므로, 그 실행의 정답 판정이
    # degrade 됐다는 사실 자체를 리포트로 드러낸다(집계: report._degraded_correctness_count).
    # 사유와 규모는 chat_json 쪽 라벨 로그와 짝을 이룬다.
    if not out_counts:
        out["answer_correctness_degraded"] = True
        print(f"[Eval] answer_correctness degrade — {degrade_reason or '원인 미상'}")
    return out


def _response_relevancy(judge, question: str, answer: str):
    """RAGAS AnswerRelevancy: 답변→질문 strictness(3)회 생성 → 원 질문과 코사인 평균. 모두 회피성이면 0."""
    if not (answer or "").strip():
        return 0.0
    # 답변으로부터 질문 n회 생성 (inner 동시성 1이면 기존 순차와 동일, 결과는 순서 보존)
    drafts = parallel_map(
        lambda _i: _chat(judge, _ragas_prompt(_RELEVANCY_INSTRUCTION, _SCHEMA_RELEVANCY,
                                              _RELEVANCY_EXAMPLES, {"response": answer}),
                         label="response_relevancy"),
        list(range(RELEVANCY_STRICTNESS)),
        _inner_concurrency(),
    )
    gen_qs, noncommittal = [], []
    for d in drafts:
        q = d.get("question")
        # LLM으로부터 생성된 str이 잘 존재하는지 판별 후 -> 판별에 필요한것 저장
        if isinstance(q, str) and q.strip():
            gen_qs.append(q)
            noncommittal.append(1 if _truthy(d.get("noncommittal")) else 0)
    if not gen_qs:
        # 질문 생성이 한 번도 성공 못 함(무응답·파싱 실패) — 답변이 무관해서 나온 0 이 아니다.
        print(f"[Eval] response_relevancy 0 — 질문 생성 {RELEVANCY_STRICTNESS}회 모두 실패")
        return 0.0
    all_noncommittal = all(n == 1 for n in noncommittal) # noncommittal: 답변이 회피형(잘모르겠다.)인지 판별
    # 임베딩 실패는 이 지표만 결측으로 만든다. 예전엔 예외가 그대로 올라가 parallel_map →
    # evaluate_real_track → _ragas_track 순으로 전파돼 트랙 전체가 {} 가 됐다 —
    # 임베딩이 없는 provider(OpenRouter 등)에서 faithfulness·context_* 까지 통째로
    # 사라지고 심판 호출 비용만 버려졌다. 나머지 두 임베딩 호출부(_answer_correctness 등)는
    # 이미 같은 방식으로 가드하고 있어, 여기만 빠져 있던 것.
    try:
        vecs = _embed(judge, [question] + gen_qs)  # Embedding
    except Exception:
        return None
    if not vecs or len(vecs) < 2:
        return None
    sims = [_cosine(vecs[0], v) for v in vecs[1:]] # Cosine Similarity
    if all_noncommittal:
        # 0 은 코사인이 낮아서가 아니라 회피 판정이 전부 1 이라 곱해진 결과다. 판정기가
        # 단정형 답변을 회피형으로 오분류해도 같은 0 이 나오므로(실측), 실제 코사인과
        # 표본 수를 남겨 '진짜 회피'와 '판정기 오분류'를 사후에 가를 수 있게 한다.
        print(f"[Eval] response_relevancy 0 — 생성 질문 {len(gen_qs)}개가 모두 noncommittal "
              f"판정(코사인 평균 {sum(sims) / len(sims):.2f}, strictness={RELEVANCY_STRICTNESS})")
        return 0.0
    return sum(sims) / len(sims)


def _context_precision(judge, question: str, reference: str, contexts: list[str]):
    """RAGAS ContextPrecision: 청크마다 유용성 판정 → 순위 가중 average precision."""
    if not contexts or not (reference or "").strip():
        return None
    # RAGAS: 청크 하나씩 판정 — parallel_map 은 순서를 보존하므로
    # _average_precision(순위 가중)의 입력이 순차 실행과 동일하다.
    decisions = parallel_map(
        lambda c: _chat(judge, _ragas_prompt(_CTX_PREC_INSTRUCTION, _SCHEMA_VERDICT,
                                             _CTX_PREC_EXAMPLES,
                                             {"question": question, "context": c, "answer": reference}),
                        label="context_precision"),
        list(contexts),
        _inner_concurrency(),
    )
    verdicts = [1 if _truthy(d.get("verdict")) else 0 for d in decisions]
    return _average_precision(verdicts)

def _context_recall(judge, question: str, reference: str, contexts: list[str]):
    """RAGAS ContextRecall: 정답(reference)을 문장별로 나눠 context 귀속 여부 → 귀속 비율."""
    if not contexts or not (reference or "").strip():
        return None
    context_str = "\n".join(contexts)
    d = _chat(judge, _ragas_prompt(_CTX_RECALL_INSTRUCTION, _SCHEMA_RECALL, _CTX_RECALL_EXAMPLES,
                                   {"question": question, "context": context_str, "answer": reference}),
              label="context_recall",
              max_output_tokens=_LARGE_JUDGE_MAX_OUTPUT_TOKENS)
    cls = _as_list(d, "classifications")
    if not cls:
        return None
    return sum(1 for c in cls if _truthy(c.get("attributed"))) / len(cls)


def _aspect_critic(judge, definition: str, user_input: str, response: str, contexts: list[str]) -> int:
    """RAGAS AspectCritic: definition 기준 이진 판정(strictness=1 → 단일 호출)."""
    instruction = _ASPECT_INSTRUCTION_TMPL.format(definition=definition)
    inp = {"user_input": user_input, "response": response, "retrieved_contexts": contexts}
    d = _chat(judge, _ragas_prompt(instruction, _SCHEMA_VERDICT, [], inp), label="aspect_critic")
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

def _chat(judge, prompt: str, max_output_tokens: int | None = None, label: str = "",
          cache_prefix: str = "", json_schema: dict | None = None) -> dict:
    """RAGAS 형식 프롬프트를 JSON 강제로 호출 → dict. 실패 시 {}.

    max_output_tokens 는 fused 처럼 응답 구조가 큰 호출만 명시한다(미지정=provider 기본).
    label 은 실패 로그에 찍히는 호출 이름 — 호출부가 여럿인데 로그가 전부 같은 문구라
    어느 지표의 심판이 죽었는지 못 가렸다(정답 판정 degrade 원인 추적).

    cache_prefix 는 매 호출 동일한 앞부분이다. anthropic 에서만 system 블록으로 올라가
    캐시 지점이 되고(두 번째 호출부터 그 몫이 입력가의 0.1 배), 다른 provider 에서는
    prompt 앞에 도로 붙어 예전과 바이트가 같은 한 덩어리가 된다 — 캐싱과 무관한
    provider 의 프롬프트를 바꾸지 않기 위해서다. 기본값 "" 은 기존 호출부(지표별 legacy
    경로)가 한 덩어리 프롬프트를 그대로 쓰게 둔다.

    json_schema 도 anthropic 전용이다(output_config.format). 나머지 provider 는 무시한다."""
    kwargs: dict = {"label": label}
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    if json_schema is not None:
        kwargs["json_schema"] = json_schema
    if cache_prefix:
        kwargs["cache_prefix"] = cache_prefix
    return llm_provider.chat_json("", prompt, **kwargs)


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


def _merge_task_values(tasks: list, values: list) -> dict:
    """병렬 태스크 결과를 트랙 dict 으로 합친다.
    대부분의 지표는 스칼라를 돌려주지만, answer_correctness 는 부가 플래그를 함께 실어야 해서
    dict 조각을 돌려준다 — dict 이면 펼쳐 넣고 아니면 태스크 키에 매핑한다."""
    out: dict = {}
    for (key, _), value in zip(tasks, values):
        if isinstance(value, dict):
            out.update(value)
        else:
            out[key] = value
    return out


def _drop_none(d: dict) -> dict:
    """value가 None이면 버리기"""
    return {k: v for k, v in d.items() if v is not None}


# ══════════════════════════════════════════════════════════════════
#  진입 시 RAGAS 측정 + 트랙별 값 접근자 (diagnose 가 소비하는 tier3 측정 API)
#    실제 트랙  = record.ragas        (검색결과 컨텍스트로 생성한 답)
#    오라클 트랙 = record.oracle_ragas  (gold 컨텍스트로 생성한 답)
#  자원(_ctx.ragas_fn)은 agent 가 set_context 로 주입한다. 임계값 판정은 diagnose 소관.
# ══════════════════════════════════════════════════════════════════

def _compute_ragas_real(record: EvalRecord) -> None:
    """실제 트랙 RAGAS 를 record.ragas 에 계산·저장. (STEP3-2, diagnose 진입 시 1회.)

    성공/실패 판정 전에 항상 돌린다 — 두 가지가 이걸 필요로 한다:
      1. diagnose 의 정답 강등 판정(_f1_ok 이 record.ragas_answer_correctness 를 읽는다)
      2. report/scoring 의 RAGAS 평균 — 실제 트랙만 쓰므로, 성공 probe 도 점수를 가져야
         '진단이 돌아간 실패 probe'만의 편향된 평균이 되지 않는다.

    비용 게이트는 DEEP 유지 — 그 미만 모드에선 LLM 을 한 번도 부르지 않는다."""
    if active_mode() < Mode.DEEP:
        return
    _ensure_ragas(record, "real")


def _compute_ragas_oracle(record: EvalRecord) -> None:
    """오라클 트랙 RAGAS 를 record.oracle_ragas 에 계산·저장.

    실패로 판정된 probe 에서만 부른다 — 오라클 값의 소비처가 B그룹(생성 실패) 라벨과
    _oracle_ok 뿐이고, report/scoring 의 평균은 실제 트랙만 쓰기 때문이다. 성공 probe 는
    진단 자체를 건너뛰므로 이 비용(트랙 하나치 LLM 호출)을 지불할 이유가 없다.

    lazy(_faith_oracle 등이 알아서 _ensure_ragas 호출)로 두지 않고 여기서 명시적으로 채우는
    이유: _oracle_ok 이 읽는 record.oracle_ragas_answer_correctness 는 dict 만 보는 property 라
    ensure 를 트리거하지 않는다 — lazy 로 두면 오라클 쪽 answer_correctness 강등이 조용히 죽는다."""
    if active_mode() < Mode.DEEP:
        return
    _ensure_ragas(record, "oracle")


def _ensure_ragas(record: EvalRecord, track: str):
    """트랙 RAGAS 점수를 record 에 계산·저장(트랙별 1회만).
    빈 결과({})여도 *_done 플래그로 '시도함'을 기록해 같은 트랙 재-LLM호출을 막는다.
    (oracle 답이 없으면 _ctx.ragas_fn 이 {} 를 돌려준다.)"""
    if _ctx.ragas_fn is None:
        return
    if track == "oracle":
        if not record.oracle_ragas_done:
            record.oracle_ragas_done = True
            record.oracle_ragas = _ctx.ragas_fn(record, "oracle") or {}
    elif not record.ragas_done:
        record.ragas_done = True
        record.ragas = _ctx.ragas_fn(record, "real") or {}


def _abstention_judged(record: EvalRecord):
    """AspectCritic 기권 판정. tier3, DEEP+ / 미측정·자원없음 None(→ 마커 휴리스틱 폴백).

    memoize 는 record.aspect(실행 단위) — signals(=diagnosis_cache)는 index_config·코퍼스
    버전으로만 무효화돼서, 매 실행 새로 생성되는 generated_answer 에 물린 판정엔 못 쓴다.
    실패({})도 None 으로 남겨 같은 실행에서 재호출을 막는다(ragas_done 과 같은 규약).
    """
    if active_mode() < Mode.DEEP or _ctx.ragas_fn is None:
        return None
    if "abstention" not in record.aspect:
        verdict = (_ctx.ragas_fn(record, "abstention") or {}).get("abstention")
        record.aspect["abstention"] = None if verdict is None else bool(verdict)
    return record.aspect["abstention"]


def _reasoning_mode_oracle(record: EvalRecord):
    """오라클 답변의 추론 실패 모드(문자열). tier3, DEEP+ / 오라클 답·자원 없거나 미상이면 None.
    memoize 는 _abstention_judged 와 같은 이유로 record.aspect(실행 단위)."""
    if active_mode() < Mode.DEEP or _ctx.ragas_fn is None or record.oracle_answer is None:
        return None
    if "reasoning_mode" not in record.aspect:
        mode = (_ctx.ragas_fn(record, "reasoning_mode") or {}).get("reasoning_mode")
        record.aspect["reasoning_mode"] = mode if mode in _REASONING_MODES else None
    return record.aspect["reasoning_mode"]


def _faith(record: EvalRecord):
    """faithfulness(충실도) 값 — 실제 트랙. tier3, DEEP+ / 미측정 None."""
    if active_mode() < Mode.DEEP:
        return None
    _ensure_ragas(record, "real")
    return record.ragas.get("faithfulness")


def _faith_oracle(record: EvalRecord):
    """faithfulness(충실도) 값 — 오라클 트랙. tier3, DEEP+ / 미측정 None."""
    if active_mode() < Mode.DEEP:
        return None
    _ensure_ragas(record, "oracle")
    return record.oracle_ragas.get("faithfulness")


def _correctness_counts_oracle(record: EvalRecord):
    """오라클 트랙 (TP, FP, FN) 카운트. tier3, DEEP+ / factual 성분 미측정(degraded)이면 None."""
    if active_mode() < Mode.DEEP:
        return None
    _ensure_ragas(record, "oracle")
    d = record.oracle_ragas
    if "answer_correctness_fn" not in d:
        return None
    return (d["answer_correctness_tp"], d["answer_correctness_fp"], d["answer_correctness_fn"])


def _ctx_precision(record: EvalRecord):
    """context_precision(검색 컨텍스트 유용성) — 실제 트랙. tier3, DEEP+ / 미측정 None.
    지금까지 소비처가 리포트 평균뿐이라, 라벨이 읽어도 LLM 추가 호출은 없다."""
    if active_mode() < Mode.DEEP:
        return None
    _ensure_ragas(record, "real")
    return record.ragas.get("context_precision")


def _rel(record: EvalRecord):
    """response_relevancy(관련성) 값 — 실제 트랙. tier3, DEEP+ / 미측정 None."""
    if active_mode() < Mode.DEEP:
        return None
    _ensure_ragas(record, "real")
    return record.ragas.get("response_relevancy")


def _rel_oracle(record: EvalRecord):
    """response_relevancy(관련성) 값 — 오라클 트랙. tier3, DEEP+ / 미측정 None."""
    if active_mode() < Mode.DEEP:
        return None
    _ensure_ragas(record, "oracle")
    return record.oracle_ragas.get("response_relevancy")


    # answer_correctness 값은 record.ragas_answer_correctness / oracle_ragas_answer_correctness
    # 속성으로 노출된다(EvalRecord). diagnose 의 정답 강등 판정이 그 속성을 직접 읽는다.
