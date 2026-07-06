"""
agents/eval/ragas_eval.py
STEP3-2: LLM 진단 (RAGAS 점수 측정)

설계 문서 '3-2단계: LLM 진단' 구현.
LLM-as-Judge 로 4개 RAGAS 지표 + 커스텀 AspectCritic 을 측정한다.
    - 실제 트랙  : Faithfulness, Context Precision/Recall, Response Relevancy
    - 오라클 트랙 : Faithfulness, Response Relevancy (gold context 투입 결과)
    - AspectCritic: staleness / contradiction 등 커스텀 이진 판정

비용·재현성 주의(설계): LLM 호출은 지표당 1~3회. 그래서 기본 비활성이며,
`EVAL_ENABLE_LLM=1` 로 명시적으로 켤 때만 동작한다(설계: '사용자가 수동 활성화').
켜져 있어도 ragas 미설치·키 없음·호출 실패면 조용히 빈 dict 를 돌려주고(폴백)
규칙 지표(STEP3-1)만으로 진단이 진행되게 한다.

[구현 포인트]
    - 아래 evaluate_* 는 RAGAS 연동 지점의 골격이다. ragas 버전에 맞춰
      metrics.collections(Faithfulness/ContextPrecision/...)를 실제로 호출하도록 채운다.
    - LLM-as-Judge 완화기법(응답모델≠평가모델, temperature=0, position swap,
      다수결)을 평가 LLM 구성에 반영한다.
"""
from __future__ import annotations

import os

from agents.eval.types import Branch, EvalRecord


# ── 활성화 여부 ───────────────────────────────────────────────────

def llm_eval_enabled() -> bool:
    """LLM(RAGAS) 진단 활성화 여부. 기본 꺼짐."""
    return os.getenv("EVAL_ENABLE_LLM", "").strip().lower() in ("1", "true", "yes", "on")


def _evaluator_llm():
    """
    평가용 LLM 핸들 로드. 없으면 None.
    [구현 포인트] 설계 원칙상 응답 모델(gpt-4o-mini)과 다른 모델(예: gpt-4o)로.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        # ragas 래퍼는 버전마다 다름 → 실제 연동 시 여기서 구성
        from ragas.llms import llm_factory
        return llm_factory(os.getenv("EVAL_JUDGE_MODEL", "gpt-4o"))
    except Exception:
        return None


# ── 메인 진입 ─────────────────────────────────────────────────────

def evaluate(record: EvalRecord) -> None:
    """
    브랜치에 따라 실제/오라클 트랙 RAGAS 지표와 AspectCritic 을 계산해
    record.ragas / record.oracle_ragas / record.aspect 에 채운다.
    (활성화 안 됐거나 실패하면 아무것도 채우지 않고 넘어간다 = 폴백)

    설계 STEP3-2 표의 트랙 선택:
        성공/검색실패              → 스킵
        검색+생성실패/애매함(생성)   → 오라클 트랙
        검색 부분실패/애매함(컨텍스트) → 실제 트랙 (+ 오라클)
    """
    if not llm_eval_enabled():
        return

    llm = _evaluator_llm()
    if llm is None:
        print("[Eval] STEP3-2: 평가 LLM 미설정 → RAGAS 스킵(규칙 지표만 사용)")
        return

    run_real, run_oracle = _tracks_for(record.branch)
    try:
        if run_real:
            record.ragas = evaluate_real_track(record, llm)
        if run_oracle and record.oracle_answer is not None:
            record.oracle_ragas = evaluate_oracle_track(record, llm)
        record.aspect = evaluate_aspect_critics(record, llm)
    except Exception as e:  # 폴백: 어떤 실패도 파이프라인을 멈추지 않음
        print(f"[Eval] STEP3-2: RAGAS 측정 실패({e}) → 규칙 지표로 진행")


def _tracks_for(branch: str) -> tuple[bool, bool]:
    """(실제 트랙 실행?, 오라클 트랙 실행?) — 설계 STEP3-2 표."""
    real = branch in (
        Branch.RETRIEVAL_PARTIAL, Branch.RETRIEVAL_PARTIAL_GEN_FAIL,
        Branch.AMBIGUOUS_CONTEXT, Branch.NO_ANSWER_VIOLATION,
    )
    oracle = branch in (
        Branch.RETRIEVAL_GEN_FAIL, Branch.RETRIEVAL_PARTIAL_GEN_FAIL,
        Branch.AMBIGUOUS_CONTEXT, Branch.AMBIGUOUS_GEN,
    )
    return real, oracle


# ── 트랙별 측정 (골격) ────────────────────────────────────────────

def evaluate_real_track(record: EvalRecord, llm) -> dict:
    """
    실제 검색·생성 결과에 대한 RAGAS 지표.
    반환 키: faithfulness, context_precision, context_recall, response_relevancy
    [구현 포인트] ragas.metrics.collections 로 실제 계산.
    """
    # TODO: RAGAS 실제 연동. 예)
    #   from ragas.metrics.collections import Faithfulness, ContextPrecision, ...
    #   faithfulness = await Faithfulness(llm=llm).ascore(
    #       user_input=record.probe.question,
    #       response=record.generated_answer,
    #       retrieved_contexts=record.retrieved_context)
    return {}


def evaluate_oracle_track(record: EvalRecord, llm) -> dict:
    """
    gold context 로 생성한 답(oracle_answer)에 대한 RAGAS 지표.
    반환 키: faithfulness, response_relevancy
    [구현 포인트] ragas 연동.
    """
    return {}


def evaluate_aspect_critics(record: EvalRecord, llm) -> dict:
    """
    커스텀 AspectCritic(이진) 결과. 설계 §6 커스텀 Finding 탐지용.
        staleness      : 오래된 정보 포함?
        contradiction  : 컨텍스트와 모순?
    반환 예: {"staleness": 0, "contradiction": 1}
    [구현 포인트] ragas.metrics.AspectCritic 연동. 사용자 정의 definition 주입.
    """
    return {}
