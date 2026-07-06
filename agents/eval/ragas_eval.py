"""
agents/eval/ragas_eval.py
STEP3-2: LLM 진단 (RAGAS 지표 측정)

설계 문서 '3-2단계: LLM 진단'을 구현한다. RAGAS 4개 지표 + 커스텀 AspectCritic 을
**LLM-as-Judge** 로 측정한다.
    - 실제 트랙  : Faithfulness, Context Precision/Recall, Response Relevancy
    - 오라클 트랙 : Faithfulness, Response Relevancy (gold context 투입 결과)
    - AspectCritic: staleness / contradiction 이진 판정

구현 메모:
    RAGAS 라이브러리는 langchain 버전에 매우 민감해(설치돼 있어도 import가 깨지는 경우가 많음)
    파이프라인을 라이브러리에 묶지 않는다. 대신 RAGAS 논문/문서에 정의된 **각 지표의 알고리즘을
    OpenAI 로 직접 구현**한다(청구 분해→근거 대조 등). 출력 스키마·의미는 RAGAS와 동일하다.
    (환경이 ragas 라이브러리를 지원하면 evaluate_*_track 내부만 교체 가능.)

비용·재현성(설계):
    - 기본 비활성. `EVAL_ENABLE_LLM=1` 일 때만 동작(사용자 수동 활성화).
    - 응답 모델 ≠ 평가 모델(EVAL_JUDGE_MODEL, 기본 gpt-4o), temperature=0 고정(재현성).
    - 키 없음·호출 실패·JSON 파싱 실패 → 조용히 건너뛰고(폴백) 규칙 지표(STEP3-1)로 진행.
"""
from __future__ import annotations

import json
import math
import os

from agents.eval.types import Branch, EvalRecord


# ── 활성화 / 심판 LLM ─────────────────────────────────────────────

def llm_eval_enabled() -> bool:
    """LLM(RAGAS) 진단 활성화 여부. 기본 꺼짐."""
    return os.getenv("EVAL_ENABLE_LLM", "").strip().lower() in ("1", "true", "yes", "on")


def _judge():
    """평가(심판)용 OpenAI 클라이언트와 모델명. 키·라이브러리 없으면 None."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    # 설계 원칙: 응답 모델(gpt-4o-mini)과 다른 모델로 채점
    return OpenAI(), os.getenv("EVAL_JUDGE_MODEL", "gpt-4o")


# ── 메인 진입 ─────────────────────────────────────────────────────

def evaluate(record: EvalRecord) -> None:
    """
    브랜치에 따라 실제/오라클 트랙 RAGAS 지표와 AspectCritic 을 계산해
    record.ragas / record.oracle_ragas / record.aspect 에 채운다.
    비활성·실패 시 아무것도 채우지 않는다(폴백).

    트랙 선택(설계 STEP3-2 표):
        성공/검색실패              → 스킵
        검색+생성실패/애매함(생성)   → 오라클 트랙
        검색 부분실패/애매함(컨텍스트) → 실제 트랙 (+ 오라클)
    """
    if not llm_eval_enabled():
        return
    if record.branch in (Branch.SUCCESS, Branch.NO_ANSWER_OK):
        return  # 진단할 게 없으면 LLM 호출 안 함

    judge = _judge()
    if judge is None:
        print("[Eval] STEP3-2: 평가 LLM 미설정(OPENAI_API_KEY) → RAGAS 스킵")
        return

    run_real, run_oracle = _tracks_for(record.branch)
    try:
        if run_real:
            record.ragas = evaluate_real_track(record, judge)
        if run_oracle and record.oracle_answer is not None:
            record.oracle_ragas = evaluate_oracle_track(record, judge)
        record.aspect = evaluate_aspect_critics(record, judge)
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


# ── 트랙별 측정 ───────────────────────────────────────────────────

def evaluate_real_track(record: EvalRecord, judge) -> dict:
    """
    실제 검색·생성 결과에 대한 RAGAS 지표.
    반환 키: faithfulness, response_relevancy, (+정답 있으면) context_precision, context_recall
    """
    q = record.probe.question
    ans = record.generated_answer
    ctx = record.retrieved_context
    ref = record.probe.ground_truth

    out: dict = {}
    out["faithfulness"] = _faithfulness(judge, q, ans, ctx)
    out["response_relevancy"] = _response_relevancy(judge, q, ans)
    if ref:  # reference 있어야 Context Precision/Recall 계산 가능
        out["context_precision"] = _context_precision(judge, q, ref, ctx)
        out["context_recall"] = _context_recall(judge, ref, ctx)
    return _drop_none(out)


def evaluate_oracle_track(record: EvalRecord, judge) -> dict:
    """gold context 로 생성한 답(oracle_answer)에 대한 RAGAS 지표. 반환: faithfulness, response_relevancy."""
    q = record.probe.question
    ans = record.oracle_answer or ""
    ctx = record.oracle_context or record.retrieved_context  # gold context 우선
    return _drop_none({
        "faithfulness": _faithfulness(judge, q, ans, ctx),
        "response_relevancy": _response_relevancy(judge, q, ans),
    })


def evaluate_aspect_critics(record: EvalRecord, judge) -> dict:
    """
    커스텀 AspectCritic(이진). 설계 §6 커스텀 Finding 탐지용.
        staleness     : 오래된(시점 지난) 정보 포함?
        contradiction : 컨텍스트와 모순?
    반환 예: {"staleness": 0, "contradiction": 1}
    [구현 포인트] 사용자 설정 definition 을 추가로 주입할 수 있음.
    """
    q = record.probe.question
    ans = record.generated_answer
    ctx = "\n\n".join(record.retrieved_context)
    return {
        "staleness": _aspect_critic(
            judge, "답변 또는 컨텍스트에 시점이 지나 더 이상 유효하지 않은(오래된) 정보가 포함되어 있는가?",
            q, ans, ctx),
        "contradiction": _aspect_critic(
            judge, "답변이 제공된 컨텍스트와 사실상 모순되는 내용을 포함하는가?",
            q, ans, ctx),
    }


# ── RAGAS 지표 알고리즘 (LLM-as-Judge) ───────────────────────────

def _faithfulness(judge, question: str, answer: str, contexts: list[str]):
    """
    답변을 검증 가능한 주장으로 분해 → 각 주장이 컨텍스트로 뒷받침되는 비율.
    (컨텍스트에 없는 내용을 지어냈으면 낮음 = 환각 탐지)
    """
    if not (answer or "").strip() or not contexts:
        return None
    ctx = "\n\n".join(contexts)
    data = _chat_json(judge,
        system=("너는 사실 검증기다. 답변을 독립적으로 검증 가능한 원자적 주장(claim)들로 나누고, "
                "각 주장이 오직 [컨텍스트]만으로 추론 가능한지 판정하라. "
                '반드시 JSON {"claims":[{"claim":str,"supported":bool}]} 형식으로만 답하라.'),
        user=f"[질문]\n{question}\n\n[컨텍스트]\n{ctx}\n\n[답변]\n{answer}")
    claims = _as_list(data, "claims")
    if not claims:
        return None
    supported = sum(1 for c in claims if _truthy(c.get("supported")))
    return supported / len(claims)


def _response_relevancy(judge, question: str, answer: str):
    """답변만 보고 역으로 질문 N개 생성 → 원 질문과의 임베딩 코사인 유사도 평균."""
    if not (answer or "").strip():
        return 0.0
    data = _chat_json(judge,
        system=("주어진 [답변]만 보고, 이 답변이 대답하고 있는 질문 3개를 생성하라. "
                '반드시 JSON {"questions":[str, str, str]} 형식으로만 답하라.'),
        user=f"[답변]\n{answer}")
    gen_qs = [q for q in _as_list(data, "questions") if isinstance(q, str) and q.strip()]
    if not gen_qs:
        return None
    vecs = _embed(judge, [question] + gen_qs)
    if not vecs or len(vecs) < 2:
        return None
    qv, sims = vecs[0], [_cosine(vecs[0], v) for v in vecs[1:]]
    return sum(sims) / len(sims) if sims else 0.0


def _context_precision(judge, question: str, reference: str, contexts: list[str]):
    """
    각 청크가 정답에 유용한지 판정 → 순위 가중 정밀도.
    관련 청크가 상위에 있을수록 점수 높음(순서 중요).
    """
    if not contexts:
        return None
    verdicts = _chunk_relevance(judge, question, reference, contexts)
    num_relevant = sum(1 for v in verdicts if v)
    if num_relevant == 0:
        return 0.0
    total, hits = 0.0, 0
    for k, rel in enumerate(verdicts, start=1):
        if rel:
            hits += 1
            total += hits / k          # 이 위치까지의 Precision@k
    return total / num_relevant


def _chunk_relevance(judge, question: str, reference: str, contexts: list[str]) -> list[bool]:
    """청크별 유용성 판정 리스트(청크 순서대로 bool)."""
    numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts))
    data = _chat_json(judge,
        system=("각 컨텍스트 청크가 [질문]에 답(=[정답]의 근거)하는 데 유용한지 판정하라. "
                '반드시 JSON {"verdicts":[bool, ...]} 형식으로, 청크 순서대로만 답하라.'),
        user=f"[질문]\n{question}\n\n[정답]\n{reference}\n\n[청크들]\n{numbered}")
    v = [_truthy(x) for x in _as_list(data, "verdicts")][:len(contexts)]
    v += [False] * (len(contexts) - len(v))     # 부족분 패딩
    return v


def _context_recall(judge, reference: str, contexts: list[str]):
    """정답(reference)을 주장으로 분해 → 각 주장이 컨텍스트에서 찾아지는 비율(누락 감지)."""
    if not contexts or not (reference or "").strip():
        return None
    ctx = "\n\n".join(contexts)
    data = _chat_json(judge,
        system=("[정답]을 문장 단위의 원자적 주장으로 나누고, 각 주장이 [컨텍스트]로 뒷받침(귀속)되는지 판정하라. "
                '반드시 JSON {"claims":[{"claim":str,"attributed":bool}]} 형식으로만 답하라.'),
        user=f"[컨텍스트]\n{ctx}\n\n[정답]\n{reference}")
    claims = _as_list(data, "claims")
    if not claims:
        return None
    return sum(1 for c in claims if _truthy(c.get("attributed"))) / len(claims)


def _aspect_critic(judge, definition: str, question: str, answer: str, context: str) -> int:
    """자유 형식 기준(definition)에 답변이 해당하면 1, 아니면 0.
    [구현 포인트] 편향 완화로 온도 변화 다수결(3회)·position swap 을 추가할 수 있음(현재 temp=0 단일)."""
    data = _chat_json(judge,
        system=(f'다음 기준에 해당하면 1, 아니면 0으로 판정하라. 기준: "{definition}" '
                '반드시 JSON {"verdict":0 또는 1} 형식으로만 답하라.'),
        user=f"[질문]\n{question}\n\n[컨텍스트]\n{context}\n\n[답변]\n{answer}")
    return 1 if _truthy((data or {}).get("verdict")) else 0


# ── OpenAI 호출 헬퍼 ─────────────────────────────────────────────

def _chat_json(judge, system: str, user: str) -> dict:
    """JSON 응답 강제 chat 호출 → dict. 실패 시 {}."""
    client, model = judge
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content or "{}"
    try:
        obj = json.loads(content)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def _embed(judge, texts: list[str]) -> list[list[float]]:
    """텍스트 리스트 → 임베딩 벡터 리스트. (심판 클라이언트 재사용)"""
    client, _ = judge
    model = os.getenv("EVAL_EMBED_MODEL", "text-embedding-3-small")
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


# ── 순수 유틸 ─────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _as_list(data, key: str) -> list:
    """data[key] 가 리스트면 반환, 아니면 []."""
    if isinstance(data, dict) and isinstance(data.get(key), list):
        return data[key]
    return []


def _truthy(v) -> bool:
    """bool/1/'1'/'true'/'yes' 를 True 로."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v == 1
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "t")
    return False


def _drop_none(d: dict) -> dict:
    """None 값 키 제거(측정 실패한 지표는 리포트/진단에서 빠지도록)."""
    return {k: v for k, v in d.items() if v is not None}
