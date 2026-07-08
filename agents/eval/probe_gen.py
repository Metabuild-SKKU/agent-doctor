"""
agents/eval/probe_gen.py
STEP1: Probe(진단용 질문) 생성

설계 문서 'STEP 1: Probe 생성' 구현.
Probe = RAG 파이프라인을 진단하기 위한 테스트 질문 집합.

Probe 소스별 신뢰도 (설계 §3):
    user_log(실사용 쿼리)  >  taxonomy(사람 작성)  >  llm_generated(자동 생성)

생성 우선순위:
    1) state.user_questions 가 있으면 → user_log Probe
    2) 없으면 → 청크 기반 자동 생성(llm_generated)

현재 구현: Single-Hop Specific(단일 청크·사실 기반) 질문만 LLM으로 생성한다
(청크 내용을 그대로 베끼지 않도록 질문·정답을 LLM이 새로 구성 — retrieval_temp.py의
_llm_generate 와 동일한 폴백 규칙: OPENAI_API_KEY 없거나 호출 실패 시 휴리스틱 추출로 대체).

[구현 포인트]  (다음 단계로 남겨둠)
    - Single-Hop Abstract / Multi-Hop(bridge·comparison·aggregation) 추가.
    - DataMorgana 20%, 무응답(Held-out·False Premise) 5% 비중 혼합.
    - RAGAS TestsetGenerator 식 지식그래프(청크 간 관계) 기반 시나리오 생성으로 확장.
    - eval_probes.json 영속화 + 문서 diff 기반 증분 생성(골든 테스트셋 재사용).
    - gold_doc_id/gold_char_span 스키마 추가(재청킹에도 안 깨지는 기준).
"""
from __future__ import annotations

import json
import os
import re

from core.schema import Chunk, Probe
from core.state import AgentDoctorState

# 자동 생성 기본 개수 (설계: testset_size=5~10 으로 시작해 비용 확인 후 확대)
DEFAULT_TESTSET_SIZE = 5


def generate_probes(state: AgentDoctorState) -> list[Probe]:
    """
    state 를 보고 Probe 리스트를 생성한다.

    읽기: state.user_questions, state.chunks
    """
    """
    user log 기반 프로브 생성
    if state.user_questions:
        probes = _from_user_questions(state.user_questions)
        print(f"[Eval] STEP1: user_log Probe {len(probes)}개 생성")
        return probes
    """
        
    probes = _from_chunks(state.chunks, DEFAULT_TESTSET_SIZE)
    print(f"[Eval] STEP1: llm_generated(폴백) Probe {len(probes)}개 생성")
    return probes


# ── user_log: 실사용 질문 기반 ────────────────────────────────────

def _from_user_questions(questions: list[str]) -> list[Probe]:
    """
    사용자가 넘긴 질문을 Probe 로 변환.
    ground_truth·gold_chunk_ids 가 없으므로 recall@k·F1 은 계산 불가 →
    RAGAS 의 무정답 지표(Faithfulness, Response Relevancy)로만 평가된다.
    """
    probes = []
    for i, q in enumerate(questions):
        q = (q or "").strip()
        if not q:
            continue
        probes.append(Probe(
            probe_id=f"probe_user_{i:03d}",
            question=q,
            source="user_log",
            answer_exists=None,      # 미상 (Oracle 테스트로 추후 판정)
            ground_truth=None,
            gold_chunk_ids=[],
            qtype=None,
        ))
    return probes


# ── llm_generated: 청크 기반 (Single-Hop Specific) ────────────────

_SENT_SPLIT = re.compile(r"(?<=[.!?。？！\n])\s+")


def _from_chunks(chunks: list[Chunk], size: int) -> list[Probe]:
    """
    청크마다 Single-Hop Specific Probe 를 만든다.
    각 청크에 대해 LLM 생성을 시도하고, 실패/미설정 시 휴리스틱 추출로 대체한다.
    """
    probes: list[Probe] = []
    # 너무 짧은 청크는 스킵, 앞에서부터 size 개 사용
    usable = [c for c in chunks if c.text and len(c.text.strip()) >= 20]
    for i, chunk in enumerate(usable[:size]):
        question, ground_truth = _llm_generate_single_hop(chunk.text) or _heuristic_single_hop(chunk.text)
        probes.append(Probe(
            probe_id=f"probe_gen_{i:03d}",
            question=question,
            source="llm_generated",
            expected_difficulty="medium",
            answer_exists=True,
            ground_truth=ground_truth,
            gold_chunk_ids=[chunk.chunk_id],
            qtype=None,                 # 단일홉
        ))
    return probes


def _heuristic_single_hop(text: str) -> tuple[str, str]:
    """LLM 미사용/실패 시 폴백: 청크 앞부분을 질문 주제로, 청크 전문을 정답으로 그대로 사용."""
    topic = _topic_of(text)
    return f"{topic}에 대해 설명해줘.", text.strip()


def _llm_generate_single_hop(chunk_text: str) -> tuple[str, str] | None:
    """
    OpenAI 로 Single-Hop Specific (질문, 정답) 쌍을 생성한다.
    청크 문장을 그대로 베끼지 않도록 질문·정답 모두 새로 구성하게 지시한다.
    키/라이브러리 없거나 호출·파싱 실패 시 None(호출부가 휴리스틱으로 대체).
    """
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    try:
        client = OpenAI()
        model = os.getenv("EVAL_GEN_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content":
                    "너는 RAG 파이프라인 평가용 테스트 질문을 설계하는 평가자다. "
                    "주어진 문서 조각(컨텍스트) 하나만으로 답할 수 있는, 실제 사용자가 물어볼 법한 "
                    "구체적인 사실 기반 질문(Single-Hop Specific) 하나와 그 정답을 만들어라. "
                    "질문과 정답 모두 컨텍스트 문장을 그대로 베끼지 말고 자기 말로 다시 구성하되, "
                    "정답은 컨텍스트에 있는 사실에서 벗어나면 안 된다. "
                    "반드시 {\"question\": str, \"ground_truth\": str} 형태의 JSON으로만 답하라."},
                {"role": "user", "content": f"[컨텍스트]\n{chunk_text}"},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        question = (data.get("question") or "").strip()
        ground_truth = (data.get("ground_truth") or "").strip()
        if not question or not ground_truth:
            return None
        return question, ground_truth
    except Exception as e:
        print(f"[Eval] STEP1: Probe LLM 생성 실패({e}) → 휴리스틱 폴백")
        return None


def _topic_of(text: str) -> str:
    """청크 첫 문장/앞부분을 질문의 주제 문구로 사용."""
    text = text.strip()
    first = _SENT_SPLIT.split(text, maxsplit=1)[0]
    first = first.strip().strip("#•-*> ").strip()
    # 너무 길면 앞 40자만
    return first[:40] if len(first) > 40 else first
