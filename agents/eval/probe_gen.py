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

[구현 포인트]  (현재는 파이프라인이 끊기지 않게 하는 최소 폴백만 구현)
    - 실제 자동 생성은 RAGAS TestsetGenerator(지식그래프 + 시나리오)로 교체한다.
      · 75% RAGAS / 20% DataMorgana / 5% 무응답(Held-out·False Premise) 혼합
      · 단일홉·멀티홉(bridge/comparison/aggregation) 다양화
    - ⚠️ 아래 폴백은 "청크에서 직접" 질문을 만든다. 이는 설계가 경고하는
      "AI가 만들고 AI가 평가" 문제에 해당하므로, 신뢰할 수 있는 평가에는
      반드시 외부 생성/사람 검수를 붙여야 한다. 지금은 골격 검증용이다.
"""
from __future__ import annotations

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


# ── llm_generated: 청크 기반 폴백 ─────────────────────────────────

_SENT_SPLIT = re.compile(r"(?<=[.!?。？！\n])\s+")


def _from_chunks(chunks: list[Chunk], size: int) -> list[Probe]:
    """
    청크에서 최소한의 (질문, 정답, gold_chunk_id) 튜플을 만드는 폴백.
    [구현 포인트] RAGAS TestsetGenerator 로 대체 예정.
    """
    probes: list[Probe] = []
    # 너무 짧은 청크는 스킵, 앞에서부터 size 개 사용
    usable = [c for c in chunks if c.text and len(c.text.strip()) >= 20]
    for i, chunk in enumerate(usable[:size]):
        topic = _topic_of(chunk.text)
        probes.append(Probe(
            probe_id=f"probe_gen_{i:03d}",
            question=f"{topic}에 대해 설명해줘.",
            source="llm_generated",
            expected_difficulty="medium",
            answer_exists=True,
            ground_truth=chunk.text.strip(),
            gold_chunk_ids=[chunk.chunk_id],
            qtype=None,                 # 단일홉
        ))
    return probes


def _topic_of(text: str) -> str:
    """청크 첫 문장/앞부분을 질문의 주제 문구로 사용."""
    text = text.strip()
    first = _SENT_SPLIT.split(text, maxsplit=1)[0]
    first = first.strip().strip("#•-*> ").strip()
    # 너무 길면 앞 40자만
    return first[:40] if len(first) > 40 else first
