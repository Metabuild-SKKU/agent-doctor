"""
agents/eval/retrieval_temp.py
STEP2: 각 Probe로 검색 + 답변 생성  ── ⚠️ 임시 파일(TEMPORARY) ──

╔══════════════════════════════════════════════════════════════════════╗
║  이 파일은 **임시**다. Index Agent가 아직 "검색 리트리버"를 제공하지     ║
║  않으므로, 그때까지만 Eval이 자체적으로 검색을 구성해 쓰기 위한 것이다.   ║
║                                                                        ║
║  Index Agent가 검색 인터페이스(예: build_retriever/search)를 제공하면    ║
║  → 이 파일을 **통째로 삭제**하고, agent.py 는 Index 쪽 검색을 호출한다.  ║
║  → 답변 생성(generate_answer)만 Eval 로 옮겨 유지하면 된다.             ║
╚══════════════════════════════════════════════════════════════════════╝

임시 구현이지만 임베딩/검색 자체는 규칙대로 공통 모듈
`agents/index/qdrant_store.py` 의 embed()/search() 를 사용한다(직접 모델 로드 금지).
여기서 임시로 떠안는 것은 "청크 적재 + top-k 검색 오케스트레이션" 뿐이다.

폴백 설계 (AGENTS.md): 라이브러리 미설치·검색 실패 시 조용히 대체 경로로.
    - 임베딩 없음/벡터 검색 실패 → 키워드 검색
    - LLM 미설정(OPENAI_API_KEY 없음) → 추출식(top 컨텍스트) 답변
"""
from __future__ import annotations

import os

from core.schema import Chunk
from agents.index.qdrant_store import (
    build_client, ensure_collection, upsert_chunks, embed, search, VECTOR_DIM,
)


# ── 검색 인덱스 준비 (임시) ───────────────────────────────────────

def build_eval_index(chunks: list[Chunk]):
    """
    state.chunks 를 eval 전용 Qdrant 클라이언트에 적재하고 반환. (임시)

    LangGraph 한 프로세스 안에서도 Index Agent 의 in-memory 클라이언트는
    Eval 로 넘어오지 않으므로(별 인스턴스), state.chunks(임베딩 포함)를
    다시 upsert 한다. (serve/api.py 의 init_qdrant 와 같은 패턴)

    임베딩이 하나도 없으면 None 을 반환 → 호출부에서 키워드 검색으로 폴백.
    """
    embedded = [c for c in chunks if c.embedding]
    if not embedded:
        print("[Eval] STEP2(임시): 임베딩 없음 → 키워드 검색 모드")
        return None
    try:
        url = os.getenv("QDRANT_URL", ":memory:")
        key = os.getenv("QDRANT_API_KEY")
        client = build_client(url=url, api_key=key)
        dim = len(embedded[0].embedding) or VECTOR_DIM
        ensure_collection(client, vector_dim=dim)
        upsert_chunks(client, embedded)
        return client
    except Exception as e:  # 폴백: 준비 실패 시 키워드 검색
        print(f"[Eval] STEP2(임시): 검색 인덱스 준비 실패({e}) → 키워드 검색 폴백")
        return None


# ── 검색 (임시) ───────────────────────────────────────────────────

def retrieve(client, chunks: list[Chunk], question: str, top_k: int) -> list[dict]:
    """
    질문으로 상위 top_k 청크 검색. 결과 dict: {score, text, chunk_id, metadata}.
    벡터 검색 우선, 실패하면 키워드 검색으로 폴백. (임시)
    """
    if client is not None:
        try:
            query_vec = embed(question)
            hits = search(client, query_vec, top_k=top_k)
            if hits:
                return hits
        except Exception as e:
            print(f"[Eval] 벡터 검색 실패({e}) → 키워드 검색 폴백")
    return _keyword_search(chunks, question, top_k)


def _keyword_search(chunks: list[Chunk], query: str, top_k: int) -> list[dict]:
    """단어 포함 개수 기반 키워드 검색 (serve/api.py 폴백과 동일 전략)."""
    q = query.lower()
    words = [w for w in q.split() if w]
    scored = []
    for c in chunks:
        text = c.text or ""
        score = sum(1 for w in words if w in text.lower())
        if score > 0:
            scored.append({
                "score": float(score),
                "text": text,
                "chunk_id": c.chunk_id,
                "metadata": c.metadata,
            })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ── 답변 생성 (Index 개발 후에도 유지될 부분) ────────────────────

def generate_answer(question: str, contexts: list[str]) -> str:
    """
    검색된 컨텍스트로 답변 생성.

    [구현 포인트] 실제 RAG 생성기로 교체.
        - LLM(OpenAI 등) 프롬프트: 컨텍스트만 근거로 답하고, 없으면 기권하도록.
        - 응답 모델 ≠ 평가 모델 원칙(설계 §LLM-as-Judge) 유지.
    폴백: OPENAI_API_KEY 없거나 실패하면 top 컨텍스트를 그대로 돌려주는 추출식.
    """
    answer = _llm_generate(question, contexts)
    if answer is not None:
        return answer
    # 폴백: 추출식 (top-1 컨텍스트). 생성 결함이 아니라 골격 동작용임.
    return contexts[0].strip() if contexts else ""


def _llm_generate(question: str, contexts: list[str]) -> str | None:
    """OpenAI 로 답변 생성. 키/라이브러리 없거나 실패하면 None."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    try:
        client = OpenAI()
        context_block = "\n\n".join(f"- {c}" for c in contexts)
        model = os.getenv("EVAL_GEN_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content":
                    "너는 사내 문서 QA 어시스턴트다. 아래 컨텍스트만 근거로 한국어로 "
                    "간결히 답하라. 컨텍스트에 근거가 없으면 '제공된 정보로는 알 수 없습니다'라고 답하라."},
                {"role": "user", "content": f"[컨텍스트]\n{context_block}\n\n[질문]\n{question}"},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[Eval] LLM 생성 실패({e}) → 추출식 폴백")
        return None
