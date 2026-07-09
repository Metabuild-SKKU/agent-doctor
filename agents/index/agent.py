"""
agents/index/agent.py
Index Agent — 문서를 청킹 → 임베딩 → Qdrant 저장

읽기: state.documents
쓰기: state.chunks, state.status

[팀원 구현 포인트]
  청킹 전략 교체: _chunk_text() 함수 수정
    - 현재: 고정 크기(character 기준)
    - 개선: 문장/단락 경계 존중, 의미 단위 청킹

  임베딩 모델 교체: qdrant_store.embed() 함수 수정
    - 현재: paraphrase-multilingual-MiniLM-L12-v2 (384차원, 무료)
    - 운영:  text-embedding-3-small (1536차원, OpenAI 유료)
    - 고급:  BGE-M3 (dense+sparse 동시, 한국어 강함)
"""
from __future__ import annotations

import os

from core.schema import Chunk
from core.state import AgentDoctorState
from agents.index.qdrant_store import (
    build_client, ensure_collection, upsert_chunks, embed, VECTOR_DIM
)


# ── 청킹 ──────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    텍스트를 chunk_size(문자 수) 단위로 분할.

    [팀원 구현 포인트] 더 정교한 청킹으로 교체 가능:
      - 문장 경계 분할: kss.split_sentences(text)
      - 단락 기준 분할: text.split("\n\n")
      - 의미 기반 분할: SemanticChunker 등
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        next_start = start + chunk_size - chunk_overlap
        if next_start <= start:
            break
        start = next_start
    return chunks


# ── 메인 ──────────────────────────────────────────────────────────

def run(state: AgentDoctorState) -> AgentDoctorState:
    """
    Index Agent 진입점.

    읽기: state.documents
    쓰기: state.chunks, state.status, state.error
    """
    state.current_agent = "index"
    print(f"[Index] 문서 {len(state.documents)}개 처리 시작")

    if not state.documents:
        state.status = "error"
        state.error = "문서가 없습니다. Ingest Agent 완료 여부를 확인하세요."
        return state

    chunk_size    = state.index_config.get("chunk_size", 512)
    chunk_overlap = state.index_config.get("chunk_overlap", 50)

    # Qdrant 클라이언트 초기화
    qdrant_url = os.getenv("QDRANT_URL", ":memory:")
    qdrant_key = os.getenv("QDRANT_API_KEY", None)
    client = build_client(url=qdrant_url, api_key=qdrant_key)
    ensure_collection(client, vector_dim=VECTOR_DIM)

    all_chunks: list[Chunk] = []

    for doc in state.documents:
        texts = _chunk_text(doc.content, chunk_size, chunk_overlap)
        title = doc.metadata.get("title", doc.doc_id)
        print(f"[Index]  └ '{title}' → {len(texts)}개 청크 생성")

        for i, text in enumerate(texts):
            vector = embed(text)
            chunk = Chunk(
                chunk_id=f"{doc.doc_id}_chunk_{i:03d}",
                doc_id=doc.doc_id,
                text=text,
                embedding=vector,
                metadata={**doc.metadata, "chunk_index": i, "source": doc.source},
            )
            all_chunks.append(chunk)

    # Qdrant에 저장
    upsert_chunks(client, all_chunks)

    state.chunks = all_chunks
    state.status = "indexed"

    print(f"[Index] 완료 — 총 {len(all_chunks)}개 청크 (dim={VECTOR_DIM})")
    return state
