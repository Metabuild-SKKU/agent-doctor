"""
agents/index/agent.py
Index Agent — 문서를 청킹하고 임베딩을 생성해 state.chunks에 저장

읽기: state.documents
쓰기: state.chunks, state.status

[팀원 구현 포인트]
  청킹 전략 교체: _chunk_text() 함수 수정
    - 현재: 고정 크기(character 기준)
    - 개선: 문장/단락 경계 존중, 의미 단위 청킹

  임베딩 모델 교체: _get_model() 함수 수정
    - 현재: paraphrase-multilingual-MiniLM-L12-v2 (384차원, 무료)
    - 운영:  text-embedding-3-small (1536차원, OpenAI 유료)
    - 고급:  BGE-M3 (dense+sparse 동시, 한국어 강함)
"""
from __future__ import annotations

from core.schema import Chunk
from core.state import AgentDoctorState

# ── 임베딩 모델 ────────────────────────────────────────────────────

try:
    from sentence_transformers import SentenceTransformer
    _MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # 한국어 지원, 384차원
    _model: SentenceTransformer | None = None

    def _get_model() -> SentenceTransformer:
        global _model
        if _model is None:
            print(f"[Index] 모델 로드 중: {_MODEL_NAME} (첫 실행 시 자동 다운로드)")
            _model = SentenceTransformer(_MODEL_NAME)
            print(f"[Index] 모델 로드 완료 (차원: {_model.get_embedding_dimension()})")
        return _model

    def _embed(text: str) -> list[float]:
        return _get_model().encode(text, normalize_embeddings=True).tolist()

except ImportError:
    print("[Index] sentence-transformers 미설치 → pip install sentence-transformers")

    def _embed(text: str) -> list[float]:  # type: ignore
        """설치 전 임시: 랜덤 벡터 (파이프라인 테스트용)"""
        import random
        return [random.uniform(-1, 1) for _ in range(384)]


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
        if next_start <= start:   # 무한루프 방지
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

    all_chunks: list[Chunk] = []

    for doc in state.documents:
        texts = _chunk_text(doc.content, chunk_size, chunk_overlap)
        title = doc.metadata.get("title", doc.doc_id)
        print(f"[Index]  └ '{title}' → {len(texts)}개 청크 생성")

        for i, text in enumerate(texts):
            embedding = _embed(text)
            chunk = Chunk(
                chunk_id=f"{doc.doc_id}_chunk_{i:03d}",
                doc_id=doc.doc_id,
                text=text,
                embedding=embedding,
                metadata={**doc.metadata, "chunk_index": i, "source": doc.source},
            )
            all_chunks.append(chunk)

    state.chunks = all_chunks
    state.status = "indexed"

    print(f"[Index] 완료 — 총 {len(all_chunks)}개 청크 (임베딩 차원: {len(all_chunks[0].embedding) if all_chunks else 0})")
    return state
