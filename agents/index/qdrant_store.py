"""
agents/index/qdrant_store.py
Qdrant 공통 모듈 — Index Agent와 MCP 서버 양쪽에서 import해서 사용

[팀원 구현 포인트]
  운영 전환 시 build_client()의 url/api_key만 바꾸면 됨:
    build_client(url="https://xxx.qdrant.io", api_key="...")
"""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION = "agent_doctor"
VECTOR_DIM = 384  # paraphrase-multilingual-MiniLM-L12-v2 기준


# ── 클라이언트 ────────────────────────────────────────────────────

def build_client(url: str = ":memory:", api_key: str | None = None) -> QdrantClient:
    """
    Qdrant 클라이언트 생성.

    url=":memory:"  → 인메모리 (테스트용, 프로세스 종료 시 사라짐)
    url="http://localhost:6333" → 로컬 Docker Qdrant
    url="https://xxx.qdrant.io" → Qdrant Cloud (운영)
    """
    if url == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(url=url, api_key=api_key)


def ensure_collection(client: QdrantClient, vector_dim: int = VECTOR_DIM) -> None:
    """컬렉션이 없으면 생성 (있으면 스킵)."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
        print(f"[Qdrant] 컬렉션 생성: {COLLECTION} (dim={vector_dim})")


# ── 저장 ──────────────────────────────────────────────────────────

def upsert_chunks(client: QdrantClient, chunks: list) -> None:
    """
    Chunk 리스트를 Qdrant에 upsert.
    chunk.embedding이 없으면 스킵.
    """
    points = []
    for i, chunk in enumerate(chunks):
        if not chunk.embedding:
            continue
        points.append(
            PointStruct(
                id=i,
                vector=chunk.embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id":   chunk.doc_id,
                    "text":     chunk.text,
                    "metadata": chunk.metadata,
                },
            )
        )
    if points:
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"[Qdrant] {len(points)}개 청크 저장 완료")


# ── 검색 ──────────────────────────────────────────────────────────

def search(client: QdrantClient, query_vector: list[float], top_k: int = 3) -> list[dict]:
    """벡터 유사도 검색. 결과를 dict 리스트로 반환."""
    # qdrant-client 1.7+ 은 query_points(), 구버전은 search()
    try:
        hits = client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=top_k,
        ).points
    except AttributeError:
        hits = client.search(
            collection_name=COLLECTION,
            query_vector=query_vector,
            limit=top_k,
        )
    return [
        {
            "score":    hit.score,
            "text":     hit.payload.get("text", ""),
            "metadata": hit.payload.get("metadata", {}),
            "chunk_id": hit.payload.get("chunk_id", ""),
        }
        for hit in hits
    ]


# ── 임베딩 ────────────────────────────────────────────────────────

_model = None  # 모델 싱글톤 (첫 호출 시 로드)


def embed(text: str) -> list[float]:
    """
    텍스트 → 벡터. Index Agent와 API 서버 양쪽에서 공유.

    [팀원 구현 포인트] 운영 모델로 교체:
        from openai import OpenAI
        return OpenAI().embeddings.create(
            input=text, model="text-embedding-3-small"
        ).data[0].embedding
    """
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except ImportError:
            _model = "random"  # 폴백 표시용

    if _model == "random":
        import random
        return [random.uniform(-1, 1) for _ in range(VECTOR_DIM)]

    return _model.encode(text, normalize_embeddings=True).tolist()
