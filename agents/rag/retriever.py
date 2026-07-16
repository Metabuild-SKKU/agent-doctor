"""
Index, Eval, Serve, RAG가 공통으로 쓰는 검색 interface
*검색 방식 선택, fallback 흐름
- Index: chunking/embedding/DB
- Eval: RAG evaluate
- Serve: API/MCP serve search result
- RAG: search result -> answer
"""
from __future__ import annotations

import json
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from agents.index.qdrant_store import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    VECTOR_DIM,
    build_client,
    embed,
    ensure_collection,
    hybrid_search,
    keyword_search,
    rerank,
    search as dense_search,
    upsert_chunks,
)
from core.schema import Chunk

"""
문서 임베딩 시 쓰는 모델 = 질문 임베딩 시 쓰는 모델
최종 점수는 dense 70% + keyword 30%로 합산해서 사용
reranker은 정확도 향상에는 좋지만 느리고 모델 로드 비용 때문에 False
-> index가 만든 DB와 RAG 검색이 같은 조건으로 동작하기 위한 설정값
"""
@dataclass(frozen=True)
class RetrievalSettings:
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimension: int | None = VECTOR_DIM
    top_k: int = 5
    use_hybrid: bool = False
    hybrid_dense_weight: float = 0.7
    use_reranker: bool = False
    reranker_model: str = DEFAULT_RERANKER_MODEL
    qdrant_url: str = ":memory:"
    qdrant_api_key: str | None = None
    recreate_collection_on_dimension_mismatch: bool = False

# true, yes, y, on, 1 -> true로 처리 (bool 정규화)
def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)

# 문자열 -> int 형으로 변환
def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

# chunk metadata를 읽어서 index 설정 복원
def _first_metadata(chunks: list[Chunk | dict]) -> dict:
    if not chunks:
        return {}
    first = chunks[0]
    if isinstance(first, dict):
        return first.get("metadata", {}) or {}
    return first.metadata or {}

# chunk안의 embedding 길이 바탕으로 벡터 차원 추론
def _first_embedding_dim(chunks: list[Chunk | dict]) -> int | None:
    for chunk in chunks:
        embedding = chunk.get("embedding") if isinstance(chunk, dict) else chunk.embedding
        if embedding:
            return len(embedding)
    return None

# 우선순위: 명시적 config > chunk metadata > 기본값/env
# 핵심 설정 정리 함수 -> eval/serve/RAG 어디서 호출하든 같은 설정으로 검색할 수 있도록
def resolve_retrieval_settings(
    chunks: list[Chunk | dict],
    config: dict | None = None,
) -> RetrievalSettings:
    """Merge explicit config, chunk metadata, and env into one retrieval config."""
    config = config or {}
    metadata = _first_metadata(chunks)

    def pick(name: str, default: Any = None) -> Any:
        if name in config:
            return config[name]
        if name in metadata:
            return metadata[name]
        return default

    embedding_dimension = _as_int(
        pick("embedding_dimension", _first_embedding_dim(chunks)),
        _first_embedding_dim(chunks) or VECTOR_DIM,
    )
    top_k = _as_int(pick("top_k", 5), 5) or 5

    return RetrievalSettings(
        embedding_model=str(pick("embedding_model", DEFAULT_EMBEDDING_MODEL)),
        embedding_dimension=embedding_dimension,
        top_k=max(1, top_k),
        use_hybrid=_as_bool(pick("use_hybrid", False)),
        hybrid_dense_weight=float(pick("hybrid_dense_weight", 0.7)),
        use_reranker=_as_bool(pick("use_reranker", False)),
        reranker_model=str(pick("reranker_model", DEFAULT_RERANKER_MODEL)),
        qdrant_url=str(config.get("qdrant_url") or os.getenv("QDRANT_URL", ":memory:")),
        qdrant_api_key=config.get("qdrant_api_key") or os.getenv("QDRANT_API_KEY"),
        recreate_collection_on_dimension_mismatch=_as_bool(
            pick("recreate_collection_on_dimension_mismatch", False)
        ),
    )

# langgraph state -> chunk 객체, chunks.json -> dict (정규화)
def _chunk_to_dict(chunk: Chunk | dict) -> dict:
    if isinstance(chunk, dict):
        return {
            "chunk_id": chunk.get("chunk_id", ""),
            "doc_id": chunk.get("doc_id", ""),
            "text": chunk.get("text", "") or "",
            "page": chunk.get("page"),
            "section": chunk.get("section"),
            "char_span": chunk.get("char_span"),
            "token_count": chunk.get("token_count"),
            "parent_id": chunk.get("parent_id"),
            "hash": chunk.get("hash"),
            "embedding": chunk.get("embedding"),
            "sparse_vector": chunk.get("sparse_vector"),
            "metadata": chunk.get("metadata", {}) or {},
        }
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "text": chunk.text or "",
        "page": chunk.page,
        "section": chunk.section,
        "char_span": chunk.char_span,
        "token_count": chunk.token_count,
        "parent_id": chunk.parent_id,
        "hash": chunk.hash,
        "embedding": chunk.embedding,
        "sparse_vector": chunk.sparse_vector,
        "metadata": chunk.metadata or {},
    }

# dict 형태 chunk -> chunk 객체로 복원
# list가 된 char_span -> tuple로 복원
def _scope_id(chunks: list[dict]) -> str:
    """Return a stable id for the exact chunk set this retriever should search."""
    rows = [
        {
            "chunk_id": chunk.get("chunk_id", ""),
            "doc_id": chunk.get("doc_id", ""),
            "hash": chunk.get("hash"),
            "text": chunk.get("text", ""),
        }
        for chunk in chunks
    ]
    raw = json.dumps(sorted(rows, key=lambda item: item["chunk_id"]), sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _with_scope(chunks: list[dict], scope_id: str) -> list[dict]:
    scoped = []
    for chunk in chunks:
        metadata = dict(chunk.get("metadata", {}) or {})
        metadata["retrieval_scope_id"] = scope_id
        scoped.append({**chunk, "metadata": metadata})
    return scoped


def _chunk_from_dict(data: dict) -> Chunk:
    span = data.get("char_span")
    if isinstance(span, list):
        span = tuple(span)
    return Chunk(
        chunk_id=data.get("chunk_id", ""),
        doc_id=data.get("doc_id", ""),
        text=data.get("text", "") or "",
        page=data.get("page"),
        section=data.get("section"),
        char_span=span,
        token_count=data.get("token_count"),
        parent_id=data.get("parent_id"),
        hash=data.get("hash"),
        embedding=data.get("embedding"),
        sparse_vector=data.get("sparse_vector"),
        metadata=data.get("metadata", {}) or {},
    )


class Retriever:
    """A small, reusable retrieval facade with dense/hybrid/rerank/fallback."""

    def __init__(
        self,
        chunks: list[Chunk | dict],
        settings: RetrievalSettings,
        client: QdrantClient | None = None,
    ) -> None:
        self.settings = settings
        self.chunks = [_chunk_to_dict(chunk) for chunk in chunks]
        self.client = client
        self.chunk_ids = {
            chunk.get("chunk_id", "")
            for chunk in self.chunks
            if chunk.get("chunk_id")
        }
        self.retrieval_scope_id = (
            _first_metadata(self.chunks).get("retrieval_scope_id")
            if self.chunks
            else None
        )

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        return self.search_with_details(query, top_k=top_k)["results"]

    def _vector_candidate_k(self, candidate_k: int) -> int:
        if not self.chunk_ids:
            return candidate_k
        return max(candidate_k, min(max(len(self.chunks), candidate_k * 8), 200))

    def _current_results(self, results: list[dict]) -> list[dict]:
        if not self.chunk_ids:
            return results
        return [
            item
            for item in results
            if item.get("chunk_id") in self.chunk_ids
        ]

    """
    1) 빈 query -> 빈 결과
    2) top_k 결정
    3) reranker 쓰면 후보를 top_k * 4개 가져옴
    4) Qdrant client -> dense/hybrid 검색
    5) 실패 or 결과 X -> keyword fallback
    6) reranker = True면 재정렬
    7) 최종 result 반환
    """
    def search_with_details(self, query: str, top_k: int | None = None) -> dict:
        if not query.strip():
            return {
                "query": query,
                "search_mode": "none",
                "reranked": False,
                "fallback_used": False,
                "results": [],
            }

        requested_top_k = max(1, int(top_k or self.settings.top_k))
        candidate_k = requested_top_k * 4 if self.settings.use_reranker else requested_top_k
        vector_candidate_k = self._vector_candidate_k(candidate_k)
        results: list[dict] = []
        mode = "keyword"
        fallback_used = self.client is None

        if self.client is not None:
            try:
                query_vector = embed(
                    query,
                    model_name=self.settings.embedding_model,
                    vector_dim=self.settings.embedding_dimension,
                )
                if self.settings.use_hybrid:
                    mode = "hybrid"
                    results = hybrid_search(
                        self.client,
                        query_vector=query_vector,
                        query=query,
                        chunks=self.chunks,
                        top_k=vector_candidate_k,
                        dense_weight=self.settings.hybrid_dense_weight,
                        retrieval_scope_id=self.retrieval_scope_id,
                    )
                else:
                    mode = "dense"
                    results = dense_search(
                        self.client,
                        query_vector,
                        top_k=vector_candidate_k,
                        retrieval_scope_id=self.retrieval_scope_id,
                    )
                results = self._current_results(results)
            except Exception as exc:
                print(f"[Retriever] vector search failed, using keyword fallback: {exc}")
                results = []
                fallback_used = True

        if not results:
            mode = "keyword"
            fallback_used = True
            results = keyword_search(self.chunks, query, top_k=candidate_k)

        reranked = False
        if self.settings.use_reranker and results:
            results = rerank(
                query,
                results,
                model_name=self.settings.reranker_model,
                top_k=requested_top_k,
            )
            reranked = True
        else:
            results = results[:requested_top_k]

        return {
            "query": query,
            "search_mode": mode,
            "reranked": reranked,
            "fallback_used": fallback_used,
            "results": results,
        }

# eval의 retrieval_temp.py 대체하는 부분
def build_retriever(
    chunks: list[Chunk | dict],
    config: dict | None = None,
    client: QdrantClient | None = None,
) -> Retriever:
    """Build a retriever from indexed chunks.

    If chunks contain embeddings, they are upserted into Qdrant. If embeddings
    are missing and no client is provided, the returned retriever still works
    via keyword_search().
    """
    settings = resolve_retrieval_settings(chunks, config)
    raw_chunks = [_chunk_to_dict(chunk) for chunk in chunks]
    embedded = [_chunk_from_dict(chunk) for chunk in raw_chunks if chunk.get("embedding")]
    if embedded:
        raw_chunks = _with_scope(raw_chunks, _scope_id(raw_chunks))
        embedded = [_chunk_from_dict(chunk) for chunk in raw_chunks if chunk.get("embedding")]

    if embedded:
        try:
            client = client or build_client(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
            vector_dim = len(embedded[0].embedding or []) or settings.embedding_dimension or VECTOR_DIM
            ensure_collection(
                client,
                vector_dim=vector_dim,
                recreate_on_mismatch=settings.recreate_collection_on_dimension_mismatch,
            )
            upsert_chunks(client, embedded)
        except Exception as exc:
            print(f"[Retriever] Qdrant setup failed, using keyword fallback: {exc}")
            client = None
    return Retriever(raw_chunks, settings, client=client)

# serve 쪽에서 chunk.json 읽을 때 helper
def load_chunks(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
