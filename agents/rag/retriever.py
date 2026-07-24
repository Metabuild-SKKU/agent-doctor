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
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from agents.index.qdrant_store import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    VECTOR_DIM,
    build_client,
    delete_document_chunks,
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

def _populate(
    raw_chunks: list[dict],
    scope_id: str,
    settings: RetrievalSettings,
    client: QdrantClient | None,
    delete_doc_ids: list[str] | None,
) -> tuple[list[dict], QdrantClient | None, bool]:
    """청크를 Qdrant 에 적재하고 (스코프가 찍힌 청크, 클라이언트, 캐시 가능 여부) 를 돌려준다.

    임베딩이 없거나 적재에 실패하면 client=None 으로 떨어지고, 호출부는
    keyword_search() 폴백으로 계속 동작한다. 이 함수가 Qdrant 쓰기의 유일한 지점이다.

    세 번째 값은 이 결과를 캐시해도 되는지다. client=None 이 나오는 경우가 둘인데
    성격이 다르다 — 임베딩이 아예 없으면 그게 정상 상태(keyword 전용)라 캐시해도 되지만,
    적재에 실패해서 None 이면 다음 호출에서 다시 시도해야 한다. 실패를 캐시하면 그 프로세스
    전체가 재시도 없이 keyword 로 굳는다.
    """
    embedded = [_chunk_from_dict(chunk) for chunk in raw_chunks if chunk.get("embedding")]
    if embedded:
        raw_chunks = _with_scope(raw_chunks, scope_id)
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
            if delete_doc_ids:
                delete_document_chunks(
                    client,
                    list(delete_doc_ids),
                    retrieval_scope_id=scope_id,
                )
            upsert_chunks(client, embedded)
        except Exception as exc:
            print(f"[Retriever] Qdrant setup failed, using keyword fallback: {exc}")
            return raw_chunks, None, False
    return raw_chunks, client, True


def build_retriever(
    chunks: list[Chunk | dict],
    config: dict | None = None,
    client: QdrantClient | None = None,
    delete_doc_ids: list[str] | None = None,
) -> Retriever:
    """Build a retriever from indexed chunks (항상 새로 적재한다).

    If chunks contain embeddings, they are upserted into Qdrant. If embeddings
    are missing and no client is provided, the returned retriever still works
    via keyword_search().

    같은 프로세스에서 같은 청크를 여러 번 검색한다면 get_retriever() 를 쓸 것 —
    이 함수는 호출할 때마다 컬렉션 준비와 upsert 를 다시 한다.
    """
    settings = resolve_retrieval_settings(chunks, config)
    raw_chunks = [_chunk_to_dict(chunk) for chunk in chunks]
    raw_chunks, client, _ = _populate(
        raw_chunks, _scope_id(raw_chunks), settings, client, delete_doc_ids
    )
    return Retriever(raw_chunks, settings, client=client)


# ── 적재 캐시 ────────────────────────────────────────────────────
# Index → Eval → (Optimize → Index → Eval)* 로 도는 동안 같은 청크 집합을 매번
# 다시 upsert 하던 문제를 없앤다.

_cache_lock = threading.Lock()
_cached_key: tuple | None = None
_cached_payload: tuple[list[dict], QdrantClient | None] | None = None


def _population_key(
    raw_chunks: list[dict], scope_id: str, settings: RetrievalSettings
) -> tuple:
    """적재 결과를 좌우하는 값만 키에 넣는다.

    청크 집합(scope_id)과 임베딩 정체성, 저장소 좌표만 보고, top_k 처럼 검색 시점에만
    쓰는 설정은 넣지 않는다. Index 와 Eval 이 서로 다른 config dict(전자는 기본값이
    병합된 전체 config, 후자는 state.index_config)를 넘겨도 같은 키로 모이게 하려는 것.

    embedding_model 이 키에 있어야 하는 이유: scope_id 는 chunk_id/doc_id/hash/text 만
    해싱하는데 hash 는 sha256(text) 라 임베딩과 무관하다. Optimize 의
    swap_embedding_model 처방(reindex=True)으로 같은 텍스트를 다른 모델로 재임베딩하면
    scope_id 가 그대로고, 새 모델 차원까지 같으면 키가 완전히 일치해 upsert 를 건너뛴다.
    그러면 질의만 새 모델로 임베딩되고 저장된 벡터는 옛 모델 것이라 유사도가 무의미해진다.

    남는 구멍: 모델명이 같은데 벡터만 다른 경우(수동 재임베딩·모델 버전 변경)는 여전히
    충돌한다. 임베딩 전체를 해싱하면 막을 수 있지만 청크 수에 비례해 매 호출 비용이 든다 —
    그런 상황에서는 reset_retriever_cache() 로 비운다.

    recreate_collection_on_dimension_mismatch 는 일부러 넣지 않는다. 이 플래그는 적재
    "결과"가 아니라 mismatch 를 만났을 때의 처리 방식을 정할 뿐이고, Index 가 이 플래그를
    one-shot 으로 소비해 끄기 때문에(index/agent.py) 키에 넣으면 Index(True)와
    Eval(False)이 갈려 같은 청크를 두 번 적재하게 된다. 차원 자체는 이미 키에 있어
    진짜 차원 변경은 어차피 새 키가 되고, 실패한 적재는 캐시되지 않으므로 플래그를 켠
    재시도는 언제나 새로 적재된다.
    """
    return (
        scope_id,
        settings.embedding_model,
        _first_embedding_dim(raw_chunks) or settings.embedding_dimension,
        settings.qdrant_url,
        settings.qdrant_api_key,
    )


def get_retriever(
    chunks: list[Chunk | dict],
    config: dict | None = None,
    client: QdrantClient | None = None,
    delete_doc_ids: list[str] | None = None,
) -> Retriever:
    """build_retriever 의 캐시 버전 — 같은 청크 집합이면 Qdrant 적재를 건너뛴다.

    Index·Eval·Serve 가 모두 이것을 호출하면 파이프라인 1회 실행에서 컬렉션 준비와
    upsert 는 정확히 한 번만 일어난다. 청크나 저장소 좌표가 바뀌면(재색인 등)
    키가 달라져 자동으로 다시 적재한다.

    주의: 캐시가 맞으면 delete_doc_ids 도 건너뛴다. 청크 집합이 같다는 것은
    지우고 다시 넣어도 결과가 같다는 뜻이라 안전하다.

    적재에 실패한 결과는 캐시하지 않는다 — 다음 호출에서 다시 시도한다. 이때 기존 캐시를
    비우지는 않는다. 슬롯이 하나뿐이라도 방금 실패한 키와 앞서 성공한 키는 서로 다른 청크
    집합이고, 그 키로 다시 들어오면 옛 항목은 여전히 유효하다.
    """
    global _cached_key, _cached_payload

    settings = resolve_retrieval_settings(chunks, config)
    raw_chunks = [_chunk_to_dict(chunk) for chunk in chunks]
    scope_id = _scope_id(raw_chunks)
    key = _population_key(raw_chunks, scope_id, settings)

    with _cache_lock:
        if _cached_key == key and _cached_payload is not None:
            raw_chunks, cached_client = _cached_payload
        else:
            raw_chunks, cached_client, cacheable = _populate(
                raw_chunks, scope_id, settings, client, delete_doc_ids
            )
            if cacheable:
                _cached_key = key
                _cached_payload = (raw_chunks, cached_client)

    return Retriever(raw_chunks, settings, client=cached_client)


def reset_retriever_cache() -> None:
    """적재 캐시를 비운다(테스트·장기 실행 프로세스에서 인덱스를 강제로 다시 만들 때)."""
    global _cached_key, _cached_payload
    with _cache_lock:
        _cached_key = None
        _cached_payload = None

# serve 쪽에서 chunk.json 읽을 때 helper
def load_chunks(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
