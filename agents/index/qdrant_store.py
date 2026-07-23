# Index와 Serve가 같이 쓰는 저장/검색 유틸.
# 저장할 때와 검색할 때 embedding model/dim이 같아야 한다.
from __future__ import annotations

import hashlib
import math
import os
import re
import uuid
from collections import Counter
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

try:  # qdrant-client<1.15에는 native hybrid query 모델이 없을 수 있다.
    from qdrant_client.models import Prefetch, Rrf, RrfQuery
except ImportError:  # pragma: no cover - installed client version dependent
    Prefetch = Rrf = RrfQuery = None

COLLECTION = "agent_doctor"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
VECTOR_DIM = 1024

_models: dict[str, Any] = {}
_failed_models: set[str] = set()
_rerankers: dict[str, Any] = {}
_failed_rerankers: set[str] = set()
_collection_native_hybrid_cache: dict[int, bool] = {}


# 테스트에서는 in-memory, 운영에서는 실제 Qdrant endpoint로 붙는다.
def build_client(url: str = ":memory:", api_key: str | None = None) -> QdrantClient:
    if url == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(url=url, api_key=api_key)


# Qdrant client 버전별 응답 차이를 여기서만 흡수한다.
def _collection_vector_size(client: QdrantClient) -> int | None:
    try:
        vectors = client.get_collection(COLLECTION).config.params.vectors
        if isinstance(vectors, dict):
            dense = vectors.get(DENSE_VECTOR_NAME)
            if dense is not None and hasattr(dense, "size"):
                return int(dense.size)
            return None
        if hasattr(vectors, "size"):
            return int(vectors.size)
    except Exception:
        return None
    return None


def _collection_has_native_hybrid(client: QdrantClient) -> bool:
    cache_key = id(client)
    if cache_key in _collection_native_hybrid_cache:
        return _collection_native_hybrid_cache[cache_key]
    try:
        params = client.get_collection(COLLECTION).config.params
        vectors = params.vectors
        sparse_vectors = getattr(params, "sparse_vectors", None) or {}
        has_native = (
            isinstance(vectors, dict)
            and DENSE_VECTOR_NAME in vectors
            and SPARSE_VECTOR_NAME in sparse_vectors
        )
    except Exception:
        _collection_native_hybrid_cache[cache_key] = False
        return False
    _collection_native_hybrid_cache[cache_key] = has_native
    return has_native


def _clear_collection_shape_cache(client: QdrantClient) -> None:
    _collection_native_hybrid_cache.pop(id(client), None)


def _query_filter(retrieval_scope_id: str | None) -> Filter | None:
    if not retrieval_scope_id:
        return None
    return Filter(
        must=[
            FieldCondition(
                key="retrieval_scope_id",
                match=MatchValue(value=retrieval_scope_id),
            )
        ]
    )


def _sparse_vector(data: dict | None) -> SparseVector | None:
    if not data:
        return None
    indices = data.get("indices") or []
    values = data.get("values") or []
    if not indices or not values or len(indices) != len(values):
        return None
    return SparseVector(
        indices=[int(index) for index in indices],
        values=[float(value) for value in values],
    )


def _hit_to_result(hit) -> dict:
    payload = hit.payload or {}
    return {
        "score": float(hit.score),
        "text": payload.get("text", ""),
        "metadata": payload.get("metadata", {}),
        "chunk_id": payload.get("chunk_id", ""),
        "doc_id": payload.get("doc_id", ""),
        "section": payload.get("section"),
        "char_span": payload.get("char_span"),
        "token_count": payload.get("token_count"),
        "parent_id": payload.get("parent_id"),
        "hash": payload.get("hash"),
        "retrieval_scope_id": payload.get("retrieval_scope_id"),
    }


def ensure_collection(
    client: QdrantClient,
    vector_dim: int = VECTOR_DIM,
    recreate_on_mismatch: bool = False,
) -> None:
    # 차원이 다른 컬렉션에 그대로 덮어쓰면 검색이 깨져서, 명시 옵션 없이는 막는다.
    existing = [collection.name for collection in client.get_collections().collections]
    if COLLECTION in existing:
        current_dim = _collection_vector_size(client)
        if current_dim is None or current_dim == vector_dim:
            if _collection_has_native_hybrid(client):
                return
            if recreate_on_mismatch:
                client.delete_collection(collection_name=COLLECTION)
                _clear_collection_shape_cache(client)
            else:
                print(
                    f"[Qdrant] legacy dense-only 컬렉션 사용: {COLLECTION} "
                    "(native hybrid를 쓰려면 컬렉션 재생성이 필요합니다)"
                )
                return
        else:
            if not recreate_on_mismatch:
                raise ValueError(
                    f"Qdrant 벡터 차원이 다릅니다: 기존={current_dim}, 요청={vector_dim}. "
                    "recreate_collection_on_dimension_mismatch를 켜거나 새 컬렉션을 사용하세요."
                )
            client.delete_collection(collection_name=COLLECTION)
            _clear_collection_shape_cache(client)
        if COLLECTION in [collection.name for collection in client.get_collections().collections]:
            return

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(size=vector_dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(),
        },
    )
    _clear_collection_shape_cache(client)
    print(f"[Qdrant] 컬렉션 준비: {COLLECTION} (dim={vector_dim})")


# 같은 chunk_id는 같은 point id를 쓰게 해서 재색인을 안전하게 만든다.
def _point_id(chunk_id: str, retrieval_scope_id: str | None = None) -> str:
    scope = retrieval_scope_id or "global"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"agent-doctor:{scope}:{chunk_id}"))


# Serve가 payload를 그대로 읽으므로 provenance 필드는 빼지 않는다.
def upsert_chunks(client: QdrantClient, chunks: list) -> None:
    use_native_hybrid = _collection_has_native_hybrid(client)
    points = []
    for chunk in chunks:
        if not chunk.embedding:
            continue
        metadata = chunk.metadata or {}
        retrieval_scope_id = metadata.get("retrieval_scope_id")
        sparse = _sparse_vector(chunk.sparse_vector)
        if use_native_hybrid:
            vector = {DENSE_VECTOR_NAME: chunk.embedding}
        else:
            vector = chunk.embedding
        if use_native_hybrid and sparse is not None:
            vector[SPARSE_VECTOR_NAME] = sparse
        payload = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "text": chunk.text,
            "section": chunk.section,
            "char_span": chunk.char_span,
            "token_count": chunk.token_count,
            "parent_id": chunk.parent_id,
            "hash": chunk.hash,
            "metadata": metadata,
            "sparse_vector": chunk.sparse_vector,
            "retrieval_scope_id": retrieval_scope_id,
        }
        points.append(
            PointStruct(
                id=_point_id(chunk.chunk_id, retrieval_scope_id),
                vector=vector,
                payload=payload,
            )
        )
    if points:
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"[Qdrant] {len(points)}개 청크 저장 완료")


# 재색인 대상 문서의 옛 chunk가 검색에 섞이지 않도록 먼저 지운다.
def delete_document_chunks(
    client: QdrantClient,
    doc_ids: list[str],
    retrieval_scope_id: str | None = None,
) -> None:
    unique_ids = sorted({doc_id for doc_id in doc_ids if doc_id})
    if not unique_ids:
        return
    try:
        must = [
            FieldCondition(
                key="doc_id",
                match=MatchAny(any=unique_ids),
            )
        ]
        if retrieval_scope_id:
            must.append(
                FieldCondition(
                    key="retrieval_scope_id",
                    match=MatchValue(value=retrieval_scope_id),
                )
            )
        client.delete(
            collection_name=COLLECTION,
            points_selector=Filter(must=must),
        )
    except Exception:
        # MatchAny를 지원하지 않는 구버전에서는 문서별로 삭제한다.
        for doc_id in unique_ids:
            must = [
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id),
                )
            ]
            if retrieval_scope_id:
                must.append(
                    FieldCondition(
                        key="retrieval_scope_id",
                        match=MatchValue(value=retrieval_scope_id),
                    )
                )
            client.delete(
                collection_name=COLLECTION,
                points_selector=Filter(must=must),
            )


def search(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int = 5,
    retrieval_scope_id: str | None = None,
) -> list[dict]:
    query_filter = _query_filter(retrieval_scope_id)
    # dense 검색 결과를 Serve가 쓰는 공통 dict 모양으로 맞춘다.
    native_hybrid_collection = _collection_has_native_hybrid(client)
    try:
        if native_hybrid_collection and hasattr(client, "query_points"):
            kwargs = {
                "collection_name": COLLECTION,
                "query": query_vector,
                "using": DENSE_VECTOR_NAME,
                "limit": top_k,
            }
            if query_filter is not None:
                kwargs["query_filter"] = query_filter
            hits = client.query_points(**kwargs).points
        else:
            search_kwargs = {
                "collection_name": COLLECTION,
                "query_vector": query_vector,
                "limit": top_k,
            }
            if query_filter is not None:
                search_kwargs["query_filter"] = query_filter
            hits = client.search(**search_kwargs)
    except Exception:
        if not hasattr(client, "query_points"):
            raise
        kwargs = {
            "collection_name": COLLECTION,
            "query": query_vector,
            "limit": top_k,
        }
        if query_filter is not None:
            kwargs["query_filter"] = query_filter
        hits = client.query_points(**kwargs).points
    return [_hit_to_result(hit) for hit in hits]


# 형태소 분석기 없이도 테스트/하이브리드 검색이 돌아가게 가볍게 쪼갠다.
def _tokens(text: str) -> list[str]:
    return re.findall(r"[가-힣]+|[A-Za-z][A-Za-z0-9_+.-]*|\d+", text.lower())


# 나중에 Qdrant sparse vector로 옮기기 쉽게 indices/values 형태로 맞춰 둔다.
def build_sparse_vector(text: str, dimensions: int = 2**20) -> dict:
    counts = Counter(_tokens(text))
    values: dict[int, float] = {}
    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "big") % dimensions
        values[index] = values.get(index, 0.0) + 1.0 + math.log(count)
    norm = math.sqrt(sum(value * value for value in values.values())) or 1.0
    ordered = sorted(values.items())
    return {
        "indices": [index for index, _ in ordered],
        "values": [value / norm for _, value in ordered],
    }


# BM25 서버가 없어도 비교 가능한 lexical 점수를 만든다.
def _keyword_score(query: str, text: str) -> float:
    query_terms = Counter(_tokens(query))
    text_terms = Counter(_tokens(text))
    if not query_terms or not text_terms:
        return 0.0
    matched = 0
    for term, count in query_terms.items():
        hits = text_terms.get(term, 0)
        if not re.fullmatch(r"[a-z0-9_+.-]+", term):
            hits += sum(
                text_count
                for token, text_count in text_terms.items()
                if token != term and term in token
            )
        matched += min(count, hits)
    return matched / max(1, sum(query_terms.values()))


def _field(chunk: Any, name: str, default: Any = None) -> Any:
    if isinstance(chunk, dict):
        return chunk.get(name, default)
    return getattr(chunk, name, default)


def keyword_search(
    chunks: list[Any],
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Dependency-light lexical fallback used by Eval, Serve, and RAG."""
    if top_k <= 0 or not query.strip():
        return []

    scored = []
    for chunk in chunks:
        text = _field(chunk, "text", "") or ""
        score = _keyword_score(query, text)
        if score <= 0:
            continue
        scored.append(
            {
                "score": float(score),
                "text": text,
                "metadata": _field(chunk, "metadata", {}) or {},
                "chunk_id": _field(chunk, "chunk_id", "") or "",
                "doc_id": _field(chunk, "doc_id", "") or "",
                "section": _field(chunk, "section"),
                "char_span": _field(chunk, "char_span"),
                "token_count": _field(chunk, "token_count"),
                "parent_id": _field(chunk, "parent_id"),
                "hash": _field(chunk, "hash"),
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def hybrid_search(
    client: QdrantClient,
    query_vector: list[float],
    query: str,
    chunks: list[dict],
    top_k: int = 5,
    dense_weight: float = 0.7,
    retrieval_scope_id: str | None = None,
) -> list[dict]:
    native_results = _native_hybrid_search(
        client,
        query_vector=query_vector,
        query=query,
        top_k=top_k,
        dense_weight=dense_weight,
        retrieval_scope_id=retrieval_scope_id,
    )
    if native_results is not None:
        return native_results

    # dense 점수와 lexical 점수를 chunk_id 기준으로 합친다.
    dense_weight = min(1.0, max(0.0, float(dense_weight)))
    dense_results = search(
        client,
        query_vector,
        top_k=max(top_k * 4, 20),
        retrieval_scope_id=retrieval_scope_id,
    )
    dense_by_id = {item["chunk_id"]: item for item in dense_results}

    lexical_by_id: dict[str, tuple[float, dict]] = {}
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "")
        score = _keyword_score(query, chunk.get("text", ""))
        if score > 0:
            lexical_by_id[chunk_id] = (score, chunk)

    max_dense = max((max(0.0, item["score"]) for item in dense_results), default=1.0) or 1.0
    candidates = set(dense_by_id) | set(lexical_by_id)
    fused = []
    for chunk_id in candidates:
        dense_item = dense_by_id.get(chunk_id)
        lexical_score, raw = lexical_by_id.get(chunk_id, (0.0, {}))
        dense_score = max(0.0, dense_item["score"]) / max_dense if dense_item else 0.0
        score = dense_weight * dense_score + (1.0 - dense_weight) * lexical_score
        base = dense_item or {
            "chunk_id": chunk_id,
            "doc_id": raw.get("doc_id", ""),
            "text": raw.get("text", ""),
            "section": raw.get("section"),
            "metadata": raw.get("metadata", {}),
        }
        fused.append({**base, "score": float(score)})

    fused.sort(key=lambda item: item["score"], reverse=True)
    return fused[:top_k]


def _native_hybrid_search(
    client: QdrantClient,
    *,
    query_vector: list[float],
    query: str,
    top_k: int,
    dense_weight: float,
    retrieval_scope_id: str | None,
) -> list[dict] | None:
    if Prefetch is None or Rrf is None or RrfQuery is None:
        return None
    sparse = _sparse_vector(build_sparse_vector(query))
    if sparse is None or not _collection_has_native_hybrid(client):
        return None

    dense_weight = min(1.0, max(0.0, float(dense_weight)))
    sparse_weight = 1.0 - dense_weight
    candidate_k = max(top_k * 4, 20)
    query_filter = _query_filter(retrieval_scope_id)
    try:
        hits = client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                Prefetch(
                    query=sparse,
                    using=SPARSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=candidate_k,
                ),
                Prefetch(
                    query=query_vector,
                    using=DENSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=candidate_k,
                ),
            ],
            query=RrfQuery(rrf=Rrf(weights=[sparse_weight, dense_weight])),
            query_filter=query_filter,
            limit=top_k,
        ).points
    except Exception as exc:
        print(f"[Qdrant] native hybrid search 실패, local fusion 사용: {exc}")
        return None

    return [_hit_to_result(hit) for hit in hits]


def rerank(
    query: str,
    results: list[dict],
    model_name: str = DEFAULT_RERANKER_MODEL,
    top_k: int = 5,
) -> list[dict]:
    # reranker가 있으면 한 번 더 정렬하고, 없으면 기존 순서를 유지한다.
    if not results:
        return []
    if model_name not in _rerankers and model_name not in _failed_rerankers:
        try:
            from sentence_transformers import CrossEncoder

            _rerankers[model_name] = CrossEncoder(model_name)
        except Exception as exc:
            _failed_rerankers.add(model_name)
            print(f"[Index] reranker 로드 실패, 기존 순위 유지: {exc}")

    model = _rerankers.get(model_name)
    if model is None:
        return results[:top_k]

    scores = model.predict([(query, item.get("text", "")) for item in results])
    reranked = [
        {**item, "retrieval_score": item.get("score", 0.0), "score": float(score)}
        for item, score in zip(results, scores)
    ]
    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


# 모델 가중치를 못 받는 환경에서도 테스트가 흔들리지 않게 결정적으로 만든다.
def _fallback_embedding(text: str, vector_dim: int) -> list[float]:
    vector = [0.0] * vector_dim
    features = _tokens(text)
    features.extend(text[index : index + 3] for index in range(max(0, len(text) - 2)))
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
        index = int.from_bytes(digest[:8], "big") % vector_dim
        sign = 1.0 if digest[8] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _get_embedding_model(model_name: str) -> Any | None:
    # 프로세스 내 1회 로드(전역 캐시). 실패 시 None → 호출부가 fallback 사용.
    if model_name not in _models and model_name not in _failed_models:
        try:
            from sentence_transformers import SentenceTransformer

            _models[model_name] = SentenceTransformer(model_name)
        except Exception as exc:
            _failed_models.add(model_name)
            print(f"[Index] 임베딩 모델 로드 실패, deterministic fallback 사용: {exc}")
    return _models.get(model_name)


def embed(
    text: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    vector_dim: int | None = None,
) -> list[float]:
    # sentence-transformers가 있으면 실제 모델, 없으면 fallback을 쓴다.
    dimension = int(vector_dim or VECTOR_DIM)
    model = _get_embedding_model(model_name)
    if model is None:
        return _fallback_embedding(text, dimension)
    return model.encode(text, normalize_embeddings=True).tolist()


def embed_batch(
    texts: list[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    vector_dim: int | None = None,
    batch_size: int | None = None,
) -> list[list[float]]:
    """텍스트 리스트를 배치 인코딩한다(색인용). 결과는 입력 순서 그대로.

    batch_size: None → env INDEX_EMBED_BATCH(기본 32). 1이면 텍스트별 단건
    encode 루프로 폴백해 기존 embed() 결과와 완전히 동일하게 만든다(kill-switch).
    질의 임베딩은 계속 단건 embed()를 쓴다 — 저장 벡터만 같은 방식으로 일괄
    전환되므로 배치 인코딩의 float 미세차가 검색 순위 비교를 왜곡하지 않는다.
    """
    if not texts:
        return []
    dimension = int(vector_dim or VECTOR_DIM)
    if batch_size is None:
        try:
            batch_size = int(os.getenv("INDEX_EMBED_BATCH", "32"))
        except ValueError:
            batch_size = 32
    model = _get_embedding_model(model_name)
    if model is None:
        return [_fallback_embedding(text, dimension) for text in texts]
    if batch_size <= 1:
        return [model.encode(text, normalize_embeddings=True).tolist() for text in texts]
    return [
        vector.tolist()
        for vector in model.encode(
            texts, normalize_embeddings=True, batch_size=batch_size
        )
    ]


def count_tokens(
    text: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> int:
    # tokenizer가 있으면 그걸 쓰고, 없으면 대략적인 토큰 수로 기록한다.
    model = _models.get(model_name)
    tokenizer = getattr(model, "tokenizer", None) if model is not None else None
    if tokenizer is None:
        return max(1, len(_tokens(text)))
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return max(1, len(_tokens(text)))
