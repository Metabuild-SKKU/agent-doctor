# Index와 Serve가 같이 쓰는 저장/검색 유틸.
# 저장할 때와 검색할 때 embedding model/dim이 같아야 한다.
from __future__ import annotations

import hashlib
import math
import os
import re
import time
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
    VectorParams,
)

COLLECTION = "agent_doctor"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
VECTOR_DIM = 1024

def _env_float(name: str, default: float) -> float:
    """환경변수 float 파싱 — 비정수/오타면 기본값으로 폴백. import 시점 크래시 방지
    (오타 하나로 모듈 전체 import 가 실패하지 않도록)."""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


_models: dict[str, Any] = {}
# model_name → 마지막 로드 실패 시각(monotonic). 실패를 영구 캐시하면 일시적
# 원인(네트워크 등) 후에도 프로세스 내내 fallback 임베딩만 조용히 쓰게 되므로,
# 쿨다운이 지나면 재시도한다.
_failed_models: dict[str, float] = {}
_FAILED_MODEL_RETRY_SEC = _env_float("INDEX_EMBED_MODEL_RETRY_SEC", 300.0)
# reranker_model → 마지막 로드 실패 시각(monotonic). embedding 모델과 같은 쿨다운
# 정책을 따른다 — 영구 캐시하면 일시적 실패 후에도 프로세스 내내 리랭킹이 죽는다.
_rerankers: dict[str, Any] = {}
_failed_rerankers: dict[str, float] = {}
_FAILED_RERANKER_RETRY_SEC = _env_float("INDEX_RERANKER_RETRY_SEC", 300.0)


# 테스트에서는 in-memory, 운영에서는 실제 Qdrant endpoint로 붙는다.
def build_client(url: str = ":memory:", api_key: str | None = None) -> QdrantClient:
    if url == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(url=url, api_key=api_key)


# Qdrant client 버전별 응답 차이를 여기서만 흡수한다.
def _collection_vector_size(
    client: QdrantClient,
    collection_name: str = COLLECTION,
) -> int | None:
    try:
        vectors = client.get_collection(collection_name).config.params.vectors
        if hasattr(vectors, "size"):
            return int(vectors.size)
    except Exception:
        return None
    return None


def ensure_collection(
    client: QdrantClient,
    vector_dim: int = VECTOR_DIM,
    recreate_on_mismatch: bool = False,
    collection_name: str = COLLECTION,
) -> None:
    # 차원이 다른 컬렉션에 그대로 덮어쓰면 검색이 깨져서, 명시 옵션 없이는 막는다.
    existing = [collection.name for collection in client.get_collections().collections]
    if collection_name in existing:
        current_dim = _collection_vector_size(client, collection_name)
        if current_dim is None or current_dim == vector_dim:
            return
        if not recreate_on_mismatch:
            raise ValueError(
                f"Qdrant 벡터 차원이 다릅니다: 기존={current_dim}, 요청={vector_dim}. "
                "recreate_collection_on_dimension_mismatch를 켜거나 새 컬렉션을 사용하세요."
            )
        client.delete_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )
    print(f"[Qdrant] 컬렉션 준비: {collection_name} (dim={vector_dim})")


# 같은 chunk_id는 같은 point id를 쓰게 해서 재색인을 안전하게 만든다.
def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"agent-doctor:{chunk_id}"))


# Serve가 payload를 그대로 읽으므로 provenance 필드는 빼지 않는다.
def upsert_chunks(
    client: QdrantClient,
    chunks: list,
    collection_name: str = COLLECTION,
) -> None:
    points = []
    for chunk in chunks:
        if not chunk.embedding:
            continue
        metadata = chunk.metadata or {}
        points.append(
            PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=chunk.embedding,
                payload={
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
                    "retrieval_scope_id": metadata.get("retrieval_scope_id"),
                    "index_cache_key": metadata.get("index_cache_key"),
                },
            )
        )
    if points:
        client.upsert(collection_name=collection_name, points=points)
        print(f"[Qdrant] {len(points)}개 청크 저장 완료")


def collection_index_cache_key(
    client: QdrantClient,
    collection_name: str = COLLECTION,
) -> str | None:
    """컬렉션 payload에 기록한 논리 인덱스 키를 한 건 읽는다."""
    try:
        points, _ = client.scroll(
            collection_name=collection_name,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return None
        payload = points[0].payload or {}
        key = payload.get("index_cache_key")
        if not key:
            key = (payload.get("metadata") or {}).get("index_cache_key")
        return str(key) if key else None
    except Exception:
        return None


# 재색인 대상 문서의 옛 chunk가 검색에 섞이지 않도록 먼저 지운다.
def delete_document_chunks(
    client: QdrantClient,
    doc_ids: list[str],
    collection_name: str = COLLECTION,
) -> None:
    unique_ids = sorted({doc_id for doc_id in doc_ids if doc_id})
    if not unique_ids:
        return
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchAny(any=unique_ids),
                    )
                ]
            ),
        )
    except Exception:
        # MatchAny를 지원하지 않는 구버전에서는 문서별로 삭제한다.
        for doc_id in unique_ids:
            client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id),
                        )
                    ]
                ),
            )


def search(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int = 5,
    retrieval_scope_id: str | None = None,
    collection_name: str = COLLECTION,
) -> list[dict]:
    query_filter = None
    if retrieval_scope_id:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="retrieval_scope_id",
                    match=MatchValue(value=retrieval_scope_id),
                )
            ]
        )
    # dense 검색 결과를 Serve가 쓰는 공통 dict 모양으로 맞춘다.
    try:
        kwargs = {
            "collection_name": collection_name,
            "query": query_vector,
            "limit": top_k,
        }
        if query_filter is not None:
            kwargs["query_filter"] = query_filter
        hits = client.query_points(**kwargs).points
    except AttributeError:
        kwargs = {
            "collection_name": collection_name,
            "query_vector": query_vector,
            "limit": top_k,
        }
        if query_filter is not None:
            kwargs["query_filter"] = query_filter
        hits = client.search(**kwargs)
    return [
        {
            "score": float(hit.score),
            "text": hit.payload.get("text", ""),
            "metadata": hit.payload.get("metadata", {}),
            "chunk_id": hit.payload.get("chunk_id", ""),
            "doc_id": hit.payload.get("doc_id", ""),
            "section": hit.payload.get("section"),
            "char_span": hit.payload.get("char_span"),
            "token_count": hit.payload.get("token_count"),
            "parent_id": hit.payload.get("parent_id"),
            "hash": hit.payload.get("hash"),
            "retrieval_scope_id": hit.payload.get("retrieval_scope_id"),
        }
        for hit in hits
    ]


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
    collection_name: str = COLLECTION,
) -> list[dict]:
    # dense 점수와 lexical 점수를 chunk_id 기준으로 합친다.
    dense_weight = min(1.0, max(0.0, float(dense_weight)))
    dense_results = search(
        client,
        query_vector,
        top_k=max(top_k * 4, 20),
        retrieval_scope_id=retrieval_scope_id,
        collection_name=collection_name,
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


def rerank(
    query: str,
    results: list[dict],
    model_name: str = DEFAULT_RERANKER_MODEL,
    top_k: int = 5,
) -> list[dict]:
    # reranker가 있으면 한 번 더 정렬하고, 없으면 기존 순서를 유지한다.
    if not results:
        return []
    if model_name not in _rerankers:
        failed_at = _failed_rerankers.get(model_name)
        in_cooldown = (
            failed_at is not None
            and time.monotonic() - failed_at < _FAILED_RERANKER_RETRY_SEC
        )
        if not in_cooldown:
            try:
                from sentence_transformers import CrossEncoder

                _rerankers[model_name] = CrossEncoder(model_name)
                _failed_rerankers.pop(model_name, None)
            except Exception as exc:
                _failed_rerankers[model_name] = time.monotonic()
                print(
                    f"[Index] reranker 로드 실패, 기존 순위 유지 "
                    f"({_FAILED_RERANKER_RETRY_SEC:.0f}초 후 재시도): {exc}"
                )

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
    # 실패는 쿨다운(_FAILED_MODEL_RETRY_SEC) 후 재시도한다 — fallback 임베딩은
    # 의미 검색이 안 되는 해시 벡터라, 이 상태로 Eval/Optimize가 돌면 점수 전체가
    # 무효가 되므로 일시적 실패에서 반드시 복구 기회를 줘야 한다.
    if model_name not in _models:
        failed_at = _failed_models.get(model_name)
        in_cooldown = (
            failed_at is not None
            and time.monotonic() - failed_at < _FAILED_MODEL_RETRY_SEC
        )
        if not in_cooldown:
            try:
                from sentence_transformers import SentenceTransformer

                _models[model_name] = SentenceTransformer(model_name)
                _failed_models.pop(model_name, None)
            except Exception as exc:
                _failed_models[model_name] = time.monotonic()
                print(
                    f"[Index] 임베딩 모델 '{model_name}' 로드 실패 — deterministic "
                    f"fallback 사용, {_FAILED_MODEL_RETRY_SEC:.0f}초 후 재시도: {exc}"
                )
    return _models.get(model_name)


def embedding_is_fallback(model_name: str = DEFAULT_EMBEDDING_MODEL) -> bool:
    """지금 이 모델로 임베딩하면 (해시) fallback 이 되는지 여부.

    호출부(index)가 청크에 임베딩 provenance 를 기록하고, 모델 복구 후 fallback 으로
    색인된 청크를 강제 재임베딩할지 판단하는 데 쓴다. 쿨다운 중이면 로드를 시도하지
    않으므로 계속 True 를 돌려준다(복구 시도는 쿨다운 경과 후). None → fallback."""
    return _get_embedding_model(model_name) is None


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
    # 순수 dict 조회로만 캐시된 모델을 본다(부작용 없음). 프로덕션 유일 호출부
    # (index/agent.py)는 항상 embed 뒤에 불려 모델이 이미 캐시에 있으므로 실제
    # tokenizer를 쓴다. 여기서 _get_embedding_model 로 지연 로드하면 모델 없는
    # 환경에서 HF 다운로드를 유발해 토큰 세기 한 번이 수 초 블로킹된다(리뷰 #36).
    model = _models.get(model_name)
    tokenizer = getattr(model, "tokenizer", None) if model is not None else None
    if tokenizer is None:
        return max(1, len(_tokens(text)))
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return max(1, len(_tokens(text)))
