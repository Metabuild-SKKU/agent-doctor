"""
Index와 Serve가 함께 사용하는 임베딩·Qdrant·검색 모듈.

문서를 저장할 때와 질문을 검색할 때 반드시 같은 임베딩 모델을 사용해야 한다.
기본 모델은 Notion 설계에서 선택한 BAAI/bge-m3이다.
"""
from __future__ import annotations

import hashlib
import math
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
    VectorParams,
)

COLLECTION = "agent_doctor"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
VECTOR_DIM = 1024

_models: dict[str, Any] = {}
_failed_models: set[str] = set()
_rerankers: dict[str, Any] = {}
_failed_rerankers: set[str] = set()


def build_client(url: str = ":memory:", api_key: str | None = None) -> QdrantClient:
    """환경에 맞는 Qdrant 클라이언트를 생성한다."""
    if url == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(url=url, api_key=api_key)


def _collection_vector_size(client: QdrantClient) -> int | None:
    """Qdrant 버전 차이를 흡수해 기존 컬렉션의 벡터 차원을 읽는다."""
    try:
        vectors = client.get_collection(COLLECTION).config.params.vectors
        if hasattr(vectors, "size"):
            return int(vectors.size)
    except Exception:
        return None
    return None


def ensure_collection(
    client: QdrantClient,
    vector_dim: int = VECTOR_DIM,
    recreate_on_mismatch: bool = False,
) -> None:
    """
    컬렉션이 없으면 생성한다.

    임베딩 모델을 바꿔 차원이 달라졌을 때는 기본적으로 오류를 내고,
    config에서 명시적으로 허용한 경우에만 기존 컬렉션을 다시 만든다.
    """
    existing = [collection.name for collection in client.get_collections().collections]
    if COLLECTION in existing:
        current_dim = _collection_vector_size(client)
        if current_dim is None or current_dim == vector_dim:
            return
        if not recreate_on_mismatch:
            raise ValueError(
                f"Qdrant 벡터 차원이 다릅니다: 기존={current_dim}, 요청={vector_dim}. "
                "recreate_collection_on_dimension_mismatch를 켜거나 새 컬렉션을 사용하세요."
            )
        client.delete_collection(collection_name=COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )
    print(f"[Qdrant] 컬렉션 준비: {COLLECTION} (dim={vector_dim})")


def _point_id(chunk_id: str) -> str:
    """재인덱싱해도 같은 청크가 같은 point id를 사용하도록 만든다."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"agent-doctor:{chunk_id}"))


def upsert_chunks(client: QdrantClient, chunks: list) -> None:
    """임베딩이 있는 Chunk를 안정적인 ID로 Qdrant에 저장한다."""
    points = []
    for chunk in chunks:
        if not chunk.embedding:
            continue
        points.append(
            PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=chunk.embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "section": chunk.section,
                    "metadata": chunk.metadata,
                    "sparse_vector": chunk.sparse_vector,
                },
            )
        )
    if points:
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"[Qdrant] {len(points)}개 청크 저장 완료")


def delete_document_chunks(client: QdrantClient, doc_ids: list[str]) -> None:
    """재인덱싱 대상 문서의 기존 point를 지워 오래된 청크가 남지 않게 한다."""
    unique_ids = sorted({doc_id for doc_id in doc_ids if doc_id})
    if not unique_ids:
        return
    try:
        client.delete(
            collection_name=COLLECTION,
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
                collection_name=COLLECTION,
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
) -> list[dict]:
    """dense cosine similarity 검색 결과를 공통 dict 형식으로 반환한다."""
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
            "score": float(hit.score),
            "text": hit.payload.get("text", ""),
            "metadata": hit.payload.get("metadata", {}),
            "chunk_id": hit.payload.get("chunk_id", ""),
            "doc_id": hit.payload.get("doc_id", ""),
            "section": hit.payload.get("section"),
        }
        for hit in hits
    ]


def _tokens(text: str) -> list[str]:
    """한국어·영어·숫자를 함께 다루는 가벼운 lexical tokenizer."""
    return re.findall(r"[가-힣]+|[A-Za-z][A-Za-z0-9_+.-]*|\d+", text.lower())


def build_sparse_vector(text: str, dimensions: int = 2**20) -> dict:
    """
    토큰을 안정적인 정수 공간에 해싱한 sparse vector를 만든다.

    현재는 API의 hybrid fusion에 사용하고, 이후 Qdrant named sparse vector로
    옮길 수 있도록 indices/values 형태를 유지한다.
    """
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


def _keyword_score(query: str, text: str) -> float:
    """외부 BM25 서버 없이도 재현 가능한 lexical 점수를 계산한다."""
    query_terms = Counter(_tokens(query))
    text_terms = Counter(_tokens(text))
    if not query_terms or not text_terms:
        return 0.0
    matched = sum(min(count, text_terms.get(term, 0)) for term, count in query_terms.items())
    return matched / max(1, sum(query_terms.values()))


def hybrid_search(
    client: QdrantClient,
    query_vector: list[float],
    query: str,
    chunks: list[dict],
    top_k: int = 5,
    dense_weight: float = 0.7,
) -> list[dict]:
    """dense 결과와 lexical 결과를 같은 chunk_id 기준으로 결합한다."""
    dense_weight = min(1.0, max(0.0, float(dense_weight)))
    dense_results = search(client, query_vector, top_k=max(top_k * 4, 20))
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
    """CrossEncoder reranker를 적용하고, 사용할 수 없으면 기존 순서를 유지한다."""
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


def _fallback_embedding(text: str, vector_dim: int) -> list[float]:
    """
    모델을 설치하거나 내려받을 수 없는 환경에서 쓰는 결정적 fallback.

    같은 텍스트는 항상 같은 벡터가 되므로 기존 random fallback보다
    테스트와 키워드 유사도 확인에 적합하다.
    """
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


def embed(
    text: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    vector_dim: int | None = None,
) -> list[float]:
    """텍스트를 정규화된 dense vector로 변환한다."""
    dimension = int(vector_dim or VECTOR_DIM)
    if model_name not in _models and model_name not in _failed_models:
        try:
            from sentence_transformers import SentenceTransformer

            _models[model_name] = SentenceTransformer(model_name)
        except Exception as exc:
            _failed_models.add(model_name)
            print(f"[Index] 임베딩 모델 로드 실패, deterministic fallback 사용: {exc}")

    model = _models.get(model_name)
    if model is None:
        return _fallback_embedding(text, dimension)
    return model.encode(text, normalize_embeddings=True).tolist()
