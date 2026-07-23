# Index와 Serve가 같이 쓰는 저장/검색 유틸.
# 저장할 때와 검색할 때 embedding model/dim이 같아야 한다.
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

_models: dict[tuple[str, str], Any] = {}
_failed_models: set[tuple[str, str]] = set()
_rerankers: dict[str, Any] = {}
_failed_rerankers: set[str] = set()


# 테스트에서는 in-memory, 운영에서는 실제 Qdrant endpoint로 붙는다.
def build_client(url: str = ":memory:", api_key: str | None = None) -> QdrantClient:
    if url == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(url=url, api_key=api_key)


# Qdrant client 버전별 응답 차이를 여기서만 흡수한다.
def _collection_vector_size(client: QdrantClient) -> int | None:
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
    # 차원이 다른 컬렉션에 그대로 덮어쓰면 검색이 깨져서, 명시 옵션 없이는 막는다.
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


# 같은 chunk_id는 같은 point id를 쓰게 해서 재색인을 안전하게 만든다.
def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"agent-doctor:{chunk_id}"))


# Serve가 payload를 그대로 읽으므로 provenance 필드는 빼지 않는다.
def upsert_chunks(client: QdrantClient, chunks: list) -> None:
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
                },
            )
        )
    if points:
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"[Qdrant] {len(points)}개 청크 저장 완료")


# 재색인 대상 문서의 옛 chunk가 검색에 섞이지 않도록 먼저 지운다.
def delete_document_chunks(client: QdrantClient, doc_ids: list[str]) -> None:
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
    retrieval_scope_id: str | None = None,
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
            "collection_name": COLLECTION,
            "query": query_vector,
            "limit": top_k,
        }
        if query_filter is not None:
            kwargs["query_filter"] = query_filter
        hits = client.query_points(**kwargs).points
    except AttributeError:
        kwargs = {
            "collection_name": COLLECTION,
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
) -> list[dict]:
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


def resolve_embedding_device(device: str = "auto") -> str:
    """요청한 임베딩 장치를 실제 사용 가능한 장치로 정규화한다."""
    requested = str(device or "auto").strip().lower()
    if requested == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if requested.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return requested
        except Exception:
            pass
        print("[Index] CUDA를 사용할 수 없어 CPU 임베딩으로 전환합니다.")
        return "cpu"
    return requested


def _load_embedding_model(model_name: str, device: str) -> tuple[Any | None, str]:
    """장치별 모델을 캐시하고 GPU 로드 실패 시 같은 모델을 CPU에서 다시 시도한다."""
    actual_device = resolve_embedding_device(device)
    key = (model_name, actual_device)
    if key not in _models and key not in _failed_models:
        try:
            from sentence_transformers import SentenceTransformer

            _models[key] = SentenceTransformer(model_name, device=actual_device)
        except Exception as exc:
            _failed_models.add(key)
            if actual_device != "cpu":
                print(f"[Index] GPU 임베딩 모델 로드 실패, CPU로 재시도: {exc}")
                return _load_embedding_model(model_name, "cpu")
            print(f"[Index] 임베딩 모델 로드 실패, deterministic fallback 사용: {exc}")
    if key in _failed_models and actual_device != "cpu":
        return _load_embedding_model(model_name, "cpu")
    return _models.get(key), actual_device


def _is_cuda_oom(exc: Exception) -> bool:
    """PyTorch 버전별 CUDA OOM 예외 표현 차이를 흡수한다."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return "cuda" in str(exc).lower() and "out of memory" in str(exc).lower()


def _encode_batch(
    model: Any,
    texts: list[str],
    batch_size: int,
    device: str,
) -> list[list[float]]:
    """CUDA OOM이면 배치 크기를 절반씩 줄여 같은 모델로 재시도한다."""
    current_batch_size = batch_size
    while True:
        try:
            vectors = model.encode(
                texts,
                batch_size=current_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vectors.tolist()
        except Exception as exc:
            if not device.startswith("cuda") or not _is_cuda_oom(exc):
                raise
            if current_batch_size <= 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
            print(
                f"[Index] GPU 메모리 부족, 배치 크기를 "
                f"{current_batch_size}로 줄여 재시도합니다."
            )


def embed_batch(
    texts: list[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    vector_dim: int | None = None,
    device: str = "auto",
    batch_size: int = 16,
) -> list[list[float]]:
    """문서 청크 여러 개를 같은 모델과 장치에서 배치 임베딩한다."""
    if not texts:
        return []
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("embedding_batch_size는 1 이상의 정수여야 합니다.")

    dimension = int(vector_dim or VECTOR_DIM)
    model, actual_device = _load_embedding_model(model_name, device)
    if model is None:
        return [_fallback_embedding(text, dimension) for text in texts]

    try:
        return _encode_batch(model, texts, batch_size, actual_device)
    except Exception as exc:
        if not actual_device.startswith("cuda") or not _is_cuda_oom(exc):
            raise
        print(f"[Index] GPU 메모리 부족이 계속되어 CPU 임베딩으로 전환합니다: {exc}")
        cpu_model, cpu_device = _load_embedding_model(model_name, "cpu")
        if cpu_model is None:
            return [_fallback_embedding(text, dimension) for text in texts]
        return _encode_batch(cpu_model, texts, batch_size, cpu_device)


def embed(
    text: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    vector_dim: int | None = None,
    device: str = "auto",
) -> list[float]:
    # 질의 임베딩 등 기존 단건 호출은 배치 인터페이스 한 건으로 호환한다.
    return embed_batch(
        [text],
        model_name=model_name,
        vector_dim=vector_dim,
        device=device,
        batch_size=1,
    )[0]


def count_tokens(
    text: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> int:
    # tokenizer가 있으면 그걸 쓰고, 없으면 대략적인 토큰 수로 기록한다.
    model = next(
        (cached for (name, _device), cached in _models.items() if name == model_name),
        None,
    )
    tokenizer = getattr(model, "tokenizer", None) if model is not None else None
    if tokenizer is None:
        return max(1, len(_tokens(text)))
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return max(1, len(_tokens(text)))
