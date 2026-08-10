# Index와 Serve가 같이 쓰는 저장/검색 유틸.
# 저장할 때와 검색할 때 embedding model/dim이 같아야 한다.
from __future__ import annotations

import hashlib
import inspect
import math
import os
import re
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from weakref import WeakKeyDictionary

from agents.ingest.document_type import has_math_signal
from core.llm_clients import (
    OPENROUTER_BASE_URL,
    normalize_provider,
    openai_embed,
)
from core.llm_retry import run_with_retry
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

_LATEX_FRAC_RE = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_PI_FRACTION_RE = re.compile(r"(π|pi)\s*/\s*([0-9]+)", re.IGNORECASE)
_KOREAN_FRAC_RE = re.compile(
    r"([0-9]+)\s*분의\s*(파이|π|pi|[A-Za-z가-힣0-9]+)",
    re.IGNORECASE,
)
_MATH_SYMBOL_ALIASES = (
    ("π", "pi"),
    ("파이", "pi"),
    ("∞", "infinity"),
    ("무한대", "infinity"),
    ("≤", "<="),
    ("≥", ">="),
)

def _env_float(name: str, default: float) -> float:
    """환경변수 float 파싱 — 비정수/오타면 기본값으로 폴백. import 시점 크래시 방지
    (오타 하나로 모듈 전체 import 가 실패하지 않도록)."""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    """환경변수 int 파싱 — 비정수/오타면 기본값으로 폴백(_env_float 와 같은 규약)."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# (model_name, device) → 로드된 모델. 장치가 키에 들어가는 이유는 같은 모델을
# CUDA 와 CPU 로 동시에 들고 있어야 하기 때문이다 — GPU OOM 이 계속되면 CPU 로
# 폴백하는데, 그때 CPU 모델을 새로 로드해도 GPU 모델은 캐시에 남는다.
# 키를 문자열과 튜플로 섞어 쓰면 같은 모델이 두 벌 상주하고(각 수 GB),
# count_tokens 처럼 한쪽 키로만 조회하는 곳이 조용히 캐시 미스가 된다.
_models: dict[tuple[str, str], Any] = {}
# (model_name, device) → 마지막 로드 실패 시각(monotonic). 실패를 영구 캐시하면
# 일시적 원인(네트워크 등) 후에도 프로세스 내내 fallback 임베딩만 조용히 쓰게
# 되므로, 쿨다운이 지나면 재시도한다.
_failed_models: dict[tuple[str, str], float] = {}
# model_name → tokenizer(또는 로드 실패를 뜻하는 None). 전체 모델과 별개로 캐시한다 —
# API provider 로 색인하면 모델은 안 올라오지만 token_count 는 여전히 필요하다.
_tokenizers: dict[str, Any | None] = {}
# 이미 안내한 임베딩 경로(색인은 스레드로 병렬 호출한다). 경로별로 한 번씩 찍는다 —
# 하나로 묶으면 먼저 찍힌 경로가 나머지를 삼켜, API 로 시작한 실행이 도중에
# 해시 fallback 으로 열화돼도 그 안내가 나오지 않는다.
_embed_routes_notified: set[str] = set()
_embed_route_lock = threading.Lock()
_FAILED_MODEL_RETRY_SEC = _env_float("INDEX_EMBED_MODEL_RETRY_SEC", 300.0)
# reranker_model → 마지막 로드 실패 시각(monotonic). embedding 모델과 같은 쿨다운
# 정책을 따른다 — 영구 캐시하면 일시적 실패 후에도 프로세스 내내 리랭킹이 죽는다.
_rerankers: dict[str, Any] = {}
_failed_rerankers: dict[str, float] = {}
# model_name → 실제로 적용된 입력 토큰 상한(None = 상한 없이 로드됨). 폴백으로 상한이 빠진
# 실행을 리포트가 구분할 수 있어야 한다 — reranker_status 는 그 경우에도 "ready" 라
# capped/uncapped 가 안 남고, 다음 처방 판정이 둘을 같은 실행으로 취급한다.
_reranker_max_lengths: dict[str, int | None] = {}
_FAILED_RERANKER_RETRY_SEC = _env_float("INDEX_RERANKER_RETRY_SEC", 300.0)
# 리랭커(cross-encoder) 입력 토큰 상한. 0 이하면 모델 기본값(이 모델은 8192)을 쓴다.
#
# [역할] 폭주 차단용 안전망이지 비용 절감 장치가 아니다. 기본값 1024 는 **정책 내 구성에서는
# 발동하지 않는다** — 실측(한국어, 이 모델 tokenizer): 1024자=584토큰 / 1500자(정책 최대,
# optimizer.DEFAULT_CONSTRAINTS)=854토큰 이라, 한국어는 약 1800자를 넘어야 걸린다.
# 게다가 sentence-transformers 의 텍스트 패딩은 batch-longest(padding=True)라 상한보다 짧은
# 입력의 계산량은 상한과 무관하다. 즉 이 값을 낮추지 않는 한 리랭크 비용은 안 줄어든다.
#
# [왜 그래도 두나] 상한이 아예 없으면 한 쌍이 8192토큰까지 열린다. 정책 밖 구성(수동
# chunk_size, 청킹 전략 교체로 길어진 청크)이나 코퍼스 특성에 따라 그 꼬리가 실제로 열리므로
# 최악값을 8배 좁혀 둔다.
#
# [왜 더 낮추지 않나] 512(약 868자)로 낮추면 정책상 합법인 chunk_size 후보가 리랭커 입력에서만
# 잘려, Optimize 의 청크 크기 비교가 '진짜 품질 차이'인지 '뒤가 잘려서'인지 구분되지 않는다.
# 비용을 실제로 누르려면 상한이 아니라 rerank_candidates(쌍 수)를 줄이거나 리랭커를 끄는 쪽이다.
# 실측 근거는 새로 붙은 계측(runtime_summary.search / reranker)으로 다음 실행에서 잡는다.
_RERANKER_MAX_LENGTH = _env_int("INDEX_RERANKER_MAX_LENGTH", 1024)
_collection_native_hybrid_cache: WeakKeyDictionary[QdrantClient, dict[str, bool]] = (
    WeakKeyDictionary()
)
# Serve의 동시 검색이 같은 CrossEncoder를 중복 로드하지 않도록 모델 캐시와
# 실패 쿨다운 갱신을 한 임계구역에서 처리한다. 추론 자체는 병렬 검색을 막지 않는다.
_reranker_lock = threading.Lock()


def _accepts_max_length(cross_encoder_cls) -> bool:
    """생성자가 max_length 를 받나 — 예외로 떠보지 않고 시그니처로 판정한다.

    TypeError 를 잡아 재시도하면 생성자 **내부**에서 난 무관한 TypeError 까지 '상한 미지원'
    으로 오진하고, 그 오진의 대가로 2GB 대 모델을 한 번 더 로드한다. 시그니처 조회는 공짜다.
    **kwargs 를 받는 래퍼는 지원으로 본다 — 넘겨봐야 알 수 있고, 틀렸다면 로드 실패로
    드러나는 편이 상한만 조용히 빠지는 것보다 낫다. 시그니처를 못 읽으면 보수적으로 False.
    """
    try:
        params = inspect.signature(cross_encoder_cls).parameters
    except (TypeError, ValueError):
        return False
    if "max_length" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _new_cross_encoder(cross_encoder_cls, model_name: str):
    """입력 길이 상한(_RERANKER_MAX_LENGTH)을 걸어 CrossEncoder 를 만든다.

    상한 인자를 못 받는 구현이면 상한 없이 만들되 조용히 넘어가지 않는다 — 상한이 빠지면
    리랭크 비용이 다시 chunk_size 에 비례해 열리므로, 로그로 드러내야 원인을 찾을 수 있다.

    경고 문구는 ASCII 기호로만 쓴다. 이 모듈은 진입점의 인코딩 보정
    (core.console.force_utf8_stdio)을 거치지 않고 import 될 수 있어, cp949 콘솔에서
    UnicodeEncodeError 가 나면 호출부(_load_reranker)의 except 가 로드 실패로 삼킨다.
    """
    if _RERANKER_MAX_LENGTH <= 0:
        _reranker_max_lengths[model_name] = None
        return cross_encoder_cls(model_name)
    if not _accepts_max_length(cross_encoder_cls):
        print("[Index] reranker 구현이 max_length 를 받지 않아 입력 길이 상한 없이 로드한다 "
              "(청크가 크면 리랭크 비용이 그만큼 커진다)")
        _reranker_max_lengths[model_name] = None
        return cross_encoder_cls(model_name)
    _reranker_max_lengths[model_name] = _RERANKER_MAX_LENGTH
    return cross_encoder_cls(model_name, max_length=_RERANKER_MAX_LENGTH)


def reranker_max_length(model_name: str = DEFAULT_RERANKER_MODEL) -> int | None:
    """이 모델에 실제로 적용된 입력 토큰 상한. 상한 없이 로드됐거나 아직 안 실렸으면 None."""
    return _reranker_max_lengths.get(model_name)


def _load_reranker(model_name: str) -> tuple[Any | None, str]:
    """reranker를 한 번만 로드하고 준비 상태를 정규화해 반환한다."""
    with _reranker_lock:
        model = _rerankers.get(model_name)
        if model is not None:
            return model, "ready"

        failed_at = _failed_rerankers.get(model_name)
        if (
            failed_at is not None
            and time.monotonic() - failed_at < _FAILED_RERANKER_RETRY_SEC
        ):
            return None, "cooldown"

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            _failed_rerankers[model_name] = time.monotonic()
            print(f"[Index] reranker 의존성 없음, 기존 순위 유지: {exc}")
            return None, "dependency_missing"

        try:
            # 모델 생성까지 lock 안에서 수행해야 동시에 들어온 요청이 같은
            # 대형 모델을 각각 메모리에 올리는 것을 막을 수 있다.
            model = _new_cross_encoder(CrossEncoder, model_name)
        except Exception as exc:
            _failed_rerankers[model_name] = time.monotonic()
            print(
                f"[Index] reranker 로드 실패, 기존 순위 유지 "
                f"({_FAILED_RERANKER_RETRY_SEC:.0f}초 후 재시도): {exc}"
            )
            return None, "model_load_failed"

        _rerankers[model_name] = model
        _failed_rerankers.pop(model_name, None)
        return model, "ready"


def probe_reranker_capability(
    model_name: str = DEFAULT_RERANKER_MODEL,
    *,
    smoke_test: bool = True,
) -> dict[str, Any]:
    """Index가 Optimize에 전달할 reranker 실행 가능성을 실제 모델로 확인한다."""
    resolved_name = str(model_name or DEFAULT_RERANKER_MODEL)
    model, load_status = _load_reranker(resolved_name)
    if model is None:
        return {
            "status": "unavailable",
            "model": resolved_name,
            "checked_at": time.time(),
            "retryable": load_status not in {"dependency_missing"},
            "reason": load_status,
        }

    if smoke_test:
        try:
            scores = list(model.predict([("질문", "문서")]))
            if len(scores) != 1:
                raise ValueError(f"smoke 점수 개수 불일치: {len(scores)} != 1")
            float(scores[0])
        except Exception as exc:
            with _reranker_lock:
                if _rerankers.get(resolved_name) is model:
                    _rerankers.pop(resolved_name, None)
                _failed_rerankers[resolved_name] = time.monotonic()
            print(f"[Index] reranker smoke inference 실패: {exc}")
            return {
                "status": "unavailable",
                "model": resolved_name,
                "checked_at": time.time(),
                "retryable": True,
                "reason": "inference_failed",
            }

    return {
        "status": "verified",
        "model": resolved_name,
        "checked_at": time.time(),
        "retryable": False,
        "reason": None,
    }


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


def _collection_has_native_hybrid(
    client: QdrantClient,
    collection_name: str = COLLECTION,
) -> bool:
    cache = _collection_native_hybrid_cache.setdefault(client, {})
    if isinstance(cache, bool):
        cache = {COLLECTION: cache}
        _collection_native_hybrid_cache[client] = cache
    if collection_name in cache:
        return cache[collection_name]
    try:
        params = client.get_collection(collection_name).config.params
        vectors = params.vectors
        sparse_vectors = getattr(params, "sparse_vectors", None) or {}
        has_native = (
            isinstance(vectors, dict)
            and DENSE_VECTOR_NAME in vectors
            and SPARSE_VECTOR_NAME in sparse_vectors
        )
    except Exception:
        return False
    cache[collection_name] = has_native
    return has_native


def _clear_collection_shape_cache(
    client: QdrantClient,
    collection_name: str | None = None,
) -> None:
    if collection_name is None:
        _collection_native_hybrid_cache.pop(client, None)
        return
    cache = _collection_native_hybrid_cache.get(client)
    if cache is not None:
        cache.pop(collection_name, None)


def _is_collection_shape_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "vector" in message
        and (
            "not found" in message
            or "name" in message
            or "existing vector" in message
            or "dense" in message
            or "sparse" in message
        )
    )


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
        "original_char_span": payload.get("original_char_span"),
        "duplicate_spans": payload.get("duplicate_spans"),
        "token_count": payload.get("token_count"),
        "parent_id": payload.get("parent_id"),
        "hash": payload.get("hash"),
        "retrieval_scope_id": payload.get("retrieval_scope_id"),
    }


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
            if _collection_has_native_hybrid(client, collection_name):
                return
            if recreate_on_mismatch:
                print(
                    f"[Qdrant] legacy dense-only 컬렉션 재생성: {collection_name} "
                    "(native hybrid shape로 마이그레이션)"
                )
                client.delete_collection(collection_name=collection_name)
                _clear_collection_shape_cache(client, collection_name)
            else:
                print(
                    f"[Qdrant] legacy dense-only 컬렉션 사용: {collection_name} "
                    "(native hybrid를 쓰려면 컬렉션 재생성이 필요합니다)"
                )
                return
        else:
            if not recreate_on_mismatch:
                raise ValueError(
                    f"Qdrant 벡터 차원이 다릅니다: 기존={current_dim}, 요청={vector_dim}. "
                    "recreate_collection_on_dimension_mismatch를 켜거나 새 컬렉션을 사용하세요."
                )
            client.delete_collection(collection_name=collection_name)
            _clear_collection_shape_cache(client, collection_name)
        if collection_name in [collection.name for collection in client.get_collections().collections]:
            return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(size=vector_dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(),
        },
    )
    _clear_collection_shape_cache(client, collection_name)
    print(f"[Qdrant] 컬렉션 준비: {collection_name} (dim={vector_dim})")


# 같은 chunk_id는 같은 point id를 쓰게 해서 재색인을 안전하게 만든다.
def _point_id(chunk_id: str, retrieval_scope_id: str | None = None) -> str:
    scope = retrieval_scope_id or "global"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"agent-doctor:{scope}:{chunk_id}"))


# Serve가 payload를 그대로 읽으므로 provenance 필드는 빼지 않는다.
def upsert_chunks(
    client: QdrantClient,
    chunks: list,
    collection_name: str = COLLECTION,
) -> None:
    def _points(use_native_hybrid: bool) -> list[PointStruct]:
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
                "original_char_span": chunk.original_char_span,
                "duplicate_spans": chunk.duplicate_spans,
                "token_count": chunk.token_count,
                "parent_id": chunk.parent_id,
                "hash": chunk.hash,
                "metadata": metadata,
                "sparse_vector": chunk.sparse_vector,
                "retrieval_scope_id": retrieval_scope_id,
                "index_cache_key": metadata.get("index_cache_key"),
            }
            points.append(
                PointStruct(
                    id=_point_id(chunk.chunk_id, retrieval_scope_id),
                    vector=vector,
                    payload=payload,
                )
            )
        return points

    use_native_hybrid = _collection_has_native_hybrid(client, collection_name)
    points = _points(use_native_hybrid)
    if not points:
        return
    try:
        client.upsert(collection_name=collection_name, points=points)
    except Exception as exc:
        if not _is_collection_shape_error(exc):
            raise
        _clear_collection_shape_cache(client, collection_name)
        points = _points(_collection_has_native_hybrid(client, collection_name))
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
    retrieval_scope_id: str | None = None,
    collection_name: str = COLLECTION,
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
            collection_name=collection_name,
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
                collection_name=collection_name,
                points_selector=Filter(must=must),
            )


def search(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int = 5,
    retrieval_scope_id: str | None = None,
    collection_name: str = COLLECTION,
) -> list[dict]:
    query_filter = _query_filter(retrieval_scope_id)
    # dense 검색 결과를 Serve가 쓰는 공통 dict 모양으로 맞춘다.
    def _query_once(native_hybrid_collection: bool):
        kwargs = {
            "collection_name": collection_name,
            "query": query_vector,
            "limit": top_k,
        }
        if native_hybrid_collection:
            kwargs["using"] = DENSE_VECTOR_NAME
        if query_filter is not None:
            kwargs["query_filter"] = query_filter
        if hasattr(client, "query_points"):
            return client.query_points(**kwargs).points
        if not native_hybrid_collection and hasattr(client, "search"):
            search_kwargs = {
                "collection_name": collection_name,
                "query_vector": query_vector,
                "limit": top_k,
            }
            if query_filter is not None:
                search_kwargs["query_filter"] = query_filter
            return client.search(**search_kwargs)
        raise AttributeError("Qdrant client has neither query_points nor legacy search")

    try:
        hits = _query_once(_collection_has_native_hybrid(client, collection_name))
    except Exception as exc:
        if not _is_collection_shape_error(exc):
            raise
        _clear_collection_shape_cache(client, collection_name)
        hits = _query_once(_collection_has_native_hybrid(client, collection_name))
    return [_hit_to_result(hit) for hit in hits]


# 형태소 분석기 없이도 테스트/하이브리드 검색이 돌아가게 가볍게 쪼갠다.
def normalize_math_text(text: str) -> str:
    """수식/한국어 표현 차이를 sparse·keyword 검색 힌트로 확장한다."""
    value = text or ""
    if not has_math_signal(value):
        return value

    aliases: list[str] = []

    for numerator, denominator in _LATEX_FRAC_RE.findall(value):
        num = _normalize_math_piece(numerator)
        den = _normalize_math_piece(denominator)
        if num and den:
            aliases.extend([f"{num}/{den}", f"{num} over {den}"])

    for denominator, numerator in _KOREAN_FRAC_RE.findall(value):
        num = _normalize_math_piece(numerator)
        den = _normalize_math_piece(denominator)
        if num and den:
            aliases.extend([f"{num}/{den}", f"{num} over {den}"])

    for numerator, denominator in _PI_FRACTION_RE.findall(value):
        num = _normalize_math_piece(numerator)
        den = _normalize_math_piece(denominator)
        if num and den:
            aliases.extend([f"{num}/{den}", f"{num} over {den}"])
            if num == "pi":
                aliases.append(f"π/{den}")

    for left, right in _MATH_SYMBOL_ALIASES:
        if left in value:
            aliases.append(right)
        if right in value:
            aliases.append(left)

    for base, power in re.findall(r"([A-Za-z가-힣]+)\s*\^\s*\{?([0-9]+)\}?", value):
        aliases.append(f"{base}{power}")

    if not aliases:
        return value
    unique_aliases = []
    for alias in aliases:
        alias = alias.strip()
        if alias and alias not in unique_aliases:
            unique_aliases.append(alias)
    return f"{value} {' '.join(unique_aliases)}"


def _normalize_math_piece(value: str) -> str:
    value = (value or "").strip().lower()
    value = value.replace("\\", "").replace("{", "").replace("}", "")
    return {"π": "pi", "파이": "pi"}.get(value, value)


def _tokens(text: str) -> list[str]:
    expanded = normalize_math_text(text)
    return re.findall(r"[가-힣]+|[A-Za-z][A-Za-z0-9_+./-]*|\d+(?:/\d+)?", expanded.lower())


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
                "original_char_span": _field(chunk, "original_char_span"),
                "duplicate_spans": _field(chunk, "duplicate_spans"),
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
    native_results = _native_hybrid_search(
        client,
        query_vector=query_vector,
        query=query,
        top_k=top_k,
        dense_weight=dense_weight,
        retrieval_scope_id=retrieval_scope_id,
        collection_name=collection_name,
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


def _native_hybrid_search(
    client: QdrantClient,
    *,
    query_vector: list[float],
    query: str,
    top_k: int,
    dense_weight: float,
    retrieval_scope_id: str | None,
    collection_name: str = COLLECTION,
) -> list[dict] | None:
    if Prefetch is None or Rrf is None or RrfQuery is None:
        return None
    sparse = _sparse_vector(build_sparse_vector(query))
    if sparse is None or not _collection_has_native_hybrid(client, collection_name):
        return None

    dense_weight = min(1.0, max(0.0, float(dense_weight)))
    sparse_weight = 1.0 - dense_weight
    candidate_k = max(top_k * 4, 20)
    query_filter = _query_filter(retrieval_scope_id)
    try:
        hits = client.query_points(
            collection_name=collection_name,
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


def rerank_with_status(
    query: str,
    results: list[dict],
    model_name: str = DEFAULT_RERANKER_MODEL,
    top_k: int = 5,
) -> tuple[list[dict], str, float]:
    """재정렬 결과·실행 상태·**추론에만 든 시간(초)** 을 함께 반환한다.

    시간을 호출부가 아니라 여기서 재는 이유가 둘 있다.
      1) 모델 로드 제외가 구조로 보장된다 — _load_reranker 는 타이머 바깥이다. 호출부에서
         재면 쿨다운 만료 직후 첫 쿼리가 2GB 대 재로드를 '쌍당 비용'으로 뒤집어쓴다.
      2) seam 이 하나로 유지된다 — 이 함수만 패치하면 테스트에서 모델 로드가 일어나지 않는다.
         호출부에 사전 로드를 따로 두면 그 패치를 우회해 단위 테스트가 실모델을 내려받는다.
    실행하지 못한 경우(후보 없음·로드 실패·쿨다운·추론 실패)는 0.0 — 리랭크한 적이 없는
    시간을 쌍당 비용 집계에 섞지 않는다.
    """
    if not results:
        return [], "not_attempted", 0.0
    model, load_status = _load_reranker(model_name)
    if model is None:
        return results[:top_k], load_status, 0.0

    try:
        started = time.monotonic()
        scores = list(
            model.predict([(query, item.get("text", "")) for item in results])
        )
        elapsed = time.monotonic() - started
        if len(scores) != len(results):
            raise ValueError(
                f"reranker 점수 개수 불일치: {len(scores)} != {len(results)}"
            )
    except Exception as exc:
        # 깨진 모델 객체로 매 요청마다 같은 실패를 반복하지 않고 쿨다운 후 다시 로드한다.
        with _reranker_lock:
            # 다른 요청이 이미 새 모델을 넣었다면 그 객체까지 지우지 않는다.
            if _rerankers.get(model_name) is model:
                _rerankers.pop(model_name, None)
            _failed_rerankers[model_name] = time.monotonic()
        print(
            f"[Index] reranker 추론 실패, 기존 순위 유지 "
            f"({_FAILED_RERANKER_RETRY_SEC:.0f}초 후 재시도): {exc}"
        )
        return results[:top_k], "inference_failed", 0.0
    reranked = [
        {**item, "retrieval_score": item.get("score", 0.0), "score": float(score)}
        for item, score in zip(results, scores)
    ]
    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k], "applied", elapsed


def rerank(
    query: str,
    results: list[dict],
    model_name: str = DEFAULT_RERANKER_MODEL,
    top_k: int = 5,
) -> list[dict]:
    """기존 호출자를 위해 결과 리스트만 반환하는 호환 API."""
    reranked, _status, _seconds = rerank_with_status(
        query,
        results,
        model_name=model_name,
        top_k=top_k,
    )
    return reranked


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


def resolve_embedding_provider(provider: str | None = None) -> str:
    """임베딩을 어디서 계산할지. None → env INDEX_EMBED_PROVIDER(기본 openrouter).

    [기본값이 openrouter 인 이유]
    이 프로젝트는 OpenRouter 예산이 확보된 상태로 운영한다는 전제다. 그 전제에서는
    CPU 로 돌릴 이유가 사실상 없다 — 실측(AGENTS.md "임베딩 provider" 절)에서 로컬 CPU 는
    2 chunks/sec, OpenRouter 는 동시 8에서 371 chunks/sec 였다. 26MB 코퍼스 환산으로
    2.3시간 vs 0.7분이고 비용은 $0.06 다. 100배 넘는 시간 차이를 몇 센트로 사는
    셈이라, "예산이 있는데 CPU 를 쓰는" 선택지는 실질적으로 없다고 보고 기본값을
    빠른 쪽에 뒀다. 예산이 없거나 오프라인이면 local 로 내린다.

    바꿔도 안전하다: 로컬 BAAI/bge-m3 와 OpenRouter baai/bge-m3 의 벡터는 코사인
    0.99997 로 사실상 같고 차원도 1024 로 동일해, provider 를 바꿔도 컬렉션을 다시
    만들 필요가 없다.

    미지원 값은 openrouter 로 떨어뜨리지 않고 그대로 돌려준다 — 호출부가 로컬로
    처리하므로, 오타가 "조용히 API 과금" 이 아니라 "조용히 로컬" 로 끝난다."""
    raw = provider if provider is not None else os.getenv("INDEX_EMBED_PROVIDER")
    return normalize_provider(raw) or "openrouter"


def resolve_query_embedding_provider(provider: str | None = None) -> str:
    """질의 임베딩을 어디서 계산할지.

    우선순위: 인자 > INDEX_QUERY_EMBED_PROVIDER > INDEX_EMBED_PROVIDER(색인 축) > openrouter.

    기본값은 색인과 같은 openrouter 다(그 이유는 resolve_embedding_provider 참고 —
    예산이 있는 전제에서 CPU 를 쓸 이유가 없다는 판단).

    그런데도 별개 축으로 둔다. 값이 같아도 성질이 다르기 때문이다 — 색인은
    대량·일회성이라 재시도가 남는 장사지만, 질의는 단건·대화형이라 재시도가 그대로
    사용자 지연이 된다. 실측 429 가 19% 라 단건 질의가 재시도에 걸리면 /search 한 건이
    수십 초 블로킹될 수 있다. 그때 색인은 그대로 두고 이 축만 local 로 내리면 된다.

    섞어도 안전하다 — 로컬과 OpenRouter 의 bge-m3 벡터는 코사인 0.99997 이고 차원도
    같아, 색인을 API 로 질의를 로컬로 계산해도 순위가 흔들리지 않는다
    (실측 — AGENTS.md "임베딩 provider" 절 참고).

    주의: 질의 경로에는 색인 같은 "시끄럽게 실패" 가 없다. agents/rag/retriever.py 가
    벡터 검색 예외를 잡아 keyword 로 내리므로, 키가 없거나 API 가 계속 실패하면
    멈추는 대신 검색 품질만 조용히 떨어진다. 그래서 retriever 진입 시
    query_embedding_config_error() 로 설정 오류만 골라 한 번 끊는다."""
    # 미지정이면 색인 축을 따른다. 이 폴백이 없으면 색인 에러가 시킨 대로
    # INDEX_EMBED_PROVIDER=local 만 고친 사용자가 곧바로 질의 preflight 에서 두 번째
    # 에러를 만난다 — 첫 에러가 시킨 대로 했는데 안 되는 상태다. 오프라인·무키 환경이
    # 정확히 이 경로를 밟는다(.env.example 머리말의 "키가 없어도 폴백으로 돈다" 약속).
    #
    # CLI 는 이미 이렇게 동작한다(core/embedding_cli.py 의 query_target = --query-embed
    # or --embed). env 만 다르게 두면 같은 설정을 어디에 쓰느냐로 결과가 갈린다.
    # 명시값은 그대로 이기므로 "별개 축" 설계는 유지된다.
    raw = provider if provider is not None else (
        os.getenv("INDEX_QUERY_EMBED_PROVIDER") or os.getenv("INDEX_EMBED_PROVIDER")
    )
    return normalize_provider(raw) or "openrouter"


def query_embedding_config_error(provider: str | None = None) -> str | None:
    """질의 임베딩 설정이 애초에 불가능한 상태면 사유 문자열, 정상이면 None.

    "설정이 틀렸다" 와 "API 가 잠깐 흔들린다" 를 가르기 위한 함수다. 후자는
    retriever 의 keyword 폴백으로 흡수하는 게 맞지만, 전자는 폴백하면 안 된다 —
    키를 안 넣은 실행이 영구히 keyword 검색으로 도는데 증상은 "검색 품질이 좀
    나쁘다" 로만 나타나 원인을 찾을 수 없다. 호출은 하지 않고 키 유무만 본다.

    색인 경로는 _embed_via_openrouter 가 곧바로 예외를 던져 이미 시끄럽지만,
    질의 경로는 retriever 가 예외를 삼키므로 진입 시점에 한 번 끊어야 한다."""
    if resolve_query_embedding_provider(provider) != "openrouter":
        return None
    if os.getenv("OPENROUTER_API_KEY"):
        return None
    return (
        "INDEX_QUERY_EMBED_PROVIDER=openrouter 인데 OPENROUTER_API_KEY 가 없습니다. "
        "키를 채우거나 INDEX_QUERY_EMBED_PROVIDER=local 로 두세요 "
        "(그대로 두면 모든 질의가 keyword 검색으로 조용히 떨어집니다)."
    )


def _openrouter_embed_model(model_name: str) -> str:
    """로컬 모델명을 OpenRouter 철자로. env 로 직접 지정할 수도 있다.

    로컬은 "BAAI/bge-m3", OpenRouter 는 "baai/bge-m3" 로 대소문자만 다르다.
    다른 모델을 로컬 이름 그대로 넘기면 OpenRouter 가 404 를 내는데, 그게
    조용한 오배선보다 낫다."""
    override = os.getenv("INDEX_EMBED_MODEL_OPENROUTER")
    if override:
        return override
    return model_name.lower()


def _notify_embed_route_once(route: str, message: str) -> None:
    """임베딩 경로를 경로마다 실행당 한 번씩 알린다.

    청크마다 찍으면 정작 봐야 할 로그를 덮는다(graph_index 의
    _notify_llm_extraction_once 와 같은 패턴). 한 번은 반드시 남겨야 하는 이유는,
    provider 가 env 로 정해져 실행 기록만 봐서는 이 색인이 어디서 계산됐는지
    알 수 없기 때문이다 — 비용과 속도가 100배 넘게 갈리는 축이다.

    플래그를 경로별로 나누는 이유: 하나로 묶으면 먼저 찍힌 경로가 나머지를 삼킨다.
    특히 API 로 시작한 실행이 도중에 해시 fallback 으로 열화됐을 때 그 안내가
    억제되는데, 그건 "어디서 계산됐는지 기록한다" 는 목적과 정확히 반대다."""
    with _embed_route_lock:
        if route in _embed_routes_notified:
            return
        _embed_routes_notified.add(route)
    print(message)


def _embed_via_openrouter(texts: list[str], model_name: str) -> list[list[float]]:
    """OpenRouter 임베딩. 실패는 예외로 올린다(해시 fallback 없음).

    로컬 경로는 모델 로드 실패 시 해시 벡터로 떨어지되 embedding_fallback 을 기록해
    나중에 복구할 수 있다(agents/index/agent.py 의 재임베딩 교체 경로). API 가 조용히
    실패하면 기록할 provenance 자체가 없어 어느 벡터가 쓰레기인지 구분할 수 없고,
    결국 전체 재색인을 해야 한다 — 실패보다 비싸다.

    429 는 동시성과 무관하게 상시 섞여 나온다(실측: 동시 1 에서도 요청의 19%).
    요청당 한도가 아니라 상위 제공자의 transient 로 보이므로 동시성을 낮추는 게
    아니라 재시도로 흡수한다."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "INDEX_EMBED_PROVIDER=openrouter 인데 OPENROUTER_API_KEY 가 없습니다. "
            "키를 채우거나 INDEX_EMBED_PROVIDER=local 로 두세요."
        )

    model = _openrouter_embed_model(model_name)
    # 0 이나 음수면 range(step=0) 이 ValueError 로 죽는다. 바로 아래 concurrency 와
    # 같은 방어를 건다 — 설정 오타가 크래시가 되면 안 된다.
    batch_size = max(1, _env_int("INDEX_EMBED_API_BATCH", 64))
    concurrency = max(1, _env_int("INDEX_EMBED_CONCURRENCY", 8))
    groups = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    _notify_embed_route_once(
        "openrouter",
        f"[Index] 임베딩을 OpenRouter({model})로 계산합니다 — 요청당 {batch_size}건, "
        f"동시 {concurrency}. 로컬로 돌리려면 INDEX_EMBED_PROVIDER=local."
    )

    def _one(group: list[str]) -> list[list[float]]:
        result = run_with_retry(
            lambda: openai_embed(
                group, model,
                api_key=api_key, base_url=OPENROUTER_BASE_URL, tag="Index",
            ),
            label="임베딩",
            tag="Index",
        )
        # 개수가 어긋나면 예외로 끊는다. 호출부가 zip(청크, 벡터) 로 짝짓기 때문에
        # 짧게 온 응답은 뒤쪽 청크를 조용히 색인에서 지운다 — 벤치에서 재시도 없이
        # 429 를 삼켰을 때 1,000청크 중 170개가 사라진 것과 같은 실패 모양이고,
        # 그때처럼 "성공한 색인" 으로 보이는 게 가장 나쁘다.
        if len(result) != len(group):
            raise RuntimeError(
                f"OpenRouter 임베딩 응답 개수가 어긋납니다: 요청 {len(group)} / "
                f"응답 {len(result)} (model={model}). 부분 응답을 그대로 쓰면 "
                f"벡터가 엉뚱한 청크에 붙습니다."
            )
        return result

    if len(groups) == 1:
        return _one(groups[0])

    # 입력 순서를 유지해야 한다 — 호출부가 청크와 zip 으로 짝짓는다.
    vectors: list[list[list[float]]] = [[] for _ in groups]
    with ThreadPoolExecutor(max_workers=min(concurrency, len(groups))) as pool:
        futures = {pool.submit(_one, group): idx for idx, group in enumerate(groups)}
        for future in as_completed(futures):
            vectors[futures[future]] = future.result()
    return [vector for group in vectors for vector in group]


def resolve_embedding_device(device: str | None = None) -> str:
    """요청한 임베딩 장치를 실제 사용 가능한 장치로 정규화한다.

    None → env INDEX_EMBED_DEVICE(기본 auto). auto 는 CUDA 가 있으면 cuda, 없으면
    cpu. cuda 를 명시했는데 쓸 수 없으면 경고하고 cpu 로 내린다 — 조용히 내리면
    "GPU 로 돌리는 중" 이라 믿은 실행이 CPU 속도(실측 2 chunks/sec)로 기어간다."""
    requested = str(
        device if device is not None else os.getenv("INDEX_EMBED_DEVICE", "auto")
    ).strip().lower() or "auto"
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
        print("[Index] CUDA 를 사용할 수 없어 CPU 임베딩으로 전환합니다.")
        return "cpu"
    return requested


def _load_embedding_model(
    model_name: str,
    device: str | None = None,
) -> tuple[Any | None, str]:
    """(모델, 실제 장치). 장치별로 캐시하고 실패 쿨다운·GPU→CPU 폴백을 함께 적용한다.

    실패 시 None → 호출부가 fallback 사용. 실패는 쿨다운(_FAILED_MODEL_RETRY_SEC)
    후 재시도한다 — fallback 임베딩은 의미 검색이 안 되는 해시 벡터라, 이 상태로
    Eval/Optimize 가 돌면 점수 전체가 무효가 되므로 반드시 복구 기회를 줘야 한다."""
    actual_device = resolve_embedding_device(device)
    key = (model_name, actual_device)
    cached = _models.get(key)
    if cached is not None:
        return cached, actual_device

    failed_at = _failed_models.get(key)
    in_cooldown = (
        failed_at is not None
        and time.monotonic() - failed_at < _FAILED_MODEL_RETRY_SEC
    )
    if in_cooldown:
        # GPU 가 쿨다운 중이면 해시 fallback 보다 CPU 실모델이 낫다. CPU 마저
        # 쿨다운이면 아래 재귀가 곧바로 None 을 돌려준다(무한 재귀 없음).
        if actual_device != "cpu":
            return _load_embedding_model(model_name, "cpu")
        return None, actual_device

    try:
        from sentence_transformers import SentenceTransformer

        _models[key] = SentenceTransformer(model_name, device=actual_device)
        _failed_models.pop(key, None)
    except Exception as exc:
        _failed_models[key] = time.monotonic()
        if actual_device != "cpu":
            print(
                f"[Index] 임베딩 모델 '{model_name}' 을 {actual_device} 로 로드하지 "
                f"못해 CPU 로 재시도합니다 ({_FAILED_MODEL_RETRY_SEC:.0f}초 후 "
                f"{actual_device} 재시도): {exc}"
            )
            return _load_embedding_model(model_name, "cpu")
        print(
            f"[Index] 임베딩 모델 '{model_name}' 로드 실패, deterministic "
            f"fallback 사용, {_FAILED_MODEL_RETRY_SEC:.0f}초 후 재시도: {exc}"
        )
    return _models.get(key), actual_device


def _get_embedding_model(
    model_name: str,
    device: str | None = None,
) -> Any | None:
    """_load_embedding_model 의 모델만 돌려주는 얇은 래퍼(기존 호출부 호환)."""
    model, _actual_device = _load_embedding_model(model_name, device)
    return model


def embedding_is_fallback(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str | None = None,
    provider: str | None = None,
) -> bool:
    """지금 이 모델로 임베딩하면 (해시) fallback 이 되는지 여부.

    호출부(index)가 청크에 임베딩 provenance 를 기록하고, 모델 복구 후 fallback 으로
    색인된 청크를 강제 재임베딩할지 판단하는 데 쓴다. GPU 가 쿨다운이어도 CPU 실모델을
    쓸 수 있으면 False 다 — 해시 벡터를 쓰는지가 기준이지 어느 장치인지가 아니다.
    Eval 의 로컬 임베딩 가용성 판정(agents/eval/llm_provider.py)도 이 함수를 본다.

    API provider 는 해시 fallback 이라는 상태 자체가 없다(성공 아니면 예외)."""
    if resolve_embedding_provider(provider) == "openrouter":
        return False
    return _get_embedding_model(model_name, device) is None


def _is_cuda_oom(exc: Exception) -> bool:
    """PyTorch 버전별 CUDA OOM 예외 표현 차이를 흡수한다."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def _encode_batch(
    model: Any,
    texts: list[str],
    batch_size: int,
    device: str,
) -> list[list[float]]:
    """CUDA OOM 이면 배치 크기를 절반씩 줄여 같은 모델로 재시도한다.

    batch_size 는 한 번에 GPU 로 올리는 양이라 OOM 의 직접 원인이고, 줄여서 다시
    하면 대개 통과한다. 1 까지 줄여도 안 되면 포기하고 올려보낸다 — 호출부가
    CPU 로 내린다."""
    current_batch_size = batch_size
    while True:
        try:
            vectors = model.encode(
                texts,
                batch_size=current_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [vector.tolist() for vector in vectors]
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
            # 이 print 는 except 안이라 문구에 cp949 불가 문자를 쓰면 안 된다.
            # 터지면 UnicodeEncodeError 가 올라가는데, 아래 embed_batch 의
            # _is_cuda_oom(exc) 가 False 라 그대로 재전파돼 배치 축소와 CPU 폴백이
            # 통째로 죽는다 — OOM 을 처리하려던 코드가 OOM 처리를 없앤다.
            # (AGENTS.md 코드 컨벤션, tests/test_console_encoding.py 가 고정한다)
            print(
                f"[Index] GPU 메모리 부족, 배치 크기를 {current_batch_size} 로 "
                f"줄여 재시도합니다."
            )


def embed_batch(
    texts: list[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    vector_dim: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    provider: str | None = None,
) -> list[list[float]]:
    """텍스트 리스트를 배치 인코딩한다(색인용). 결과는 입력 순서 그대로.

    provider: None → env INDEX_EMBED_PROVIDER(기본 openrouter). local 이 아니면
    아래 device/batch_size 는 쓰이지 않는다.
    batch_size: None → env INDEX_EMBED_BATCH(기본 32). 1이면 모델이 텍스트별로
    인코딩하므로 단건 embed() 와 결과가 같아진다(kill-switch).
    device: None → env INDEX_EMBED_DEVICE(기본 auto).

    CUDA OOM 은 두 단계로 흡수한다 — 먼저 배치를 절반씩 줄여 재시도하고,
    1 까지 줄여도 안 되면 CPU 모델로 옮겨 같은 텍스트를 다시 인코딩한다."""
    if not texts:
        return []
    if resolve_embedding_provider(provider) == "openrouter":
        return _embed_via_openrouter(texts, model_name)
    dimension = int(vector_dim or VECTOR_DIM)
    if batch_size is None:
        try:
            batch_size = int(os.getenv("INDEX_EMBED_BATCH", "32"))
        except ValueError:
            batch_size = 32
    batch_size = max(1, batch_size)

    model, actual_device = _load_embedding_model(model_name, device)
    if model is None:
        _notify_embed_route_once(
            "hash_fallback",
            f"[Index] 임베딩 모델 '{model_name}' 을 못 써 해시 fallback 벡터로 "
            f"색인합니다 — 의미 검색이 되지 않습니다."
        )
        return [_fallback_embedding(text, dimension) for text in texts]
    _notify_embed_route_once(
        f"local:{actual_device}",
        f"[Index] 임베딩을 로컬 {actual_device.upper()}({model_name})로 계산합니다 "
        f"— 비용 0, 외부 호출 없음."
    )

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
    device: str | None = None,
    provider: str | None = None,
) -> list[float]:
    """단건 임베딩(질의용). batch_size=1 로 embed_batch 를 한 번 타 경로를 통일한다.

    provider 를 명시하지 않으면 색인이 아니라 **질의 축**(INDEX_QUERY_EMBED_PROVIDER,
    기본 openrouter)을 읽는다 — 두 축을 나눈 이유는 resolve_query_embedding_provider 참고."""
    return embed_batch(
        [text],
        model_name=model_name,
        vector_dim=vector_dim,
        batch_size=1,
        device=device,
        provider=resolve_query_embedding_provider(provider),
    )[0]


def _get_tokenizer(model_name: str) -> Any | None:
    """임베딩 모델의 tokenizer 만 따로 로드한다(전체 모델 없이). 실패하면 None.

    API provider 로 색인하면 sentence-transformers 모델이 아예 안 올라오는데,
    청크 token_count 는 여전히 실제 tokenizer 로 세야 한다 — 그 값이 optimize 의
    chunk_size 처방 근거이고, 어림짐작으로 바뀌면 처방이 조용히 무뎌진다.

    성공이든 실패든 결과를 캐시한다. 실패를 캐시하지 않으면 청크마다 네트워크를
    두드려 색인이 기어간다."""
    if model_name in _tokenizers:
        return _tokenizers[model_name]
    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:
        print(
            f"[Index] '{model_name}' tokenizer 를 불러오지 못해 토큰 수를 어림짐작으로 "
            f"기록합니다 (optimize 의 chunk_size 처방 근거가 느슨해집니다): {exc}"
        )
    _tokenizers[model_name] = tokenizer
    return tokenizer


def count_tokens(
    text: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> int:
    # tokenizer가 있으면 그걸 쓰고, 없으면 대략적인 토큰 수로 기록한다.
    # 순수 dict 조회로만 캐시된 모델을 본다(부작용 없음). 프로덕션 유일 호출부
    # (index/agent.py)는 항상 embed 뒤에 불려 모델이 이미 캐시에 있으므로 실제
    # tokenizer를 쓴다. 여기서 _get_embedding_model 로 지연 로드하면 모델 없는
    # 환경에서 HF 다운로드를 유발해 토큰 세기 한 번이 수 초 블로킹된다(리뷰 #36).
    #
    # 캐시 키가 (model_name, device) 라 장치를 모르는 여기서는 이름이 같은 항목
    # 아무거나 쓴다 — tokenizer 는 장치와 무관하게 같다. 이름만으로 조회하면
    # 항상 캐시 미스가 나서 실제 tokenizer 대신 어림짐작으로 조용히 떨어진다
    # (그 토큰 수는 optimize 의 chunk_size 처방에 쓰인다).
    # list() 로 스냅샷을 뜬다 — 색인은 스레드로 병렬 호출하므로, 다른 스레드가
    # 모델을 로드하는 중이면 순회 도중 "dictionary changed size during iteration"
    # 이 난다. 토큰 세기 하나 때문에 문서 처리가 통째로 실패하면 안 된다.
    model = next(
        (value for (name, _device), value in list(_models.items())
         if name == model_name),
        None,
    )
    tokenizer = getattr(model, "tokenizer", None) if model is not None else None
    if tokenizer is None:
        # API provider 로 색인하면 로컬 모델이 아예 안 올라와 위 조회가 항상 빈다.
        # 그때는 tokenizer 만 따로 받는다 — 전체 모델(2.2GB)과 달리 수 MB 라
        # 리뷰 #36 이 막으려던 "토큰 세기 한 번이 수 초 블로킹" 에 해당하지 않는다.
        tokenizer = _get_tokenizer(model_name)
    if tokenizer is None:
        return max(1, len(_tokens(text)))
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return max(1, len(_tokens(text)))
