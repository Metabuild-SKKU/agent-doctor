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
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from agents.index.qdrant_store import (
    COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    VECTOR_DIM,
    build_client,
    collection_index_cache_key,
    delete_document_chunks,
    embed,
    ensure_collection,
    query_embedding_config_error,
    hybrid_search,
    keyword_search,
    rerank_with_status,
    reranker_max_length,
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
    rerank_candidates: int = 20
    # MMR(Maximal Marginal Relevance) 다양성 재정렬. 후보풀에서 관련성과 상호
    # 다양성을 mmr_lambda 로 균형해 top_k 를 고른다(나열형 질문의 중복 잠식 완화).
    use_mmr: bool = False
    mmr_lambda: float = 0.5
    mmr_candidates: int = 20
    qdrant_url: str = ":memory:"
    qdrant_api_key: str | None = None
    collection_name: str = COLLECTION
    index_cache_key: str = ""
    replace_collection: bool = False
    reuse_existing_collection: bool = False
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
    rerank_candidates = (
        _as_int(pick("rerank_candidates", 20), 20) or 20
    )
    mmr_candidates = _as_int(pick("mmr_candidates", 20), 20) or 20

    return RetrievalSettings(
        embedding_model=str(pick("embedding_model", DEFAULT_EMBEDDING_MODEL)),
        embedding_dimension=embedding_dimension,
        top_k=max(1, top_k),
        use_hybrid=_as_bool(pick("use_hybrid", False)),
        hybrid_dense_weight=float(pick("hybrid_dense_weight", 0.7)),
        use_reranker=_as_bool(pick("use_reranker", False)),
        reranker_model=str(
            pick("reranker_model", DEFAULT_RERANKER_MODEL)
            or DEFAULT_RERANKER_MODEL
        ),
        rerank_candidates=max(1, rerank_candidates),
        use_mmr=_as_bool(pick("use_mmr", False)),
        mmr_lambda=float(pick("mmr_lambda", 0.5)),
        mmr_candidates=max(1, mmr_candidates),
        qdrant_url=str(config.get("qdrant_url") or os.getenv("QDRANT_URL", ":memory:")),
        qdrant_api_key=config.get("qdrant_api_key") or os.getenv("QDRANT_API_KEY"),
        collection_name=str(pick("qdrant_collection_name", COLLECTION)),
        index_cache_key=str(pick("index_cache_key", "")),
        replace_collection=_as_bool(pick("replace_qdrant_collection", False)),
        reuse_existing_collection=_as_bool(
            pick("reuse_existing_collection", False)
        ),
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


def _cosine(a: list[float], b: list[float]) -> float:
    """두 임베딩의 코사인 유사도. 크기 0 이거나 차원 불일치면 0."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _dedup_by_chunk_id(results: list[dict]) -> list[dict]:
    """같은 chunk_id 는 첫 등장(=상위 순위)만 남긴다.

    중복이 남으면 같은 본문이 top_k 슬롯을 두 번 먹는다(실측: top-5 에 chunk_005 가 두 번
    잡혀 실제 후보가 4개). 리랭커·MMR 도 같은 쌍을 두 번 계산하고, Eval 의 recall·중복 신호가
    그만큼 왜곡된다. 융합(hybrid)은 chunk_id 로 접지만, 네이티브 RRF·keyword 폴백처럼 접지
    않는 경로가 있어 최종 결과 쪽에서 한 번 더 보장한다.

    id 가 비어 있는 항목(payload 결측 → _hit_to_result·keyword_search 가 "" 로 채움)은 접지
    않고 그대로 통과시킨다 — 서로 다른 청크인데 id 만 없는 것들이 "" 하나로 뭉쳐 조용히
    버려지는 걸 막는다(중복 제거의 근거는 '같은 id'이지 '둘 다 id 가 없음'이 아니다).
    """
    seen: set[str] = set()
    deduped = []
    for item in results:
        chunk_id = item.get("chunk_id")
        if chunk_id:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
        deduped.append(item)
    # 정상 경로에선 chunk_id 가 f"{doc_id}_chunk_{idx:03d}" 라 중복이 안 생긴다. 그래서 여기서
    # 뭔가 접혔다면 상류(Index) 쪽 중복이고, 조용히 접으면 그 원인이 로그에서 사라진다.
    if len(deduped) < len(results):
        print(f"[Retriever] 중복 chunk_id {len(results) - len(deduped)}건 제거 — 상류 중복 의심")
    return deduped


def _mmr_select(
    results: list[dict], top_k: int, lambda_: float, embeddings: list[list[float] | None]
) -> list[dict] | None:
    """MMR(Maximal Marginal Relevance)로 후보풀에서 top_k 를 고른다.

    각 단계에서 `lambda*관련성 - (1-lambda)*이미뽑힌것과의최대유사도` 가 가장 큰 후보를
    고른다. 관련성은 이미 계산된 검색 점수(score, [0,1] 로 정규화)를, 다양성은 청크
    임베딩의 코사인을 쓴다 — 질의 벡터에 의존하지 않아 dense/hybrid/keyword/rerank 결과
    모두에 동일하게 적용된다.

    embeddings 는 results 와 같은 순서의 청크 임베딩 리스트다(호출부가 chunk_id 로
    _chunks_by_id 에서 조회해 넘긴다 — 검색 결과 dict 에 embedding 을 싣지 않아도 되게).
    하나라도 없으면 다양성을 잴 수 없어 None 을 돌려주고, 호출부는 기존 점수 순서를
    그대로 쓴다(안전 폴백)."""
    if len(embeddings) != len(results) or any(not emb for emb in embeddings):
        return None
    scores = [float(r.get("score") or 0.0) for r in results]
    hi = max(scores) if scores else 0.0
    lo = min(scores) if scores else 0.0
    span = hi - lo
    rel = [(s - lo) / span if span > 0 else 1.0 for s in scores]

    remaining = list(range(len(results)))
    selected: list[int] = []
    while remaining and len(selected) < top_k:
        best_idx = None
        best_score = None
        for i in remaining:
            diversity = max(
                (_cosine(embeddings[i], embeddings[j]) for j in selected),
                default=0.0,
            )
            mmr = lambda_ * rel[i] - (1.0 - lambda_) * diversity
            if best_score is None or mmr > best_score:
                best_score, best_idx = mmr, i
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [results[i] for i in selected]


class Retriever:
    """A small, reusable retrieval facade with dense/hybrid/rerank/fallback."""

    def __init__(
        self,
        chunks: list[Chunk | dict],
        settings: RetrievalSettings,
        client: QdrantClient | None = None,
    ) -> None:
        self.settings = settings
        # 같은 chunk_id 가 두 번 들어오면 lexical 경로(keyword_search·BM25 융합)가 같은 본문을
        # 두 번 후보로 올려 top_k 슬롯을 먹는다. 검색 결과 쪽에서 접으면 자른 뒤라 후보가
        # 그만큼 비므로, 입력에서 한 번 접어 슬롯 수를 유지한다.
        self.chunks = _dedup_by_chunk_id([_chunk_to_dict(chunk) for chunk in chunks])
        self.client = client
        self.chunk_ids = {
            chunk.get("chunk_id", "")
            for chunk in self.chunks
            if chunk.get("chunk_id")
        }
        self._chunks_by_id = {
            chunk.get("chunk_id", ""): chunk
            for chunk in self.chunks
            if chunk.get("chunk_id")
        }
        self.retrieval_scope_id = (
            _first_metadata(self.chunks).get("retrieval_scope_id")
            if self.chunks
            else None
        )
        self._dense_only: "Retriever | None" = None

    def dense_only_view(self) -> "Retriever | None":
        """같은 인덱스를 dense 단일 채널로만 보는 Retriever. 하이브리드가 아니면 None.

        Eval 이 융합 손실(한 채널은 gold 를 상위에 뒀는데 융합이 밀어냄)을 판정하려면 융합 전
        채널별 순위가 필요하다. 설정만 바꾼 뷰라 임베딩·컬렉션은 그대로 공유하고, probe 마다
        새로 만들지 않도록 여기서 캐시한다(생성 비용은 청크 dict 복사뿐이지만 코퍼스가 크면
        그것도 probe 수만큼 반복된다).

        캐시에 락을 걸지 않는다 — 동시 호출이면 뷰가 중복 생성될 수 있지만 생성이 순수하고
        (모델 로드 없이 dict 복사뿐) 결과가 동등해서 무해하다. Eval 은 검색을 순차 구간에서만
        돌리므로(LLM 호출만 병렬) 실제로는 경합이 생기지 않는다.
        """
        if not self.settings.use_hybrid:
            return None                  # 융합이 없으면 대조할 채널도 없다
        if self._dense_only is None:
            self._dense_only = Retriever(
                self.chunks,
                replace(self.settings, use_hybrid=False),
                client=self.client,
            )
        return self._dense_only

    def search(
        self,
        query: str,
        top_k: int | None = None,
        apply_rerank: bool | None = None,
    ) -> list[dict]:
        return self.search_with_details(
            query, top_k=top_k, apply_rerank=apply_rerank
        )["results"]

    def _vector_candidate_k(self, candidate_k: int) -> int:
        if not self.chunk_ids:
            return candidate_k
        return max(candidate_k, min(max(len(self.chunks), candidate_k * 8), 200))

    def _current_results(self, results: list[dict]) -> list[dict]:
        """검색 결과를 현재 청크 집합 기준으로 좁히고 payload 를 현재 값으로 덮는다."""
        if not self.chunk_ids:
            return _dedup_by_chunk_id(results)
        current_results = []
        for item in _dedup_by_chunk_id(results):
            chunk = self._chunks_by_id.get(item.get("chunk_id"))
            if chunk is None:
                continue
            current_results.append(
                {
                    **item,
                    "doc_id": chunk.get("doc_id", ""),
                    "text": chunk.get("text", ""),
                    "section": chunk.get("section"),
                    "char_span": chunk.get("char_span"),
                    "token_count": chunk.get("token_count"),
                    "parent_id": chunk.get("parent_id"),
                    "hash": chunk.get("hash"),
                    "metadata": chunk.get("metadata", {}),
                }
            )
        return current_results

    """
    1) 빈 query -> 빈 결과
    2) top_k 결정
    3) reranker 쓰면 설정된 rerank_candidates만큼 후보를 가져옴
    4) Qdrant client -> dense/hybrid 검색
    5) 실패 or 결과 X -> keyword fallback
    6) reranker = True면 재정렬
    7) 최종 result 반환

    apply_rerank 로 리랭크 단계만 끌 수 있다(기본은 설정값). Eval 의 순위 측정이
    "리랭크 이전 융합 순위"를 재야 하기 때문이다 — 자세한 배경은 아래 주석 참고.
    """
    def search_with_details(
        self,
        query: str,
        top_k: int | None = None,
        apply_rerank: bool | None = None,
    ) -> dict:
        # 검색 한 건의 총시간. 리랭크 시간(rerank_seconds)과 함께 실어야 '느린 게 검색이냐
        # 생성이냐, 검색이면 그중 리랭크가 얼마냐'가 한 번의 실행으로 갈린다 — 지금까지는
        # 분자(리랭크)만 있고 분모(검색 총시간)가 없어 콘솔 로그를 눈으로 대조해야 했다.
        search_started = time.monotonic()
        use_reranker = (
            self.settings.use_reranker if apply_rerank is None else bool(apply_rerank)
        )
        if not query.strip():
            return {
                "query": query,
                "search_mode": "none",
                "reranker_enabled": use_reranker,
                "reranker_attempted": False,
                "reranked": False,
                "reranker_status": (
                    "not_attempted"
                    if use_reranker
                    else "disabled"
                ),
                "reranker_fallback_used": False,
                "mmr_enabled": self.settings.use_mmr,
                "mmr_applied": False,
                "search_fallback_used": False,
                "fallback_used": False,
                # 정상 경로와 키 집합을 맞춘다 — 소비처(Eval 진단)가 키 유무로 분기한다.
                "rerank_candidate_count": 0,
                "pre_rerank_ids": [],
                "rerank_seconds": 0.0,
                "rerank_pairs": 0,
                "search_seconds": round(time.monotonic() - search_started, 3),
                "results": [],
            }

        requested_top_k = max(1, int(top_k or self.settings.top_k))
        # 리랭커·MMR 은 top_k 보다 넓은 후보풀이 있어야 재정렬·다양화를 할 수 있다.
        # 리랭크 몫은 use_reranker(= apply_rerank 반영)로 본다 — 순위 측정용 호출에서
        # 리랭크를 껐으면 그만큼 넓은 풀이 필요 없다. MMR 은 그 override 와 무관하게 돈다.
        pool_sizes = [requested_top_k]
        if use_reranker:
            pool_sizes.append(self.settings.rerank_candidates)
        if self.settings.use_mmr:
            pool_sizes.append(self.settings.mmr_candidates)
        candidate_k = max(pool_sizes)
        vector_candidate_k = self._vector_candidate_k(candidate_k)
        results: list[dict] = []
        mode = "keyword"
        fallback_used = self.client is None

        if self.client is not None:
            # 설정 오류는 폴백 대상이 아니다. 아래 except 가 잡는 건 "API 가 잠깐
            # 흔들린다" 이고 그건 keyword 로 내리는 게 맞지만, 키를 안 넣은 실행까지
            # 같이 흡수하면 모든 질의가 영구히 keyword 로 돌면서 증상은 "검색 품질이
            # 좀 나쁘다" 로만 보인다. try 밖에서 먼저 끊는다.
            config_error = query_embedding_config_error()
            if config_error:
                raise RuntimeError(f"[Retriever] {config_error}")
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
                        collection_name=self.settings.collection_name,
                    )
                else:
                    mode = "dense"
                    results = dense_search(
                        self.client,
                        query_vector,
                        top_k=vector_candidate_k,
                        retrieval_scope_id=self.retrieval_scope_id,
                        collection_name=self.settings.collection_name,
                    )
                results = self._current_results(results)
                if use_reranker:
                    results = results[:candidate_k]
            except Exception as exc:
                print(f"[Retriever] vector search failed, using keyword fallback: {exc}")
                results = []
                fallback_used = True

        if not results:
            mode = "keyword"
            fallback_used = True
            # self.chunks 가 이미 chunk_id 로 접혀 있어 여기서 중복이 생기지 않는다
            # (접는 자리를 입력으로 올린 이유는 __init__ 주석 참고 — 자른 뒤 접으면 슬롯이 빈다).
            results = keyword_search(self.chunks, query, top_k=candidate_k)

        # 리랭크 직전 후보 순서. Eval 이 "리랭커가 gold 를 봤나(후보창 안)"와
        # "보고도 떨어뜨렸나(강등)"를 가르는 유일한 신호라 결과에 함께 싣는다.
        # 리랭크가 results 를 덮어쓰므로 이 시점에 떠 두지 않으면 복원할 수 없다.
        # candidate_k 로 자르는 게 중요하다 — 벡터 검색은 후보를 최대 200개까지 넉넉히
        # 가져오는데(_vector_candidate_k), 리랭커가 실제로 받는 건 앞 candidate_k 개뿐이다.
        # 안 자르면 '리랭커가 본 목록'이 아니게 되고, 리랭커가 꺼진 검색에서도 200개짜리
        # 목록이 매 record 에 실려 state 스냅샷·Eval 캐시가 불어난다.
        pre_rerank_ids = [item.get("chunk_id", "") for item in results[:candidate_k]]

        reranker_attempted = bool(use_reranker and results)
        reranked = False
        reranker_status = (
            "not_attempted"
            if use_reranker
            else "disabled"
        )
        # MMR 이 최종 top_k 선택을 맡으면 리랭커는 후보풀을 유지한다(그래야 다양화 여지가 남음).
        rerank_top_k = candidate_k if self.settings.use_mmr else requested_top_k
        # 리랭크 실측 시간·쌍 수를 결과에 싣는다 — 검색 시간의 대부분이 여기서 나오는데
        # (쌍 수 × 쌍당 텍스트 길이), 지금까지 리포트에는 실행 여부만 있고 비용이 없었다.
        # chunk_size 처방으로 쌍당 비용이 뛰어도 Optimize 가 그걸 못 보고 품질만 비교한다.
        # 시간·쌍 수는 리랭크가 실제로 돈 경우(applied)만, 반드시 **함께** 센다. 둘의 기준이
        # 어긋나면 집계 ms_per_pair(=Σseconds/Σpairs)가 양쪽으로 왜곡된다:
        #   로드 실패·쿨다운 — predict 가 아예 안 돌아 시간 0, 쌍만 잡히면 아래로 희석
        #   inference_failed — predict 는 끝까지 돌고 결과 검증에서 실패, 시간만 잡히면 위로 부풀림
        # 실패한 시도의 벽시계는 버린다(그 사실은 reranker_status/failed 집계가 이미 알린다) —
        # 이 지표의 질문은 '실제로 재정렬한 1쌍이 얼마였나'뿐이다.
        rerank_seconds = 0.0
        rerank_pairs = 0
        if reranker_attempted:
            attempted_pairs = len(results)
            # 시간은 rerank_with_status 가 추론 구간만 재서 돌려준다 — 여기서 감싸면 모델
            # 로드가 섞이고, 사전 로드를 따로 부르면 그게 별도 seam 이 돼 이 함수를 패치한
            # 테스트가 실모델을 내려받는다(둘 다 겪었다).
            results, reranker_status, measured = rerank_with_status(
                query,
                results,
                model_name=self.settings.reranker_model,
                top_k=rerank_top_k,
            )
            reranked = reranker_status == "applied"
            rerank_seconds = measured if reranked else 0.0
            rerank_pairs = attempted_pairs if reranked else 0

        mmr_applied = False
        if self.settings.use_mmr and len(results) > requested_top_k:
            # 임베딩은 검색 결과 dict 가 아니라 원본 청크(_chunks_by_id)에서 chunk_id 로
            # 조회한다 — keyword/dense/hybrid/rerank 어느 경로든 결과에 embedding 을 싣지
            # 않으므로(과거 MMR 이 항상 no-op 이던 원인) 여기서 확실히 붙여 넘긴다.
            embeddings = [
                (self._chunks_by_id.get(r.get("chunk_id")) or {}).get("embedding")
                for r in results
            ]
            selected = _mmr_select(
                results, requested_top_k, self.settings.mmr_lambda, embeddings
            )
            if selected is not None:
                results = selected
                mmr_applied = True
        results = results[:requested_top_k]

        return {
            "query": query,
            "search_mode": mode,
            "reranker_enabled": use_reranker,
            "reranker_attempted": reranker_attempted,
            "reranked": reranked,
            "reranker_status": reranker_status,
            "reranker_fallback_used": (
                reranker_attempted and reranker_status != "applied"
            ),
            "mmr_enabled": self.settings.use_mmr,
            "mmr_applied": mmr_applied,
            "search_fallback_used": fallback_used,
            "fallback_used": fallback_used,
            "rerank_candidate_count": candidate_k,
            "pre_rerank_ids": pre_rerank_ids,
            "rerank_seconds": round(rerank_seconds, 3),
            "rerank_pairs": rerank_pairs,
            # 상한이 실제로 걸린 채 돌았는지 — 폴백으로 빠진 실행과 구분해야 다음 처방 판정이
            # capped/uncapped 를 같은 실행으로 묶지 않는다.
            "rerank_max_length": (
                reranker_max_length(self.settings.reranker_model) if reranked else None
            ),
            "search_seconds": round(time.monotonic() - search_started, 3),
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
            existing = {
                item.name
                for item in client.get_collections().collections
            }
            reuse_requested = (
                settings.reuse_existing_collection
                and settings.collection_name in existing
            )
            stored_index_key = (
                collection_index_cache_key(
                    client,
                    collection_name=settings.collection_name,
                )
                if reuse_requested
                else None
            )
            if (
                reuse_requested
                and settings.index_cache_key
                and stored_index_key == settings.index_cache_key
            ):
                ensure_collection(
                    client,
                    vector_dim=vector_dim,
                    recreate_on_mismatch=False,
                    collection_name=settings.collection_name,
                )
                return raw_chunks, client, True
            if settings.replace_collection or reuse_requested:
                if settings.collection_name in existing:
                    client.delete_collection(
                        collection_name=settings.collection_name
                    )
            ensure_collection(
                client,
                vector_dim=vector_dim,
                recreate_on_mismatch=settings.recreate_collection_on_dimension_mismatch,
                collection_name=settings.collection_name,
            )
            if delete_doc_ids:
                delete_document_chunks(
                    client,
                    list(delete_doc_ids),
                    retrieval_scope_id=scope_id,
                    collection_name=settings.collection_name,
                )
            upsert_chunks(
                client,
                embedded,
                collection_name=settings.collection_name,
            )
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
_MAX_CACHED_INDEXES = 2
_cached_entries: OrderedDict[
    tuple,
    tuple[list[dict], QdrantClient | None, str],
] = OrderedDict()


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
        settings.index_cache_key or _legacy_payload_signature(raw_chunks),
        settings.embedding_model,
        _first_embedding_dim(raw_chunks) or settings.embedding_dimension,
        settings.qdrant_url,
        settings.qdrant_api_key,
        settings.collection_name,
    )


def _legacy_payload_signature(raw_chunks: list[dict]) -> str:
    """논리 인덱스 키가 없는 옛 청크는 Qdrant payload 전체로 충돌을 막는다."""
    raw = json.dumps(
        sorted(raw_chunks, key=lambda item: item.get("chunk_id", "")),
        sort_keys=True,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _evict_oldest_index() -> None:
    """가장 오래된 프로세스 내 적재 캐시만 제거한다.

    임의의 custom collection은 다른 파이프라인 소유일 수 있어 여기서 삭제하지
    않는다. Index가 관리하는 고정 슬롯은 새 버전을 쓸 때 _populate가 정확한
    대상 슬롯만 교체한다.
    """
    if not _cached_entries:
        return
    _cached_entries.popitem(last=False)


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
    settings = resolve_retrieval_settings(chunks, config)
    raw_chunks = [_chunk_to_dict(chunk) for chunk in chunks]
    scope_id = _scope_id(raw_chunks)
    key = _population_key(raw_chunks, scope_id, settings)

    with _cache_lock:
        if key in _cached_entries:
            _, cached_client, _ = _cached_entries.pop(key)
            if any(chunk.get("embedding") for chunk in raw_chunks):
                raw_chunks = _with_scope(raw_chunks, scope_id)
            _cached_entries[key] = (
                raw_chunks,
                cached_client,
                settings.collection_name,
            )
        else:
            # 고정 2-slot의 한쪽을 새 버전으로 덮어쓸 때는 그 물리 슬롯의
            # 이전 캐시 항목만 제거한다. 단순 LRU로 baseline을 먼저 지우면
            # 같은 Optimize 방문의 후속 처방이 다시 실패했을 때 롤백할 수 없다.
            for cached_key, payload in list(_cached_entries.items()):
                if payload[2] == settings.collection_name:
                    _cached_entries.pop(cached_key)
            if len(_cached_entries) >= _MAX_CACHED_INDEXES:
                _evict_oldest_index()
            raw_chunks, cached_client, cacheable = _populate(
                raw_chunks, scope_id, settings, client, delete_doc_ids
            )
            if cacheable:
                _cached_entries[key] = (
                    raw_chunks,
                    cached_client,
                    settings.collection_name,
                )

    return Retriever(raw_chunks, settings, client=cached_client)


def reset_retriever_cache() -> None:
    """적재 캐시를 비운다(테스트·장기 실행 프로세스에서 인덱스를 강제로 다시 만들 때)."""
    with _cache_lock:
        _cached_entries.clear()

# serve 쪽에서 chunk.json 읽을 때 helper
def load_chunks(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
