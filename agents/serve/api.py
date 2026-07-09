"""
FastAPI 검색 서버 — Qdrant 검색 결과를 HTTP로 제공
MCP 서버와 챗봇 UI 양쪽에서 이 API를 호출함.

실행:
  python agents/serve/api.py --chunks-file chunks.json --port 8766
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client.models import PointStruct

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.index.qdrant_store import (
    COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    build_client,
    embed,
    ensure_collection,
    hybrid_search,
    rerank,
    search as qdrant_search,
)

# ── FastAPI 앱 ────────────────────────────────────────────────────

app = FastAPI(title="Agent Doctor API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 챗봇 UI 연결 허용
    allow_methods=["*"],
    allow_headers=["*"],
)

_client = None
_chunks_raw: list[dict] = []
_index_settings: dict = {}


# ── 시작 시 Qdrant 초기화 ─────────────────────────────────────────

def init_qdrant(chunks_file: str) -> None:
    global _client, _chunks_raw, _index_settings

    _chunks_raw = json.loads(Path(chunks_file).read_text(encoding="utf-8"))
    print(f"[API] 청크 {len(_chunks_raw)}개 로드")
    first_metadata = _chunks_raw[0].get("metadata", {}) if _chunks_raw else {}
    _index_settings = {
        "embedding_model": first_metadata.get(
            "embedding_model", DEFAULT_EMBEDDING_MODEL
        ),
        "embedding_dimension": first_metadata.get("embedding_dimension"),
        "top_k": int(first_metadata.get("top_k", 5)),
        "use_hybrid": bool(first_metadata.get("use_hybrid", False)),
        "hybrid_dense_weight": float(
            first_metadata.get("hybrid_dense_weight", 0.7)
        ),
        "use_reranker": bool(first_metadata.get("use_reranker", False)),
        "reranker_model": first_metadata.get(
            "reranker_model", DEFAULT_RERANKER_MODEL
        ),
    }

    has_embedding = any(c.get("embedding") for c in _chunks_raw)
    if not has_embedding:
        print("[API] 임베딩 없음 → 키워드 검색 모드")
        return

    url = os.getenv("QDRANT_URL", ":memory:")
    key = os.getenv("QDRANT_API_KEY", None)
    client = build_client(url=url, api_key=key)

    vector_dim = len(next(c["embedding"] for c in _chunks_raw if c.get("embedding")))
    ensure_collection(client, vector_dim=vector_dim)

    points = [
        PointStruct(
            id=i,
            vector=c["embedding"],
            payload={
                "text":     c.get("text", ""),
                "doc_id":   c.get("doc_id", ""),
                "chunk_id": c.get("chunk_id", ""),
                "metadata": c.get("metadata", {}),
            },
        )
        for i, c in enumerate(_chunks_raw)
        if c.get("embedding")
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    _client = client
    print(f"[API] Qdrant 준비 완료 ({len(points)}개 벡터)")


# ── 엔드포인트 ────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": len(_chunks_raw),
        "qdrant": _client is not None,
        "index_settings": _index_settings,
    }


@app.get("/search")
def search(query: str, top_k: int | None = None):
    """Index에서 저장한 설정과 같은 모델·검색 방식으로 청크를 찾는다."""
    if not _chunks_raw:
        raise HTTPException(status_code=503, detail="인덱싱된 문서가 없습니다.")
    if not query.strip():
        raise HTTPException(status_code=400, detail="query가 비어 있습니다.")

    requested_top_k = top_k or int(_index_settings.get("top_k", 5))
    if requested_top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k는 1 이상이어야 합니다.")
    candidate_k = requested_top_k * 4 if _index_settings.get("use_reranker") else requested_top_k
    results = []

    if _client:
        try:
            query_vec = embed(
                query,
                model_name=_index_settings.get(
                    "embedding_model", DEFAULT_EMBEDDING_MODEL
                ),
                vector_dim=_index_settings.get("embedding_dimension"),
            )
            if _index_settings.get("use_hybrid"):
                results = hybrid_search(
                    _client,
                    query_vector=query_vec,
                    query=query,
                    chunks=_chunks_raw,
                    top_k=candidate_k,
                    dense_weight=_index_settings.get("hybrid_dense_weight", 0.7),
                )
            else:
                results = qdrant_search(_client, query_vec, top_k=candidate_k)
        except Exception as e:
            print(f"[API] 벡터 검색 실패: {e} → 키워드 검색으로 폴백", flush=True)
            results = []

    if not results:
        q = query.lower()
        scored = []
        for c in _chunks_raw:
            text = c.get("text", "")
            score = sum(1 for w in q.split() if w in text.lower())
            if score > 0:
                scored.append(
                    {
                        "chunk_id": c.get("chunk_id", ""),
                        "doc_id": c.get("doc_id", ""),
                        "section": c.get("section"),
                        "text": text,
                        "metadata": c.get("metadata", {}),
                        "score": float(score),
                    }
                )
        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:candidate_k]

    if _index_settings.get("use_reranker"):
        results = rerank(
            query,
            results,
            model_name=_index_settings.get(
                "reranker_model", DEFAULT_RERANKER_MODEL
            ),
            top_k=requested_top_k,
        )
    else:
        results = results[:requested_top_k]

    return {
        "query": query,
        "search_mode": (
            "hybrid" if _index_settings.get("use_hybrid") else "dense"
        ),
        "reranked": bool(_index_settings.get("use_reranker")),
        "results": results,
    }


@app.get("/documents")
def documents():
    """인덱싱된 문서 목록 반환."""
    seen = {}
    for c in _chunks_raw:
        doc_id = c.get("doc_id", "")
        seen[doc_id] = c.get("metadata", {}).get("title", doc_id)
    return {"total": len(seen), "documents": [{"doc_id": k, "title": v} for k, v in seen.items()]}


# ── 진입점 ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-file", required=True)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    init_qdrant(args.chunks_file)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
