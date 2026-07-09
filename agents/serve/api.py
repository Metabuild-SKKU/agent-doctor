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
    build_client, ensure_collection, embed, search as qdrant_search, COLLECTION,
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


# ── 시작 시 Qdrant 초기화 ─────────────────────────────────────────

def init_qdrant(chunks_file: str) -> None:
    global _client, _chunks_raw

    _chunks_raw = json.loads(Path(chunks_file).read_text(encoding="utf-8"))
    print(f"[API] 청크 {len(_chunks_raw)}개 로드")

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
    }


@app.get("/search")
def search(query: str, top_k: int = 3):
    """쿼리와 유사한 청크 반환."""
    if not _chunks_raw:
        raise HTTPException(status_code=503, detail="인덱싱된 문서가 없습니다.")

    results = []

    if _client:
        # 벡터 검색
        try:
            query_vec = embed(query)
            results = qdrant_search(_client, query_vec, top_k=top_k)
        except Exception as e:
            print(f"[API] 벡터 검색 실패: {e} → 키워드 검색으로 폴백", flush=True)
            results = []

    if not results:
        # 키워드 검색 (폴백)
        q = query.lower()
        scored = []
        for c in _chunks_raw:
            text = c.get("text", "")
            score = sum(1 for w in q.split() if w in text.lower())
            if score > 0:
                scored.append({"text": text, "metadata": c.get("metadata", {}), "score": float(score)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:top_k]

    return {"query": query, "results": results}


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
