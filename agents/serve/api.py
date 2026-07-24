"""FastAPI server for Agent Doctor retrieval and RAG answers.

Run:
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.rag.generator import answer_question
from agents.rag.retriever import Retriever, get_retriever
# 지문 알고리즘은 Serve agent 와 공유한다(agents/serve/fingerprint.py) — agent.py 가
# 지문 하나 때문에 이 모듈(uvicorn/fastapi/qdrant_client)을 import 하지 않도록 분리.
from agents.serve.fingerprint import corpus_fingerprint

app = FastAPI(title="Agent Doctor API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_retriever: Retriever | None = None
_chunks_raw: list[dict] = []
_chunks_file: str | None = None
_fingerprint: str = ""


def init_qdrant(chunks_file: str) -> None:
    global _retriever, _chunks_raw, _chunks_file, _fingerprint

    _chunks_file = chunks_file
    _chunks_raw = json.loads(Path(chunks_file).read_text(encoding="utf-8"))
    _fingerprint = corpus_fingerprint(_chunks_raw)
    print(f"[API] loaded {len(_chunks_raw)} chunks (fingerprint {_fingerprint})")

    _retriever = get_retriever(
        _chunks_raw,
        {
            "qdrant_url": os.getenv("QDRANT_URL", ":memory:"),
            "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
        },
    )
    if _retriever.client is None:
        print("[API] Qdrant unavailable or embeddings missing; using keyword fallback")
    else:
        print("[API] RAG retriever ready")


def _require_retriever() -> Retriever:
    if not _chunks_raw or _retriever is None:
        raise HTTPException(status_code=503, detail="No indexed documents are loaded.")
    return _retriever


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": len(_chunks_raw),
        "qdrant": _retriever is not None and _retriever.client is not None,
        "fingerprint": _fingerprint,
    }


@app.post("/reload")
def reload():
    """chunks.json 을 다시 읽어 retriever 를 재구축한다.

    Serve(agent.py)가 새 코퍼스를 chunks.json 에 쓴 뒤, 이미 실행 중인 이 서버가
    낡은 코퍼스를 서빙하고 있을 때 호출한다. init_qdrant 를 그대로 재실행해
    _chunks_raw·_retriever·_fingerprint 를 최신 파일 내용으로 교체한다."""
    if not _chunks_file:
        raise HTTPException(status_code=503, detail="No chunks file to reload from.")
    init_qdrant(_chunks_file)
    return {"status": "reloaded", "chunks": len(_chunks_raw), "fingerprint": _fingerprint}


@app.get("/search")
def search(query: str, top_k: int = 3):
    """Return chunks relevant to the query."""
    retriever = _require_retriever()
    retrieval = retriever.search_with_details(query, top_k=top_k)
    return {
        "query": query,
        "results": retrieval["results"],
        "search_mode": retrieval["search_mode"],
        "fallback_used": retrieval["fallback_used"],
        "reranked": retrieval["reranked"],
    }


@app.get("/answer")
def answer(query: str, top_k: int = 3):
    """Run retrieval plus answer generation and return the RAG payload."""
    retriever = _require_retriever()
    return answer_question(query, retriever, top_k=top_k)


@app.get("/documents")
def documents():
    """Return indexed document ids and titles."""
    seen = {}
    for chunk in _chunks_raw:
        doc_id = chunk.get("doc_id", "")
        seen[doc_id] = chunk.get("metadata", {}).get("title", doc_id)
    return {
        "total": len(seen),
        "documents": [
            {"doc_id": doc_id, "title": title}
            for doc_id, title in seen.items()
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-file", required=True)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    init_qdrant(args.chunks_file)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
