"""FastAPI server for Agent Doctor retrieval and RAG answers.

Run:
  python agents/serve/api.py --chunks-file chunks.json --port 8766
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.rag.generator import answer_question
from agents.rag.retriever import Retriever, get_retriever, load_chunks

app = FastAPI(title="Agent Doctor API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_retriever: Retriever | None = None
_chunks_raw: list[dict] = []


def init_qdrant(chunks_file: str) -> None:
    """Load serialized chunks and prepare the shared retriever."""
    global _retriever, _chunks_raw

    _chunks_raw = load_chunks(chunks_file)
    print(f"[API] loaded {len(_chunks_raw)} chunks")
    _retriever = get_retriever(_chunks_raw)
    if _retriever.client is None:
        print("[API] Qdrant unavailable or embeddings missing; keyword fallback enabled")
    else:
        print("[API] RAG retriever ready")


def _require_retriever() -> Retriever:
    if _retriever is None or not _chunks_raw:
        raise HTTPException(status_code=503, detail="indexed chunks are not loaded")
    return _retriever


def _public_index_settings(retriever: Retriever | None) -> dict:
    if retriever is None:
        return {}
    settings = retriever.settings
    return {
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "top_k": settings.top_k,
        "use_hybrid": settings.use_hybrid,
        "hybrid_dense_weight": settings.hybrid_dense_weight,
        "use_reranker": settings.use_reranker,
        "reranker_model": settings.reranker_model,
        "recreate_collection_on_dimension_mismatch": (
            settings.recreate_collection_on_dimension_mismatch
        ),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": len(_chunks_raw),
        "qdrant": bool(_retriever and _retriever.client is not None),
        "index_settings": _public_index_settings(_retriever),
    }


@app.get("/search")
def search(query: str, top_k: int | None = None):
    """Search indexed chunks using the same retriever that Eval/RAG use."""
    retriever = _require_retriever()
    if not query.strip():
        raise HTTPException(status_code=400, detail="query must not be blank")
    if top_k is not None and top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be positive")
    return retriever.search_with_details(query, top_k=top_k)


@app.get("/answer")
def answer(query: str, top_k: int | None = None):
    """Search documents and generate a grounded answer."""
    retriever = _require_retriever()
    if not query.strip():
        raise HTTPException(status_code=400, detail="query must not be blank")
    if top_k is not None and top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be positive")
    return answer_question(query, retriever, top_k=top_k)


@app.get("/documents")
def documents():
    """Return indexed document list."""
    seen = {}
    for chunk in _chunks_raw:
        doc_id = chunk.get("doc_id", "")
        seen[doc_id] = chunk.get("metadata", {}).get("title", doc_id)
    return {
        "total": len(seen),
        "documents": [{"doc_id": doc_id, "title": title} for doc_id, title in seen.items()],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-file", required=True)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    init_qdrant(args.chunks_file)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
