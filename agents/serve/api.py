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
# 지문·서빙설정 로직은 Serve agent 와 공유한다(agents/serve/*.py) — agent.py 가 이 모듈
# (uvicorn/fastapi/qdrant_client)을 import 하지 않도록 순수 모듈로 분리.
from agents.serve.serve_config import (
    read_serve_config, serving_fingerprint, generation_subset,
)

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
# Optimize(B그룹)가 고른 생성 설정. init_qdrant 가 서빙 설정 사이드카에서 읽어 주입한다.
# 비어 있으면(사이드카 없음) generator 기본값(현 동작)을 쓴다.
_generation_config: dict = {}


def configure_generation(generation_config: dict | None) -> None:
    """서빙에 쓸 generation_config를 설정한다(파이프라인 Serve 단계에서 호출)."""
    global _generation_config
    _generation_config = dict(generation_config or {})

_GENERATION_CONFIG_KEYS = (
    "context_compression",
    "context.compression.enabled",
    "context_compression_max_contexts",
    "context_filter_max_contexts",
    "context_compression_min_contexts",
    "context_filter_min_contexts",
    "context_compression_max_sentences",
    "context_filter_max_sentences",
)


def init_qdrant(chunks_file: str) -> None:
    global _retriever, _chunks_raw, _chunks_file, _fingerprint

    _chunks_file = chunks_file
    _chunks_raw = json.loads(Path(chunks_file).read_text(encoding="utf-8"))
    # 파이프라인이 고른 검색·생성 설정을 사이드카에서 읽어 retriever/generator 에 주입한다.
    # (없으면 빈 dict → 기본 동작). 지문에 설정을 포함해, 설정만 바뀌어도 reload 가 걸린다.
    serve_config = read_serve_config(chunks_file)
    _fingerprint = serving_fingerprint(_chunks_raw, serve_config)
    configure_generation(generation_subset(serve_config))
    print(f"[API] loaded {len(_chunks_raw)} chunks (fingerprint {_fingerprint}), "
          f"serve_config {len(serve_config)}개 키")

    _retriever = get_retriever(
        _chunks_raw,
        {
            # 최적화된 top_k·reranker·mmr·hybrid 등(get_retriever 가 아는 키만 사용).
            **serve_config,
            "qdrant_url": os.getenv("QDRANT_URL", ":memory:"),
            "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
        },
    )
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
        "use_mmr": settings.use_mmr,
        "mmr_lambda": settings.mmr_lambda,
        "recreate_collection_on_dimension_mismatch": (
            settings.recreate_collection_on_dimension_mismatch
        ),
    }


def _answer_generation_config(retriever: Retriever) -> dict:
    settings = retriever.settings
    config = _stored_generation_config()
    config.update(_generation_config)
    env_context_compression = os.getenv("RAG_CONTEXT_COMPRESSION")
    if env_context_compression is not None:
        config["context_compression"] = env_context_compression
        config["context.compression.enabled"] = env_context_compression

    enabled = config.get("context_compression")
    if enabled is None:
        enabled = config.get("context.compression.enabled")
    if enabled is not None:
        config["context_compression"] = enabled
        config["context.compression.enabled"] = enabled

    config.update(
        {
        "top_k": settings.top_k,
        "use_hybrid": settings.use_hybrid,
        "use_reranker": settings.use_reranker,
        }
    )
    return config


def _stored_generation_config() -> dict:
    config: dict = {}
    for chunk in _chunks_raw:
        metadata = chunk.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            continue
        for key in _GENERATION_CONFIG_KEYS:
            value = metadata.get(key)
            if value is not None and key not in config:
                config[key] = value
    return config


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": len(_chunks_raw),
        "qdrant": _retriever is not None and _retriever.client is not None,
        "fingerprint": _fingerprint,
        "index_settings": _public_index_settings(_retriever),
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
    return answer_question(
        query,
        retriever,
        top_k=top_k,
        config=_answer_generation_config(retriever),
    )


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
