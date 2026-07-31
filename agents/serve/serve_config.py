"""서빙 설정 사이드카 — Serve agent(쓰기)와 API 서버(읽기)가 공유하는 순수 모듈.

API 는 별도 프로세스로 `--chunks-file` 만 받아 기동하므로, 파이프라인이 고른 검색·생성
설정(top_k·reranker·MMR·generation 플래그 등)이 부모 프로세스 전역값으로는 전달되지
않는다. 그래서 chunks.json 옆에 서빙 관련 설정 subset 을 사이드카 JSON 으로 남기고,
API 가 기동/reload 시 이를 읽어 retriever 구성과 generation 설정에 주입한다.

fingerprint.py 처럼 순수 함수만 두어 Serve agent 가 uvicorn/fastapi/qdrant 를 import
하지 않고도 이 모듈을 쓸 수 있게 한다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agents.serve.fingerprint import corpus_fingerprint

# 서빙 검색을 좌우하는 index_config 키(청킹처럼 이미 청크에 구운 재색인 키는 제외).
# embedding_model/dimension 은 질의 임베딩이 색인과 같아야 하므로 포함한다.
RETRIEVAL_KEYS: tuple[str, ...] = (
    "top_k",
    "use_hybrid",
    "hybrid_dense_weight",
    "use_reranker",
    "reranker_model",
    "rerank_candidates",
    "use_mmr",
    "mmr_lambda",
    "mmr_candidates",
    "embedding_model",
    "embedding_dimension",
)

# generator 가 프롬프트/온도로 소비하는 생성 설정 키(B그룹 Tier1).
GENERATION_KEYS: tuple[str, ...] = (
    "temperature",
    "grounding_strict",
    "require_citation",
    "restate_question",
    "completeness_mode",
    "abstention_strict",
    "generation_model",
    "context_compression",
    "context.compression.enabled",
    "context_compression_max_contexts",
    "context_filter_max_contexts",
    "context_compression_min_contexts",
    "context_filter_min_contexts",
    "context_compression_max_sentences",
    "context_filter_max_sentences",
)

SERVE_CONFIG_KEYS: tuple[str, ...] = RETRIEVAL_KEYS + GENERATION_KEYS


def extract_serve_config(index_config: dict | None) -> dict:
    """index_config 에서 서빙에 필요한 키만 추린다(존재하는 키만)."""
    cfg = index_config or {}
    return {key: cfg[key] for key in SERVE_CONFIG_KEYS if key in cfg}


def generation_subset(serve_config: dict | None) -> dict:
    """serve_config 에서 generator 가 쓰는 생성 키만 추린다(configure_generation 용)."""
    cfg = serve_config or {}
    return {key: cfg[key] for key in GENERATION_KEYS if key in cfg}


def sidecar_path(chunks_file: str | Path) -> Path:
    """chunks.json 옆 서빙 설정 사이드카 경로(chunks.json → chunks.json.serve.json)."""
    return Path(f"{chunks_file}.serve.json")


def write_serve_config(chunks_file: str | Path, index_config: dict | None) -> dict:
    """서빙 설정 subset 을 사이드카에 쓴다. 실제로 쓴 설정을 반환한다."""
    config = extract_serve_config(index_config)
    sidecar_path(chunks_file).write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return config


def read_serve_config(chunks_file: str | Path) -> dict:
    """사이드카에서 서빙 설정을 읽는다. 없거나 깨졌으면 빈 dict(=현 기본 동작)."""
    path = sidecar_path(chunks_file)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def serving_fingerprint(chunks: list[dict], serve_config: dict | None) -> str:
    """코퍼스 + 서빙 설정의 결합 지문.

    corpus_fingerprint 만으로는 generation 처럼 재색인 없는(청크 불변) 설정 변경을
    구분하지 못해, 이미 실행 중인 API 가 낡은 설정을 그대로 서빙한다. 설정 해시를 더해
    설정만 바뀌어도 지문이 달라지게 하여 Serve 가 /reload 를 트리거하게 한다."""
    corpus = corpus_fingerprint(chunks)
    payload = json.dumps(serve_config or {}, sort_keys=True, ensure_ascii=False)
    cfg_hash = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    return f"{corpus}-{cfg_hash}"
