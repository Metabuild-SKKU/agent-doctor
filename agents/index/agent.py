# Index 단계에서 만지는 상태:
# - read: state.source_url, state.source_type, state.documents,
#         state.index_config, state.reindex_required,
#         state.optimization_history, state.active_index_key, state.index_cache
# - write: state.chunks, state.index_artifacts, state.reindex_required,
#          state.index_cache, state.active_index_key, state.index_cache_hit,
#          state.status, state.error, state.current_agent
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from agents.index.graph_index import build_graph_artifacts
from agents.index.qdrant_store import (
    DEFAULT_EMBEDDING_MODEL,
    build_sparse_vector,
    count_tokens,
    embed,
    embed_batch,
    embedding_is_fallback,
    resolve_embedding_device,
)
from agents.rag.retriever import get_retriever, reset_retriever_cache
from core.schema import Chunk, Document, IndexSnapshot
from core.state import AgentDoctorState
from core.timing import StageTimer


@dataclass
class _ChunkDraft:
    text: str
    section: str | None = None
    start: int = 0
    end: int = 0


@dataclass
class _SectionDraft:
    text: str
    section: str | None
    start: int
    end: int


@dataclass(frozen=True)
class IndexTools:
    # 실험/테스트 때 저장소, 임베딩, 그래프 구현만 바꿔 끼우기 위한 얇은 묶음.
    get_retriever: Callable[..., Any]
    embed: Callable[..., list[float]]
    count_tokens: Callable[..., int]
    build_sparse_vector: Callable[..., dict]
    build_graph_artifacts: Callable[..., dict]
    # 배치 임베딩(없으면 단건 embed 루프 폴백) — 필드 끝에 default 로 두어
    # embed 만 주입하는 기존 테스트/실험 코드가 그대로 동작한다.
    embed_batch: Callable[..., list[list[float]]] | None = None
    # "지금 임베딩하면 fallback 인가" 술어(없으면 provenance 미기록=항상 실제로 간주).
    # 모델 로드 실패로 만든 해시 fallback 벡터를 청크에 표시하고, 복구 후 강제
    # 재임베딩할지 판단하는 데 쓴다. default 로 둬 기존 주입 코드 호환.
    embedding_is_fallback: Callable[..., bool] | None = None


# Ingest가 넘겨준 Document도 Index 경계에서 한 번 더 확인한다.
class _DocumentSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    format: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("doc_id", "source", "format", "content")
    @classmethod
    def value_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("빈 문자열일 수 없습니다.")
        return value


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _validate_document(document: Document) -> None:
    if not isinstance(document, Document):
        raise TypeError(f"Document 타입이 아닙니다: {type(document).__name__}")
    try:
        _DocumentSchema.model_validate(
            {
                "doc_id": document.doc_id,
                "source": document.source,
                "format": document.format,
                "content": _normalize_text(document.content),
                "metadata": document.metadata,
            }
        )
    except ValidationError as exc:
        raise ValueError(f"{document.doc_id or '(doc_id 없음)'}: 문서 검증 실패: {exc}") from exc


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# char_span이 원문 좌표를 가리키도록 앞뒤 공백만 보정한다.
def _trimmed_slice(text: str, start: int, end: int) -> tuple[str, int, int]:
    raw = text[start:end]
    left = len(raw) - len(raw.lstrip())
    right = len(raw.rstrip())
    trimmed_start = start + left
    trimmed_end = start + right
    return text[trimmed_start:trimmed_end], trimmed_start, trimmed_end


# Markdown 제목은 section 이름으로 남기고, 위치값은 원문 기준을 유지한다.
def _split_markdown_sections(text: str) -> list[_SectionDraft]:
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    heading_path: list[str] = []
    sections: list[_SectionDraft] = []
    current_section: str | None = None
    section_start = 0
    cursor = 0

    def flush(end: int) -> None:
        body, body_start, body_end = _trimmed_slice(text, section_start, end)
        if body:
            sections.append(
                _SectionDraft(
                    text=body,
                    section=current_section,
                    start=body_start,
                    end=body_end,
                )
            )

    for line in text.splitlines(keepends=True):
        match = heading_pattern.match(line.rstrip("\r\n"))
        if not match:
            cursor += len(line)
            continue
        flush(cursor)
        section_start = cursor
        level = len(match.group(1))
        title = match.group(2).strip()
        heading_path = heading_path[: level - 1]
        heading_path.append(title)
        current_section = " > ".join(heading_path)
        cursor += len(line)

    flush(len(text))
    if sections:
        return sections
    body, start, end = _trimmed_slice(text, 0, len(text))
    return [_SectionDraft(text=body, section=None, start=start, end=end)] if body else []


# recursive chunker가 문맥 경계에서 끊도록 후보 순서를 둔다.
def _preferred_boundary(text: str, start: int, hard_end: int) -> int:
    minimum = start + max(1, (hard_end - start) // 2)
    for separator in ("\n\n", "\n", ". ", "。", "? ", "! ", " "):
        position = text.rfind(separator, minimum, hard_end)
        if position >= minimum:
            return position + len(separator)
    return hard_end


def _fixed_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    *,
    base_offset: int = 0,
    section: str | None = None,
) -> list[_ChunkDraft]:
    if not text:
        return []
    chunks: list[_ChunkDraft] = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        chunk, trimmed_start, trimmed_end = _trimmed_slice(text, start, end)
        if chunk:
            chunks.append(
                _ChunkDraft(
                    text=chunk,
                    section=section,
                    start=base_offset + trimmed_start,
                    end=base_offset + trimmed_end,
                )
            )
        if end >= len(text):
            break
    return chunks


def _recursive_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    *,
    base_offset: int = 0,
    section: str | None = None,
) -> list[_ChunkDraft]:
    if not text:
        return []
    if len(text) <= chunk_size:
        return [
            _ChunkDraft(
                text=text,
                section=section,
                start=base_offset,
                end=base_offset + len(text),
            )
        ]

    chunks: list[_ChunkDraft] = []
    start = 0
    while start < len(text):
        hard_end = min(len(text), start + chunk_size)
        end = hard_end if hard_end == len(text) else _preferred_boundary(text, start, hard_end)
        chunk, trimmed_start, trimmed_end = _trimmed_slice(text, start, end)
        if chunk:
            chunks.append(
                _ChunkDraft(
                    text=chunk,
                    section=section,
                    start=base_offset + trimmed_start,
                    end=base_offset + trimmed_end,
                )
            )
        if end >= len(text):
            break
        next_start = max(start + 1, end - chunk_overlap)
        while next_start < end and text[next_start].isspace():
            next_start += 1
        start = next_start
    return chunks


def _fixed_strategy(
    document: Document,
    chunk_size: int,
    chunk_overlap: int,
) -> list[_ChunkDraft]:
    text, start, _ = _trimmed_slice(document.content, 0, len(document.content))
    return _fixed_chunks(
        text,
        chunk_size,
        chunk_overlap,
        base_offset=start,
    )


def _markdown_strategy(
    document: Document,
    _chunk_size: int,
    _chunk_overlap: int,
) -> list[_ChunkDraft]:
    return [
        _ChunkDraft(
            text=section.text,
            section=section.section,
            start=section.start,
            end=section.end,
        )
        for section in _split_markdown_sections(document.content)
    ]


def _recursive_strategy(
    document: Document,
    chunk_size: int,
    chunk_overlap: int,
) -> list[_ChunkDraft]:
    text, start, _ = _trimmed_slice(document.content, 0, len(document.content))
    return _recursive_chunks(
        text,
        chunk_size,
        chunk_overlap,
        base_offset=start,
    )


def _markdown_recursive_strategy(
    document: Document,
    chunk_size: int,
    chunk_overlap: int,
) -> list[_ChunkDraft]:
    drafts: list[_ChunkDraft] = []
    for section in _split_markdown_sections(document.content):
        drafts.extend(
            _recursive_chunks(
                section.text,
                chunk_size,
                chunk_overlap,
                base_offset=section.start,
                section=section.section,
            )
        )
    return drafts


ChunkStrategy = Callable[[Document, int, int], list[_ChunkDraft]]

CHUNK_STRATEGIES: dict[str, ChunkStrategy] = {
    "fixed": _fixed_strategy,
    "markdown": _markdown_strategy,
    "recursive": _recursive_strategy,
    "markdown_recursive": _markdown_recursive_strategy,
}

CHUNK_STAGE_ALIASES: dict[str | int, str] = {
    1: "fixed",
    "1": "fixed",
    "stage_1": "fixed",
    2: "recursive",
    "2": "recursive",
    "stage_2": "recursive",
    3: "markdown_recursive",
    "3": "markdown_recursive",
    "stage_3": "markdown_recursive",
}


# Notion의 (1)(2)(3) 외 실험 chunker를 붙일 때 쓴다.
def register_chunk_strategy(name: str, strategy: ChunkStrategy) -> None:
    normalized = name.strip()
    if not normalized:
        raise ValueError("chunk_strategy 이름은 빈 문자열일 수 없습니다.")
    CHUNK_STRATEGIES[normalized] = strategy


def _resolve_chunk_strategy(strategy: str | int) -> str:
    resolved = CHUNK_STAGE_ALIASES.get(strategy, strategy)
    if isinstance(resolved, str):
        resolved = resolved.strip()
    if resolved in CHUNK_STRATEGIES:
        return str(resolved)
    choices = ", ".join(CHUNK_STRATEGIES)
    stages = "1=fixed, 2=recursive, 3=markdown_recursive"
    raise ValueError(
        f"지원하지 않는 chunk_strategy입니다: {strategy}. 선택값: {choices}; 단계: {stages}"
    )


def _configured_chunk_strategy(config: dict) -> str:
    raw_strategy = config.get(
        "chunk_stage",
        config.get("chunk_strategy", "markdown_recursive"),
    )
    return _resolve_chunk_strategy(raw_strategy)


def _default_tools() -> IndexTools:
    return IndexTools(
        get_retriever=get_retriever,
        embed=embed,
        count_tokens=count_tokens,
        build_sparse_vector=build_sparse_vector,
        build_graph_artifacts=build_graph_artifacts,
        embed_batch=embed_batch,
        embedding_is_fallback=embedding_is_fallback,
    )


# Index 본문에서는 여기만 호출해서 chunking 전략을 교체한다.
def _chunk_document(
    document: Document,
    chunk_size: int,
    chunk_overlap: int,
    strategy: str | int = "markdown_recursive",
) -> list[_ChunkDraft]:
    resolved_strategy = _resolve_chunk_strategy(strategy)
    chunker = CHUNK_STRATEGIES[resolved_strategy]
    return chunker(document, chunk_size, chunk_overlap)


# 청크/임베딩 결과를 바꾸는 설정만 재사용 판단에 반영한다.
def _index_signature(config: dict) -> str:
    relevant = {
        "chunk_size": config["chunk_size"],
        "chunk_overlap": config["chunk_overlap"],
        "chunk_strategy": _configured_chunk_strategy(config),
        "embedding_model": config["embedding_model"],
        "embedding_dimension": config.get("embedding_dimension", 1024),
        "use_hybrid": config.get("use_hybrid", False),
        "deduplicate": config.get("deduplicate", True),
    }
    return _sha256(json.dumps(relevant, sort_keys=True, ensure_ascii=False))


def _graph_cache_signature(config: dict) -> dict:
    """그래프 결과만 바꾸는 설정은 임베딩 재사용 signature와 분리한다."""
    graph_config = {
        key: value
        for key, value in config.items()
        if key.startswith("graph_")
    }
    extraction = str(config.get("graph_extraction", "auto"))
    graph_config["llm_available"] = bool(
        extraction in {"auto", "llm"} and os.getenv("OPENAI_API_KEY")
    )
    return graph_config


def _index_cache_key(documents: list[Document], config: dict) -> str:
    """원문과 인덱스 산출 설정으로 결정되는 롤백 캐시 키를 만든다."""
    corpus = [
        {
            "doc_id": document.doc_id,
            "source": document.source,
            "format": document.format,
            "content_hash": _sha256(_normalize_text(document.content)),
            "metadata": document.metadata,
        }
        for document in documents
    ]
    payload = {
        "schema_version": 2,
        "index_signature": _index_signature(config),
        "graph_signature": _graph_cache_signature(config),
        "collection_namespace": config.get(
            "qdrant_collection_namespace_resolved",
            "",
        ),
        # deduplicate=True일 때 동일 본문 중 먼저 나온 문서가 승자가 되므로
        # 입력 순서까지 fingerprint에 보존해야 provenance가 뒤바뀌지 않는다.
        "documents": corpus,
    }
    return _sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    )


def _collection_slots(
    state: AgentDoctorState,
    config: dict,
) -> tuple[str, str]:
    """코퍼스/사용자 namespace별 고정 Qdrant 슬롯 두 개를 만든다."""
    explicit = str(
        config.get("qdrant_collection_namespace", "")
    ).strip()
    previous = str(
        state.index_artifacts.get("qdrant_collection_namespace", "")
    ).strip()
    source_identity = (
        {
            "source_type": state.source_type,
            "source_url": state.source_url,
        }
        if state.source_type or state.source_url
        else {
            "document_sources": sorted(
                {
                    document.source
                    for document in state.documents
                }
            )
        }
    )
    if explicit:
        prefix = f"agent_doctor_{_sha256(explicit)[:12]}"
    elif previous.startswith("agent_doctor_"):
        prefix = previous
    else:
        namespace_source = json.dumps(
            source_identity,
            sort_keys=True,
            ensure_ascii=False,
        )
        prefix = f"agent_doctor_{_sha256(namespace_source)[:12]}"
    config["qdrant_collection_namespace_resolved"] = prefix
    return f"{prefix}_slot_0", f"{prefix}_slot_1"


def _pending_baseline_index_key(state: AgentDoctorState) -> str:
    """현재 처방이 실패했을 때 돌아가야 할 baseline 인덱스 키를 찾는다."""
    for item in reversed(state.optimization_history):
        metadata = getattr(item, "metadata", {}) or {}
        if not metadata.get("pending"):
            continue
        key = metadata.get("before_index_key")
        if key:
            return str(key)
    return ""


def _next_collection_name(
    state: AgentDoctorState,
    slots: tuple[str, str],
    protected_key: str = "",
) -> str:
    """보호할 baseline 반대편의 고정 슬롯을 새 인덱스 생성 공간으로 고른다."""
    active_collection = str(
        state.index_artifacts.get("qdrant_collection_name", "")
    )
    lookup_key = protected_key or state.active_index_key
    if lookup_key:
        for snapshot in state.index_cache:
            if snapshot.cache_key == lookup_key:
                active_collection = snapshot.collection_name
                break
    slot_0, slot_1 = slots
    return slot_1 if active_collection == slot_0 else slot_0


def _cache_limit(config: dict) -> int:
    if not config.get("rollback_cache_enabled", True):
        return 0
    try:
        requested = int(config.get("rollback_cache_max_versions", 2))
    except (TypeError, ValueError):
        requested = 2
    return max(1, min(2, requested))


def _find_index_snapshot(
    state: AgentDoctorState,
    cache_key: str,
) -> IndexSnapshot | None:
    """캐시 hit를 LRU 최신 위치로 옮겨 다음 축출에서 보호한다."""
    if _cache_limit(state.index_config) == 0:
        state.index_cache = []
        return None
    for index, snapshot in enumerate(state.index_cache):
        if snapshot.cache_key != cache_key:
            continue
        state.index_cache.append(state.index_cache.pop(index))
        return state.index_cache[-1]
    return None


def _store_index_snapshot(
    state: AgentDoctorState,
    cache_key: str,
    collection_name: str,
    config: dict,
) -> None:
    """현재 인덱스를 저장하고 현재/직전 두 버전만 남긴다."""
    limit = _cache_limit(config)
    if limit == 0:
        state.index_cache = []
        return
    state.index_cache = [
        snapshot
        for snapshot in state.index_cache
        if snapshot.cache_key != cache_key
    ]
    state.index_cache.append(
        IndexSnapshot(
            cache_key=cache_key,
            # Chunk는 이후 경로에서 제자리 수정하지 않고 dataclasses.replace로 교체한다.
            # 활성 state와 객체를 공유해 동일 임베딩을 메모리에 한 벌 더 복제하지 않는다.
            chunks=list(state.chunks),
            index_artifacts=deepcopy(state.index_artifacts),
            collection_name=collection_name,
        )
    )
    pinned_key = _pending_baseline_index_key(state)
    pinned = next(
        (
            snapshot
            for snapshot in state.index_cache
            if snapshot.cache_key == pinned_key
        ),
        None,
    )
    current = state.index_cache[-1]
    if (
        limit == 2
        and pinned is not None
        and pinned.cache_key != current.cache_key
    ):
        state.index_cache = [pinned, current]
    else:
        state.index_cache = state.index_cache[-limit:]


def _refresh_runtime_metadata(
    chunks: list[Chunk],
    config: dict,
) -> list[Chunk]:
    """재인덱싱 없이 바뀌는 검색 설정을 청크 provenance에도 반영한다."""
    refreshed = []
    for chunk in chunks:
        metadata = {
            **(chunk.metadata or {}),
            "hybrid_dense_weight": float(
                config.get("hybrid_dense_weight", 0.7)
            ),
            "use_reranker": bool(config.get("use_reranker", False)),
            "reranker_model": config.get("reranker_model"),
            "top_k": int(config.get("top_k", 5)),
            "qdrant_collection_name": config.get(
                "qdrant_collection_name"
            ),
            "index_cache_key": config.get("index_cache_key"),
        }
        refreshed.append(replace(chunk, metadata=metadata))
    return refreshed


def _validate_config(config: dict) -> None:
    chunk_size = config["chunk_size"]
    overlap = config["chunk_overlap"]
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size는 1 이상의 정수여야 합니다.")
    if not isinstance(overlap, int) or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap은 0 이상 chunk_size 미만이어야 합니다.")
    _configured_chunk_strategy(config)
    top_k = config.get("top_k", 5)
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k는 1 이상의 정수여야 합니다.")
    batch_size = config.get("embedding_batch_size", 16)
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("embedding_batch_size는 1 이상의 정수여야 합니다.")
    device = config.get("embedding_device", "auto")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("embedding_device는 비어 있지 않은 문자열이어야 합니다.")
    if device.strip().lower() not in {"auto", "cpu", "cuda"} and not re.fullmatch(
        r"cuda:\d+", device.strip().lower()
    ):
        raise ValueError("embedding_device는 auto, cpu, cuda 또는 cuda:N이어야 합니다.")


def _page_of_span(document: Document, char_span: tuple[int, int]) -> int | None:
    """청크가 원본 문서의 몇 페이지에서 나왔는지 (1-based). 모르면 None.

    Ingest(agents/ingest/preprocess.py)가 PDF 에서만 document.metadata["page_spans"]
    를 채운다. char_span 과 같은 좌표계(Document.content 기준)라 시작 위치만 대조하면
    된다. 청크가 페이지 경계를 걸치면 "시작한 페이지"로 본다 — 인용 표기 목적이라
    첫 페이지가 사람이 찾아가기에 맞다.
    """
    spans = document.metadata.get("page_spans")
    if not isinstance(spans, (list, tuple)):
        return None

    start = char_span[0]
    last_nonempty: int | None = None
    for page_no, span in enumerate(spans, start=1):
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        try:
            page_start, page_end = int(span[0]), int(span[1])
        except (TypeError, ValueError):
            # 이 함수의 방침은 "깨진 입력이면 조용히 None". page_spans 는 직렬화를
            # 거쳐 오는 값이라 숫자가 아닐 수 있는데, 여기서 예외가 나면 청킹 루프
            # 한복판에서 터진다 — 출처 표기용 부가 정보 때문에 색인을 죽이지 않는다.
            continue
        if page_start <= start < page_end:
            return page_no
        # 청크 시작이 페이지 사이 구분자에 걸린 경우(공백만 있는 위치)를 대비해
        # "여기까지 지나온 마지막 페이지"를 기억해둔다.
        if page_start <= start:
            last_nonempty = page_no
    return last_nonempty


def _chunk_metadata(
    document: Document,
    config: dict,
    *,
    chunk_index: int,
    document_hash: str,
    chunk_hash: str,
    char_span: tuple[int, int],
    chunk_strategy: str,
    signature: str,
    embedding_dimension: int,
    embedding_fallback: bool = False,
) -> dict[str, Any]:
    # page_spans 는 문서 단위 정보라 청크마다 복사하면 payload 가 페이지 수만큼 불어난다.
    # 아래에서 이 청크의 "page" 하나로 접어 넣으므로 원본 목록은 뺀다.
    doc_metadata = {k: v for k, v in document.metadata.items() if k != "page_spans"}

    # Serve는 Qdrant payload만 보고 검색 옵션을 복원하므로 retrieval 설정도 같이 저장한다.
    return {
        **doc_metadata,
        "chunk_index": chunk_index,
        "source": document.source,
        "document_hash": document_hash,
        "chunk_hash": chunk_hash,
        "char_span": [char_span[0], char_span[1]],
        # 출처 표기용 페이지 번호. document.metadata 의 page_spans 를 그대로 물려받으면
        # 청크마다 문서 전체 span 목록이 payload 에 복사되므로, 여기서 이 청크의
        # 페이지 하나로 접고 원본 목록은 뺀다.
        "page": _page_of_span(document, char_span),
        "chunk_strategy": chunk_strategy,
        "index_signature": signature,
        "embedding_model": config["embedding_model"],
        "embedding_dimension": embedding_dimension,
        # 이 벡터가 (의미 없는) 해시 fallback 으로 만들어졌는지. True 면 모델 복구 후
        # 재색인 시 강제 재임베딩 대상이다(_process_document reusable 분기).
        "embedding_fallback": bool(embedding_fallback),
        "use_hybrid": bool(config.get("use_hybrid", False)),
        "hybrid_dense_weight": float(config.get("hybrid_dense_weight", 0.7)),
        "use_reranker": bool(config.get("use_reranker", False)),
        "reranker_model": config.get("reranker_model"),
        "top_k": int(config.get("top_k", 5)),
        "qdrant_collection_name": config.get("qdrant_collection_name"),
        "index_cache_key": config.get("index_cache_key"),
    }


def _span_from_chunk(chunk: Chunk) -> tuple[int, int]:
    if chunk.char_span:
        return int(chunk.char_span[0]), int(chunk.char_span[1])
    raw_span = chunk.metadata.get("char_span")
    if isinstance(raw_span, (list, tuple)) and len(raw_span) == 2:
        return int(raw_span[0]), int(raw_span[1])
    return 0, len(chunk.text)


def _parent_id(document: Document, section: str | None) -> str:
    if section:
        return f"{document.doc_id}:section:{_sha256(section)[:12]}"
    return document.doc_id


def _refresh_reused_chunk(
    chunk: Chunk,
    document: Document,
    config: dict,
    *,
    chunk_index: int,
    document_hash: str,
    chunk_hash: str,
    chunk_strategy: str,
    signature: str,
) -> Chunk:
    # 임베딩은 재사용하되, top_k 같은 실험값은 최신 config로 맞춘다.
    char_span = _span_from_chunk(chunk)
    vector_dim = len(chunk.embedding or []) or int(config.get("embedding_dimension", 1024))
    return replace(
        chunk,
        chunk_id=f"{document.doc_id}_chunk_{chunk_index:03d}",
        doc_id=document.doc_id,
        # 재청킹되면 char_span 이 달라지므로 페이지도 다시 계산한다(replace 는 옛 값을 남긴다).
        page=_page_of_span(document, char_span),
        char_span=char_span,
        parent_id=_parent_id(document, chunk.section),
        hash=chunk_hash[:16],
        metadata=_chunk_metadata(
            document,
            config,
            chunk_index=chunk_index,
            document_hash=document_hash,
            chunk_hash=chunk_hash,
            char_span=char_span,
            chunk_strategy=chunk_strategy,
            signature=signature,
            embedding_dimension=vector_dim,
            # 임베딩을 재사용하는 경로이므로 fallback 여부도 그대로 이어야 한다.
            # 여기서 기본값(False)으로 덮으면, 모델이 두 번 연속 실패하는 동안 플래그가
            # 지워져 이후 모델이 복구돼도 재임베딩 대상으로 잡히지 않는다(해시 벡터 고착).
            embedding_fallback=bool(chunk.metadata.get("embedding_fallback")),
        ),
    )


def _reembed_stale_chunks(
    document_chunks: list[Chunk],
    stale: list[tuple[int, "Chunk", str]],
    document: Document,
    config: dict,
    tools: "IndexTools",
    *,
    document_hash: str,
    chunk_strategy: str,
    signature: str,
) -> list[Chunk]:
    """fallback 으로 색인됐던 청크들을 복구된 모델로 다시 임베딩해 교체한다.

    document_chunks 의 placeholder(원본 fallback 청크) 자리를, 실제 모델 벡터와
    embedding_fallback=False 메타데이터를 가진 새 Chunk 로 바꿔 돌려준다.
    좌표·section·hash 등 임베딩 외 속성은 원본을 그대로 잇는다(재청킹 아님)."""
    texts = [chunk.text for _idx, chunk, _h in stale]
    if tools.embed_batch is not None:
        vectors = tools.embed_batch(
            texts,
            model_name=config["embedding_model"],
            vector_dim=config.get("embedding_dimension"),
        )
    else:
        vectors = [
            tools.embed(
                text,
                model_name=config["embedding_model"],
                vector_dim=config.get("embedding_dimension"),
            )
            for text in texts
        ]

    for (chunk_index, chunk, chunk_hash), vector in zip(stale, vectors):
        char_span = _span_from_chunk(chunk)
        metadata = _chunk_metadata(
            document,
            config,
            chunk_index=chunk_index,
            document_hash=document_hash,
            chunk_hash=chunk_hash,
            char_span=char_span,
            chunk_strategy=chunk_strategy,
            signature=signature,
            embedding_dimension=len(vector),
            embedding_fallback=False,
        )
        document_chunks[chunk_index] = replace(
            chunk,
            chunk_id=f"{document.doc_id}_chunk_{chunk_index:03d}",
            doc_id=document.doc_id,
            page=metadata.get("page"),
            char_span=char_span,
            parent_id=_parent_id(document, chunk.section),
            hash=chunk_hash[:16],
            embedding=vector,
            sparse_vector=(
                tools.build_sparse_vector(chunk.text)
                if config.get("use_hybrid", False)
                else chunk.sparse_vector
            ),
            metadata=metadata,
        )
    return document_chunks


def _previous_chunks_by_document(chunks: list[Chunk]) -> dict[tuple[str, str], list[Chunk]]:
    grouped: dict[tuple[str, str], list[Chunk]] = {}
    for chunk in chunks:
        doc_hash = chunk.metadata.get("document_hash")
        signature = chunk.metadata.get("index_signature")
        if doc_hash and signature:
            grouped.setdefault((doc_hash, signature), []).append(chunk)
    return grouped


@dataclass(frozen=True)
class _DocResult:
    chunks: list[Chunk]        # 이 문서에서 새로 만든/재사용한 청크
    new_hashes: set[str]       # 성공 시에만 seen_chunks에 커밋할 청크 해시
    document_hash: str
    reused: int                # 재사용 임베딩 개수 (신규는 0)
    reembedded: int = 0        # 모델 복구로 fallback 벡터를 실제 벡터로 다시 임베딩한 개수


# 문서 하나를 청크 리스트로 변환한다. 공유 상태(seen_*)는 읽기만 하고,
# 새로 본 청크 해시는 반환값으로 돌려줘서 호출자가 성공 시에만 커밋한다 —
# 처리 도중 실패한 문서의 흔적이 dedup 집합에 남으면 다른 문서의 동일 청크가
# 중복으로 오인되어 조용히 누락되기 때문.
def _process_document(
    document: Document,
    *,
    config: dict,
    tools: IndexTools,
    chunk_strategy: str,
    signature: str,
    previous: dict[tuple[str, str], list[Chunk]],
    seen_chunks: set[str],
    seen_doc_ids: dict[str, str],
    seen_documents: set[str],
    timer: StageTimer,
) -> _DocResult:
    _validate_document(document)
    normalized = _normalize_text(document.content)
    document_hash = _sha256(normalized)
    previous_hash = seen_doc_ids.get(document.doc_id)
    if previous_hash and previous_hash != document_hash:
        raise ValueError(
            f"같은 doc_id에 서로 다른 본문이 들어왔습니다: {document.doc_id}"
        )
    if config.get("deduplicate", True) and document_hash in seen_documents:
        print(f"[Index] 중복 문서 제외: {document.doc_id}")
        return _DocResult([], set(), document_hash, 0)

    new_hashes: set[str] = set()

    def _is_duplicate(chunk_hash: str) -> bool:
        return config.get("deduplicate", True) and (
            chunk_hash in seen_chunks or chunk_hash in new_hashes
        )

    reusable = previous.get((document_hash, signature), [])
    if reusable:
        # 모델이 (다시) 로드 가능해졌으면, 이전에 fallback(해시 벡터)으로 색인된 청크는
        # 재사용하면 안 된다 — 문서 벡터는 fallback 공간, 질의 벡터는 실제 모델 공간이라
        # 서로 다른 벡터 공간을 비교하게 되어 검색 점수가 무의미해진다. 이런 청크만
        # 골라 강제 재임베딩하고, 나머지는 기존대로 임베딩을 재사용한다.
        model_recovered = bool(
            tools.embedding_is_fallback
            and not tools.embedding_is_fallback(config["embedding_model"])
        )

        document_chunks: list[Chunk] = []
        stale: list[tuple[int, Chunk, str]] = []   # (chunk_index, chunk, chunk_hash) — 재임베딩 대상
        reused_count = 0
        for chunk in reusable:
            chunk_hash = _sha256(chunk.text)
            if _is_duplicate(chunk_hash):
                continue
            new_hashes.add(chunk_hash)
            chunk_index = len(document_chunks)
            was_fallback = bool(chunk.metadata.get("embedding_fallback"))
            if model_recovered and was_fallback:
                # 자리(순서)만 잡아 두고 뒤에서 실제 벡터로 채운다.
                document_chunks.append(chunk)   # placeholder, 아래서 교체
                stale.append((chunk_index, chunk, chunk_hash))
                continue
            document_chunks.append(
                _refresh_reused_chunk(
                    chunk,
                    document,
                    config,
                    chunk_index=chunk_index,
                    document_hash=document_hash,
                    chunk_hash=chunk_hash,
                    chunk_strategy=chunk_strategy,
                    signature=signature,
                )
            )
            reused_count += 1

        if stale:
            document_chunks = _reembed_stale_chunks(
                document_chunks,
                stale,
                document,
                config,
                tools,
                document_hash=document_hash,
                chunk_strategy=chunk_strategy,
                signature=signature,
            )
            print(
                f"[Index] 모델 복구 감지 → fallback 청크 재임베딩: "
                f"{document.doc_id} ({len(stale)}개, 재사용 {reused_count}개)"
            )
        else:
            print(f"[Index] 기존 임베딩 재사용: {document.doc_id} ({reused_count}개)")
        return _DocResult(
            document_chunks, new_hashes, document_hash, reused_count,
            reembedded=len(stale),
        )

    with timer.measure(f"청킹 ({document.doc_id})"):
        drafts = _chunk_document(
            document,
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            strategy=chunk_strategy,
        )
    title = document.metadata.get("title", document.doc_id)
    print(
        f"[Index] └ '{title}' → {len(drafts)}개 청크 후보 "
        f"(strategy={chunk_strategy})"
    )

    unique_drafts: list[tuple[_ChunkDraft, str]] = []
    # pass 1: dedup 판정(해시 기반 — 임베딩과 무관하므로 판정 결과는 기존과 동일)
    for draft in drafts:
        chunk_hash = _sha256(draft.text)
        if _is_duplicate(chunk_hash):
            continue
        new_hashes.add(chunk_hash)
        unique_drafts.append((draft, chunk_hash))

    # 모델 로드 실패로 해시 fallback을 쓰는 경우 provenance를 남겨, 모델이
    # 복구된 다음 실행에서 해당 청크만 실제 벡터로 다시 임베딩한다.
    fallback_now = bool(
        tools.embedding_is_fallback
        and tools.embedding_is_fallback(config["embedding_model"])
    )

    # pass 2: 살아남은 draft를 설정된 장치에서 한 번에 배치 임베딩한다.
    texts = [draft.text for draft, _chunk_hash in unique_drafts]
    embedding_device = resolve_embedding_device(config.get("embedding_device", "auto"))
    embedding_batch_size = int(config.get("embedding_batch_size", 16))
    with timer.measure(
        f"임베딩 ({document.doc_id}, {embedding_device}, batch={embedding_batch_size})",
        synchronize_cuda=embedding_device.startswith("cuda"),
    ):
        if tools.embed_batch is not None:
            vectors = tools.embed_batch(
                texts,
                model_name=config["embedding_model"],
                vector_dim=config.get("embedding_dimension"),
                device=config.get("embedding_device", "auto"),
                batch_size=embedding_batch_size,
            )
        else:
            # 기존 실험용 IndexTools 구현도 깨지지 않도록 단건 도구를 호환한다.
            vectors = [
                tools.embed(
                    text,
                    model_name=config["embedding_model"],
                    vector_dim=config.get("embedding_dimension"),
                )
                for text in texts
            ]
    if len(vectors) != len(unique_drafts):
        raise ValueError(
            "배치 임베딩 결과 수가 청크 수와 다릅니다: "
            f"청크={len(unique_drafts)}, 벡터={len(vectors)}"
        )

    document_chunks: list[Chunk] = []
    for (draft, chunk_hash), vector in zip(unique_drafts, vectors):
        chunk_index = len(document_chunks)
        char_span = (draft.start, draft.end)
        metadata = _chunk_metadata(
            document,
            config,
            chunk_index=chunk_index,
            document_hash=document_hash,
            chunk_hash=chunk_hash,
            char_span=char_span,
            chunk_strategy=chunk_strategy,
            signature=signature,
            embedding_dimension=len(vector),
            embedding_fallback=fallback_now,
        )
        chunk = Chunk(
            chunk_id=f"{document.doc_id}_chunk_{chunk_index:03d}",
            doc_id=document.doc_id,
            text=draft.text,
            page=metadata.get("page"),
            section=draft.section,
            char_span=char_span,
            token_count=tools.count_tokens(
                draft.text,
                model_name=config["embedding_model"],
            ),
            parent_id=_parent_id(document, draft.section),
            hash=chunk_hash[:16],
            embedding=vector,
            sparse_vector=(
                tools.build_sparse_vector(draft.text)
                if config.get("use_hybrid", False)
                else None
            ),
            metadata=metadata,
        )
        document_chunks.append(chunk)
    return _DocResult(document_chunks, new_hashes, document_hash, 0)


def run(state: AgentDoctorState, tools: IndexTools | None = None) -> AgentDoctorState:
    """Index Agent 진입점과 전체 실행 시간을 함께 기록한다."""
    timer = StageTimer("Index")
    try:
        return _run(state, tools, timer)
    finally:
        timer.finish()


# Eval/Optimize가 config를 바꿔 다시 호출하는 흐름을 전제로 둔 Index 본체.
def _run(
    state: AgentDoctorState,
    tools: IndexTools | None,
    timer: StageTimer,
) -> AgentDoctorState:
    tools = tools or _default_tools()
    state.current_agent = "index"
    state.error = None
    print(f"[Index] 문서 {len(state.documents)}개 처리 시작")

    if not state.documents:
        state.status = "error"
        state.error = "문서가 없습니다. Ingest Agent 완료 여부를 확인하세요."
        return state

    config = {
        "chunk_size": state.index_config.get("chunk_size", 600),
        "chunk_overlap": state.index_config.get("chunk_overlap", 80),
        "embedding_model": state.index_config.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        "embedding_dimension": state.index_config.get("embedding_dimension", 1024),
        **state.index_config,
    }
    try:
        _validate_config(config)
        collection_slots = _collection_slots(state, config)
        target_key = _index_cache_key(state.documents, config)
        config["index_cache_key"] = target_key
    except Exception as exc:
        state.status = "error"
        state.error = f"Index 실패: {exc}"
        print(f"[Index] 오류: {exc}")
        return state
    force_rebuild = bool(
        state.reindex_required
        and state.active_index_key == target_key
    )
    snapshot = (
        None
        if force_rebuild
        else _find_index_snapshot(state, target_key)
    )
    protected_key = _pending_baseline_index_key(state)
    if snapshot is not None:
        collection_name = snapshot.collection_name
    elif state.active_index_key == target_key and not force_rebuild:
        collection_name = str(
            state.index_artifacts.get(
                "qdrant_collection_name",
                _next_collection_name(
                    state,
                    collection_slots,
                    protected_key,
                ),
            )
        )
    else:
        collection_name = _next_collection_name(
            state,
            collection_slots,
            protected_key,
        )
    config["qdrant_collection_name"] = collection_name
    graph_output_root = Path(
        str(config.get("graph_output_dir", "output/index_graph"))
    )
    config["graph_output_dir"] = str(
        graph_output_root / collection_name
    )
    state.index_cache_hit = False

    # 런타임 설정 변경(False)이고 논리 인덱스도 같을 때만 재색인을 건너뛴다.
    # True이면 같은 fingerprint라도 손상 복구/명시적 재생성 요청으로 보고
    # 비활성 슬롯에 다시 만든다.
    if (
        state.chunks
        and not state.reindex_required
        and (
            state.active_index_key == target_key
            or not state.active_index_key
        )
    ):
        state.chunks = _refresh_runtime_metadata(state.chunks, config)
        state.active_index_key = target_key
        state.reindex_required = True
        state.status = "indexed"
        state.index_artifacts = {
            **state.index_artifacts,
            "reindex_skipped": True,
            "index_cache_hit": False,
            "active_index_key": target_key,
            "qdrant_collection_name": collection_name,
            "qdrant_collection_namespace": config[
                "qdrant_collection_namespace_resolved"
            ],
            "reused_embeddings": len(state.chunks),
            "skip_reason": "검색 시점 설정만 변경됨",
        }
        _store_index_snapshot(
            state,
            target_key,
            collection_name,
            config,
        )
        print("[Index] 검색 시점 설정만 변경됨 - 기존 인덱스 재사용")
        return state

    if snapshot is not None:
        restored_chunks = _refresh_runtime_metadata(
            list(snapshot.chunks),
            config,
        )
        restored_artifacts = {
            **deepcopy(snapshot.index_artifacts),
            "index_cache_hit": True,
            "active_index_key": target_key,
            "qdrant_collection_name": snapshot.collection_name,
            "qdrant_collection_namespace": config[
                "qdrant_collection_namespace_resolved"
            ],
        }
        # 같은 프로세스에서는 retriever의 2-slot 캐시가 그대로 반환된다. 캐시가
        # 유실됐어도 원격 Qdrant 슬롯이 남아 있으면 upsert 없이 다시 연결한다.
        # 슬롯까지 없어진 경우에만 저장된 임베딩으로 복구하며 재임베딩은 하지 않는다.
        config["reuse_existing_collection"] = True
        try:
            tools.get_retriever(restored_chunks, config)
        except Exception as exc:
            state.status = "error"
            state.error = f"Index 캐시 복원 실패: {exc}"
            print(f"[Index] 오류: {state.error}")
            return state

        # 외부 저장소 재연결까지 성공한 뒤 공유 상태를 한 번에 바꾼다.
        state.chunks = restored_chunks
        state.index_artifacts = restored_artifacts
        state.active_index_key = target_key
        state.index_cache_hit = True
        state.reindex_required = True
        state.status = "indexed"
        if state.index_config.get("recreate_collection_on_dimension_mismatch"):
            state.index_config["recreate_collection_on_dimension_mismatch"] = False
        print(f"[Index] 롤백 인덱스 캐시 복원: {target_key[:12]}")
        return state

    try:
        # 새 버전은 비활성 슬롯을 완전히 교체한다. 고정 슬롯 두 개만 사용하므로
        # 프로세스가 재시작돼도 Qdrant 컬렉션이 버전 수만큼 누적되지 않는다.
        config["replace_qdrant_collection"] = True
        _validate_config(config)
        chunk_strategy = _configured_chunk_strategy(config)
        signature = _index_signature(config)
        previous = _previous_chunks_by_document(state.chunks)
        seen_documents: set[str] = set()
        seen_doc_ids: dict[str, str] = {}
        seen_chunks: set[str] = set()
        all_chunks: list[Chunk] = []
        reused_count = 0
        reembedded_count = 0

        failed_documents: list[dict] = []

        for document in state.documents:
            try:
                res = _process_document(
                    document,
                    config=config,
                    tools=tools,
                    chunk_strategy=chunk_strategy,
                    signature=signature,
                    previous=previous,
                    seen_chunks=seen_chunks,
                    seen_doc_ids=seen_doc_ids,
                    seen_documents=seen_documents,
                    timer=timer,
                )
            except Exception as exc:
                doc_id = str(getattr(document, "doc_id", "<unknown>"))
                failed_documents.append({"doc_id": doc_id, "error": str(exc)})
                print(f"[Index] 문서 처리 실패(건너뜀): {doc_id} — {exc}")
                continue

            # 성공한 문서만 공유 상태에 반영한다. 실패 문서의 doc_id가 seen_doc_ids에
            # 남으면 아래 delete_document_chunks가 기존 벡터를 지우는데 새 청크는
            # upsert되지 않아 벡터 스토어에서 그 문서가 통째로 사라진다.
            seen_chunks |= res.new_hashes
            seen_doc_ids[document.doc_id] = res.document_hash
            seen_documents.add(res.document_hash)
            all_chunks.extend(res.chunks)
            reused_count += res.reused
            reembedded_count += res.reembedded

        if not all_chunks:
            failure_summary = (
                f" (문서 {len(failed_documents)}개 처리 실패, "
                f"첫 오류: {failed_documents[0]['error']})"
                if failed_documents
                else ""
            )
            raise ValueError(f"검증과 중복 제거 후 저장할 청크가 없습니다.{failure_summary}")

        if failed_documents:
            print(
                f"[Index] 경고: 문서 {len(failed_documents)}개 처리 실패 — "
                f"나머지 {len(seen_doc_ids)}개는 정상 인덱싱 "
                f"(상세: index_artifacts['failed_documents'])"
            )

        vector_dim = len(next(chunk.embedding for chunk in all_chunks if chunk.embedding))
        # fallback 벡터를 실제 벡터로 재임베딩한 경우, 적재 캐시를 비운다.
        # get_retriever 의 캐시 키(_population_key)는 (scope_id, 모델명, 차원, 저장소)만
        # 보는데, scope_id 는 hash=sha256(text) 라 임베딩과 무관하다. 모델명·차원이 그대로면
        # 벡터만 바뀐 이 전환은 키가 충돌해 옛 fallback 컬렉션이 재사용된다(retriever.py
        # "남는 구멍" 주석 참고). 명시적 reset 으로 새 벡터가 실제 upsert 되게 한다.
        if reembedded_count:
            reset_retriever_cache()
        # 컬렉션 준비·증분 삭제·upsert를 공통 retriever에 위임한다. 뒤이어 도는
        # Eval/Serve가 같은 청크로 get_retriever를 부르면 이 적재 결과를 그대로 쓴다.
        # 재생성 플래그는 config를 통해 ensure_collection까지 전달된다.
        with timer.measure("Qdrant 적재"):
            tools.get_retriever(all_chunks, config, delete_doc_ids=list(seen_doc_ids))
        # one-shot: 재생성 플래그는 소비 즉시 끈다. 켠 채로 두면 이후 모든
        # 재색인과 retriever(resolve_retrieval_settings)까지 차원 가드가
        # 풀린 채 남아, mismatch 시 에러 대신 컬렉션이 조용히 삭제된다.
        if state.index_config.get("recreate_collection_on_dimension_mismatch"):
            state.index_config["recreate_collection_on_dimension_mismatch"] = False

        state.chunks = all_chunks
        if config.get("graph_enabled", True):
            with timer.measure("그래프 생성"):
                state.index_artifacts = tools.build_graph_artifacts(all_chunks, config)
        else:
            state.index_artifacts = {}
        state.index_artifacts.update(
            {
                "documents": len(seen_documents),
                "chunks": len(all_chunks),
                "reused_embeddings": reused_count,
                "reembedded_fallback": reembedded_count,
                "chunk_strategy": chunk_strategy,
                "embedding_model": config["embedding_model"],
                "embedding_dimension": vector_dim,
                "embedding_device": resolve_embedding_device(
                    config.get("embedding_device", "auto")
                ),
                "embedding_batch_size": int(config.get("embedding_batch_size", 16)),
                "failed_documents": failed_documents,
                "index_cache_hit": False,
                "active_index_key": target_key,
                "qdrant_collection_name": collection_name,
                "qdrant_collection_namespace": config[
                    "qdrant_collection_namespace_resolved"
                ],
            }
        )
        state.active_index_key = target_key
        state.reindex_required = True
        _store_index_snapshot(
            state,
            target_key,
            collection_name,
            config,
        )
        state.status = "indexed"
        print(f"[Index] 완료 - 총 {len(all_chunks)}개 청크 (dim={vector_dim})")
    except Exception as exc:
        state.status = "error"
        state.error = f"Index 실패: {exc}"
        print(f"[Index] 오류: {exc}")

    return state
