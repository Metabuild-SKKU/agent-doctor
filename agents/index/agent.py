# Index는 Ingest가 만든 document를 검색 가능한 chunk로 바꾸는 단계다.
# Optimize가 index_config를 바꾼 뒤 다시 호출할 수 있으므로 설정값은 여기서만 해석한다.
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from agents.index.graph_index import build_graph_artifacts
from agents.index.qdrant_store import (
    DEFAULT_EMBEDDING_MODEL,
    build_client,
    build_sparse_vector,
    count_tokens,
    delete_document_chunks,
    embed,
    ensure_collection,
    upsert_chunks,
)
from core.schema import Chunk, Document
from core.state import AgentDoctorState


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

# 테스트나 실험에서 embedding, Qdrant, graph 구현만 바꿔 끼우기 위한 묶음.
@dataclass(frozen=True)
class IndexTools:
    build_client: Callable[..., Any]
    ensure_collection: Callable[..., Any]
    delete_document_chunks: Callable[..., Any]
    upsert_chunks: Callable[..., Any]
    embed: Callable[..., list[float]]
    count_tokens: Callable[..., int]
    build_sparse_vector: Callable[..., dict]
    build_graph_artifacts: Callable[..., dict]


# Ingest 출력이 흔들리면 뒤 단계가 전부 깨져서 Index 경계에서 한 번 더 막는다.
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

# 해시와 char_span이 흔들리지 않도록 최소한의 정규화만 한다.
def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# Index가 처리하기 전에 공통 Document 계약을 확인한다.
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


# 문서와 chunk 중복 판별에 같은 해시 함수를 쓴다.
def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# Eval이 원문 위치를 다시 찾을 수 있어야 해서 char_span은 원문 기준으로 유지한다.
def _trimmed_slice(text: str, start: int, end: int) -> tuple[str, int, int]:
    raw = text[start:end]
    left = len(raw) - len(raw.lstrip())
    right = len(raw.rstrip())
    trimmed_start = start + left
    trimmed_end = start + right
    return text[trimmed_start:trimmed_end], trimmed_start, trimmed_end

# 제목 구조는 section으로 남기고, start/end는 Document.content 기준 좌표로 둔다.
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

# recursive chunking에서 문단이 깨지는 것을 줄이기 위한 경계 우선순위.
def _preferred_boundary(text: str, start: int, hard_end: int) -> int:
    minimum = start + max(1, (hard_end - start) // 2)
    for separator in ("\n\n", "\n", ". ", "。", "? ", "! ", " "):
        position = text.rfind(separator, minimum, hard_end)
        if position >= minimum:
            return position + len(separator)
    return hard_end


# fixed strategy는 가장 단순한 baseline이라 의도적으로 문맥 경계를 보지 않는다.
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


# chunk_size는 맞추되 가능하면 문단이나 문장 경계에서 끊는다.
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


# fixed chunker를 Document 단위 strategy 인터페이스에 맞춘 wrapper.
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


# 구조 보존만 확인할 때 쓰는 markdown-only 전략.
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


# recursive chunker를 Document 단위 strategy 인터페이스에 맞춘 wrapper.
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


# 기본 전략: 문서 구조를 먼저 살리고, 긴 섹션만 다시 자른다.
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


# Notion의 (1)(2)(3) 외 전략을 비교할 때 여기로 등록한다.
def register_chunk_strategy(name: str, strategy: ChunkStrategy) -> None:
    normalized = name.strip()
    if not normalized:
        raise ValueError("chunk_strategy 이름은 빈 문자열일 수 없습니다.")
    CHUNK_STRATEGIES[normalized] = strategy


# 숫자 stage와 문자열 strategy 이름을 실제 strategy key로 통일한다.
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


# Optimize가 chunk_stage나 chunk_strategy 중 무엇을 바꿔도 같은 경로로 처리한다.
def _configured_chunk_strategy(config: dict) -> str:
    raw_strategy = config.get(
        "chunk_stage",
        config.get("chunk_strategy", "markdown_recursive"),
    )
    return _resolve_chunk_strategy(raw_strategy)


# 기본 구현은 현재 index module 안의 Qdrant, embedding, graph 함수를 그대로 쓴다.
def _default_tools() -> IndexTools:
    return IndexTools(
        build_client=build_client,
        ensure_collection=ensure_collection,
        delete_document_chunks=delete_document_chunks,
        upsert_chunks=upsert_chunks,
        embed=embed,
        count_tokens=count_tokens,
        build_sparse_vector=build_sparse_vector,
        build_graph_artifacts=build_graph_artifacts,
    )


# 예전 테스트/호출부가 쓰는 fixed-size helper라 인터페이스를 유지한다.
def _chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, int, int]]:
    trimmed, start, _ = _trimmed_slice(text, 0, len(text))
    return [
        (draft.text, draft.start, draft.end)
        for draft in _fixed_chunks(
            trimmed,
            chunk_size,
            chunk_overlap,
            base_offset=start,
        )
    ]


# 실제 Index 흐름은 이 함수만 통해 chunking 전략을 갈아끼운다.
def _chunk_document(
    document: Document,
    chunk_size: int,
    chunk_overlap: int,
    strategy: str | int = "markdown_recursive",
) -> list[_ChunkDraft]:
    resolved_strategy = _resolve_chunk_strategy(strategy)
    chunker = CHUNK_STRATEGIES[resolved_strategy]
    return chunker(document, chunk_size, chunk_overlap)


# 이 값이 같으면 기존 embedding을 재사용할 수 있다.
def _index_signature(config: dict) -> str:
    relevant = {
        "chunk_size": config["chunk_size"],
        "chunk_overlap": config["chunk_overlap"],
        "chunk_strategy": _configured_chunk_strategy(config),
        "embedding_model": config["embedding_model"],
        "embedding_dimension": config.get("embedding_dimension", 1024),
        "use_hybrid": config.get("use_hybrid", False),
    }
    return _sha256(json.dumps(relevant, sort_keys=True, ensure_ascii=False))


# Optimize가 잘못된 값을 넣었을 때 Index 초입에서 바로 멈춘다.
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


# Serve는 Qdrant payload만 보고 검색 옵션을 복원하므로 retrieval 설정도 chunk에 저장한다.
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
) -> dict[str, Any]:
    return {
        **document.metadata,
        "chunk_index": chunk_index,
        "source": document.source,
        "document_hash": document_hash,
        "chunk_hash": chunk_hash,
        "char_span": [char_span[0], char_span[1]],
        "chunk_strategy": chunk_strategy,
        "index_signature": signature,
        "embedding_model": config["embedding_model"],
        "embedding_dimension": embedding_dimension,
        "use_hybrid": bool(config.get("use_hybrid", False)),
        "hybrid_dense_weight": float(config.get("hybrid_dense_weight", 0.7)),
        "use_reranker": bool(config.get("use_reranker", False)),
        "reranker_model": config.get("reranker_model"),
        "top_k": int(config.get("top_k", 5)),
    }


# 예전 chunk에 char_span 필드가 없으면 metadata에 남은 값으로 복구한다.
def _span_from_chunk(chunk: Chunk) -> tuple[int, int]:
    if chunk.char_span:
        return int(chunk.char_span[0]), int(chunk.char_span[1])
    raw_span = chunk.metadata.get("char_span")
    if isinstance(raw_span, (list, tuple)) and len(raw_span) == 2:
        return int(raw_span[0]), int(raw_span[1])
    return 0, len(chunk.text)


# section이 있으면 나중에 parent-child retrieval로 확장할 수 있게 parent_id를 분리한다.
def _parent_id(document: Document, section: str | None) -> str:
    if section:
        return f"{document.doc_id}:section:{_sha256(section)[:12]}"
    return document.doc_id


# embedding은 재사용하되 top_k 같은 실험값은 최신 config로 맞춘다.
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
    char_span = _span_from_chunk(chunk)
    vector_dim = len(chunk.embedding or []) or int(config.get("embedding_dimension", 1024))
    return replace(
        chunk,
        chunk_id=f"{document.doc_id}_chunk_{chunk_index:03d}",
        doc_id=document.doc_id,
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
        ),
    )


# 이전 실행에서 같은 문서와 같은 Index 설정으로 만든 chunk를 찾기 위한 lookup.
def _previous_chunks_by_document(chunks: list[Chunk]) -> dict[tuple[str, str], list[Chunk]]:
    grouped: dict[tuple[str, str], list[Chunk]] = {}
    for chunk in chunks:
        doc_hash = chunk.metadata.get("document_hash")
        signature = chunk.metadata.get("index_signature")
        if doc_hash and signature:
            grouped.setdefault((doc_hash, signature), []).append(chunk)
    return grouped


# Optimize가 config를 바꾼 뒤 다시 들어오는 Index Agent의 진입점.
def run(state: AgentDoctorState, tools: IndexTools | None = None) -> AgentDoctorState:
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
        chunk_strategy = _configured_chunk_strategy(config)
        signature = _index_signature(config)
        previous = _previous_chunks_by_document(state.chunks)
        seen_documents: set[str] = set()
        seen_doc_ids: dict[str, str] = {}
        seen_chunks: set[str] = set()
        all_chunks: list[Chunk] = []
        reused_count = 0

        for document in state.documents:
            _validate_document(document)
            normalized = _normalize_text(document.content)
            document_hash = _sha256(normalized)
            previous_hash = seen_doc_ids.get(document.doc_id)
            if previous_hash and previous_hash != document_hash:
                raise ValueError(
                    f"같은 doc_id에 서로 다른 본문이 들어왔습니다: {document.doc_id}"
                )
            seen_doc_ids[document.doc_id] = document_hash
            if config.get("deduplicate", True) and document_hash in seen_documents:
                print(f"[Index] 중복 문서 제외: {document.doc_id}")
                continue
            seen_documents.add(document_hash)

            reusable = previous.get((document_hash, signature), [])
            if reusable:
                document_chunks: list[Chunk] = []
                for chunk in reusable:
                    chunk_hash = _sha256(chunk.text)
                    if config.get("deduplicate", True) and chunk_hash in seen_chunks:
                        continue
                    seen_chunks.add(chunk_hash)
                    document_chunks.append(
                        _refresh_reused_chunk(
                            chunk,
                            document,
                            config,
                            chunk_index=len(document_chunks),
                            document_hash=document_hash,
                            chunk_hash=chunk_hash,
                            chunk_strategy=chunk_strategy,
                            signature=signature,
                        )
                    )
                all_chunks.extend(document_chunks)
                reused_count += len(document_chunks)
                print(f"[Index] 기존 임베딩 재사용: {document.doc_id} ({len(document_chunks)}개)")
                continue

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

            document_chunks: list[Chunk] = []
            for draft in drafts:
                chunk_hash = _sha256(draft.text)
                if config.get("deduplicate", True) and chunk_hash in seen_chunks:
                    continue
                seen_chunks.add(chunk_hash)

                vector = tools.embed(
                    draft.text,
                    model_name=config["embedding_model"],
                    vector_dim=config.get("embedding_dimension"),
                )
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
                )
                chunk = Chunk(
                    chunk_id=f"{document.doc_id}_chunk_{chunk_index:03d}",
                    doc_id=document.doc_id,
                    text=draft.text,
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
            all_chunks.extend(document_chunks)

        if not all_chunks:
            raise ValueError("검증과 중복 제거 후 저장할 청크가 없습니다.")

        vector_dim = len(next(chunk.embedding for chunk in all_chunks if chunk.embedding))
        client = tools.build_client(
            url=os.getenv("QDRANT_URL", ":memory:"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        tools.ensure_collection(
            client,
            vector_dim=vector_dim,
            recreate_on_mismatch=bool(
                config.get("recreate_collection_on_dimension_mismatch", False)
            ),
        )
        tools.delete_document_chunks(client, list(seen_doc_ids))
        tools.upsert_chunks(client, all_chunks)

        state.chunks = all_chunks
        if config.get("graph_enabled", True):
            state.index_artifacts = tools.build_graph_artifacts(all_chunks, config)
        else:
            state.index_artifacts = {}
        state.index_artifacts.update(
            {
                "documents": len(seen_documents),
                "chunks": len(all_chunks),
                "reused_embeddings": reused_count,
                "chunk_strategy": chunk_strategy,
                "embedding_model": config["embedding_model"],
                "embedding_dimension": vector_dim,
            }
        )
        state.status = "indexed"
        print(f"[Index] 완료 - 총 {len(all_chunks)}개 청크 (dim={vector_dim})")
    except Exception as exc:
        state.status = "error"
        state.error = f"Index 실패: {exc}"
        print(f"[Index] 오류: {exc}")

    return state
