"""
Index Agent — 문서 검증·중복 제거·청킹·임베딩·Qdrant 저장.

읽기: state.documents, state.index_config
쓰기: state.chunks, state.index_artifacts, state.status, state.error
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

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

class _DocumentSchema(BaseModel):
    """Ingest의 dataclass를 Index 경계에서 다시 검증하는 Pydantic schema."""

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
    """Unicode와 줄바꿈을 통일하되 문단 구조는 보존한다."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _validate_document(document: Document) -> None:
    """Index가 처리하기 전에 공통 Document 계약을 확인한다."""
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


def _trimmed_slice(text: str, start: int, end: int) -> tuple[str, int, int]:
    """원문 slice의 공백을 제거하고 보정된 원문 좌표를 함께 반환한다."""
    raw = text[start:end]
    left = len(raw) - len(raw.lstrip())
    right = len(raw.rstrip())
    trimmed_start = start + left
    trimmed_end = start + right
    return text[trimmed_start:trimmed_end], trimmed_start, trimmed_end


def _split_markdown_sections(text: str) -> list[_SectionDraft]:
    """Markdown 제목 경계를 찾되 모든 위치는 원본 Document 좌표로 유지한다."""
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


def _preferred_boundary(text: str, start: int, hard_end: int) -> int:
    """문단→줄→문장→공백 순서로 가장 가까운 경계를 찾는다."""
    minimum = start + max(1, (hard_end - start) // 2)
    for separator in ("\n\n", "\n", ". ", "。", "? ", "! ", " "):
        position = text.rfind(separator, minimum, hard_end)
        if position >= minimum:
            return position + len(separator)
    return hard_end


def _recursive_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    *,
    base_offset: int = 0,
    section: str | None = None,
) -> list[_ChunkDraft]:
    """문맥 경계를 우선하고 각 결과의 원문 char span을 보존한다."""
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


def _chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, int, int]]:
    """텍스트와 원본 기준 (start, end)를 반환하는 호환용 공개 함수."""
    trimmed, start, _ = _trimmed_slice(text, 0, len(text))
    return [
        (draft.text, draft.start, draft.end)
        for draft in _recursive_chunks(
            trimmed,
            chunk_size,
            chunk_overlap,
            base_offset=start,
        )
    ]


def _chunk_document(document: Document, chunk_size: int, chunk_overlap: int) -> list[_ChunkDraft]:
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


def _index_signature(config: dict) -> str:
    """청크나 임베딩 결과에 영향을 주는 설정만 fingerprint에 포함한다."""
    relevant = {
        "chunk_size": config["chunk_size"],
        "chunk_overlap": config["chunk_overlap"],
        "chunk_strategy": config.get("chunk_strategy", "markdown_recursive"),
        "embedding_model": config["embedding_model"],
        "embedding_dimension": config.get("embedding_dimension", 1024),
        "use_hybrid": config.get("use_hybrid", False),
    }
    return _sha256(json.dumps(relevant, sort_keys=True, ensure_ascii=False))


def _validate_config(config: dict) -> None:
    chunk_size = config["chunk_size"]
    overlap = config["chunk_overlap"]
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size는 1 이상의 정수여야 합니다.")
    if not isinstance(overlap, int) or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap은 0 이상 chunk_size 미만이어야 합니다.")
    top_k = config.get("top_k", 5)
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k는 1 이상의 정수여야 합니다.")


def _previous_chunks_by_document(chunks: list[Chunk]) -> dict[tuple[str, str], list[Chunk]]:
    grouped: dict[tuple[str, str], list[Chunk]] = {}
    for chunk in chunks:
        doc_hash = chunk.metadata.get("document_hash")
        signature = chunk.metadata.get("index_signature")
        if doc_hash and signature:
            grouped.setdefault((doc_hash, signature), []).append(chunk)
    return grouped


def run(state: AgentDoctorState) -> AgentDoctorState:
    """Notion에서 정한 Index 파이프라인을 실행하고 항상 state를 반환한다."""
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
                all_chunks.extend(reusable)
                reused_count += len(reusable)
                print(f"[Index] 기존 임베딩 재사용: {document.doc_id} ({len(reusable)}개)")
                continue

            drafts = _chunk_document(
                document,
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"],
            )
            title = document.metadata.get("title", document.doc_id)
            print(f"[Index] └ '{title}' → {len(drafts)}개 청크 후보")

            document_chunks: list[Chunk] = []
            for draft in drafts:
                chunk_hash = _sha256(draft.text)
                if config.get("deduplicate", True) and chunk_hash in seen_chunks:
                    continue
                seen_chunks.add(chunk_hash)

                vector = embed(
                    draft.text,
                    model_name=config["embedding_model"],
                    vector_dim=config.get("embedding_dimension"),
                )
                chunk_index = len(document_chunks)
                metadata = {
                    **document.metadata,
                    "chunk_index": chunk_index,
                    "source": document.source,
                    "document_hash": document_hash,
                    "chunk_hash": chunk_hash,
                    "char_span": [draft.start, draft.end],
                    "index_signature": signature,
                    "embedding_model": config["embedding_model"],
                    "embedding_dimension": len(vector),
                    "use_hybrid": bool(config.get("use_hybrid", False)),
                    "hybrid_dense_weight": float(config.get("hybrid_dense_weight", 0.7)),
                    "use_reranker": bool(config.get("use_reranker", False)),
                    "reranker_model": config.get("reranker_model"),
                    "top_k": int(config.get("top_k", 5)),
                }
                chunk = Chunk(
                    chunk_id=f"{document.doc_id}_chunk_{chunk_index:03d}",
                    doc_id=document.doc_id,
                    text=draft.text,
                    section=draft.section,
                    char_span=(draft.start, draft.end),
                    token_count=count_tokens(
                        draft.text,
                        model_name=config["embedding_model"],
                    ),
                    parent_id=(
                        f"{document.doc_id}:section:{_sha256(draft.section)[:12]}"
                        if draft.section
                        else document.doc_id
                    ),
                    hash=chunk_hash[:16],
                    embedding=vector,
                    sparse_vector=(
                        build_sparse_vector(draft.text)
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
        client = build_client(
            url=os.getenv("QDRANT_URL", ":memory:"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        ensure_collection(
            client,
            vector_dim=vector_dim,
            recreate_on_mismatch=bool(
                config.get("recreate_collection_on_dimension_mismatch", False)
            ),
        )
        delete_document_chunks(client, list(seen_doc_ids))
        upsert_chunks(client, all_chunks)

        state.chunks = all_chunks
        if config.get("graph_enabled", True):
            state.index_artifacts = build_graph_artifacts(all_chunks, config)
        else:
            state.index_artifacts = {}
        state.index_artifacts.update(
            {
                "documents": len(seen_documents),
                "chunks": len(all_chunks),
                "reused_embeddings": reused_count,
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
