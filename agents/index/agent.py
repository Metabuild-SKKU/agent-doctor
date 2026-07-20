# Index 단계에서 만지는 상태:
# - read: state.documents, state.index_config
# - write: state.chunks, state.index_artifacts, state.status, state.error
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


@dataclass(frozen=True)
class IndexTools:
    # 실험/테스트 때 저장소, 임베딩, 그래프 구현만 바꿔 끼우기 위한 얇은 묶음.
    build_client: Callable[..., Any]
    ensure_collection: Callable[..., Any]
    delete_document_chunks: Callable[..., Any]
    upsert_chunks: Callable[..., Any]
    embed: Callable[..., list[float]]
    count_tokens: Callable[..., int]
    build_sparse_vector: Callable[..., dict]
    build_graph_artifacts: Callable[..., dict]


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
        build_client=build_client,
        ensure_collection=ensure_collection,
        delete_document_chunks=delete_document_chunks,
        upsert_chunks=upsert_chunks,
        embed=embed,
        count_tokens=count_tokens,
        build_sparse_vector=build_sparse_vector,
        build_graph_artifacts=build_graph_artifacts,
    )


# 예전 fixed-size 호출부는 깨지지 않게 그대로 둔다.
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
    }
    return _sha256(json.dumps(relevant, sort_keys=True, ensure_ascii=False))


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
    # Serve는 Qdrant payload만 보고 검색 옵션을 복원하므로 retrieval 설정도 같이 저장한다.
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
        document_chunks: list[Chunk] = []
        for chunk in reusable:
            chunk_hash = _sha256(chunk.text)
            if _is_duplicate(chunk_hash):
                continue
            new_hashes.add(chunk_hash)
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
        print(f"[Index] 기존 임베딩 재사용: {document.doc_id} ({len(document_chunks)}개)")
        return _DocResult(document_chunks, new_hashes, document_hash, len(document_chunks))

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
        if _is_duplicate(chunk_hash):
            continue
        new_hashes.add(chunk_hash)

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
    return _DocResult(document_chunks, new_hashes, document_hash, 0)


# Eval/Optimize가 config를 바꿔 다시 호출하는 흐름을 전제로 둔 Index 본체.
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
        # one-shot: 재생성 플래그는 소비 즉시 끈다. 켠 채로 두면 이후 모든
        # 재색인과 retriever(resolve_retrieval_settings)까지 차원 가드가
        # 풀린 채 남아, mismatch 시 에러 대신 컬렉션이 조용히 삭제된다.
        if state.index_config.get("recreate_collection_on_dimension_mismatch"):
            state.index_config["recreate_collection_on_dimension_mismatch"] = False
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
                "failed_documents": failed_documents,
            }
        )
        state.status = "indexed"
        print(f"[Index] 완료 - 총 {len(all_chunks)}개 청크 (dim={vector_dim})")
    except Exception as exc:
        state.status = "error"
        state.error = f"Index 실패: {exc}"
        print(f"[Index] 오류: {exc}")

    return state
