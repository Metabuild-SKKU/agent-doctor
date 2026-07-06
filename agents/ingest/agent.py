"""
agents/ingest/agent.py
Ingest Agent — 데이터 소스에서 문서를 수집해 Document 리스트로 반환

담당: 지민

[구현 포인트]
  1. state.source_type 보고 맞는 수집기 호출
  2. Document 객체 만들어서 state.documents에 추가
  3. state 반환

[지원 source_type]
  - "notion" : Notion 페이지 (NOTION_TOKEN 필요)
  - "file"   : 로컬 파일 (.txt / .md / .pdf)
  - "gdrive" : Google Drive (TODO)
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from core.schema import Document
from core.state import AgentDoctorState


# ── Notion ────────────────────────────────────────────────────────

def _ingest_notion(source_url: str) -> list[Document]:
    """Notion 페이지 전체를 텍스트로 수집"""
    try:
        from notion_client import Client
    except ImportError:
        raise ImportError("pip install notion-client")

    from agents.ingest.oauth import get_notion_token
    token = get_notion_token()  # .env 토큰 or OAuth 자동 처리

    client = Client(auth=token)

    # URL에서 32자리 hex page_id 추출
    # 지원 형식:
    #   https://notion.so/Title-360f03a0822180bab3eac6472512572e
    #   https://app.notion.com/p/360f03a0822180bab3eac6472512572e
    import re
    match = re.search(r"([0-9a-f]{32})", source_url.replace("-", ""))
    if not match:
        raise ValueError(f"URL에서 page_id를 찾을 수 없습니다: {source_url}")
    page_id = match.group(1)

    page = client.pages.retrieve(page_id=page_id)
    title = _notion_title(page)
    content = _notion_blocks_to_text(client, page_id)

    return [Document(
        doc_id=str(uuid.uuid4()),
        source=source_url,
        format="notion",
        content=content,
        metadata={
            "title": title,
            "page_id": page_id,
        },
    )]


def _notion_title(page: dict) -> str:
    for prop in page.get("properties", {}).values():
        if prop.get("type") == "title":
            return "".join(t.get("plain_text", "") for t in prop.get("title", []))
    return "Untitled"


def _notion_blocks_to_text(client, block_id: str, depth: int = 0) -> str:
    """블록 재귀 순회해서 텍스트 추출"""
    lines = []
    cursor = None

    while True:
        resp = client.blocks.children.list(
            block_id=block_id, start_cursor=cursor, page_size=100
        )
        for block in resp.get("results", []):
            text = _notion_block_text(block)
            if text:
                lines.append("  " * depth + text)
            if block.get("has_children"):
                lines.append(_notion_blocks_to_text(client, block["id"], depth + 1))

        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")

    return "\n".join(filter(None, lines))


def _notion_block_text(block: dict) -> str:
    """블록 타입 → 텍스트 변환"""
    btype = block.get("type", "")
    data = block.get(btype, {})
    text = "".join(rt.get("plain_text", "") for rt in data.get("rich_text", []))

    prefix = {
        "heading_1": "# ", "heading_2": "## ", "heading_3": "### ",
        "bulleted_list_item": "• ", "numbered_list_item": "1. ",
        "to_do": "☐ ", "quote": "> ", "callout": "📌 ",
    }.get(btype, "")

    return f"{prefix}{text}" if text else ""


# ── 로컬 파일 ─────────────────────────────────────────────────────

def _ingest_file(source_url: str) -> list[Document]:
    """로컬 파일 수집 (.txt / .md / .pdf)"""
    path = Path(source_url)
    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {source_url}")

    suffix = path.suffix.lower()

    if suffix in (".txt", ".md"):
        content = path.read_text(encoding="utf-8")
        fmt = "md" if suffix == ".md" else "txt"

    elif suffix == ".pdf":
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pip install pdfplumber")
        with pdfplumber.open(path) as pdf:
            content = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
        fmt = "pdf"

    else:
        raise ValueError(f"지원 안 하는 형식: {suffix}  (지원: .txt .md .pdf)")

    return [Document(
        doc_id=str(uuid.uuid4()),
        source=str(path.resolve()),
        format=fmt,
        content=content,
        metadata={"filename": path.name},
    )]


# ── JSON Corpus ───────────────────────────────────────────────────

def _ingest_json_corpus(source_url: str) -> list[Document]:
    """
    직무체험 — 전처리된 corpus.json 로드

    기대 형식:
        [{"id": "doc_0", "text": "...", "source": "고용동향.pdf"}, ...]
    """
    import json

    path = Path(source_url)
    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {source_url}")

    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("corpus.json은 리스트 형식이어야 합니다: [{id, text, source}, ...]")

    docs = []
    seen_ids: set[str] = set()
    for i, item in enumerate(data):
        content = item.get("text")
        if not content:
            raise ValueError(f"item[{i}] (id={item.get('id', '?')})에 'text' 필드가 없습니다")

        doc_id = item.get("id", str(uuid.uuid4()))
        if doc_id in seen_ids:
            raise ValueError(f"item[{i}]의 doc_id '{doc_id}'가 앞선 항목과 중복됩니다")
        seen_ids.add(doc_id)

        src = item.get("source", str(path.resolve()))
        fmt = Path(src).suffix.lstrip(".").lower() or "txt"

        docs.append(Document(
            doc_id  = doc_id,
            source  = src,
            format  = fmt,
            content = content,
            metadata= {"source_file": item.get("source", path.name)},
        ))

    return docs


# ── Google Drive (TODO) ───────────────────────────────────────────

def _ingest_gdrive(source_url: str) -> list[Document]:
    """Google Drive 수집 — 추후 구현"""
    raise NotImplementedError(
        "gdrive 수집은 아직 미구현입니다.\n"
        "구현 참고: https://developers.google.com/drive/api/quickstart/python"
    )


# ── 라우팅 테이블 ─────────────────────────────────────────────────

_INGESTERS = {
    "notion":      _ingest_notion,
    "file":        _ingest_file,
    "json_corpus": _ingest_json_corpus,
    "gdrive":      _ingest_gdrive,
}


# ── run() ─────────────────────────────────────────────────────────

def run(state: AgentDoctorState) -> AgentDoctorState:
    """
    Ingest Agent 진입점.

    읽기: state.source_url, state.source_type
    쓰기: state.documents, state.status, state.error
    """
    state.current_agent = "ingest"
    print(f"[Ingest] {state.source_type}: {state.source_url}")

    ingester = _INGESTERS.get(state.source_type)
    if ingester is None:
        state.status = "error"
        state.error = f"지원 안 하는 source_type: '{state.source_type}' | 지원: {list(_INGESTERS)}"
        print(f"[Ingest] 오류: {state.error}")
        return state

    try:
        docs = ingester(state.source_url)
        state.documents = docs
        print(f"[Ingest] 완료 — {len(docs)}개 문서 수집")
    except NotImplementedError as e:
        state.status = "error"
        state.error = str(e)
        print(f"[Ingest] 미구현: {e}")
    except Exception as e:
        state.status = "error"
        state.error = f"수집 실패: {e}"
        print(f"[Ingest] 오류: {e}")

    return state
