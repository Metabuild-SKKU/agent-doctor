"""
Agent Doctor MCP server.

이 파일은 Claude Desktop 같은 MCP 클라이언트가 Agent Doctor의 Serve API를
도구처럼 호출할 수 있게 해준다. 검색/답변 생성 자체는 여기서 직접 하지 않고,
이미 검증된 `agents/serve/api.py`에 HTTP로 위임한다.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


def _fastmcp_available() -> bool:
    try:
        return importlib.util.find_spec("mcp.server.fastmcp") is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _missing_fastmcp_message() -> str:
    return (
        "MCP 실행에는 `mcp.server.fastmcp.FastMCP`가 필요합니다. "
        "`pip install 'mcp[cli]<2'`로 MCP 1.x를 설치하거나, "
        "MCP 2.x용 서버 구현(`mcp.server.mcpserver.MCPServer`)으로 갱신하세요."
    )


if _fastmcp_available():
    from mcp.server.fastmcp import FastMCP
else:  # pragma: no cover - 테스트 환경에서 mcp 패키지가 없는 경우만 대체한다.
    class FastMCP:  # type: ignore[no-redef]
        """테스트 환경에 mcp 패키지가 없어도 tool 함수는 import할 수 있게 하는 얇은 대체물."""

        def __init__(self, name: str):
            self.name = name

        def tool(self):
            def decorator(fn):
                return fn

            return decorator

        def run(self, *, transport: str = "stdio") -> None:
            raise RuntimeError(_missing_fastmcp_message())


def _log(message: str) -> None:
    print(f"[MCP] {message}", file=sys.stderr, flush=True)


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        _log(f"{name}={raw!r} 값이 정수가 아니어서 기본값 {default}를 사용합니다.")
        return default
    if minimum is not None and value < minimum:
        _log(f"{name}={raw!r} 값이 너무 작아서 기본값 {default}를 사용합니다.")
        return default
    return value


API_URL = os.getenv("AGENT_DOCTOR_API_URL", "http://localhost:8766").rstrip("/")
CHUNKS_FILE = os.getenv("AGENT_DOCTOR_CHUNKS_FILE", "")
AUTO_START_API = os.getenv("AGENT_DOCTOR_MCP_AUTOSTART", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
STARTUP_RETRIES = _env_int("AGENT_DOCTOR_MCP_STARTUP_RETRIES", 10, minimum=0)
MAX_TOP_K = _env_int("AGENT_DOCTOR_MCP_MAX_TOP_K", 20, minimum=1)
SNIPPET_CHARS = _env_int("AGENT_DOCTOR_MCP_SNIPPET_CHARS", 0, minimum=0)

mcp = FastMCP("agent-doctor")

_api_autostart_attempted = False


def _is_local_api() -> bool:
    host = urlparse(API_URL).hostname
    return host in {None, "", "localhost", "127.0.0.1", "::1"}


def _api_port() -> str:
    parsed = urlparse(API_URL)
    if parsed.port is not None:
        return str(parsed.port)
    return "443" if parsed.scheme == "https" else "80"


def _api_get(
    path: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float = 10,
    ensure_local_api: bool = True,
) -> dict:
    if ensure_local_api and _is_local_api() and not _ensure_api_running():
        raise RuntimeError("Serve API가 준비되지 않았습니다. 파이프라인 Serve 단계를 먼저 실행하세요.")

    response = requests.get(f"{API_URL}{path}", params=params or {}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _ensure_api_running() -> bool:
    """로컬 MCP 데모에서 Serve API가 꺼져 있으면 chunks.json으로 자동 기동한다."""
    global _api_autostart_attempted

    if _health_ok():
        return True

    if not AUTO_START_API or not _is_local_api():
        return False

    if _api_autostart_attempted:
        return False
    _api_autostart_attempted = True

    chunks_path = Path(CHUNKS_FILE) if CHUNKS_FILE else None
    if chunks_path is None or not chunks_path.exists():
        _log("chunks 파일이 없어 Serve API를 자동 시작할 수 없습니다.")
        return False

    api_py = Path(__file__).parent / "api.py"
    subprocess.Popen(
        [
            sys.executable,
            str(api_py),
            "--chunks-file",
            str(chunks_path),
            "--port",
            _api_port(),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    _log("Serve API 자동 시작 중...")
    for _ in range(STARTUP_RETRIES):
        time.sleep(0.5)
        if _health_ok(timeout=1):
            _log("Serve API 준비 완료")
            return True

    _log("Serve API 자동 시작 실패")
    return False


def _health_ok(*, timeout: float = 2) -> bool:
    try:
        response = requests.get(f"{API_URL}/health", timeout=timeout)
        response.raise_for_status()
        return True
    except Exception:
        return False


def _clean_query(query: str) -> str:
    text = (query or "").strip()
    if not text:
        raise ValueError("query는 비어 있을 수 없습니다.")
    return text


def _top_k(value: int | None, *, default: int) -> int:
    try:
        parsed = int(value if value is not None else default)
    except (TypeError, ValueError) as exc:
        raise ValueError("top_k는 정수여야 합니다.") from exc
    if parsed <= 0:
        raise ValueError("top_k는 1 이상이어야 합니다.")
    if parsed > MAX_TOP_K:
        raise ValueError(f"top_k는 {MAX_TOP_K} 이하여야 합니다.")
    return parsed


def _shorten(text: str, *, limit: int = SNIPPET_CHARS) -> str:
    text = " ".join((text or "").split())
    if limit <= 0:
        return text
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _source_label(item: dict) -> str:
    metadata = item.get("metadata", {}) or {}
    title = metadata.get("title") or item.get("title") or item.get("doc_id") or "unknown"
    chunk_id = item.get("chunk_id")
    section = item.get("section") or metadata.get("section")
    detail = " / ".join(str(v) for v in (section, chunk_id) if v)
    return f"{title} ({detail})" if detail else str(title)


def _format_score(score: object) -> str:
    if isinstance(score, (int, float)):
        return f" score={float(score):.3f}"
    return ""


def _format_search_response(data: dict) -> str:
    results = data.get("results", []) or []
    if not results:
        return "관련 청크를 찾지 못했습니다."

    mode = data.get("search_mode") or "unknown"
    fallback = ", keyword fallback" if data.get("fallback_used") else ""
    lines = [f"검색 결과 {len(results)}개 (mode={mode}{fallback})"]
    for index, item in enumerate(results, 1):
        score = _format_score(item.get("score"))
        lines.append(
            f"\n[{index}]{score}\n"
            f"출처: {_source_label(item)}\n"
            f"내용: {_shorten(item.get('text', ''))}"
        )
    return "\n".join(lines)


def _format_answer_response(data: dict) -> str:
    answer = (data.get("answer") or "").strip()
    if not answer:
        return "답변을 만들 수 있는 근거를 찾지 못했습니다."

    mode = data.get("generation_mode")
    lines = [answer]
    if mode:
        lines.append(f"\n생성 방식: {mode}")

    citations = data.get("citations", []) or []
    if citations:
        lines.append("\n근거:")
        for citation in citations[:5]:
            score = _format_score(citation.get("score"))
            title = citation.get("title") or citation.get("doc_id") or "unknown"
            chunk_id = citation.get("chunk_id") or "-"
            lines.append(f"- {title} / {chunk_id}{score}")
    return "\n".join(lines)


@mcp.tool()
def health_check() -> str:
    """Agent Doctor Serve API 상태와 현재 인덱스 설정을 확인합니다."""
    try:
        data = _api_get("/health", timeout=5, ensure_local_api=False)
    except Exception as exc:
        return f"상태 확인 실패: {exc}"

    settings = data.get("index_settings", {}) or {}
    setting_bits = []
    for key in ("top_k", "use_reranker", "use_hybrid", "use_mmr", "embedding_model"):
        if key in settings:
            setting_bits.append(f"{key}={settings[key]}")

    lines = [
        f"상태: {data.get('status', 'unknown')}",
        f"청크 수: {data.get('chunks', 0)}",
        f"Qdrant 사용: {bool(data.get('qdrant'))}",
    ]
    if data.get("fingerprint"):
        lines.append(f"fingerprint: {data['fingerprint']}")
    if setting_bits:
        lines.append("설정: " + ", ".join(setting_bits))
    return "\n".join(lines)


@mcp.tool()
def search_docs(query: str, top_k: int = 3) -> str:
    """질문과 가까운 문서 청크를 검색합니다."""
    try:
        params = {"query": _clean_query(query), "top_k": _top_k(top_k, default=3)}
        data = _api_get("/search", params=params, timeout=15)
    except Exception as exc:
        return f"검색 실패: {exc}"
    return _format_search_response(data)


@mcp.tool()
def ask_docs(query: str, top_k: int = 5) -> str:
    """검색 결과를 근거로 RAG 답변을 생성합니다."""
    try:
        params = {"query": _clean_query(query), "top_k": _top_k(top_k, default=5)}
        data = _api_get("/answer", params=params, timeout=60)
    except Exception as exc:
        return f"답변 생성 실패: {exc}"
    return _format_answer_response(data)


@mcp.tool()
def list_documents() -> str:
    """현재 인덱싱된 문서 목록을 확인합니다."""
    try:
        data = _api_get("/documents", timeout=10)
    except Exception as exc:
        return f"문서 목록 조회 실패: {exc}"

    docs = data.get("documents", []) or []
    if not docs:
        return "인덱싱된 문서가 없습니다."

    lines = [f"총 {data.get('total', len(docs))}개 문서"]
    for doc in docs:
        title = doc.get("title") or doc.get("doc_id") or "unknown"
        lines.append(f"- {title} (id: {doc.get('doc_id', '-')})")
    return "\n".join(lines)


if __name__ == "__main__":
    _log(f"Serve API: {API_URL}")
    _ensure_api_running()
    _log("MCP 서버 시작 (stdio)")
    mcp.run(transport="stdio")
