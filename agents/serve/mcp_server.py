"""
MCP 서버 — FastAPI를 통해 문서 검색 결과를 Claude에 제공

Claude Desktop이 이 파일을 stdio 프로세스로 실행함.
검색은 직접 하지 않고 api.py(FastAPI)에 위임.

[향후 운영 전환]
  AGENT_DOCTOR_API_URL 환경변수를 클라우드 URL로 바꾸면 끝:
    AGENT_DOCTOR_API_URL=https://agentdoctor.com/api
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from mcp.server.fastmcp import FastMCP

API_URL = os.getenv("AGENT_DOCTOR_API_URL", "http://localhost:8766")
CHUNKS_FILE = os.getenv("AGENT_DOCTOR_CHUNKS_FILE", "")

mcp = FastMCP("agent-doctor")


def _ensure_api_running() -> None:
    """api.py가 안 떠있으면 자동 시작."""
    try:
        requests.get(f"{API_URL}/health", timeout=2)
        return  # 이미 실행 중
    except Exception:
        pass

    if not CHUNKS_FILE or not Path(CHUNKS_FILE).exists():
        print("[MCP] chunks 파일 없음 → API 서버 시작 불가", file=sys.stderr, flush=True)
        return

    api_py = Path(__file__).parent / "api.py"
    port = API_URL.split(":")[-1].rstrip("/")
    subprocess.Popen(
        [sys.executable, str(api_py), "--chunks-file", CHUNKS_FILE, "--port", port],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("[MCP] API 서버 시작 중...", file=sys.stderr, flush=True)
    for _ in range(10):
        time.sleep(0.5)
        try:
            requests.get(f"{API_URL}/health", timeout=1)
            print("[MCP] API 서버 준비 완료", file=sys.stderr, flush=True)
            return
        except Exception:
            pass
    print("[MCP] API 서버 시작 실패", file=sys.stderr, flush=True)


_ensure_api_running()


# ── MCP 툴 ────────────────────────────────────────────────────────

@mcp.tool()
def search_docs(query: str) -> str:
    """문서에서 관련 내용을 검색합니다."""
    try:
        resp = requests.get(f"{API_URL}/search", params={"query": query, "top_k": 3}, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except Exception as e:
        return f"검색 실패: {e}"

    if not results:
        return "관련 내용을 찾을 수 없습니다."

    parts = []
    for i, r in enumerate(results, 1):
        title = r.get("metadata", {}).get("title", "")
        score = r.get("score", "")
        score_str = f" (유사도: {score:.3f})" if isinstance(score, float) else ""
        parts.append(f"[{i}]{score_str} {r['text']}\n출처: {title}")
    return "\n\n".join(parts)


@mcp.tool()
def list_documents() -> str:
    """인덱싱된 문서 목록을 반환합니다."""
    try:
        resp = requests.get(f"{API_URL}/documents", timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"문서 목록 조회 실패: {e}"

    docs = data.get("documents", [])
    if not docs:
        return "인덱싱된 문서가 없습니다."

    lines = [f"- {d['title']} (id: {d['doc_id']})" for d in docs]
    return f"총 {data['total']}개 문서:\n" + "\n".join(lines)


# ── 진입점 ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[MCP] API 서버: {API_URL}", file=sys.stderr, flush=True)
    print("[MCP] MCP 서버 시작 (stdio)", file=sys.stderr, flush=True)
    mcp.run(transport="stdio")
