"""
Serve Agent — 청크를 저장하고 Claude Desktop 설정 자동 등록 (테스트용)

읽기: state.chunks
쓰기: state.mcp_endpoint, state.status
"""
from __future__ import annotations

import dataclasses
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

from core.state import AgentDoctorState

MCP_SERVER  = Path(__file__).parent / "mcp_server.py"
API_SERVER  = Path(__file__).parent / "api.py"
CHUNKS_FILE = Path(__file__).parent.parent.parent / "chunks.json"
API_PORT    = int(os.getenv("AGENT_DOCTOR_API_PORT", "8766"))
# api.py 는 uvicorn 리스닝 전에 청크 로드 + retriever 구축(임베딩 업서트)을 하므로
# 코퍼스가 크면 기동에 수십 초가 걸릴 수 있다 → 시간 기반 대기.
API_START_TIMEOUT = float(os.getenv("AGENT_DOCTOR_API_START_TIMEOUT", "30"))
API_LOG_FILE = CHUNKS_FILE.parent / "api_server.log"


def _find_claude_config() -> Path:
    """
    Claude Desktop 설정 파일 경로 탐색.
    일반 설치: %APPDATA%/Claude/claude_desktop_config.json
    Store 앱:  %LOCALAPPDATA%/Packages/Claude_*/LocalCache/Roaming/Claude/claude_desktop_config.json
    """
    # Store 앱 경로 (Microsoft Store 설치)
    local = os.environ.get("LOCALAPPDATA", "")
    matches = glob.glob(os.path.join(local, "Packages", "Claude_*"))
    if matches:
        return Path(matches[0]) / "LocalCache" / "Roaming" / "Claude" / "claude_desktop_config.json"

    # 일반 설치 경로
    return Path(os.environ.get("APPDATA", "")) / "Claude" / "claude_desktop_config.json"


CLAUDE_CONFIG = _find_claude_config()


def _serialize_chunks(state: AgentDoctorState) -> str:
    """state.chunks → JSON (embedding 포함, sparse_vector 제외)"""
    data = []
    for chunk in state.chunks:
        d = dataclasses.asdict(chunk)
        d.pop("sparse_vector", None)   # embedding은 벡터 검색에 필요하므로 유지
        data.append(d)
    return json.dumps(data, ensure_ascii=False, default=str, indent=2)


def _start_api_server() -> bool:
    """FastAPI 서버를 백그라운드 프로세스로 시작. 기동 확인 성공 여부를 반환."""
    health_url = f"http://localhost:{API_PORT}/health"

    # 이미 실행 중이면 스킵
    try:
        requests.get(health_url, timeout=2)
        print(f"[Serve] API 서버 이미 실행 중 (port {API_PORT})")
        return True
    except Exception:
        pass

    # 자식 출력을 로그 파일로 남겨 기동 실패 원인을 확인할 수 있게 한다.
    log_file = open(API_LOG_FILE, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [sys.executable, str(API_SERVER), "--chunks-file", str(CHUNKS_FILE), "--port", str(API_PORT)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    finally:
        log_file.close()  # 자식이 핸들을 상속하므로 부모 쪽은 닫아도 된다

    deadline = time.monotonic() + API_START_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            print(
                f"[Serve] API 서버 프로세스 조기 종료 (exit {proc.returncode}) "
                f"— {API_LOG_FILE} 확인"
            )
            return False
        try:
            requests.get(health_url, timeout=1)
            print(f"[Serve] API 서버 시작 완료 (port {API_PORT})")
            return True
        except Exception:
            time.sleep(0.5)

    print(
        f"[Serve] API 서버가 {API_START_TIMEOUT:.0f}초 안에 응답하지 않음 "
        f"— {API_LOG_FILE} 확인"
    )
    return False


def _register_to_claude_desktop() -> None:
    """
    claude_desktop_config.json에 agent-doctor MCP 서버 자동 등록.
    기존 설정은 유지하고 mcpServers 항목만 추가/업데이트.
    """
    # 기존 설정 읽기 (없으면 새로 생성)
    CLAUDE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    if CLAUDE_CONFIG.exists():
        config = json.loads(CLAUDE_CONFIG.read_text(encoding="utf-8"))
    else:
        config = {}

    # mcpServers 항목 추가/업데이트
    config.setdefault("mcpServers", {})
    config["mcpServers"]["agent-doctor"] = {
        "command": sys.executable,
        "args": [str(MCP_SERVER)],
        "env": {
            "AGENT_DOCTOR_API_URL":    f"http://localhost:{API_PORT}",
            "AGENT_DOCTOR_CHUNKS_FILE": str(CHUNKS_FILE),  # api.py 자동 시작용
        },
    }

    CLAUDE_CONFIG.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"[Serve] Claude Desktop 설정 자동 등록 완료 → {CLAUDE_CONFIG}")


def run(state: AgentDoctorState) -> AgentDoctorState:
    """
    Serve Agent 진입점.

    읽기: state.chunks
    쓰기: state.mcp_endpoint, state.status, state.error
    """
    state.current_agent = "serve"
    print(f"[Serve] 청크 {len(state.chunks)}개 처리 중")

    if not state.chunks:
        state.status = "error"
        state.error  = "청크가 없습니다. Index Agent가 완료됐는지 확인하세요."
        return state

    try:
        # 1. 청크 저장
        CHUNKS_FILE.write_text(_serialize_chunks(state), encoding="utf-8")
        print(f"[Serve] 청크 저장 → {CHUNKS_FILE}")

        # 2. FastAPI 서버 백그라운드 시작 — 실패 시 죽은 엔드포인트 등록을 막는다
        if not _start_api_server():
            state.status = "error"
            state.error = (
                f"Serve 실패: API 서버가 시작되지 않았습니다 (port {API_PORT}) "
                f"— {API_LOG_FILE} 확인"
            )
            return state

        # 3. Claude Desktop 설정 자동 등록
        _register_to_claude_desktop()

        # 4. 완료
        state.mcp_endpoint = f"http://localhost:{API_PORT}"
        state.status = "done"

        print("\n" + "=" * 50)
        print(f"[Serve] 완료!")
        print(f"   API 서버: http://localhost:{API_PORT}")
        print(f"   Claude Desktop을 재시작하면 MCP 연결됩니다.")
        print("=" * 50)

    except Exception as e:
        state.status = "error"
        state.error  = f"Serve 실패: {e}"
        print(f"[Serve] 오류: {e}")

    return state
