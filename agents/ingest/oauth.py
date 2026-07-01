"""
agents/ingest/oauth.py
Notion 인증 처리 — 두 가지 방식 모두 지원

방식 1 (Access Token): .env에 NOTION_TOKEN 직접 입력
방식 2 (OAuth):        브라우저로 이용자 Notion 연결

agent.py에서 get_notion_token() 하나만 호출하면 됨
"""
from __future__ import annotations

import json
import os
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import requests

# ── 설정 ──────────────────────────────────────────────────────────

REDIRECT_URI  = "http://localhost:8765/callback"
AUTH_URL      = "https://api.notion.com/v1/oauth/authorize"
TOKEN_URL     = "https://api.notion.com/v1/oauth/token"

# 발급된 토큰을 저장할 파일 (같은 폴더에 생성됨, .gitignore에 추가 권장)
TOKENS_FILE = Path(__file__).parent / "tokens.json"


# ── 토큰 저장 / 로드 ──────────────────────────────────────────────

def _load_all() -> dict:
    if TOKENS_FILE.exists():
        return json.loads(TOKENS_FILE.read_text(encoding="utf-8"))
    return {}


def save_token(user_id: str, token: str) -> None:
    """이용자별 토큰 저장"""
    data = _load_all()
    data[user_id] = token
    TOKENS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_token(user_id: str) -> str | None:
    """저장된 토큰 반환 (없으면 None)"""
    return _load_all().get(user_id)


# ── 방식 1: Access Token ──────────────────────────────────────────

def get_token_from_env() -> str | None:
    """.env의 NOTION_TOKEN 반환"""
    return os.getenv("NOTION_TOKEN")


# ── 방식 2: OAuth ─────────────────────────────────────────────────

def _build_auth_url() -> tuple[str, str]:
    """
    Notion OAuth 인증 URL 생성
    state: CSRF 방지용 1회성 랜덤 값
    반환: (url, state)
    """
    client_id = os.getenv("NOTION_CLIENT_ID")
    if not client_id:
        raise ValueError(".env에 NOTION_CLIENT_ID가 없습니다")

    state = secrets.token_urlsafe(16)
    params = {
        "client_id":     client_id,
        "redirect_uri":  REDIRECT_URI,
        "response_type": "code",
        "owner":         "user",   # 이용자 개인 Notion 접근
        "state":         state,
    }
    return f"{AUTH_URL}?{urlencode(params)}", state


def _exchange_code(code: str) -> str:
    """
    Authorization code → Access token 교환
    Notion은 Basic Auth (client_id:client_secret) 방식 사용
    """
    client_id     = os.getenv("NOTION_CLIENT_ID")
    client_secret = os.getenv("NOTION_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError(".env에 NOTION_CLIENT_ID 또는 NOTION_CLIENT_SECRET이 없습니다")

    resp = requests.post(
        TOKEN_URL,
        json={
            "grant_type":   "authorization_code",
            "code":         code,
            "redirect_uri": REDIRECT_URI,
        },
        auth=(client_id, client_secret),  # HTTP Basic Auth
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def run_oauth_flow(user_id: str = "default") -> str:
    """
    브라우저 OAuth 흐름 전체 실행

    흐름:
      1. 인증 URL 생성
      2. 브라우저 자동 오픈
      3. 이용자가 Notion 로그인 + 승인
      4. localhost:8765/callback 으로 code 수신
      5. code → token 교환
      6. 토큰 저장 후 반환
    """
    auth_url, expected_state = _build_auth_url()

    # 콜백 결과 저장용 (스레드 간 공유)
    result = {"code": None, "error": None}
    done   = threading.Event()

    class _CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            # Notion이 보내는 쿼리: ?code=xxx&state=yyy
            params = parse_qs(urlparse(self.path).query)
            code   = params.get("code",  [None])[0]
            state  = params.get("state", [None])[0]
            error  = params.get("error", [None])[0]

            if error:
                result["error"] = error
                self._html("❌ 연결 실패. 창을 닫아주세요.")

            elif state != expected_state:
                # state 불일치 → CSRF 공격 가능성, 거부
                result["error"] = "state_mismatch"
                self._html("❌ 보안 오류. 다시 시도해주세요.")

            else:
                result["code"] = code
                self._html("✅ Notion 연결 완료! 이 창을 닫아도 됩니다.")

            done.set()

        def _html(self, msg: str):
            body = f"<html><body><h2>{msg}</h2></body></html>".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass  # 서버 로그 숨김

    # 콜백 서버를 백그라운드 스레드로 실행
    server = HTTPServer(("localhost", 8765), _CallbackHandler)
    t = threading.Thread(target=server.handle_request)
    t.start()

    # 브라우저 오픈
    print("[OAuth] 브라우저에서 Notion 로그인 후 연결을 승인해주세요...")
    webbrowser.open(auth_url)

    # 최대 2분 대기
    done.wait(timeout=120)
    t.join()

    if result["error"]:
        raise RuntimeError(f"OAuth 실패: {result['error']}")
    if not result["code"]:
        raise RuntimeError("OAuth 타임아웃 (2분 초과)")

    # code → token
    token = _exchange_code(result["code"])
    save_token(user_id, token)
    print(f"[OAuth] 완료 — 토큰 저장됨 (user_id: {user_id})")
    return token


# ── 통합 진입점 (agent.py에서 이것만 호출) ───────────────────────

def get_notion_token(user_id: str = "default", use_oauth: bool = False) -> str:
    """
    토큰 우선순위:
      1. use_oauth=False → .env의 NOTION_TOKEN 사용
      2. 저장된 OAuth 토큰이 있으면 재사용
      3. 없으면 OAuth 흐름 새로 실행

    Args:
        user_id:   이용자 식별자 (멀티 유저 대비)
        use_oauth: True면 액세스 토큰 무시하고 OAuth 강제
    """
    # 방식 1: 환경변수 우선
    if not use_oauth:
        token = get_token_from_env()
        if token:
            return token

    # 방식 2: 저장된 OAuth 토큰 재사용
    token = load_token(user_id)
    if token:
        return token

    # 방식 2: 새로 OAuth 실행
    return run_oauth_flow(user_id)
