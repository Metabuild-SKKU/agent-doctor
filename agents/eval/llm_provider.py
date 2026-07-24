"""
agents/eval/llm_provider.py
Eval Agent 가 쓰는 LLM 호출(OpenAI/Gemini/GitHub Models)을 provider 하나로 추상화한다.

OpenAI API 토큰 승인 전까지 무료 대체 provider 로 브릿지한다:
    EVAL_LLM_PROVIDER=gemini  → Google AI Studio 무료 Gemini API
    EVAL_LLM_PROVIDER=github  → GitHub Models(무료, GitHub PAT 인증, OpenAI 호환 API)
토큰 승인 후에는 EVAL_LLM_PROVIDER=openai(기본값)로 되돌리거나 env 변수를
지우면 원래 동작으로 복귀한다.
"""
from __future__ import annotations

import json
import os

from core.llm_clients import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    GITHUB_MODELS_BASE_URL,
    gemini_chat,
    gemini_embed,
    openai_chat,
    openai_embed,
)
from core.llm_retry import run_with_retry


def _provider() -> str:
    return os.getenv("EVAL_LLM_PROVIDER", "openai").strip().lower()


def has_key() -> bool:
    """활성 provider의 API 키가 설정돼 있는지."""
    provider = _provider()
    if provider == "gemini":
        return bool(os.getenv("GEMINI_API_KEY"))
    if provider == "github":
        return bool(os.getenv("GITHUB_TOKEN"))
    return bool(os.getenv("OPENAI_API_KEY"))


# ── rate limit(429) 재시도 (core/llm_retry.py 공용 구현) ──────────
# env(EVAL_LLM_RETRY_WAIT/EVAL_LLM_MAX_RETRIES)·jitter 동작은 core 쪽 참고.

def _run_with_retry(fn, label: str = "LLM"):
    return run_with_retry(fn, label, tag="Eval")


# ── 답변 생성 (retrieval_temp.py 가 사용) ─────────────────────────

def generate_text(system: str, user: str, model: str | None = None) -> str | None:
    """일반 텍스트 응답 생성. 키/라이브러리 없거나 호출 실패 시 None."""
    if not has_key():
        return None

    def _do():
        provider = _provider()
        if provider == "gemini":
            return _gemini_generate(
                system, user, model or os.getenv("EVAL_GEN_MODEL_GEMINI", "gemini-flash-latest"))
        elif provider == "github":
            return _github_generate(
                system, user, model or os.getenv("EVAL_GEN_MODEL_GITHUB", "openai/gpt-4o-mini"))
        return _openai_generate(
            system, user, model or os.getenv("EVAL_GEN_MODEL", "gpt-4o-mini"))

    try:
        text = _run_with_retry(_do, "생성")
        return (text or "").strip()
    except ImportError:
        return None
    except Exception as e:
        print(f"[Eval] LLM 생성 실패({e}) → 추출식 폴백")
        return None


# ── JSON 강제 채점 호출 (ragas_eval.py 가 사용) ───────────────────

def chat_json(
    system: str,
    user: str,
    model: str | None = None,
    *,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict:
    """JSON 응답 강제 chat 호출 → dict. JSON 파싱 실패 시 {} (API 예외는 호출부로 전파).

    상한에 걸려 잘린 응답은 JSON 파싱에 실패해 {} 가 된다 — 호출부가 "빈 응답"과
    구분하지 못하므로, 구조가 큰 응답을 기대하는 쪽은 max_output_tokens 를 올려 잡을 것."""
    def _do():
        provider = _provider()
        if provider == "gemini":
            return _gemini_generate(
                system, user, model or os.getenv("EVAL_JUDGE_MODEL_GEMINI", "gemini-flash-latest"),
                json_mode=True, max_output_tokens=max_output_tokens)
        elif provider == "github":
            return _github_generate(
                system, user, model or os.getenv("EVAL_JUDGE_MODEL_GITHUB", "openai/gpt-4o"),
                json_mode=True, max_output_tokens=max_output_tokens)
        return _openai_generate(
            system, user, model or os.getenv("EVAL_JUDGE_MODEL", "gpt-4o"),
            json_mode=True, max_output_tokens=max_output_tokens)

    raw = _run_with_retry(_do, "심판")
    try:
        obj = json.loads(raw or "{}")
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


# ── 임베딩 (ragas_eval.py 가 사용) ────────────────────────────────
# GitHub Models 는 embeddings 엔드포인트를 제공하지 않아, github provider 에서도
# 임베딩만은 OpenAI 클라이언트(OPENAI_API_KEY)로 폴백한다 — 없으면 호출부가
# except 로 잡아 스킵(response_relevancy 등 임베딩 의존 지표만 빠짐).

def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """텍스트 리스트 → 임베딩 벡터 리스트. (API 예외는 호출부로 전파; rate limit 은 재시도)"""
    def _do():
        if _provider() == "gemini":
            return _gemini_embed(texts, model or os.getenv("EVAL_EMBED_MODEL_GEMINI", "gemini-embedding-001"))
        return _openai_embed(texts, model or os.getenv("EVAL_EMBED_MODEL", "text-embedding-3-small"))

    return _run_with_retry(_do, "임베딩")


# ── provider 별 transport (core/llm_clients.py 공용 구현에 위임) ──
# 모델명 규약: GitHub Models 는 "<publisher>/<model>" 형식(예: openai/gpt-4o-mini).
# Gemini 모델명/무료 티어 한도는 Google AI Studio 콘솔 참고.

def _openai_generate(
    system: str, user: str, model: str, json_mode: bool = False,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> str:
    return openai_chat(
        system, user, model, json_mode=json_mode,
        max_output_tokens=max_output_tokens, tag="Eval",
    )


def _openai_embed(texts: list[str], model: str) -> list[list[float]]:
    return openai_embed(texts, model, tag="Eval")


def _github_generate(
    system: str, user: str, model: str, json_mode: bool = False,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> str:
    return openai_chat(
        system, user, model, json_mode=json_mode, max_output_tokens=max_output_tokens,
        api_key=os.getenv("GITHUB_TOKEN"), base_url=GITHUB_MODELS_BASE_URL, tag="Eval",
    )


def _gemini_generate(
    system: str, user: str, model: str, json_mode: bool = False,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> str:
    return gemini_chat(
        system, user, model, json_mode=json_mode,
        max_output_tokens=max_output_tokens, tag="Eval",
    )


def _gemini_embed(texts: list[str], model: str) -> list[list[float]]:
    return gemini_embed(texts, model, tag="Eval")
