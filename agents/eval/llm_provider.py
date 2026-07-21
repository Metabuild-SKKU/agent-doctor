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

from core.llm_retry import run_with_retry
from core.llm_usage import log_usage

GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"


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


# ── 토큰 사용량·비용 로깅 (core/llm_usage.py 공용 구현) ──────────
# LLM_LOG_USAGE=0 (또는 EVAL_LOG_USAGE=0) 으로 끄기. 단가표도 core 쪽에 있다.

def _log_usage(model: str, prompt_tokens, output_tokens) -> None:
    log_usage(model, prompt_tokens, output_tokens, tag="Eval")


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

def chat_json(system: str, user: str, model: str | None = None) -> dict:
    """JSON 응답 강제 chat 호출 → dict. JSON 파싱 실패 시 {} (API 예외는 호출부로 전파)."""
    def _do():
        provider = _provider()
        if provider == "gemini":
            return _gemini_generate(
                system, user, model or os.getenv("EVAL_JUDGE_MODEL_GEMINI", "gemini-flash-latest"),
                json_mode=True)
        elif provider == "github":
            return _github_generate(
                system, user, model or os.getenv("EVAL_JUDGE_MODEL_GITHUB", "openai/gpt-4o"),
                json_mode=True)
        return _openai_generate(
            system, user, model or os.getenv("EVAL_JUDGE_MODEL", "gpt-4o"),
            json_mode=True)

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


# ── OpenAI 구현 ───────────────────────────────────────────────────

def _openai_generate(system: str, user: str, model: str, json_mode: bool = False) -> str:
    from openai import OpenAI
    client = OpenAI()
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **kwargs,
    )
    if resp.usage:
        _log_usage(model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
    return resp.choices[0].message.content or ""


def _openai_embed(texts: list[str], model: str) -> list[list[float]]:
    from openai import OpenAI
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=texts)
    if resp.usage:
        _log_usage(model, resp.usage.prompt_tokens, 0)
    return [d.embedding for d in resp.data]


# ── GitHub Models 구현 (OpenAI 호환 API, GitHub PAT 인증) ─────────
# 모델명은 "<publisher>/<model>" 형식(예: openai/gpt-4o-mini, meta/Llama-3.3-70B-Instruct).
# 사용 가능한 모델·요율 제한은 https://github.com/marketplace/models 참고.

def _github_generate(system: str, user: str, model: str, json_mode: bool = False) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=GITHUB_MODELS_BASE_URL, api_key=os.getenv("GITHUB_TOKEN"))
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **kwargs,
    )
    if resp.usage:
        _log_usage(model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
    return resp.choices[0].message.content or ""


# ── Gemini 구현 (google-genai SDK) ────────────────────────────────
# 참고: Google AI Studio 콘솔에서 실제 사용 가능한 모델명을 확인할 것
# (모델명/무료 티어 한도는 시점에 따라 바뀔 수 있음).

def _gemini_generate(system: str, user: str, model: str, json_mode: bool = False) -> str:
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = {"temperature": 0, "system_instruction": system}
    if json_mode:
        config["response_mime_type"] = "application/json"
    resp = client.models.generate_content(model=model, contents=user, config=config)
    usage = getattr(resp, "usage_metadata", None)
    if usage:
        # 과금되는 출력 = 답변(candidates) + 내부 사고(thoughts). 추론 모델(3.5-flash 등)은
        # thoughts 가 답변보다 클 수 있어 candidates 만 세면 비용이 절반 이하로 과소집계된다.
        out = (usage.candidates_token_count or 0) + (getattr(usage, "thoughts_token_count", 0) or 0)
        _log_usage(model, usage.prompt_token_count, out)
    return resp.text or ""


def _gemini_embed(texts: list[str], model: str) -> list[list[float]]:
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    resp = client.models.embed_content(model=model, contents=texts)
    return [e.values for e in resp.embeddings]
