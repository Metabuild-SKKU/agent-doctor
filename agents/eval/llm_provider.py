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
import time

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


# ── rate limit(429) 재시도 ────────────────────────────────────────
# Gemini 무료 티어 등에서 "too many requests"(429)가 나면 잠시 대기 후 재시도한다.
#   EVAL_LLM_RETRY_WAIT   대기 시간(초, 기본 5)
#   EVAL_LLM_MAX_RETRIES  재시도 횟수 상한(기본 5). 소진하면 마지막 예외를 그대로 올린다.
# rate limit 이 아닌 예외(인증 실패·잘못된 모델명 등)는 재시도 없이 즉시 전파한다.

def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _is_rate_limit(exc: Exception) -> bool:
    """rate limit(429/quota/RESOURCE_EXHAUSTED) 계열 예외인지 status/메시지로 느슨히 판별.
    provider별 예외 타입을 직접 import 하지 않기 위함."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 429:
        return True
    msg = str(exc).lower()
    markers = ("429", "too many requests", "rate limit", "ratelimit",
               "resource_exhausted", "resourceexhausted", "quota", "exceeded")
    return any(m in msg for m in markers)


def _run_with_retry(fn, label: str = "LLM"):
    """fn()을 호출하되 rate limit 예외면 EVAL_LLM_RETRY_WAIT초 대기 후 재시도.
    최대 EVAL_LLM_MAX_RETRIES회까지 재시도하고, 그래도 실패하면 마지막 예외를 전파한다."""
    max_retries = _env_int("EVAL_LLM_MAX_RETRIES", 5)
    wait = _env_float("EVAL_LLM_RETRY_WAIT", 5.0)
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:
            if not _is_rate_limit(e) or attempt >= max_retries:
                raise
            attempt += 1
            print(f"[Eval] {label} rate limit(429) — {wait:.0f}초 대기 후 재시도 ({attempt}/{max_retries})")
            time.sleep(wait)


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
    return resp.choices[0].message.content or ""


def _openai_embed(texts: list[str], model: str) -> list[list[float]]:
    from openai import OpenAI
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=texts)
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
    return resp.text or ""


def _gemini_embed(texts: list[str], model: str) -> list[list[float]]:
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    resp = client.models.embed_content(model=model, contents=texts)
    return [e.values for e in resp.embeddings]
