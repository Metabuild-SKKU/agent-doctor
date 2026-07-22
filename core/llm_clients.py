"""
core/llm_clients.py
LLM provider transport 공용 구현 (OpenAI / GitHub Models / Gemini).

agents/eval/llm_provider.py 와 agents/rag/generator.py 에 복붙돼 있던
"클라이언트 생성 → 호출 → usage 로깅" 계층만 모은다. provider 선택·폴백 체인,
env 규약(EVAL_LLM_PROVIDER vs RAG_LLM_PROVIDER), 키 부재 처리, 재시도 래핑은
호출하는 쪽 모듈이 그대로 가진다 — 여기는 "키가 이미 준비된 1회 호출"만 담당.
"""
from __future__ import annotations

import os

from core.llm_usage import log_usage

GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"


def openai_chat(
    system: str,
    user: str,
    model: str,
    *,
    json_mode: bool = False,
    api_key: str | None = None,
    base_url: str | None = None,
    tag: str = "LLM",
) -> str:
    """OpenAI 호환 chat 1회 호출(temperature=0) → 응답 텍스트("" 가능).

    base_url/api_key 를 주면 GitHub Models 등 OpenAI 호환 엔드포인트 겸용."""
    from openai import OpenAI

    client_kwargs = {}
    if base_url:
        client_kwargs["base_url"] = base_url
    if api_key:
        client_kwargs["api_key"] = api_key
    client = OpenAI(**client_kwargs)
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
        log_usage(model, resp.usage.prompt_tokens, resp.usage.completion_tokens, tag=tag)
    return resp.choices[0].message.content or ""


def gemini_chat(
    system: str,
    user: str,
    model: str,
    *,
    json_mode: bool = False,
    tag: str = "LLM",
) -> str:
    """Gemini chat 1회 호출(temperature=0, google-genai SDK) → 응답 텍스트("" 가능)."""
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config: dict = {"temperature": 0, "system_instruction": system}
    if json_mode:
        config["response_mime_type"] = "application/json"
    resp = client.models.generate_content(model=model, contents=user, config=config)
    usage = getattr(resp, "usage_metadata", None)
    if usage:
        # 과금되는 출력 = 답변(candidates) + 내부 사고(thoughts). 추론 모델(3.5-flash 등)은
        # thoughts 가 답변보다 클 수 있어 candidates 만 세면 비용이 절반 이하로 과소집계된다.
        out = (usage.candidates_token_count or 0) + (getattr(usage, "thoughts_token_count", 0) or 0)
        log_usage(model, usage.prompt_token_count, out, tag=tag)
    return resp.text or ""


def openai_embed(texts: list[str], model: str, *, tag: str = "LLM") -> list[list[float]]:
    """OpenAI embeddings 1회 호출 → 벡터 리스트(입력 순서 유지)."""
    from openai import OpenAI

    resp = OpenAI().embeddings.create(model=model, input=texts)
    if resp.usage:
        log_usage(model, resp.usage.prompt_tokens, 0, tag=tag)
    return [d.embedding for d in resp.data]


def gemini_embed(texts: list[str], model: str, *, tag: str = "LLM") -> list[list[float]]:
    """Gemini embed_content 1회 호출 → 벡터 리스트(입력 순서 유지)."""
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    resp = client.models.embed_content(model=model, contents=texts)
    return [e.values for e in resp.embeddings]
