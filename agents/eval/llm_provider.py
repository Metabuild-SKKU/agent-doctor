"""
agents/eval/llm_provider.py
Eval Agent 가 쓰는 LLM 호출(OpenAI/Gemini)을 provider 하나로 추상화한다.

OpenAI API 토큰 승인 전까지 EVAL_LLM_PROVIDER=gemini 로 Google AI Studio 무료
Gemini API를 임시로 쓰기 위한 다리. 토큰 승인 후에는 EVAL_LLM_PROVIDER=openai
(기본값)로 되돌리거나 env 변수를 지우면 원래 동작으로 복귀한다.
"""
from __future__ import annotations

import json
import os


def _provider() -> str:
    return os.getenv("EVAL_LLM_PROVIDER", "openai").strip().lower()


def has_key() -> bool:
    """활성 provider의 API 키가 설정돼 있는지."""
    if _provider() == "gemini":
        return bool(os.getenv("GEMINI_API_KEY"))
    return bool(os.getenv("OPENAI_API_KEY"))


# ── 답변 생성 (retrieval_temp.py 가 사용) ─────────────────────────

def generate_text(system: str, user: str, model: str | None = None) -> str | None:
    """일반 텍스트 응답 생성. 키/라이브러리 없거나 호출 실패 시 None."""
    if not has_key():
        return None
    try:
        if _provider() == "gemini":
            text = _gemini_generate(
                system, user, model or os.getenv("EVAL_GEN_MODEL_GEMINI", "gemini-2.5-flash"))
        else:
            text = _openai_generate(
                system, user, model or os.getenv("EVAL_GEN_MODEL", "gpt-4o-mini"))
        return (text or "").strip()
    except ImportError:
        return None
    except Exception as e:
        print(f"[Eval] LLM 생성 실패({e}) → 추출식 폴백")
        return None


# ── JSON 강제 채점 호출 (ragas_eval.py 가 사용) ───────────────────

def chat_json(system: str, user: str, model: str | None = None) -> dict:
    """JSON 응답 강제 chat 호출 → dict. JSON 파싱 실패 시 {} (API 예외는 호출부로 전파)."""
    if _provider() == "gemini":
        raw = _gemini_generate(
            system, user, model or os.getenv("EVAL_JUDGE_MODEL_GEMINI", "gemini-2.5-flash"),
            json_mode=True)
    else:
        raw = _openai_generate(
            system, user, model or os.getenv("EVAL_JUDGE_MODEL", "gpt-4o"),
            json_mode=True)
    try:
        obj = json.loads(raw or "{}")
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


# ── 임베딩 (ragas_eval.py 가 사용) ────────────────────────────────

def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """텍스트 리스트 → 임베딩 벡터 리스트. (API 예외는 호출부로 전파)"""
    if _provider() == "gemini":
        return _gemini_embed(texts, model or os.getenv("EVAL_EMBED_MODEL_GEMINI", "text-embedding-004"))
    return _openai_embed(texts, model or os.getenv("EVAL_EMBED_MODEL", "text-embedding-3-small"))


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
