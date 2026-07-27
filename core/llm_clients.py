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

# LangSmith 트레이싱(선택). @traceable 은 LANGSMITH_TRACING 이 꺼져 있으면 스스로 no-op 이라
# 무조건 붙여도 켜지 않는 한 오버헤드가 없다. langsmith 패키지가 없을 때만 대비해
# 아무것도 안 하는 데코레이터로 폴백한다 — "키/패키지 없어도 파이프라인이 폴백 동작"
# 규약을 여기서도 지킨다.
try:
    from langsmith import traceable, get_current_run_tree
except ImportError:  # pragma: no cover - langsmith 미설치 환경 폴백
    def traceable(*d_args, **d_kwargs):
        def _decorator(fn):
            return fn
        # @traceable (인자 없이) / @traceable(run_type=...) 두 형태 모두 지원
        if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
            return d_args[0]
        return _decorator

    def get_current_run_tree():  # langsmith 없으면 붙일 트레이스도 없다
        return None

GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"


def _attach_usage(model: str, input_tokens: int, output_tokens: int) -> None:
    """현재 @traceable 트레이스에 토큰 사용량을 실어 LangSmith UI 의 Tokens/Cost 를 채운다.

    우리 코드는 usage 를 log_usage(콘솔)로만 흘려보내서 @traceable 이 토큰을 몰랐다 —
    그래서 UI 의 Tokens/Cost 가 빈칸이었다. LangSmith 규약 필드로 넘겨준다:
      - usage_metadata(토큰): Tokens 컬럼을 채운다.
      - ls_model_name(모델명): metadata 최상위에 둬야 LangSmith 가 내장 단가표
        (예: gemini-3.1-flash-lite BUILT-IN, $0.25/$1.50)와 매칭해 Cost 를 계산한다.
        usage_metadata 딕셔너리 '안'에 넣으면 매칭 위치가 아니라 Cost 가 빈칸으로 남는다.
    트레이싱 OFF/미설치면 get_current_run_tree()가 None 이라 조용히 skip(무해)."""
    run = get_current_run_tree()
    if run is None:
        return
    run.metadata["usage_metadata"] = {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int((input_tokens or 0) + (output_tokens or 0)),
    }
    run.metadata["ls_model_name"] = model  # ← 최상위(단가 매칭용). 위 dict 안이 아님.


@traceable(run_type="llm", name="openai_chat")
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
        _attach_usage(model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
    return resp.choices[0].message.content or ""


@traceable(run_type="llm", name="gemini_chat")
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
        _attach_usage(model, usage.prompt_token_count, out)
    return resp.text or ""


# NOTE: embed 함수에는 @traceable 을 붙이지 않는다 — 반환값이 1024차원 float 벡터 × N개라
# 트레이스 페이로드가 수 MB로 폭증해 전송이 끊기고(ConnectionAborted), 관측 가치도 낮다.
# 1단계 목적(LLM 프롬프트/응답/토큰/비용)은 chat 트레이스만으로 충분하다.
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
