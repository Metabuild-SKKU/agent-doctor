"""
RAG의 답변 생성 파트 
retriever에서 찾은 chunk을 이용해서 답변 생성 
LLM 설정 X, LLM 호출 실패 시 -> 추출식 fallback 사용 
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

from agents.rag.retriever import Retriever

# 답변, context, citation, 검색 상세 정보 보관 
@dataclass
class GeneratedAnswer:
    question: str
    answer: str
    contexts: list[str]
    citations: list[dict]
    generation_mode: str
    retrieval: dict

# 1) LLM 답변 2) 관련 있는 context 그대로 반환 
# context 없으면 빈 문자열 반환 
def generate_answer(
    question: str,
    contexts: list[str],
    *,
    provider: str | None = None,
    model: str | None = None,
    config: dict | None = None,
    max_context_chars: int = 12000,
) -> str:
    """Generate an answer from contexts, falling back to the top context."""
    # 공백 context를 제거해서 LLM에 빈 근거가 들어가지 않도록 
    cleaned_contexts = [context.strip() for context in contexts if context and context.strip()]
    if not cleaned_contexts:
        return ""

    # LLM 답변 생성 -> 실패 시 none 반환, extractive fallback으로 넘어가기 
    answer = _llm_generate(
        question,
        cleaned_contexts,
        provider=provider,
        model=model,
        config=config,
        max_context_chars=max_context_chars,
    )
    if answer:
        return answer
    return _extractive_answer(cleaned_contexts)

# retriever 검색 ~ 답변 생성 (전체 RAG 흐름) 
def answer_question(
    question: str,
    retriever: Retriever,
    *,
    top_k: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    config: dict | None = None,
) -> dict:
    """Run retrieval + answer generation and return a JSON-ready payload."""
    # 질문으로 관련 chunk를 검색, 검색 방식과 fallback 반환 
    retrieval = retriever.search_with_details(question, top_k=top_k)

    # 검색 결과에서 답변 생성 -> 본문 text만 추출
    contexts = [item.get("text", "") for item in retrieval["results"]]

    # 검색된 context 기반 -> 최종 답변 
    answer = generate_answer(question, contexts, provider=provider, model=model, config=config)

    # 실제 LLM provider -> llm, X -> extractive 기록 
    generation_mode = "llm" if _has_provider(provider, config=config) else "extractive"

    # top context 그대로 반환 -> fallback
    if answer == _extractive_answer(contexts):
        generation_mode = "extractive"

    result = GeneratedAnswer(
        question=question,
        answer=answer,
        contexts=contexts,
        citations=_citations(retrieval["results"]),
        generation_mode=generation_mode,
        retrieval=retrieval,
    )
    return asdict(result)

"""
answer_text(): 최종 답변 문자열만 반환 
_citations(): 어떤 chunk/doc/section 근거로 답했는지 추적하기 위해 검색 결과를 citation 형태로 정리
_extractive_answer(): LLM 사용 불가, 가장 관련 있는 top context 반환 
_has_provider(): 사용할 수 있는 LLM provider가 설정되어 있는지 확인 
_llm_generate()
"""
def answer_text(
    question: str,
    retriever: Retriever,
    *,
    top_k: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    config: dict | None = None,
) -> str:
    """Run the full RAG flow and return only the answer string."""
    return answer_question(
        question,
        retriever,
        top_k=top_k,
        provider=provider,
        model=model,
        config=config,
    )["answer"]


# 검색 결과를 근거 정보 형태로 변환
# Eval/Serve에서 어떤 문서와 chunk를 보고 답했는지 확인할 때 사용
def _citations(results: list[dict]) -> list[dict]:
    citations = []
    for rank, item in enumerate(results, 1):
        metadata = item.get("metadata", {}) or {}
        # rank는 검색 결과 순서, chunk_id/doc_id/score는 추적용 metadata
        citations.append(
            {
                "rank": rank,
                "chunk_id": item.get("chunk_id", ""),
                "doc_id": item.get("doc_id", ""),
                "title": metadata.get("title") or item.get("doc_id", ""),
                "section": item.get("section"),
                "score": item.get("score"),
                "char_span": item.get("char_span"),
            }
        )
    return citations


def _extractive_answer(contexts: list[str]) -> str:
    for context in contexts:
        if context and context.strip():
            return context.strip()
    return ""


# config 안에서 여러 후보 key 중 첫 번째 유효 값 반환 (이름 달라도 설정 동일하게)
def _config_value(config: dict | None, *names: str) -> Any:
    if not config:
        return None
    for name in names:
        value = config.get(name)
        if value is not None and value != "":
            return value
    return None


# LLM provider 결정 
# 우선순위: provider > config 값 > 환경변수 > auto
def _selected_provider(provider: str | None = None, config: dict | None = None) -> str:
    return str(
        provider
        or _config_value(config, "rag_llm_provider", "llm_provider", "response_provider")
        or os.getenv("RAG_LLM_PROVIDER")
        or "auto"
    ).lower()


# LLM model 결정 
# 우선순위: 함수 인자 model > config 값 > 환경변수
def _selected_model(
    provider_name: str,
    model: str | None = None,
    config: dict | None = None,
) -> str | None:
    provider_key = provider_name.replace("_models", "")
    return (
        model
        or _config_value(
            config,
            "rag_llm_model",
            "llm_model",
            "response_model",
            f"rag_{provider_key}_model",
            f"{provider_key}_model",
        )
        or os.getenv("RAG_LLM_MODEL")
    )


# API key X, provider가 선택되어 있어도 LLM 호출 X
def _has_provider(provider: str | None = None, config: dict | None = None) -> bool:
    selected = _selected_provider(provider, config)
    if selected in {"openai", "auto"} and os.getenv("OPENAI_API_KEY"):
        return True
    if selected in {"gemini", "auto"} and os.getenv("GEMINI_API_KEY"):
        return True
    if selected in {"github", "github_models", "auto"} and (
        os.getenv("GITHUB_MODELS_TOKEN") or os.getenv("GITHUB_TOKEN")
    ):
        return True
    return False


# LLM provider를 선택 -> 실제 provider별 생성 함수를 호출
# auto 모드에서는 OpenAI -> Gemini -> GitHub 순서로 사용 가능한 provider 시도
# 모두 실패하면 None을 반환해 extractive fallback으로 넘어가기 
def _llm_generate(
    question: str,
    contexts: list[str],
    *,
    provider: str | None = None,
    model: str | None = None,
    config: dict | None = None,
    max_context_chars: int = 12000,
) -> str | None:
    selected = _selected_provider(provider, config)
    providers = ["openai", "gemini", "github"] if selected == "auto" else [selected]
    system, user = _build_prompt(question, contexts, max_context_chars=max_context_chars)

    for name in providers:
        # config/env에서 다시 결정
        selected_model = _selected_model(name, model, config)
        try:
            if name == "openai" and os.getenv("OPENAI_API_KEY"):
                return _openai_generate(system, user, model=selected_model, config=config)
            if name == "gemini" and os.getenv("GEMINI_API_KEY"):
                return _gemini_generate(system, user, model=selected_model, config=config)
            if name in {"github", "github_models"} and (
                os.getenv("GITHUB_MODELS_TOKEN") or os.getenv("GITHUB_TOKEN")
            ):
                return _github_generate(system, user, model=selected_model, config=config)
        except Exception as exc:
            print(f"[RAG] {name} generation failed, trying fallback: {exc}")
    return None


# LLM에 전달할 system/user prompt
def _build_prompt(
    question: str,
    contexts: list[str],
    *,
    max_context_chars: int,
) -> tuple[str, str]:
    context_block = ""
    for index, context in enumerate(contexts, 1):
        # context에 번호를 붙여 답변에서 근거 번호를 표시할 수 있게 한다.
        next_block = f"[{index}]\n{context.strip()}\n\n"
        if len(context_block) + len(next_block) > max_context_chars:
            break
        context_block += next_block

    # context 밖의 내용을 지어내지 않도록 답변 규칙을 고정한다.
    system = (
        "너는 사내 문서 QA 어시스턴트다. 반드시 제공된 컨텍스트만 근거로 한국어로 답하라. "
        "컨텍스트에 근거가 없으면 '제공된 정보로는 알 수 없습니다'라고 답하라. "
        "답변 끝에는 근거 번호를 대괄호로 표시하라."
    )
    user = f"[컨텍스트]\n{context_block.strip()}\n\n[질문]\n{question.strip()}"
    return system, user


def _openai_generate(
    system: str,
    user: str,
    *,
    model: str | None = None,
    config: dict | None = None,
) -> str | None:
    from openai import OpenAI

    client_kwargs: dict[str, Any] = {"api_key": os.getenv("OPENAI_API_KEY")}
    base_url = (
        _config_value(config, "rag_openai_base_url", "openai_base_url")
        or os.getenv("RAG_OPENAI_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
    )
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    response = client.chat.completions.create(
        model=model or os.getenv("RAG_OPENAI_MODEL", "gpt-4o"),
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = response.choices[0].message.content
    return content.strip() if content else None


# GitHub Models
def _github_generate(
    system: str,
    user: str,
    *,
    model: str | None = None,
    config: dict | None = None,
) -> str | None:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("GITHUB_MODELS_TOKEN") or os.getenv("GITHUB_TOKEN"),
        base_url=(
            _config_value(config, "rag_github_base_url", "github_models_base_url")
            or os.getenv("RAG_GITHUB_BASE_URL")
            or os.getenv("GITHUB_MODELS_BASE_URL")
            or "https://models.github.ai/inference"
        ),
    )
    response = client.chat.completions.create(
        model=model or os.getenv("RAG_GITHUB_MODEL", "openai/gpt-4o"),
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = response.choices[0].message.content
    return content.strip() if content else None


# Gemini REST API
def _gemini_generate(
    system: str,
    user: str,
    *,
    model: str | None = None,
    config: dict | None = None,
) -> str | None:
    import requests

    api_key = os.getenv("GEMINI_API_KEY")
    selected_model = model or os.getenv("RAG_GEMINI_MODEL", "gemini-1.5-flash")
    api_base = (
        _config_value(config, "rag_gemini_base_url", "gemini_base_url")
        or os.getenv("RAG_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    )
    url = f"{api_base}/models/{selected_model}:generateContent"
    response = requests.post(
        url,
        params={"key": api_key},
        json={
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{system}\n\n{user}"}],
                }
            ],
            "generationConfig": {"temperature": 0},
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    candidates = data.get("candidates") or []

    # 하나의 문자열로 합치기
    parts = (
        candidates[0]
        .get("content", {})
        .get("parts", [])
        if candidates
        else []
    )
    text = "".join(str(part.get("text", "")) for part in parts).strip()
    return text or None
