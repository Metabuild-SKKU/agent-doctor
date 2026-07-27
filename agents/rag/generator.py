"""
RAG의 답변 생성 파트
retriever에서 찾은 chunk을 이용해서 답변 생성
LLM 설정 X, LLM 호출 실패 시 -> 추출식 fallback 사용
"""
from __future__ import annotations

import os
import threading
from dataclasses import asdict, dataclass
from typing import Any

from agents.rag.retriever import Retriever
from core.llm_clients import GITHUB_MODELS_BASE_URL, gemini_chat, openai_chat
from core.llm_retry import run_with_retry

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
    *,
    allow_common_model: bool = True,
) -> str | None:
    provider_key = provider_name.replace("_models", "")
    provider_model = _config_value(
        config,
        f"rag_{provider_key}_model",
        f"{provider_key}_model",
    )
    common_model = None
    if allow_common_model:
        common_model = (
            _config_value(config, "rag_llm_model", "llm_model", "response_model")
            or os.getenv("RAG_LLM_MODEL")
        )
    return (
        model
        or provider_model
        or common_model
    )


# API key X, provider가 선택되어 있어도 LLM 호출 X
def _github_api_key(*, allow_repo_token: bool = False) -> str | None:
    token = os.getenv("GITHUB_MODELS_TOKEN")
    if token:
        return token
    if allow_repo_token:
        return os.getenv("GITHUB_TOKEN")
    return None


def _has_provider(provider: str | None = None, config: dict | None = None) -> bool:
    selected = _selected_provider(provider, config)
    if selected in {"openai", "auto"} and os.getenv("OPENAI_API_KEY"):
        return True
    if selected in {"gemini", "auto"} and os.getenv("GEMINI_API_KEY"):
        return True
    if selected == "auto" and _github_api_key(allow_repo_token=False):
        return True
    if selected in {"github", "github_models"} and _github_api_key(allow_repo_token=True):
        return True
    return False


# 이미 경고한 미지원 provider 값(Eval/Optimize 가 스레드로 병렬 호출하므로 lock 으로 보호).
_warned_providers: set[str] = set()
_warned_providers_lock = threading.Lock()


def _warn_unknown_provider_once(selected: str) -> None:
    """미지원 provider 경고를 provider 값당 한 번만 출력한다."""
    with _warned_providers_lock:
        if selected in _warned_providers:
            return
        _warned_providers.add(selected)
    print(f"[RAG] 알 수 없는 provider '{selected}' — extractive 답변으로 폴백 "
          f"(RAG_LLM_PROVIDER: openai|gemini|github|auto)")


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
    if selected != "auto" and selected not in {"openai", "gemini", "github", "github_models"}:
        # 오타 등 미지원 값이면 아래 루프의 어떤 분기에도 안 걸려 로그 없이
        # extractive 로 저하되므로, 원인을 명시하고 바로 폴백한다.
        # 설정값 문제라 질문마다 같은 줄이 반복될 뿐이므로 provider 당 한 번만 —
        # Eval/Optimize 병렬 실행 시 수백 줄이 로그를 덮는 것을 막는다.
        _warn_unknown_provider_once(selected)
        return None
    providers = ["openai", "gemini", "github"] if selected == "auto" else [selected]
    system, user = _build_prompt(question, contexts, max_context_chars=max_context_chars)

    for name in providers:
        # config/env에서 다시 결정
        selected_model = _selected_model(
            name,
            model,
            config,
            allow_common_model=selected != "auto",
        )
        try:
            # rate limit(429)은 재시도 — 재시도 없이 폴백되면 병렬 실행 시 429 가
            # 조용히 추출식 답변으로 대체돼 평가 결과가 왜곡된다. 그 외 예외는 즉시
            # 다음 provider 폴백(기존 동작).
            if name == "openai" and os.getenv("OPENAI_API_KEY"):
                result = run_with_retry(
                    lambda: _openai_generate(system, user, model=selected_model, config=config),
                    "생성", tag="RAG")
                if result:
                    return result
            if name == "gemini" and os.getenv("GEMINI_API_KEY"):
                result = run_with_retry(
                    lambda: _gemini_generate(system, user, model=selected_model, config=config),
                    "생성", tag="RAG")
                if result:
                    return result
            if name in {"github", "github_models"}:
                # auto 모드에선 전용 토큰(GITHUB_MODELS_TOKEN)이 있을 때만 GitHub를 시도한다
                # (범용 GITHUB_TOKEN이 우연히 있다고 auto가 GitHub로 새지 않도록).
                if _github_api_key(allow_repo_token=selected != "auto"):
                    result = run_with_retry(
                        lambda: _github_generate(system, user, model=selected_model, config=config),
                        "생성", tag="RAG")
                    if result:
                        return result
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


# provider 별 transport 는 core/llm_clients.py 공용 구현에 위임한다.
# 여기 래퍼는 RAG 규약(키 없으면 None, 빈 응답 None, RAG_* 모델 env)만 담당.

# OpenAI (openai SDK)
def _openai_generate(
    system: str,
    user: str,
    *,
    model: str | None = None,
    config: dict | None = None,   # 미사용(호출부 시그니처 호환용)
) -> str | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    selected_model = model or os.getenv("RAG_OPENAI_MODEL", "gpt-4o")
    return openai_chat(system, user, selected_model, tag="RAG").strip() or None


# GitHub Models (OpenAI 호환 API, GitHub PAT 인증)
def _github_generate(
    system: str,
    user: str,
    *,
    model: str | None = None,
    config: dict | None = None,   # 미사용(호출부 시그니처 호환용)
) -> str | None:
    # 토큰 해석: GITHUB_MODELS_TOKEN 우선, 없으면 GITHUB_TOKEN.
    # (auto 모드에서 GITHUB_TOKEN으로 새는 것은 위 호출부 가드가 이미 막는다)
    api_key = _github_api_key(allow_repo_token=True)
    if not api_key:
        return None
    selected_model = model or os.getenv("RAG_GITHUB_MODEL", "openai/gpt-4o")
    return openai_chat(
        system, user, selected_model,
        api_key=api_key, base_url=GITHUB_MODELS_BASE_URL, tag="RAG",
    ).strip() or None


# Gemini (google-genai SDK)
def _gemini_generate(
    system: str,
    user: str,
    *,
    model: str | None = None,
    config: dict | None = None,   # base_url 등은 SDK가 처리하므로 현재 미사용(시그니처 호환용)
) -> str | None:
    if not os.getenv("GEMINI_API_KEY"):
        return None
    selected_model = model or os.getenv("RAG_GEMINI_MODEL", "gemini-flash-latest")
    return gemini_chat(system, user, selected_model, tag="RAG").strip() or None
