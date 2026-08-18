"""
RAG의 답변 생성 파트
retriever에서 찾은 chunk을 이용해서 답변 생성
LLM 설정 X, LLM 호출 실패 시 -> 추출식 fallback 사용
"""
from __future__ import annotations

import os
import re
import threading
from dataclasses import asdict, dataclass
from typing import Any

from agents.rag.retriever import Retriever
from core.llm_clients import (
    GITHUB_MODELS_BASE_URL,
    OPENROUTER_BASE_URL,
    PROVIDER_ALIASES,
    normalize_provider,
    gemini_chat,
    openai_chat,
)
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


@dataclass(frozen=True)
class PromptContext:
    citation_index: int
    text: str

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

    prompt_contexts = _compress_contexts_for_question(question, cleaned_contexts, config=config)

    # LLM 답변 생성 -> 실패 시 none 반환, extractive fallback으로 넘어가기
    answer = _llm_generate(
        question,
        prompt_contexts,
        provider=provider,
        model=model,
        config=config,
        max_context_chars=max_context_chars,
    )
    if answer:
        return answer
    return _extractive_answer([context.text for context in prompt_contexts])

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
    cleaned_contexts = [
        context.strip() for context in contexts if context and context.strip()
    ]
    prompt_contexts = _compress_contexts_for_question(
        question,
        cleaned_contexts,
        config=config,
    )
    if answer == _extractive_answer([context.text for context in prompt_contexts]):
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


_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "what", "when", "where", "how",
    "에", "의", "을", "를", "이", "가", "은", "는", "와", "과", "로", "으로",
    "및", "또는", "그리고", "대해", "설명", "하십시오", "주세요",
}


def _compress_contexts_for_question(
    question: str,
    contexts: list[str],
    *,
    config: dict | None = None,
) -> list[PromptContext]:
    """Keep question-relevant sentences before generation when explicitly enabled."""
    original = [
        PromptContext(citation_index=index, text=context)
        for index, context in enumerate(contexts, 1)
    ]

    if not _context_compression_enabled(config):
        return original

    query_tokens = _keywords(question)
    query_numbers = set(_NUMBER_RE.findall(question))
    if not query_tokens and not query_numbers:
        return original

    max_sentences = _positive_int(
        _config_value(config, "context_compression_max_sentences", "context_filter_max_sentences"),
        4,
    )
    raw_max_contexts = _config_value(
        config,
        "context_compression_max_contexts",
        "context_filter_max_contexts",
    )
    max_contexts = (
        _positive_int(raw_max_contexts, len(contexts))
        if raw_max_contexts is not None
        else None
    )
    min_contexts = min(
        len(contexts),
        _positive_int(
            _config_value(config, "context_compression_min_contexts", "context_filter_min_contexts"),
            1,
        ),
    )

    compressed: list[tuple[float, bool, PromptContext]] = []
    matched_any = False
    matched_count = 0
    for context_index, context in enumerate(contexts):
        sentences = _split_sentences(context)
        scored: list[tuple[float, int, str]] = []
        for sentence_index, sentence in enumerate(sentences):
            score = _sentence_relevance_score(sentence, query_tokens, query_numbers)
            if score > 0:
                scored.append((score, sentence_index, sentence))

        if scored:
            matched_any = True
            matched_count += 1
            best_score = max(item[0] for item in scored)
            picked = sorted(
                sorted(scored, key=lambda item: item[0], reverse=True)[:max_sentences],
                key=lambda item: item[1],
            )
            compressed.append(
                (
                    best_score,
                    True,
                    PromptContext(
                        citation_index=context_index + 1,
                        text=" ".join(item[2] for item in picked),
                    ),
                )
            )
        else:
            fallback_text = " ".join(sentences[:max_sentences])
            compressed.append(
                (
                    0.0,
                    False,
                    PromptContext(citation_index=context_index + 1, text=fallback_text),
                )
            )

    if not matched_any:
        return original

    target_contexts = max_contexts if max_contexts is not None else matched_count
    if compressed and not compressed[0][1] and matched_count > 0 and len(compressed) > 1:
        target_contexts = max(target_contexts, 2)
    target_contexts = min(len(compressed), max(min_contexts, target_contexts))
    compressed = sorted(
        compressed,
        key=lambda item: (
            item[2].citation_index != 1,
            -item[0],
            not item[1],
            item[2].citation_index,
        ),
    )[:target_contexts]
    compressed = sorted(compressed, key=lambda item: item[2].citation_index)
    return [context for _score, _matched, context in compressed if context.text.strip()] or original


def _context_compression_enabled(config: dict | None = None) -> bool:
    value = _config_value(
        config,
        "context_compression",
        "context_filtering",
        "context.compression.enabled",
    )
    if value is None:
        value = os.getenv("RAG_CONTEXT_COMPRESSION")
    return _as_bool(value)


def _split_sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
    return sentences or [text.strip()]


def _sentence_relevance_score(
    sentence: str,
    query_tokens: set[str],
    query_numbers: set[str],
) -> float:
    sentence_tokens = _keywords(sentence)
    overlap = len(query_tokens & sentence_tokens)
    sentence_numbers = set(_NUMBER_RE.findall(sentence))
    number_overlap = len(query_numbers & sentence_numbers)
    return float(overlap) + (2.0 * number_overlap)


def _keywords(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in _TOKEN_RE.findall(text or ""):
        for token in _expand_keyword(raw.lower()):
            if len(token) > 1 and token not in _STOPWORDS:
                tokens.add(token)
    return tokens


def _expand_keyword(token: str) -> set[str]:
    variants = {token}
    stripped = _strip_korean_suffix(token)
    variants.add(stripped)
    if re.search(r"[가-힣]", stripped):
        variants.update(_hangul_ngrams(stripped))
    return {variant for variant in variants if variant}


def _strip_korean_suffix(token: str) -> str:
    suffixes = (
        "으로부터", "로부터", "까지", "부터", "에서는", "에서", "에게", "께서",
        "으로", "라고", "이라", "이며", "이고", "입니다", "입니까", "합니다",
        "하세요", "되는", "된다", "되며", "된다면", "은", "는", "이", "가",
        "을", "를", "의", "에", "도", "만", "와", "과", "로",
    )
    for suffix in suffixes:
        if token.endswith(suffix) and len(token) > len(suffix) + 1:
            return token[: -len(suffix)]
    return token


def _hangul_ngrams(token: str) -> set[str]:
    compact = re.sub(r"[^0-9A-Za-z가-힣]", "", token)
    if len(compact) < 3:
        return set()
    ngrams: set[str] = set()
    for size in (2, 3, 4):
        if len(compact) < size:
            continue
        ngrams.update(compact[index:index + size] for index in range(len(compact) - size + 1))
    return ngrams


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


# config 안에서 여러 후보 key 중 첫 번째 유효 값 반환 (이름 달라도 설정 동일하게)
def _config_value(config: dict | None, *names: str) -> Any:
    if not config:
        return None
    for name in names:
        value = config.get(name)
        if value is not None and value != "":
            return value
    return None


# 같은 문자열을 RAG_LLM_PROVIDER 와 EVAL_LLM_PROVIDER 어느 쪽에 넣어도 동작하도록 맞춘
# 철자표. agents/eval/llm_provider.py 의 _PROVIDER_ALIASES 와 항상 같은 값을 유지할 것
# (tests/test_provider_notices.py 가 두 표의 일치를 핀으로 잡는다).
# 한쪽만 받아주면 "OpenRouter 로 다 바꿨다"고 믿은 실행에서 심판은 OpenRouter 로 돌고
# 답변 생성만 extractive 로 저하돼, 평가가 저하된 RAG 를 재는 상태가 된다.
_PROVIDER_ALIASES = PROVIDER_ALIASES


# LLM provider 결정
# 우선순위: provider > config 값 > 환경변수 > auto
# 여기서 철자를 정규화하므로 아래 _has_provider()/_llm_generate() 는 정규형만 본다.
def _selected_provider(provider: str | None = None, config: dict | None = None) -> str:
    selected = str(
        provider
        or _config_value(config, "rag_llm_provider", "llm_provider", "response_provider")
        or os.getenv("RAG_LLM_PROVIDER")
        or "auto"
    ).strip().lower()
    return _PROVIDER_ALIASES.get(selected, selected)


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
    if selected in {"openrouter", "auto"} and os.getenv("OPENROUTER_API_KEY"):
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
          f"(RAG_LLM_PROVIDER: openai|gemini|github|openrouter|auto)")


# auto 는 OpenAI 를 먼저 시도한다 — 다른 용도로 OPENAI_API_KEY 를 넣으면 Gemini 로 돌던
# 답변 생성이 말없이 OpenAI 로 옮겨간다. 그 전환만 프로세스당 한 번 알린다
# (질문마다 찍으면 Eval/Optimize 병렬 실행에서 수백 줄이 된다).
# 시도 시점이 아니라 첫 성공 직후에 찍는다 — 만료된 키면 실제로는 Gemini 로
# 폴백하는데 "OpenAI 로 답했다"는 줄만 남는다.
_notified_auto_openai = False
_auto_openai_notice_lock = threading.Lock()


def _notify_auto_openai_once() -> None:
    global _notified_auto_openai
    with _auto_openai_notice_lock:
        if _notified_auto_openai:
            return
        _notified_auto_openai = True
    print("[RAG] auto → OpenAI 로 답변 생성 중(OPENAI_API_KEY 감지). "
          "고정하려면 RAG_LLM_PROVIDER=gemini|openai|github|openrouter")


# LLM provider를 선택 -> 실제 provider별 생성 함수를 호출
# auto 모드에서는 OpenAI -> Gemini -> GitHub -> OpenRouter 순서로 사용 가능한 provider 시도
# (OpenRouter를 맨 뒤에 둔 것은 기존 우선순위를 그대로 보존하기 위함 — 키를 안 넣은
#  사용자에겐 동작이 전혀 바뀌지 않는다. 고정하려면 RAG_LLM_PROVIDER=openrouter)
# 모두 실패하면 None을 반환해 extractive fallback으로 넘어가기
def _llm_generate(
    question: str,
    contexts: list[str] | list[PromptContext],
    *,
    provider: str | None = None,
    model: str | None = None,
    config: dict | None = None,
    max_context_chars: int = 12000,
) -> str | None:
    selected = _selected_provider(provider, config)
    if selected != "auto" and selected not in {
        "openai", "gemini", "github", "github_models", "openrouter",
    }:
        # 오타 등 미지원 값이면 아래 루프의 어떤 분기에도 안 걸려 로그 없이
        # extractive 로 저하되므로, 원인을 명시하고 바로 폴백한다.
        # 설정값 문제라 질문마다 같은 줄이 반복될 뿐이므로 provider 당 한 번만 —
        # Eval/Optimize 병렬 실행 시 수백 줄이 로그를 덮는 것을 막는다.
        _warn_unknown_provider_once(selected)
        return None
    providers = (
        ["openai", "gemini", "github", "openrouter"] if selected == "auto" else [selected]
    )
    system, user = _build_prompt(
        question, contexts, max_context_chars=max_context_chars, config=config
    )

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
                    if selected == "auto":
                        _notify_auto_openai_once()
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
            if name == "openrouter" and os.getenv("OPENROUTER_API_KEY"):
                result = run_with_retry(
                    lambda: _openrouter_generate(system, user, model=selected_model, config=config),
                    "생성", tag="RAG")
                if result:
                    return result
        except Exception as exc:
            print(f"[RAG] {name} generation failed, trying fallback: {exc}")
    return None


# LLM에 전달할 system/user prompt
def _gen_flag(config: dict | None, name: str, default: bool) -> bool:
    """generation_config 의 bool 플래그를 읽는다(문자열 'true' 등도 정규화). 없으면 default."""
    if not config or name not in config:
        return default
    value = config.get(name)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


# 답변 언어. 기본은 한국어 — 기존 KorQuAD 실행이 그대로 돌아야 한다.
#
# **왜 손잡이가 필요한가.** 언어가 프롬프트에 박혀 있으면 영어 코퍼스에서 질문은 영어인데
# 답변만 한국어로 나오고, char-F1(문자 겹침)이 **구조적으로 0** 이 된다. 그러면 _is_success
# 의 lexical 축이 항상 실패라 **검색이 완벽해도 실패로 집계된다**(실측: DragonBall probe 에서
# recall=1.00·gold 2/2 검색인데 f1=0.00 → generation_hallucination 오진).
#
# **한 문장만 바꾸는 걸론 안 된다 — 실측으로 확인했다.** 처음엔 지시문 한 조각만
# ("질문과 같은 언어로") 갈아끼우고 "언어는 모델이 알아서 맞춘다" 고 뒀는데, 나머지 프롬프트
# (역할 문장·기권 지시·근거 표시 지시·[컨텍스트]/[질문] 머리말)가 전부 한국어라 모델이
# 프롬프트 언어를 따라갔다. DragonBall 10건 재실행에서 영어 질문 10건이 **전부 한국어로**
# 답했고, 기권 답변은 예시 문구('제공된 정보로는 알 수 없습니다')를 글자 그대로 복사했다.
# 그래서 조각이 아니라 **프롬프트 전체를 언어별로** 둔다.
#
# "match" 는 질문의 문자 구성으로 고른다. 처음엔 감지기를 피하려고 모델에 맡겼지만 위처럼
# 실패했고, 판정은 '한글이 있나' 하나라 오판 여지가 거의 없다(정답 채점 단위를 고르는
# metrics_basic.scoring_unit 과 같은 방식이다).

_PROMPT_KO = {
    "role": "너는 사내 문서 QA 어시스턴트다.",
    "grounded": ("반드시 제공된 컨텍스트만 근거로 한국어로 답하라. "
                 "컨텍스트에 근거가 없으면 '제공된 정보로는 알 수 없습니다'라고 답하라."),
    "loose": "한국어로 답하라.",
    "abstention_relaxed": ("컨텍스트에 질문과 관련된 근거가 조금이라도 있으면 그것을 바탕으로 "
                           "최대한 답하라. 명백히 근거가 전혀 없을 때에만 "
                           "'제공된 정보로는 알 수 없습니다'라고 답하라."),
    "abstention_strict": ("조금이라도 확신이 없으면 지어내지 말고 반드시 "
                          "'제공된 정보로는 알 수 없습니다'라고 답하라."),
    "completeness": "질문에 여러 항목이나 하위 질문이 있으면 빠짐없이 모두 답하라.",
    "restate": "답변을 시작하기 전에 질문을 한 문장으로 재진술하라.",
    "citation": "답변 끝에는 근거 번호를 대괄호로 표시하라.",
    "context_header": "[컨텍스트]",
    "question_header": "[질문]",
}

# 기권 문구는 metrics_basic._ABSTENTION_MARKERS 에 등록된 표현이어야 한다 — 아니면 모델이
# 기권했는데 파이프라인이 '답했다' 로 읽어 generation_* 라벨이 통째로 어긋난다.
# "cannot answer" 가 등록돼 있어 아래 문구가 그대로 잡힌다(테스트가 이 계약을 고정한다).
_PROMPT_EN = {
    "role": "You are a QA assistant for internal company documents.",
    "grounded": ("Answer in English using only the provided context. "
                 "If the context does not support an answer, reply "
                 "'I cannot answer based on the provided information.'"),
    "loose": "Answer in English.",
    "abstention_relaxed": ("If the context contains any evidence related to the question, "
                           "answer as fully as you can from it. Only when there is clearly "
                           "no evidence at all, reply "
                           "'I cannot answer based on the provided information.'"),
    "abstention_strict": ("If you are not fully certain, do not make anything up; reply "
                          "'I cannot answer based on the provided information.'"),
    "completeness": ("If the question has several parts or sub-questions, "
                     "answer every one of them."),
    "restate": "Restate the question in one sentence before you begin your answer.",
    "citation": "End your answer with the supporting source numbers in square brackets.",
    "context_header": "[Context]",
    "question_header": "[Question]",
}

_PROMPTS = {"ko": _PROMPT_KO, "en": _PROMPT_EN}


def _answer_language(question: str, config: dict | None) -> str:
    """답변 언어 키("ko"|"en"). config > env(RAG_ANSWER_LANGUAGE) > 기본(한국어).

    "match" 는 질문에 한글이 있는지로 고른다 — 한글이 하나라도 있으면 한국어다.
    """
    raw = _config_value(config, "answer_language")
    if raw is None:
        raw = os.getenv("RAG_ANSWER_LANGUAGE", "")
    key = str(raw or "").strip().lower()
    if key == "match":
        return "ko" if _has_hangul(question or "") else "en"
    return key if key in _PROMPTS else "ko"


def _has_hangul(text: str) -> bool:
    """한글 포함 여부 — **채점 단위 판정과 같은 구현을 쓴다.**

    주석에서 "metrics_basic.scoring_unit 과 같은 방식" 이라고 선언한 계약을 코드로 묶는다.
    복제해 두면 한쪽만 한글 블록을 넓혔을 때 혼합 문자 질문에서 생성 언어와 채점 단위가
    갈리고, 이 파일이 고친 f1=0 오진이 그 부분집합에서 조용히 재현된다(어떤 테스트도 못 잡음).
    """
    from agents.eval.metrics_basic import has_hangul
    return has_hangul(text)


def _build_prompt(
    question: str,
    contexts: list[str] | list[PromptContext],
    *,
    max_context_chars: int,
    config: dict | None = None,
) -> tuple[str, str]:
    context_block = ""
    for index, context in enumerate(contexts, 1):
        if isinstance(context, PromptContext):
            citation_index = context.citation_index
            context_text = context.text
        else:
            citation_index = index
            context_text = context
        # context에 번호를 붙여 답변에서 근거 번호를 표시할 수 있게 한다.
        next_block = f"[{citation_index}]\n{context_text.strip()}\n\n"
        if len(context_block) + len(next_block) > max_context_chars:
            break
        context_block += next_block

    # generation_config 플래그로 시스템 프롬프트를 조립한다. rules.py 처방은 플래그만
    # 넘기고 실제 문구는 여기서 소유한다(use_hybrid/use_reranker와 같은 패턴).
    # 기본값(grounding_strict·require_citation=True, 나머지 False)은 과거 하드코딩
    # 프롬프트를 그대로 재현하므로, Optimize가 손대기 전엔 동작이 바뀌지 않는다.
    text = _PROMPTS[_answer_language(question, config)]
    clauses = [text["role"]]
    if _gen_flag(config, "grounding_strict", True):
        clauses.append(text["grounded"])
    else:
        clauses.append(text["loose"])
    # 기권 성향은 한 축의 양방향이라 둘을 동시에 싣지 않는다.
    #
    # 어느 쪽이 이길지는 여기서 정하지 않는다 — 처방 적용 시점에 config_mapper 의
    # EXCLUSIVE_FLAG_PAIRS 가 반대쪽을 내려서(마지막에 쓴 쪽이 이긴다) 정상 경로에서는
    # 애초에 한쪽만 True 로 들어온다. 아래 elif 는 손으로 고친 config 등 그 경로를 안 거친
    # 입력에 대한 백스톱일 뿐이다. 여기서 우선순위로 푸는 건 안 된다: 진 쪽 처방이 영구히
    # no-op 이 되어 롤백 + blacklist 로 사라지고, 그 라벨의 레버가 죽는다.
    if _gen_flag(config, "abstention_relaxed", False):
        clauses.append(text["abstention_relaxed"])
    elif _gen_flag(config, "abstention_strict", False):
        clauses.append(text["abstention_strict"])
    if _gen_flag(config, "completeness_mode", False):
        clauses.append(text["completeness"])
    if _gen_flag(config, "restate_question", False):
        clauses.append(text["restate"])
    if _gen_flag(config, "require_citation", True):
        clauses.append(text["citation"])
    system = " ".join(clauses)
    # 머리말도 함께 바꾼다 — 지시문만 영어로 두고 사용자 메시지가 한국어면 모델이 다시
    # 프롬프트 언어를 따라간다(실측에서 그렇게 무너졌다).
    user = (f"{text['context_header']}\n{context_block.strip()}\n\n"
            f"{text['question_header']}\n{question.strip()}")
    return system, user


# provider 별 transport 는 core/llm_clients.py 공용 구현에 위임한다.
# 여기 래퍼는 RAG 규약(키 없으면 None, 빈 응답 None, RAG_* 모델 env)만 담당.

# OpenAI (openai SDK)
def _generation_temperature(config: dict | None) -> float:
    """generation_config 의 temperature(기본 0.0). 파싱 실패 시 0.0."""
    raw = _config_value(config, "temperature")
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


# 답변 생성 출력 상한. 공용 기본값(2048)은 컨텍스트가 긴 코퍼스에서 부족했다 —
# KorQuAD 위키 문서(청크 39~50개)로 돌린 실행에서 solar-pro-3 답변이
# finish_reason=length 로 잘렸다. 잘린 답변은 그대로 채점에 들어가 faithfulness·
# answer_correctness 를 부당하게 떨어뜨리므로, 진단 품질에 직접 영향을 준다.
# 상한은 실사용분만 과금되니 올려도 정상 답변의 비용은 그대로다.
DEFAULT_GENERATION_MAX_TOKENS = 4096


def _generation_max_tokens(config: dict | None) -> int:
    """답변 생성 출력 상한. config > env(RAG_MAX_OUTPUT_TOKENS) > 기본값."""
    raw = _config_value(config, "generation_max_tokens", "max_output_tokens")
    if raw is None:
        raw = os.getenv("RAG_MAX_OUTPUT_TOKENS")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_GENERATION_MAX_TOKENS
    return value if value > 0 else DEFAULT_GENERATION_MAX_TOKENS


def _openai_generate(
    system: str,
    user: str,
    *,
    model: str | None = None,
    config: dict | None = None,
) -> str | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    selected_model = model or os.getenv("RAG_OPENAI_MODEL", "gpt-4o")
    return openai_chat(
        system, user, selected_model,
        max_output_tokens=_generation_max_tokens(config),
        temperature=_generation_temperature(config), tag="RAG",
    ).strip() or None


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
        api_key=api_key, base_url=GITHUB_MODELS_BASE_URL,
        max_output_tokens=_generation_max_tokens(config),
        temperature=_generation_temperature(config), tag="RAG",
    ).strip() or None


# OpenRouter (OpenAI 호환 API, 단일 키로 다수 publisher 모델)
def _openrouter_generate(
    system: str,
    user: str,
    *,
    model: str | None = None,
    config: dict | None = None,
) -> str | None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    # 모델명은 "publisher/model" 형식이어야 한다(예: anthropic/claude-sonnet-4.5).
    # 형식이 틀리면 404가 나고 호출부가 다음 provider로 폴백한다.
    selected_model = model or os.getenv("RAG_OPENROUTER_MODEL", "openai/gpt-4o")
    return openai_chat(
        system, user, selected_model,
        api_key=api_key, base_url=OPENROUTER_BASE_URL,
        max_output_tokens=_generation_max_tokens(config),
        temperature=_generation_temperature(config), tag="RAG",
    ).strip() or None


# Gemini (google-genai SDK)
def _gemini_generate(
    system: str,
    user: str,
    *,
    model: str | None = None,
    config: dict | None = None,
) -> str | None:
    if not os.getenv("GEMINI_API_KEY"):
        return None
    selected_model = model or os.getenv("RAG_GEMINI_MODEL", "gemini-flash-latest")
    return gemini_chat(
        system, user, selected_model,
        max_output_tokens=_generation_max_tokens(config),
        temperature=_generation_temperature(config), tag="RAG",
    ).strip() or None
