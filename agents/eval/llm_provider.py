"""
agents/eval/llm_provider.py
Eval Agent 가 쓰는 LLM 호출(OpenAI/Gemini/GitHub Models/OpenRouter)을 provider 하나로 추상화한다.

OpenAI API 토큰 승인 전까지 무료 대체 provider 로 브릿지한다:
    EVAL_LLM_PROVIDER=gemini  → Google AI Studio 무료 Gemini API
    EVAL_LLM_PROVIDER=github  → GitHub Models(무료, GitHub PAT 인증, OpenAI 호환 API)
토큰 승인 후에는 EVAL_LLM_PROVIDER=openai(기본값)로 되돌리거나 env 변수를
지우면 원래 동작으로 복귀한다.

    EVAL_LLM_PROVIDER=openrouter → OpenRouter(유료, 다수 모델 단일 키, OpenAI 호환 API)
OpenRouter 는 브릿지가 아니라 상시 사용을 전제한 provider 다. 모델명은 "publisher/model"
형식(예: anthropic/claude-sonnet-4.5)을 쓴다. 임베딩 엔드포인트는 제공하지 않는다
(embed_texts 주석 참고).
"""
from __future__ import annotations

import json
import os
import threading

from core.llm_clients import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    GITHUB_MODELS_BASE_URL,
    OPENROUTER_BASE_URL,
    gemini_chat,
    gemini_embed,
    openai_chat,
    openai_embed,
)
from core.llm_retry import run_with_retry


# 새 provider 를 추가할 때는 transport(_*_generate)·has_key()·여기 셋을 함께 고쳐야 한다.
# 여기 빠뜨리면 그 provider 는 "미지원 값"으로 판정돼 openai 로 폴백하고 transport 는
# 도달 불가 코드가 된다 — 파일 내 위치가 달라 git 이 충돌로 잡아주지 않는 실수다.
_KNOWN_PROVIDERS = {"openai", "gemini", "github", "openrouter"}
# 같은 문자열을 EVAL_LLM_PROVIDER 와 RAG_LLM_PROVIDER 어느 쪽에 넣어도 동작하도록 맞춘
# 철자표. agents/rag/generator.py 의 _PROVIDER_ALIASES 와 항상 같은 값을 유지할 것
# (tests/test_provider_notices.py 가 두 표의 일치를 핀으로 잡는다).
_PROVIDER_ALIASES = {
    "github_models": "github",
    "open_router": "openrouter",
    "open-router": "openrouter",
    "openrouter_ai": "openrouter",
    "openrouter.ai": "openrouter",
}
# 이미 경고한 미지원 provider 값(Eval 은 스레드로 병렬 호출하므로 lock 으로 보호).
_warned_providers: set[str] = set()
_warned_providers_lock = threading.Lock()


def _warn_unknown_provider_once(raw: str) -> None:
    """미지원 EVAL_LLM_PROVIDER 값 경고를 값당 한 번만 출력한다."""
    with _warned_providers_lock:
        if raw in _warned_providers:
            return
        _warned_providers.add(raw)
    print(f"[Eval] 알 수 없는 EVAL_LLM_PROVIDER '{raw}' — openai 로 폴백 "
          f"(openai|gemini|github|openrouter)")


def _provider() -> str:
    """활성 provider. 오타 등 미지원 값은 openai 로 떨어지므로 경고를 남긴다 —
    Gemini/OpenRouter 로 돌린다고 믿은 실행이 조용히 OpenAI 로 과금되는 걸 막기 위함."""
    raw = os.getenv("EVAL_LLM_PROVIDER", "openai").strip().lower()
    if not raw:  # 빈 값은 "기본값" 의사표시로 보고 경고하지 않는다.
        return "openai"
    raw = _PROVIDER_ALIASES.get(raw, raw)
    if raw not in _KNOWN_PROVIDERS:
        _warn_unknown_provider_once(raw)
        return "openai"
    return raw


def has_key() -> bool:
    """활성 provider의 API 키가 설정돼 있는지."""
    provider = _provider()
    if provider == "gemini":
        return bool(os.getenv("GEMINI_API_KEY"))
    if provider == "github":
        return bool(os.getenv("GITHUB_TOKEN"))
    if provider == "openrouter":
        return bool(os.getenv("OPENROUTER_API_KEY"))
    return bool(os.getenv("OPENAI_API_KEY"))


# ── rate limit(429) 재시도 (core/llm_retry.py 공용 구현) ──────────
# env(EVAL_LLM_RETRY_WAIT/EVAL_LLM_MAX_RETRIES)·jitter 동작은 core 쪽 참고.

def _run_with_retry(fn, label: str = "LLM"):
    return run_with_retry(fn, label, tag="Eval")


# ── JSON 강제 채점 호출 (probe_gen.py / metrics_ragas.py 가 사용) ─
# 참고: STEP2 답변 생성은 이 모듈이 아니라 agents/rag/generator.py 가 담당한다
# (그쪽은 RAG_LLM_PROVIDER / RAG_*_MODEL 계열 env 를 쓴다).

def chat_json(
    system: str,
    user: str,
    model: str | None = None,
    *,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict:
    """JSON 응답 강제 chat 호출 → dict. 실패 시 {} (API 예외는 호출부로 전파).

    빈 응답 / 파싱 실패 / 타입 불일치는 사유별 로그를 남기고 {} 를 돌려준다. Gemini 가
    dict 를 [ {…} ] 로 감싸 반환한 경우(길이 1 리스트)는 언랩해 dict 로 돌려준다.
    상한에 걸려 잘린 응답은 JSON 파싱에 실패해 {} 가 되므로, 구조가 큰 응답을 기대하는
    쪽은 max_output_tokens 를 올려 잡을 것."""
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
        elif provider == "openrouter":
            # 주의: response_format=json_object 지원은 OpenRouter 에서 모델마다 다르다.
            # 미지원 모델을 쓰면 파싱 실패 → 아래 {} 폴백으로 조용히 흘러가므로,
            # 심판 모델은 JSON 모드를 지원하는 것으로 고를 것.
            return _openrouter_generate(
                system, user, model or os.getenv("EVAL_JUDGE_MODEL_OPENROUTER", "openai/gpt-4o"),
                json_mode=True, max_output_tokens=max_output_tokens)
        return _openai_generate(
            system, user, model or os.getenv("EVAL_JUDGE_MODEL", "gpt-4o"),
            json_mode=True, max_output_tokens=max_output_tokens)

    raw = _run_with_retry(_do, "심판")
    # 실패를 전부 {} 로 뭉개면 호출부가 "빈 응답/파싱 실패/타입 불일치"를 못 가린다(예전엔
    # 이 침묵이 Gemini 리스트 래핑을 "빈 응답"으로 오인해 쓰레기 Probe 폴백을 유발했다).
    # 사유별로 로그만 남기고 반환은 {} 로 유지한다 — 호출부 넷이 전부 {} 를 "결측"으로
    # 흡수하는 구조라, 예외로 바꾸면 그 넷을 다 손대야 한다.
    if not (raw or "").strip():
        print("[Eval] chat_json 빈 응답 → {}")
        return {}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[Eval] chat_json JSON 파싱 실패(앞 80자: {raw[:80]!r}) → {{}}")
        return {}
    if isinstance(obj, dict):
        return obj
    # Gemini 는 json_mode 에서 스키마 강제 없이 mime_type 만 지정돼(core/llm_clients.py
    # gemini_chat) dict 를 [ {…} ] 로 한 겹 감싸 반환하는 경우가 있다. 길이 1 리스트에
    # dict 하나면 언랩한다. 길이 2+ 나 원소가 dict 가 아니면 스키마 위반이라 억지로 풀지
    # 않는다 — 잘못된 데이터를 정상인 척 통과시키지 않기 위해서다.
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        return obj[0]
    print(f"[Eval] chat_json 타입 불일치({type(obj).__name__}"
          f"{f', len={len(obj)}' if isinstance(obj, list) else ''}) → {{}}")
    return {}


# ── 임베딩 (metrics_ragas.py 가 사용) ─────────────────────────────
# GitHub Models 와 OpenRouter 는 embeddings 엔드포인트를 제공하지 않는다(OpenRouter 는
# 카탈로그 337개 중 임베딩 모델 0개 — 붙일 대상 자체가 없다). 그 조합에서는
# OPENAI_API_KEY 로 폴백하고, 그것도 없으면 Index 가 이미 쓰는 로컬 BGE-M3 로 계산한다.
# 셋 다 불가할 때만 결측이 된다.

# 임베딩 경로 전환/불가를 실행당 한 번만 알린다. 안 그러면 probe·트랙마다 같은 줄이 찍혀
# 정작 봐야 할 로그를 덮는다(agents/index/graph_index.py 의 _notify_llm_extraction_once 와 같은 패턴).
_embed_notified = False
_embed_notice_lock = threading.Lock()


def _notify_embed_once(message: str) -> None:
    global _embed_notified
    with _embed_notice_lock:
        if _embed_notified:
            return
        _embed_notified = True
    print(message)


def _api_embeddings_available() -> bool:
    """활성 provider 로 임베딩 API 를 부를 수 있는지. 키 유무만 본다(호출은 안 한다)."""
    if _provider() == "gemini":
        return bool(os.getenv("GEMINI_API_KEY"))
    return bool(os.getenv("OPENAI_API_KEY"))


def _local_embeddings_available() -> bool:
    """로컬 BGE-M3 로 임베딩할 수 있는지.

    모델 로드에 실패하면 qdrant_store 는 해시 기반 벡터를 돌려준다 — 검색에서는 '품질
    저하'지만 채점에 쓰면 무의미한 코사인 값이 정상 점수처럼 리포트에 박힌다. 결측보다
    나쁘므로 그 경우는 쓰지 않는다."""
    try:
        from agents.index.qdrant_store import embedding_is_fallback
        return not embedding_is_fallback()
    except Exception:
        return False


def embeddings_available() -> bool:
    """임베딩 의존 지표를 계산할 수 있는지(API 또는 로컬 모델)."""
    return _api_embeddings_available() or _local_embeddings_available()


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """텍스트 리스트 → 임베딩 벡터 리스트. (API 예외는 호출부로 전파; rate limit 은 재시도)

    우선순위: 활성 provider 의 임베딩 API > 로컬 BGE-M3 > 결측(빈 리스트).
    API 키가 있으면 기존 동작 그대로다 — 로컬은 임베딩 API 가 없는 조합
    (OpenRouter·GitHub Models 등)에서만 쓰인다.

    로컬 폴백이 필요한 이유: response_relevancy 가 결측이면 diagnose 의
    bad_gold_answer / bad_gold_answer_oracle 이 rel 을 AND 조건으로 요구해 영구히
    침묵하고, 그 라벨에 걸린 probe 자동 재생성 루프까지 멈춘다. 지표 하나가 아니라
    진단 기능이 빠지는 문제라 무료로 쓸 수 있는 로컬 모델로 메운다."""
    if not texts:
        return []

    if _api_embeddings_available():
        def _do():
            if _provider() == "gemini":
                return _gemini_embed(texts, model or os.getenv("EVAL_EMBED_MODEL_GEMINI", "gemini-embedding-001"))
            return _openai_embed(texts, model or os.getenv("EVAL_EMBED_MODEL", "text-embedding-3-small"))

        return _run_with_retry(_do, "임베딩")

    if _local_embeddings_available():
        # AGENTS.md 규약대로 공통 모듈을 통해서만 임베딩한다(직접 모델 로드 금지).
        from agents.index.qdrant_store import embed_batch
        _notify_embed_once(
            f"[Eval] EVAL_LLM_PROVIDER={_provider()} 는 임베딩 엔드포인트가 없어 "
            f"로컬 임베딩(BGE-M3)으로 계산합니다 — 비용 0, 외부 호출 없음. "
            f"참고: 임베딩 모델이 바뀌면 코사인 분포가 달라지므로 "
            f"response_relevancy 값을 API 임베딩 실행과 직접 비교하지 마세요.")
        return embed_batch(texts)

    _notify_embed_once(
        f"[Eval] EVAL_LLM_PROVIDER={_provider()} 는 임베딩 엔드포인트가 없고 "
        f"OPENAI_API_KEY 도, 로컬 임베딩 모델도 쓸 수 없어 임베딩 의존 지표"
        f"(response_relevancy)를 건너뜁니다 — bad_gold_answer 라벨도 함께 침묵합니다.")
    return []


# ── provider 별 transport (core/llm_clients.py 공용 구현에 위임) ──
# 모델명 규약: GitHub Models·OpenRouter 는 "<publisher>/<model>" 형식(예: openai/gpt-4o-mini).
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


def _openrouter_generate(
    system: str, user: str, model: str, json_mode: bool = False,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> str:
    return openai_chat(
        system, user, model, json_mode=json_mode, max_output_tokens=max_output_tokens,
        api_key=os.getenv("OPENROUTER_API_KEY"), base_url=OPENROUTER_BASE_URL, tag="Eval",
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
