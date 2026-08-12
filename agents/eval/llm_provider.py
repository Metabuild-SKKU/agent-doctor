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

    EVAL_LLM_PROVIDER=anthropic → Anthropic Messages API(유료, 심판 전용 권장)
심판 품질이 모든 최적화 판단의 타당성을 결정하므로 별도 축으로 둔다. OpenRouter 를 거치지
않고 직접 붙는 이유는 output_config.format(JSON 스키마 강제)·prompt caching 때문이다 —
그 둘이 fused RAGAS 의 결손 보수 경로와 입력 비용을 동시에 없앤다. 모델명은 접두사 없는
정식 ID(예: claude-sonnet-5)를 쓰고, 임베딩 엔드포인트는 제공하지 않는다.
"""
from __future__ import annotations

import json
import os
import threading

from core.llm_clients import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    GITHUB_MODELS_BASE_URL,
    OPENROUTER_BASE_URL,
    PROVIDER_ALIASES,
    normalize_provider,
    anthropic_chat,
    gemini_chat,
    gemini_embed,
    openai_chat,
    openai_embed,
)
from core.llm_retry import run_with_retry


# 새 provider 를 추가할 때는 transport(_*_generate)·has_key()·여기 셋을 함께 고쳐야 한다.
# 여기 빠뜨리면 그 provider 는 "미지원 값"으로 판정돼 openai 로 폴백하고 transport 는
# 도달 불가 코드가 된다 — 파일 내 위치가 달라 git 이 충돌로 잡아주지 않는 실수다.
_KNOWN_PROVIDERS = {"openai", "gemini", "github", "openrouter", "anthropic"}
# 같은 문자열을 EVAL_LLM_PROVIDER 와 RAG_LLM_PROVIDER 어느 쪽에 넣어도 동작하도록 맞춘
# 철자표. agents/rag/generator.py 의 _PROVIDER_ALIASES 와 항상 같은 값을 유지할 것
# (tests/test_provider_notices.py 가 두 표의 일치를 핀으로 잡는다).
_PROVIDER_ALIASES = PROVIDER_ALIASES
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
          f"(openai|gemini|github|openrouter|anthropic)")


# 임베딩축 전용 값 집합. 심판축(_KNOWN_PROVIDERS)을 재사용하면 양방향으로 틀린다 —
# 임베딩 엔드포인트가 없는 anthropic·github 를 받아 조용히 다른 모델로 새고(코사인 분포가
# 달라져 실행 간 비교가 깨진다), 정작 "임베딩 실행 위치" 축의 대표 값인 local 은 거부해
# 심판축(과금 경로)으로 떨어뜨린다. 형제 축 INDEX_EMBED_PROVIDER 와 값이 맞아야
# "같은 계약" 이라는 이 축의 설명이 사실이 된다.
_EMBED_PROVIDERS = {"openai", "gemini", "openrouter", "local"}


def _warn_unusable_embed_provider_once(raw: str) -> None:
    """쓸 수 없는 EVAL_EMBED_PROVIDER 값 경고를 값당 한 번만 출력한다.

    같은 set 을 공유하면 EVAL_LLM_PROVIDER 에 같은 오타를 낸 실행이 이 경고를 삼킨다.
    사유를 둘로 가른다 - 오타와 "심판으로는 되지만 임베딩은 안 되는 값"은 고치는 방법이
    다르다(전자는 철자, 후자는 축 선택)."""
    key = f"embed:{raw}"
    with _warned_providers_lock:
        if key in _warned_providers:
            return
        _warned_providers.add(key)
    why = ("는 임베딩 엔드포인트가 없습니다"
           if normalize_provider(raw) in _KNOWN_PROVIDERS else "는 지원하지 않는 값입니다")
    print(f"[Eval] EVAL_EMBED_PROVIDER '{raw}'{why} · 심판 provider 를 따릅니다 "
          f"({'|'.join(sorted(_EMBED_PROVIDERS))})")


def _provider() -> str:
    """활성 provider. 오타 등 미지원 값은 openai 로 떨어지므로 경고를 남긴다 —
    Gemini/OpenRouter 로 돌린다고 믿은 실행이 조용히 OpenAI 로 과금되는 걸 막기 위함."""
    raw = os.getenv("EVAL_LLM_PROVIDER", "openai").strip().lower()
    if not raw:  # 빈 값은 "기본값" 의사표시로 보고 경고하지 않는다.
        return "openai"
    raw = normalize_provider(raw)
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
    if provider == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
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
    label: str = "",
    json_schema: dict | None = None,
    cache_prefix: str = "",
) -> dict:
    """JSON 응답 강제 chat 호출 → dict. 실패 시 {} (API 예외는 호출부로 전파).

    빈 응답 / 파싱 실패 / 타입 불일치는 사유별 로그를 남기고 {} 를 돌려준다. Gemini 가
    dict 를 [ {…} ] 로 감싸 반환한 경우(길이 1 리스트)는 언랩해 dict 로 돌려준다.
    상한에 걸려 잘린 응답은 JSON 파싱에 실패해 {} 가 되므로, 구조가 큰 응답을 기대하는
    쪽은 max_output_tokens 를 올려 잡을 것.

    label 은 실패 로그에 찍히는 호출 이름(어느 지표가 죽었는지 구분용).

    json_schema 는 **anthropic provider 에서만** 쓰인다(output_config.format). 나머지
    provider 는 "아무 JSON" 강제(json_object)만 지원하므로 무시한다 — 스키마를 넘겨도
    동작이 나빠지지 않고, 호출부가 provider 를 알 필요도 없다.

    cache_prefix 는 "매 호출 동일해서 캐시할 수 있는 앞부분" 이다. system 과 구분하는
    이유는 **다른 provider 의 프롬프트를 바꾸지 않기 위해서**다 — anthropic 에서만
    system 블록으로 올려 cache_control 을 걸고, 나머지 provider 에서는 user 앞에 도로
    붙여 예전과 바이트가 같은 한 덩어리를 만든다. 그냥 system 에 실으면 지시문이
    user → system 으로 역할이 바뀌어, 캐싱과 무관한 provider 의 판정 분포까지 건드리게
    된다(같은 글자라도 역할이 다르면 모델이 다르게 받아들일 수 있다)."""
    def _do():
        provider = _provider()
        # anthropic 이 아니면 캐시 지점이라는 개념이 없다 → 원래대로 user 앞에 이어붙인다.
        sys_text = system
        user_text = user
        if cache_prefix:
            if provider == "anthropic":
                sys_text = f"{system}\n\n{cache_prefix}" if system else cache_prefix
            else:
                user_text = cache_prefix + user
        if provider == "gemini":
            return _gemini_generate(
                sys_text, user_text, model or os.getenv("EVAL_JUDGE_MODEL_GEMINI", "gemini-flash-latest"),
                json_mode=True, max_output_tokens=max_output_tokens)
        elif provider == "github":
            return _github_generate(
                sys_text, user_text, model or os.getenv("EVAL_JUDGE_MODEL_GITHUB", "openai/gpt-4o"),
                json_mode=True, max_output_tokens=max_output_tokens)
        elif provider == "openrouter":
            # 주의: response_format=json_object 지원은 OpenRouter 에서 모델마다 다르다.
            # 미지원 모델을 쓰면 파싱 실패 → 아래 {} 폴백으로 조용히 흘러가므로,
            # 심판 모델은 JSON 모드를 지원하는 것으로 고를 것.
            return _openrouter_generate(
                sys_text, user_text, model or os.getenv("EVAL_JUDGE_MODEL_OPENROUTER", "openai/gpt-4o"),
                json_mode=True, max_output_tokens=max_output_tokens)
        elif provider == "anthropic":
            return _anthropic_generate(
                sys_text, user_text, model or os.getenv("EVAL_JUDGE_MODEL_ANTHROPIC", "claude-sonnet-5"),
                json_schema=json_schema, max_output_tokens=max_output_tokens)
        return _openai_generate(
            sys_text, user_text, model or os.getenv("EVAL_JUDGE_MODEL", "gpt-4o"),
            json_mode=True, max_output_tokens=max_output_tokens)

    raw = _run_with_retry(_do, "심판")
    # 실패를 전부 {} 로 뭉개면 호출부가 "빈 응답/파싱 실패/타입 불일치"를 못 가린다(예전엔
    # 이 침묵이 Gemini 리스트 래핑을 "빈 응답"으로 오인해 쓰레기 Probe 폴백을 유발했다).
    # 사유별로 로그만 남기고 반환은 {} 로 유지한다 — 호출부 넷이 전부 {} 를 "결측"으로
    # 흡수하는 구조라, 예외로 바꾸면 그 넷을 다 손대야 한다.
    where = f" [{label}]" if label else ""
    if not (raw or "").strip():
        print(f"[Eval] chat_json{where} 빈 응답 → {{}}")
        return {}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # 응답 길이와 그 호출에 적용된 상한을 함께 남긴다 — 상한 절단이면 끝이 잘린 채로
        # 끝나므로 "모델이 JSON 을 못 만든 것"과 "만들다 잘린 것"을 로그만으로 가를 수 있다.
        # 상한값을 같이 찍는 이유: 길이는 '문자 수'고 상한은 '토큰 수'라 길이만으로는
        # 상한에 붙었는지 판정할 수 없다(한국어는 문자당 토큰 비율이 커서 더 그렇다).
        print(f"[Eval] chat_json{where} JSON 파싱 실패({len(raw)}자, "
              f"상한 {max_output_tokens}토큰, "
              f"앞 80자: {raw[:80]!r}, 끝 40자: {raw[-40:]!r}) → {{}}")
        return {}
    if isinstance(obj, dict):
        return obj
    # Gemini 는 json_mode 에서 스키마 강제 없이 mime_type 만 지정돼(core/llm_clients.py
    # gemini_chat) dict 를 [ {…} ] 로 한 겹 감싸 반환하는 경우가 있다. 길이 1 리스트에
    # dict 하나면 언랩한다. 길이 2+ 나 원소가 dict 가 아니면 스키마 위반이라 억지로 풀지
    # 않는다 — 잘못된 데이터를 정상인 척 통과시키지 않기 위해서다.
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        return obj[0]
    print(f"[Eval] chat_json{where} 타입 불일치({type(obj).__name__}"
          f"{f', len={len(obj)}' if isinstance(obj, list) else ''}) → {{}}")
    return {}


# ── 임베딩 (metrics_ragas.py 가 사용) ─────────────────────────────
# OpenRouter 는 임베딩 엔드포인트(/api/v1/embeddings)를 제공한다. 예전 주석은 "카탈로그에
# 임베딩 모델 0개" 라고 적혀 있었는데 사실이 아니었다 — baai/bge-m3 를 포함해 31개가 있고,
# bge-m3 는 $0.01/1M 이라 사실상 공짜다. Index 가 쓰는 로컬 모델과 같은 모델이고
# 벡터도 코사인 0.99997 로 사실상 동일하다(실측 — AGENTS.md "임베딩 provider" 절 참고).
#
# GitHub Models·Anthropic 은 임베딩 엔드포인트가 없다. 그 조합에서는 OPENAI_API_KEY 로
# 폴백하고, 그것도 없으면 Index 가 이미 쓰는 로컬 BGE-M3 로 계산한다.
# 셋 다 불가할 때만 결측이 된다.
#
# 임베딩 경로는 기본적으로 **심판 provider 를 따라간다**. OPENROUTER_API_KEY 가 있다고 해서
# 심판이 anthropic/github 인 실행을 임의로 OpenRouter 임베딩으로 보내지는 않는다 — 그렇게
# 하면 심판 설정을 하나도 안 바꾼 사람의 실행이 OpenRouter 가용성에 새로 묶이고, 예전에
# 오프라인으로 돌던 조합이 남의 장애에 같이 죽는다(아래 embed_texts 독스트링이 설명하는
# bad_gold_answer 침묵 → probe 재생성 정지가 정확히 그 대가다).
#
# 심판축과 임베딩축을 나누고 싶으면 EVAL_EMBED_PROVIDER 로 **명시**한다(_embed_provider).
# Index 쪽의 INDEX_EMBED_PROVIDER 와 같은 모양이고, 기본값은 "심판을 따른다" 라 아무것도
# 설정하지 않은 실행의 동작은 바뀌지 않는다.

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


def _embed_provider() -> str:
    """임베딩을 실제로 부를 provider. 기본은 심판 provider 와 같다.

    anthropic·github 는 임베딩 엔드포인트가 없어, 그 심판을 쓰면 OPENROUTER_API_KEY 가
    있어도 2GB 로컬 모델을 띄우게 된다(실측: 로컬 16.8s vs OpenRouter 3.1s, 벡터는
    코사인 1.00000 으로 동일). 그 낭비를 피하려면 임베딩축만 따로 지정한다.

    **키가 있다고 자동 전환하지는 않는다.** 그러면 심판 설정을 안 바꾼 사람의 실행이
    OpenRouter 가용성에 새로 묶이고, 예전에 오프라인으로 돌던 조합이 남의 장애에 같이
    죽는다. 전환은 EVAL_EMBED_PROVIDER 를 적은 사람만 받는다 — Index 의
    INDEX_EMBED_PROVIDER 와 같은 계약이다.

    받는 값은 _EMBED_PROVIDERS — openai·gemini·openrouter·local. anthropic·github 는
    심판으로는 되지만 임베딩 엔드포인트가 없어 여기서는 거부한다(그대로 두면 아래 분기가
    openai 로 떨어져, 'anthropic 으로 임베딩한다'고 적어둔 실행이 실제로는 OpenAI 에
    과금 호출을 한다 - 심판축의 폴백 사슬과 달리 여기서는 사용자가 직접 고른 값이라
    조용히 바꾸면 안 된다).

    **미지원 값은 심판 provider 로 떨어뜨린다(경고 후).** 형제 축
    (qdrant_store.resolve_embedding_provider)은 같은 상황에서 local 로 떨어뜨린다 -
    "오타가 '조용히 API 과금' 이 아니라 '조용히 로컬' 로 끝나게" 라는 fail-safe 다.
    여기서 방향이 다른 이유는 두 축의 비용 구조가 달라서다: 색인은 코퍼스 전량이라
    오타 한 번이 통째로 과금되지만, 채점 임베딩은 probe 수십 건이라 금액이 작고 대신
    **경로가 바뀌면 코사인 분포가 달라져 실행 간 response_relevancy 비교가 깨진다**.
    그래서 여기서는 "설정이 무시됐다(= 미지정과 같은 동작)" 를 택했다. 이 선택이
    안전한 건 가장 흔한 오타 대상인 local 이 이제 정식 값이기 때문이다."""
    raw = os.getenv("EVAL_EMBED_PROVIDER", "").strip().lower()
    if not raw:
        return _provider()
    provider = normalize_provider(raw)
    if provider not in _EMBED_PROVIDERS:
        _warn_unusable_embed_provider_once(raw)
        return _provider()
    return provider


def _embed_axis_label() -> str:
    """폴백 안내에 쓸 "어느 설정 때문인가" 문구.

    축을 나눠 놓고 로그가 심판축 이름만 말하면 판정 주체와 문구가 어긋난다 - 사용자가
    EVAL_EMBED_PROVIDER 를 적었는데 "EVAL_LLM_PROVIDER=openrouter 는 임베딩 엔드포인트가
    없어" 라고 찍히면 엉뚱한 변수를 들여다보게 된다. 두 축이 같을 때만 심판축 이름을 쓴다."""
    embed = _embed_provider()
    if embed == _provider():
        return f"EVAL_LLM_PROVIDER={embed}"
    return f"EVAL_EMBED_PROVIDER={embed}"


def _api_embeddings_available() -> bool:
    """활성 임베딩 provider 로 임베딩 API 를 부를 수 있는지. 키 유무만 본다(호출은 안 한다)."""
    provider = _embed_provider()
    if provider == "local":
        # 사용자가 로컬을 못 박았다 - 키가 있어도 API 를 부르지 않는다.
        # embed_texts 가 아래 로컬 분기로 이어진다.
        return False
    if provider == "gemini":
        return bool(os.getenv("GEMINI_API_KEY"))
    if provider == "openrouter":
        return bool(os.getenv("OPENROUTER_API_KEY"))
    return bool(os.getenv("OPENAI_API_KEY"))


def _local_embeddings_available() -> bool:
    """로컬 BGE-M3 로 임베딩할 수 있는지.

    모델 로드에 실패하면 qdrant_store 는 해시 기반 벡터를 돌려준다 — 검색에서는 '품질
    저하'지만 채점에 쓰면 무의미한 코사인 값이 정상 점수처럼 리포트에 박힌다. 결측보다
    나쁘므로 그 경우는 쓰지 않는다.

    provider="local" 을 못 박는다. Index 의 기본 임베딩 provider 는 openrouter 라
    그냥 물으면 "해시 fallback 아님"(=API 경로엔 그 상태가 없음)이 돌아와, 정작
    로컬 모델이 못 뜨는 환경에서도 True 가 된다."""
    try:
        from agents.index.qdrant_store import embedding_is_fallback
        return not embedding_is_fallback(provider="local")
    except Exception:
        return False


def embeddings_available() -> bool:
    """임베딩 의존 지표를 계산할 수 있는지(API 또는 로컬 모델)."""
    return _api_embeddings_available() or _local_embeddings_available()


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """텍스트 리스트 → 임베딩 벡터 리스트. (rate limit 은 재시도)

    우선순위: 활성 provider 의 임베딩 API > 로컬 BGE-M3 > 결측(빈 리스트).
    로컬은 임베딩 API 가 없는 조합(GitHub Models·Anthropic 등)이나 키가 없을 때,
    그리고 **API 호출이 재시도 끝에 실패했을 때** 쓰인다. 로컬마저 쓸 수 없으면
    그때만 API 예외가 호출부로 전파된다.

    주의: provider 를 바꾸면 임베딩 모델이 바뀌고 코사인 분포도 달라진다.
    response_relevancy 를 실행 간에 비교하려면 한 번 정한 뒤 고정할 것.

    로컬 폴백이 필요한 이유: response_relevancy 가 결측이면 diagnose 의
    bad_gold_answer / bad_gold_answer_oracle 이 rel 을 AND 조건으로 요구해 영구히
    침묵하고, 그 라벨에 걸린 probe 자동 재생성 루프까지 멈춘다. 지표 하나가 아니라
    진단 기능이 빠지는 문제라 무료로 쓸 수 있는 로컬 모델로 메운다."""
    if not texts:
        return []

    if _api_embeddings_available():
        def _do():
            # _api_embeddings_available 과 같은 해석기를 써야 한다 — 다르면
            # "쓸 수 있다"고 판정한 provider 와 실제로 부르는 곳이 어긋난다.
            provider = _embed_provider()
            if provider == "gemini":
                return _gemini_embed(texts, model or os.getenv("EVAL_EMBED_MODEL_GEMINI", "gemini-embedding-001"))
            if provider == "openrouter":
                return _openrouter_embed(
                    texts,
                    model or os.getenv("EVAL_EMBED_MODEL_OPENROUTER", "baai/bge-m3"),
                )
            return _openai_embed(texts, model or os.getenv("EVAL_EMBED_MODEL", "text-embedding-3-small"))

        try:
            return _run_with_retry(_do, "임베딩")
        except Exception as exc:      # noqa: BLE001 — 폴백 판단만 하고 못 메우면 되던진다
            # 임베딩 API 가 죽었다고 진단 기능까지 같이 죽을 이유는 없다. 재시도를 다
            # 쓴 뒤에도 실패하면, 쓸 수 있는 로컬 모델이 있는 한 그쪽으로 계속한다 —
            # 아래 "로컬 폴백이 필요한 이유" 의 침묵 연쇄가 외부 장애로 열리는 것을 막는다.
            # 로컬도 못 쓰면 전파한다(무의미한 값으로 메우지 않는다 — _local_embeddings_available 참고).
            if not _local_embeddings_available():
                raise
            api_failure = exc
    else:
        api_failure = None

    if _local_embeddings_available():
        # AGENTS.md 규약대로 공통 모듈을 통해서만 임베딩한다(직접 모델 로드 금지).
        from agents.index.qdrant_store import embed_batch
        if api_failure is not None:
            head = f"[Eval] 임베딩 API 호출이 실패해({api_failure}) "
        elif _embed_provider() == "local":
            head = "[Eval] EVAL_EMBED_PROVIDER=local 이라 "
        else:
            head = f"[Eval] {_embed_axis_label()} 는 임베딩 엔드포인트가 없어 "
        _notify_embed_once(
            head +
            f"로컬 임베딩(BGE-M3)으로 계산합니다 · 비용 0, 외부 호출 없음. "
            f"참고: 임베딩 모델이 바뀌면 코사인 분포가 달라지므로 "
            f"response_relevancy 값을 API 임베딩 실행과 직접 비교하지 마세요.")
        # provider 를 못 박는다 — Index 의 기본값은 openrouter 라 그냥 부르면
        # "비용 0, 외부 호출 없음" 이라 찍어놓고 실제로는 과금 호출을 한다.
        return embed_batch(texts, provider="local")

    _notify_embed_once(
        f"[Eval] {_embed_axis_label()} 의 임베딩 키도, OPENAI_API_KEY 도, "
        f"로컬 임베딩 모델도 쓸 수 없어 임베딩 의존 지표(response_relevancy)를 "
        f"건너뜁니다 · bad_gold_answer 라벨도 함께 침묵합니다.")
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


def _openrouter_embed(texts: list[str], model: str) -> list[list[float]]:
    return openai_embed(
        texts, model,
        api_key=os.getenv("OPENROUTER_API_KEY"), base_url=OPENROUTER_BASE_URL, tag="Eval",
    )


def _anthropic_generate(
    system: str, user: str, model: str, json_schema: dict | None = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> str:
    """Anthropic transport. 다른 transport 와 달리 json_mode 대신 json_schema 를 받는다 —
    Messages API 에는 스키마 없는 JSON 모드가 없기 때문이다(core/llm_clients.py 참고).
    temperature 도 넘기지 않는다(Sonnet 5 이후 세대는 비기본 sampling 값을 400 으로 거부)."""
    return anthropic_chat(
        system, user, model, json_schema=json_schema,
        max_output_tokens=max_output_tokens,
        api_key=os.getenv("ANTHROPIC_API_KEY"), tag="Eval",
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
