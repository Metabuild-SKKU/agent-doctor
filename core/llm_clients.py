"""
core/llm_clients.py
LLM provider transport 공용 구현 (OpenAI / GitHub Models / OpenRouter / Gemini / Anthropic).

agents/eval/llm_provider.py 와 agents/rag/generator.py 에 복붙돼 있던
"클라이언트 생성 → 호출 → usage 로깅" 계층만 모은다. provider 선택·폴백 체인,
env 규약(EVAL_LLM_PROVIDER vs RAG_LLM_PROVIDER), 키 부재 처리, 재시도 래핑은
호출하는 쪽 모듈이 그대로 가진다 — 여기는 "키가 이미 준비된 1회 호출"만 담당.

GitHub Models 와 OpenRouter 는 둘 다 OpenAI 호환이라 openai_chat 의 base_url/api_key
인자만으로 붙는다. 전용 transport 함수는 없다.

Anthropic 만 전용 transport(anthropic_chat)를 둔다. OpenRouter 로 anthropic/claude-*
를 부르면 코드 변경이 0 이지만, Messages API 의 output_config.format(json_schema)·
prompt caching·thinking 제어에 닿을 수 없다. RAGAS fused 판정은 그 셋이 전부 필요하다
(스키마 강제로 결손 보수 경로가 불필요해지고, 지시문 캐싱으로 입력비가 0.1 배가 된다).
"""
from __future__ import annotations

import os
import threading

from core.llm_usage import log_usage

GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# 리랭크는 OpenAI 호환 규약이 아니라 Cohere 스타일 전용 엔드포인트다(OpenAI SDK 에 대응
# 메서드가 없어 openai_chat/openai_embed 처럼 base_url 만 갈아끼우는 방식이 안 통한다).
OPENROUTER_RERANK_URL = f"{OPENROUTER_BASE_URL}/rerank"

# provider 철자표 — 같은 값을 EVAL_LLM_PROVIDER / RAG_LLM_PROVIDER / INDEX_LLM_PROVIDER
# 어디에 넣어도 같게 해석되도록 한 곳에서 소유한다. 예전엔 Eval·RAG 가 각자 복사본을
# 두고 테스트로 대칭만 고정했는데, 나중에 붙은 Index 가 세 번째 표 없이 원문 비교만 해서
# 같은 값이 두 곳은 OpenRouter, Index 만 keyword 로 갈렸다.
PROVIDER_ALIASES = {
    "github_models": "github",
    "open_router": "openrouter",
    "open-router": "openrouter",
    "openrouter_ai": "openrouter",
    "openrouter.ai": "openrouter",
}


def normalize_provider(raw: str | None) -> str:
    """provider 문자열을 정규 표기로. 공백·대소문자·별칭을 흡수한다(미지원 값은 그대로)."""
    value = str(raw or "").strip().lower()
    return PROVIDER_ALIASES.get(value, value)

# 출력 토큰 기본 상한. 상한이 없으면 모델이 같은 문장을 반복 생성하며 최대치(64K)까지
# 달려도 아무도 막지 않는다 — 실제로 한 번 일어났고(응답 65,521 토큰, 그 1회로 $0.10),
# 잘린 응답이 JSON 파싱에 실패해 호출부가 조용히 휴리스틱으로 폴백하면서 쓰레기 Probe 를
# 만들어냈다. 상한은 비용 방어이자 "잘림"을 조기에 드러내는 장치다.
DEFAULT_MAX_OUTPUT_TOKENS = 2048

# 두 가지를 구분한다 — 예전엔 한 목록이 둘을 겸해서, 한쪽만 해당하는 모델을 넣으면
# 나머지 하나가 잘못 적용됐다.
#   (1) API 규약이 다른 계열: max_tokens 를 400 으로 거부하고(max_completion_tokens 만
#       허용), temperature 도 1 외의 값을 받지 않는다. OpenAI o-series/gpt-5 가 여기.
#   (2) 규약은 같지만 추론 토큰을 뱉어 출력 예산만 크게 필요한 계열: DeepSeek 등.
#       여기에 (1)을 적용하면 temperature 가 버려져 Optimize 의 generation.temperature
#       스윕이 조용히 no-op 이 된다 — 실제로 지원하는 모델인데도.
# 미확인: GitHub Models 엔드포인트가 max_completion_tokens 를 받는지는 검증하지 못했다.
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")
# gpt-5-chat-* 은 접두사만 같고 추론 모델이 아니다 — temperature 를 그대로 받는다.
_NON_REASONING_PREFIXES = ("gpt-5-chat",)
# (2) 계열. 목록은 필연적으로 밀리므로 아래 finish_reason=length 재시도가 최종 안전망이다.
_LARGE_OUTPUT_PREFIXES = ("deepseek",)

# 추론 토큰을 뱉는 모델의 출력 상한 하한선. 내부 추론 토큰이 이 상한을 함께 소진하고
# output 으로 과금되므로, 2048 이면 추론에서 다 쓰고 본문이 빈 채로 돌아올 수 있다
# (비용은 그대로). OpenAI 문서 권장 예약량 25K 를 하한으로 둔다 — 상한일 뿐 실사용분만
# 과금된다.
_REASONING_MIN_OUTPUT_TOKENS = 25_000

# (2) 계열도 같은 하한을 쓴다. 한때 이 계열만 8K 로 낮췄다가 되돌렸다 —
# 상한을 낮추면 그 상한을 넘는 호출이 "잘린 응답 값 + 재시도 값" 을 모두 지불하게 되어,
# 정작 상한이 막으려던 폭주 시나리오에서 더 비싸진다(8K+25K=33K > 25K). 상한은
# 실사용분만 과금되므로 정상 호출의 비용은 값과 무관하고, 낮춰서 얻는 이득이 없다.
# 이 값으로 시작하는 호출은 retry_cap > cap 이 거짓이라 재시도 경로가 열리지 않는다.
# json_mode 가 아닌 OpenRouter+추론off 호출은 이 승급을 건너뛰므로(위 openai_chat 참고)
# 그쪽은 재시도가 열려 있다 — 산문은 잘려도 부분 답변이 쓸모 있다는 판단이다.
#
# 실측(deepseek-v4-flash-0731, 한국어 RAGAS fused 판정 1회, top_k=5):
#   입력 2,514 / 출력 2,708 토큰 — 추론 토큰 포함. 기본 2048 로는 잘렸다.
# 정상 사용량은 상한의 11% 수준이라, 25K 는 "예산" 이 아니라 폭주 차단선으로만 작동한다.
_LARGE_OUTPUT_MIN_TOKENS = _REASONING_MIN_OUTPUT_TOKENS

# 추론 모델에서 temperature 가 버려졌다는 사실을 모델당 한 번 알린다.
# Optimize 의 generation.temperature sweep 이 조용히 no-op 이 되는 걸 드러내기 위함.
_warned_ignored_temperature: set[str] = set()
_warn_lock = threading.Lock()

# 이미 출력한 1회성 경고 키. Eval 은 스레드로 병렬 호출하므로 _warn_lock 으로 보호한다.
_warned_once: set[str] = set()


def _warn_once(key: str, message: str) -> bool:
    """같은 key 의 경고를 프로세스당 한 번만 출력한다. 출력했으면 True.

    probe·트랙마다 같은 줄이 찍히면 정작 봐야 할 진단 로그를 덮는다
    (agents/eval/llm_provider.py 의 _notify_embed_once 와 같은 패턴)."""
    with _warn_lock:
        if key in _warned_once:
            return False
        _warned_once.add(key)
    print(message)
    return True


# ── Anthropic 모델 능력표 ─────────────────────────────────────────
# .env 에서 모델명만 갈아끼워도 동작하게 하려면 모델별로 갈리는 축을 코드가 알아야 한다.
# 이 표가 없으면 갈림이 전부 "조용한 저하"로 나타난다 — haiku 로 내리면 캐시가 안 걸리고
# (최소 4096 토큰 미달), sonnet-4-6 으로 내리면 스키마 강제가 사라져 결손 보수 경로가
# 되살아나는데, 어느 쪽도 에러를 내지 않는다.
#
# 필드: (입력 $/1M, 출력 $/1M, 캐시 최소 토큰, output_config.format 지원, thinking 끌 수 있음)
#
# thinking 축이 특히 중요하다 — 이 transport 는 판정용이라 thinking 을 끄는 것이 기본인데:
#   fable/mythos : {"type":"disabled"} 가 400. thinking 을 끌 수 없다(항상 켜짐).
#   opus-5       : effort<=high 에서만 disabled 허용. effort 를 안 보내면 기본 high 라 통과.
#   그 외        : disabled 허용.
# 생략하면 안 된다는 점도 같이 봐야 한다 — sonnet-5/opus-5 는 thinking 을 생략하면
# adaptive 가 켜지고, max_tokens 가 사고+본문을 함께 제한해 본문이 잘린다.
_ANTHROPIC_MODELS: dict[str, tuple[float, float, int, bool, bool]] = {
    # (in, out, cache_min, schema, can_disable_thinking)
    "claude-fable-5":    (10.00, 50.00,  512, True,  False),
    "claude-mythos-5":   (10.00, 50.00,  512, True,  False),
    "claude-opus-5":     ( 5.00, 25.00,  512, True,  True),
    "claude-opus-4-8":   ( 5.00, 25.00, 1024, True,  True),
    "claude-opus-4-7":   ( 5.00, 25.00, 2048, False, True),
    "claude-opus-4-6":   ( 5.00, 25.00, 4096, False, True),
    "claude-opus-4-5":   ( 5.00, 25.00, 4096, True,  True),
    # sonnet-5 정가. 2026-08-31 까지는 인트로 $2/$10 이지만 정가로 잡는다 —
    # 추정이 과소되는 것보다 과대되는 쪽이 안전하고, 인트로가 끝나도 표를 안 고쳐도 된다.
    "claude-sonnet-5":   ( 3.00, 15.00, 1024, True,  True),
    "claude-sonnet-4-6": ( 3.00, 15.00, 1024, False, True),
    "claude-sonnet-4-5": ( 3.00, 15.00, 1024, False, True),
    "claude-haiku-4-5":  ( 1.00,  5.00, 4096, True,  True),
}
# 표에 없는 모델의 기본값. 신모델은 대개 현세대 규약을 따르므로 그쪽에 맞추고,
# 단가만 None 으로 둬서 "단가 미등록" 으로 드러나게 한다(0 으로 뭉개지 않는다).
_ANTHROPIC_UNKNOWN = (None, None, 1024, True, True)

# 캐시 토큰 과금 배수(5분 TTL 기준). 쓰기는 입력가의 1.25 배, 읽기는 0.1 배다.
# 이걸 무시하고 전부 입력가로 계산하면 캐시가 잘 걸릴수록 비용이 과대집계된다.
_ANTHROPIC_CACHE_WRITE_MULT = 1.25
_ANTHROPIC_CACHE_READ_MULT = 0.10


def _anthropic_caps(model: str) -> tuple:
    """모델 능력표 조회. 접두 매칭(긴 키 우선) — 표에 없으면 현세대 기본값."""
    name = _bare_model_name(model)
    for key in sorted(_ANTHROPIC_MODELS, key=len, reverse=True):
        if name.startswith(key):
            return _ANTHROPIC_MODELS[key]
    return _ANTHROPIC_UNKNOWN


# ── JSON Schema 조립 헬퍼 (output_config.format 용) ────────────────
# Anthropic structured outputs 는 모든 object 에 additionalProperties: false 와
# required 전량 명시를 요구한다. 하나만 빠져도 요청이 400 이고, 스키마를 쓰는 곳은
# 여럿이라(agents/eval/metrics_ragas.py, agents/eval/probe_gen.py) 규칙의 소유자를
# 한 곳에 둔다 — 손으로 쓰면 additionalProperties 를 빠뜨리기 쉽다.
#
# 미지원 키워드도 함께 기억해 둘 것: minimum/maximum/minLength/maxLength/multipleOf
# 같은 수치·길이 제약과 재귀 스키마는 쓸 수 없다. 값의 범위를 좁혀야 하면 enum 을 쓴다.

def strict_object(props: dict) -> dict:
    """모든 키가 required 이고 추가 키를 막는 object 스키마."""
    return {"type": "object", "properties": props,
            "required": list(props), "additionalProperties": False}


def array_of(items: dict) -> dict:
    """항목 스키마가 items 인 배열."""
    return {"type": "array", "items": items}


SCHEMA_STR = {"type": "string"}
SCHEMA_INT = {"type": "integer"}
# 0/1 이진 판정. minimum/maximum 을 못 쓰므로 enum 으로 좁힌다.
SCHEMA_INT01 = {"type": "integer", "enum": [0, 1]}


def _openrouter_reasoning_disabled() -> bool:
    """OpenRouter 호출에서 추론 토큰을 끌지. 기본 끔(=True).

    추론 토큰은 output 으로 과금되는데 RAGAS 판정에서 비용의 대부분을 차지한다.
    끄면 싸지지만 판정 품질이 떨어질 수 있다 — 근거 뒷받침 여부를 따지는 작업이라
    추론이 도움이 되는 쪽이다. 점수가 거칠어지면 OPENROUTER_REASONING=1 로 되살린다.
    """
    return (os.getenv("OPENROUTER_REASONING", "0").strip().lower()
            not in {"1", "true", "yes", "on"})


def _bare_model_name(model: str) -> str:
    """'publisher/model' 형식에서 모델명만. GitHub Models·OpenRouter 가 이 형식을 쓴다."""
    return model.rsplit("/", 1)[-1].strip().lower()


def _is_reasoning_model(model: str) -> bool:
    """max_completion_tokens 만 받고 temperature 를 거부하는 계열인지(OpenAI o-series/gpt-5)."""
    name = _bare_model_name(model)
    if name.startswith(_NON_REASONING_PREFIXES):
        return False
    return name.startswith(_REASONING_MODEL_PREFIXES)


def _needs_large_output(model: str) -> bool:
    """API 규약은 일반 모델과 같지만 추론 토큰 때문에 출력 예산이 크게 필요한 계열인지."""
    return _bare_model_name(model).startswith(_LARGE_OUTPUT_PREFIXES)


def _warn_temperature_ignored_once(model: str, temperature: float, tag: str) -> None:
    with _warn_lock:
        if model in _warned_ignored_temperature:
            return
        _warned_ignored_temperature.add(model)
    print(f"[{tag}] {model} 은 temperature 를 받지 않아 요청값({temperature})이 무시됩니다 "
          f"— Optimize 의 generation.temperature 조정도 이 모델에선 no-op 입니다.")


def openrouter_reported_cost(usage) -> float | None:
    """OpenRouter 응답 usage 에서 실제 과금액(USD)을 꺼낸다. 없으면 None.

    OpenRouter 는 요청에 usage.include 를 주면 응답 usage 에 cost 를 실어 보낸다.
    단가표 추정과 달리 모델을 바꿔도 표를 고칠 필요가 없고 값이 틀릴 일도 없다.
    OpenAI SDK 의 CompletionUsage 는 스키마에 없는 필드를 model_extra 로 흘리므로
    속성 접근과 model_extra 를 모두 시도한다."""
    cost = getattr(usage, "cost", None)
    if cost is None:
        extra = getattr(usage, "model_extra", None) or {}
        cost = extra.get("cost")
    try:
        return float(cost) if cost is not None else None
    except (TypeError, ValueError):
        return None


def _known_cost_usd(base_url: str | None, usage) -> float | None:
    """transport 가 아는 실제 비용(USD). 모르면 None → 호출부가 단가표로 추정한다.

    "publisher/model" 형식을 무료로 단정하던 예전 규칙을 대체한다 — GitHub Models(무료)와
    OpenRouter(유료)가 같은 형식이라, 그 규칙 아래서는 OpenRouter 유료 호출이 전부
    $0 으로 조용히 기록됐다. 엔드포인트를 아는 여기서 판정하는 게 안전하다."""
    if base_url == GITHUB_MODELS_BASE_URL:
        return 0.0   # GitHub Models 무료 티어
    if base_url == OPENROUTER_BASE_URL:
        # 못 읽으면 None → "단가 미등록"으로 드러난다. $0 으로 뭉개지 않는다.
        return openrouter_reported_cost(usage)
    return None


def openai_chat(
    system: str,
    user: str,
    model: str,
    *,
    json_mode: bool = False,
    api_key: str | None = None,
    base_url: str | None = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    temperature: float = 0.0,
    tag: str = "LLM",
) -> str:
    """OpenAI 호환 chat 1회 호출 → 응답 텍스트("" 가능).

    temperature 기본 0(결정적). RAG 답변 생성만 호출부에서 조정하고, 판정·합성 등
    구조가 중요한 호출은 기본 0을 유지한다. 추론 모델(o-series/gpt-5)은 temperature 를
    받지 않아 무시되고(1회 경고), 출력 상한은 추론 토큰 몫까지 25K 로 올려 잡는다.
    base_url/api_key 를 주면 GitHub Models·OpenRouter 등 OpenAI 호환 엔드포인트 겸용."""
    from openai import OpenAI

    client_kwargs = {}
    if base_url:
        client_kwargs["base_url"] = base_url
    if api_key:
        client_kwargs["api_key"] = api_key
    client = OpenAI(**client_kwargs)
    base_kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    if base_url == OPENROUTER_BASE_URL:
        # 응답 usage 에 실제 과금액(cost)을 함께 받는다 — 추가 비용·지연 없음.
        extra: dict = {"usage": {"include": True}}
        if _openrouter_reasoning_disabled():
            # 추론 토큰은 output 으로 과금된다. 실측(deepseek, RAGAS 판정 1회)에서 출력
            # 2,708 토큰 중 대부분이 추론이었고 JSON 본문은 수백 토큰이었다.
            # 주의: include_reasoning=false 는 "안 보여줄 뿐" 이라 여전히 과금된다.
            # 실제로 끄려면 reasoning.enabled=false 여야 한다.
            extra["reasoning"] = {"enabled": False}
        base_kwargs["extra_body"] = extra

    reasoning = _is_reasoning_model(model)
    if reasoning and temperature != 1.0:
        _warn_temperature_ignored_once(model, temperature, tag)

    # 추론을 꺼둔 호출에는 큰 출력 예산 승급을 건너뛴다 — 그 승급은 추론 토큰이 상한을
    # 먹어 본문이 잘리는 걸 막으려던 것이고, 추론이 없으면 근거가 사라진다. 남겨두면
    # 호출부가 정한 상한(RAG 답변 4096 등)이 조용히 25K 로 덮여 폭주 여지만 커진다.
    #
    # 단 json_mode 는 예외다. 잘린 JSON 은 파싱이 실패해 전량 손실이고, 아래 재시도가
    # 반드시 걸려 "잘린 값 + 재시도 값" 을 둘 다 지불한다. 산문은 잘려도 부분 답변이
    # 쓸모 있어 좁은 상한이 이득이지만, JSON 은 애초에 넉넉히 주는 쪽이 싸다.
    # (실측: 이 예외 없이 fused RAGAS 를 돌리면 4096 에서 잘려 결손 13건·재시도 4건,
    #  재시도 1건당 4,096+25,000=29,096 토큰. 승급하면 잘림 0건.)
    reasoning_off = (base_url == OPENROUTER_BASE_URL
                     and _openrouter_reasoning_disabled())
    cap = max_output_tokens
    if reasoning:
        cap = max(cap, _REASONING_MIN_OUTPUT_TOKENS)
    elif _needs_large_output(model) and (json_mode or not reasoning_off):
        cap = max(cap, _LARGE_OUTPUT_MIN_TOKENS)

    def _call(limit: int):
        kwargs = dict(base_kwargs)
        if reasoning:
            kwargs["max_completion_tokens"] = limit
        else:
            kwargs["max_tokens"] = limit
            kwargs["temperature"] = temperature
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        if resp.usage:
            log_usage(
                model, resp.usage.prompt_tokens, resp.usage.completion_tokens, tag=tag,
                cost_usd=_known_cost_usd(base_url, resp.usage),
            )
        choice = resp.choices[0]
        return choice, (choice.message.content or "")

    choice, content = _call(cap)
    truncated = getattr(choice, "finish_reason", None) == "length"

    # 상한에 걸려 쓸 수 없는 응답이면 상한을 올려 딱 1회만 다시 부른다.
    # 접두사 목록(_LARGE_OUTPUT_PREFIXES)은 새 모델이 나올 때마다 밀리므로, 목록이 못
    # 잡은 경우의 최종 안전망이다. "쓸 수 없다"는 본문이 비었거나 JSON 을 기대한 경우로
    # 한정한다 — 산문이 길어서 잘린 건 부분 답변이라도 쓸모가 있고, 무조건 재시도하면
    # 긴 답변마다 비용이 두 배가 된다. 재시도는 1회 고정이라 비용 상한이 예측 가능하다.
    if truncated and (json_mode or not content.strip()):
        # 재시도 목표치는 추론 몫까지 감안한 고정값. 배수(cap×N)로 늘리면 상한이 없어져,
        # 같은 문장을 반복 생성하는 모델에 훨씬 큰 예산을 태우게 된다(파일 상단 주석의
        # 65,521 토큰 사고가 그 사례). 이미 이 값 이상이면 다시 불러도 의미가 없다.
        retry_cap = _REASONING_MIN_OUTPUT_TOKENS
        if retry_cap > cap:
            print(f"[{tag}] {model} 응답이 출력 상한({cap})에서 잘렸습니다 "
                  f"— 상한 {retry_cap} 으로 1회 재시도합니다.")
            choice, content = _call(retry_cap)
            truncated = getattr(choice, "finish_reason", None) == "length"
            cap = retry_cap

    # 재시도 후에도 잘렸으면(또는 재시도 대상이 아니면) 사유를 드러낸다 — 여기서 안 찍으면
    # 호출부가 "빈 응답"으로 흡수해 원인 추적이 불가능하다.
    if truncated:
        print(f"[{tag}] {model} 응답이 출력 상한({cap})에서 잘렸습니다(finish_reason=length).")
    return content


def gemini_chat(
    system: str,
    user: str,
    model: str,
    *,
    json_mode: bool = False,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    temperature: float = 0.0,
    tag: str = "LLM",
) -> str:
    """Gemini chat 1회 호출(google-genai SDK) → 응답 텍스트("" 가능).

    temperature 기본 0(결정적). RAG 답변 생성만 호출부에서 조정한다.
    주의: 추론 모델은 내부 사고(thoughts)도 이 상한을 함께 소진한다. JSON 구조가 온전해야
    하는 호출(Probe 합성 등)은 호출부에서 상한을 넉넉히 주는 게 안전하다."""
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config: dict = {
        "temperature": temperature,
        "system_instruction": system,
        "max_output_tokens": max_output_tokens,
    }
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


def _anthropic_text(content) -> str:
    """응답 content 블록에서 text 만 이어붙인다. text 블록이 없으면 ""."""
    return "".join(getattr(b, "text", "") or ""
                   for b in (content or []) if getattr(b, "type", None) == "text")


def _anthropic_cost_usd(caps: tuple, usage) -> float | None:
    """Anthropic usage → 실제 과금 추정(USD). 단가 미등록이면 None.

    캐시 토큰을 입력가 그대로 세면 안 된다 — 쓰기는 1.25 배, 읽기는 0.1 배다.
    RAGAS 판정처럼 지시문이 매번 같은 호출은 대부분이 캐시 읽기라, 이걸 무시하면
    실제보다 몇 배 비싸게 집계돼 "캐싱이 효과가 없다" 는 잘못된 결론이 나온다."""
    in_rate, out_rate = caps[0], caps[1]
    if in_rate is None or out_rate is None:
        return None
    fresh = getattr(usage, "input_tokens", 0) or 0
    write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    read = getattr(usage, "cache_read_input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    return (
        fresh * in_rate
        + write * in_rate * _ANTHROPIC_CACHE_WRITE_MULT
        + read * in_rate * _ANTHROPIC_CACHE_READ_MULT
        + out * out_rate
    ) / 1_000_000


def anthropic_chat(
    system: str,
    user: str,
    model: str,
    *,
    json_schema: dict | None = None,
    api_key: str | None = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    cache_system: bool = True,
    tag: str = "LLM",
) -> str:
    """Anthropic Messages API 1회 호출 → 응답 텍스트("" 가능).

    openai_chat 과 달리 **temperature 인자가 없다.** Sonnet 5 이후 세대는 기본값이 아닌
    sampling 파라미터(temperature/top_p/top_k)를 400 으로 거부한다. 인자를 두지 않으면
    호출부가 실수로 넘길 수도 없다 — 이 transport 는 판정·합성 전용이라 결정성을 프롬프트로
    잡고, 온도를 흔들어야 하는 RAG 답변 생성은 애초에 이 함수를 쓰지 않는다.

    json_schema 를 주면 output_config.format 으로 **스키마를 강제**한다. OpenAI 의
    json_object 처럼 "아무 JSON" 을 강제하는 모드는 Messages API 에 없고, 개방형 스키마도
    불가하다(additionalProperties: false 가 강제). 그래서 스키마를 안 주면 강제 없이
    프롬프트에만 의존한다 — 그 경우 OpenRouter 로 부르는 것과 차이가 없다.

    system 은 캐시 지점이다. 매 호출 동일한 지시문·스키마·few-shot 을 system 에 두고
    질문·컨텍스트는 user 에 두면, 두 번째 호출부터 그 몫이 입력가의 0.1 배가 된다.
    (모델별 최소 길이 미달이면 조용히 캐시가 안 걸리므로 첫 호출에서 확인해 경고한다.)

    주의 — 캐시 엔트리는 **첫 응답이 시작된 뒤에야** 읽을 수 있다. Eval 은 probe 를
    parallel_map 으로 동시 채점하므로, 실행 시작 시 같이 출발한 EVAL_LLM_CONCURRENCY 개
    호출은 전부 캐시 미스이고 각자 쓰기(1.25배)를 지불한다. "판정 1건 50% 절감" 은 캐시가
    더워진 정상 상태의 수치라 실행 전체로는 그만큼 나오지 않는다.

    실측 기반 추정(probe 30, 실제 30 + 오라클 15 호출, EVAL_LLM_CONCURRENCY=10):
        캐시 없음        $0.998
        현재(첫 물결 미스) $0.778   절감 22.1%
        전부 히트         $0.518   절감 48.1%
    동시성이 클수록 첫 물결이 커져 더 희석된다. 트랙별 첫 1건으로 캐시를 데운 뒤 나머지를
    팬아웃하면 이상치에 가까워지지만, 그건 호출부(agents/eval/agent.py)의 실행 순서
    문제라 여기서 손대지 않는다."""
    import anthropic

    caps = _anthropic_caps(model)
    _, _, cache_min, supports_schema, can_disable_thinking = caps

    kwargs: dict = {}

    # ── thinking: 판정에는 불필요하고, max_tokens 를 본문과 나눠 쓰므로 끄는 게 안전하다.
    cap = max_output_tokens
    if can_disable_thinking:
        kwargs["thinking"] = {"type": "disabled"}
    else:
        # fable/mythos 계열은 thinking 을 끌 수 없다(파라미터를 보내면 400). 생략하면
        # 항상 켜진 채로 돌고 사고 토큰이 max_tokens 를 함께 소진하므로 상한을 올려 잡는다.
        cap = max(cap, _REASONING_MIN_OUTPUT_TOKENS)
        _warn_once(
            f"anthropic-thinking-{model}",
            f"[{tag}] {model} 은 thinking 을 끌 수 없어 사고 토큰이 함께 과금됩니다 "
            f"— 출력 상한을 {cap} 으로 올려 잡습니다(판정 비용이 크게 늘 수 있습니다).")

    # ── 스키마 강제: 지원 모델에서만. 미지원 모델에서 조용히 빠지면 이 transport 를
    #    만든 이유(결손 보수 경로 제거)가 사라지므로 반드시 알린다.
    if json_schema:
        if supports_schema:
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": json_schema}}
        else:
            _warn_once(
                f"anthropic-schema-{model}",
                f"[{tag}] {model} 은 output_config.format(스키마 강제)을 지원하지 않습니다 "
                f"— JSON 은 프롬프트 지시로만 유도되고, 결손 시 보수 경로가 다시 돕니다.")

    # ── 캐시 지점: system 블록. 본문이 비면 아예 보내지 않는다.
    if system and system.strip():
        block: dict = {"type": "text", "text": system}
        if cache_system:
            block["cache_control"] = {"type": "ephemeral"}
        kwargs["system"] = [block]

    client = anthropic.Anthropic(**({"api_key": api_key} if api_key else {}))

    def _call(limit: int):
        resp = client.messages.create(
            model=model,
            max_tokens=limit,
            messages=[{"role": "user", "content": user}],
            **kwargs,
        )
        usage = getattr(resp, "usage", None)
        if usage:
            fresh = getattr(usage, "input_tokens", 0) or 0
            write = getattr(usage, "cache_creation_input_tokens", 0) or 0
            read = getattr(usage, "cache_read_input_tokens", 0) or 0
            # 입력 토큰은 세 갈래의 합으로 넘긴다 — fresh 만 넘기면 캐시가 잘 걸릴수록
            # 토큰 열이 작아져 "입력이 줄었다" 는 거짓 인상을 준다(비용만 준 것이다).
            log_usage(model, fresh + write + read,
                      getattr(usage, "output_tokens", 0) or 0,
                      tag=tag, cost_usd=_anthropic_cost_usd(caps, usage))
            if cache_system and system and system.strip() and not (write or read):
                _warn_once(
                    f"anthropic-cache-{model}",
                    f"[{tag}] {model} 에서 프롬프트 캐시가 걸리지 않았습니다 "
                    f"(캐시 최소 {cache_min}토큰, 이번 입력 {fresh}토큰) "
                    f"— 지시문이 최소 길이에 못 미치면 에러 없이 무시됩니다.")
        return resp, _anthropic_text(getattr(resp, "content", None))

    resp, content = _call(cap)
    stop = getattr(resp, "stop_reason", None)

    # 안전 분류기 거부는 HTTP 200 + stop_reason="refusal" 로 온다. content 가 비어
    # 있으므로 여기서 안 찍으면 호출부가 "빈 응답" 으로 흡수해 원인 추적이 불가능하다.
    if stop == "refusal":
        details = getattr(resp, "stop_details", None)
        print(f"[{tag}] {model} 이 요청을 거부했습니다"
              f"(refusal, category={getattr(details, 'category', None)}).")
        return content

    # openai_chat 의 finish_reason=="length" 대응물. 잘린 JSON 은 파싱이 통째로 실패하므로
    # 상한을 올려 딱 1회만 다시 부른다(재시도 1회 고정 = 비용 상한이 예측 가능).
    if stop == "max_tokens" and (json_schema or not content.strip()):
        retry_cap = _REASONING_MIN_OUTPUT_TOKENS
        if retry_cap > cap:
            print(f"[{tag}] {model} 응답이 출력 상한({cap})에서 잘렸습니다 "
                  f"— 상한 {retry_cap} 으로 1회 재시도합니다.")
            resp, content = _call(retry_cap)
            stop, cap = getattr(resp, "stop_reason", None), retry_cap

    if stop == "max_tokens":
        print(f"[{tag}] {model} 응답이 출력 상한({cap})에서 잘렸습니다(stop_reason=max_tokens).")
    return content


def openai_embed(
    texts: list[str],
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    tag: str = "LLM",
) -> list[list[float]]:
    """OpenAI 호환 embeddings 1회 호출 → 벡터 리스트(입력 순서 유지).

    base_url/api_key 를 주면 OpenRouter 등 OpenAI 호환 엔드포인트 겸용 —
    openai_chat 과 같은 규약이다. OpenRouter 는 임베딩 엔드포인트에서도
    usage.include 로 실제 과금액을 실어 보낸다(실측 확인).

    재시도는 하지 않는다. 호출부가 core.llm_retry.run_with_retry 로 감쌀 것 —
    OpenRouter 임베딩은 동시성과 무관하게 429 가 상시 섞여 나오고(실측:
    동시 1 에서도 요청의 19%), 재시도 없이 삼키면 그 청크의 벡터가 조용히
    빠진 컬렉션이 만들어진다."""
    from openai import OpenAI

    client_kwargs = {}
    if base_url:
        client_kwargs["base_url"] = base_url
    if api_key:
        client_kwargs["api_key"] = api_key

    create_kwargs: dict = {"model": model, "input": texts}
    if base_url == OPENROUTER_BASE_URL:
        create_kwargs["extra_body"] = {"usage": {"include": True}}

    resp = OpenAI(**client_kwargs).embeddings.create(**create_kwargs)
    if resp.usage:
        # 임베딩은 출력 토큰이 없다. cost 를 안 넘기면 OpenRouter 유료 호출이
        # 전부 $0 으로 조용히 집계된다.
        log_usage(
            model, resp.usage.prompt_tokens, 0, tag=tag,
            cost_usd=_known_cost_usd(base_url, resp.usage),
        )
    # index 로 정렬한다. 스펙상 입력 순서대로 오지만 그건 응답 본문이 보장하는 게
    # 아니라 관례이고(그래서 index 필드가 따로 있다), 이제 색인 벡터가 이 경로를 탄다.
    # 순서가 한 칸만 어긋나도 벡터가 엉뚱한 청크에 붙어 검색이 조용히 망가진다.
    return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


def openrouter_rerank(
    query: str,
    documents: list[str],
    model: str,
    *,
    api_key: str,
    timeout: float = 60.0,
    tag: str = "LLM",
) -> list[tuple[int, float]]:
    """OpenRouter /rerank 1회 호출 → [(입력 인덱스, 관련도 점수)].

    응답은 점수 내림차순이라 입력 순서가 아니다. 인덱스를 그대로 실어 돌려주고
    호출부가 제자리에 꽂는다 — 여기서 정렬해 버리면 "응답이 짧게 왔다"(문서 일부가
    빠졌다)를 호출부가 알아챌 방법이 없어진다.

    top_n 은 보내지 않는다. 보내면 그 개수만 응답에 오는데, 호출부(rerank_with_status)는
    후보 전원의 점수를 받아 정렬하는 규약이라 빠진 문서를 실패로 읽는다. 비용도 안 준다 —
    top_n 은 응답만 자를 뿐 채점(과금) 대상 문서는 그대로다(Cohere 계열은 search 1건
    단위, Voyage 계열은 보낸 토큰 단위 — 어느 쪽이든 top_n 과 무관하다).

    재시도는 하지 않는다 — openai_embed 와 같은 규약으로, 호출부가
    core.llm_retry.run_with_retry 로 감싼다. 상태 코드를 예외에 실어 올려야
    is_transient 가 429/5xx 만 골라 재시도할 수 있어 status_code 를 붙인다.
    """
    import requests

    resp = requests.post(
        OPENROUTER_RERANK_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": model, "query": query, "documents": documents},
        timeout=timeout,
    )
    if resp.status_code >= 400:
        error = RuntimeError(
            f"OpenRouter rerank 실패(status {resp.status_code}, model={model}): "
            f"{resp.text[:300]}"
        )
        # 문자열 매칭이 아니라 상태 코드로 재시도 여부가 갈리게 한다(core/llm_retry.py).
        error.status_code = resp.status_code
        raise error

    payload = resp.json()
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError(
            f"OpenRouter rerank 응답에 results 가 없습니다(model={model}): "
            f"{str(payload)[:300]}"
        )

    usage = payload.get("usage") or {}
    cost = usage.get("cost") if isinstance(usage, dict) else None
    try:
        cost_usd = float(cost) if cost is not None else None
    except (TypeError, ValueError):
        cost_usd = None
    # 과금 단위가 모델마다 다르다(Cohere 계열 search 1건 / Voyage 계열 토큰). 단가표로
    # 추정하지 않고 응답 usage 의 실비를 그대로 기록한다 — 응답이 토큰 수를 줄 때만 싣고,
    # cost 를 못 읽으면 None 으로 "단가 미등록"이 된다. $0 으로 뭉개져 조용히 사라지지 않는다.
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    log_usage(model, prompt_tokens or 0, 0, tag=tag, cost_usd=cost_usd)

    scored: list[tuple[int, float]] = []
    for item in results:
        if not isinstance(item, dict):
            raise RuntimeError(f"OpenRouter rerank 응답 항목 형식 오류: {str(item)[:120]}")
        score = item.get("relevance_score")
        if score is None:
            score = item.get("score")
        scored.append((int(item.get("index", -1)), float(score)))
    return scored


def gemini_embed(texts: list[str], model: str, *, tag: str = "LLM") -> list[list[float]]:
    """Gemini embed_content 1회 호출 → 벡터 리스트(입력 순서 유지)."""
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    resp = client.models.embed_content(model=model, contents=texts)
    return [e.values for e in resp.embeddings]
