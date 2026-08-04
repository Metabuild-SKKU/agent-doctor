"""
core/llm_clients.py
LLM provider transport 공용 구현 (OpenAI / GitHub Models / OpenRouter / Gemini).

agents/eval/llm_provider.py 와 agents/rag/generator.py 에 복붙돼 있던
"클라이언트 생성 → 호출 → usage 로깅" 계층만 모은다. provider 선택·폴백 체인,
env 규약(EVAL_LLM_PROVIDER vs RAG_LLM_PROVIDER), 키 부재 처리, 재시도 래핑은
호출하는 쪽 모듈이 그대로 가진다 — 여기는 "키가 이미 준비된 1회 호출"만 담당.

GitHub Models 와 OpenRouter 는 둘 다 OpenAI 호환이라 openai_chat 의 base_url/api_key
인자만으로 붙는다. 전용 transport 함수는 없다.
"""
from __future__ import annotations

import os
import threading

from core.llm_usage import log_usage

GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

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


def _openrouter_reported_cost(usage) -> float | None:
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
        return _openrouter_reported_cost(usage)
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
