"""
core/llm_retry.py
LLM rate limit(429) 재시도 공용 구현.
agents/eval/llm_provider.py 와 agents/rag/generator.py 가 공용으로 쓴다.

Gemini 무료 티어 등에서 "too many requests"(429)가 나면 잠시 대기 후 재시도한다.
    EVAL_LLM_RETRY_WAIT   기본 대기 시간(초, 기본 5). 실제 대기는 여기에 full jitter
                          (×1.0~2.0 무작위 배수)를 곱한다 — 병렬 워커들이 동시에 429를
                          맞고 같은 시각에 재돌진(thundering herd)하는 것을 막기 위함.
    EVAL_LLM_MAX_RETRIES  재시도 횟수 상한(기본 5). 소진하면 마지막 예외를 그대로 올린다.
rate limit 이 아닌 예외(인증 실패·잘못된 모델명 등)는 재시도 없이 즉시 전파한다.
"""
from __future__ import annotations

import os
import random
import time


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def is_rate_limit(exc: Exception) -> bool:
    """rate limit(429/quota/RESOURCE_EXHAUSTED) 계열 예외인지 status/메시지로 느슨히 판별.
    provider별 예외 타입을 직접 import 하지 않기 위함."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 429:
        return True
    msg = str(exc).lower()
    markers = ("429", "too many requests", "rate limit", "ratelimit",
               "resource_exhausted", "resourceexhausted", "quota")
    return any(m in msg for m in markers)


def run_with_retry(fn, label: str = "LLM", tag: str = "Eval"):
    """fn()을 호출하되 rate limit 예외면 대기 후 재시도.
    최대 EVAL_LLM_MAX_RETRIES회까지 재시도하고, 그래도 실패하면 마지막 예외를 전파한다.
    tag 는 재시도 로그의 접두([Eval]/[RAG]) 표시용."""
    max_retries = _env_int("EVAL_LLM_MAX_RETRIES", 5)
    base_wait = _env_float("EVAL_LLM_RETRY_WAIT", 5.0)
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:
            if not is_rate_limit(e) or attempt >= max_retries:
                raise
            attempt += 1
            wait = base_wait * (1.0 + random.random())  # full jitter
            print(f"[{tag}] {label} rate limit(429) — {wait:.0f}초 대기 후 재시도 ({attempt}/{max_retries})")
            time.sleep(wait)
