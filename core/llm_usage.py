"""
core/llm_usage.py
LLM 호출별 토큰 사용량·추정 비용 로깅.
agents/eval/llm_provider.py 와 agents/rag/generator.py 가 공용으로 쓴다.

호출마다 그 호출의 입력/출력 토큰 수·추정 비용과 프로세스 누적치를 출력한다.
비용은 API 가 알려주지 않으므로 토큰 × 단가표(_PRICES_USD_PER_1M)로 추정한다.
무료 티어 키면 실제 청구는 0원이고 이 금액은 유료 환산 참고치다.
    LLM_LOG_USAGE=0  으로 끄기 (기본 켜짐; 기존 EVAL_LOG_USAGE 도 인식)
"""
from __future__ import annotations

import os
import threading

# USD per 1M tokens (input, output). 유료 티어 텍스트 기준, 2026-07 요금표.
# 출처: ai.google.dev/gemini-api/docs/pricing, openai.com/api/pricing
# 미등록 모델은 토큰만 찍힌다. "publisher/model" 형식(GitHub Models)은 무료로 취급.
_PRICES_USD_PER_1M = {
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.1-pro": (2.00, 12.00),
    "gemini-3-flash": (0.50, 3.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-flash-latest": (0.30, 2.50),
    "gemini-embedding-001": (0.15, 0.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "text-embedding-3-small": (0.02, 0.0),
}

_totals = {"calls": 0, "prompt": 0, "output": 0, "cost": 0.0}
# 병렬 LLM 호출(core/parallel.py) 시 누적 갱신 경쟁·로그 줄 섞임을 막는 락.
_lock = threading.Lock()


def _enabled() -> bool:
    val = os.getenv("LLM_LOG_USAGE") or os.getenv("EVAL_LOG_USAGE") or "1"
    return val.strip().lower() not in {"0", "false", "off"}


def _estimate_cost_usd(model: str, prompt_tokens: int, output_tokens: int) -> float | None:
    """토큰 수 → 추정 비용(USD). GitHub Models(publisher/model)는 0, 단가 미등록이면 None."""
    if "/" in model:
        return 0.0
    for key in sorted(_PRICES_USD_PER_1M, key=len, reverse=True):
        if model.startswith(key):
            in_rate, out_rate = _PRICES_USD_PER_1M[key]
            return (prompt_tokens * in_rate + output_tokens * out_rate) / 1_000_000
    return None


def _fmt_usd(cost: float) -> str:
    return f"${cost:.2f}" if cost >= 0.01 else f"${cost:.6f}"


def log_usage(model: str, prompt_tokens, output_tokens, tag: str = "LLM") -> None:
    """호출 1건의 사용량을 기록·출력한다. tag 는 호출 주체 표시용([Eval]/[RAG])."""
    if not _enabled():
        return
    p, o = int(prompt_tokens or 0), int(output_tokens or 0)
    cost = _estimate_cost_usd(model, p, o)
    with _lock:
        _totals["calls"] += 1
        _totals["prompt"] += p
        _totals["output"] += o
        cost_part = ""
        if cost is not None:
            _totals["cost"] += cost
            cost_part = f" ≈ {_fmt_usd(cost)}"
        print(f"[{tag}] 토큰({model}): 입력 {p} / 출력 {o}{cost_part}"
              f" — 누적 {_totals['calls']}회, 입력 {_totals['prompt']:,} / 출력 {_totals['output']:,}"
              f" ≈ {_fmt_usd(_totals['cost'])}")


def print_summary(tag: str = "LLM") -> None:
    """프로세스 누적 사용량·추정 비용을 한 줄로 출력한다. LLM 호출이 없었으면 침묵."""
    if not _enabled():
        return
    with _lock:
        if not _totals["calls"]:
            return
        print(f"[{tag}] LLM 사용 합계: {_totals['calls']}회 호출, "
              f"입력 {_totals['prompt']:,} / 출력 {_totals['output']:,} 토큰 ≈ {_fmt_usd(_totals['cost'])}")
