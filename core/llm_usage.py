"""
core/llm_usage.py
LLM 단계별 호출 횟수·추정 비용 집계.
agents/eval/llm_provider.py 와 agents/rag/generator.py 가 공용으로 쓴다.

집계 단위는 "호출"이 아니라 "스텝"이다 — 호출마다 한 줄씩 찍으면 Eval 한 번에
수십 줄이 쌓여, 정작 봐야 할 probe 별 진단 결과가 묻힌다. 대신 step() 으로 구간을
열어 두면 그 구간이 끝날 때 호출 수·비용과 프로세스 누적치를 한 줄로 요약한다.
비용은 API 가 알려주지 않으므로 토큰 × 단가표(_PRICES_USD_PER_1M)로 추정한다.
무료 티어 키면 실제 청구는 0원이고 이 금액은 유료 환산 참고치다.
    LLM_LOG_USAGE=0        전부 끄기
    LLM_LOG_USAGE=1        스텝 요약만 (기본)
    (기존 EVAL_LOG_USAGE 도 같은 값으로 인식)
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager

# USD per 1M tokens (input, output). 유료 티어 텍스트 기준, 2026-07 요금표.
# 출처: ai.google.dev/gemini-api/docs/pricing, openai.com/api/pricing
# 미등록 모델은 비용을 계산하지 않고 호출 수를 별도로 표시한다.
# "publisher/model" 형식(GitHub Models)은 단가가 0인 무료 모델로 취급한다.
_PRICES_USD_PER_1M = {
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.1-pro": (2.00, 12.00),
    "gemini-3-flash": (0.50, 3.00),
    "gemini-2.5-flash": (0.30, 2.50),
    # alias — 2026-07 현재 gemini-3.5-flash를 가리킴. Google이 alias를 옮기면 같이 갱신할 것.
    "gemini-flash-latest": (1.50, 9.00),
    "gemini-embedding-001": (0.15, 0.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    # Index 그래프 entity 추출 기본 모델(core/state.py graph_llm_model)과 그 형제들.
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1": (2.00, 8.00),
    "text-embedding-3-small": (0.02, 0.0),
}

_totals = {
    "calls": 0,
    "prompt": 0,
    "output": 0,
    "cost": 0.0,
    "unpriced_calls": 0,
}
# 에이전트(Ingest/Index/Eval/Optimize)별 누적 — 파이프라인 종료 표(print_agent_table)용.
# 귀속 키는 열려 있는 스텝의 tag 를 우선한다(아래 log_usage 주석 참고).
_by_agent: dict[str, dict] = {}
# 병렬 LLM 호출(core/parallel.py) 시 누적 갱신 경쟁·로그 줄 섞임을 막는 락.
_lock = threading.Lock()

# 현재 열려 있는 스텝(step() 이 관리). ThreadPoolExecutor 워커에서도 같은 스텝에
# 귀속돼야 하므로 threading.local 이 아니라 모듈 전역 + _lock 으로 다룬다 —
# Eval STEP2/STEP3-2 의 LLM 호출은 core/parallel.py 의 워커 스레드에서 일어난다.
_current_step: dict | None = None
def _setting() -> str:
    return (os.getenv("LLM_LOG_USAGE") or os.getenv("EVAL_LOG_USAGE") or "1").strip().lower()


def _enabled() -> bool:
    return _setting() not in {"0", "false", "off"}


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
    """합계·표용. 1센트를 넘으면 유효자리를 잃지 않게 4자리까지 보여준다
    (스모크 실행은 통째로 1센트 미만이라 2자리면 전부 $0.04 로 뭉개진다)."""
    return f"${cost:.4f}" if cost >= 0.01 else f"${cost:.6f}"


def snapshot_usage() -> dict[str, int | float]:
    """단계 시작 시점의 프로세스 누적 사용량을 복사한다."""
    with _lock:
        return dict(_totals)


def log_usage(model: str, prompt_tokens, output_tokens, tag: str = "LLM") -> None:
    """호출 1건의 사용량을 기록한다. tag 는 호출 주체 표시용([Eval]/[RAG]).

    출력은 하지 않는다 — 열려 있는 스텝에 누적만 하고, 요약은 step() 이
    구간 끝에서 한 줄로 찍는다.
    """
    if not _enabled():
        return
    p, o = int(prompt_tokens or 0), int(output_tokens or 0)
    cost = _estimate_cost_usd(model, p, o)
    with _lock:
        _totals["calls"] += 1
        _totals["prompt"] += p
        _totals["output"] += o
        if cost is None:
            _totals["unpriced_calls"] += 1
        else:
            _totals["cost"] += cost

        # 에이전트 귀속: 열린 스텝의 tag 를 우선한다. Eval STEP2 의 답변 생성은
        # agents/rag/generator.py 를 거쳐 tag="RAG" 로 들어오는데, 그 호출은 Eval 이
        # 지불한 비용이므로 tag 만 믿으면 Eval 집계에서 통째로 빠진다.
        owner = _current_step["tag"] if _current_step else tag
        bucket = _by_agent.setdefault(
            owner,
            {
                "calls": 0,
                "prompt": 0,
                "output": 0,
                "cost": 0.0,
                "unpriced_calls": 0,
            },
        )
        bucket["calls"] += 1
        bucket["prompt"] += p
        bucket["output"] += o
        if cost is None:
            bucket["unpriced_calls"] += 1
        else:
            bucket["cost"] += cost

        if _current_step is not None:
            _current_step["calls"] += 1
            _current_step["prompt"] += p
            _current_step["output"] += o
            if cost is None:
                _current_step["unpriced_calls"] += 1
            else:
                _current_step["cost"] += cost

# ── 스텝 단위 집계 ────────────────────────────────────────────────

_HEADER_WIDTH = 62


def _display_width(text: str) -> int:
    """한글·박스 문자를 2칸으로 세어 폭을 맞춘다(콘솔 고정폭 기준).
    len() 으로 재면 한글 제목이 든 헤더만 짧게 그려져 구분선 끝이 들쭉날쭉해진다."""
    return sum(2 if ord(ch) > 0x2500 else 1 for ch in text)


def _header(tag: str, n: int, title: str) -> str:
    left = f"─── {tag} STEP{n} · {title} "
    return left + "─" * max(3, _HEADER_WIDTH - _display_width(left))


@contextmanager
def step(tag: str, n: int, title: str):
    """STEP 구간을 열고, 끝날 때 그 구간의 LLM 사용량을 한 줄로 요약한다.

    구간 안에서 일어난 모든 log_usage 는 (호출부의 tag 와 무관하게) 이 스텝에 귀속된다.
    예외가 나도 스텝은 반드시 닫는다 — 다음 스텝의 집계가 오염되지 않도록.
    스텝은 항상 순차로 열리고 닫히므로 전역 하나로 충분하다(중첩은 지원하지 않는다).
    """
    global _current_step
    if not _enabled():
        yield
        return

    print(_header(tag, n, title))
    with _lock:
        _current_step = {
            "tag": tag,
            "n": n,
            "title": title,
            "calls": 0,
            "prompt": 0,
            "output": 0,
            "cost": 0.0,
            "unpriced_calls": 0,
        }
    started_at = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - started_at
        with _lock:
            rec, _current_step = _current_step, None
            cumulative = dict(_totals)
        if rec is not None:
            line = _step_line(rec, cumulative)
            if line:
                print(line)
            print(f"[{tag}] STEP{n} 소요: {elapsed:.1f}s")


def _step_line(rec: dict, cumulative: dict) -> str:
    """LLM 호출이 있었던 단계만 요구된 사용량 필드로 한 줄 출력한다."""
    if not rec["calls"]:
        return ""
    return (
        f"[{rec['tag']}] LLM 사용 | STEP{rec['n']} {rec['title']}: "
        f"호출 {rec['calls']}회, 비용 ≈ {_fmt_usd(rec['cost'])}, "
        f"단가 미등록 {rec['unpriced_calls']}회 | "
        f"누적 호출 {cumulative['calls']}회, "
        f"누적 비용 ≈ {_fmt_usd(cumulative['cost'])}, "
        f"누적 단가 미등록 {cumulative['unpriced_calls']}회"
    )


def print_summary(
    tag: str = "LLM",
    *,
    stage: str = "전체",
    since: dict[str, int | float] | None = None,
) -> None:
    """한 단계의 호출 횟수·비용과 프로세스 누적치를 한 줄로 출력한다."""
    if not _enabled():
        return
    with _lock:
        baseline = since or {
            "calls": 0,
            "cost": 0.0,
            "unpriced_calls": 0,
        }
        stage_calls = int(_totals["calls"]) - int(baseline.get("calls", 0))
        stage_cost = float(_totals["cost"]) - float(baseline.get("cost", 0.0))
        stage_unpriced = int(_totals["unpriced_calls"]) - int(
            baseline.get("unpriced_calls", 0)
        )
        if stage_calls <= 0:
            return
        print(
            f"[{tag}] LLM 사용 | {stage}: "
            f"호출 {stage_calls}회, 비용 ≈ {_fmt_usd(stage_cost)}, "
            f"단가 미등록 {stage_unpriced}회 | "
            f"누적 호출 {_totals['calls']}회, "
            f"누적 비용 ≈ {_fmt_usd(_totals['cost'])}, "
            f"누적 단가 미등록 {_totals['unpriced_calls']}회"
        )


def totals_by_agent() -> dict[str, dict]:
    """에이전트별 누적 사용량 스냅샷(읽기 전용 복사본)."""
    with _lock:
        return {k: dict(v) for k, v in _by_agent.items()}


_HEADERS = ("에이전트", "LLM호출", "토큰(입/출)", "추정비용", "단가미등록")


def _rule(widths: list[int], left: str, mid: str, right: str) -> str:
    return left + mid.join("─" * (w + 2) for w in widths) + right


def _row(widths: list[int], cells: tuple[str, ...]) -> str:
    """각 칸을 폭에 맞춰 채운다. 첫 칸은 좌측, 나머지 숫자 칸은 우측 정렬."""
    out = []
    for i, (w, text) in enumerate(zip(widths, cells)):
        pad = max(0, w - _display_width(text))
        out.append(" " + (text + " " * pad if i == 0 else " " * pad + text) + " ")
    return "│" + "│".join(out) + "│"


def print_agent_table() -> None:
    """파이프라인 종료 시 에이전트별 LLM 사용량 표. 호출이 0인 에이전트 행은 생략."""
    if not _enabled():
        return
    agents = {k: v for k, v in totals_by_agent().items() if v["calls"]}
    if not agents:
        return

    rows = [
        (
            name,
            f"{v['calls']:,}",
            f"{v['prompt']:,}/{v['output']:,}",
            _fmt_usd(v["cost"]),
            f"{v['unpriced_calls']:,}",
        )
        for name, v in sorted(agents.items(), key=lambda kv: -kv[1]["cost"])
    ]
    total = ("합계", f"{_totals['calls']:,}",
             f"{_totals['prompt']:,}/{_totals['output']:,}", _fmt_usd(_totals["cost"]),
             f"{_totals['unpriced_calls']:,}")
    # 폭은 실제 내용에서 구한다 — 고정폭이면 토큰 수가 커질 때(수십만) 칸을 삐져나온다.
    widths = [max(_display_width(r[i]) for r in (_HEADERS, *rows, total))
              for i in range(len(_HEADERS))]

    print("\n  전체 LLM 사용")
    print("  " + _rule(widths, "┌", "┬", "┐"))
    print("  " + _row(widths, _HEADERS))
    print("  " + _rule(widths, "├", "┼", "┤"))
    for r in rows:
        print("  " + _row(widths, r))
    print("  " + _rule(widths, "├", "┼", "┤"))
    print("  " + _row(widths, total))
    print("  " + _rule(widths, "└", "┴", "┘"))
