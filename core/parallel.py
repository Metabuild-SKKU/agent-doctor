"""
core/parallel.py
LLM 호출(네트워크 I/O 대기) 병렬 실행 헬퍼.
agents/eval/probe_gen.py(STEP1 합성)와 agents/eval/agent.py(STEP2 답변 생성)가 쓴다.

스레드 기반(ThreadPoolExecutor)인 이유: 병목이 API 응답 대기(I/O)라 GIL 영향이
없고, 프로세스 분리 없이 기존 코드(전역 상태·클라이언트)를 그대로 쓸 수 있다.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(fn: Callable[[T], R], items: Iterable[T], max_workers: int) -> list[R]:
    """items 각각에 fn 을 적용해 입력 순서 그대로 결과 리스트를 반환한다.

    - max_workers <= 1 이면 executor 없이 순수 순차 실행 — 기존(병렬화 이전) 동작을
      바이트 단위로 보존한다(EVAL_LLM_CONCURRENCY=1 로 병렬화를 완전히 끌 수 있음).
    - 워커 예외는 삼키지 않고 결과 수집 시점에 그대로 전파한다 — 폴백이 필요한
      작업은 fn 안에서 자체 처리할 것(probe 합성의 휴리스틱 폴백 등).
    """
    items = list(items)
    if max_workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        return list(pool.map(fn, items))
