"""
core/parallel.py
LLM 호출(네트워크 I/O 대기) 병렬 실행 헬퍼.
agents/eval/probe_gen.py(STEP1 합성)와 agents/eval/agent.py(STEP2 답변 생성)가 쓴다.

스레드 기반(ThreadPoolExecutor)인 이유: 병목이 API 응답 대기(I/O)라 GIL 영향이
없고, 프로세스 분리 없이 기존 코드(전역 상태·클라이언트)를 그대로 쓸 수 있다.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar

from core import progress

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    max_workers: int,
    label: str | None = None,
) -> list[R]:
    """items 각각에 fn 을 적용해 입력 순서 그대로 결과 리스트를 반환한다.

    - max_workers <= 1 이면 executor 없이 순수 순차 실행 — 기존(병렬화 이전) 동작을
      바이트 단위로 보존한다(EVAL_LLM_CONCURRENCY=1 로 병렬화를 완전히 끌 수 있음).
    - 워커 예외는 삼키지 않고 결과 수집 시점에 그대로 전파한다 — 폴백이 필요한
      작업은 fn 안에서 자체 처리할 것(probe 합성의 휴리스틱 폴백 등).
    - label 을 주면 진행률을 주기적으로 한 줄씩 찍는다(core/progress.py). 안 주면
      지금까지처럼 조용히 돈다. 라벨은 호출부만 아는 정보라 여기서 만들지 않는다.

    구현 메모: 완료 개수를 세려면 결과가 나오는 대로 받아야 해서 pool.map 대신
    submit + as_completed 를 쓴다. pool.map 과 달리 결과가 완료 순으로 오므로
    제출할 때 받아 둔 인덱스 자리에 채워 넣어 **입력 순서 계약을 지킨다**
    (RAGAS context_precision 의 순위 가중 평균처럼 순서에 의미가 있는 소비처가 있다).
    두 방식 모두 항목 전부를 즉시 제출하고 예외를 그대로 올리므로 나머지 계약은
    그대로다 — 다만 여러 항목이 실패하면 '입력 순서상 첫 예외' 가 아니라 '가장 먼저
    끝난 예외' 가 올라온다.
    """
    items = list(items)
    reporter = progress.start(label, len(items))
    if max_workers <= 1 or len(items) <= 1:
        results: list[R] = []
        for item in items:
            results.append(fn(item))
            progress.tick(reporter)
        progress.finish(reporter)
        return results

    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        index_of = {pool.submit(fn, item): i for i, item in enumerate(items)}
        ordered: list[R] = [None] * len(items)  # type: ignore[list-item]
        for future in as_completed(index_of):
            # future.result() 의 예외는 잡지 않는다 — 여기서 삼키면 호출부가 실패를
            # 성공한 빈 결과로 오인한다(위 계약).
            ordered[index_of[future]] = future.result()
            progress.tick(reporter)
        progress.finish(reporter)
        return ordered
