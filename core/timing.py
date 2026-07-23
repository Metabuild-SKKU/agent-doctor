"""에이전트 단계별 실행 시간을 일관된 로그 형식으로 측정한다."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


def _synchronize_cuda() -> None:
    """GPU 비동기 작업이 끝난 시점을 기준으로 시간을 재도록 동기화한다."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        # torch 미설치·CPU 환경에서도 시간 측정 자체는 계속한다.
        pass


class StageTimer:
    """한 에이전트의 단계 시간과 전체 시간을 표준 로그로 출력한다."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self._total_started = time.perf_counter()
        self._finished = False

    def begin(self, *, synchronize_cuda: bool = False) -> float:
        """측정 시작 시각을 반환한다."""
        if synchronize_cuda:
            _synchronize_cuda()
        return time.perf_counter()

    def end(
        self,
        stage: str,
        started: float,
        *,
        synchronize_cuda: bool = False,
    ) -> float:
        """단계 종료 시간을 계산하고 즉시 로그로 남긴다."""
        if synchronize_cuda:
            _synchronize_cuda()
        elapsed = time.perf_counter() - started
        print(f"[{self.tag}] 시간 | {stage}: {elapsed:.3f}초")
        return elapsed

    @contextmanager
    def measure(
        self,
        stage: str,
        *,
        synchronize_cuda: bool = False,
    ) -> Iterator[None]:
        """짧은 코드 블록의 실행 시간을 예외 경로까지 포함해 측정한다."""
        started = self.begin(synchronize_cuda=synchronize_cuda)
        try:
            yield
        finally:
            self.end(stage, started, synchronize_cuda=synchronize_cuda)

    def finish(self) -> float:
        """에이전트 전체 시간을 한 번만 출력한다."""
        if self._finished:
            return 0.0
        self._finished = True
        elapsed = time.perf_counter() - self._total_started
        print(f"[{self.tag}] 시간 | 전체: {elapsed:.3f}초")
        return elapsed
