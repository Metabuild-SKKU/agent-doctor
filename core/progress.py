"""
core/progress.py
긴 단계가 침묵하지 않도록 "주기적 한 줄"을 찍는 최소 진행률 리포터.

왜 tqdm 이 아닌가:
    core/run_logger.py 의 _Tee 가 stdout 을 콘솔과 output/logs/*.log 에 **동시에**
    쓴다. `\\r` 로 같은 줄을 덮어쓰는 진행 바는 파일 쪽에 프레임 수백 개를 그대로
    박아 로그를 읽을 수 없게 만든다. 그래서 줄바꿈으로 끝나는 한 줄만 찍는다.
    tqdm/rich 기본 글자('█','▏','░')가 cp949 에서 깨지는 문제도 같이 피한다
    (AGENTS.md 코드 컨벤션 참고 — 여기 문구는 전부 cp949 안전).

왜 건수가 아니라 시간 기준인가:
    항목당 소요가 들쭉날쭉해서(RAGAS 는 건당 30초, PDF 페이지는 0.1초) 건수 기준이면
    빠른 구간엔 도배되고 느린 구간엔 다시 침묵한다. 마지막 출력 후 PROGRESS_INTERVAL_SEC
    초가 지났을 때만 찍으면 단계 성격과 무관하게 출력 밀도가 일정해진다.

    부작용으로 **짧은 단계는 완전히 침묵한다** — 첫 줄이 나올 만큼 오래 걸리지 않은
    단계는 완료 줄도 찍지 않아 기존 출력이 그대로 보존된다.

끄는 법(core/llm_usage.py 의 _enabled() 와 같은 규약):
    PROGRESS_LOG=0|false|off   전부 끔 (기본은 켬)
    PROGRESS_INTERVAL_SEC=10   출력 주기(초)

이 모듈은 **출력만** 한다. 켜고 끄는 것이 계산 결과를 바꾸지 않는다.
"""
from __future__ import annotations

import os
import threading
import time

# 기본 주기. 실측 기준으로 고른 값이다 — Eval STEP3 RAGAS(552초/18건)는 건당 30초라
# 어차피 완료마다 한 줄(18줄)이고, Ingest PDF 추출(104초/878페이지)과 임베딩
# (67.8초/3,880청크)은 이 주기에 걸려 각각 10줄 안팎으로 눌린다.
_DEFAULT_INTERVAL_SEC = 10.0

# 여러 스레드가 같은 리포터를 두드려도(parallel_map 워커) 줄이 섞이지 않게 한다.
_print_lock = threading.Lock()


def _enabled() -> bool:
    return (os.getenv("PROGRESS_LOG") or "1").strip().lower() not in {"0", "false", "off"}


def _interval_sec() -> float:
    """출력 주기(초). 잘못된 값은 기본값으로 흘린다 — 진행률 설정 때문에 실행이 죽으면 안 된다."""
    try:
        value = float(os.getenv("PROGRESS_INTERVAL_SEC", "").strip() or _DEFAULT_INTERVAL_SEC)
    except ValueError:
        return _DEFAULT_INTERVAL_SEC
    return value if value > 0 else _DEFAULT_INTERVAL_SEC


def _fmt_duration(seconds: float) -> str:
    """60초 미만은 '45s', 그 이상은 '9m12s'. 9분짜리 단계를 '552s' 로 읽게 두지 않는다.

    경계가 60초인 이유: 878페이지 실측에서 경과 시간이 10초마다 찍히는데, 경계를
    90초에 두면 '...70s · 80s · 1m31s' 처럼 한 칸이 건너뛴 것처럼 보인다."""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m{seconds % 60:02d}s"


class Progress:
    """완료 건수를 세다가 주기가 지나면 한 줄 찍는다. start() 로만 만든다."""

    def __init__(self, label: str, total: int, interval_sec: float):
        self._label = label
        self._total = total
        self._interval = interval_sec
        self._done = 0
        self._started_at = time.monotonic()
        self._last_print_at = self._started_at
        self._printed = False

    def tick(self, count: int = 1) -> None:
        """항목 count 개가 끝났음을 알린다. 주기가 안 됐으면 세기만 하고 조용히 넘어간다."""
        with _print_lock:
            self._done += count
            now = time.monotonic()
            # 마지막 한 건은 finish() 가 완료 줄로 찍으므로 여기서 중복 출력하지 않는다.
            if self._done >= self._total:
                return
            if now - self._last_print_at < self._interval:
                return
            self._last_print_at = now
            self._emit(now)

    def finish(self) -> None:
        """단계 종료. 진행 줄을 한 번이라도 찍었을 때만 완료 줄을 남긴다.

        조용히 끝난 짧은 단계에 완료 줄만 덜렁 붙이면 기존 출력에 없던 잡음이 되므로,
        '침묵했으면 끝까지 침묵' 을 지킨다."""
        with _print_lock:
            if not self._printed:
                return
            self._emit(time.monotonic(), final=True)

    # ── 내부 ────────────────────────────────────────────────
    def _emit(self, now: float, final: bool = False) -> None:
        elapsed = now - self._started_at
        percent = int(self._done * 100 / self._total) if self._total else 100
        line = f"{self._label} {self._done}/{self._total} ({percent}%)"
        if final:
            line += f" · 완료 {_fmt_duration(elapsed)}"
        else:
            line += f" · 경과 {_fmt_duration(elapsed)}"
            # 남은 시간은 지금까지의 평균 속도로 민 추정치다. 병렬 구간에서는 초반
            # 완료가 몰려 낙관적으로 나왔다가 수렴하므로 '약' 을 붙여 둔다.
            if self._done > 0 and self._done < self._total:
                remaining = elapsed / self._done * (self._total - self._done)
                line += f" · 남은 약 {_fmt_duration(remaining)}"
        # flush: _Tee 는 줄 버퍼라 대개 바로 나가지만, 진행률은 "살아 있다" 를 보여주는
        # 것이 목적이라 버퍼에 걸려 늦게 나오면 의미가 없다.
        print(line, flush=True)
        self._printed = True

    # 테스트용 — 실제 출력 없이 상태를 확인한다.
    @property
    def printed(self) -> bool:
        return self._printed


def start(label: str | None, total: int) -> Progress | None:
    """진행률 리포터를 만든다. 끈 상태·라벨 없음·항목 없음이면 None(= 호출부는 아무것도 안 함).

    라벨은 호출부가 준다 — 'STEP3 RAGAS 실제 트랙' 처럼 어느 단계인지 알아야
    진행률이 의미를 갖는데, 그건 parallel_map 이나 encode 가 알 수 없는 정보다.
    라벨을 안 주면 지금까지처럼 조용히 돈다(기본이 침묵이라 새 소음이 생기지 않는다)."""
    if not label or total <= 0 or not _enabled():
        return None
    return Progress(label, total, _interval_sec())


def tick(reporter: Progress | None, count: int = 1) -> None:
    """None 안전 tick — 호출부에서 `if reporter:` 를 반복하지 않게 한다."""
    if reporter is not None:
        reporter.tick(count)


def finish(reporter: Progress | None) -> None:
    """None 안전 finish."""
    if reporter is not None:
        reporter.finish()
