"""
core/progress.py
긴 단계가 침묵하지 않도록 진행 상태를 한 줄씩 찍는 최소 리포터.

**이것은 완료 이벤트 기반이지 타이머(heartbeat) 기반이 아니다.**
    줄은 호출부가 tick() 을 부를 때, 즉 **항목이 하나 끝났을 때만** 나간다.
    PROGRESS_MIN_INTERVAL_SEC 은 "이 주기마다 찍는다"가 아니라 "줄 사이 최소 간격"
    이다 — 도배를 막는 상한이지 출력 보장이 아니다.

    그래서 다음은 **여전히 침묵한다**(알려진 한계):
      - 항목 하나가 오래 걸릴 때. RAGAS 1건이 3분 걸리면 그 3분은 조용하다.
      - 모든 워커가 동시에 같은 긴 API 호출에 걸려 있을 때.
    항목이 여러 개라 완료가 흩뿌려지는 구간(probe 18건, PDF 878페이지, 청크 수천
    건)이 이 모듈이 노리는 대상이고, 실제 침묵 구간도 대부분 거기였다.
    단건 long call 까지 덮으려면 별도 타이머 스레드가 필요하다 — 이 모듈 범위 밖이다.

왜 tqdm 이 아닌가:
    core/run_logger.py 의 _Tee 가 stdout 을 콘솔과 output/logs/*.log 에 **동시에**
    쓴다. `\\r` 로 같은 줄을 덮어쓰는 진행 바는 파일 쪽에 프레임 수백 개를 그대로
    박아 로그를 읽을 수 없게 만든다. 그래서 줄바꿈으로 끝나는 한 줄만 찍는다.
    tqdm/rich 기본 글자('█','▏','░')가 cp949 에서 깨지는 문제도 같이 피한다
    (AGENTS.md 코드 컨벤션 참고 — 여기 문구는 전부 cp949 안전).

왜 최소 간격을 건수가 아니라 시간으로 두는가:
    항목당 소요가 들쭉날쭉해서(RAGAS 는 건당 30초, PDF 페이지는 0.1초) "N건마다"
    로 두면 빠른 구간엔 도배되고 느린 구간엔 다시 침묵한다. 시간으로 두면 단계
    성격과 무관하게 출력 밀도의 상한이 일정해진다.

    부작용으로 **짧은 단계는 완전히 침묵한다** — 첫 줄이 나갈 만큼 오래 걸리지 않은
    단계는 완료 줄도 찍지 않아 기존 출력이 그대로 보존된다.

끝맺음은 두 갈래다:
    finish()  정상 완료 — '완료' 줄
    abort()   예외로 중단 — '중단' 줄. 이게 없으면 로그가 '12/30' 같은 중간 상태에서
              끊겨, 읽는 사람이 "멈춘 건지 죽은 건지" 를 구분할 수 없다.

끄는 법(core/llm_usage.py 의 _enabled() 와 같은 규약):
    PROGRESS_LOG=0|false|off       전부 끔 (기본은 켬)
    PROGRESS_MIN_INTERVAL_SEC=10   줄 사이 최소 간격(초)

이 모듈은 **출력만** 한다. 켜고 끄는 것이 계산 결과를 바꾸지 않는다.
다만 진행률을 붙이려고 호출부가 구조를 바꾼 곳까지 그렇다는 뜻은 아니다 —
agents/index/qdrant_store.py 는 창을 세려고 encode() 를 여러 번 부르는데, 그 분할은
PROGRESS_LOG 와 무관하게 항상 적용된다(자세한 내용은 그쪽 _encode_batch docstring).
"""
from __future__ import annotations

import os
import threading
import time

# 줄 사이 기본 최소 간격. 실측 기준으로 고른 값이다 — Eval STEP3 RAGAS(552초/18건)는
# 건당 30초라 이 간격에 걸리지 않고 완료마다 한 줄(18줄)이 나가고, Ingest PDF
# 추출(878페이지 2m36s)과 임베딩(3,880청크)은 이 간격에 눌려 10줄 안팎이 된다.
_DEFAULT_MIN_INTERVAL_SEC = 10.0

# 여러 스레드가 같은 리포터를 두드려도(parallel_map 워커) 줄이 섞이지 않게 한다.
_print_lock = threading.Lock()


def _enabled() -> bool:
    return (os.getenv("PROGRESS_LOG") or "1").strip().lower() not in {"0", "false", "off"}


def _min_interval_sec() -> float:
    """줄 사이 최소 간격(초). '이 주기마다 찍는다'가 아니라 '이보다 촘촘히는 안 찍는다'다.

    잘못된 값은 기본값으로 흘린다 — 진행률 설정 오타 때문에 파이프라인이 죽으면 안 된다."""
    try:
        value = float(os.getenv("PROGRESS_MIN_INTERVAL_SEC", "").strip()
                      or _DEFAULT_MIN_INTERVAL_SEC)
    except ValueError:
        return _DEFAULT_MIN_INTERVAL_SEC
    return value if value > 0 else _DEFAULT_MIN_INTERVAL_SEC


def _fmt_duration(seconds: float) -> str:
    """60초 미만은 '45s', 그 이상은 '9m12s'. 9분짜리 단계를 '552s' 로 읽게 두지 않는다.

    경계가 60초인 이유: 878페이지 실측에서 경과 시간이 10초마다 찍히는데, 경계를
    90초에 두면 '...70s · 80s · 1m31s' 처럼 한 칸이 건너뛴 것처럼 보인다."""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m{seconds % 60:02d}s"


class Progress:
    """완료 건수를 세다가 최소 간격이 지났으면 한 줄 찍는다. start() 로만 만든다.

    출력 시점은 전적으로 tick() 호출에 달려 있다 — 아무도 tick() 을 안 부르는 동안은
    시간이 아무리 지나도 조용하다(모듈 docstring 의 '완료 이벤트 기반' 참고)."""

    def __init__(self, label: str, total: int, min_interval_sec: float,
                 eta: bool = True):
        self._label = label
        self._total = total
        self._min_interval = min_interval_sec
        self._eta = eta
        self._done = 0
        self._started_at = time.monotonic()
        self._last_print_at = self._started_at
        self._printed = False
        self._closed = False

    def tick(self, count: int = 1) -> None:
        """항목 count 개가 끝났음을 알린다. 최소 간격이 안 지났으면 세기만 한다."""
        with _print_lock:
            self._done += count
            now = time.monotonic()
            # 마지막 한 건은 finish() 가 완료 줄로 찍으므로 여기서 중복 출력하지 않는다.
            if self._done >= self._total:
                return
            if now - self._last_print_at < self._min_interval:
                return
            self._last_print_at = now
            self._emit(now)

    def finish(self) -> None:
        """정상 완료. 진행 줄을 한 번이라도 찍었을 때만 완료 줄을 남긴다.

        조용히 끝난 짧은 단계에 완료 줄만 덜렁 붙이면 기존 출력에 없던 잡음이 되므로,
        '침묵했으면 끝까지 침묵' 을 지킨다."""
        self._close(final="완료")

    def abort(self, reason: str = "") -> None:
        """예외로 중단. 진행 줄을 찍은 적이 있으면 중단 줄로 닫는다.

        이게 없으면 로그가 '12/30 (40%)' 같은 중간 상태에서 그냥 끊긴다 — 읽는 사람이
        '아직 도는 중인지, 예외로 죽었는지' 를 구분할 수 없다. 관측성이 목적인
        모듈에서 그 구분이 안 되는 건 그 자체로 결함이다.

        finish() 와 마찬가지로 침묵했던 단계는 중단 줄도 안 찍는다 — 예외는 어차피
        전파돼 호출부가 제 방식으로 보고한다. 여기서 굳이 한 줄을 더 얹지 않는다."""
        self._close(final="중단", note=reason)

    # ── 내부 ────────────────────────────────────────────────
    def _close(self, final: str, note: str = "") -> None:
        with _print_lock:
            # finally 와 except 가 겹쳐 두 번 닫히는 배선에서도 줄은 한 번만 나간다.
            if not self._printed or self._closed:
                return
            self._closed = True
            self._emit(time.monotonic(), final=final, note=note)

    def _emit(self, now: float, final: str = "", note: str = "") -> None:
        elapsed = now - self._started_at
        percent = int(self._done * 100 / self._total) if self._total else 100
        line = f"{self._label} {self._done}/{self._total} ({percent}%)"
        if final:
            line += f" · {final} {_fmt_duration(elapsed)}"
            if note:
                line += f" ({note})"
        else:
            line += f" · 경과 {_fmt_duration(elapsed)}"
            # 남은 시간은 지금까지의 평균 속도로 민 추정치다. 병렬 구간에서는 초반
            # 완료가 몰려 낙관적으로 나왔다가 수렴하므로 '약' 을 붙여 둔다.
            # eta=False 는 이 외삽 자체가 성립하지 않는 구간용 — anthropic 배치처럼
            # 전 항목이 한꺼번에 끝나는 곳에서는 첫 완료 기준 외삽이 실측 대비 60배까지
            # 부풀었다(1/100 시점 '남은 약 285m' → 실제 4.7분).
            if self._eta and self._done > 0 and self._done < self._total:
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


def start(label: str | None, total: int, *, eta: bool = True) -> Progress | None:
    """진행률 리포터를 만든다. 끈 상태·라벨 없음·항목 없음이면 None(= 호출부는 아무것도 안 함).

    라벨은 호출부가 준다 — 'STEP3 RAGAS 실제 트랙' 처럼 어느 단계인지 알아야
    진행률이 의미를 갖는데, 그건 parallel_map 이나 encode 가 알 수 없는 정보다.
    라벨을 안 주면 지금까지처럼 조용히 돈다(기본이 침묵이라 새 소음이 생기지 않는다).

    eta=False 는 '남은 약 …' 추정을 끈다 — 완료가 한꺼번에 몰리는 구간(배치)에서는
    평균 속도 외삽이 수십 배로 틀리므로, 잘못된 숫자보다 없는 숫자가 낫다."""
    if not label or total <= 0 or not _enabled():
        return None
    return Progress(label, total, _min_interval_sec(), eta=eta)


def tick(reporter: Progress | None, count: int = 1) -> None:
    """None 안전 tick — 호출부에서 `if reporter:` 를 반복하지 않게 한다."""
    if reporter is not None:
        reporter.tick(count)


def finish(reporter: Progress | None) -> None:
    """None 안전 finish (정상 완료)."""
    if reporter is not None:
        reporter.finish()


def abort(reporter: Progress | None, reason: str = "") -> None:
    """None 안전 abort (예외 중단)."""
    if reporter is not None:
        reporter.abort(reason)
