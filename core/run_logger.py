"""
core/run_logger.py
파이프라인 실행 출력(print)을 콘솔과 로그 파일에 동시에 남기기 위한 Tee.

기존 print(...) 호출을 하나도 바꾸지 않고, sys.stdout/stderr 를 "콘솔 + 파일"에
동시에 쓰는 Tee 로 교체한다. 엔트리포인트(run_local_pipeline, graph.run_pipeline)
에서 setup_run_logging() 을 한 번만 부르면 그 뒤 모든 출력이 로그 파일에도 쌓인다.
"""
from __future__ import annotations

import atexit
import os
import sys
from datetime import datetime


class _Tee:
    """write/flush 를 여러 스트림에 그대로 위임하는 최소 Tee (콘솔+파일 동시 출력용).

    Windows 콘솔은 기본 인코딩이 cp949 등 UTF-8이 아닌 경우가 많아, 이모지나
    em-dash 같은 문자를 담은 print() 가 UnicodeEncodeError 를 던질 수 있다.
    그 예외가 agent.py 의 try/except 에 잡혀 정상 완료된 단계를 error 로
    오염시키는 걸 막기 위해, 인코딩 실패한 스트림에는 대체 문자로 바꿔 쓴다.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except UnicodeEncodeError:
                encoding = getattr(s, "encoding", None) or "ascii"
                s.write(data.encode(encoding, errors="replace").decode(encoding))
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        # tqdm/컬러 판별 등이 콘솔 여부를 물어볼 때: 하나라도 tty 면 tty 로 취급.
        return any(getattr(s, "isatty", lambda: False)() for s in self._streams)


_installed = False


def setup_run_logging(log_dir: str = "output/logs", prefix: str = "run") -> str | None:
    """stdout/stderr 를 콘솔+파일 Tee 로 교체하고 생성된 로그 파일 경로를 반환한다.

    한 프로세스에서 한 번만 설치한다(중복 호출은 무시하고 None 반환) — graph 와
    run_local_pipeline 가 겹쳐 불려도 이중 Tee 가 쌓이지 않는다.
    파일 열기에 실패하면(권한 등) 조용히 콘솔만 유지한다 — 로깅은 부가 기능이라
    파이프라인 실행을 막지 않는다.
    """
    global _installed
    if _installed:
        return None
    try:
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"{prefix}_{ts}.log")
        f = open(path, "a", encoding="utf-8", buffering=1)  # 줄 버퍼 → 실시간 tail 가능
    except OSError as e:
        print(f"[log] 파일 로깅 비활성(로그 파일 열기 실패: {e})")
        return None

    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_stdout, f)
    sys.stderr = _Tee(orig_stderr, f)

    def _restore_and_close():
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        f.close()

    atexit.register(_restore_and_close)
    _installed = True
    print(f"[log] 실행 로그 저장: {path}")
    return path
