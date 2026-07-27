"""
core/console.py
표준 출력/오류 스트림의 인코딩을 UTF-8 로 고정한다.

왜 필요한가:
    한글 Windows 콘솔의 기본 인코딩은 cp949 라 '—'(em dash) · '↳' · '⚠' 같은
    문자를 못 쓴다. 그런 문자가 든 print() 는 그 자리에서 UnicodeEncodeError 를
    던지는데, agents/eval/agent.py 의 포괄 except 가 이걸 "평가 실패" 로 잡아
    **정상 완료된 STEP 을 status="error" 로 뒤집는다**. 로그 한 줄 때문에
    파이프라인이 실패한 것처럼 보고되는 것이다.

    지금까지는 run_logger 의 Tee 가 인코딩 실패를 대체 문자로 흡수해 막고 있었다.
    그 방어는 "로그 파일 기능" 에 딸려 있어, Tee 를 설치하지 않는 경로(단위 테스트,
    모듈 직접 호출)는 무방비였다. 인코딩은 로깅과 독립된 관심사이므로 여기서
    프로세스 진입 시점에 한 번 확정한다.

사용:
    엔트리포인트(graph.py, run_local_pipeline.py, tests/*)에서 다른 무거운
    import 보다 먼저 부른다.

        from core.console import force_utf8_stdio
        force_utf8_stdio()

    errors="replace" 를 함께 주는 이유: UTF-8 로 재설정할 수 없는 환경(파이프가
    닫힌 경우 등)에서도 최소한 예외로 죽지는 않게 하는 보험이다. 문자가 '?' 로
    깨질지언정 파이프라인 상태를 오염시키지는 않는다.
"""
from __future__ import annotations

import sys

_applied = False


def force_utf8_stdio() -> bool:
    """sys.stdout/stderr 를 UTF-8(errors=replace)로 재설정한다.

    한 프로세스에서 한 번만 적용한다(중복 호출은 무시). 재설정을 지원하지 않는
    스트림(이미 교체된 Tee, io 객체가 아닌 더미 등)이면 조용히 건너뛴다 —
    인코딩 보정은 부가 기능이라 이것 때문에 실행이 막히면 안 된다.

    Returns:
        실제로 재설정을 적용했으면 True, 이미 적용됐거나 불가능하면 False.
    """
    global _applied
    if _applied:
        return False

    changed = False
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue    # Tee/더미 스트림 — 자체 인코딩 방어에 맡긴다
        try:
            reconfigure(encoding="utf-8", errors="replace")
            changed = True
        except (ValueError, OSError):
            pass        # 이미 detach 됐거나 재설정 불가 — 무시하고 진행

    _applied = True
    return changed
