"""
core 패키지 초기화.

여기서 콘솔 인코딩을 프로세스 진입 시점에 한 번 확정한다. 이유:
    한글 Windows 콘솔의 기본 인코딩은 cp949 라 '—'(em dash) 같은 문자를 담은
    print() 가 그 자리에서 UnicodeEncodeError 를 던진다. 그 예외가 eval/RAG 등의
    포괄 except 에 잡히면 **정상 완료된 STEP 이 status="error" 로 뒤집히거나**
    (agents/eval/agent.py), 정상적인 LLM 생성이 조용히 폴백으로 강등된다
    (agents/rag/generator.py). 로그 한 줄 때문에 파이프라인 결과가 오염되는 것이다.

    기존엔 force_utf8_stdio() 를 엔트리포인트(graph.py, run_local_pipeline.py 등)
    마다 개별로 불렀는데, 그 방어를 빠뜨린 경로(단위 테스트를 __main__ 으로 직접
    실행, 모듈 직접 호출)는 무방비였다. 모든 에이전트·테스트가 core 를 import 하므로,
    여기서 한 번 걸어 두면 진입점이 무엇이든 인코딩이 확정된다. force_utf8_stdio()
    는 idempotent 하고(중복 무시), Tee/재설정 불가 스트림은 조용히 건너뛰므로
    run_logger 의 Tee 나 이미 UTF-8 인 환경과 충돌하지 않는다.
"""
from __future__ import annotations

from core.console import force_utf8_stdio

force_utf8_stdio()
