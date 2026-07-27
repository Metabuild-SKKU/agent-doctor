"""
tests 패키지 초기화.

테스트는 엔트리포인트(graph.py 등)를 거치지 않고 에이전트를 직접 부르므로,
콘솔 인코딩 보정도 스스로 해야 한다. 이게 없으면 한글 Windows 기본 콘솔(cp949)
에서 '—'·'↳' 같은 문자를 print 하는 순간 UnicodeEncodeError 가 나고, 그 예외가
agents/eval/agent.py 의 포괄 except 에 잡혀 정상 완료된 STEP 을 status="error"
로 뒤집는다 — 테스트가 프로덕션 코드가 아니라 콘솔 인코딩 때문에 깨진다.
"""
from core.console import force_utf8_stdio

force_utf8_stdio()
