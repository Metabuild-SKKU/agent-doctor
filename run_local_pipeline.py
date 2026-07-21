"""
run_local_pipeline.py
전체 파이프라인 로컬 스모크 — 외부 API/크리덴셜 없이 로컬 파일로 Ingest→Index→Eval→Optimize 실행.

  python run_local_pipeline.py
  EVAL_MODE=full python run_local_pipeline.py     # 진단 깊이 조절

Serve(Ctrl+C 로 멈춰야 하는 API 서버)는 스킵한다. 서버까지 띄우려면 graph.py 의 run_pipeline 사용:
  from graph import run_pipeline
  run_pipeline("sample_docs/hr_policy.md", source_type="file",
               user_questions=["재택근무 며칠까지 가능해?"])
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.run_logger import setup_run_logging
setup_run_logging(prefix="local_pipeline")  # 이후 모든 print 를 콘솔+로그파일에 동시 출력

from core.state import AgentDoctorState
from agents.ingest.agent import run as ingest_run
from agents.index.agent import run as index_run
from agents.eval.agent import run as eval_run
from agents.optimize.agent import run as optimize_run

state = AgentDoctorState()
state.source_url = "sample_docs/hr_policy.md"
state.source_type = "file"
state.user_questions = [
    "재택근무 며칠까지 가능해?",
    "연차는 며칠이야?",
    "성과급은 언제 나와?",
]

STEPS = [
    ("Ingest", ingest_run),
    ("Index", index_run),
    ("Eval", eval_run),
    ("Optimize", optimize_run),
]

for name, fn in STEPS:
    print("\n" + "=" * 56)
    print(f"  {name}")
    print("=" * 56)
    state = fn(state)
    if state.error:
        print(f"[중단] {name} 오류: {state.error}")
        sys.exit(1)

print("\n" + "=" * 56)
print("  결과 요약")
print("=" * 56)
print(f"문서:   {len(state.documents)}개")
print(f"청크:   {len(state.chunks)}개")
print(f"프로브: {len(state.probes)}개")
if state.report:
    from agents.optimize import gate
    # score_pass = Eval 점수 판정, gate = serve/optimize 운영 게이트(점수 + 검색 바닥선)
    print(f"overall={state.report.overall_score}  "
          f"score_pass={state.report.pass_threshold}  "
          f"gate_pass={gate.passes_report(state.report)}")
    print(f"findings_summary: {state.report.findings_summary}")
print(f"index_config: {state.index_config}")
print("\n전체 파이프라인 로컬 스모크 완료 [OK]")
