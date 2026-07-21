"""
run_local_pipeline.py
전체 파이프라인 로컬 스모크 — Ingest→Index→Eval→Optimize 실행.

소스는 env 로 고른다(state.source_type/source_url 로 들어감):
  python run_local_pipeline.py                                   # 기본: korquad (data/corpus.jsonl)
  SOURCE_TYPE=file SOURCE_URL=sample_docs/hr_policy.md python run_local_pipeline.py

korquad 설정(둘 다 있어야 함 — corpus 는 Ingest, qa 는 Eval):
  Ingest : SOURCE_TYPE=korquad, SOURCE_URL=data/corpus.jsonl (원문 복원)
  Eval   : EVAL_PROBE_SOURCE=taxonomy → EVAL_TAXONOMY_QA(기본 data/qa_pairs.jsonl) 로드.
           gold 좌표용 corpus 는 state.source_url(=위 SOURCE_URL)을 그대로 재사용(단일화)
  규모   : KORQUAD_MAX_DOCS(문서 수)·KORQUAD_QA_LIMIT(질문 수) — 기본 소규모 스모크
           (전체를 쓰려면 두 값을 0/미설정. 아래는 setdefault 라 shell·.env 로 덮인다)

  EVAL_MODE=deep EVAL_ENABLE_LLM=1 python run_local_pipeline.py  # 생성·RAGAS 채점(API 비용)

Serve(API 서버)는 스킵한다. 서버까지 띄우려면 graph.py 의 run_pipeline 사용.
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

SOURCE_TYPE = os.getenv("SOURCE_TYPE", "korquad").strip().lower()

state = AgentDoctorState()
state.source_type = SOURCE_TYPE
if SOURCE_TYPE == "korquad":
    state.source_url = os.getenv("SOURCE_URL", "data/corpus.jsonl")
    # korquad 는 qa 도 taxonomy 로 함께 세팅(shell/.env 로 덮게 setdefault).
    os.environ.setdefault("EVAL_PROBE_SOURCE", "taxonomy")
    os.environ.setdefault("KORQUAD_MAX_DOCS", "20")          # 스모크 규모(전체는 0/미설정)
    os.environ.setdefault("KORQUAD_QA_LIMIT", "50")
elif SOURCE_TYPE == "file":
    state.source_url = os.getenv("SOURCE_URL", "sample_docs/hr_policy.md")
    state.user_questions = [
        "재택근무 며칠까지 가능해?",
        "연차는 며칠이야?",
        "성과급은 언제 나와?",
    ]
else:
    state.source_url = os.getenv("SOURCE_URL", "")

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
    print(f"overall={state.report.overall_score}  pass={state.report.pass_threshold}")
    print(f"findings_summary: {state.report.findings_summary}")
print(f"index_config: {state.index_config}")
print("\n전체 파이프라인 로컬 스모크 완료 [OK]")
