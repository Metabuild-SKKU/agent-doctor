"""
run_local_pipeline.py
전체 파이프라인 로컬 스모크 — Ingest→Index→Eval→Optimize 실행.

데이터셋은 PIPELINE_DATASET 로 고른다:
  python run_local_pipeline.py                          # 기본: korquad (data/)
  PIPELINE_DATASET=file python run_local_pipeline.py    # sample_docs/hr_policy.md 데모

korquad 설정(둘 다 있어야 함 — corpus 는 Ingest, qa 는 Eval):
  Ingest : source_type="korquad", source_url=data/corpus.jsonl (원문 복원)
  Eval   : EVAL_PROBE_SOURCE=taxonomy → data/qa_pairs.jsonl 의 사람 정답+gold 를 로드
  규모   : KORQUAD_MAX_DOCS(문서 수)·KORQUAD_QA_LIMIT(질문 수) — 기본 소규모 스모크
           (전체를 쓰려면 두 값을 크게/미설정. 아래는 setdefault 라 shell·.env 로 덮인다)

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

DATASET = os.getenv("PIPELINE_DATASET", "korquad").strip().lower()

state = AgentDoctorState()
if DATASET == "korquad":
    # 두 설정을 여기서 고정한다(shell/.env 로 덮을 수 있게 setdefault).
    state.source_type = "korquad"
    state.source_url = os.getenv("KORQUAD_CORPUS", "data/corpus.jsonl")
    os.environ.setdefault("EVAL_PROBE_SOURCE", "taxonomy")   # qa 를 taxonomy 로 주입
    os.environ.setdefault("KORQUAD_MAX_DOCS", "20")          # 스모크 규모(전체는 0/미설정)
    os.environ.setdefault("KORQUAD_QA_LIMIT", "50")
    # taxonomy 경로는 EVAL_TAXONOMY_QA/CORPUS 로도 조정 가능(기본 data/*.jsonl)
else:
    state.source_type = "file"
    state.source_url = "sample_docs/hr_policy.md"
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
    print(f"overall={state.report.overall_score}  pass={state.report.pass_threshold}")
    print(f"findings_summary: {state.report.findings_summary}")
print(f"index_config: {state.index_config}")
print("\n전체 파이프라인 로컬 스모크 완료 [OK]")
