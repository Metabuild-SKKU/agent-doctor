# Eval Agent 테스트 (Mock Chunks 사용)
# Index Agent 없이도 단독으로 실행 가능. 외부 API(OpenAI/RAGAS) 없이 폴백 경로로 동작.

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# .env 로딩(EVAL_MODE / EVAL_ENABLE_LLM 등). agents import 전에 실행해야 import 시점 env 읽기에도 반영됨.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.llm_usage import print_summary
from core.schema import Chunk
from core.state import AgentDoctorState
from agents.index.qdrant_store import embed
from agents.eval.agent import run
from agents.eval.types import resolve_mode, DEFAULT_TOP_K
from agents.rag.retriever import build_retriever

# ── 1) 순수 규칙 지표 단위 확인 (의존성 0, 결정적) ────────────────
from agents.eval.metrics import token_f1, recall_at_k, is_abstention

print("=" * 50)
print("Eval Agent 테스트 - (1) 규칙 지표 단위 확인")
print("=" * 50)

assert round(token_f1("재택근무는 주 2일 가능", "재택근무는 주 2일 가능"), 3) == 1.0
assert token_f1("전혀 다른 문장", "재택근무 규정") == 0.0
assert recall_at_k(["c1"], ["c1", "c2"]) == 1.0
assert recall_at_k(["c1", "c3"], ["c1", "c2"]) == 0.5
assert recall_at_k([], ["c1"]) == -1.0            # gold 없음
assert is_abstention("제공된 정보로는 알 수 없습니다") is True
assert is_abstention("제공된 컨텍스트에 따르면 재택근무는 주 2일입니다") is False  # 근거 인용은 기권 아님(오탐 회귀)
print("규칙 지표 단위 확인 통과 [OK]")

# ── 2) Mock Chunks (Index 결과 시뮬레이션) ───────────────────────
raw = [
    ("doc_001_chunk_000", "재택근무는 주 2일까지 가능하며 팀장 승인 후 사용합니다."),
    ("doc_001_chunk_001", "재택근무 신청은 전날 오후 6시까지 슬랙으로 제출해야 합니다."),
    ("doc_002_chunk_000", "신입사원 온보딩 기간은 2주이며 첫 주는 교육입니다."),
    ("doc_002_chunk_001", "연차는 15일이고 반차는 4시간 기준입니다."),
    ("doc_002_chunk_002", "경조사 휴가는 별도 규정을 따르며 최대 5일입니다."),
]
mock_chunks = [
    Chunk(chunk_id=cid, doc_id=cid.rsplit("_chunk_", 1)[0], text=text, embedding=embed(text))
    for cid, text in raw
]

# ── 3) Eval 실행 (자동 Probe 생성 경로) ──────────────────────────
print("\n" + "=" * 50)
print("Eval Agent 테스트 - (2) run(state) 전체 실행")
print("=" * 50)

state = AgentDoctorState()
state.chunks = mock_chunks

result = run(state)

print("\n── 생성된 Probe(질문/정답) + 검색 재현 ──")
# STEP2(검색+답변 생성)는 run() 내부 지역 변수(EvalRecord)라 state 밖으로 안 나온다.
# 검색(로컬 Qdrant, 무료)만 재현해 QA셋 품질을 눈검사한다 — 생성답변은 LLM 재호출
# 비용(probe당 1회)이 들고 위 [N/M] 평가 로그에 이미 출력돼 있어 여기선 재생성하지 않는다.
_retriever = build_retriever(mock_chunks, state.index_config)
for p in result.probes:
    print(f"\n[{p.metadata.get('gen_method', p.source)}] qtype={p.qtype}")
    print(f"  질문:      {p.question}")
    print(f"  정답:      {p.ground_truth}")
    hits = _retriever.search(p.question, top_k=DEFAULT_TOP_K)
    print(f"  검색결과:  {[h.get('chunk_id') for h in hits]}")

print(f"\n상태:        {result.status}")
print(f"반복:        {result.iteration}")
if result.error:
    print(f"오류: {result.error}")
    sys.exit(1)

report = result.report
print(f"probe 수:    {len(result.probes)}")
print(f"overall:     {report.overall_score}")
print(f"pass:        {report.pass_threshold}")
print(f"oracle_acc:  {report.oracle_accuracy}")
print(f"ragas_scores:{report.ragas_scores}")
print(f"findings_summary:{report.findings_summary}")

print("\n── Finding 목록 ──")
if not report.findings:
    print("  (없음)")
for f in report.findings:
    mark = "확정" if f.confirmed else "예비"
    print(f"  [{f.severity}] {f.type} / {f.label} "
          f"({mark}·{f.metadata.get('group')}그룹)")

# 계약 검증: run 은 항상 state 를 반환, report 존재
assert result is state, "run()은 동일 state 를 반환해야 함"
assert result.report is not None, "report 가 생성되어야 함"

# findings_summary 계약: 필수 키 존재 + 개수 정합 + 확정우선 정렬
fs = report.findings_summary
for key in ("mode", "total", "confirmed", "preliminary",
            "confirmed_labels", "preliminary_labels"):
    assert key in fs, f"findings_summary 에 {key} 누락"
assert fs["total"] == len(report.findings) == fs["confirmed"] + fs["preliminary"]
# 확정 우선 정렬: 확정 findings 가 예비보다 앞
confirmed_flags = [f.confirmed for f in report.findings]
assert confirmed_flags == sorted(confirmed_flags, reverse=True), "확정 우선 정렬이 아님"
# 진단 모드는 EVAL_MODE(.env/환경변수)로 정해진다 — 하드코딩 대신 resolve_mode 로 대조.
assert fs["mode"] == resolve_mode(), "findings_summary.mode 가 EVAL_MODE 와 일치해야 함"
print("\n전체 파이프라인 스모크 테스트 통과 [OK]")
print_summary(tag="Test")  # run() 내부 + 위 재현 섹션까지 포함한 스크립트 전체 합계
