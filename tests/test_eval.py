# Eval Agent 테스트 (Mock Chunks 사용)
# Index Agent 없이도 단독으로 실행 가능. 외부 API(OpenAI/RAGAS) 없이 폴백 경로로 동작.

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import Chunk
from core.state import AgentDoctorState
from agents.index.qdrant_store import embed
from agents.eval.agent import run

# ── 1) 순수 규칙 지표 단위 확인 (의존성 0, 결정적) ────────────────
from agents.eval.metrics import token_f1, recall_at_k, decide_branch, is_abstention
from agents.eval.types import Branch

print("=" * 50)
print("Eval Agent 테스트 - (1) 규칙 지표 단위 확인")
print("=" * 50)

assert round(token_f1("재택근무는 주 2일 가능", "재택근무는 주 2일 가능"), 3) == 1.0
assert token_f1("전혀 다른 문장", "재택근무 규정") == 0.0
assert recall_at_k(["c1"], ["c1", "c2"]) == 1.0
assert recall_at_k(["c1", "c3"], ["c1", "c2"]) == 0.5
assert recall_at_k([], ["c1"]) == -1.0            # gold 없음
assert decide_branch(1.0, 0.9, 0.0, True, False) == Branch.SUCCESS
assert decide_branch(0.0, 0.0, 0.9, True, False) == Branch.RETRIEVAL_FAIL
assert decide_branch(0.0, 0.0, 0.0, True, False) == Branch.RETRIEVAL_GEN_FAIL
assert decide_branch(1.0, 0.2, 0.9, True, False) == Branch.AMBIGUOUS_CONTEXT
assert decide_branch(1.0, 0.2, 0.2, True, False) == Branch.AMBIGUOUS_GEN
assert decide_branch(0.0, 0.0, 0.0, False, True) == Branch.NO_ANSWER_OK
assert decide_branch(0.0, 0.9, 0.0, False, False) == Branch.NO_ANSWER_VIOLATION
assert is_abstention("제공된 정보로는 알 수 없습니다") is True
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

print("\n── Finding 목록 ──")
if not report.findings:
    print("  (없음)")
for f in report.findings:
    print(f"  [{f.severity}] {f.type} / {f.label} ({f.metadata.get('group')}그룹)")

# 계약 검증: run 은 항상 state 를 반환, report 존재
assert result is state, "run()은 동일 state 를 반환해야 함"
assert result.report is not None, "report 가 생성되어야 함"
print("\n전체 파이프라인 스모크 테스트 통과 [OK]")
