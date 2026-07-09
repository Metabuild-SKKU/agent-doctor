# Index Agent 테스트 (Mock Documents 사용)
# Ingest Agent 없이도 단독으로 테스트 가능

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import Document
from core.state import AgentDoctorState
from agents.index.agent import run

# ── Mock Documents (Ingest 결과 시뮬레이션) ────────────────────────
# 실제 사용 시 Ingest Agent 결과인 state.documents 사용
mock_docs = [
    Document(
        doc_id="doc_001",
        source="https://notion.so/example-page",
        format="notion",
        content=(
            "재택근무는 주 2일까지 가능합니다. 팀장 승인 후 사용하세요. "
            "재택근무 신청은 전날 오후 6시까지 슬랙으로 제출해야 합니다. "
            "재택근무 장비는 회사 지급 노트북만 사용 가능합니다."
        ),
        metadata={"title": "사내 규정 2024", "author": "HR팀"},
    ),
    Document(
        doc_id="doc_002",
        source="https://notion.so/hr-guide",
        format="notion",
        content=(
            "신입사원 온보딩 기간은 2주입니다. 첫 주는 교육, 둘째 주는 실무 배치입니다. "
            "연차는 15일이며 반차는 4시간 기준입니다. "
            "경조사 휴가는 별도 규정을 따릅니다."
        ),
        metadata={"title": "HR 가이드", "author": "HR팀"},
    ),
]

# ── 실행 ──────────────────────────────────────────────────────────
state = AgentDoctorState()
state.documents = mock_docs
# 단독 테스트에서는 외부 LLM 호출 없이 재현 가능한 keyword graph를 사용한다.
state.index_config["graph_extraction"] = "keyword"

print("=" * 50)
print("Index Agent 테스트 시작")
print("=" * 50)

result = run(state)

# ── 결과 출력 ─────────────────────────────────────────────────────
print(f"\n상태:    {result.status}")
print(f"청크 수: {len(result.chunks)}")

if result.error:
    print(f"오류: {result.error}")
else:
    print("\n── 청크 목록 ──")
    for chunk in result.chunks:
        embedding_preview = chunk.embedding[:3] if chunk.embedding else []
        print(f"\n[{chunk.chunk_id}]")
        print(f"  텍스트:     {chunk.text[:60]}...")
        print(f"  임베딩 차원: {len(chunk.embedding)}")
        print(f"  임베딩 앞부분: {[round(v, 4) for v in embedding_preview]}...")
        print(f"  메타데이터: {chunk.metadata}")
        print(f"  char_span: {chunk.char_span}  token_count: {chunk.token_count}  hash: {chunk.hash}")

    # ── 간단 유사도 검색 테스트 ──────────────────────────────────
    print("\n── 검색 테스트 ──")
    try:
        import numpy as np
        from agents.index.qdrant_store import embed

        query = "재택근무 며칠이야?"
        query_vec = np.array(embed(query))

        scored = []
        for chunk in result.chunks:
            vec = np.array(chunk.embedding)
            score = float(np.dot(query_vec, vec))   # cosine (normalize=True)
            scored.append((score, chunk))
        scored.sort(reverse=True)

        print(f"쿼리: '{query}'")
        for score, chunk in scored[:2]:
            print(f"  [{score:.4f}] {chunk.text[:60]}...")

    except ImportError:
        print("numpy 미설치 → 검색 테스트 생략 (pip install numpy)")
