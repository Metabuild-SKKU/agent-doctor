# Ingest → Index → Serve 전체 파이프라인 테스트
# Notion 페이지 수집 → 임베딩 → Qdrant → API 서버 → MCP 등록

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from core.state import AgentDoctorState
from agents.ingest.agent import run as ingest_run
from agents.index.agent import run as index_run
from agents.serve.agent import run as serve_run

NOTION_URL = os.getenv("TEST_NOTION_URL", "여기에_노션_URL_입력")

print("=" * 50)
print("Pipeline 테스트: Ingest → Index → Serve")
print("=" * 50)

state = AgentDoctorState()
state.source_url  = "data/pdf_corpus.json"
state.source_type = "json_corpus"

# 1단계: Ingest
print("\n[1/3] Ingest Agent 실행 중...")
state = ingest_run(state)
if state.error:
    print(f"  오류: {state.error}")
    sys.exit(1)
for doc in state.documents:
    print(f"  └ '{doc.metadata.get('title', doc.doc_id)}' ({len(doc.content)}자)")

# 2단계: Index
print("\n[2/3] Index Agent 실행 중...")
state = index_run(state)
if state.error:
    print(f"  오류: {state.error}")
    sys.exit(1)
print(f"  청크 {len(state.chunks)}개 (임베딩 {len(state.chunks[0].embedding)}차원)")

# 3단계: Serve
print("\n[3/3] Serve Agent 실행 중...")
state = serve_run(state)
if state.error:
    print(f"  오류: {state.error}")
    sys.exit(1)

print(f"\n{'=' * 50}")
print("✅ 전체 파이프라인 완료!")
print(f"  문서:  {len(state.documents)}개")
print(f"  청크:  {len(state.chunks)}개")
print(f"  API:   {state.mcp_endpoint}/search?query=검색어")
print("=" * 50)
print("\nClaude Desktop 재시작하면 MCP 연결됩니다.")
print("API 서버 종료하려면 Ctrl+C")
input()
