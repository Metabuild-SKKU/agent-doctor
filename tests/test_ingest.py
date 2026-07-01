# Ingest Agent 수동 테스트

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from agents.ingest.agent import run
from core.state import AgentDoctorState

# ── 테스트할 소스 설정 ─────────────────────────────────────────
# source_type: "notion" | "file" | "gdrive"
# notion 사용 시 .env에 NOTION_TOKEN 또는 OAuth 토큰 필요

state = AgentDoctorState(
    source_url="https://app.notion.com/p/360f03a0822180bab3eac6472512572e?source=copy_link",
    source_type="notion",
)

# ── 실행 ──────────────────────────────────────────────────────
result = run(state)
print(f"\n문서 수: {len(result.documents)}")

if result.documents:
    doc = result.documents[0]
    print(f"doc_id  : {doc.doc_id}")
    print(f"format  : {doc.format}")
    print(f"metadata: {doc.metadata}")
    print(f"\n내용 앞부분:\n{doc.content[:300]}")
else:
    print(f"오류: {result.error}")
