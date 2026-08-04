# Notion OAuth 흐름 테스트

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#
# 사전 준비:
#   1. notion.so/my-integrations 에서 Public integration 생성
#   2. Redirect URI: http://localhost:8765/callback 등록
#   3. .env에 아래 두 값 추가
#      NOTION_CLIENT_ID=xxx
#      NOTION_CLIENT_SECRET=xxx

from dotenv import load_dotenv
load_dotenv()

from agents.ingest.oauth import get_notion_token, load_token

print("=" * 50)
print("Notion OAuth 테스트")
print("=" * 50)

# 브라우저 자동 오픈 → Notion 로그인 → 승인 → 토큰 저장
token = get_notion_token(user_id="test_user", use_oauth=True)

print(f"\n토큰 앞 20자: {token[:20]}...")
print(f"저장 확인:    {load_token('test_user')[:20]}...")
print("\n✅ OAuth 성공 — 이제 tests/test_ingest.py 돌려봐")
