# Serve Agent

최적화된 청크를 MCP 서버로 서빙해 Claude 등 외부 AI가 문서를 검색할 수 있게 함.

---

## 역할

파이프라인에서 가장 마지막으로 실행됨.

```
[Optimize Agent] → state.chunks
                        ↓
                 [Serve Agent]
                        ↓
             Claude Desktop에 MCP 등록
                        ↓
              Claude가 문서 검색 가능
```

---

## 현재 구현 (테스트용 / 로컬)

Index Agent 미완성 상태에서 테스트하기 위한 임시 구현.

```
state.chunks → chunks.json (로컬 파일)
mcp_server.py → chunks.json 읽어서 키워드 검색
claude_desktop_config.json → 자동 등록
Claude Desktop 재시작 → 연결 완료
```

**한계**: 내 PC에서만 동작. 진짜 서비스 아님.

---

## 전체 아키텍처 (현재 vs 목표)

### 현재 (로컬 데모)

```
이용자 PC
├── Agent Doctor 실행
├── chunks.json (로컬 저장)
├── mcp_server.py (로컬 실행)
└── Claude Desktop ← MCP 자동 등록
```

### 목표 (실제 서비스)

```
이용자 브라우저
    ↓ Notion 링크 입력
클라우드 서버 (Agent Doctor)
    ↓ 파이프라인 실행
Qdrant Cloud (벡터 DB)  ← RAG 저장
    ↓
MCP 서버 (클라우드, 항상 실행)
    ↓
https://agentdoctor.com/mcp/{user_id}
    ↓
이용자 → Claude Desktop에 이 URL 한 번만 추가
```

---

## MCP 연결 방식 (현재 vs 향후)

### 현재 — stdio (Claude Desktop)
Claude Desktop이 mcp_server.py를 직접 프로세스로 실행.
이용자 PC에 파일이 있어야 동작.

```json
{
  "mcpServers": {
    "agent-doctor": {
      "command": "python",
      "args": ["mcp_server.py", "--chunks-file", "chunks.json"]
    }
  }
}
```

### 향후 — URL (클라우드 배포 시)
클라우드 MCP 서버 URL을 이용자가 한 번만 등록.
이후 어디서든 본인 문서 검색 가능.

```json
{
  "mcpServers": {
    "agent-doctor": {
      "url": "https://agentdoctor.com/mcp/{user_id}"
    }
  }
}
```

> Claude.ai 웹에서 원격 MCP 지원 중 (베타). 지원되면 이용자가 URL만 붙여넣으면 자동 연결.

---

## 향후 확장 계획

### 1단계 — Qdrant 연동 (Index Agent 완성 후)

`mcp_server.py`의 `_search()` 함수 교체:

```python
# 현재 (키워드 매칭)
score = sum(1 for word in query.split() if word in text)

# 교체 (Qdrant 벡터 검색)
results = qdrant_client.search(
    collection_name="agent_doctor",
    query_vector=embed(query),
    limit=top_k
)
```

### 2단계 — 클라우드 배포

```
Railway / Render / AWS
    → MCP 서버 상시 운영
    → 이용자별 collection_name 분리
    → URL 방식으로 연결
```

### 3단계 — 웹 챗봇 (선택)

MCP 없이도 쓸 수 있게 웹페이지 챗봇 추가.
Claude Desktop 없는 이용자도 사용 가능.

```
웹페이지 챗봇
    ↓ 질문 입력
FastAPI 백엔드
    ↓ Qdrant 검색
LLM (Claude API / OpenAI)
    ↓ 답변 생성
웹페이지에 출력
```

---

## 테스트

```bash
python tests/test_pipeline.py
```

Notion 수집 → 인덱싱 → Serve까지 전체 실행. 완료 후 Claude Desktop 재시작하면 연결됨.

---

## 파일 구조

```
agents/serve/
├── agent.py       # run() — 청크 저장 + Claude Desktop 설정 자동 등록
├── mcp_server.py  # FastMCP 서버 (search_docs, list_documents 툴)
└── README.md      # 이 파일
```

---

## MCP 툴 목록

| 툴 | 설명 |
|----|------|
| `search_docs(query)` | 문서에서 관련 청크 검색 (top 3 반환) |
| `list_documents()` | 인덱싱된 문서 목록 조회 |
