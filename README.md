# Agent Doctor

> 데이터 소스를 연결하면 자동으로 RAG를 구성·진단·최적화하고, AI가 바로 쓸 수 있는 MCP 서버로 제공하는 멀티 에이전트 시스템

---

## 프로젝트 개요

이용자가 Notion, Google Drive, 파일 등 데이터 소스를 연결하면:

1. 문서를 자동으로 수집하고 전처리
2. 청크로 분할 후 벡터화해서 저장
3. RAG 품질을 자동으로 진단
4. 품질 미달이면 파라미터 자동 조정 후 재진단
5. 최적화된 RAG를 MCP 서버로 외부 AI에 제공

```
이용자: Notion 링크 입력
           ↓
[Ingest] → [Index] → [Eval] → [Optimize] → [Serve]
                        ↑________↓ (품질 미달이면 반복, 최대 3회)
           ↓
"Claude에서 이 주소로 연결하세요: http://localhost:8765/xxx"
```

---

## 팀원별 담당

| 에이전트 | 파일 |
|---------|------|
| Ingest Agent | `agents/ingest/agent.py` |
| Index Agent | `agents/index/agent.py` |
| Eval Agent | `agents/eval/agent.py` |
| Optimize Agent | `agents/optimize/agent.py` |
| Serve Agent | `agents/serve/agent.py` |

**각자 자기 `agent.py`의 `run(state)` 함수만 구현하면 됨.**  
`graph.py`는 건드리지 않아도 됨.

---

## 프로젝트 구조

```
agent_doctor_v2/
├── core/
│   ├── schema.py          # 공통 데이터 모델 (Document, Chunk, Probe 등)
│   └── state.py           # LangGraph 공유 상태 정의
├── agents/
│   ├── ingest/
│   │   ├── agent.py       # 데이터 수집 (Notion / Google Drive / 파일)
│   │   ├── oauth.py       # Notion 인증 (Access Token / OAuth)
│   │   └── README.md
│   ├── index/
│   │   ├── agent.py       # 청크 분할 + 임베딩 + Qdrant 저장
│   │   ├── qdrant_store.py  # Qdrant 클라이언트 / 검색 / 임베딩 공통 모듈
│   │   └── graph_index.py   # NetworkX 그래프 / Mermaid / PyVis
│   ├── eval/
│   │   └── agent.py       # RAG 품질 진단
│   ├── optimize/
│   │   └── agent.py       # 파라미터 자동 조정
│   └── serve/
│       ├── agent.py       # 청크 저장 + API 서버 시작 + Claude Desktop 등록
│       ├── api.py         # FastAPI 검색 서버 (Qdrant → HTTP)
│       ├── mcp_server.py  # MCP 서버 (Claude Desktop ↔ API)
│       └── README.md
├── tests/
│   ├── test_ingest.py     # Ingest Agent 단독 테스트
│   ├── test_oauth.py      # Notion OAuth 테스트
│   └── test_pipeline.py   # Ingest → Index → Serve 전체 파이프라인 테스트
├── graph.py               # LangGraph 파이프라인 (Orchestrator)
├── requirements.txt
└── .env.example
```

---

## 핵심 개념

### 공유 상태 (AgentDoctorState)

모든 에이전트는 `AgentDoctorState`를 읽고 써서 다음 에이전트에 데이터를 전달함.

```python
state.documents    # Ingest가 채움 → Index가 읽음
state.chunks       # Index가 채움  → Eval이 읽음
state.report       # Eval이 채움   → Optimize가 읽음
state.index_config # Optimize가 수정 → Index 재실행 시 적용
state.mcp_endpoint # Serve가 채움
```

### 에이전트 구현 규칙

```python
# 반드시 이 시그니처를 유지할 것
def run(state: AgentDoctorState) -> AgentDoctorState:
    # state 읽고
    # 처리하고
    # state 수정해서 반환
    return state
```

모든 지표가 기준값 이상이면 `report.pass_threshold = True` → Serve로 이동.


## 환경 설정

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 환경변수 설정
cp .env.example .env
# .env 열어서 NOTION_TOKEN, TEST_NOTION_URL 등 입력
```

---

## 테스트

### 전체 흐름

```
.env 설정 (NOTION_TOKEN, TEST_NOTION_URL)
        ↓
python tests/test_pipeline.py
        ↓
  [1] Ingest Agent
      Notion API → 페이지 블록 파싱 → Document 객체 생성
        ↓
  [2] Index Agent
      Document 검증·중복 제거 → Markdown/문단 경계 청킹
      → BGE-M3 임베딩 → Qdrant upsert → graph artifact 생성
      → state.chunks, state.index_artifacts 업데이트
        ↓
  [3] Serve Agent
      chunks.json 저장(임베딩 포함)
      → api.py 백그라운드 시작 → Qdrant 재로드
      → claude_desktop_config.json 자동 등록
        ↓
  Claude Desktop 재시작
        ↓
  agent-doctor MCP 도구 활성화
  → search_docs("질문") → api.py → Qdrant 벡터 검색 → 결과 반환
```

### 사용 기술

| 단계 | 기술 | 설명 |
|------|------|------|
| 문서 수집 | Notion API | 블록 단위 재귀 파싱, 페이지네이션 처리 |
| 인증 | Access Token / OAuth2 | `.env` 토큰 또는 브라우저 OAuth 흐름 |
| 청킹 | Markdown + recursive boundary | 기본 600자, overlap 80 |
| 임베딩 | sentence-transformers | `BAAI/bge-m3` (1024차원, 다국어) |
| 벡터 DB | Qdrant in-memory | 운영 시 `QDRANT_URL`로 Cloud 전환 |
| 검색 | Dense baseline | config로 Hybrid/Reranker 활성화 |
| 그래프 | NetworkX | Mermaid, GraphML, PyVis 출력 |
| 검색 API | FastAPI + uvicorn | `GET /search?query=...` → 벡터 검색 결과 반환 |
| MCP 서버 | FastMCP (stdio) | Claude Desktop이 프로세스로 직접 실행 |
| 상태 공유 | LangGraph `AgentDoctorState` | 에이전트 간 데이터 전달 |

### 에이전트 단독 테스트

자기 담당 에이전트만 테스트할 때 사용. mock 데이터로 실행되므로 의존성 없음.

```bash
python tests/test_ingest.py   # Notion 수집 단독 테스트
python tests/test_oauth.py    # Notion OAuth 브라우저 흐름 테스트
```

### 전체 파이프라인 테스트

```bash
python tests/test_pipeline.py
```

### API 직접 확인

파이프라인 실행 후 브라우저에서:

```
http://localhost:8766/health                    # 서버 상태 및 청크 수 확인
http://localhost:8766/documents                 # 인덱싱된 문서 목록
http://localhost:8766/search?query=재택근무     # 벡터 검색 테스트
```

### 현재 한계 및 향후 계획

| 항목 | 현재 (로컬 테스트) | 향후 (운영) |
|------|------------------|------------|
| 임베딩 모델 | BAAI/bge-m3 (무료, 1024차원) | Eval 결과에 따라 교체 |
| 벡터 DB | Qdrant in-memory (프로세스 내) | Qdrant Cloud (영구 저장) |
| MCP 연결 | stdio (로컬 파일 경로) | URL 방식 (클라우드 배포 후) |
| 청킹 전략 | Markdown 구조 + 경계 기반 분할 | Semantic/parent-child 실험 |

---

## 실행

```bash
python graph.py  # 전체 파이프라인 (graph.py 완성 후)
```

---

## 데이터 모델 (core/schema.py)

### Document
Ingest Agent가 생성. 원본 문서 단위.
```python
Document(doc_id, source, format, content, metadata)
```

### Chunk
Index Agent가 생성. 분할된 텍스트 단위.
```python
Chunk(chunk_id, doc_id, text, embedding, sparse_vector, ...)
```

### Probe
Eval Agent가 생성. 진단용 질문.
> **주의**: 청크에서 질문을 뽑으면 안 됨. 외부 지식 기반으로 생성해야 함.

### Finding
Eval Agent가 생성. 진단 결과 단위.
```python
# type: "gap" | "contradiction" | "duplicate" | "staleness" | "retrieval_failure" | "generation_failure"
Finding(finding_id, type, severity, description, affected_chunks, prescription)
```

### DiagnosticReport
Eval Agent의 최종 리포트. Optimize/Serve Agent가 읽음.
```python
DiagnosticReport(ragas_scores, findings, overall_score, pass_threshold)
```

---

## 기술 스택

| 역할 | 기술 |
|------|------|
| 에이전트 오케스트레이션 | LangGraph |
| 데이터 수집 | - |
| 파일 파싱 | - |
| 임베딩 | sentence-transformers (테스트) / OpenAI (운영) |
| 벡터 DB | Qdrant (in-memory → Cloud) |
| 검색 API | FastAPI + uvicorn |
| 진단 지표 | RAGAS |
| MCP 서버 | FastMCP |
