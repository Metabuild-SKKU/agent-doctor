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
agent_doctor/
├── core/
│   ├── schema.py     # 공통 데이터 모델 (Document, Chunk, Probe 등)
│   └── state.py      # LangGraph 공유 상태 정의
├── agents/
│   ├── ingest/
│   │   └── agent.py  # 데이터 수집 (Notion / Google Drive / 파일)
│   ├── index/
│   │   └── agent.py  # 청크 분할 + 벡터화 + 저장
│   ├── eval/
│   │   └── agent.py  # RAG 품질 진단
│   ├── optimize/
│   │   └── agent.py  # 파라미터 자동 조정
│   └── serve/
│       └── agent.py  # MCP 서버 생성 + 서빙
├── graph.py          # LangGraph 파이프라인 (Orchestrator)
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
# .env 파일 열어서 키 입력
```


---

## 실행

```bash
cd C:\SKKU\agent_doctor
python graph.py
```

또는 코드에서:

```python
from graph import run_pipeline

result = run_pipeline(
    source_url="https://notion.so/your-page-id",
    source_type="notion",
    user_questions=["휴가 정책이 어떻게 돼?"],  # 없으면 자동 생성
)

print(result.mcp_endpoint)  # Claude에 연결할 주소
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
| 임베딩 | - |
| RAG 프레임워크 | - |
| 진단 지표 | RAGAS, - |
| MCP 서버 | - |
