# AGENTS.md

AI 코딩 에이전트가 이 저장소에서 작업할 때 반드시 지켜야 할 규칙입니다.
사람용 소개·설치법은 [README.md](README.md)를 참고하세요.

---

## 1. 에이전트 계약(Contract) 규칙

이 프로젝트에서 가장 중요하고, 가장 실수하기 쉬운 부분입니다.

### 필수 시그니처
모든 에이전트는 아래 시그니처를 **그대로** 유지해야 합니다.

```python
def run(state: AgentDoctorState) -> AgentDoctorState:
    # 1. state 읽고
    # 2. 처리하고
    # 3. state 수정해서
    return state          # ← 반드시 state를 반환할 것
```

- **절대 `None`을 반환하지 말 것.** `pass`만 있는 함수는 `None`을 반환해 LangGraph 상태를 깨뜨립니다.
  성공·스킵·수동 조치·오류 어느 경로든 반드시 같은 `state`를 반환합니다.
- **`graph.py`는 수정하지 말 것.** 오케스트레이터이며, 각 에이전트는 자기 `agent.py`의 `run()`만 구현합니다.
- 새 파라미터·설정은 하드코딩하지 말고 `state.index_config` 등 상태를 통해 전달합니다.
- 오류는 예외를 그대로 던지기보다 `state.status = "error"`, `state.error = "..."`로 기록하고 `state`를 반환하는 기존 패턴을 따릅니다 (`agents/ingest/agent.py` 참고).

### 상태 필드 소유권
각 필드는 정해진 생산자만 쓰고, 정해진 소비자만 읽습니다. 남의 필드를 덮어쓰지 마세요.

| 필드 | 쓰는 에이전트(생산자) | 읽는 에이전트(소비자) |
|------|----------------------|----------------------|
| `source_url`, `source_type`, `user_questions` | (파이프라인 입력) | Ingest / Index / Eval |
| `documents` | Ingest | Index |
| `chunks` | Index | Eval, Serve |
| `index_config` | Optimize (수정) | Index |
| `index_artifacts` | Index | (그래프 산출물·통계 출력) |
| `index_cache`, `active_index_key`, `index_cache_hit` | Index | Index, Eval, Optimize |
| `probes`, `report` | Eval | Optimize, Serve, `graph.py` 분기 |
| `diagnosis_cache`, `diagnosis_cache_version` | Eval | Eval |
| `eval_cache`, `active_eval_key`, `eval_cache_hit` | Eval | Eval, Optimize |
| `optimization_history` | Optimize | Optimize (다음 라운드), `route_after_eval()` (pending 확인) |
| `iteration`, `max_iterations` | 반복 제어 | `route_after_eval()` |
| `mcp_endpoint` | Serve | (최종 출력) |
| `status`, `error`, `current_agent` | 모든 에이전트 | 오케스트레이터 (`route_after_optimize()`가 `status`로 분기) |

---

## 2. 코드 컨벤션

- **주석·docstring은 한국어로 작성**합니다. 기존 코드 스타일과 일치시키세요.
- 각 `agent.py` 상단에는 그 에이전트의 **읽기/쓰기 상태 필드**를 docstring으로 명시합니다 (기존 파일 참고).
- **임베딩·검색은 공통 모듈을 통해서만** 수행합니다. 직접 모델을 로드하지 말고 `agents/index/qdrant_store.py`의 `embed()` / `search()`를 사용하세요. Index Agent와 API 서버가 같은 벡터 공간을 공유해야 합니다.
- **청킹/임베딩 전략 교체 지점**이 정해져 있습니다:
  - 청킹: `agents/index/agent.py`의 전략 레지스트리 — `state.index_config["chunk_strategy"]`로
    선택(`fixed`/`markdown`/`recursive`/`markdown_recursive`), 새 전략은 `register_chunk_strategy()`로 등록
  - 임베딩 모델: `agents/index/qdrant_store.py`의 `embed()` (기본 `BAAI/bge-m3`, 1024차원)
- 문서 임베딩과 질의 임베딩은 반드시 같은 모델·차원을 사용해야 합니다. 기존 컬렉션과 차원이
  다르면 오류가 나며, 재생성을 허용하려면 `index_config["recreate_collection_on_dimension_mismatch"] = True`를
  명시적으로 설정합니다.
- 새 의존성을 추가하면 `requirements.txt`에 담당 에이전트 주석과 함께 기록합니다.
- 폴백 설계를 유지합니다: 라이브러리 미설치·검색 실패 시 조용히 대체 경로로 넘어가는 기존 패턴(예: sentence-transformers 미설치 → random 벡터, 벡터 검색 실패 → 키워드 검색)을 깨지 마세요.
- 개발 환경은 **Windows / PowerShell** 기준입니다. 경로 구분자와 셸 명령 구문에 유의하세요.

---

## 3. 아키텍처 요약

데이터 소스를 연결하면 자동으로 RAG를 구성·진단·최적화하고, 완성된 검색을 MCP 서버로 외부 AI에 제공하는 LangGraph 멀티 에이전트 파이프라인입니다.

```
[Ingest] → [Index] → [Eval] → [Optimize] → [Serve]
  수집       벡터화     진단        최적화        제공
                        ↑___________↓
              품질 미달 시 Index부터 재실행 (최대 max_iterations회)
```

- **Ingest** (`agents/ingest/`) — Notion·로컬 파일(txt/md/pdf)·json_corpus 수집 → `documents`. (`oauth.py`: Notion 인증)
- **Index** (`agents/index/`) — 검증·중복제거 → 전략 청킹 → bge-m3 임베딩 → Qdrant 저장 → `chunks`, `index_artifacts`. (`qdrant_store.py`: 클라이언트·검색·임베딩 공통 모듈, `graph_index.py`: 그래프 산출물)
- **Eval** (`agents/eval/`) — Probe 생성 → 검색·생성 → 규칙지표·RAGAS(옵션) → 16개 라벨 원인 진단 → `probes`, `report`. (`EVAL_MODE`로 진단 깊이 조절)
- **Optimize** (`agents/optimize/`) — 진단 라벨 기반 처방을 한 번에 하나씩 적용/롤백 → `index_config`, `optimization_history`. (planner → optimizer → config_mapper → history → reporter)
- **RAG** (`agents/rag/`) — 검색(`retriever.py`) + 답변 생성(`generator.py`, LLM 폴백 포함) 공통 모듈. Serve API와 Eval이 함께 사용. (그래프 노드는 아님)
- **Serve** (`agents/serve/`) — 청크 저장 + FastAPI(`api.py`: `/search`·`/answer`) 기동 + MCP 서버(`mcp_server.py`: `search_docs`/`ask_docs`/`list_documents`) 등록 → `mcp_endpoint`.

### 분기 로직
`graph.py`의 `route_after_eval()` / `route_after_optimize()`가 흐름을 결정합니다.
- `report.pass_threshold`가 `True` → **Serve** (종료)
- `iteration >= max_iterations` → **Serve**. 단, 마지막 처방이 아직 유지/롤백 판정 전(pending)이면 마지막으로 한 번 더 **Optimize**
- 그 외(품질 미달) → **Optimize**
- Optimize 후: `status`가 `applied`/`rolled_back`(config 변경) → **Index** 재색인, 그 외(제안·유지·수동·스킵) → **Serve**

### 설계 포인트
- **api.py ↔ mcp_server.py 분리**: MCP 서버는 검색을 직접 하지 않고 FastAPI에 위임합니다. 운영 전환 시 `AGENT_DOCTOR_API_URL`만 클라우드 URL로 바꾸면 됩니다.
- **공유 상태 패턴**: 모든 에이전트가 `core/state.py`의 `AgentDoctorState` 하나를 릴레이하며 데이터를 전달합니다. 데이터 모델은 `core/schema.py`에 정의되어 있습니다.
