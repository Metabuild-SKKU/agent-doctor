# Agent Doctor

> 데이터 소스를 연결하면 RAG 파이프라인을 자동으로 구성·진단·최적화하고, AI 클라이언트가 바로 쓸 수 있는 MCP 서버로 제공하는 멀티 에이전트 시스템

---

## 개요

Notion·파일·QA 데이터셋 등 데이터 소스를 입력하면 다음을 자동으로 수행합니다.

1. **Ingest** — 문서를 수집·전처리
2. **Index** — 청킹 → 임베딩 → 벡터 DB(Qdrant) 저장
3. **Eval** — RAG 품질을 진단하고 실패 원인을 라벨링
4. **Optimize** — 품질 미달이면 파라미터를 조정해 재진단 (최대 3회 반복)
5. **Serve** — 최적화된 RAG를 MCP 서버로 노출

```
[Ingest] → [Index] → [Eval] → [Optimize] → [Serve]
                        ↑__________↓
                   품질 미달 시 재색인·재진단 (최대 3회)
```

파이프라인은 [LangGraph](https://github.com/langchain-ai/langgraph)로 오케스트레이션되며, 모든 단계는 공유 상태 객체(`AgentDoctorState`)를 통해 데이터를 주고받습니다.

---

## 빠른 시작

```bash
# 1. 설치
pip install -r requirements.txt

# 2. 환경변수
cp .env.example .env
#   OPENAI_API_KEY, NOTION_TOKEN 등 필요한 값 입력 (없어도 로컬 폴백으로 동작)

# 3. 전체 파이프라인 실행
python graph.py     # 기본: 레포에 포함된 sample_docs/hr_policy.md 로 바로 실행

# 다른 소스를 쓰려면 환경변수로 지정
SOURCE_TYPE=notion  SOURCE_URL=https://notion.so/...  python graph.py
SOURCE_TYPE=korquad SOURCE_URL=data/corpus.jsonl      python graph.py   # data/ 준비 필요
```

`SOURCE_TYPE`을 지정하지 않으면 레포에 포함된 샘플 문서로 바로 실행됩니다.
`korquad`(한국어 QA 데이터셋)로 정량 진단을 돌리려면 `data/corpus.jsonl`·`data/qa_pairs.jsonl`을 직접 준비해야 합니다 (자세한 내용은 [`data/README.md`](data/README.md) 참고).

### 실행 후 API 확인

파이프라인이 Serve 단계까지 완료하면 로컬 FastAPI 서버가 뜹니다 (기본 포트 8766).

```
http://localhost:8766/health                # 서버 상태 · 로드된 청크 수
http://localhost:8766/documents             # 인덱싱된 문서 목록
http://localhost:8766/search?query=재택근무  # 벡터 검색
http://localhost:8766/answer?query=재택근무  # 검색 + LLM 답변 생성
```

---

## 지원 데이터 소스

| `SOURCE_TYPE` | 설명 |
|---|---|
| `korquad` | KorQuAD 형식 QA 데이터셋 (정답·gold 포함 → 정량 진단 가능) |
| `file` | 로컬 파일 (`.txt` / `.md` / `.pdf`) |
| `json_corpus` | 전처리된 JSON 코퍼스 |
| `notion` | Notion API (Access Token 또는 OAuth) |
| `gdrive` | *(미구현)* |

> 진단(Recall·F1·RAGAS)을 제대로 돌리려면 정답과 gold 근거가 있는 소스(`korquad`)가 필요합니다. 정답이 없는 소스는 검색·파이프라인 동작 확인용입니다.

---

## 다른 팀의 RAG를 로그로 진단하기

우리 파이프라인(Ingest→Index)을 거치지 않고, **이미 운영 중인 남의 RAG가 남긴 실행 로그**(질문·검색 컨텍스트·답변)만으로 품질을 진단할 수 있습니다("로그 리플레이" 모드).

```bash
# 골든셋(질문·정답·정답 근거) 없이는 검색축 라벨 3종이 침묵합니다 — 골든셋이 진단의 기본 입력입니다.
python -m agents.eval.replay <log.jsonl> --golden=<golden.xlsx|csv|jsonl|json>

# 정답지가 아예 없으면 명시적으로 옵트인(얕은 진단만 가능)
python -m agents.eval.replay <log.jsonl> --no-golden

# 브라우저로 바로 여는 진단서 HTML을 원하면
python tools/run_replay_report.py <log.jsonl>
```

로그 스키마(JSON Lines, 한 줄 = 요청 1건)와 골든셋 병합 규칙은 [`docs/external_rag_log_intake.md`](docs/external_rag_log_intake.md)에 정리돼 있습니다.

이 모드는 `graph.py`의 LangGraph 루프(Ingest→Index→Eval→Optimize→Serve)에 물리지 않습니다 — 남의 인덱스라 재색인·재검증(Optimize)이 성립하지 않기 때문입니다. 대신 원인 라벨(`ext_` 접두어 7종)에 매핑된 처방을 사람이 읽는 권고 카드로만 돌려줍니다.

---

## 진단과 최적화

### 진단 (Eval)

각 진단 질문(Probe)에 대해 검색·답변을 실행하고, 규칙 지표(Recall@k, 문자 단위 F1)와 선택적 RAGAS(LLM-as-Judge)로 채점한 뒤, 실패 원인을 세분화 라벨로 분류합니다.

- **A. 검색 실패** — 낮은 순위, 어휘/의미 불일치, 미검색 등
- **B. 생성 실패** — 환각, 멀티홉 결합 오류, 부분 답변, 무응답 미기권 등
- **C. 컨텍스트 구조** — 청크 경계 분할, 노이즈 간섭 등
- **D. 데이터** — 코퍼스 결손, 정답셋 오류 등

> **원칙**: 진단 질문은 청크 자체가 아니라 외부 지식(QA 데이터셋·사용자 로그 등)에서 생성합니다. 청크에서 질문을 뽑으면 진단이 순환 논리가 되기 때문입니다.

### 점수와 통과 기준

| 점수 | 범위 | 용도 |
|---|---|---|
| `overall_score` | 0–1 | 품질 단일 축 — Optimize의 탐색 신호 |
| `composite_score` | 0–100 | 품질 × 신뢰도의 조화 평균 — **통과 판정·리포트 헤드라인** |

`composite_score ≥ 90`이고 `mean_recall_at_k ≥ 0.6`이면 통과해 Serve로 이동하고, 미달이면 Optimize가 개입합니다.
근거가 없는 축은 막지 않습니다 — `mean_recall_at_k`가 없으면(gold 청크 미지정) 바닥선을 적용하지 않고, `composite_score`가 없으면(구버전·평가 신호 없음) Eval의 `pass_threshold`를 대신 씁니다.

### 최적화 (Optimize)

진단 라벨에 매핑된 처방을 **한 번에 하나씩** 적용 → 재색인·재진단으로 검증 → 개선되면 유지, 아니면 롤백합니다. 내부 스윕과 RAGBuilder 백엔드를 지원합니다.

---

## 아키텍처

모든 에이전트는 동일한 순수 함수 시그니처를 따르며, 공유 상태만으로 통신합니다.

```python
def run(state: AgentDoctorState) -> AgentDoctorState:
    ...  # state를 읽고, 처리하고, 갱신해 반환
    return state
```

| 필드 | 생성 | 소비 |
|---|---|---|
| `documents` | Ingest | Index |
| `chunks` | Index | Eval · Optimize · Serve |
| `index_config` | Optimize | Index (재실행 시) |
| `report` | Eval | Optimize · 라우팅 |
| `mcp_endpoint` | Serve | (최종 출력) |

```
agent_doctor/
├── graph.py              # LangGraph 파이프라인 (오케스트레이터)
├── core/                 # 공유 스키마(Document·Chunk·Probe·Finding) · 상태 정의
├── agents/
│   ├── ingest/           # 데이터 수집 (Notion / 파일 / QA 데이터셋)
│   ├── index/            # 청킹 · 임베딩 · Qdrant 저장
│   ├── eval/             # 품질 진단 (규칙 지표 + RAGAS + 원인 라벨링)
│   ├── optimize/         # 라벨 기반 처방 · 적용/검증/롤백
│   ├── rag/              # 공용 검색 + 답변 생성 (Eval · Serve가 공유)
│   └── serve/            # FastAPI 서버 + MCP 서버 + Claude Desktop 등록
├── data/                 # 평가 데이터셋 (KorQuAD 등, 별도 준비)
├── sample_docs/          # 샘플 문서
├── docs/                 # 설계 문서 (로그 리플레이 모드 등)
├── tools/                # 단독 실행 스크립트 (진단서 HTML 생성 등)
└── tests/
```

각 에이전트 디렉터리에는 세부 동작을 설명하는 `README.md`가 있습니다.

---

## MCP 연동

Serve 단계는 청크를 저장하고 로컬 API 서버를 띄운 뒤, Claude Desktop 설정에 `agent-doctor` MCP 서버를 자동 등록합니다. Claude Desktop을 재시작하면 다음 도구가 활성화됩니다.

- `search_docs` — 벡터 검색
- `ask_docs` — 검색 + RAG 답변
- `list_documents` — 인덱싱된 문서 목록

MCP 서버(stdio)는 검색을 로컬 FastAPI 서버에 위임합니다.
원격 Serve API로 운영할 때는 `AGENT_DOCTOR_API_URL`만 배포 URL로 바꾸면 같은 MCP 툴이 원격 API를 직접 호출합니다.

주요 MCP 환경 변수:

| 변수 | 기본값 | 설명 |
|---|---:|---|
| `AGENT_DOCTOR_API_URL` | `http://localhost:8766` | MCP 툴이 호출할 Serve API 주소 |
| `AGENT_DOCTOR_CHUNKS_FILE` | 없음 | 로컬 API 자동 기동에 사용할 chunks JSON 경로 |
| `AGENT_DOCTOR_MCP_AUTOSTART` | `1` | 로컬 API가 꺼져 있을 때 `api.py` 자동 기동 여부 |
| `AGENT_DOCTOR_MCP_STARTUP_RETRIES` | `10` | 자동 기동 뒤 `/health` 확인 재시도 횟수 |
| `AGENT_DOCTOR_MCP_MAX_TOP_K` | `20` | MCP 툴에서 허용하는 최대 `top_k` |
| `AGENT_DOCTOR_MCP_SNIPPET_CHARS` | `0` | 검색 결과 본문 절단 길이. `0`이면 전문 반환 |

---

## 기술 스택

| 역할 | 기술 |
|---|---|
| 오케스트레이션 | LangGraph |
| 데이터 수집 | Notion API · pdfplumber |
| 임베딩 | sentence-transformers `BAAI/bge-m3` (1024차원, 한국어 지원) |
| 벡터 DB | Qdrant (기본 in-memory, `QDRANT_URL`로 Cloud 전환) |
| 검색 API | FastAPI + uvicorn |
| 답변 생성 | OpenAI / Gemini / GitHub Models (키 없으면 추출식 폴백) |
| 진단 지표 | 규칙 지표 + RAGAS (LLM-as-Judge, 자체 구현) |
| 최적화 | 라벨 기반 처방 + RAGBuilder |
| MCP 서버 | FastMCP (stdio) |

> RAGAS 라이브러리는 LangChain 1.x / LangGraph와 의존성이 충돌해 직접 포함하지 않고, 프롬프트·알고리즘을 동일하게 재구현했습니다.

---

## 테스트

테스트 러너나 빌드 시스템 없이 개별 스크립트로 실행합니다.

```bash
# 에이전트 단독 (mock 데이터, 외부 의존성 없음)
python tests/check_eval.py               # Eval STEP1~5
python tests/check_index.py              # Index

# unittest 계열
python -m unittest tests.test_index_unit -v
python -m unittest tests.test_optimizer tests.test_optimize_agent tests.test_config_mapper tests.test_ragbuilder_adapter

# 외부 API 필요 (없으면 자동 스킵)
python tests/check_ragas_eval.py         # RAGAS 실측 (OpenAI 키, ~$0.01–0.05)
python tests/check_pipeline.py           # Ingest → Index → Serve (실제 Notion)
```

---

## 현재 한계

- Qdrant는 프로세스 내 in-memory로 동작하며, `QDRANT_URL`을 지정하지 않으면 실행 간 영속되지 않습니다.
- MCP 전송은 stdio 방식(Claude Desktop이 프로세스를 직접 실행)이며, 호스팅 URL 방식은 계획 중입니다.
- Google Drive 수집은 미구현입니다.
