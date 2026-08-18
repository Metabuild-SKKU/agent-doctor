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
| `runtime_capabilities` | Index | Optimize |
| `index_cache`, `active_index_key`, `index_cache_hit` | Index | Index, Eval, Optimize |
| `probes`, `report` | Eval | Optimize, Serve, `graph.py` 분기 |
| `diagnosis_cache`, `diagnosis_cache_version` | Eval | Eval |
| `eval_cache`, `active_eval_key`, `eval_cache_hit` | Eval | Eval, Optimize |
| `optimization_history`, `blocked_action_attempts`, `completed_action_studies` | Optimize | Optimize (다음 라운드), `route_after_eval()` (pending 확인) |
| `blacklist`, `completed_prescriptions` (구버전 표시용) | Optimize | 리포트 호환 경로만 — 실행 제어에는 쓰지 않는다 |
| `iteration`, `max_iterations`, `optimize_visit_count`, `max_optimize_visits` | 반복 제어 | Optimize, `route_after_eval()` |
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
  - 임베딩 모델: `agents/index/qdrant_store.py`의 `embed()` (기본 `bge-m3`, 1024차원)
  - 임베딩 실행 위치: `INDEX_EMBED_PROVIDER`(색인) / `INDEX_QUERY_EMBED_PROVIDER`(질의).
    기본 `openrouter`이며 `local`로 내릴 수 있습니다. 질의 축을 안 정하면 색인 축을
    따릅니다. 아래 "임베딩 provider" 참고
  - 리랭크 실행 위치: `INDEX_RERANKER_PROVIDER`(기본 `local`) / `INDEX_RERANKER_DEVICE`.
    임베딩과 기본값이 반대이고 경로를 바꾸면 모델도 바뀝니다. 아래 "리랭크 provider" 참고
- 문서 임베딩과 질의 임베딩은 반드시 같은 모델·차원을 사용해야 합니다. 기존 컬렉션과 차원이
  다르면 오류가 나며, 재생성을 허용하려면 `index_config["recreate_collection_on_dimension_mismatch"] = True`를
  명시적으로 설정합니다.
- **`try` 블록 안의 `print()` 문구에는 cp949에 없는 기호(em-dash `—`, 이모지, `✓`, `⚠`, `↳` 등)를 쓰지 않습니다.** 진입점이 `core.console.force_utf8_stdio()`를 부르면 보호되지만, 그걸 거치지 않는 경로(모듈 단독 실행·import)에서는 한국어 Windows 콘솔에서 `UnicodeEncodeError`가 나고, 바깥 `except`가 그 예외를 삼켜 폴백·복구 경로까지 무력화됩니다(pytest는 stdout을 utf-8로 캡처해 이 경로를 못 잡습니다). 한글 본문과 `·`, `-`, `→`는 cp949에서 안전합니다. 주석·docstring은 무해합니다.
- 리랭커 입력 토큰 상한은 `INDEX_RERANKER_MAX_LENGTH`(기본 1024, `0` 이하면 모델 기본값 8192)로 조절합니다. **폭주 차단용 안전망이지 비용 절감 장치가 아닙니다** — 기본 1024는 정책 내 구성(최대 1500자 ≈ 854토큰)에서는 발동하지 않고, 패딩이 batch-longest라 상한보다 짧은 입력의 계산량도 줄지 않습니다. 리랭크 비용을 실제로 줄이려면 `rerank_candidates`(쌍 수)를 낮추거나 리랭커를 끄세요.
- 새 의존성을 추가하면 `requirements.txt`에 담당 에이전트 주석과 함께 기록합니다.
- 폴백 설계를 유지합니다: 라이브러리 미설치·검색 실패 시 조용히 대체 경로로 넘어가는 기존 패턴(예: sentence-transformers 미설치 → random 벡터, 벡터 검색 실패 → 키워드 검색)을 깨지 마세요.
- 개발 환경은 **Windows / PowerShell** 기준입니다. 경로 구분자와 셸 명령 구문에 유의하세요.

---

### 임베딩 provider (색인·질의·채점)

임베딩은 **어느 모델**이냐와 **어디서 계산하느냐**가 분리돼 있습니다. 모델은 `bge-m3`로
고정이고, 계산 위치만 env로 고릅니다.

| env | 기본값 | 대상 |
|---|---|---|
| `INDEX_EMBED_PROVIDER` | `openrouter` | 문서 색인 |
| `INDEX_QUERY_EMBED_PROVIDER` | `INDEX_EMBED_PROVIDER` 따름 (없으면 `openrouter`) | 검색 질의 |
| `INDEX_EMBED_DEVICE` | `auto` | `local`일 때 `cuda`/`cpu` |
| `EVAL_EMBED_PROVIDER` | (미설정 = 심판 provider 따름) | Eval 축(RAGAS relevancy) — `openrouter`/`openai`/`gemini`/`local`. **명시하면 강제값**: 못 쓰면 결측이지 다른 provider 로 새지 않는다. 심판=anthropic 이면 로컬 폴백이 리랭커와 VRAM 을 다투므로 `openrouter` 권장 |

**강제값 정책은 "받는 값을 적었는데 그 경로를 못 쓸 때"의 이야기입니다.** 받는 값 밖
(오타, 그리고 임베딩 엔드포인트가 없는 `anthropic`·`github`)은 "설정을 못 읽었다 = 안 적은
것"으로 보고 기본 폴백 사슬을 그대로 탑니다 — 철자 하나가 진단을 통째로 끄는 것이 더
나쁘기 때문입니다. 색인축은 같은 상황을 `local`로 떨어뜨리는데
(`qdrant_store.resolve_embedding_provider` — "오타가 '조용히 API 과금'이 아니라 '조용히
로컬'로"), 채점축이 방향을 달리하는 이유는 비용 구조입니다: 색인은 코퍼스 전량이라 오타
한 번이 통째로 과금되지만, 채점 임베딩은 probe 수십 건이라 금액이 작고 대신 **경로가
바뀌면 코사인 분포가 달라져 실행 간 비교가 깨집니다.** 색인축의 fail-safe 가 막으려는 건
"조용히 API 과금"인데, 채점축은 방향 대신 **"조용히"를 없앱니다** — 경고가 이번 실행의
귀착지 provider 이름을 찍으므로, `local`의 철자 오타처럼 값 목록으로 못 막는 잔여도
로그에 남습니다.

실측(한국어 1,000청크 — 측정 도구 `tools/bench_embedding.py` 는 측정값을 여기와 커밋에 박제한 뒤 제거했습니다. 재검증이 필요하면 `git log --diff-filter=D -- tools/bench_embedding.py` 로 복원): 로컬 CPU 2 chunks/sec vs
OpenRouter 동시 8에서 371 chunks/sec. 26MB 코퍼스 환산으로 **2.3시간 vs 0.7분 / $0.06**
입니다.

**기본값이 `openrouter`인 이유**는 이 프로젝트가 OpenRouter 예산이 확보된 상태로
운영된다는 전제 때문입니다. 그 전제에서는 CPU로 돌릴 이유가 사실상 없습니다 —
100배 넘는 시간 차이를 몇 센트로 사는 셈이라 "예산이 있는데 CPU"라는 선택지가
실질적으로 없다고 보고 기본값을 빠른 쪽에 뒀습니다. 예산이 없거나 오프라인이면
`local`로 내리면 됩니다.

바꿔도 안전합니다: 로컬 `BAAI/bge-m3`와 OpenRouter `baai/bge-m3`의 벡터는
**코사인 0.99997**이고 차원도 같아, provider를 바꿔도 컬렉션을 다시 만들 필요가 없고
색인·질의를 서로 다른 경로로 계산해도 순위가 흔들리지 않습니다.

실행할 때 바꾸려면 플래그를 씁니다. 엔트리포인트가 `load_dotenv(override=True)`라
셸 값보다 `.env`가 이기므로, `INDEX_EMBED_PROVIDER=local python graph.py`는 조용히
무시됩니다. 플래그는 `.env` 로드 뒤에 적용돼 항상 이깁니다.

```bash
python graph.py --embed openrouter        # API
python graph.py --embed gpu               # 로컬 CUDA (없으면 경고 후 CPU)
python graph.py --embed cpu               # 로컬 CPU
python graph.py --embed openrouter --query-embed cpu   # 색인만 API
```

`run_local_pipeline.py`와 `tests/run_corpus.py`도 같은 플래그를 받습니다. 소스 선택
(`SOURCE_TYPE`/`SOURCE_URL`)은 기존대로 env 규약입니다.

테스트는 예외입니다. `tests/conftest.py`가 스위트 전역에서 두 값을 `local`로 고정합니다
— 키가 있는 개발 머신에서 스위트를 돌릴 때마다 실제 API가 호출돼 조용히 과금되는 것을
막기 위함이며, **프로덕션 기본값과는 무관합니다.**

색인 중 임베딩 API가 **일시적으로** 실패(5xx·타임아웃)해 문서가 빠지면
`state.status`가 `indexed`가 아니라 `partial`이 됩니다. 라우팅은 `error`만 보므로
파이프라인은 계속 진행되지만, 불완전한 컬렉션 위에서 나온 Eval 점수가 "이 코퍼스의
성능"으로 오인되지 않게 표시가 남습니다. 빈 문서나 `doc_id` 충돌 같은 **영구** 실패는
다시 돌려도 같으므로 `partial`로 올리지 않습니다(`index_artifacts['failed_documents']`의
`transient` 플래그로 구분).

**`openrouter`는 `OPENROUTER_API_KEY`가 필요합니다.** 없으면 색인은 예외로 멈추고,
질의는 retriever 진입 시 preflight가 끊습니다 — 둘 다 조용히 로컬로 새지 않습니다.
질의를 폴백에 맡기면 모든 검색이 keyword로 떨어지면서 증상은 "검색 품질이 좀 나쁘다"
로만 보여 원인 추적이 불가능하기 때문입니다.

두 축을 나눈 이유는 성질이 달라서입니다. 색인은 대량·일회성이라 429 재시도가 남는
장사지만, 질의는 단건·대화형이라 재시도가 그대로 사용자 지연이 됩니다. 실측 429가
19%라 단건 질의가 재시도에 걸리면 `/search` 한 건이 수십 초 블로킹될 수 있고,
그때 `INDEX_QUERY_EMBED_PROVIDER=local`만 내리면 됩니다.

Eval의 RAGAS `response_relevancy` 임베딩은 기본적으로 `EVAL_LLM_PROVIDER`를 따르고,
`EVAL_EMBED_PROVIDER`로 따로 지정할 수 있습니다(위 표). `anthropic`·`github`는 임베딩
엔드포인트가 없어 이 값으로는 받지 않습니다 — 받아도 계산할 곳이 없어 위 문단의 폴백
사슬로 되돌아오기 때문입니다. 되돌아온 실행은 심판축 사슬을 그대로 타므로
`OPENAI_API_KEY`가 있으면 OpenAI(`text-embedding-3-small`)로 나갈 수 있습니다. 거부가
막는 것은 그 호출이 아니라 **침묵**입니다 — 경고가 이번 실행의 귀착지를 이름으로 찍습니다.
**provider를 바꾸면 코사인 분포가 달라지므로 실행 간 비교를 하려면 한 번 정한 뒤
고정하세요.**

---

### 리랭크 provider (cross-encoder)

리랭크도 **어디서 계산하느냐**를 고를 수 있습니다. 임베딩과 같은 3지선다지만
**기본값이 반대(`local`)**이고, 임베딩과 달리 **경로를 바꾸면 모델도 바뀝니다.**

| env | 기본값 | 대상 |
|---|---|---|
| `INDEX_RERANKER_PROVIDER` | `local` | `local` \| `openrouter` |
| `INDEX_RERANKER_MODEL_OPENROUTER` | `voyageai/rerank-2.5-lite` | `openrouter`일 때 부를 모델 |
| `INDEX_RERANKER_DEVICE` | `INDEX_EMBED_DEVICE` 따름 (없으면 `auto`) | `local`일 때 `cuda`/`cpu` |

```bash
python graph.py --rerank gpu              # 로컬 CUDA (없으면 경고 후 CPU)
python graph.py --rerank cpu              # 로컬 CPU
python graph.py --rerank openrouter       # API (질의마다 과금)
python graph.py --embed openrouter --rerank gpu   # 임베딩만 API, 리랭크는 로컬 GPU
```

**기본값이 `local`인 이유는 과금 시점입니다.** 임베딩은 색인 1회지만 리랭크는
**질의마다** 부과됩니다. Eval 한 번이 질문 수만큼이고 Optimize는 같은 셋을 반복
평가하므로, 기본값을 API로 두면 실행할수록 조용히 곱해집니다. 임베딩 쪽 근거("예산이
있으면 API가 100배 빠르다")는 여기 그대로 적용되지 않습니다 — 리랭크 후보는 보통
20건이라 로컬 GPU로도 한 자릿수 ms입니다.

API 모델 기본값이 `voyageai/rerank-2.5-lite`인 이유: 토큰 과금($0.02/1M)이라 이
워크로드(후보 20건 × ~292토큰 ≈ 검색당 $0.00012, KorQuAD 스모크 한 바퀴 ≈ $0.03)에서
`cohere/rerank-v3.5`($0.001/search)보다 약 8배 싸고, 컨텍스트도 32K로 후보 전체가
들어갑니다(v3.5는 4K라 정책 내 구성에서도 넘칠 수 있음). 무료 nvidia 계열은 free 티어의
분당·일당 상한이 Eval의 버스트 호출과 충돌하고(한도에 걸리면 리랭크가 조용히 빠진 채
점수가 나옴) 프롬프트 로깅 정책도 있어 기본값으로 두지 않습니다.

**바꾸면 안전하지 않습니다.** 로컬 `BAAI/bge-reranker-v2-m3`와 Voyage/Cohere 계열은
서로 다른 모델이라 점수 스케일도 순위도 다릅니다(임베딩의 코사인 0.99997 같은 호환성이
없습니다). OpenRouter 카탈로그에 bge 계열이 없어 이름 변환도 불가능합니다. 그래서 실제
경로가 실행마다 남습니다 — `runtime_summary.reranker.routes`(리포트)와
`runtime_capabilities.reranker.route`(Index preflight). **경로가 다른 두 실행의 점수를
Optimize의 처방 효과로 읽지 마세요.**

`openrouter`인데 `OPENROUTER_API_KEY`가 없으면 리랭크만 건너뛰고 원순위를 유지합니다
(임베딩처럼 예외로 끊지 않습니다 — 리랭커는 optional이라 여기서 던지면 키 오타 하나로
모든 검색이 죽습니다). 대신 capability에 `api_key_missing`이 재시도 불가로 남아 Index
preflight 단계에서 드러납니다.

로컬 경로에서 CUDA OOM이 나면(로드·추론 모두) 경고하고 CPU로 내려 그 실행 동안
고정합니다. 쿨다운 후 같은 GPU로 재시도하면 같은 자리에서 계속 죽기 때문입니다.

API 경로에는 입력 길이 상한(`INDEX_RERANKER_MAX_LENGTH`)이 걸리지 않습니다 — 토크나이저가
로컬에만 있기 때문이고, 리포트에는 `max_lengths: [None]`으로 남습니다. 정책 밖 구성으로
후보 전체가 모델 컨텍스트(voyage-2.5-lite 기준 32K)를 넘기면 API가 400을 내고, 그 실행의
리랭크는 쿨다운 동안 건너뜁니다(원순위 유지). 후보 20건 x 1500자 기준 약 17K라 정책 내
구성에서는 닿지 않습니다.

## 3. 아키텍처 요약

데이터 소스를 연결하면 자동으로 RAG를 구성·진단·최적화하고, 완성된 검색을 MCP 서버로 외부 AI에 제공하는 LangGraph 멀티 에이전트 파이프라인입니다.

```
[Ingest] → [Index] → [Eval] → [Optimize] → [Serve]
  수집       벡터화     진단        최적화        제공
                        ↑___________↓
              품질 미달 시 Index부터 재실행 (최대 max_iterations회)
```

- **Ingest** (`agents/ingest/`) — Notion·로컬 파일(txt/md/pdf)·json_corpus 수집 → `documents`. (`oauth.py`: Notion 인증)
- **Index** (`agents/index/`) — 검증·중복제거 → 전략 청킹 → bge-m3 임베딩(OpenRouter 또는 로컬, `INDEX_EMBED_PROVIDER`) → Qdrant 저장 → `chunks`, `index_artifacts`. (`qdrant_store.py`: 클라이언트·검색·임베딩 공통 모듈, `graph_index.py`: 그래프 산출물)
- **Eval** (`agents/eval/`) — Probe 생성 → 검색·생성 → 규칙지표·RAGAS(옵션) → 16개 라벨 원인 진단 → `probes`, `report`. (`EVAL_MODE`로 진단 깊이 조절)
- **Optimize** (`agents/optimize/`) — 진단 라벨 기반 처방을 한 번에 하나씩 적용/롤백 → `index_config`, `optimization_history`. (planner → optimizer → config_mapper → history → reporter)
- **RAG** (`agents/rag/`) — 검색(`retriever.py`) + 답변 생성(`generator.py`, LLM 폴백 포함) 공통 모듈. Serve API와 Eval이 함께 사용. (그래프 노드는 아님)
- **Serve** (`agents/serve/`) — 청크 저장 + FastAPI(`api.py`: `/search`·`/answer`) 기동 + MCP 서버(`mcp_server.py`: `search_docs`/`ask_docs`/`list_documents`) 등록 → `mcp_endpoint`.

### 분기 로직
`graph.py`의 `route_after_eval()` / `route_after_optimize()`가 흐름을 결정합니다.
- `gate.passes_report(report)`가 `True`(composite ≥ 90 + recall 바닥선) → **Serve** (종료)
- `iteration >= max_iterations` → **Serve**. 단, 마지막 처방이 아직 유지/롤백 판정 전(pending)이면 마지막으로 한 번 더 **Optimize**
- 그 외(품질 미달) → **Optimize**
- Optimize 후: `status`가 `applied`/`rolled_back`(config 변경) → **Index** 재색인, 그 외(제안·유지·수동·스킵) → **Serve**

### 설계 포인트
- **api.py ↔ mcp_server.py 분리**: MCP 서버는 검색을 직접 하지 않고 FastAPI에 위임합니다. 운영 전환 시 `AGENT_DOCTOR_API_URL`만 클라우드 URL로 바꾸면 됩니다.
- **공유 상태 패턴**: 모든 에이전트가 `core/state.py`의 `AgentDoctorState` 하나를 릴레이하며 데이터를 전달합니다. 데이터 모델은 `core/schema.py`에 정의되어 있습니다.
