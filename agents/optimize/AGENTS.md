# Optimize Agent 작업 규칙

이 파일은 `agents/optimize/`와 그 하위 파일에 적용된다. 저장소 루트의
[`AGENTS.md`](../../AGENTS.md)를 함께 따르며, 충돌할 때는 루트 계약과 실제 공통
스키마를 우선한다. 사람용 설계 소개는 [`README.md`](README.md), 설계 배경은
[`CONTEXT.md`](CONTEXT.md), 시점별 구현 현황은 [`PROGRESS.md`](PROGRESS.md)를
참고한다.

## 1. 작업 전 확인 순서

1. 루트 `AGENTS.md`의 상태 소유권과 `run()` 계약을 읽는다.
2. `git status --short`와 작업 대상의 diff를 확인한다. 현재 워킹트리에는 다른
   작업자의 미커밋 변경이 있으므로 관련 없는 수정은 정리하거나 덮어쓰지 않는다.
3. `core/state.py`, `core/schema.py`, `graph.py`와 Optimize 소비자인
   `agents/index/agent.py`, Serve/Eval 코드를 확인한다.
4. 이 디렉터리의 실제 코드와 테스트를 우선하고, `README.md`, `CONTEXT.md`, 첨부
   메모는 설계 의도로 사용한다. 문서와 코드가 다르면 차이를 숨기지 말고 테스트와
   진행상황 문서를 함께 갱신한다.
5. 새 처방이나 config를 추가하기 전에 Eval이 해당 라벨·신호를 생산하는지, 그리고
   Index/Serve/Eval이 변경값을 실제로 소비하는지 확인한다.

## 2. 최상위 계약

`agent.py`의 공개 진입점은 아래 시그니처를 그대로 유지한다.

```python
def run(state: AgentDoctorState) -> AgentDoctorState:
    return state
```

- `pass`나 암시적 `None` 반환을 금지한다. 성공·스킵·수동 조치·오류를 포함한 모든
  경로에서 반드시 같은 `state` 객체를 반환한다.
- `agent.py` 상단 docstring에 읽는 필드와 쓰는 필드를 한국어로 명시한다.
- 기본 입력은 `state.report`, `state.index_config`, `state.iteration`이다.
- Optimize가 직접 갱신할 수 있는 기존 공통 필드는 `state.index_config`,
  `state.iteration`, `state.status`, `state.error`, `state.current_agent`뿐이다.
  `report`, `probes`, `chunks`, `documents` 등 다른 에이전트의 결과를 덮어쓰지 않는다.
- decision/result/report/history 같은 새 상태 필드가 필요하면 먼저
  `AgentDoctorState`의 정식 필드와 직렬화 계약을 합의한다. README의 동적 속성 예시는
  임시 아이디어이지 확정 계약이 아니다.
- 예외는 바깥으로 그대로 전파하지 말고 `state.status = "error"`, 원인을 알 수 있는
  `state.error`, `state.current_agent = "optimize"`를 기록한 뒤 `state`를 반환한다.
  다만 실패를 `applied`나 `skipped`로 위장하지 않는다.
- `iteration`은 후보 trial 수가 아니라 라벨 처리 단계를 센다. 이전에 실제 적용한
  라벨이 없거나 새 request의 라벨과 다를 때만 `agent.py`에서 정확히 한 번 증가시킨다.
  같은 라벨의 top-k sweep 후보와 후속 처방은 반복 횟수를 소비하지 않으며 Eval도
  `iteration`을 증가시키지 않는다.
- `graph.py`는 이 모듈 작업 범위에서 수정하지 않는다. 사용자 선택 분기나 새 route가
  필요하면 상태·오케스트레이터 소유자와 계약을 먼저 확정하고 통합 차단 요인으로
  기록한다.

## 3. 제품 원칙: 진단 후 단일 처방

AgentDoctor의 차별점은 brute-force로 최적값만 찾는 것이 아니라 실패 원인과 처방
이유를 설명하는 데 있다. 다음 원칙을 깨지 않는다.

- 라벨 판별과 oracle/topic/fact 신호 계산은 Eval 책임이다. Optimize는
  `DiagnosticReport.findings`와 `Finding.label`/`metadata`를 신뢰하고 처방만 만든다.
  코퍼스 재조회, LLM 재진단, 라벨 재판정 로직을 Optimize에 넣지 않는다.
- 한 번에 최상위 라벨 하나와 처방 하나만 선택한다. 한 라벨의 여러 config를 한꺼번에
  바꾸지 말고, `rules.py`에 정의된 순서대로 하나씩 적용하고 Eval로 검증한다.
- 성공하면 유지하고 다음 진단으로 진행한다. 무개선·악화 또는 전역 하한선 위반이면
  이전 config로 rollback하고 `(label, prescription_id)`를 blacklist에 기록한 뒤 다음
  후보를 시도한다.
- `diagnosis_confidence`는 진단 신호의 명확도를 나타내는 규칙 상수이고, `impact`는
  처방 후 실측되는 효과다. 두 값을 같은 변수나 의미로 사용하지 않는다.
- RAGBuilder가 반환한 best config는 대리 파이프라인의 유망 후보일 뿐이다.
  사용자 파이프라인에서 다시 Eval하기 전에는 개선 성공으로 기록하지 않는다.

자동 처방 우선순위는 D 그룹을 먼저 사람에게 표면화한 뒤, 실행 가능한 큐에서는
`A > C > B` 순서를 사용한다. 같은 그룹의 MVP 점수는 다음과 같다.

```text
priority = (len(finding.affected_probes) × diagnosis_confidence) / prescription_cost
```

빈도는 최소 1로 계산하고, 비용은 런타임 변경 1, 재색인 변경 3을 기본값으로 삼는다.
사후 실측 전의 `impact`는 이 공식에 넣지 않는다.

## 4. 사용자 선택과 종료 분기

최종 목표에는 개선안만 받을지 실제 최적화를 진행할지 사용자가 선택하는 분기가
포함된다. 새 상태 이름을 만들기보다 `schemas.py`의 계약을 재사용한다.

| 상황 | `OptimizeDecision.mode` | 결과 상태 | 의도한 다음 흐름 |
| --- | --- | --- | --- |
| 개선안만 요청 | `propose_only` | `proposed` | Serve 또는 END, 통합 계약 필요 |
| 명시적으로 적용 선택 | `apply_optimize` | 적용 전 `proposed`, 적용 후 `applied` | Index/Eval 검증 |
| 임계값 이미 통과 | `use_current` | `already_optimal` | Serve |
| 실행 가능한 처방 없음 | `use_current` | `skipped` | Serve |
| 데이터·평가셋 문제 | `manual_required` | `manual_required` | 사람 조치 안내 후 Serve/END |
| 실행 실패 | 해당 요청 모드 | `failed` | `status/error` 기록, 통합 오류 처리 |

- 사용자 선택 입력을 어느 UI/API/state 필드에서 받을지는 아직 정해지지 않았다. 입력이
  없는데 승인을 받은 것처럼 자동 적용하지 말고, 계약을 먼저 확정한다.
- 현재 `planner.py`는 actionable finding이 있으면 `apply_optimize`와
  `requires_user_confirmation=False`를 반환한다. `propose_only`는 schema 뼈대만 있으므로
  사용자 선택 분기가 구현됐다고 간주하지 않는다.
- `manual_required`와 actionable finding이 동시에 있을 때 자동 처방을 계속할지 여부도
  미합의다. 임의로 한쪽을 버리지 말고 decision/report에 둘 다 보존한다.
- 현재 그래프는 `Eval -> Optimize -> Index`로 고정돼 있다. 위 표의 Serve/END 분기는
  `graph.py` 수정 금지 계약 때문에 Optimize 단독 작업으로 완성할 수 없다.

## 5. 라벨과 처방 규칙

`rules.py`의 `LABEL_TO_PRESCRIPTIONS`를 라벨·처방의 단일 진실 원천으로 사용한다.
AGENTS/README에 별도 처방 테이블을 복제하지 않는다.

- `status="ready"`이고 처방 목록이 있는 라벨만 planner의 자동 처방 후보가 될 수 있다.
  `ready`는 pipeline-ready라는 뜻이 아니며, optimizer의 concrete 값 변환·capability·
  constraint와 downstream 소비 계약까지 통과해야 실제 적용할 수 있다.
- `draft`, `unassigned`, `manual` 라벨은 optimizer나 adapter에 보내지 않는다.
- D 그룹의 `corpus_gap`, `corpus_gap_partial_hop`, `bad_gold_answer`는 config로 해결하지
  않는다. 문서 추가, probe/ground-truth 검수 같은 사람 조치를 리포트한다.
- `rules.py`는 선언적 데이터 테이블이다. 실행, 우선순위 계산, adapter 호출 로직을
  넣지 않는다.
- 처방은 가벼운 순서의 리스트를 유지하고 각 항목의 `id`, `patch`, `reindex`, `cost`,
  `applies_when`, target metric과 trade-off를 일관되게 기록한다.
- `applies_when`은 Eval의 `Finding.metadata`에 실제 존재하는 신호와 대조해야 한다.
  신호 계약이 없는 처방을 추측으로 선택하지 않는다.
- 새 라벨은 진단 신호 이름으로 짓는다. `overchunking`처럼 처방 방법을 원인인 것처럼
  확정하지 말고, Eval이 고유하고 저비용인 신호를 생산할 수 있을 때만 `ready`로
  승격한다.

## 6. config와 재색인 계약

- planner/optimizer/adapter 내부에서는 `retriever.top_k`, `chunker.chunk_size` 같은
  canonical path를 사용한다. 실제 flat `state.index_config` 변환은
  `config_mapper.py`를 통해서만 수행한다.
- `rules.py`의 `"increase"`, `"decrease"`, `"upgrade"` 같은 symbolic patch를 mapper에
  직접 넘기지 않는다. optimizer가 현재값을 기준으로 concrete하고 bounded한 후보값으로
  해석한 뒤 constraint를 검증해야 한다. 문자열을 그대로 state에 저장하면 Index에서
  숫자 연산이 깨진다.
- mapper는 지원하지 않는 key를 억지로 state에 추가하지 않고 `ignored_keys`와 warning에
  남긴다. 적용 전후는 반드시 `ConfigDiff`로 보존한다.
- optimizer가 capability와 constraint를 검증한 값만 mapper에 전달한다. 현재 주요 안전
  범위는 `top_k 1..20`, `chunk_size 200..1500`, `chunk_overlap 0..300`이며 overlap은
  현재 chunk size의 40%를 넘지 않는다.
- 임베딩 모델, chunk size/overlap, chunking strategy 변경은 `reindex=True`다. 반드시
  Index를 경유해 재임베딩·재저장한 뒤 Eval한다.
- top-k, hybrid, reranker, context ordering 같은 런타임 변경도 실제 검색/생성 소비자가
  변경값을 읽는 경로가 있어야 검증할 수 있다.
- 현재 `agents/index/agent.py`가 실제로 읽는 값은 `chunk_size`와 `chunk_overlap`뿐이다.
  `embedding_model`, `use_hybrid`, `top_k`, reranker 계열은 현재 파이프라인 전체에서
  적용되지 않거나 별도 요청값으로 처리된다. downstream 소비가 확인되지 않은 patch는
  `applied` 또는 개선 완료로 기록하지 않는다.
- 현재 flat `use_hybrid=True` patch를 mapper에 직접 넣으면 canonical 변환 과정에서
  의도와 반대로 `False`가 될 수 있다. ready 처방별 mapper 계약 테스트가 생기기 전에는
  이 직접 경로를 사용하지 말고 `retriever.search_type="hybrid"` 같은 검증된 canonical
  표현을 사용한다.
- 임베딩 모델·벡터 차원 변경은 Index 팀의 `qdrant_store.py` 계약과 컬렉션 재생성을
  수반한다. Optimize에서 모델을 직접 로드하거나 `VECTOR_DIM`을 임의 변경하지 않는다.

## 7. 파일별 책임

| 파일 | 책임 | 넣지 말아야 할 것 |
| --- | --- | --- |
| `agent.py` | planner → optimizer → mapper → history → reporter 연결, state 갱신 | 진단 규칙, backend 세부 구현 |
| `planner.py` | finding 분류, `applies_when`, 우선순위, 단일 후보/request/decision 생성 | state 직접 수정, config 적용 |
| `rules.py` | 라벨별 선언적 처방 데이터와 조회 함수 | 실행·탐색 로직 |
| `schemas.py` | Optimize 내부 dataclass와 Literal 계약 | 실행 로직, 다른 Optimize 모듈 import |
| `optimizer.py` | capability/constraint, backend 선택, 공통 결과 정규화와 안전 fallback | state 직접 수정 |
| `config_mapper.py` | canonical config를 state 형식으로 변환·적용하고 diff 생성 | 처방 선택, backend 정책 |
| `history.py` | 전후 config/metric, blacklist, 유지·rollback 판정 | 새 진단 수행 |
| `reporter.py` | 적용/제안/수동/실패별 사용자 설명 | config 변경 |
| `adapters/*.py` | 외부 backend payload 변환, 실행 경계, 결과 정규화 | 라벨 재판정, 전역 정책 |

`internal_adapter.py`는 향후 자체 search-space 탐색 backend 자리이며 현재 의도적으로
비어 있다. 첫 후보 선택이나 단순 patch fallback을 이 파일에 임시 구현하지 않는다.
검증된 규칙 후보를 순서대로 선택하는 공통 fallback은 `optimizer.py` 책임이다.

RAGBuilder 연동에서는 다음 경계를 지킨다.

- 제한된 hyperparameter search에만 사용한다. AutoRAG adapter는 아직 없으며 복합 원인
  전체 탐색은 향후 과제다.
- 현재 adapter는 단일 optimized stage만 허용한다. retrieval과 chunking을 한 request에
  섞지 않는다.
- 현재 planner는 `request.search_space`와 RAGBuilder `input_source`를 채우지 않는다.
  adapter는 빈 search space를 `skipped`로 처리하므로, backend 연결 시 optimizer가
  처방 후보를 concrete search space로 만들고 필요한 입력을 명시적으로 전달해야 한다.
- 현재 보장된 objective mapping은 context recall/precision 중심이다. generation 또는
  faithfulness 최적화가 지원된다고 가정하지 않는다.
- 외부 라이브러리의 전체 Optuna study/trial history를 얻는다고 가정하거나 직접
  `optuna` 의존성을 추가하지 않는다. 실제 반환 계약으로 검증된 정보만 사용한다.
- 기본 테스트는 mock 또는 주입 client로 결정적이어야 한다. 실제 corpus/API key를 쓰는
  검증은 선택적 통합 테스트와 격리 환경에서 수행한다.

## 8. 오류·이력·리포트

- `applied`는 config가 실제로 변경되고 다음 Eval 검증을 기다리는 상태이지, 성능 개선이
  증명됐다는 뜻이 아니다. `OptimizationResult.improved`는 평가 전 `None`을 유지한다.
- 값이 모두 필터링됨, 미지원 backend/path/objective, 외부 adapter 실패를 구분 가능한
  status/error/warning으로 남긴다. 빈 `internal_adapter`로 조용히 fallback하지 않는다.
- history에는 request/trial ID, iteration, label, prescription ID, backend, before/after
  config와 metric, target metrics, rollback 이유를 저장한다.
- blacklist는 `(failure_label, prescription_id)` 단위로 만든다. 같은 실패 처방이 반복돼
  그래프가 무한 순환하지 않도록 한다.
- rollback 비교는 반올림하지 않은 원값을 사용한다. 사용자 표시 점수만 반올림하고
  참고용 추정치임을 밝힌다.
- reporter는 원인, 제안/적용 처방, 실제 변경과 무시된 key, 예상 trade-off, 수동 조치,
  다음 단계를 숨김없이 보여준다.

## 9. 테스트와 완료 기준

현재 Optimize 단위 테스트는 아래 명령으로 실행한다.

```powershell
python -m unittest tests.test_config_mapper tests.test_optimizer tests.test_ragbuilder_adapter -v
python -m compileall -q agents/optimize tests/test_config_mapper.py tests/test_optimizer.py tests/test_ragbuilder_adapter.py
```

2026-07-13 현재 위 명령의 27개 테스트가 통과한다. 이는 mapper/optimizer 정책/RAGBuilder
adapter 하위 기능만 검증하며, Optimize 노드나 전체 파이프라인 성공을 뜻하지 않는다.

변경 시 최소한 다음을 지킨다.

- 수정한 모듈의 기존 테스트와 새 회귀 테스트를 함께 실행한다.
- planner에는 report 없음, threshold 통과, manual/actionable 혼합, 빈 후보, blacklist,
  복수 finding, `applies_when` 테스트를 추가한다.
- optimizer에는 backend dispatch, constraint로 빈 search space가 된 경우, 외부 실패와
  검증된 후보 fallback, 결과 정규화 테스트를 추가한다.
- agent에는 모든 분기에서 동일 state 반환, 오류 변환, 라벨 전환 시 iteration 단일 증가,
  같은 라벨 후보에서 카운터 유지, 실제 diff만 적용, 제안/수동/현재 유지 분기 테스트를
  추가한다.
- history/reporter에는 rollback·blacklist와 applied/proposed/manual/failed별 테스트를
  추가한다.
- 마지막으로 `eval -> optimize -> index -> eval` 반복 통합 테스트를 추가하되 외부 API와
  로컬 서비스에 의존하지 않게 만든다.
- 새 의존성은 루트 `requirements.txt`에 Optimize 담당 주석과 함께 기록한다. RAGBuilder
  호환성 실험용 의존성은 기본 런타임과 섞지 말고 별도 requirements/Docker 경계를
  유지한다.

## 10. 현재 진행상황과 협업 주의사항

이 절은 2026-07-13의 작업 스냅샷이다. 작업을 시작할 때 Git과 코드를 다시 확인한다.

- 구현됨/초안: `rules.py`, `planner.py`, Optimize schema, config mapper, optimizer의
  capability/constraint 정책, RAGBuilder mock·client·native 실행 경계.
- 미구현: `agent.py`, `history.py`, `reporter.py`, optimizer backend dispatch.
- `internal_adapter.py`는 의도적으로 비어 있다.
- 현재 규칙은 25개 라벨 중 `ready` 5, `draft` 17, `manual` 3이다.
- 현재 브랜치 `feature/optimize_Sungwoo`에는 RAGBuilder/mapper/optimizer/schema와 테스트의
  큰 미커밋 변경이 있다. 사용자 작업으로 간주하고 보존한다.
- 협업 브랜치 `origin/feature/optimize`에는 현재 작업트리에 없는 manual finding 보존 및
  history 유지/rollback 구현 커밋(`fb7015d`, `a52c5ba`)이 있다. 같은 기능을 새로 쓰기
  전에 해당 diff를 비교하고 팀과 통합 방향을 확인한다.
- GitHub `main`에서는 configurable index와 swappable chunk strategy 변경이 revert됐다.
  feature branch에 코드가 있다는 이유로 main/현재 pipeline이 해당 config를 소비한다고
  가정하지 않는다.
- 현재 열린 PR·issue와 GitHub Actions workflow가 없으므로 브랜치 코드와 로컬 테스트가
  진행상황 판단의 핵심 근거다.

## 11. 사람 합의 없이 결정하지 말 것

- 사용자 선택 입력을 저장할 state/API/UI 계약과 `proposed` 이후 Serve 대 END route
- optimize decision/result/report/history/blacklist/score history의 정식 state 필드
- manual과 actionable finding이 동시에 있을 때의 실행 정책
- target profile별 threshold·가중치와 전역 rollback 하한선
- `internal`을 자체 탐색 backend로 예약할지, 규칙 기반 직접 경로의 별도 이름
- Index/Serve의 top-k, reranker, hybrid, query rewrite, MMR, context compression,
  chunking strategy 지원 범위
- AutoRAG 도입 시점과 복합 원인을 AutoRAG로 보내는 정량 기준
- RAGBuilder native 결과의 전체 trial/Optuna DB 추출 및 strict hybrid 보장 방식
