# Optimize 파트 진행상황

작성 기준: `feature/optimize_Sungwoo` 브랜치의 현재 워킹트리  
작성일: 2026-07-10

## 1. 한눈에 보는 현재 상태

Optimize 파트는 **규칙·데이터 모델·계획 수립·설정 변환·RAGBuilder 어댑터의 하위 부품은 상당 부분 작성됐지만, 이를 실제 파이프라인으로 묶는 진입점은 아직 미구현** 상태다.

- 구현됨: 진단 라벨 규칙, planner의 우선순위/요청 생성, Optimize 내부 schema, config 적용과 diff, RAGBuilder mock·client 주입 경로
- 구현 중: optimizer의 capability/constraint 정책, RAGBuilder mapping 책임 분리, 관련 단위 테스트
- 미구현: `agent.py`, history, reporter, optimizer backend dispatch
- 의도적 유보: `internal_adapter.py`는 향후 자체 search space 최적화 backend를 구현할 때 사용하기 위해 빈 파일로 유지
- 통합 차단 요인: `agent.py`의 `run()`이 `pass`여서 Optimize 노드가 `None`을 반환함
- 규칙 현황: 총 25개 라벨 중 `ready` 5개, `draft` 17개, `manual` 3개
- 테스트 현황: Optimize 관련 단위 테스트 11개 통과. 실제 LangGraph 왕복, planner, history/rollback, 실제 RAGBuilder 실행 테스트는 없음

상태 표시는 다음 기준을 사용한다.

| 상태 | 의미 |
| --- | --- |
| 완료 | 현재 책임 범위의 핵심 코드와 최소 테스트가 있음 |
| 진행 중 | 일부 로직 또는 테스트는 있으나 호출 흐름/검증이 덜 연결됨 |
| 초안 | 구조와 정책은 있으나 실제 적용 범위가 제한적임 |
| 미구현 | 빈 파일 또는 `pass` 상태 |

## 2. 현재 브랜치의 작업 중 변경사항

아래 파일은 HEAD 커밋 이후 워킹트리에서 수정 중이다. 따라서 이 문서의 평가는 커밋된 코드뿐 아니라 현재 로컬 변경분까지 포함한다.

| 파일 | Git 상태 | 변경 방향 |
| --- | --- | --- |
| `adapters/ragbuilder_adapter.py` | 수정 | prescription ID 재해석을 제거하고 request의 search space/patch를 소비하도록 책임 축소 |
| `config_mapper.py` | 수정 | 처방 후보 생성기에서 canonical config를 실제 `index_config`에 적용하는 mapper로 재구성 |
| `optimizer.py` | 수정 | capability와 constraint 정책 추가 |
| `schemas.py` | 수정 | 폐기된 guardrail 필드 제거 |
| `tests/test_config_mapper.py` | 수정 | 새 mapper 책임에 맞게 테스트 변경 |
| `tests/test_ragbuilder_adapter.py` | 수정 | request가 search space를 명시하도록 테스트 변경 |
| `tests/test_optimizer.py` | 신규, 미추적 | optimizer 정책 테스트 추가 |

## 3. 실행 흐름 기준 진행상황

목표 흐름은 다음과 같다.

```text
Eval report
  -> planner: 라벨 분류, 우선순위 결정, OptimizationRequest 생성
  -> optimizer: 요청 검증, 후보 제한, backend 선택 및 결과 정규화
       -> 기본 결정 경로: 검증된 처방 후보를 순서대로 선택
       -> ragbuilder adapter: RAGBuilder search space 최적화 실행
       -> internal adapter(향후): 자체 search space 최적화 실행
  -> config_mapper: state.index_config 적용 및 ConfigDiff 생성
  -> history: 시도 기록, blacklist/rollback 정보 유지
  -> reporter: 사용자용 OptimizationReport 생성
  -> agent: state 갱신, iteration 증가, index 재진입
```

현재 연결 상태는 아래와 같다.

```text
Eval report
  -> planner (구현)
  -X-> optimizer dispatch (미구현)
  -X-> agent 통합 (미구현)
  -X-> history/reporter (미구현)
```

즉, 개별 부품의 단위 테스트 성공과 별개로 Optimize 노드 전체는 아직 실행할 수 없다.

## 4. 파일별 역할, 진행상황, 해야 할 일

### `agent.py` — 미구현, 최우선

역할:

- Optimize 노드의 유일한 진입점
- `state.report`, `state.index_config`, `state.iteration`을 읽음
- planner, optimizer, mapper, history, reporter를 순서대로 호출
- `state.index_config`, `state.iteration`, `state.status`, `state.error`, `state.current_agent`를 갱신하고 반드시 `state`를 반환

현재 상태:

- `run()`이 `pass`만 포함한다.
- LangGraph가 Optimize 경로에 진입하면 `None`이 반환되어 상태 계약이 깨진다.
- 파일 상단의 읽기/쓰기 상태 필드 docstring도 없다.

해야 할 일:

1. AGENTS.md 계약에 맞는 `run(state) -> state` 구현
2. `state.current_agent = "optimize"` 설정과 예외의 `status/error` 변환
3. `planner.plan()` 결과의 `use_current`, `manual_required`, `apply_optimize` 분기 처리
4. optimizer 결과를 mapper로 적용하고 `iteration` 증가
5. history와 reporter 결과를 state에 보관할 방식 확정
6. Optimize 노드 단위 테스트와 `eval -> optimize -> index` 통합 테스트 추가

### `planner.py` — 초안 구현

역할:

- Eval의 `Finding.label`을 manual/actionable로 분리
- 그룹과 점수로 finding 우선순위 결정
- blacklist를 반영해 처방 후보 하나를 선택
- `OptimizationRequest`와 `OptimizeDecision` 생성

현재 상태:

- report 없음, threshold 통과, 수동 조치, 자동 최적화 분기가 구현돼 있다.
- 현재 정렬은 자동 적용 라벨에 대해 A > C > B이며, D는 manual로 별도 처리한다.
- 점수는 `affected_probes 수 × diagnosis_confidence ÷ cost`를 사용한다.
- rules의 patch를 `PrescriptionCandidate`와 request로 변환한다.
- 전용 테스트가 없다.

해야 할 일:

1. `applies_when`과 `finding.metadata`를 실제로 대조해 후보를 선택
2. candidate별 priority와 trade-off를 채우기
3. optimizer가 소비할 search space 생성 책임을 명확히 하고 현재 단일 patch 후보와 정합성 맞추기
4. 빈 candidate, 모든 후보 blacklist, 복수 finding, manual+actionable 혼합 테스트 추가
5. README의 우선순위 설명(D > A > C > B)과 실제 구현(A > C > B, D 별도)을 일치시키기

### `rules.py` — 규칙 테이블 초안

역할:

- failure label별 그룹, 상태, 진단 신뢰도, target metric, 처방 목록 정의
- 자동 적용 가능 여부와 manual 여부 조회 함수 제공

현재 상태:

- 라벨 25개와 처방 45개가 정의돼 있다.
- `ready`: 5개
  - `retrieval_low_rank`
  - `retrieval_lexical_mismatch`
  - `retrieval_semantic_mismatch`
  - `retrieval_missing_gold`
  - `too_long_context`
- `draft`: 17개
- `manual`: `corpus_gap`, `corpus_gap_partial_hop`, `bad_gold_answer` 3개
- Index/Eval과 합의되지 않은 config와 signal에 TODO/BLOCKER가 남아 있다.

해야 할 일:

1. Eval이 실제 생성할 label과 `metadata` signal 계약 확정
2. Index/Serve가 실제 소비하지 않는 `reranker`, `query_rewrite`, `mmr`, `context_compression`, `chunking_strategy` 등의 처방을 계속 draft로 둘지 결정
3. 각 ready 처방의 patch가 현재 mapper에서 실제 적용되는지 계약 테스트 작성
4. 처방별 cost, trade-off, 재색인 필요 여부를 일관된 schema로 정리
5. draft 17개의 승격 조건과 담당 연동 파일 명시

### `schemas.py` — 진행 중

역할:

- planner, optimizer, adapters, mapper, history, reporter 사이의 공통 데이터 계약 정의
- request/result/patch/diff/decision/report/history 모델 제공

현재 상태:

- 11개 dataclass가 정의돼 있어 모듈 간 데이터 구조는 충분히 구체적이다.
- 현재 워킹트리에서 폐기된 guardrail 필드를 제거하는 중이다.
- 실제 `AgentDoctorState`에는 optimize decision/report/history를 담는 정식 필드가 없다.
- `OptimizationRequest.optimizer`의 기본값이 현재 `"internal"`이라, 비워 둘 `internal_adapter.py`와 MVP의 단순 후보 선택 경로가 이름상 충돌한다.

해야 할 일:

1. 모든 소비 파일이 guardrail 제거 후 schema와 일치하는지 확인
2. 단순 후보 선택 backend를 `direct`/`rules` 등으로 분리하고 `internal`은 향후 자체 탐색 adapter용으로 예약할지 결정
3. `OptimizerBackend`에 선언된 `autorag`를 미지원 상태로 명시하거나 구현 전까지 제한
4. optimize 결과를 state의 정식 필드로 추가할지, 임시 동적 속성을 사용할지 결정
5. dataclass 생성/직렬화 및 status 조합에 대한 계약 테스트 추가

### `optimizer.py` — 진행 중

역할:

- `OptimizationRequest` 검증
- capability와 안전 범위 적용
- 현재 MVP의 검증된 처방 후보 순차 선택
- 구현된 backend만 선택하고 adapter 실행
- backend 결과를 공통 `OptimizationResult`로 정규화
- adapter 실패 시 기본 결정 경로로 fallback할지 결정

현재 상태:

- capability 병합/확인, constraint 병합, 후보 필터링만 구현돼 있다.
- `chunk_overlap <= chunk_size × 0.4` 제약이 있다.
- backend 선택, request 실행, 결과 정규화는 아직 없다.
- 현재 request의 기본 backend 이름 `internal`을 그대로 사용하면 빈 `internal_adapter.py`로 연결될 수 있으므로 backend 명칭 계약을 먼저 정리해야 한다.
- 관련 단위 테스트 4개가 통과한다.

해야 할 일:

1. MVP 기본 결정 경로의 backend 이름을 `direct`/`rules` 등으로 정할지 결정하고 `internal`과 구분
2. `run()` 또는 동등한 공개 진입 함수 정의
3. 현재 MVP의 기본 결정 경로와 `ragbuilder` backend dispatch 구현
4. request의 search space 전체에 capability/constraint 필터 적용
5. 빈 search space, 미지원 backend, adapter 실패 시 검증된 처방 후보로 fallback하는 정책 구현
6. `RAGBuilderResult`를 `OptimizationResult`로 변환
7. 후보가 전부 제거된 경우 명확한 skipped/failed 결과 반환

### `config_mapper.py` — 진행 중, 단위 기능 구현

역할:

- optimizer의 canonical 경로를 실제 `state.index_config` flat key로 변환
- patch 또는 best config 적용
- 적용 전후 `ConfigDiff` 생성
- 미지원 key를 무시하고 warning으로 기록

현재 상태:

- `top_k`, `use_hybrid`, `chunk_size`, `chunk_overlap`, `embedding_model` 적용을 지원한다.
- nested/flat alias 조회, mutate 여부, non-index target 무시가 구현돼 있다.
- 관련 단위 테스트 5개가 통과한다.
- 현재 워킹트리에서 기존의 prescription-to-search-space 책임을 optimizer/planner 쪽으로 이동시키는 중이다.

해야 할 일:

1. optimizer의 constraint 검증을 거치지 않은 값을 mapper가 받을 때의 방어 정책 결정
2. rules의 ready 처방 중 mapper가 무시하는 key를 명시적으로 검증
3. `reindex_required`와 warning이 diff/history/reporter까지 전달되도록 연결
4. 실제 `state.index_config`에 적용하는 agent 통합 테스트 추가
5. mapper가 값 clamp를 담당한다는 기존 README 설명을 현재 책임과 맞게 수정

### `adapters/internal_adapter.py` — 의도적 유보

역할:

- `ragbuilder_adapter.py`와 동일한 backend 경계 역할
- optimizer가 전달한 search space를 받아 AgentDoctor 자체 최적화 알고리즘 실행
- trial별 config와 metric을 비교해 best config/result 반환
- 외부 라이브러리에 의존하지 않는 자체 탐색 전략의 구현 위치

현재 상태:

- 빈 파일이며 현재 단계에서는 의도된 상태다.
- 단순히 첫 처방을 고르거나 patch를 적용하는 fallback 파일로 사용하지 않는다.
- 자체 search space 탐색 알고리즘의 요구사항과 평가 방식이 확정될 때까지 구현을 보류한다.

향후 구현 조건:

1. 입력 search space와 trial budget 계약 확정
2. 각 trial을 평가할 metric/evaluator 연결 방식 확정
3. grid/random/Bayesian 등 자체 탐색 전략 선택
4. RAGBuilder 결과와 비교 가능한 공통 adapter result 계약 확정
5. 위 조건이 정해진 뒤 구현과 전용 테스트 추가

기존에 이 파일의 책임으로 적었던 항목은 다음과 같이 분배한다.

| 기존 책임 | 담당 파일 | 이유 |
| --- | --- | --- |
| `applies_when`을 finding signal과 대조 | `planner.py` | 진단 결과를 해석해 유효한 처방 후보를 만드는 계획 단계의 책임 |
| capability/constraint 적용 | `optimizer.py` | 어느 backend를 쓰든 동일하게 지켜야 하는 실행 전 정책 |
| 검증된 후보의 순차 선택과 fallback | `optimizer.py` | backend 공통 orchestration이며 자체 탐색 알고리즘과 구분해야 함 |
| `propose_only`, `manual_required`, `use_current` 결정 | `planner.py`와 `agent.py` | planner가 결정을 만들고 agent가 실제 pipeline 분기를 반영 |
| `OptimizationResult` 정규화 | `optimizer.py` | RAGBuilder와 향후 internal backend 결과를 같은 계약으로 맞추는 계층 |
| `needs_reindex` 계산 | `optimizer.py` | 선택된 patch/result의 실행 의미를 공통 결과에 기록 |
| patch를 `state.index_config`에 반영 | `config_mapper.py` | canonical config와 실제 state key 사이의 변환 책임 |
| status/error/iteration 등 state 갱신 | `agent.py` | LangGraph 상태 계약을 책임지는 최상위 진입점 |

### `adapters/ragbuilder_adapter.py` — 진행 중

역할:

- `OptimizationRequest`를 RAGBuilder payload로 변환
- 주입 client, 설치된 RAGBuilder, 명시적 mock 중 하나로 실행
- 외부 결과를 `RAGBuilderResult`로 정규화

현재 상태:

- mapping, payload 생성, client 호출, 실제 패키지 import 경계, mock, 결과 정규화가 구현돼 있다.
- request search space 우선 사용과 candidate patch fallback을 지원한다.
- mock과 client 주입 경로 단위 테스트 2개가 통과한다.
- 실제 RAGBuilder 패키지와 실제 데이터셋을 이용한 실행은 검증되지 않았다.
- capability/constraint 검증은 adapter에서 제거되고 optimizer 책임으로 이동 중이다.

해야 할 일:

1. optimizer가 검증된 search space만 전달하도록 호출 계약 연결
2. 실제 설치 버전의 RAGBuilder API와 payload 호환성 확인
3. 입력 source/eval dataset 누락 시 사전 검증과 오류 메시지 정리
4. 외부 실패를 optimizer의 검증된 처방 후보 선택 경로로 fallback할지 정책 확정
5. 실제 실행을 선택적 integration test로 추가

### `history.py` — 미구현

역할:

- 최적화 시도 전후 config/metric 기록
- 실패한 `(label, prescription_id)` blacklist 관리
- 성능 하락 시 rollback 판단과 이전 config 복원 지원

현재 상태:

- 빈 파일이다.
- schema에 `OptimizationHistoryItem`만 정의돼 있다.

해야 할 일:

1. history 저장 위치를 `AgentDoctorState` 정식 필드로 둘지 결정
2. append/query/blacklist 생성 함수 구현
3. baseline 대비 metric 개선·악화 판정 규칙 확정
4. rollback과 iteration 간 관계 구현
5. 같은 처방 무한 반복 방지 테스트 추가

### `reporter.py` — 미구현

역할:

- decision/result/diff/history를 사용자용 `OptimizationReport`로 변환
- 적용 내용, 무시된 설정, trade-off, 수동 조치, 다음 단계를 설명

현재 상태:

- 빈 파일이다.
- schema에 `OptimizationReport`만 정의돼 있다.

해야 할 일:

1. applied/proposed/manual/failed별 report builder 구현
2. ignored key와 warning을 사용자에게 숨기지 않고 요약
3. ready가 아닌 finding에 대한 수동 안내 문구 정의
4. CLI/API가 report를 노출할 state 필드 또는 출력 계약 연결

### `README.md` — 문서 초안, 현재 코드와 일부 불일치

역할:

- Optimize 목표, 전체 흐름, 모듈 책임, MVP 구현 순서 설명

현재 상태:

- 설계 의도와 파일별 계획은 잘 정리돼 있다.
- 현재 코드보다 오래된 내용이 남아 있다.
  - `agent.py` fallback 흐름은 아직 없음
  - internal adapter를 단순한 안전 patch 생성기로 설명하지만, 실제 의도는 향후 자체 search space 최적화 backend임
  - mapper가 숫자를 clamp한다고 설명하지만 현재는 optimizer가 필터 정책을 가짐
  - RAGBuilder를 skeleton/mock 수준으로 설명하지만 현재 adapter는 실제 실행 경계까지 구현됨
  - 문제 우선순위 설명과 planner 구현이 다름

해야 할 일:

1. 이 진행상황 문서를 기준으로 구현 완료 후 README 갱신
2. 목표 설계와 현재 구현 상태를 별도 절로 분리
3. 상태 필드와 graph 분기 방식이 확정되면 흐름도 수정

### `CONTEXT.md` — 설계 배경 문서

역할:

- 방식 2(진단 기반 단일 처방·순차 검증), 우선순위 공식, rollback 원칙, 팀 간 미합의 사항 보존

현재 상태:

- 구현 방향을 이해하는 데 유용하지만 미해결 합의와 과거 계획이 섞여 있다.
- 코드의 source of truth로 사용하기에는 현재 구현과 차이가 있다.

해야 할 일:

1. 미해결 항목을 issue/checklist 형태로 분리
2. 확정된 결정은 README 또는 코드 docstring으로 승격
3. 폐기된 guardrail 개념과 현재 metric 하한선 정책을 정리

### `__init__.py`, `adapters/__init__.py` — 빈 패키지 표시 파일

역할:

- Python package 경계를 표시

현재 상태:

- 빈 파일이며 현재로서는 문제없다.

해야 할 일:

- 외부 공개 API가 확정된 뒤에만 필요한 symbol을 제한적으로 export한다.

## 5. 관련 테스트 파일

### `tests/test_config_mapper.py` — 통과

- canonical/flat alias 조회
- canonical path의 실제 index key 변환
- patch 적용과 diff
- 비변경 실행
- 지원하지 않는 target 무시

추가 필요:

- constraint 경계값과 mapper 연동
- reindex flag 전달
- ready 처방별 실제 적용 가능성

### `tests/test_optimizer.py` — 통과, 미추적 파일

- constraint 적용
- chunk overlap 비율 제한
- 보수적 capability 기본값
- flat alias constraint 병합

추가 필요:

- backend dispatch와 fallback
- request 전체 search space 필터
- `OptimizationResult` 생성

### `tests/test_ragbuilder_adapter.py` — 통과

- 명시적 mock 실행과 결과 정규화
- 주입 client 결과의 canonical path 복원

추가 필요:

- 빈 search space
- candidate patch fallback
- 외부 예외/failed result
- 실제 RAGBuilder 선택적 통합 테스트

### 현재 없는 테스트

- planner 단위 테스트
- internal adapter 단위 테스트는 자체 최적화 알고리즘 구현 시 추가
- history/rollback 테스트
- reporter 테스트
- Optimize agent 단위 테스트
- `eval -> optimize -> index -> eval` 반복 통합 테스트

## 6. 권장 구현 순서

1. `planner.py`에서 `applies_when`과 finding signal 대조 및 후보 순서 확정
2. `optimizer.py`의 기본 후보 선택, RAGBuilder dispatch, 공통 결과 정규화 구현
3. `agent.py`에서 planner → optimizer → mapper를 연결하고 항상 state 반환
4. state에 decision/result/report/history 필드를 정식 추가할지 결정
5. `history.py`의 append와 blacklist부터 구현
6. `reporter.py`의 applied/manual/failed 요약 구현
7. Optimize 노드와 반복 그래프 통합 테스트 추가
8. 실제 RAGBuilder 호환성 검증
9. `rules.py`의 draft 라벨을 Eval/Index 계약에 맞춰 단계적으로 ready로 승격
10. 자체 search space 최적화 요구사항이 확정되면 `internal_adapter.py` 구현

## 7. 검증 기록

Optimize 관련 테스트 실행:

```powershell
python -m unittest discover -s tests -p test_config_mapper.py -v
python -m unittest discover -s tests -p test_optimizer.py -v
python -m unittest discover -s tests -p test_ragbuilder_adapter.py -v
```

결과: 총 11개 테스트 통과.

전체 테스트 discovery는 Optimize와 무관한 환경 문제로 실패했다.

- `pytest` 패키지 미설치
- Notion OAuth 환경변수 미설정
- Windows cp949 콘솔에서 Index 로그의 em dash 출력 오류
- `tests/test_pipeline.py`가 Ingest 실패 시 import 도중 종료

따라서 현재 확인 가능한 결론은 **Optimize 하위 모듈 3개 영역의 단위 테스트는 통과하지만, 전체 pipeline 성공을 증명하지는 않는다**는 것이다.
