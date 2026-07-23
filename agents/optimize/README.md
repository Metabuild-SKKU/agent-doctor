# Optimize Module Plan

## 목적

Optimize 모듈은 Eval 단계에서 생성된 진단 결과를 읽고, RAG pipeline의 어떤 설정을 조정할지 결정한다. 단순히 점수가 낮은 지표에 대해 여러 설정을 한꺼번에 바꾸는 방식이 아니라, failure label을 기반으로 문제 원인을 분류하고, 그 원인에 맞는 처방과 search space를 만든다.

AgentDoctor의 차별점은 최적값을 brute force로 찾는 것보다, 왜 현재 RAG가 실패했는지와 어떤 처방을 먼저 시도해야 하는지를 설명하는 데 있다.

## 전체 흐름

```text
Eval report 읽기
  -> failure label 확인
  -> optimize 필요 여부 판단
  -> 처방 가능/불가능 분류
  -> search space 생성
  -> 사용자 선택 분기
       - 제안만 생성
       - optimize 적용
       - 이미 최적 상태
       - 수동 개입 필요
  -> optimizer 실행
  -> config patch 또는 best config 생성
  -> state.index_config 병합
  -> optimization history 기록
  -> 사용자용 처방 리포트 생성
  -> 재색인 또는 종료 흐름으로 전달
```

## 사용자 선택 분기

Optimize는 항상 설정을 즉시 바꾸지 않는다. 진단 결과에 따라 다음 상태 중 하나를 남긴다.

- `proposed`: 개선안을 제안하고 종료한다.
- `applied`: 처방을 실제 config에 적용한다.
- `already_optimal`: 현재 설정이 충분히 좋아서 변경하지 않는다.
- `manual_required`: corpus gap, bad gold answer처럼 config 수정으로 해결할 수 없는 문제를 안내한다.
- `failed`: optimize 요청 생성 또는 실행 중 실패했다.

`graph.py`에는 optimize 이후 조건부 branch가 구현돼 있다(`route_after_optimize`).

```text
applied / rolled_back -> index (재색인 필요)
그 외(proposed/already_optimal/manual_required/failed/verified) -> serve
```

또한 반복 예산(`max_iterations`)을 다 썼어도 직전에 적용한 처방이 아직 판정(유지/롤백)
안 됐으면, `route_after_eval`이 마지막으로 한 번 더 optimize로 보내 판정 기회를 준다
(그렇지 않으면 나빠진 처방이 판정 없이 그대로 서빙될 수 있었다).

pending 이력에는 적용 직전의 `before_index_key`와 `before_eval_key`도 저장한다.
Index/Eval 캐시는 이 두 키를 롤백 기준본으로 고정해, 후보를 여러 번 시험하더라도
최대 두 슬롯 안에서 기준본과 현재 후보만 유지한다. 처방이 실패하면 config뿐 아니라
해당 인덱스와 진단 결과를 캐시에서 바로 복원할 수 있다.

## 문제 그룹

Eval에서 넘어온 finding label은 우선 다음 그룹으로 분류한다.

| 그룹 | 의미 | 예시 label | 기본 대응 |
| --- | --- | --- | --- |
| A | 검색 실패 | `retrieval_low_rank`, `retrieval_lexical_mismatch`, `retrieval_missing_gold` | retrieval/index config 조정 |
| B | 생성 실패 | `generation_hallucination`, `generation_partial_answer` | generation config 또는 prompt 계열 조정 |
| C | context 구조 문제 | `too_long_context`, `lost_in_the_middle`, `context_noise_interference` | top-k, reorder, compression 계열 조정 |
| D | 데이터/평가셋 문제 | `corpus_gap`, `bad_gold_answer`, `corpus_gap_partial_hop` | 자동 optimize 대신 수동 조치 안내 |

여러 문제가 동시에 있을 때의 우선순위는 MVP 기준으로 다음 순서를 따른다.

```text
데이터 문제(D) -> 검색 문제(A) -> context 구조 문제(C) -> 생성 문제(B)
```

D 그룹은 pipeline config로 해결할 수 없는 경우가 많으므로, 실제 patch보다 리포트와 사용자 안내가 중요하다.

## Search Space 설계

Planner는 최상위 failure label을 기준으로 조정 가능한 config 후보를 만든다.

예시:

| Label | 처방 | Search space |
| --- | --- | --- |
| `retrieval_low_rank` | reranker 도입 | `use_reranker`, `reranker_model`, `rerank_top_n` |
| `retrieval_lexical_mismatch` | hybrid search 적용 | `use_hybrid`, `retriever_type`, `bm25_top_k`, `rrf_weight` |
| `retrieval_missing_gold` | top-k 또는 chunk 조정 | `top_k`, `chunk_size`, `chunk_overlap` |
| `retrieval_incomplete_enumeration` | 동적 검색량 확대 | `top_k`, `adaptive_retrieval` |
| `too_long_context` | context 축소 | `top_k`, `chunk_size`, `context_compression` |
| `lost_in_the_middle` | context 재정렬 | `context_reorder` |
| `generation_hallucination` | grounding 강화 | `generation_config.grounding_strict`, `require_citation` |
| `corpus_gap` | 문서 추가 요청 | config 없음 |
| `bad_gold_answer` | 평가셋 검수 요청 | config 없음 |

현재 Index Agent가 실제로 사용하는 값은 `chunk_size`, `chunk_overlap`, `embedding_model`, `use_hybrid` 중심이다. 그 외 필드는 향후 Index/Serve/Eval 연동을 위한 확장 필드로 둔다.

## RAGBuilder / AutoRAG 연동 방향

AgentDoctor는 RAGBuilder나 AutoRAG의 내부 최적화 로직을 직접 구현하지 않고 wrapper로 감싼다.

```text
OptimizationRequest
  -> RAGOptimizerWrapper
  -> RAGBuilderAdapter 또는 AutoRAGAdapter
  -> best_config
  -> OptimizationResult
```

- RAGBuilder: 원인이 비교적 명확하고 hyperparameter 튜닝으로 해결 가능한 경우 사용한다.
- AutoRAG: 원인이 복합적이거나 pipeline 전체 탐색이 필요한 경우 추후 사용한다.
- MVP에서는 `ragbuilder_adapter.py`를 skeleton 또는 mock 형태로 둔다.

## History / Rollback

같은 처방을 반복하지 않고, 처방 실패 시 되돌리기 위해 optimization history를 기록한다.

`AgentDoctorState`에 정식 필드로 구현돼 있다(`core/state.py`). 원소 타입은
`OptimizationHistoryItem`(schemas.py)이며, core가 optimize를 import하지 않도록
state 쪽 타입 선언은 느슨하게 둔다.

```python
state.optimization_history: list        # OptimizationHistoryItem 목록
state.blacklist: set                    # {(label, prescription_id), ...}
state.optimization_report: object | None  # 이번 방문의 사용자 리포트(OptimizationReport)
```

판정이 다음 Eval 재측정 후에야 가능하다는 시점 문제 때문에, 이력은 2단계로 기록된다.
처방을 적용한 시점엔 before만 채운 pending 항목을 만들고(`history.create_pending_item`),
다음 Optimize 방문에서 재측정된 리포트로 확정한다(`history.finalize_item`).

실제 rollback 기준(`history.judge`, `history.check_floor`에 구현됨):

- ragas 하한선(FLOORS)을 위반하면 무조건 rollback한다. 현재 더미값이며,
  실제 파이프라인을 돌려본 실험 결과로 나중에 튜닝하기로 팀 결정.
- Eval이 계산한 종합 점수(`overall_score`)가 이전보다 오르지 않으면 rollback한다.
- 실패한 `(label, prescription_id)` 조합은 blacklist로 기록해 재시도를 막는다.

## 파일별 구현 계획

### `agent.py`

Optimize 모듈의 최상위 진입점이다.

- `state.report` 확인
- planner가 있으면 planner를 사용해 request 생성
- planner가 없으면 report 기반 fallback request 생성
- optimizer 실행
- config mapper로 `state.index_config` 병합
- history 기록
- reporter로 사용자용 리포트 생성
- `state.optimize_decision` 또는 동등한 상태값 저장

### `planner.py`

A 담당 파일이다.

- Eval report의 finding label을 읽는다.
- optimize 필요 여부를 판단한다.
- 처방 후보와 search space를 만든다.
- `OptimizationRequest`를 생성한다.

### `rules.py`

A/B 공동 구현 파일이다.

- label별 처방 후보를 정의한다.
- label의 그룹(A/B/C/D), 비용, target metric, trade-off를 관리한다.
- MVP에서는 dict 기반 규칙으로 시작하고, 추후 점수 기반 priority 계산을 추가한다.

### `schemas.py`

A 담당 파일이다.

Optimize 내부 데이터 모델을 정의한다.

- `OptimizationRequest`
- `OptimizationResult`
- `PrescriptionCandidate`
- `ConfigPatch`
- `MetricGoal`
- `Guardrail`
- `OptimizationHistoryItem`

B 파트에서는 A schema가 없을 때를 대비해 `optimizer.py`에 임시 fallback dataclass를 둘 수 있다.

### `optimizer.py`

B 담당 파일이다.

- `OptimizationRequest`를 검증한다.
- adapter를 선택한다.
- internal, ragbuilder, 추후 autorag adapter로 실행을 위임한다.
- 결과를 `OptimizationResult` 형태로 정리한다.
- unknown adapter는 internal로 fallback한다.

### `adapters/internal_adapter.py`

B 담당 파일이다.

- 외부 도구 없이 label 기반 config patch를 생성한다.
- MVP에서는 안전한 patch만 만든다.
- D 그룹처럼 자동 처리할 수 없는 문제는 `manual_required`로 반환한다.

### `adapters/ragbuilder_adapter.py`

B 담당 파일이다.

- AgentDoctor의 추상 request를 RAGBuilder가 이해할 search space로 변환한다.
- 실제 RAGBuilder 연동 전까지는 skeleton 또는 mock result를 반환한다.
- TODO로 실제 연동 지점을 명확히 남긴다.

### `config_mapper.py`

B 담당 파일이다.

- `OptimizationResult.best_config` 또는 `config_patch`를 `state.index_config`에 병합한다.
- 허용된 config key만 적용한다.
- 숫자형 config는 안전 범위로 clamp한다.
- 알 수 없는 key는 warning으로 남긴다.

### `history.py`

A 담당 또는 B 연결 파일이다.

- optimization trial 기록을 추가한다.
- before/after config를 저장한다.
- rollback에 필요한 정보를 유지한다.
- 실패한 처방 blacklist를 관리할 수 있도록 확장한다.

### `reporter.py`

B 담당 파일이다.

사용자에게 보여줄 처방 리포트를 생성한다.

리포트에는 다음을 포함한다.

- 문제 원인
- 적용 또는 제안된 처방
- 변경 config
- 예상 trade-off
- 수동 개입 필요 여부
- 다음 단계 안내

## MVP 구현 순서

1. ~~파일 구조와 README 정리~~ 완료
2. ~~B fallback dataclass 정의~~ 완료
3. internal adapter의 label 기반 patch 구현 — 미착수(`internal_adapter.py`는 향후 자체
   탐색 backend용으로 의도적 유보, PROGRESS.md 참고)
4. ~~config mapper 구현~~ 완료
5. ~~reporter 구현~~ 완료
6. ~~agent 흐름 연결~~ 완료(Phase 1 순방향 + Phase 2 방문 간 판정·롤백)
7. ~~history 기록~~ 완료(동적 속성이 아니라 state 정식 필드로 구현)
8. ~~단위 테스트 추가~~ 완료(59개 통과, PROGRESS.md 7-1절 참고)
9. ~~graph 조건부 branch~~ 완료(`route_after_optimize`, `route_after_eval`)

남은 항목은 3번(자체 탐색 backend)뿐이며, 이건 요구사항이 확정될 때까지 의도적으로
미루기로 한 것이다.

