# Optimizer 구현·연동 계획

> 상태: 1차 구현 완료, 팀 연동 합의 항목 추적 중
> 작성 목적: `optimizer.py`와 `internal_adapter.py` 담당 작업, 다른 Optimize/Eval/Index
> 모듈과의 연결 계약, 테스트 및 완료 기준을 한곳에 기록한다.
> 기준 설계: [`PARAM_TUNING_PROPOSAL.md`](PARAM_TUNING_PROPOSAL.md)

---

## 1. 이번 구현에서 확정한 방향

Planner는 파라미터 종류와 관계없이 canonical 경로 하나와 후보 리스트를 같은 형식으로
전달한다.

```python
{"retriever.top_k": [8, 12, 16]}
{"chunker.chunk_size": [400, 600, 800]}
{"chunker.chunk_overlap": [40, 80, 120]}
```

별도의 "계산형/sweep형/사전선별형" 분류를 만들지 않는다. Optimizer가 canonical
파라미터 경로를 보고 해당 파라미터의 비용 특성에 맞는 실행 정책을 선택한다.

| canonical 파라미터 | 후보 처리 정책 | 비싼 Index/Eval 횟수 |
| --- | --- | --- |
| `retriever.top_k` | 후보가 하나면 1회 검증, 여러 개면 후보별 실제 Eval 후 best 선택 | 후보 수만큼 |
| `chunker.chunk_size` | 실제 청커 사전검증으로 예상 best 하나 선택 | 1회 |
| `chunker.chunk_overlap` | 경계 복구·중복량 사전검증으로 예상 best 하나 선택 | 1회 |
| 단일 토글 파라미터 | 안전검사 후 전달된 값 하나 적용 | 1회 |

정책 선택 예시는 다음과 같다.

```python
PARAMETER_POLICIES = {
    "retriever.top_k": evaluate_candidates_with_pipeline,
    "chunker.chunk_size": select_chunk_size_with_prescreener,
    "chunker.chunk_overlap": select_overlap_with_prescreener,
}
```

핵심 원칙은 다음과 같다.

1. `state.iteration`은 후보 trial 수가 아니라 **연속된 라벨 처리 횟수**를 센다.
2. 후보 진행률은 별도 state 카운터를 만들지 않고 `trial_results`에서 계산한다.
3. top-k 후보 하나가 실패해도 처방 전체를 즉시 실패·blacklist 처리하지 않는다.
4. top-k 후보를 모두 평가한 뒤 최고 점수를 최초 baseline과 비교해 한 번만 결론 낸다.
5. chunk 후보는 DB를 후보마다 재구성하지 않고, 싼 사전검증 후 예상 best 하나만 적용한다.
6. 후보 적용 전에는 capability, state mapping, 안전범위를 모두 통과해야 한다.
7. 실제 유지·롤백 판정은 Eval이 만든 `overall_score`를 기준으로 한다.
8. `graph.py`는 수정하지 않는다.

---

## 2. 용어와 반복 구조

### 2.1 Study

한 진단 라벨의 한 파라미터 축을 최적화하는 전체 작업이다.

권장 식별 단위는 다음과 같다.

```text
(failure_label, prescription_id, canonical_parameter_path)
```

예:

```text
(retrieval_low_rank, increase_top_k, retriever.top_k)
```

### 2.2 Trial

Study 안에서 후보값 하나를 실제 또는 사전 평가한 결과다.

```text
Study: retrieval_low_rank / retriever.top_k
  Trial 1: top_k=8  -> overall_score=0.68
  Trial 2: top_k=12 -> overall_score=0.82
  Trial 3: top_k=16 -> overall_score=0.78
  결론: baseline 0.70보다 높은 top_k=12 채택
```

### 2.3 카운터 계약

| 값 | 의미 | 증가 시점 |
| --- | --- | --- |
| `state.iteration` | 연속된 라벨 처리 횟수 | 이전 처리 라벨이 없거나 새 라벨과 다를 때 1회 |
| `state.max_iterations` | 처리할 수 있는 최대 라벨 단계 수 | 사용자/파이프라인 설정 |

별도 `trial_count` 상태 필드는 추가하지 않는다. 현재 후보 진행률은
`len(trial_results)`와 미평가 candidate fingerprint로 계산한다. Planner가 sweep 후보를
2~3개로 제한하고, 기존 `request.max_trials`는 후보 목록을 넘어가지 않게 하는 adapter
안전 상한으로만 유지한다.

Iteration 증가는 다음 규칙 하나로 처리한다.

```python
if previous_label is None or previous_label != request.failure_label:
    state.iteration += 1
```

같은 라벨의 top-k 후보 `[8, 12, 16]`을 차례로 평가해도 iteration은 증가하지 않는다.
`A -> B -> A`처럼 이전 단계와 다른 라벨로 전환되면 각 전환에서 증가한다.

이미 시작한 동일 라벨의 후보 평가는 `state.iteration >= state.max_iterations`여도
마무리할 수 있어야 한다. 반복 예산 소진은 **다른 라벨로 넘어가는 것만 차단**한다.
동일 라벨이 후보 소진 뒤 다시 무한히 열리는 문제는 추가 카운터가 아니라 study 완료
표시와 blacklist로 막는다.

---

## 3. 전체 실행 흐름

### 3.1 새 study 시작

```text
Eval Finding
  -> Planner가 라벨 하나와 후보 목록 생성
  -> active study 생성
  -> 이전 처리 라벨이 없거나 다를 때만 state.iteration += 1
  -> Optimizer가 첫 후보 또는 사전선별 best 반환
```

### 3.2 Active study 재개

```text
Index/Eval 결과
  -> 직전에 적용한 후보의 점수 기록
  -> 같은 study의 Internal Adapter 재호출
     -> 다음 후보가 있으면 needs_evaluation
     -> 후보가 끝났으면 completed + best_config
```

Active study가 있는 동안에는 Planner를 다시 호출해 다른 라벨을 선택하지 않는다.

### 3.3 Study 종료

```text
best 후보 점수 > 최초 baseline 점수
  -> best 후보 채택

모든 후보 점수 <= 최초 baseline 점수
  -> 최초 config 유지/복원
  -> study 최종 실패
  -> 이때만 처방 전체 blacklist 여부 판정
```

중간 후보가 `pass_threshold=True`를 달성하면 나머지 후보를 생략하고 조기 종료할 수 있다.

---

## 4. 성우 담당 파일

### 4.1 `optimizer.py` — 필수 수정

`optimizer.py`는 상태를 직접 수정하지 않는 실행 조율·안전검사·결과 정규화 계층으로
유지한다.

#### 해야 할 일

- [x] `internal` backend의 `backend_not_implemented` 조기 반환 제거
- [x] `BACKEND_SUPPORTED_PATHS["internal"]` 정의
- [x] Planner 후보를 canonical 단일 축 search space로 정규화
- [x] capability와 `STATE_MAPPABLE_PATHS` 검사
- [x] 안전범위로 후보 필터링
- [x] 필터 결과가 비면 원인을 구분한 `skipped` 결과 반환
- [x] 선택된 candidate와 필터링된 search space로 새 request를 만들어 adapter 호출
- [x] Internal Adapter의 반환 타입과 config를 재검증
- [x] `needs_evaluation`을 적용 가능한 `OptimizationResult`로 정규화
- [x] `completed`를 best 적용 또는 baseline 유지 결과로 정규화
- [x] `failed`/`skipped`를 공통 오류 계약으로 정규화
- [x] trial/study 상태를 다음 방문에 전달할 수 있도록 metadata에 보존
- [x] 입력 request, candidate, search space를 변경하지 않음

#### Internal 결과 정규화 규칙

| `InternalAdapterResult` | `OptimizationResult` |
| --- | --- |
| `needs_evaluation + next_config` | `status="proposed"`, 다음 후보 `ConfigPatch`, `improved=None` |
| `completed + best_config`가 현재값과 다름 | `status="proposed"`, best `ConfigPatch` |
| `completed + best_config`가 현재값과 같음 | config 변경 없는 완료/유지 결과 |
| `skipped` | `status="skipped"`, 구체적인 `error_code` |
| `failed` 또는 잘못된 반환 타입 | `status="failed"`, error와 adapter 정보 |

반환 metadata에는 최소한 다음을 보존한다.

```python
{
    "active_study": True,
    "parameter_path": "retriever.top_k",
    "study_baseline_config": {...},
    "trial_results": [...],
    "objective_metric": "overall_score",
    "direction": "maximize",
    "best_score": 0.82,
    "budget_used": 2,
    "max_trials": 3,
    "stop_reason": "candidate_requires_evaluation | pass_threshold_reached | search_budget_finished",
    "filtered_search_space": {...},
}
```

입력에 들어온 문서 본문이나 `gold_spans` 전체는 결과 metadata/history에 복사하지 않는다.
결과에는 후보별 점수와 선택 근거 요약만 남긴다.

### 4.2 `adapters/internal_adapter.py` — 필수 검토, 필요한 부분만 수정

현재 구현은 이미 다음 기능을 제공한다.

- evaluator가 없으면 이전 trial 관측을 읽고 다음 미평가 후보 반환
- evaluator가 있으면 같은 호출에서 후보를 순서대로 평가
- 고정 study baseline 지원
- 후보 fingerprint와 중복 평가 방지
- `max_trials`, `min_delta`, objective direction 처리
- `pass_threshold` 조기 종료
- baseline과 후보 중 best 선택

따라서 범용 상태 기계를 다시 작성하거나 별도의 파라미터 유형을 추가하지 않는다.
Optimizer가 canonical 경로별 정책을 선택하고, Internal Adapter에는 평가 방법만 전달한다.

#### Top-k 경로

`retriever.top_k`는 후보마다 실제 Index/Eval이 필요한 경로다.

```text
evaluator 없음
  -> 이전 trial_results 읽기
  -> 다음 미평가 후보를 next_config로 반환
  -> 전부 평가됐으면 best_config 반환
```

#### Chunk 경로

`chunker.chunk_size`와 `chunker.chunk_overlap`은 싼 사전검증으로 같은 Optimize 방문에서
예상 best를 고르는 경로다.

```text
싼 evaluator 있음
  -> 모든 안전한 후보를 in-process 평가
  -> 예상 best_config 하나 반환
  -> Index/Eval에는 선택값 하나만 전달
```

#### 해야 할 일

- [x] Optimizer가 전달한 canonical 파라미터 경로와 실행 정책 조합 검증
- [x] candidate 실패가 study 전체를 중단하지 않도록 격리
- [x] 모든 후보를 고정된 최초 baseline과 비교
- [x] 후보 순서와 무관하게 점수 기준 best 선택
- [x] `needs_evaluation`과 `completed`를 명확히 구분
- [x] search space 밖 관측과 반환값 거부
- [x] `max_trials`를 넘은 관측이 best에 영향을 주지 않게 유지
- [x] trial 결과에 config, score, status, fingerprint, metric 기록
- [x] 반환값과 입력값의 불변성 유지

`internal_adapter.py` 자체는 위 기능을 이미 제공하고 있어 이번 1차 구현에서는 수정하지
않았다. 대신 `optimizer.py`의 dispatch·재검증과 Agent 재개 흐름을 연결하고 기존 Adapter
테스트를 그대로 통과시키는 방식으로 재사용했다.

### 4.3 `adapters/chunk_prescreener.py` — 신규 파일 권장

chunk 전용 기하 사전검증을 범용 adapter에 섞지 않기 위해 별도 helper로 둔다.

#### 입력

- 현재 chunk config
- Planner가 만든 안전범위 전 후보
- 영향받는 문서의 본문 또는 실제 청커를 호출할 수 있는 preview 입력
- `gold_spans`
- 진단 라벨

#### `chunk_size` 평가 기준

`chunk_overlap`은 현재값으로 고정한다.

1. `span_fit_rate`: gold span 길이가 후보 chunk size 이내인 비율
2. `full_span_containment`: 실제 dry-run 청크 하나가 정답 전체를 포함하는 비율
3. `context_waste`: 정답 포함 청크 길이에서 정답 길이를 뺀 값
4. `chunk_count`: 예상 청크 수

`boundary_cut_rate`는 chunk size의 주 목적 지표가 아니라 고정 overlap 조건에서의
보조 안전검사로만 사용한다.

#### `chunk_overlap` 평가 기준

`chunk_size`는 현재값으로 고정한다.

1. `boundary_crossing_rate`: gold span이 청크 경계를 넘는 비율
2. `boundary_recovery_rate`: overlap 적용 후 어느 청크가 정답 전체를 포함하는 비율
3. `unrecovered_cut_rate`: overlap 후에도 정답 전체를 담지 못하는 비율
4. `duplication_ratio`: overlap으로 중복 처리되는 문자 비율

#### 후보 선택 방식

문서에는 가중치 합산 공식이 정의돼 있지 않으므로 임의의 점수 가중치를 만들지 않는다.
설명 가능한 우선순위 비교 또는 문서에서 합의한 결정론적 기준을 사용한다.

현재 Index 구현에서 `chunk_size`와 `chunk_overlap`은 문자열 위치를 사용하는 문자 단위다.
사전검증도 `gold_spans`의 문자 절대좌표와 같은 단위를 사용한다.

가능하면 실제 Index 청커의 pure dry-run 함수를 재사용한다. Optimize 안에 별도 청커를
복제하지 않는다.

### 4.4 테스트 — 성우 담당 변경과 함께 작성

- [x] `tests/test_optimizer.py`
- [x] `tests/test_internal_adapter.py` 기존 회귀 테스트 통과
- [x] 신규 `tests/test_chunk_prescreener.py`
- [x] `tests/test_planner.py` 후보 전달 계약 테스트
- [x] `tests/test_optimize_agent.py` top-k sweep·iteration 통합 테스트
- [ ] 실제 Index/Eval 전체 그래프 통합 테스트

---

## 5. 다른 파일과 연결해야 할 작업

아래 작업은 성우 담당 코드가 실제 파이프라인에서 호출·재개되기 위해 필요하다.

### 5.1 `planner.py`

현재 Planner는 기본적으로 `optimizer="rules"`, `max_trials=1` 요청을 만든다. 다음 계약을
추가해야 한다.

- [x] `confirmed=True` finding만 비싼 trial 대상으로 선택
- [x] 모든 파라미터를 `{canonical_path: [후보값...]}` 형식으로 전달
- [ ] top-k/chunk 후보값 자체를 계산하는 공식은 Eval/Planner 담당자와 합의
- [x] 준비된 chunk 후보 리스트와 사전검증 근거 전달
- [x] 내부 평가가 필요한 파라미터 요청에 `optimizer="internal"` 설정
- [x] `max_trials`를 전달받은 후보 수와 맞춤
- [x] active study가 있으면 Agent가 Planner 호출 전에 재개

현재 후보 입력은 정식 schema를 늘리지 않고 다음 임시 계약을 사용한다.

```python
Finding.metadata["parameter_candidates"] = {
    "retriever.top_k": [8, 12, 16],
}
```

후보 산출 공식을 합의하면 이 metadata를 생산하는 쪽만 교체하면 되고, Optimizer 이하의
sweep 구현은 그대로 유지할 수 있다.

권장 요청 예시는 다음과 같다.

```python
OptimizationRequest(
    request_id="stable-study-request-id",
    iteration=state.iteration,
    baseline_config=dict(state.index_config),
    failure_label="retrieval_low_rank",
    search_space={"retriever.top_k": [8, 12, 16]},
    optimizer="internal",
    max_trials=3,
    metadata={
        "study_baseline_config": dict(state.index_config),
        "trial_results": [],
    },
)
```

### 5.2 `agent.py`

현재는 config patch를 적용할 때마다 `state.iteration`을 증가시키고, 반복 예산이 소진되면
신규 후보 적용을 중단한다. 이전 처리 라벨과 현재 request 라벨을 비교하는 구조로 다음
연결이 필요하다.

- [x] active study가 있으면 Planner보다 먼저 study 재개
- [x] 이전 라벨이 없거나 `request.failure_label`과 다를 때만 `state.iteration += 1`
- [x] 같은 라벨의 다음 후보 적용 시 iteration을 올리지 않음
- [x] `state.iteration >= max_iterations`여도 동일 라벨의 남은 후보는 허용
- [x] 예산 소진 시 다른 라벨로의 전환만 차단
- [x] 직전 후보의 Eval 결과를 `InternalTrialResult` 관측으로 변환
- [x] Optimizer가 반환한 study metadata를 다음 방문까지 보존
- [x] `needs_evaluation` 후보를 config에 적용하고 Index/Eval로 보냄
- [x] study 완료 후 best를 확정하고 active study 종료
- [x] Eval의 후보별 `state.iteration` 증가 제거

권장 분기 순서는 다음과 같다.

```python
if active_study:
    observe_previous_trial()
    resume_internal_study()
else:
    request = planner.plan(...)
    previous_label = find_previous_label(...)
    label_changed = previous_label is None or previous_label != request.failure_label
    if label_changed and state.iteration >= state.max_iterations:
        stop_without_new_study()
    start_study(request)
    if label_changed:
        state.iteration += 1
```

### 5.3 `history.py`

후보별 기록과 라벨 study 최종 판정을 분리한다.

#### 후보 평가 중

```text
top_k=8  -> 0.68 기록, 다음 후보 진행
top_k=12 -> 0.82 기록, 다음 후보 진행
top_k=16 -> 0.78 기록
```

이 단계에서는 후보 하나가 baseline보다 나쁘더라도 처방 전체를 blacklist 처리하지 않는다.

#### 모든 후보 평가 후

```text
최초 baseline=0.70, best=0.82
  -> best 후보 채택

최초 baseline=0.70, 모든 후보 <= 0.70
  -> baseline 유지/복원
  -> study 최종 실패
  -> 처방 blacklist 여부 판정
```

필요 작업:

- [x] study별 trial observation 누적
- [x] 고정 baseline report/config 보존
- [x] 개별 trial status와 study status 분리
- [x] candidate 하나의 실패로 전체 처방을 blacklist하지 않음
- [x] 모든 후보 종료 후 한 번만 최종 keep/rollback 판정
- [x] candidate fingerprint 단위 재시도 방지
- [x] 최종 실패한 study만 처방 전체 blacklist 처리

### 5.4 기존 schema와 state 재사용

현재는 한 번에 라벨 하나와 후보 2~3개만 처리하므로 `OptimizationStudy` 같은 새 schema나
`state.active_optimization` 필드를 추가하지 않는다. 기존 모델의 역할만 조합한다.

| 기존 모델 | 사용하는 정보 |
| --- | --- |
| `OptimizationRequest` | 라벨, 최초 baseline config, 후보 목록, `max_trials` |
| `InternalTrialResult` | 후보 하나의 config, score, metrics, status |
| `InternalAdapterResult` | 다음 후보, 현재 best, 누적 trial 결과 |
| `OptimizationHistoryItem` | Index/Eval 방문 사이에 진행 상황 보존 |
| `state.optimization_history` | 진행 중 항목과 완료 이력 저장 |

진행 중 sweep은 최신 `OptimizationHistoryItem.metadata`에 최소 정보만 기록한다.

```python
item.metadata.update({
    "active_study": True,
    "study_request": request,
    "current_candidate": {"retriever.top_k": 8},
    "study_baseline_config": {"top_k": 5},
    "trial_results": [...],
})
```

다음 Optimize 방문은 `optimization_history`에서 `active_study=True`이면서 `pending=True`인
최신 항목을 찾아 같은 request를 재구성한다. 모든 후보 평가가 끝나면 두 값을 False로
바꾸고 best config와 최종 점수를 기록한다.

이 방식은 새 상태 필드 없이 현재 요구사항을 충족한다. 다음 조건이 생길 때만 별도
`OptimizationStudy` schema를 다시 검토한다.

- 동시에 여러 study를 실행해야 함
- 일시정지·재개·외부 저장을 정식 지원해야 함
- 여러 파라미터 축과 복잡한 탐색 전략을 하나의 study에서 관리해야 함
- metadata 키가 많아져 타입 오류를 막기 어려워짐

### 5.5 Eval 연결

- [ ] top-k 계산/후보 생성용 `gold_rank`, `recall_at_k` 제공
- [x] chunk 사전검증용 기존 Probe `gold_spans` 전달
- [x] trial 비교용 동일 계약의 `overall_score` 제공
- [x] `pass_threshold` 제공

`gold_spans`가 없는 chunk 후보는 자동으로 비싼 재색인을 실행하지 않는다.

- span 있음: chunk 경로의 사전검증 정책으로 자동 적용 가능
- ground-truth 길이만 있음: 약한 근거이므로 `propose_only` 권장
- 근거 없음: `skipped` 또는 `manual_required`

### 5.6 Index 연결

Optimizer가 변경하는 값에 따라 Index가 수행해야 하는 작업과 비용이 다르다. Optimize는
변경 종류와 재색인 필요 여부를 정확히 표시하고, 실제 청크·임베딩·Qdrant 처리 방식은
Index가 담당한다.

#### 5.6.1 Top-k만 변경하는 경우

```text
top_k=5 -> top_k=12
```

top-k는 저장된 청크의 내용이나 임베딩 벡터를 바꾸지 않는다. 검색 시 기존 벡터 결과에서
상위 몇 개를 가져올지만 달라진다.

```text
기존: 상위 5개 청크 반환
변경: 상위 12개 청크 반환
```

따라서 top-k-only 변경에는 문서 재청킹과 재임베딩이 필요하지 않다. 현재 그래프 구조상
Index 노드는 거칠 수 있지만, 기존 청크와 임베딩을 재사용해야 한다. Qdrant payload의
top-k 메타데이터를 갱신하는 가벼운 작업은 발생할 수 있다.

통합 테스트에서는 다음을 확인한다.

- [x] top-k만 바꿨을 때 기존 청크 수와 텍스트가 유지됨
- [x] Index의 임베딩 함수 `embed()` 호출 횟수가 0임
- [x] 변경된 top-k가 다음 Eval 검색에 실제 사용됨
- [x] 후보별 Eval 점수가 같은 `overall_score` 계약으로 기록됨

Index의 `_index_signature()`에는 top-k가 포함되지 않고, 재사용 청크의 retrieval metadata만
새 config로 갱신한다. 기존 `test_same_signature_reuses_embeddings`와
`test_reused_chunks_refresh_retrieval_metadata`가 이 동작을 검증한다. 다만 Eval은 아직
`retrieval_temp.build_eval_index()`로 청크를 매번 다시 적재하므로, Index 임베딩 재사용과
Eval 임시 인덱스 재적재 비용은 구분해야 한다.

#### 5.6.2 Chunk size/overlap을 변경하는 경우

```text
chunk_size=512 -> chunk_size=600
```

chunk size나 overlap이 바뀌면 문서를 자르는 위치와 청크 텍스트가 달라진다.

```text
기존 512:
  chunk 1 = 0..512
  chunk 2 = 462..974

변경 600:
  chunk 1 = 0..600
  chunk 2 = 550..1150
```

청크 텍스트가 달라지므로 다음 작업이 필요하다.

```text
문서 재청킹
  -> 새 청크 임베딩
  -> Qdrant에 새 인덱스 저장
  -> Eval 재실행
```

이 작업은 top-k 변경보다 비싸므로 Planner가 후보 `[400, 600, 800]`을 보냈다고 후보마다
전체 Index/Eval을 실행하지 않는다. Optimizer가 먼저 싼 사전검증으로 예상 best 하나를
고르고, 그 값만 Index에 전달한다.

#### 5.6.3 DB 없는 chunk preview API

Preview API는 임베딩과 Qdrant 저장 없이 실제 청커가 만들 청크의 원문 좌표만 계산하는
함수다.

```python
preview_chunk_spans(
    document,
    chunk_size=600,
    chunk_overlap=50,
    strategy="markdown_recursive",
)
```

예상 반환값:

```python
[
    {"start": 0, "end": 580},
    {"start": 530, "end": 1100},
]
```

Optimizer의 chunk prescreener는 이 좌표와 Eval의 `gold_spans`를 비교해 정답 포함률과
경계 절단 여부를 계산한다. 이 단계에서는 임베딩 모델, Qdrant, 검색, LLM을 호출하지
않는다.

Optimize 안에 별도 청커를 복제하면 preview와 실제 Index의 청크 경계가 달라질 수 있다.
따라서 현재 Index의 실제 청킹 함수를 재사용하는 얇은 공개 wrapper를 Index 담당자와
협의한다.

- [ ] Index가 DB 없는 `preview_chunk_spans()` 공개 API를 제공할지 결정
- [ ] preview가 실제 Index 청커와 동일한 경계를 반환하는지 단위 테스트
- [ ] preview 호출 중 `embed()`와 Qdrant 쓰기가 발생하지 않는지 테스트

#### 5.6.4 Chunk 후보가 실패했을 때 rollback

현재 baseline이 `chunk_size=512`이고 사전선별 후보 `600`을 적용했다고 가정한다.

방법 1은 후보가 실패했을 때 기존 설정의 인덱스를 다시 만드는 것이다.

```text
512 인덱스 존재
  -> 600 인덱스 생성
  -> Eval 결과가 baseline보다 나쁨
  -> 600 인덱스 제거
  -> 512 설정으로 재청킹·재임베딩·인덱스 재생성
```

구현은 단순하지만 실패 시 비싼 DB 구성이 한 번 더 발생한다.

방법 2는 baseline 인덱스를 지우지 않고 후보용 버전 컬렉션을 별도로 만드는 것이다.

```text
baseline_collection_512 보존
  -> candidate_collection_600 생성
  -> 성공: candidate 컬렉션을 최종 사용
  -> 실패: baseline 컬렉션으로 즉시 복귀
```

롤백은 빠르지만 컬렉션 버전, alias, 저장 공간을 관리해야 한다. MVP에서 단순 재구성을
사용할지, 비용을 줄이기 위해 버전 컬렉션을 사용할지는 Index 담당자와 결정한다.

- [ ] MVP rollback을 baseline 재구성으로 할지 버전 컬렉션 전환으로 할지 결정
- [ ] 실패한 후보의 임시 인덱스 정리 책임과 시점 결정
- [ ] 버전 컬렉션을 사용하면 baseline/candidate alias 전환 테스트

#### 5.6.5 “후보 검증 1회”와 “DB 재구성 총 1회” 구분

chunk 후보 세 개 중 하나만 선택하면 후보 검증용 Index/Eval은 한 번이다.

```text
[400, 600, 800] 사전검증
  -> 600 선택
  -> 600으로 Index/Eval 1회
```

600이 성공하면 새 DB 구성도 한 번으로 끝난다. 하지만 600이 실패하고 기존 512 인덱스를
보존하지 않았다면 baseline 복원을 위해 DB를 다시 구성해야 한다.

```text
600 후보 구성: 1회
600 실패
512 baseline 복원: 추가 1회
```

따라서 현재 단순 rollback 방식의 비용은 다음과 같다.

```text
후보 검증용 Index/Eval: 1회
실패 rollback까지 포함한 DB 재구성: 최대 2회
```

실패 시에도 새 DB 구성을 총 1회로 제한하려면 baseline 인덱스를 보존하는 버전 컬렉션
방식이 필요하다.

#### 5.6.6 담당 경계

Optimizer 담당:

- [x] chunk 후보를 사전검증해 하나만 Index에 전달
- [x] top-k 변경은 `needs_reindex=False`로 표현
- [x] chunk size/overlap 변경은 `needs_reindex=True`로 표현
- [x] rollback에 필요한 baseline config와 선택 근거를 결과/history에 보존

Index 담당 또는 공동 협의:

- [x] top-k-only 변경에서 실제 Index 임베딩을 재사용
- [ ] DB 없는 실제 청커 preview 제공
- [ ] chunk 후보 인덱스 생성·삭제·복원
- [ ] baseline 재구성과 버전 컬렉션 중 rollback 전략 결정

### 5.7 `graph.py`

수정하지 않는다. 적용 또는 다음 후보가 있으면 기존 `status="applied"` 신호로 Index/Eval을
거친다. 라우팅 요구사항은 `agent.py`의 active study 처리와 상태 계약 안에서 해결한다.

---

## 6. Top-k 실행 상세

### 6.1 근거값 하나

```text
Eval: 놓친 gold_rank=11
Planner: [11]
Optimizer: 1..20 안전검사
Index/Eval: top_k=11 한 번 검증
History: baseline보다 좋아졌으면 유지, 아니면 복원
```

### 6.2 대표 후보 여러 개

```text
Planner: [8, 12, 16]

iteration 1 / retrieval_low_rank study 시작
  trial 1: top_k=8  -> Eval 0.68
  trial 2: top_k=12 -> Eval 0.82
  trial 3: top_k=16 -> Eval 0.78

최초 baseline=0.70
best=top_k 12, 0.82
최종 top_k=12
state.iteration은 전체 과정에서 1회만 증가
```

top-k는 recall만으로 best를 고르지 않는다. top-k가 커지면 recall은 좋아져도 context
노이즈와 생성 품질이 나빠질 수 있으므로 최종 비교는 `overall_score`를 사용한다.

Eval이 LLM/RAGAS FULL 모드라면 후보별 생성 비용이 생긴다. 기본 후보는 2~3개로 제한하고,
필요하면 한 번의 wide retrieval prefix 지표로 후보를 먼저 줄이는 최적화를 후속으로 둔다.

---

## 7. Chunk 실행 상세

```text
Planner: chunk_size 후보 [400, 600, 800]
Optimizer/Internal:
  실제 청커 dry-run으로 세 후보 사전검증
  예상 best=600 선택
Index/Eval:
  600만 실제 재청킹·임베딩·평가
History:
  baseline보다 좋아졌으면 유지
  아니면 baseline config로 롤백
```

사전검증 결과는 실제 검색·생성 품질을 보장하는 최종 점수가 아니라 proxy다. 사용자
리포트에는 "최적값"이 아니라 "기하 사전검증에서 선택한 예상 best"라고 표시한다.

---

## 8. 구현 순서

### Phase 1 — 계약 확정

- [x] 기존 `request_id`를 sweep이 끝날 때까지 동일하게 유지
- [x] 진행 상태는 최신 `OptimizationHistoryItem.metadata`에 저장
- [x] Planner request metadata 필드 — 현재는 임시 확장 계약
- [x] Optimizer result metadata 필드 — 현재는 임시 확장 계약
- [x] trial observation은 Optimize Agent가 현재 Eval report로 생성
- [x] 모든 후보가 baseline 이하일 때만 study 실패와 처방 blacklist

### Phase 2 — 성우 담당 단위 구현

- [x] `optimizer.py` internal dispatch
- [x] Internal 결과 공통 정규화
- [x] `internal_adapter.py` canonical 파라미터 경로별 평가 흐름 검증
- [x] top-k pipeline sweep 단위 테스트
- [x] chunk prescreener와 단위 테스트

이 단계 테스트는 mock request, mock trial observation, mock precheck evaluator를 사용하며
실제 Qdrant, 외부 API, LLM에 의존하지 않는다.

### Phase 3 — 파이프라인 연결

- [x] Planner가 internal request 생성
- [x] Agent가 active study 시작/재개
- [x] History가 trial과 study를 분리해 기록
- [x] Eval 결과를 trial observation으로 전달
- [x] 라벨 비교 기반 iteration과 후보 소진 통합 테스트

### Phase 4 — Chunk 데이터 준비

- [x] 기존 Eval Probe의 `gold_spans`를 Planner가 전달
- [x] 실제 Index 청커의 `_chunk_document()`를 임시 preview로 재사용
- [x] chunk 후보 하나만 ConfigPatch로 반환하는 단위 검증
- [ ] 공개 preview API와 전체 Index/Eval 통합 검증

---

## 9. 테스트 시나리오와 완료 기준

### 9.1 Optimizer

- [x] `internal` 요청이 더 이상 `backend_not_implemented`가 아님
- [x] 지원하지 않는 backend/path/capability는 구체적 사유로 실패 또는 스킵
- [x] top-k 1..20 밖 후보는 제거
- [x] chunk size/overlap 안전범위 적용
- [x] 한 번에 config 축 하나만 허용
- [x] Internal 반환 config가 필터된 후보 밖이면 거부
- [x] `next_config`가 `ConfigPatch`로 변환됨
- [x] 입력 request/search space가 변경되지 않음

### 9.2 Top-k study

- [x] 첫 라벨 시작 시 `state.iteration`이 1 증가
- [x] 같은 라벨 후보 세 개를 평가하는 동안 iteration이 더 증가하지 않음
- [x] 이전 단계와 다른 라벨로 전환할 때만 iteration이 다시 증가
- [x] 각 후보의 Eval 점수가 누적됨
- [x] 첫 후보 실패가 나머지 후보를 차단하지 않음
- [x] 점수 기준으로 best를 선택
- [x] 모든 후보가 baseline 이하이면 baseline 유지
- [x] Internal Adapter가 pass threshold 관측 시 조기 종료
- [x] 후보 목록 또는 `max_trials` 이후 후보는 실행하지 않음
- [x] `max_iterations` 도달 후에도 동일 라벨의 시작된 후보 평가는 완료
- [x] `max_iterations` 도달 후 다른 라벨로 전환하지 않음
- [ ] 실제 graph에서 pass threshold 조기 종료 시 active history 확정

### 9.3 Chunk 사전선별

- [x] 후보 N개가 있어도 실제 Index/Eval 후보는 하나만 반환
- [x] `gold_spans`가 없으면 자동 재색인하지 않음
- [x] chunk size 비교 시 overlap은 고정
- [x] chunk overlap 비교 시 size는 고정
- [x] 실제 청커 경계를 사용
- [x] 문자/토큰 단위를 섞지 않음
- [x] 선택 이유와 후보별 proxy 결과가 리포트용 metadata에 남음

### 9.4 회귀

- [x] 기존 rules backend 동작 유지
- [x] 기존 RAGBuilder fallback 동작 유지
- [x] config mapper 계약 유지
- [x] 모든 Optimize 경로에서 동일 state 반환
- [x] 예외는 state status/error 또는 표준 실패 결과로 변환

### 9.5 검증 명령

```powershell
python -m unittest tests.test_optimizer tests.test_internal_adapter -v
python -m unittest tests.test_config_mapper tests.test_optimize_agent tests.test_eval_iteration -v
python -m compileall -q agents/optimize tests
```

신규 prescreener 테스트가 생기면 첫 명령에 추가한다.

---

## 10. 구현 중 지키지 말아야 할 것

- `graph.py`를 수정해 trial 반복을 해결하지 않는다.
- `optimizer.py`나 adapter가 `AgentDoctorState`를 직접 수정하지 않는다.
- Planner가 넘기지 않은 symbolic patch를 Optimizer가 임의의 숫자로 추측하지 않는다.
- 후보 하나 실패를 study 전체 실패로 확정하지 않는다.
- top-k best를 recall 하나로만 고르지 않는다.
- chunk 후보마다 전체 Index/Eval을 실행하지 않는다.
- Optimize에 실제 청킹 로직을 복제하지 않는다.
- 미평가 후보를 best config로 기록하지 않는다.
- 외부 backend 실패를 검증되지 않은 후보로 조용히 fallback하지 않는다.

---

## 11. 1차 구현 이후 합의가 필요한 항목

1. `Finding.metadata["parameter_candidates"]`를 정식 schema 필드로 승격할지
2. 동일 라벨의 다른 prescription/parameter 축을 같은 iteration으로 계속 볼지
3. 후보 전부 평가 후 best가 현재 후보가 아닐 때 best 적용 뒤 Eval을 한 번 더 할지
4. top-k 후보 기본 개수와 FULL Eval 비용 상한
5. chunk 실패 롤백 시 기존 Qdrant 컬렉션 보존 전략
6. Index의 `_chunk_document()`를 공개 `preview_chunk_spans()`로 감쌀지
7. Eval의 임시 `build_eval_index()` 재적재를 제거하고 Index 검색기를 재사용할 시점
8. graph가 `pass_threshold=True`를 먼저 Serve로 보내는 경우 active history를 어디서 확정할지

권장 기본값은 다음과 같다.

- Study 단위: `(failure_label, prescription_id, canonical_parameter_path)`
- Top-k 후보: 기본 2~3개, 최대 `remaining_trial_budget`
- Trial 비교 baseline: study 시작 전 config와 report로 고정
- 최종 objective: Eval의 `overall_score`, 높을수록 좋음
- Candidate 재시도 방지: config fingerprint
- Study 전체 blacklist: 모든 유효 후보가 baseline을 개선하지 못했을 때만
- Chunk 자동 적용: `confirmed=True`이고 유효한 `gold_spans`가 있을 때만

현재 코드는 1차 구현을 진행하기 위해 다음 임시 결정을 사용한다.

- 동일 라벨의 후속 prescription도 iteration을 증가시키지 않는다.
- best가 직전 후보가 아니면 best config를 다시 적용하고 `status="applied"`로 Index/Eval에
  보낸다. 직전 후보가 best면 `verified`로 종료한다.
- Chunk 동점은 가중치 합산 없이 문서에 적은 사전순위와 baseline 변화량으로 결정한다.
- Chunk rollback은 기존 History 방식대로 baseline config를 복원한다. 컬렉션 보존은
  Index 담당 결정 전까지 구현하지 않는다.
- `graph.py`는 수정하지 않았다. 따라서 pass threshold로 곧바로 Serve될 때 실제 config는
  통과한 후보로 유지되지만, active history의 완료 표시가 남지 않는 문제는 후속 합의가
  필요하다.
