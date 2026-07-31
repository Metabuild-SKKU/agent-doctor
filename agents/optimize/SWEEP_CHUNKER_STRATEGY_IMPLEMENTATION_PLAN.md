# internal sweep에 `chunker.strategy` 축 추가 — 구현 계획

> 상태: 구현 전 설계·작업 계획
> 작업 브랜치: `feature/optimize-sweep-chunker-strategy`
> 분기 기준: `origin/main` (`ae3e46c`)
> 목표: 청킹 전략 교체(2후보)를 internal sweep으로 한 study에서 비교한다.
> 관련: `ACTION_CENTERED_OPTIMIZER_IMPLEMENTATION_PLAN.md` §2.5·§3.1·§8.0-A 선행 PR ②

---

## 1. 목적

### 1.1 현재 무슨 일이 일어나는가

`rules.py`는 청킹 전략 교체를 **2후보 스윕**으로 등록해 두었다.

```python
# agents/optimize/rules.py
"chunking_context_mismatch":   switch_to_recursive_sentence / switch_to_markdown_recursive
"retrieval_semantic_mismatch": switch_to_recursive_sentence / switch_to_markdown_recursive
```

`optimizer.py`의 제약 주석도 이 의도를 명시한다.

```python
# agents/optimize/optimizer.py:53
# rules 가 청킹 교체를 2-후보 스윕(recursive_sentence·markdown_recursive)으로 등록하므로
# 둘 다 허용해야 한다 — recursive_sentence 만 두면 markdown_recursive 후보가 실행 전에
# 필터돼 스윕이 사실상 1개가 된다(이 PR 목표인 '실행 불가 처방 언블록'과 충돌).
"chunker.strategy": {"allowed": ["recursive_sentence", "markdown_recursive"]},
```

**그런데 실제로는 sweep되지 않는다.** `internal` backend가 이 축을 지원하지 않기
때문이다.

```python
# agents/optimize/optimizer.py:138
BACKEND_SUPPORTED_PATHS: dict[str, set[str]] = {
    "rules": set(STATE_MAPPABLE_PATHS),
    "internal": {
        "retriever.top_k",
        "chunker.chunk_size",
        "chunker.chunk_overlap",
        # chunker.strategy 없음  ← 이번 작업의 대상
    },
    ...
}
```

### 1.2 현재 흐름 추적

planner는 **이미 internal을 요청한다.** `chunker.strategy`는 chunk 사전검증
경로가 아니므로 후보 수만 보고 결정하기 때문이다.

```python
# agents/optimize/planner.py:99
_CHUNK_PRECHECK_PATHS = frozenset({"chunker.chunk_size", "chunker.chunk_overlap"})
#   → chunker.strategy는 여기 없다

# planner.py  _build_request()
use_internal = (
    use_chunk_precheck
    if selected_path in _CHUNK_PRECHECK_PATHS
    else candidate_count > 1        # ← strategy는 이 경로. 후보 2개면 internal 요청
)
```

그런데 optimizer가 실행 직전에 거부한다.

```python
# agents/optimize/optimizer.py:363  _prepare_search_space()
if path not in BACKEND_SUPPORTED_PATHS.get(backend, set()):
    return {}, "unsupported_backend_path"
```

**결과**: 2후보로 등록된 청킹 전략 교체가 sweep되지 못하고, rules backend로
1회만 적용되거나 아예 건너뛰어진다.

### 1.3 이 작업이 하는 일

`internal`이 `chunker.strategy`를 지원하게 해서, 두 후보를 **한 study 안에서
실제로 비교**하고 좋은 쪽을 채택한다.

---

## 2. 사전 조사 결과 — 변경 범위가 작다

### 2.1 sweep 엔진은 후보값 타입에 무관하다

현재 지원 축 3개가 모두 숫자인 것은 **우연이며, 엔진이 숫자를 요구하지 않는다.**

```python
# agents/optimize/adapters/internal_adapter.py:252  후보 정규화
for value in values:
    if value == current or value in deduped:   # 동등 비교와 dedupe만
        continue
    deduped.append(deepcopy(value))
```

산술 연산(`float()`, `min/max`)은 전부 **점수(score)** 처리 쪽이고 후보값 쪽이
아니다. 문자열 후보에 대한 특별 처리가 필요 없다.

### 2.2 제약은 이미 범주형을 지원한다

```python
# agents/optimize/optimizer.py:425  _filter_candidate_values()
if allowed is not None and value not in allowed:
    continue
if minimum is not None or maximum is not None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        continue          # ← min/max는 숫자에만 적용, allowed와 독립
```

`chunker.strategy`는 `allowed`만 정의돼 있어 문자열 필터가 정상 동작한다.

### 2.3 나머지 계약도 이미 갖춰져 있다

| 항목 | 상태 |
| --- | --- |
| `STATE_MAPPABLE_PATHS` | ✅ `chunker.strategy` 포함 |
| `PATH_CAPABILITIES` | ✅ `chunking_strategy` 매핑 |
| `DEFAULT_CAPABILITIES["chunking_strategy"]` | ✅ `True` |
| `config_mapper` | ✅ `chunker.strategy` → `chunk_strategy` |
| `REINDEX_PATHS` | ✅ 포함 (재색인 경로 존재) |
| Index 소비 | ✅ `CHUNK_STRATEGIES` 레지스트리에서 `config["chunk_strategy"]` 소비 |

### 2.4 따라서 핵심 변경은 한 줄

```python
BACKEND_SUPPORTED_PATHS["internal"] = {
    "retriever.top_k",
    "chunker.chunk_size",
    "chunker.chunk_overlap",
    "chunker.strategy",          # ← 추가
}
```

**단, 한 줄로 끝난다고 단정하지 말 것.** §3의 확인 항목을 테스트로 먼저 고정한다.

---

## 3. 착수 전 확인할 것 (테스트로 먼저 고정)

### 3.1 fingerprint 안정성

`internal_adapter._fingerprint(config)`가 문자열 값을 안정적으로 직렬화하는가.
trial 식별과 중복 방지가 여기에 달려 있다.

```python
# internal_adapter.py:927
def _fingerprint(self, config: dict[str, Any]) -> str: ...
```

같은 문자열 후보가 항상 같은 fingerprint를 내고, 다른 후보와 충돌하지 않아야 한다.

### 3.2 baseline trial 생성

```python
# internal_adapter.py:770  _trial_config()
if trial.is_baseline:
    return self._baseline_axis_config(request, search_space)
```

범주형 축에서 baseline(현재 `chunk_strategy` 값, 기본 `markdown_recursive`)이
올바르게 만들어지는가. baseline이 후보 목록에 있는 값과 같으면
`_normalize_search_space`의 `value == current` 필터로 제외되는데
(internal_adapter.py:255), **후보 2개 중 하나가 현재값이면 sweep이 1개로 줄어든다.**

> 예: 현재 `markdown_recursive` → 후보 `[recursive_sentence, markdown_recursive]`
> → `markdown_recursive` 제외 → 실제 sweep은 `recursive_sentence` 1개 + baseline 비교
>
> 이것은 **정상 동작**이다(현재값 재적용은 no-op). 다만 "2후보 sweep"이라는
> 표현과 실제 trial 수가 다를 수 있음을 테스트로 명시한다.

### 3.3 재색인 경로

`chunker.strategy ∈ REINDEX_PATHS`이므로 `chunk_size`와 같은 재색인 경로를 타야
한다. sweep 후보를 바꿀 때마다 `state.reindex_required`가 올바르게 설정되고,
rollback 시에도 재색인 요구가 보존되는지 확인한다.

### 3.4 `chunk_precheck_context` — **제외가 맞을 가능성이 높다**

planner는 `_CHUNK_PRECHECK_PATHS`(chunk_size/overlap)에만 prescreener 컨텍스트를
붙인다. `chunker.strategy`를 여기 추가해서는 **안 된다**고 본다.

> prescreener는 gold span이 청크 경계에 걸리는지를 **기하로** 검증한다. 그런데
> strategy 교체는 **경계 생성 규칙 자체를 바꾸므로**, 기존 경계 좌표를 전제로 한
> span-경계 계산이 성립하지 않는다.

즉 strategy는 사전검증 없이 실제 sweep으로 판정하는 것이 맞다. 이 판단의 근거를
테스트와 주석으로 남긴다. (planner의 `use_internal` 분기는 이미 strategy를
`candidate_count > 1` 경로로 보내므로 **planner 변경은 불필요**하다.)

### 3.5 `allowed` 제약 필터

`allowed` 목록 밖의 값(예: 오타, 미등록 전략)이 실제로 걸러지는지 확인한다.
`_filter_candidate_values`가 min/max 없이 allowed만으로 동작해야 한다.

### 3.6 objective 스케일

`primary_metric`은 `composite_score`(0~1)이며 축 종류와 무관하다. 범주형이라고
objective가 달라지지 않는다 — 확인만 한다.

---

## 4. 변경 계획

### 4.1 `agents/optimize/optimizer.py`

```python
BACKEND_SUPPORTED_PATHS["internal"]에 "chunker.strategy" 추가
```

주석으로 근거를 남긴다: rules가 2후보로 등록했고 제약(`allowed`)도 2값을
허용하므로, sweep이 실제 비교를 수행해야 등록 의도가 실현된다.

### 4.2 `agents/optimize/planner.py`

**변경 없음.** `_CHUNK_PRECHECK_PATHS`에 추가하지 않는다(§3.4). 다만
"strategy는 prescheck 대상이 아니다"라는 판단 근거를 주석으로 남기는 것을 검토한다.

### 4.3 `agents/optimize/adapters/internal_adapter.py`

**변경 없음이 목표.** §3의 확인에서 문제가 드러나면 그때 최소 수정한다.
숫자를 가정한 코드가 발견되면 타입 분기를 넣지 말고 **타입 무관하게** 고친다.

---

## 5. 테스트 계획

### 5.1 `tests/test_internal_adapter.py`

1. 범주형 search space(`{"chunker.strategy": ["recursive_sentence", "markdown_recursive"]}`)로
   sweep이 정상 진행된다
2. 현재값과 같은 후보가 정규화 단계에서 제외된다(`value == current`)
3. 문자열 후보의 fingerprint가 안정적이고 서로 충돌하지 않는다
4. baseline trial이 올바른 축 config를 만든다
5. 후보가 baseline보다 나으면 채택, 아니면 `best_is_baseline=True`
6. `direction`/objective가 범주형에서도 동일하게 동작한다

### 5.2 `tests/test_optimizer.py`

1. `backend="internal"` + `chunker.strategy` → `unsupported_backend_path`가
   **더 이상 반환되지 않는다** (이번 변경의 핵심 회귀 테스트)
2. `allowed` 밖의 값이 걸러진다
3. capability(`chunking_strategy`)가 False면 여전히 차단된다
4. `chunker.strategy`가 재색인 필요로 표시된다
5. 단일 축 보장(multi-axis 거부)이 유지된다

### 5.3 `tests/test_planner.py`

1. `chunker.strategy` 후보 2개일 때 `optimizer="internal"`, `max_trials=2`인
   request가 생성된다
2. `chunk_precheck_context`가 **붙지 않는다**(§3.4 근거)

### 5.4 통합·회귀

- `tests/test_chunk_prescreener.py`, `tests/test_chunk_grounding_integration.py`
  — chunk_size/overlap의 기존 prescreener 경로가 영향받지 않는지
- `tests/test_optimize_agent.py` — 재색인·rollback·visit 제어
- `tests/test_index_unit.py` — `chunk_strategy` 실제 소비

---

## 6. 검증 명령

```bash
python3 -m unittest tests.test_internal_adapter tests.test_optimizer tests.test_planner
```

```bash
python3 -m unittest tests.test_chunk_prescreener tests.test_chunk_grounding_integration tests.test_optimize_agent tests.test_index_unit
```

```bash
python3 -m compileall -q agents core tests
```

---

## 7. 완료 기준

- [x] `internal` backend가 `chunker.strategy`를 거부하지 않는다
- [x] 범주형 후보가 sweep에서 실제로 평가·비교된다
- [x] 현재값과 같은 후보가 정규화 단계에서 제외된다
- [x] 문자열 fingerprint가 안정적이다
- [x] `allowed` 밖의 값이 걸러진다
- [x] `chunker.strategy`가 재색인 경로를 탄다
- [x] `chunk_precheck_context`가 strategy에 붙지 않으며 그 근거가 기록됐다
- [x] chunk_size/overlap의 기존 prescreener 동작이 변하지 않는다
- [x] `internal_adapter`에 타입 분기가 추가되지 않았다
- [x] 전체 optimize 테스트 통과

### 7.1 구현 결과 메모 — §1.2의 전제 정정

**§1.2가 "planner는 이미 internal을 요청한다"고 적었으나, rules 기본 흐름에서는
그렇지 않다.** planner는 `candidates[0].search_space`의 **후보값 개수**로
`use_internal`을 정하는데(`planner._build_request`), 두 전략은 rules.py에서
**별개의 처방 2개**로 등록돼 있어 각 candidate의 search_space가
`{"chunker.strategy": [값 1개]}`가 된다 → `candidate_count == 1` → `optimizer="rules"`.

따라서 이번 변경이 실제로 푸는 것은 다음 두 경로다.

1. **차단 해제**: 앞선 후보(예: chunk_size)가 필터돼 optimizer가 전략 후보까지
   내려온 회차에, backend가 `internal`이면 `unsupported_backend_path`로 요청 전체가
   스킵됐다. 이제 전략 축이 그 회차에서 실행된다(후보 1개 + baseline 비교).
2. **범주형 sweep 지원**: Eval이 `Finding.metadata["parameter_candidates"]`로
   `chunker.strategy` 후보 2개를 넘기면 한 study 안에서 두 전략이 실측 비교된다
   (`tests/test_planner.py::test_chunk_strategy_candidates_make_one_internal_request`).

rules.py의 처방 2개를 **한 candidate의 2후보 search_space로 합치는 일**은 이번
범위 밖이다(§4.2 "planner 변경 없음"). 그 병합은 §8이 가리키는 action 중심 전환에서
`chunker.strategy:replace` + 후보값 2개로 정의되며, 그전에 planner에서 처방을 임의로
합치면 처방 단위 blacklist·효과 귀속 계약이 흐트러진다.

---

## 8. 범위 밖

- **다른 차단 action 활성화**: `query_rewrite`, `adaptive_retrieval`,
  `context.compression` 등은 `STATE_MAPPABLE_PATHS` 미등록이거나 capability가
  꺼져 있다. 각각 별도 작업이다.
- **`embedding.model` sweep**: capability가 `False`(검증된 후보 부재)라 별개 문제다.
- **action 중심 전환**: 이 작업은 그와 독립적이며 먼저 병합돼야 한다.
  전환 후 catalog에서 이 축은 `chunker.strategy:replace` + 후보값 2개로 정의된다
  (값을 action key에 넣으면 지지 label 집합이 동일해 영구 교착이 발생한다 —
  `ACTION_CENTERED_OPTIMIZER_IMPLEMENTATION_PLAN.md` §3.1 참고).
- **새 청킹 전략 추가**: `allowed` 목록 확장은 Index의 `CHUNK_STRATEGIES` 등록과
  함께 별도로 다룬다.
