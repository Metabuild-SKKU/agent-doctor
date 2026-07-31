# Action-Centered Optimizer 전환 구현 계획

> 상태: 구현 전 설계·작업 계획
> 작업 브랜치: `feature/action-centered-optimizer`
> 최초 조사 기준: `origin/main`의 `a35677f` (**낡음 — 아래 재조사 기준으로 대체**)
> **재조사 기준: `origin/main`의 `23a6fe7`** (PR #56 병합 후)
> 목표: Optimize의 선택 중심을 failure label에서 실제 실행 가능한 config action으로 완전히 옮긴다.

### ⚠️ 기준 커밋 갱신 이력

이 문서의 §2·§6은 처음 `a35677f` 기준으로 작성됐다. 그 이후 `origin/main`이 크게
앞서가면서 전제가 달라졌다.

```text
a35677f → 23a6fe7 (optimize 관련)
  rules.py            +216 / -119
  optimizer.py        +44
  config_mapper.py    +21
  core/state.py       +29
  agents/serve/*      +400 이상
  합계                +772 / -201
```

전제를 바꾼 변경은 다음 셋이다.

1. **B그룹(generation) 5개 라벨이 `draft` → `ready`로 승격됐다.**
   `generation_config` 부재라는 블로커가 해소되어 `config_mapper`에 `generation.*`
   7경로가 추가됐고, generator가 실제로 소비한다.
2. **`chunker.strategy`·`retriever.mmr`가 실행 가능해졌다.** 초판 §2.4가 차단으로
   분류한 항목 중 넷이 이미 해제됐다.
3. **`optimizer.py`에 실행 가능성 정책이 정식으로 들어왔다.**
   `STATE_MAPPABLE_PATHS` / `PATH_CAPABILITIES` / `DEFAULT_CAPABILITIES` /
   `BACKEND_SUPPORTED_PATHS` / `DEFAULT_CONSTRAINTS`가 이미 존재한다. §6.4가 신설을
   전제로 쓰였으나 실제로는 **추출·공유**가 맞다.

§2.1·§2.4·§6.1은 재조사 결과로 교체했다. §3.1·§4.3·§4.4는 그 결과에 따라
정책을 수정했다.

### 이슈 #67과의 관계 — 이 작업을 지금 하는 이유

이슈 #67은 처방 전역 우선순위로의 **전면 전환을 반대**했다. 근거는 넷이었다.

| #67의 근거 | 재조사 결과 (`23a6fe7`) |
| --- | --- |
| ① `cost`/`confidence`가 전부 비어 정렬 재료가 없다 | **여전히 사실.** `cost` 51/51 None, `diagnosis_confidence` 25/25 None |
| ② 라벨 간 공유 처방이 적다 | **해소됨.** 실행 가능 action 16개 중 **10개가 2개 이상 라벨의 지지**를 받는다 |
| ③ A>C>B 인과가 깨진다 | **위험이 커졌다.** B그룹이 ready가 되어 실제로 A와 경쟁한다 |
| ④ 측정이 병목이다 | **대응 착수.** keep/rollback 판정에 최소 개선 마진을 도입한다(§5.11) |

따라서 이 작업의 성격을 다음과 같이 정의한다.

- **이번 전환은 성능 개선이 아니라 구조 선투자다.** ①이 비어 있는 한 점수 공식은
  사실상 `고유 probe 수 ÷ action 비용`으로 축약되며, 이는 현재 `planner._score`와
  재료가 같다. 즉 **선택 결과가 크게 달라지리라 기대해선 안 된다.**
- 그럼에도 지금 하는 이유는 ②가 해소되어 통합할 실익이 생겼고, `confidence`가
  나중에 채워질 때 **코드 변경 없이 곧바로 반영되는 자리를 미리 만들어 두기**
  위해서다.
- ③은 §4.3의 hard tier를 **불변조건**으로 격상해 방어한다.
- ④는 §5.11로 이번 범위에 포함한다.
- 이 판단이 틀렸을 경우를 대비해 §8 단계 3에 **중단 기준**을 둔다.

---

## 1. 목적

현재 Optimize는 대략 다음 순서로 동작한다.

```text
활성 Finding
  → 같은 label끼리 그룹화
  → 그룹 우선순위와 label 점수로 최상위 label 하나 선택
  → 그 label에 선언된 prescription을 순서대로 시도
  → config 하나 적용
  → Index/Eval
  → 유지 또는 rollback
```

목표 구조는 다음과 같다.

```text
모든 활성 Finding
  → 각 label이 지지하는 action 생성
  → 같은 실제 config 변경을 하나의 ActionCandidate로 통합
  → 고유 probe 커버리지·진단 신뢰도·비용·충돌을 계산
  → 최고 ActionCandidate 하나 선택
  → 후보값 하나 또는 단일 축 sweep 적용
  → Index/Eval
  → 유지 또는 rollback
  → 새 진단 결과에서 action을 다시 집계
```

최종 구조에서는 label이 처방 실행 순서를 소유하지 않는다.

- Label의 역할: 문제 진단, action 지지 근거, 영향 probe, 목표 metric 제공
- Action의 역할: 우선순위 경쟁, 후보값 탐색, 적용, 이력, 차단의 중심

이 작업은 기존 planner에 단순히 “공유 처방 보너스”를 더하는 절충안이 아니다.
데이터 모델, 선택 알고리즘, 실행 이력, blacklist, 반복 예산을 action 기준으로 함께
전환하는 작업이다.

### 유지해야 하는 안전 원칙

1. 한 번에 config 축 하나만 변경한다.
2. 실제 개선은 사용자 pipeline의 Eval 결과로 판정한다.
3. action 적용 후에는 매번 action 점수를 다시 계산한다.
4. 하한선 위반 또는 종합점수 미상승이면 rollback한다.
5. `graph.py`는 수정하지 않는다.
6. 기존 chunk 후보 경계와 prescreener fallback을 보존한다.
7. 기존 pass gate와 pending finalize를 보존한다.
8. reranker 실행 검증과 precision floor 완화를 보존한다.
9. 실행 불가능 action은 투표 전에 제외한다.
10. 같은 probe에서 파생된 여러 label을 중복 투표로 세지 않는다.

---

## 2. 현재 구조에서 확인된 문제

### 2.1 동일한 실제 변경이 여러 처방으로 중복 선언된다

현재 여러 label이 아래와 같은 실제 config 변경을 공유한다.

`23a6fe7` 기준 전수조사 결과다. ready 라벨 16개가 선언한 처방에서 canonical action
25개가 나오고, 그중 **실행 가능한 것이 16개**다. 실행 가능한 16개 중 **10개를 둘 이상의
라벨이 함께 지지한다.**

| Canonical action | 지지 | tier | 지지하는 label |
| --- | --- | --- | --- |
| `chunker.chunk_size:increase` | 3 | A | `chunking_context_mismatch`, `chunking_overchunking`, `retrieval_missing_gold` |
| `chunker.chunk_size:decrease` | 3 | A·C | `chunking_underchunking`, `retrieval_semantic_mismatch`, `too_long_context` |
| `generation.require_citation:set:true` | 3 | B | `generation_abstention_failure`, `generation_hallucination`, `generation_parametric_overreliance` |
| `generation.abstention_strict:set:true` | 3 | B | `generation_abstention_failure`, `generation_hallucination`, `generation_parametric_overreliance` |
| `chunker.chunk_overlap:increase` | 2 | A | `chunking_context_mismatch`, `retrieval_missing_gold` |
| `chunker.strategy:set:recursive_sentence` | 2 | A | `chunking_context_mismatch`, `retrieval_semantic_mismatch` |
| `chunker.strategy:set:markdown_recursive` | 2 | A | `chunking_context_mismatch`, `retrieval_semantic_mismatch` |
| `generation.temperature:decrease` | 2 | B | `generation_hallucination`, `generation_parametric_overreliance` |
| `retriever.mmr:set:true` | 2 | A·C | `retrieval_incomplete_enumeration`, `context_noise_interference` |
| `retriever.top_k:increase` | 2 | A | `retrieval_incomplete_enumeration`, `retrieval_missing_gold` |
| `retriever.top_k:decrease` | 2 | C | `lost_in_the_middle`, `too_long_context` |
| `reranker.enabled:set:true` | 1 | A | `retrieval_low_rank` |
| `reranker.candidate_count:increase` | 1 | A | `retrieval_low_rank` |
| `retriever.search_type:set:hybrid` | 1 | A | `retrieval_lexical_mismatch` |
| `generation.completeness_mode:set:true` | 1 | B | `generation_partial_answer` |
| `generation.restate_question:set:true` | 1 | B | `generation_misinterpretation` |

이 표에서 곧바로 읽히는 세 가지를 이 문서의 정책 근거로 삼는다.

1. **공유는 충분하다.** 16개 중 10개가 공유 → 통합의 실익이 있다(#67 우려 ② 해소).
2. **같은 축의 경쟁이 실재한다.** `chunker.chunk_size`(증가 3 ⟷ 감소 3),
   `retriever.top_k`(증가 2 ⟷ 감소 2), `chunker.strategy`(두 값 2 ⟷ 2).
   → §4.4의 충돌 정책이 형식이 아니라 실제로 발동한다.
3. **B그룹이 링 위에 있다.** `require_citation`·`abstention_strict`가 각각 3표이고,
   generation 경로는 재색인이 없어 비용이 A그룹 `top_k`와 같다(둘 다 1).
   비용이 약분되면 **B가 A를 점수로 이긴다.** → §4.3 hard tier가 필수다.

현재 구조에서는 같은 `top_k 증가`라도 prescription ID나 소유 label이 다르면 서로 다른
후보처럼 취급될 수 있다. 이 때문에 다음 문제가 생긴다.

- 여러 label이 같은 변경을 지지한다는 정보가 우선순위에 반영되지 않는다.
- 같은 변경이 다른 label을 통해 반복 적용될 수 있다.
- blacklist가 실제 config 변경이 아니라 label/prescription 이름에 묶인다.
- 이력에서 어떤 문제들이 공통으로 한 변경을 지지했는지 알기 어렵다.

### 2.2 현재 label 점수는 실제 action 비용과 일치하지 않을 수 있다

현재 개념상 점수는 다음과 유사하다.

```text
frequency = label의 고유 affected probe 수
confidence = rule.diagnosis_confidence, 없으면 1.0
cost = label의 첫 prescription 비용

label_score = frequency × confidence ÷ cost
```

문제점:

1. 비용이 실제 선택 action이 아니라 label의 첫 prescription에 종속될 수 있다.
2. 같은 probe에서 여러 label이 나오면 action 수준 중복 제거가 없다.
3. 동일 action이 다른 이름으로 선언되면 합산되지 않는다.
4. 반대 방향 action의 지지 강도를 비교하기 어렵다.
5. 실행 불가능 action이 먼저 선택된 뒤 optimizer에서 탈락할 수 있다.

### 2.3 실행 제어가 label/prescription 계약에 강하게 결합돼 있다

다음 영역이 현재 `(label, prescription_id)`를 중심으로 연결돼 있다.

- `planner.plan()`의 우선순위와 blacklist
- `PrescriptionCandidate.failure_label`
- `OptimizationRequest.failure_label`
- `OptimizationHistoryItem.failure_labels`
- `OptimizationHistoryItem.selected_prescription_id`
- `state.blacklist`
- `state.completed_prescriptions`
- Optimize iteration 증가 조건
- internal study 동일성 판별
- unjudgeable 재시도 제한
- reranker 전용 guardrail
- reporter의 문제 및 선택 처방 설명
- Serve 치료 경과와 해결 label 표시
- RAGBuilder payload

따라서 planner의 정렬 함수만 바꾸면 안 된다. 선택 단위와 이력 식별자를 함께 옮겨야 한다.

### 2.4 선언돼도 실행할 수 없는 action이 존재한다

대표적으로 다음 action은 rule에 존재해도 capability 또는 mapping 때문에 기본 경로에서
차단될 수 있다.

**초판의 차단 목록은 낡았다.** `23a6fe7` 기준으로 다시 조사한 결과는 다음과 같다.
판정 기준은 `optimizer.STATE_MAPPABLE_PATHS` 통과 여부와
`DEFAULT_CAPABILITIES[PATH_CAPABILITIES[path]]` 값이다.

**초판 차단 목록 7개 중 4개가 이미 해제됐다.**

| 초판이 차단으로 본 action | `23a6fe7` 실제 |
| --- | --- |
| `retriever.search_type:set:hybrid` | ✅ **해제** — `use_hybrid` 매핑, capability `hybrid_search=True` |
| `chunker.strategy:set:recursive_sentence` | ✅ **해제** — `chunk_strategy` 매핑, capability `chunking_strategy=True` |
| `mmr:set:true` | ✅ **해제** — `retriever.mmr` → `use_mmr`, capability `mmr=True` |
| `embedding.model:set:<model>` | ⚠️ 차단이지만 **이유가 다름** — 매핑은 되고 `embedding_model=False`(검증된 후보 부재) |
| `query_rewrite:set:expand` | ❌ 차단 유지 — `STATE_MAPPABLE_PATHS` 미등록 |
| `adaptive_retrieval:set:true` | ❌ 차단 유지 — 동일 |
| `context.compression.enabled:set:true` | ❌ 차단 유지 — `context_compression=False` |

**현재 차단 중인 action 9개 (전량)**

| Action | 차단 사유 |
| --- | --- |
| `query_rewrite:set:expand` | `STATE_MAPPABLE_PATHS` 미등록 |
| `adaptive_retrieval:set:true` | 〃 |
| `answer_checklist_review:set:true` | 〃 |
| `conflict_resolution_prompt:set:...` | 〃 |
| `context_ordering:set:most_relevant_edges` | 〃 |
| `noise_filter:set:true` | 〃 |
| `context.compression.enabled:set:true` | capability `context_compression=False` |
| `embedding.model:set:<model>` | capability `embedding_model=False` |
| `generation.model:set:upgrade` | capability `generation_model=False` |

두 차단 사유는 성격이 다르므로 **분리해 기록한다**(§5.7).

- `STATE_MAPPABLE_PATHS` 미등록 = mapper 계약 부재 → 영구적, catalog에 `blocked` 명시
- `capability=False` = 소비 경로는 있으나 검증된 후보/구현 부재 → 조건부, 나중에 해제 가능

Action 집계 시 rule status만 보지 않고, 현재 baseline과 runtime에서 실제 실행 가능한지
확인한 뒤 점수를 계산해야 한다.

### 2.5 sweep 가능 축이 3개뿐이다 (초판 누락)

`optimizer.BACKEND_SUPPORTED_PATHS["internal"]`은 다음 셋만 지원한다.

```text
retriever.top_k
chunker.chunk_size
chunker.chunk_overlap
```

나머지 13개 실행 가능 action은 **`rules` backend로 1회 적용만 가능하다.** 이 사실이
§4.5(후보값 통합)와 §5.8(반복 예산)에 영향을 준다.

- 후보값이 여러 개라도 sweep 대상이 아닌 축은 **후보 하나를 골라 1회 적용**해야 한다.
- `chunker.strategy`는 후보가 2개인데 internal sweep 대상이 아니다 → §3.1 참고.

---

## 3. 목표 도메인 모델

### 3.1 ActionDefinition

실제 config 변경의 정적 정의다. Label과 독립적으로 한 번만 선언한다.

```python
@dataclass(frozen=True)
class ActionDefinition:
    key: str
    canonical_path: str
    operation: Literal[
        "increase",
        "decrease",
        "set",
        "enable",
        "disable",
        "replace",
    ]
    fixed_value: Any | None
    reindex_required: bool
    base_cost: float
    capability: str | None
    prerequisites: tuple[str, ...]
    conflict_family: str
    description: str
    tradeoffs: tuple[str, ...] = ()
```

예:

```python
ActionDefinition(
    key="chunker.chunk_size:increase",
    canonical_path="chunker.chunk_size",
    operation="increase",
    fixed_value=None,
    reindex_required=True,
    base_cost=3.0,
    capability="chunking",
    prerequisites=(),
    conflict_family="chunker.chunk_size",
    description="청크 크기를 늘린다.",
)
```

#### `base_cost` 산정 — **합의 완료: 현행 이분법을 그대로 옮긴다**

`confidence`가 전부 `None`(=1.0)인 현재, `base_cost`는 점수 공식에 남은 **유일한
차별 재료**다. 그만큼 임의로 정하면 선택이 조용히 왜곡된다.

따라서 새 숫자를 만들지 않고 `planner._derive_cost`의 현행 규칙을 그대로 옮긴다.

```python
base_cost = 3.0 if reindex_required else 1.0     # 재색인 = 3, 런타임 = 1
```

- 근거는 `optimizer.REINDEX_PATHS` 하나로 통일한다(라벨의 첫 처방이 아니라 **action
  자신의** 재색인 여부 — 초판 §2.2가 지적한 왜곡의 수정).
- 세분화(실측 소요 시간·LLM 호출 수 기반)는 `confidence` 생산과 함께 **별도 작업**으로
  둔다. 근거 없는 숫자를 지금 만들면 §1의 선투자 논리와 어긋난다.
- `score_breakdown`에 `cost_source: "reindex_flag"`를 기록해, 나중에 실측 기반으로
  바뀔 때 이력에서 구분할 수 있게 한다.

**부작용을 인지할 것**: 이 이분법에서는 같은 축의 반대 방향(§4.4)과 재색인이 없는
A·B 그룹 action(§4.3)이 **분모가 같아 약분된다.** 그래서 충돌 정책과 hard tier가
필요하다 — cost로는 그 둘을 가를 수 없다.

Action key 규칙:

```text
<canonical_path>:<operation>[:<stable-value-id>]
```

예:

```text
retriever.top_k:increase
retriever.top_k:decrease
chunker.chunk_size:increase
chunker.chunk_size:decrease
reranker.enabled:enable
retriever.search_type:replace        # 후보값 [hybrid]
chunker.strategy:replace             # 후보값 [recursive_sentence, markdown_recursive]
embedding.model:replace              # 후보값 [<model>...]
```

원칙:

- action key에는 label이나 기존 prescription ID를 넣지 않는다.
- 방향 action의 실제 후보 숫자는 key에 넣지 않는다.
- **고정값 교체 action은 값을 key에 넣지 않는다.** 축 단위 `replace` action 하나로 두고,
  값은 전부 후보값으로 넘긴다. (초판 규칙에서 **변경** — 아래 이유 참고)
- boolean 축은 `set:true`/`set:false` 대신 `enable`/`disable`을 쓴다.
- 한 action은 canonical config 축 하나만 소유한다.
- 보조 안전 플래그는 patch에 추가할 수 있지만 action의 정체성은 하나로 유지한다.

#### 초판 규칙을 바꾼 이유 — starvation(영구 교착)

초판은 "고정값 교체 action은 stable value를 key에 포함한다"였다. 이 규칙을 `23a6fe7`의
`chunker.strategy`에 적용하면 **그 축이 영원히 선택되지 않는다.**

```text
chunker.strategy:set:recursive_sentence   ← chunking_context_mismatch, retrieval_semantic_mismatch
chunker.strategy:set:markdown_recursive   ← chunking_context_mismatch, retrieval_semantic_mismatch
                                             └ 지지 label 집합이 완전히 동일
```

지지 probe 집합이 같으면 두 action의 점수는 **수학적으로 항상 정확히 같다.** 그러면
§4.4의 충돌 보류 규칙에 영구히 걸려 **한 번도 시도되지 않는다.** 왕복(oscillation)의
반대인 기아(starvation)이며, `conflict_margin_ratio`를 어떤 값으로 정해도 해소되지 않는다.

이는 `origin/main`이 의도적으로 만든 설계를 되돌리는 것이기도 하다. `optimizer.py`의
제약 주석이 이를 명시한다.

```python
# rules 가 청킹 교체를 2-후보 스윕(recursive_sentence·markdown_recursive)으로 등록하므로
# 둘 다 허용해야 한다 — recursive_sentence 만 두면 markdown_recursive 후보가 실행 전에
# 필터돼 스윕이 사실상 1개가 된다(이 PR 목표인 '실행 불가 처방 언블록'과 충돌).
"chunker.strategy": {"allowed": ["recursive_sentence", "markdown_recursive"]},
```

즉 두 값은 **경쟁자가 아니라 같은 action의 두 후보**다. 값을 key에서 빼면 이 의도가
보존된다.

**정밀도 손실은 없다.** blacklist는 action key가 아니라 `ActionAttemptKey`(§3.5)로
걸리며, 여기에 `candidate_fingerprint`가 포함되어 "어떤 값으로 시도했는가"는 여전히
정확히 구분된다.

**다만 §2.5의 제약과 충돌한다.** `chunker.strategy`는 후보가 2개인데 internal sweep
대상이 아니다(`BACKEND_SUPPORTED_PATHS["internal"]`에 없음). 따라서 다음 중 하나를
합의해야 한다.

| 선택지 | 내용 | 비용 |
| --- | --- | --- |
| (a) `internal`에 `chunker.strategy` 추가 | 2후보를 정식 sweep | adapter 변경 필요 |
| (b) rules backend가 후보 하나만 적용 | 나머지 후보는 다음 방문에 | sweep 이점 상실 |
| (c) 이번 범위에서 제외 | 단일 후보로 고정 | 기존 2-후보 설계 후퇴 |

권장은 **(b)** 다. §12 권장 범위("차단 기능 활성화는 별도 PR")와 일관되고,
`ActionStudyKey`로 "이 baseline에서 어떤 후보를 이미 썼는지" 추적하면 다음 방문에
자연스럽게 나머지 후보로 넘어간다.

### 3.2 LabelActionRule

Label이 어떤 action을 왜 지지하는지 나타내는 선언이다.

```python
@dataclass(frozen=True)
class LabelActionRule:
    label: str
    action_key: str
    applies_when: dict[str, Any]
    candidate_strategy: str
    target_metrics: tuple[str, ...]
    confidence: float | None
    reason_template: str
```

변경 전:

```python
"retrieval_missing_gold": {
    "prescriptions": [
        {
            "id": "increase_top_k",
            "patch": {"top_k": "increase"},
            "reindex": False,
        },
    ],
}
```

변경 후:

```python
"retrieval_missing_gold": {
    "actions": [
        {
            "action_key": "retriever.top_k:increase",
            "candidate_strategy": "gold_rank_top_k",
        },
    ],
}
```

Label rule에는 실제 patch와 비용을 중복 선언하지 않는다.

### 3.3 ActionSupport

한 label 묶음이 action에 제공하는 런타임 근거다.

```python
@dataclass
class ActionSupport:
    action_key: str
    label: str
    group: FailureGroup
    finding_ids: list[str]
    affected_probes: set[str]
    confidence: float
    target_metrics: list[str]
    candidate_values: list[Any]
    applies_when: dict[str, Any]
    reason: str
    grounding_metadata: dict[str, Any]
```

원칙:

- 같은 label의 여러 Finding은 먼저 하나의 support로 묶는다.
- `affected_probes`는 set으로 중복 제거한다.
- preliminary Finding은 기존과 동일하게 자동 처방에서 제외한다.
- manual Finding은 투표에 넣지 않고 decision/report에 별도로 보존한다.
- `applies_when`을 만족하지 않는 support는 생성하지 않는다.

### 3.4 ActionCandidate

같은 action key를 지지하는 support를 통합한 실제 선택 단위다.

```python
@dataclass
class ActionCandidate:
    action_key: str
    definition: ActionDefinition
    supports: list[ActionSupport]
    supporting_labels: list[str]
    supporting_probes: set[str]
    opposing_action_keys: list[str]
    opposing_labels: list[str]
    search_space: dict[str, list[Any]]
    target_metrics: list[str]
    score: float
    score_breakdown: dict[str, Any]
    status: Literal["ready", "blocked", "conflicted"]
    reason: str
    metadata: dict[str, Any]
```

최종적으로 `PrescriptionCandidate`를 제거하고 `ActionCandidate`로 교체한다.
마이그레이션 중 UI 호환을 위해 한 단계 동안 두 모델을 읽을 수는 있지만, 실행 선택
단위는 하나만 유지한다.

### 3.5 ActionAttemptKey와 ActionStudyKey

기존 `(label, prescription_id)` blacklist는 실제 변경을 충분히 표현하지 못한다.

권장 식별자:

```python
@dataclass(frozen=True)
class ActionAttemptKey:
    action_key: str
    baseline_fingerprint: str
    candidate_fingerprint: str


@dataclass(frozen=True)
class ActionStudyKey:
    action_key: str
    baseline_fingerprint: str
    search_space_fingerprint: str
```

용도:

- `blocked_action_attempts`: Eval 후 실패한 정확한 action 전이
- `completed_action_studies`: 해당 baseline/search space에서 완료된 sweep
- visit-local exclusions: 일시적 capability 또는 runtime 문제
- unjudgeable attempts: 품질 실패 blacklist와 분리

Fingerprint 입력:

```text
action_key
baseline canonical effective config
candidate config
관련 runtime capability identity
```

---

## 4. Action 선택 알고리즘

### 4.1 전체 흐름

```python
def plan(state, exclusions):
    manual, actionable = split_findings(state.report.findings)
    decision = decide_mode(state, actionable, manual)
    if decision.mode != "apply_optimize":
        return None, decision

    grouped = group_findings_by_label(actionable)
    supports = build_action_supports(grouped, state)
    candidates = aggregate_action_candidates(supports, state)
    candidates = filter_ineligible_actions(candidates, state, exclusions)
    candidates = resolve_action_conflicts(candidates)
    selected = rank_action_candidates(candidates)[0]
    request = build_action_request(selected, state, manual)
    return request, decision
```

### 4.2 고유 probe 기반 가중 투표

Label 개수를 그대로 합산하면 같은 probe에서 파생된 여러 label이 여러 표가 된다.
Probe 단위로 한 번만 기여하게 한다.

권장 초기 공식:

```text
probe_support(p, action)
  = 해당 probe에서 action을 지지한 support 중 가장 높은 confidence

coverage_weight(action)
  = Σ probe_support(p, action)

action_score
  = coverage_weight(action) / action.base_cost
```

Score breakdown:

```text
supporting_label_count
supporting_probe_count
weighted_probe_support
grounded_support_count
target_metric_count
runtime/reindex cost
```

Label 다양성은 taxonomy label을 많이 만들수록 점수가 커지는 왜곡이 생길 수 있으므로,
초기에는 주 점수가 아닌 tie-breaker로 사용한다.

### 4.3 A > C > B 인과 정책

선택 중심이 action으로 이동해도 원인 그룹은 support의 속성으로 유지한다.

초기 권장안:

```text
candidate.causal_rank
  = candidate를 지지하는 label 중 가장 높은 우선순위 그룹

정렬:
  1. causal_rank 오름차순(A, C, B)
  2. action_score 내림차순
  3. grounded support 수 내림차순
  4. action 비용 오름차순
  5. action_key 사전순
```

이 구조는 label을 먼저 선택하지 않는다. Action끼리 경쟁하되 기존 검색 우선 인과 정책을
안전장치로 유지한다.

#### 🔒 이 조항은 불변조건이다 (초판에서 격상)

초판은 hard tier를 "초기 권장안"으로 두고 "실험 데이터가 쌓이면 group multiplier로
바꿀 수 있도록" 여지를 남겼다. **`23a6fe7`에서는 이를 선택지로 두면 안 된다.**

초판이 작성된 `a35677f`에서는 B그룹이 전부 `draft`라 A와 경쟁할 일이 없었다. 그래서
hard tier가 형식적 안전장치였다. 지금은 다르다.

```text
[B] generation.require_citation:enable   지지 3 probe군 ÷ 비용 1 = 3.0   ← 이긴다
[A] retriever.top_k:increase             지지 2 probe군 ÷ 비용 1 = 2.0
```

generation 경로는 **재색인이 없어 비용이 1**이고, `retriever.top_k`도 1이다. 비용이
약분되면 순수 지지 수 싸움이 되며, `require_citation`·`abstention_strict`는 각각
**3개 라벨의 지지**를 받는다. hard tier가 없으면 **검색이 새는 상태에서 생성 프롬프트를
먼저 손대는 순서 역전이 기본 시나리오가 된다**(garbage-in tuning).

따라서:

- **hard tier는 정렬 1차 키로 유지한다.** `causal_rank`가 다르면 점수를 보지 않는다.
- group multiplier(§5.4의 대안)는 **B그룹이 ready인 한 채택하지 않는다.** multiplier는
  지지 수가 충분히 몰리면 여전히 역전을 허용하기 때문이다(`10×1 > 2×3`).
- 정책 객체로 분리하는 것은 유지하되, 기본값 교체는 별도 PR과 별도 근거를 요구한다.
- §7.11 invariant에 "B그룹 action은 실행 가능한 A/C action이 남아 있는 한 선택되지
  않는다"를 추가한다.

### 4.4 반대 방향 충돌

같은 config 축에서 반대 action이 동시에 활성화될 수 있다.

예:

```text
chunker.chunk_size:increase
chunker.chunk_size:decrease
```

초기 정책:

1. 서로 다른 causal tier면 높은 tier를 우선한다.
2. 같은 tier면 grounded support가 있는 쪽을 우선한다.
3. 양쪽 근거 수준이 같으면 점수 차이를 비교한다.
4. 합의된 `conflict_margin_ratio` 이상 차이일 때만 우세 action을 선택한다.
5. 차이가 작으면 해당 config 축을 이번 방문에서 보류한다.
6. 보류 후 다음 순위의 다른 config 축 action을 검토한다.
7. 양쪽 점수와 보류 이유를 report metadata에 남긴다.

근소한 충돌을 단순 다수결로 처리하면 config가 증가와 감소를 왕복할 수 있다.

#### 실제 충돌 축에 적용한 결과 (`23a6fe7`)

이 정책을 §2.1의 충돌 3축에 적용하면 **3단계까지 가는 축은 하나뿐이다.**

| 축 | 1단계 tier | 2단계 grounded | 결과 |
| --- | --- | --- | --- |
| `retriever.top_k` | 증가 **A** vs 감소 **C** | — | ✅ **1단계 종료.** 충돌 정책 불필요 |
| `chunker.strategy` | 양쪽 A | 양쪽 없음 | — §3.1로 해소(단일 action + 후보 2개) |
| `chunker.chunk_size` | 양쪽 A¹ | **양쪽 있음**² | ⚠️ **3단계까지 진행 — 유일한 실제 대상** |

¹ 감소 쪽은 A·C 혼합이나 §4.3의 `causal_rank` = "가장 높은 tier"에 따라 A로 판정된다.
² 증가 = `chunking_overchunking`, 감소 = `too_long_context`가 각각
`_ground_chunk_size_candidates`로 근거값을 만든다.

#### 대부분의 왕복은 근거값 계산이 먼저 차단한다 (초판 누락)

`planner._ground_chunk_size_candidates`는 gold span 길이 분포(P85)로 목표값을 정한 뒤,
방향이 맞지 않으면 후보 생성 자체를 실패시킨다.

```python
if (direction == "decrease" and target >= current_int) or (
    direction == "increase" and target <= current_int
):
    metadata["status"] = "direction_conflict"
    return None, metadata
```

§4.6에 따라 **후보값이 없는 action은 점수 계산 전에 제외**되므로, 정상적으로 근거가
계산되는 상황에서는 **증가·감소가 동시에 링에 오르지 않는다.** 즉 충돌 정책이 실제로
필요한 것은 근거 계산이 실패하는 다음 경우로 한정된다.

```text
missing_gold_spans      gold span 없음
insufficient_spans      span 3개 미만
invalid_policy          정책값 무효
```

이때만 방향 키워드 폴백(`×2` / `÷2`)으로 양쪽이 살아남는다.

#### `conflict_margin_ratio` 합의값

노이즈 한 개 차이로 방향이 뒤집히지 않도록 **상대·절대 조건을 모두** 건다.

```text
우세 판정 조건 (둘 다 만족해야 함)
  (a) 상대: (우세 점수 - 열세 점수) / 우세 점수 >= 0.20
  (b) 절대: 고유 probe 지지 수 차이 >= 2

둘 중 하나라도 미충족 → 해당 축을 이번 방문 보류, 다음 순위 축으로
```

(b)가 필요한 이유는 상대 조건만 두면 `3 : 2`가 33%로 통과하는데, probe 1개 차이는
§5.11의 측정 노이즈 안에 묻히기 때문이다.

### 4.5 후보값 통합

같은 action을 지지해도 label마다 후보값이 다를 수 있다.

예:

```text
retrieval_missing_gold       → chunk_size [700, 800]
chunking_context_mismatch    → chunk_size [750, 900]
chunking_overchunking        → chunk_size [900, 1200]
```

병합 정책:

1. support별 후보값과 provenance를 보존한다.
2. action 방향에 맞지 않는 값을 제거한다.
3. stable dedupe한다.
4. absolute/max-step constraint를 적용한다.
5. baseline에서 변화량이 작은 값부터 정렬한다.
6. candidate budget까지만 search space에 포함한다.
7. 후보가 여러 개면 internal sweep 또는 chunk prescreener가 선택한다.
8. 후보가 하나면 rules backend가 한 번 적용한다.

Metadata 예:

```python
{
    "candidate_support": {
        700: ["retrieval_missing_gold"],
        750: ["chunking_context_mismatch"],
        900: [
            "chunking_context_mismatch",
            "chunking_overchunking",
        ],
    }
}
```

Chunk 후보에서는 기존 정책을 유지한다.

- 정상 intersection이 있으면 단계별 후보군 유지
- 정상 후보가 없을 때만 absolute boundary clamp
- 유효한 document/evidence span이 있을 때만 prescreener 선택
- context/dependency 부재는 rules fallback
- 실제 prescreener 실패는 숨기지 않음

### 4.6 실행 가능성은 점수 계산 전에 확인

다음 정책을 planner와 optimizer가 공유하도록 분리한다.

```text
canonical path 정규화
state mapping 지원
backend path 지원
pipeline capability
runtime capability
baseline prerequisite
numeric/allowed constraints
현재값과 같은 no-op 제거
단일 축 보장
```

Planner는 실행 가능한 candidate만 점수화한다.
Optimizer는 실행 직전 동일 정책을 재검증해 외부 입력 변조를 방어한다.

---

## 5. 구현 전 합의가 필요한 정책

### 5.1 Action identity

합의 항목:

- 방향 action key에 후보 숫자를 포함할지
- model 교체 같은 고정값 action은 값을 key에 포함할지
- multi-key patch를 금지할지

권장:

- 기본적으로 단일 canonical path
- 후보 숫자는 attempt fingerprint에만 포함
- 고정값은 action key에 포함
- 안전을 위한 부수 플래그만 atomic action의 예외로 허용

### 5.2 투표 단위

권장:

- 고유 probe 단위
- 같은 probe의 여러 label 지지는 confidence 최대값 한 번만 반영
- label 수는 설명과 tie-breaker에만 사용

### 5.3 Confidence

현재 rule의 diagnosis confidence가 비어 있는 경우가 많다.

권장:

- 1차 전환에서는 현재 `1.0` fallback 유지
- score breakdown에 `confidence_source` 기록
- 별도 Eval 작업에서 Finding별 confidence를 생산

### 5.4 Group 인과 — **합의 완료**

선택지:

- A > C > B hard tier ← **채택**
- ~~group multiplier~~ — B그룹이 ready인 한 채택하지 않는다(§4.3)
- B action 전에 A/C 해결 필수 — hard tier로 사실상 충족

결정:

- **hard tier를 불변조건으로 유지한다.** 근거와 수치는 §4.3 참고.
- 구조 전환과 그룹 정책 변경을 한 번에 섞지 않는다.
- multiplier 전환은 별도 PR + 별도 근거를 요구한다.

### 5.5 Conflict 처리

합의 항목:

- 반대 방향 action을 점수만으로 결정할지
- 근소한 충돌은 축 전체를 보류할지
- grounded evidence에 우선권을 줄지

권장:

- 근소한 충돌 보류
- grounded action 우선
- 양쪽 점수와 이유 노출

### 5.6 Candidate value merge

권장:

- 안전한 union 후 candidate budget 제한
- 변화량이 작은 후보부터
- chunk는 prescreener
- top-k 등은 internal sweep

### 5.7 Blacklist 범위

권장:

- 품질 실패는 exact `ActionAttemptKey` 차단
- 완료된 sweep는 `ActionStudyKey` 차단
- capability 미지원은 visit-local exclusion 또는 deferred 기록
- 새 baseline에서는 같은 action을 다시 시도할 수 있음

### 5.8 반복 예산

현재 iteration은 새 label 시작을 중심으로 계산될 수 있다.

새 의미:

```text
iteration = 실제 새 ActionStudy를 적용한 횟수
```

- 같은 action 내부 sweep은 iteration 하나
- baseline이 선택돼 실제 변경이 없으면 예산 미소비
- rollback 후 다른 action은 새 iteration
- 같은 action도 새 baseline에서 새 study면 새 iteration
- 절대 Optimize visit 상한은 유지

#### ⚠️ `iteration`은 `graph.py`가 종료 조건으로 읽는다 (초판 누락)

§6.16과 안전 원칙 5는 "`graph.py`를 수정하지 않는다"고만 적었으나, **`graph.py`는
`iteration`을 파이프라인 종료 조건으로 읽는다.**

```python
# graph.py:65  route_after_eval
if state.iteration >= state.max_iterations:
    if history.find_pending(...):   # 마지막 판정 기회
        return "optimize"
    return "serve"                  # ← 파이프라인 종료
```

따라서 **`graph.py`를 한 줄도 고치지 않아도, `iteration` 의미를 바꾸면 그 동작이
달라진다.** 특히 "no-op이면 예산 미소비"는 이 종료 조건의 발동을 직접 미룬다.

##### 이 문제는 `graph.py` 수정으로 해결되지 않는다

- `graph.py`는 `iteration`을 **읽기만** 한다. 증가는 `agent.py` 소관이다.
- `iteration`이 오르지 않는 경로가 반복되면 위 가드는 **영원히 발동하지 않는다.**
- 실제 최종 방어선은 `graph.py`가 모르는 값이다.

```python
# agent.py:462
if state.optimize_visit_count >= state.max_optimize_visits:
    return _stop_at_optimize_visit_limit(state)
```

즉 라우팅을 고쳐도 얻을 것이 없고, 5개 에이전트 공용 라우팅을 Optimize 하나 때문에
바꾸는 비용만 남는다. **`graph.py` 무수정 원칙을 유지한다.**

##### 대신 착수 전에 증명해야 할 것

이 프로젝트는 이미 같은 계열의 사고를 겪었다(`fix: bound optimize visits and close
completed sweeps`, `fix: finalize optimize visit limit safely`). 따라서 다음을
표로 만들어 합의한 뒤 구현에 들어간다.

| 확인 항목 | 내용 |
| --- | --- |
| 미소비 경로 목록 | `iteration`이 오르지 않는 모든 분기를 열거 |
| 수렴 증명 | 각 미소비 경로가 유한 횟수 안에 소비 경로 또는 종료로 이어짐 |
| visit 소비 여부 | 미소비 경로에서도 `optimize_visit_count`는 반드시 증가하는가 |
| 최악 방문 수 | `max_iterations=3`, `max_optimize_visits=20` 기준 상한 |
| 종료 경로 동등성 | 모든 종료 경로가 pending finalize와 baseline 복원을 보장하는가 |

**핵심 불변조건**: `optimize_visit_count`는 **어떤 경로에서도 반드시 증가한다.**
`iteration` 미소비는 허용하되, visit 미소비는 허용하지 않는다. 이것이 성립하면
최악의 경우에도 `max_optimize_visits`에서 종료가 보장된다.

### 5.9 결과 귀속

Action이 지지받았다는 사실과 label이 실제 해결됐다는 사실을 구분한다.

권장:

- 적용 당시 supporting labels/probes 저장
- 다음 Eval의 remaining labels/probes 저장
- before/after confirmed Finding으로 resolved labels 계산
- UI에서 “지지한 문제”와 “실제로 사라진 문제” 구분

### 5.10 하위 호환

권장 순서:

1. 새 action 필드 추가
2. 기존 필드를 파생값으로 함께 채움
3. 모든 소비처를 action 필드로 전환
4. 기존 필드를 deprecated
5. 저장 상태 호환 요구 확인 후 제거

### 5.11 개선 판정 최소 마진 — **합의 완료** (신설)

이슈 #67의 선결 과제 ②(측정 노이즈)에 대한 대응이다. 초판에는 없던 항목이며,
**이번 범위에 포함한다.**

#### 문제

현재 판정은 티끌만큼만 올라도 유지한다.

```python
# history.py judge()
if after_score > before_score:      # 0.0001 상승도 "개선"
    return Verdict(keep=True, ...)
```

측정 노이즈로 우연히 오른 값을 개선으로 오인하면, 롤백 안전망을 **통과하면서**
왕복이 발생한다. 롤백은 "점수가 안 올랐을 때" 작동하는데 노이즈가 점수를 올려버리기
때문에, §4.4의 충돌 정책만으로는 막을 수 없다.

#### 결정

```text
최소 개선 마진 = composite 표시 점수 기준 2점
적용 지점      = 두 곳 모두 동일한 값
```

| 적용 지점 | 함수 | 역할 |
| --- | --- | --- |
| keep/rollback 판정 | `history.judge` | 적용된 config를 유지할지 |
| sweep best 선정 | `internal_adapter._best_completed_trial` | baseline 대비 나은 후보인지 |

**⚠️ 스케일 주의.** 두 지점 모두 내부적으로 **0~1 정규화 값**을 쓴다.

```python
# history._read_score
return float(total) / 100.0          # composite 0~100 → 0~1

# planner._report_metrics
metrics["composite_score"] = float(composite_total) / 100.0
```

따라서 **표시 점수 2점 = 내부 값 `0.02`** 이다. 상수를 하드코딩하지 말고 한 곳에
정의해 두 모듈이 공유한다.

```python
MIN_IMPROVEMENT_MARGIN = 0.02        # composite 표시 기준 2점
```

`judge`만 고치고 sweep을 두면 **"sweep이 고른 최선이 judge에서 탈락"**해 예산 한 번을
통째로 날린다. 반드시 함께 적용한다.

#### 부작용 — 조기 종료 위험

마진을 올리면 롤백이 늘고, 롤백은 blacklist를 채운다.

```python
# planner.plan()
reason="처방 후보가 모두 블랙리스트에 걸림"   → use_current → serve
```

즉 **마진 ↑ → 롤백 ↑ → 후보 고갈 → 아무 개선 없이 종료**가 가능하다. 2점은 실측
테스트로 정한 값이나, 다음을 함께 기록해 사후 검증할 수 있게 한다.

- 마진 때문에 탈락한 시도 수와 그때의 점수 차이
- blacklist 고갈로 조기 종료한 횟수

#### 후속 (별도 작업)

고정값 2점은 노이즈 크기를 **가정**할 뿐 측정하지 않는다. Eval이 probe별 점수를
노출할 수 있게 되면 표준오차(`std/√n`) 기반 동적 마진으로 대체할 수 있다. 이는
Eval과의 계약 변경이 필요하므로 별도 작업으로 둔다.

---

## 6. 파일별 구현 계획

### 6.1 새 파일: `agents/optimize/action_catalog.py`

역할:

- `ActionDefinition` 레지스트리
- action key lookup
- action key uniqueness 검증
- canonical path/operation/value 계약 검증
- reindex/cost/capability/prerequisite의 단일 진실 원천
- conflict family 정의

**초판의 등록 목록(7개)은 `a35677f` 기준이라 낡았다.** `23a6fe7`에서 실행 가능한
action은 16개이며, catalog는 **차단된 9개도 blocked reason과 함께** 등록한다.

#### 우선 등록 — 실행 가능 16개

| action key | tier | reindex | sweep |
| --- | --- | --- | --- |
| `retriever.top_k:increase` | A | — | internal |
| `retriever.top_k:decrease` | C | — | internal |
| `chunker.chunk_size:increase` | A | ✔ | internal |
| `chunker.chunk_size:decrease` | A·C | ✔ | internal |
| `chunker.chunk_overlap:increase` | A | ✔ | internal |
| `chunker.strategy:replace` | A | ✔ | rules only ⚠️ |
| `reranker.enabled:enable` | A | — | rules only |
| `reranker.candidate_count:increase` | A | — | rules only |
| `retriever.search_type:replace` | A | — | rules only |
| `retriever.mmr:enable` | A·C | — | rules only |
| `generation.require_citation:enable` | B | — | rules only |
| `generation.abstention_strict:enable` | B | — | rules only |
| `generation.temperature:decrease` | B | — | rules only |
| `generation.completeness_mode:enable` | B | — | rules only |
| `generation.restate_question:enable` | B | — | rules only |
| `generation.grounding_strict:enable`¹ | B | — | rules only |

¹ `STATE_MAPPABLE_PATHS`·`PATH_CAPABILITIES`에는 있으나 `23a6fe7`의 ready 라벨 처방에서
직접 참조되지 않는다. catalog에는 등록하되 지지 label이 없으면 후보에 오르지 않는다.

**⚠️ `internal` sweep이 지원하는 축은 셋뿐이다**(§2.5). 나머지 13개는 `rules` backend로
1회 적용된다. `chunker.strategy:replace`는 후보가 2개인데 sweep 대상이 아니므로
§3.1의 (b)안을 따른다.

#### 차단 등록 — 9개 (blocked reason 필수)

| action key | blocked reason | 성격 |
| --- | --- | --- |
| `query_rewrite:*` | `not_state_mappable` | 영구 — mapper 계약 부재 |
| `adaptive_retrieval:enable` | `not_state_mappable` | 〃 |
| `answer_checklist_review:enable` | `not_state_mappable` | 〃 |
| `conflict_resolution_prompt:replace` | `not_state_mappable` | 〃 |
| `context_ordering:replace` | `not_state_mappable` | 〃 |
| `noise_filter:enable` | `not_state_mappable` | 〃 |
| `context.compression.enabled:enable` | `capability_off` | 조건부 — 소비 노드 부재 |
| `embedding.model:replace` | `capability_off` | 조건부 — 검증된 후보 부재 |
| `generation.model:replace` | `capability_off` | 조건부 — 검증된 후보 부재 |

두 사유는 해제 조건이 다르므로 **catalog에서 구분해 기록한다**(§5.7).
`capability_off`는 `DEFAULT_CAPABILITIES` 값만 바꾸면 열리지만, `not_state_mappable`은
`config_mapper` 계약과 소비 노드가 함께 추가돼야 한다.

검증:

- 같은 canonical path/operation/value 중복 금지
- 한 action은 기본적으로 config 축 하나
- mapper가 지원하지 않는 path는 명시적인 blocked reason 필요

### 6.2 새 파일: `agents/optimize/action_aggregator.py`

역할:

- Finding 그룹을 `ActionSupport`로 변환
- 동일 action key 통합
- 고유 probe 집계
- target metric union
- score와 score breakdown 계산
- conflict 분석
- deterministic ranking

공개 API:

```python
def build_action_supports(
    grouped_findings: dict[str, list[Finding]],
    state: AgentDoctorState,
) -> list[ActionSupport]:
    ...


def aggregate_action_candidates(
    supports: list[ActionSupport],
    state: AgentDoctorState,
    exclusions: set[ActionAttemptKey],
) -> list[ActionCandidate]:
    ...


def rank_action_candidates(
    candidates: list[ActionCandidate],
    policy: ActionVotePolicy,
) -> list[ActionCandidate]:
    ...
```

주의:

- Finding metadata를 다시 진단하지 않는다.
- 같은 probe 중복 투표를 제거한다.
- manual/preliminary label을 투표에 포함하지 않는다.
- score tie에서도 결과가 deterministic해야 한다.
- 실행 불가능 action은 score 계산 전에 제외한다.

### 6.3 새 파일: `agents/optimize/candidate_values.py`

현재 planner의 label별 후보값 생성과 chunk 근거화 로직을 분리한다.

이동 대상:

- top-k knee 계산
- gold rank 기반 top-k grounding
- chunk size/overlap 후보 정책
- percentile 계산
- evidence window 처리
- absolute boundary clamp
- candidate constraint 처리
- finding별 search space 생성
- symbolic fallback 판단
- chunk precheck context 생성

변경 방향:

- label/path 하드코딩을 `candidate_strategy` registry로 교체
- 각 support가 자기 후보를 만든 뒤 aggregator가 병합
- evidence analysis는 label/strategy 단위 memoize
- 기존 수학은 먼저 그대로 이동
- 별도 리팩터링은 구조 전환 뒤 진행

### 6.4 새 파일: `agents/optimize/eligibility.py`

Planner와 optimizer가 공유할 순수 실행 가능성 정책이다.

**이것은 신설이 아니라 추출이다.** `23a6fe7`의 `optimizer.py`에 이미 다음이 모두
존재하며, 새 로직을 만드는 것이 아니라 planner도 쓸 수 있게 옮기는 작업이다.

| 기존 위치 | 심볼 |
| --- | --- |
| `optimizer.py:45` | `DEFAULT_CONSTRAINTS` |
| `optimizer.py:61` | `DEFAULT_CAPABILITIES` |
| `optimizer.py:91` | `PATH_CAPABILITIES` |
| `optimizer.py:113` | `STATE_MAPPABLE_PATHS` |
| `optimizer.py:138` | `BACKEND_SUPPORTED_PATHS` |
| `optimizer.py` | `REINDEX_PATHS` |

이동 또는 공유:

- constraints
- capabilities
- path capabilities
- state mappable paths
- backend supported paths
- runtime capability merge
- candidate value filtering
- no-op filtering

주의: 옮기면서 값을 바꾸지 않는다. 특히 `DEFAULT_CONSTRAINTS["chunker.strategy"]`의
`allowed` 2값은 §3.1의 2-후보 설계 근거이므로 유지한다.

API:

```python
def prepare_action_search_space(
    action: ActionCandidate,
    baseline_config: dict[str, Any],
    *,
    backend: str,
    capabilities: dict[str, Any],
    runtime_capabilities: dict[str, Any],
    constraints: dict[str, Any] | None = None,
) -> PreparedAction | ActionRejection:
    ...
```

특수 조건:

- reranker candidate count는 reranker enabled와 runtime verified 필요
- runtime unavailable은 품질 blacklist로 분류하지 않음
- chunk overlap은 chunk size 비율 제약 유지
- multi-axis action 기본 거부

### 6.5 `agents/optimize/rules.py`

역할을 label 진단에서 action 지지로 바꾼다.

변경:

1. `LABEL_TO_PRESCRIPTIONS`를 `LABEL_TO_ACTION_RULES`로 교체
2. patch/reindex/cost 중복 제거
3. 각 항목은 action key, candidate strategy, applies_when 중심
4. target metrics와 manual action은 label 수준 유지
5. `is_actionable`, `is_manual`, `get_rule` 갱신
6. migration 중 필요하면 기존 view를 action catalog에서 파생
7. 같은 실제 변경인데 이름만 다른 prescription을 통합
8. 같은 ID지만 patch가 다르면 action key로 분리

### 6.6 `agents/optimize/schemas.py`

추가:

- `ActionOperation`
- `ActionDefinition`
- `LabelActionRule`
- `ActionSupport`
- `ActionCandidate`
- `ActionVotePolicy`
- `ActionAttemptKey`
- `ActionStudyKey`
- `SkippedAction`

`OptimizationRequest` 목표:

```python
action: ActionCandidate
supporting_labels: list[str]
supporting_probes: list[str]
opposing_labels: list[str]
search_space: dict[str, list[Any]]
target_metrics: list[str]
```

Deprecated 대상:

```text
failure_label
related_failure_labels
candidates: list[PrescriptionCandidate]
```

`OptimizationHistoryItem` 추가:

```python
action_key: str
action_attempt_key: ActionAttemptKey | None
action_study_key: ActionStudyKey | None
supporting_labels: list[str]
supporting_probes: list[str]
opposing_labels: list[str]
candidate_value: Any
resolved_labels: list[str]
remaining_labels: list[str]
```

`OptimizationReport` 추가:

```python
selected_action: str | None
supporting_labels: list[str]
supporting_probe_count: int
opposing_labels: list[str]
score_breakdown: dict[str, Any]
```

### 6.7 `agents/optimize/planner.py`

최종적으로 얇은 orchestration 계층으로 축소한다.

유지:

- report 없음/통과/manual/actionable 분기
- preliminary 제외
- manual label 보존
- request/decision ID 생성
- backend와 max trials 선택

제거 또는 이동:

- label score와 label ranking
- 대표 label 선택
- label/prescription blacklist
- label 소유 prescription 순서
- 후보값 수학 함수

새 흐름:

```python
manual, actionable = split_findings(...)
decision = decide_mode(...)
groups = group_findings_by_label(actionable)
supports = build_action_supports(groups, state)
candidates = aggregate_action_candidates(...)
selected = select_action(candidates)
request = build_action_request(selected, state)
```

Request metadata:

```text
action score/breakdown
supporting labels/probes
opposing labels
candidate provenance
baseline metrics/config
runtime capabilities
chunk precheck context
```

### 6.8 `agents/optimize/optimizer.py`

Optimizer는 action 선택을 다시 하지 않고 실행 안전성만 검증한다.

변경:

1. 여러 prescription 순회 제거
2. 선택된 action 하나 준비
3. eligibility 정책으로 재검증
4. skipped metadata를 action key로 기록
5. patch metadata에 attempt/study fingerprint 기록
6. result에 selected action 저장
7. 모든 backend가 같은 action metadata 보존
8. embedding recreate 안전 플래그 유지
9. chunk prescreener recoverable fallback 유지
10. multi-axis 방어 유지

### 6.9 `agents/optimize/agent.py`

핵심 전환 지점이다.

#### 로그

- 선택한 label 대신 선택한 action 출력
- supporting/opposing label과 probe 수 출력
- score breakdown 출력
- deferred action을 action key로 출력

#### 상태와 제외

```text
blocked_action_attempts
completed_action_studies
unjudgeable action attempts
visit-local rejected actions
```

#### Iteration

- `last_failure_label` 비교 제거
- 실제 새 ActionStudy 적용 시 증가
- 같은 study 내부 후보 순회는 추가 증가 없음
- baseline/no-op이면 증가 없음

#### Pending/history

- pending item에 action과 support snapshot 전달
- rollback 시 exact attempt 차단
- sweep 완료 시 study key 완료

#### Active study

- top-k 전용 표현을 단일 축 action study로 일반화
- 시작 당시 action/support/search space snapshot 유지
- study 종료 후 새 진단으로 action 재집계

#### Reranker guardrail

Prescription ID 분기를 action key로 교체:

```text
enable_reranker
widen_rerank_candidates
```

→

```text
reranker.enabled:set:true
reranker.candidate_count:increase
```

보존:

- runtime unknown/unavailable은 품질 blacklist 제외
- disabled 상태에서 candidate count 확대 금지
- reranker applied 불완전 시 unjudgeable rollback
- low-rank 감소와 composite 상승 시 precision floor 완화
- 같은 runtime 오류 무한 반복 방지

#### Oscillation 방지

- exact transition 차단
- inverse transition 즉시 선택 시 방문 config 확인
- 개선 근거 없는 왕복 차단
- rollback cache와 reindex OR 정책 유지

### 6.10 `agents/optimize/history.py`

변경:

1. pending item이 action/support snapshot을 받음
2. `last_failure_label` 대신 `last_action_key`
3. before/after confirmed label/probe snapshot
4. finalize 시 resolved/remaining label 계산
5. attempt/study fingerprint helper
6. action transition 재방문 helper
7. 기존 판정 수학과 floor 유지
8. 무거운 before report 참조 제거 시점 유지

### 6.11 `agents/optimize/reporter.py`

사용자 설명을 대표 label 하나에서 action 합의 중심으로 바꾼다.

표시:

- action key와 설명
- supporting labels
- 고유 probe 수
- opposing labels와 충돌 처리
- action score breakdown
- config before/after
- target metrics
- resolved/remaining labels
- keep/rollback 근거
- runtime deferred actions

실제 selected action을 직접 읽고 첫 candidate로 결과를 추측하지 않는다.

### 6.12 `agents/optimize/config_mapper.py`

핵심 mapping은 유지한다.

추가 검증:

- catalog canonical path와 mapper 일치
- hybrid는 boolean이 아니라 canonical `"hybrid"` 값 사용
- action metadata를 ConfigDiff에 보존
- unknown path는 eligibility에서 미리 차단

### 6.13 `agents/optimize/adapters/internal_adapter.py`

핵심 sweep 알고리즘은 유지한다.

변경:

- 대표 failure label 의존 제거
- action key와 supporting labels를 trial metadata에 전달
- target metric union의 objective 선택 정책 명시
- action study fingerprint와 trial fingerprint 분리
- search space 단일 축 검증 유지

### 6.14 `agents/optimize/adapters/chunk_prescreener.py`

변경:

- 여러 support에서 합친 span/document 입력
- 동일 span dedupe
- action key와 candidate provenance 기록
- missing context/dependency fallback 유지
- 실제 precheck failure 구분 유지

### 6.15 `agents/optimize/adapters/ragbuilder_adapter.py`

Payload 변경:

```text
failure_label
related_failure_labels
```

→

```text
action_key
supporting_labels
supporting_probes
opposing_labels
```

외부 호환이 필요하면 대표 label은 설명용 compatibility field로만 유지한다.

보존:

- 단일 optimized stage
- objective 지원 검증
- strict hybrid 계약
- external result validation
- rules fallback

### 6.16 `core/state.py`

추가 권장 필드:

```python
blocked_action_attempts: set = field(default_factory=set)
completed_action_studies: set = field(default_factory=set)
```

기존:

```python
blacklist
completed_prescriptions
```

마이그레이션:

1. 새 필드 추가
2. agent가 새 필드 사용
3. 기존 state compatibility converter
4. 소비처 전환 후 기존 필드 제거 여부 합의

Iteration 주석을 label 처리 단계에서 ActionStudy 적용 횟수로 변경한다.
`graph.py`는 수정하지 않는다.

### 6.17 `agents/serve/report_view.py`

변경:

- 치료 경과 카드에 action key 표시
- supporting labels 표시
- resolved/remaining labels 구분
- action score와 고유 probe 수 drill-down
- 구버전 history fallback

### 6.18 `agents/serve/web_api.py`

변경:

- Optimize ticker에서 action key 표시
- 구버전 `selected_prescription_id` fallback
- API 응답에 action metadata 포함

### 6.19 문서

수정 대상:

- 루트 `AGENTS.md`
- `agents/optimize/AGENTS.md`
- `agents/optimize/README.md`
- `agents/optimize/CONTEXT.md`
- `agents/optimize/PROGRESS.md`
- `agents/optimize/OPTIMIZER_IMPLEMENTATION_PLAN.md`
- `agents/optimize/PARAM_TUNING_PROPOSAL.md`

필수 변경:

- 최상위 label 하나를 고르는 기존 계약 교체
- action catalog를 처방 변경의 단일 진실 원천으로 명시
- blacklist/completed identity 교체
- iteration 의미 교체
- ActionCandidate와 support 설명
- A > C > B 정책을 action selector 정책으로 이동
- 기존 계획 문서에 superseded 표시 또는 새 문서 링크

---

## 7. 테스트 계획

### 7.1 새 테스트: `tests/test_action_catalog.py`

필수:

1. action key 유일성
2. 동일 path/operation/value 중복 금지
3. 모든 label action reference가 catalog에 존재
4. 기본적으로 한 action 한 config 축
5. reindex/cost/capability와 path 정책 일치
6. ready action은 mapping 또는 blocked reason 보유
7. 비canonical 값 등록 방지

### 7.2 새 테스트: `tests/test_action_aggregator.py`

필수:

1. 다른 prescription ID의 동일 top-k 증가 통합
2. 같은 probe의 여러 label을 한 번만 집계
3. 서로 다른 probe 정상 합산
4. preliminary Finding 제외
5. manual label 투표 제외
6. applies_when 불일치 제외
7. 실행 불가능 action 사전 제외
8. action 자체 비용 사용
9. A > C > B tier
10. 같은 tier에서 높은 score 선택
11. deterministic tie-break
12. 증가/감소 근소 충돌 보류
13. grounded action 우선 정책
14. supporting/opposing label 기록
15. target metrics stable union

### 7.3 `tests/test_planner.py`

변경:

- label ranking 테스트를 action ranking으로 교체
- report 없음/already optimal/manual 분기 유지
- 여러 label이 같은 action을 지지할 때 request 하나 생성
- supporting labels/probes/score breakdown
- 후보 provenance union
- evidence analysis memoization
- 여러 support span의 precheck context union
- 실행 가능한 action 없음 사유
- exact attempt 차단과 다른 candidate 허용
- completed study 재시작 방지

### 7.4 새 테스트: `tests/test_candidate_values.py`

기존 planner 후보 수학 테스트를 이동한다.

추가:

- 여러 support 후보 union/dedupe
- 방향에 어긋난 값 제거
- baseline no-op 제거
- constraint 후 stable ordering
- candidate budget
- candidate별 support provenance
- 정상 chunk 후보 보존
- dead-zone boundary clamp
- invalid/missing evidence fallback

### 7.5 `tests/test_optimizer.py`

변경:

- ActionCandidate fixture
- action 하나만 준비
- skipped metadata action key
- capability/runtime prerequisite
- exact search space 검증
- 모든 backend action metadata 보존
- chunk recoverable fallback
- 실제 precheck failure 미은폐
- request/action 불변성

### 7.6 `tests/test_optimize_agent.py`

필수:

1. 새 action 적용 시 iteration 증가
2. 같은 action internal sweep은 추가 증가 없음
3. 새 action이면 label이 같아도 증가
4. 같은 action의 새 baseline study 증가
5. baseline no-op은 예산 미소비
6. rollback exact attempt 차단
7. 다른 candidate/baseline 허용
8. inverse transition 왕복 방지
9. unjudgeable과 품질 blacklist 분리
10. visit limit에서 baseline 복원
11. pending finalize와 gate pass 유지
12. rollback 후 reindex 요구 보존
13. support snapshot history 기록
14. resolved/remaining label 계산
15. 모든 경로 동일 state 반환

### 7.7 `tests/test_enable_reranker.py`

기존 guardrail을 action key 기준으로 이식한다.

필수:

- reranker enable action
- candidate count action
- unavailable/unknown runtime deferred
- reranker disabled prerequisite
- incomplete runtime unjudgeable
- precision floor 단독 완화
- low-rank 감소 조건
- 무한 재시도 방지
- runtime metadata report 전달

### 7.8 Chunk 통합 테스트

대상:

- `tests/test_chunk_grounding_integration.py`
- `tests/test_chunk_overlap_grounding.py`
- `tests/test_chunk_prescreener.py`
- `tests/test_evidence_window.py`

추가:

- 서로 다른 label span을 같은 action에 통합
- 동일 span dedupe
- 후보값 provenance
- union context 기반 prescreener
- optional dependency 부재 fallback
- actual failure는 failed 유지

### 7.9 Adapter 테스트

대상:

- `tests/test_internal_adapter.py`
- `tests/test_ragbuilder_adapter.py`

추가:

- action key/support payload
- target metrics union
- 대표 label compatibility가 선택에 영향 없음
- trial fingerprint와 action study fingerprint 구분

### 7.10 History/Serve 테스트

대상:

- 신규 `tests/test_action_history.py`
- `tests/test_report_view.py`
- `tests/test_serve_api.py`
- `tests/test_pipeline.py`
- `tests/test_rollback_cache.py`

필수:

- 구버전 history fallback
- action 중심 치료 경과
- supporting/resolved/remaining label 표시
- rollback config 복원
- action fingerprint 직렬화
- Index/Eval cache key 복원
- graph route 무변경 동작

### 7.11 Invariant 테스트

1. 선택 action은 catalog에 존재
2. 선택 action은 eligibility 통과
3. search space는 단일 canonical 축
4. 모든 후보값은 constraint 안
5. 방향 후보는 baseline 대비 올바른 방향
6. 같은 probe는 score에 최대 한 번 기여
7. score와 breakdown 합계 일치
8. 동일 입력은 항상 같은 선택
9. blocked exact attempt 재선택 금지
10. 새 baseline/candidate 과도한 차단 금지
11. 충돌 정책 미통과 축 자동 적용 금지
12. 적용/rollback 후 config는 허용 snapshot
13. **B그룹 action은 실행 가능한 A/C action이 남아 있는 한 선택되지 않는다** (§4.3)
14. **같은 축의 여러 고정값은 단일 action의 후보로 통합된다** — 서로 경쟁하지 않는다(§3.1)
15. **어떤 축도 "지지가 동일해서" 영구 보류되지 않는다** — starvation 부재
16. **개선 마진 미달 시 keep되지 않는다** — judge와 sweep이 같은 임계를 쓴다(§5.11)

---

## 8. 구현 단계와 권장 커밋 단위

### 단계 0. Baseline 고정

- 최신 main Optimize 테스트 실행
- ready/action inventory fixture 저장
- 기존 선택 결과 characterization test
- chunk/gate/reranker 핵심 회귀 확인

**⚠️ 동점 입력은 characterization 대상에서 제외한다.** 현재 `_group_by_label`은 dict
삽입 순서에 의존하는데, §4.2·§7.11이 결정적 tie-break(`action_key` 사전순 등)를
새로 도입한다. 동점 케이스에서는 선택이 바뀌는 것이 **정상**이므로, 이를 미리
구분해 두지 않으면 단계 1·2의 "기존 동작 변화 없음" 완료 조건과 충돌한다.

- 동점 입력: 변화 허용 + 결정성만 검증
- 비동점 입력: 기존 선택과 완전 일치 요구

완료 조건:

- 변경 전 baseline 기록
- 환경 실패와 코드 실패 구분
- 동점/비동점 fixture 분리 완료

### 단계 1. Schema와 action catalog

- 새 dataclass
- action catalog
- label rule이 action key 참조
- compatibility view로 기존 planner 유지

완료 조건:

- 기존 동작 변화 없음
- catalog/rule 무결성 테스트 통과

### 단계 2. Candidate value 로직 분리

- planner에서 candidate module로 함수 이동
- strategy registry
- 기존 결과와 동일한 characterization test

완료 조건:

- 기존 planner/chunk 테스트 동일

### 단계 3. Action aggregation shadow mode

- support/candidate/score 구현
- 실제 적용은 기존 선택 유지
- legacy 선택과 action 선택 비교 로그

이 단계는 최종 절충안이 아니라 구현 검증을 위한 임시 상태다.

#### 🛑 중단 기준 (신설)

이 작업은 §1에서 밝혔듯 **재료(`cost`/`confidence`)가 비어 있는 상태의 구조 선투자**다.
그 판단이 틀렸을 경우 되돌릴 수 있도록, shadow mode 관측 결과에 중단 기준을 둔다.

shadow mode에서 legacy 선택과 action 선택을 비교해 다음을 기록한다.

```text
총 비교 횟수
선택이 달라진 횟수
그중 "공유 지지 합산" 때문에 달라진 횟수
그중 conflict 보류로 축이 바뀐 횟수
```

판단:

| 관측 | 조치 |
| --- | --- |
| 선택이 달라진 사례가 **0건** | 전환 보류. 구조만 남기고 선택 로직은 legacy 유지 |
| 달라졌으나 **전부 tie-break 차이** | 전환 보류. 실익 없음이 확인된 것 |
| 공유 합산으로 달라진 사례 **1건 이상** | 계속 진행 |

0건이어도 §6.1~§6.4의 catalog·eligibility 추출은 **유지할 가치가 있다**(중복 선언
제거, 정책 단일화). 중단은 "선택 로직 전환"에만 적용한다.

완료 조건:

- deterministic ranking
- probe dedupe와 conflict 테스트
- shadow mode 비교 로그 수집 및 중단 기준 판정

### 단계 4. Planner 선택 중심 전환

- ActionCandidate 선택
- action 중심 OptimizationRequest
- 실행 불가능 action 사전 제외
- backend 선택과 chunk context union

완료 조건:

- label 우선 선택 함수가 실행 경로에서 제거
- planner/action/optimizer 테스트 통과

### 단계 5. History·blacklist·iteration 전환

- state 새 필드
- action attempt/study fingerprint
- active study와 rollback 전환
- action study 기준 iteration
- oscillation 방지

완료 조건:

- label/prescription pair가 실행 제어에 사용되지 않음
- rollback/sweep/visit-limit/cache 테스트 통과

### 단계 6. Reranker·adapter guardrail 이관

- prescription ID 특수 분기 제거
- action key 분기
- RAGBuilder/internal/chunk metadata 변경

완료 조건:

- 기존 reranker guardrail 테스트 통과
- runtime deferred 동작 유지

### 단계 7. Reporter·Serve 전환

- action 합의 설명
- support/opposition/resolution 표시
- 구버전 history fallback

완료 조건:

- CLI, report, 웹 치료 경과가 같은 action 표시

### 단계 8. Compatibility 제거와 문서 갱신

- `PrescriptionCandidate` 제거
- legacy request/history field 정리
- compatibility 제거 여부 확정
- 관련 문서 갱신

완료 조건:

- 실행 코드에서 기존 prescription pair 제어 참조 없음
- 전체 테스트와 compileall 통과

---

## 9. 검증 명령

빠른 단위 검증:

```bash
python3 -m unittest \
  tests.test_action_catalog \
  tests.test_action_aggregator \
  tests.test_candidate_values \
  tests.test_planner \
  tests.test_optimizer
```

Optimize 전체:

```bash
python3 -m unittest \
  tests.test_config_mapper \
  tests.test_evidence_window \
  tests.test_chunk_prescreener \
  tests.test_chunk_grounding_integration \
  tests.test_chunk_overlap_grounding \
  tests.test_internal_adapter \
  tests.test_ragbuilder_adapter \
  tests.test_enable_reranker \
  tests.test_optimize_agent \
  tests.test_action_history
```

통합·소비처:

```bash
python3 -m unittest \
  tests.test_index_unit \
  tests.test_retriever_rag \
  tests.test_rollback_cache \
  tests.test_report_view \
  tests.test_serve_api \
  tests.test_pipeline
```

문법과 diff:

```bash
python3 -m compileall -q agents core tests
git diff --check
```

전체:

```bash
python3 -m unittest discover -s tests
```

외부 API·LLM·Qdrant가 필요한 테스트는 기본 단위 테스트와 분리하고, 실패 시 환경 문제와
코드 회귀를 구분해 기록한다.

---

## 10. 완료 기준

### 구조

- [ ] Label이 실행 순서를 직접 결정하지 않는다.
- [ ] config 변경은 action catalog에 한 번만 정의된다.
- [ ] 동일 action은 이름이 달라도 하나로 집계된다.
- [ ] 실행·history·blacklist identity가 action 기반이다.
- [ ] 모든 활성 label의 support가 선택에 고려된다.

### 안전

- [ ] 한 번에 config 축 하나만 변경한다.
- [ ] 실행 불가능 action은 경쟁 전에 제외된다.
- [ ] 증가/감소 충돌을 임의로 선택하지 않는다.
- [ ] **B그룹 action이 A/C를 앞질러 선택되지 않는다.**
- [ ] **개선 마진 미달을 개선으로 판정하지 않는다.**
- [ ] **지지가 동일한 축이 영구 보류되지 않는다.**
- [ ] exact 실패 transition은 재시도하지 않는다.
- [ ] 다른 baseline/candidate를 과도하게 차단하지 않는다.
- [ ] rollback cache와 reindex 요구가 보존된다.
- [ ] 절대 Optimize visit limit가 유지된다.

### 품질 판정

- [ ] 사용자 pipeline Eval 전에는 improved를 확정하지 않는다.
- [ ] composite/floor 기반 keep/rollback을 유지한다.
- [ ] supporting label과 resolved label을 구분한다.
- [ ] internal sweep은 action study 하나로 처리한다.

### 회귀 방지

- [ ] 기존 chunk 후보/prescreener 테스트 통과
- [ ] gate/pending finalize 테스트 통과
- [ ] reranker guardrail 테스트 통과
- [ ] Serve가 action history를 표시
- [ ] `graph.py` 무수정

### 설명 가능성

- [ ] action 선택 score breakdown 제공
- [ ] supporting label과 고유 probe 수 제공
- [ ] opposing label과 충돌 처리 이유 제공
- [ ] config before/after와 재색인 여부 제공
- [ ] resolved/remaining label 제공

---

## 11. 주요 위험과 완화

| 위험 | 영향 | 완화 |
| --- | --- | --- |
| 같은 probe의 다중 label 과대투표 | generic action 독점 | probe별 최대 confidence 한 번 |
| **기준 커밋 낡음** | 잘못된 전제 위 설계 | 재조사 기준 `23a6fe7` 명시, 착수 전 재확인 |
| **동점 action 영구 교착(starvation)** | **축이 한 번도 시도되지 않음** | 고정값을 key에서 빼고 후보로 통합(§3.1) |
| **측정 노이즈로 인한 가짜 개선** | **왕복이 롤백 안전망을 통과** | 최소 개선 마진 2점, judge·sweep 동시 적용(§5.11) |
| **마진 과대 → 후보 고갈 조기 종료** | 개선 기회 상실 | 마진 탈락 로그 기록 후 사후 조정(§5.11) |
| 증가/감소 action 왕복 | iteration 소모 | 근거값 `direction_conflict` 선차단 + conflict 정책(§4.4) |
| action 전체 과도한 blacklist | 유효 후보 차단 | baseline/candidate 단위 attempt key |
| label별 후보 근거 유실 | 잘못된 숫자 선택 | support별 provenance 보존 |
| capability 미지원 action 득표 | 실행 가능 action starvation | 점수 계산 전 eligibility |
| target metric union 왜곡 | objective 모호 | composite를 primary로 유지 |
| **B action 조기 실행** | **garbage-in tuning (B그룹 ready로 실재화)** | A > C > B hard tier를 **불변조건**으로(§4.3) |
| sweep 미지원 축의 다후보 | 후보 일부 미검증 | `ActionStudyKey`로 방문 간 이어서 소진(§2.5·§3.1) |
| reranker 예외 유실 | 잘못된 판정 | action key 기반 guardrail 테스트 |
| chunk 경계 회귀 | 후보 dead zone | 기존 characterization/invariant 테스트 |
| schema 동시 변경 | UI/adapter 파손 | dual-read migration |
| planner 재비대화 | 유지보수 악화 | catalog/aggregator/candidate/eligibility 분리 |

---

## 12. 구현 전 최종 합의 체크리스트

**합의 완료**

- [x] **Action key 형식** — 방향은 key에, **고정값은 key에 넣지 않고 후보값으로**(§3.1)
- [x] **A > C > B hard tier** — 유지, **불변조건으로 격상**(§4.3·§5.4)
- [x] **conflict margin** — 상대 20% **그리고** 절대 probe 2 이상(§4.4)
- [x] **개선 판정 최소 마진** — composite 2점(내부 `0.02`), judge·sweep 양쪽 동일(§5.11)
- [x] **confidence 생산** — 이번 범위 밖. `1.0` fallback 유지(§5.3), 근거는 §1
- [x] **차단 기능 활성화** — 별도 PR. catalog에는 blocked reason으로만 등록(§6.1)
- [x] **`base_cost` 산정** — 현행 이분법(재색인 3 / 런타임 1)을 그대로 이관. 세분화는 별도 작업(§3.1)
- [x] **`graph.py` 무수정 유지** — 라우팅 수정으로는 예산 문제가 해결되지 않음(§5.8)

**미합의 — 착수 전 결정 필요**

- [ ] **iteration 미소비 경로의 수렴 증명** — §5.8의 5개 확인 항목 표를 채울 것 (최우선)
- [ ] `chunker.strategy` 2후보 처리: §3.1의 (a)/(b)/(c) 중 택1 — **권장 (b)**
- [ ] `rules.py` 재작성 시점 — 단계 1에서 전면 교체 vs 단계 4까지 alias로 버티기
      (현재 flat 20 / canonical 20으로 혼재, 핵심 축 4개가 flat)
- [ ] 고유 probe 기반 투표 (§4.2 공식 확정)
- [ ] 후보값 union과 candidate budget
- [ ] exact attempt blacklist 범위
- [ ] inverse transition 차단 범위 — 정확 전이만 vs 축 전체 보류
- [ ] 새 state field와 legacy 호환 기간
- [ ] `selected_prescription_id` 제거 시점
- [ ] resolved label 계산 방식
- [ ] UI의 supporting/opposing label 표현
- [ ] shadow mode 중단 기준 수치 확정 (§8 단계 3)

권장 범위:

- 이번 작업은 선택·상태·이력 중심을 action으로 옮기는 데 집중한다.
- 현재 차단된 기능을 실제 활성화하는 작업은 별도 PR로 둔다.
- Finding confidence 생산도 별도 Eval 계약으로 둔다.
- 기존 후보값 수학과 keep/rollback 임계값은 전환 중 변경하지 않는다.

이렇게 분리하면 action 중심 구조 변경의 효과와 신규 기능 활성화의 효과를 따로 검증할 수
있다.
