# Action-Centered Optimizer 전환 구현 계획

> 상태: 구현 전 설계·작업 계획
> 작업 브랜치: `feature/action-centered-optimizer`
> 최초 조사 기준: `origin/main`의 `a35677f`
> 목표: Optimize의 선택 중심을 failure label에서 실제 실행 가능한 config action으로 완전히 옮긴다.

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

| Canonical action | 지지하는 대표 label |
| --- | --- |
| `retriever.top_k:increase` | `retrieval_missing_gold`, `retrieval_incomplete_enumeration` |
| `retriever.top_k:decrease` | `too_long_context` |
| `chunker.chunk_size:increase` | `retrieval_missing_gold`, `chunking_context_mismatch`, `chunking_overchunking` |
| `chunker.chunk_size:decrease` | `retrieval_semantic_mismatch`, `too_long_context` |
| `chunker.chunk_overlap:increase` | `retrieval_missing_gold`, `chunking_context_mismatch` |
| `reranker.enabled:set:true` | `retrieval_low_rank` |
| `reranker.candidate_count:increase` | `retrieval_low_rank` |

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

| Action | 대표 차단 이유 |
| --- | --- |
| `retriever.search_type:set:hybrid` | runtime/capability 확인 필요 |
| `embedding.model:set:<model>` | embedding model capability 및 재색인 필요 |
| `chunker.strategy:set:recursive_sentence` | state mapping/capability 확인 필요 |
| `query_rewrite:set:expand` | state mapping 또는 소비 노드 부재 |
| `mmr:set:true` | canonical mapping/소비 경로 확인 필요 |
| `adaptive_retrieval:set:true` | state mapping 또는 소비 노드 부재 |
| `context.compression.enabled:set:true` | capability 및 소비 노드 부재 |

Action 집계 시 rule status만 보지 않고, 현재 baseline과 runtime에서 실제 실행 가능한지
확인한 뒤 점수를 계산해야 한다.

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
reranker.enabled:set:true
retriever.search_type:set:hybrid
embedding.model:set:sentence-transformers/all-MiniLM-L6-v2
```

원칙:

- action key에는 label이나 기존 prescription ID를 넣지 않는다.
- 방향 action의 실제 후보 숫자는 key에 넣지 않는다.
- 고정값 교체 action은 stable value를 key에 포함한다.
- 한 action은 canonical config 축 하나만 소유한다.
- 보조 안전 플래그는 patch에 추가할 수 있지만 action의 정체성은 하나로 유지한다.

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

실험 데이터가 쌓이면 hard tier를 group multiplier로 바꿀 수 있도록 정책 객체로 분리한다.

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

### 5.4 Group 인과

선택지:

- A > C > B hard tier
- group multiplier
- B action 전에 A/C 해결 필수

권장:

- 첫 구현에서는 hard tier 유지
- 구조 전환과 그룹 정책 변경을 한 번에 섞지 않음

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

우선 등록:

```text
retriever.top_k:increase
retriever.top_k:decrease
chunker.chunk_size:increase
chunker.chunk_size:decrease
chunker.chunk_overlap:increase
reranker.enabled:set:true
reranker.candidate_count:increase
```

후속 등록:

```text
retriever.search_type:set:hybrid
embedding.model:set:<model>
chunker.strategy:set:recursive_sentence
```

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

이동 또는 공유:

- constraints
- capabilities
- path capabilities
- state mappable paths
- backend supported paths
- runtime capability merge
- candidate value filtering
- no-op filtering

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

---

## 8. 구현 단계와 권장 커밋 단위

### 단계 0. Baseline 고정

- 최신 main Optimize 테스트 실행
- ready/action inventory fixture 저장
- 기존 선택 결과 characterization test
- chunk/gate/reranker 핵심 회귀 확인

완료 조건:

- 변경 전 baseline 기록
- 환경 실패와 코드 실패 구분

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

완료 조건:

- deterministic ranking
- probe dedupe와 conflict 테스트

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
| 증가/감소 action 왕복 | iteration 소모 | conflict 정책과 transition fingerprint |
| action 전체 과도한 blacklist | 유효 후보 차단 | baseline/candidate 단위 attempt key |
| label별 후보 근거 유실 | 잘못된 숫자 선택 | support별 provenance 보존 |
| capability 미지원 action 득표 | 실행 가능 action starvation | 점수 계산 전 eligibility |
| target metric union 왜곡 | objective 모호 | composite를 primary로 유지 |
| B action 조기 실행 | garbage-in tuning | 초기 A > C > B hard tier |
| reranker 예외 유실 | 잘못된 판정 | action key 기반 guardrail 테스트 |
| chunk 경계 회귀 | 후보 dead zone | 기존 characterization/invariant 테스트 |
| schema 동시 변경 | UI/adapter 파손 | dual-read migration |
| planner 재비대화 | 유지보수 악화 | catalog/aggregator/candidate/eligibility 분리 |

---

## 12. 구현 전 최종 합의 체크리스트

- [ ] Action key 형식
- [ ] 고유 probe 기반 투표
- [ ] 초기 A > C > B hard tier 유지 여부
- [ ] conflict margin 및 grounded 우선 정책
- [ ] 후보값 union과 candidate budget
- [ ] iteration을 ActionStudy 횟수로 변경
- [ ] exact attempt blacklist 범위
- [ ] inverse transition 차단 범위
- [ ] 새 state field와 legacy 호환 기간
- [ ] `selected_prescription_id` 제거 시점
- [ ] resolved label 계산 방식
- [ ] UI의 supporting/opposing label 표현
- [ ] hybrid/embedding/chunk strategy 활성화를 별도 작업으로 둘지
- [ ] confidence 생산을 별도 Eval 작업으로 둘지

권장 범위:

- 이번 작업은 선택·상태·이력 중심을 action으로 옮기는 데 집중한다.
- 현재 차단된 기능을 실제 활성화하는 작업은 별도 PR로 둔다.
- Finding confidence 생산도 별도 Eval 계약으로 둔다.
- 기존 후보값 수학과 keep/rollback 임계값은 전환 중 변경하지 않는다.

이렇게 분리하면 action 중심 구조 변경의 효과와 신규 기능 활성화의 효과를 따로 검증할 수
있다.
