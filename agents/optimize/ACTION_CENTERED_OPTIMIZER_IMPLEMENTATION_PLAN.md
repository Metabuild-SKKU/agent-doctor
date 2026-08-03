# Action-Centered Optimizer 전환 구현 계획

> 작업 브랜치: `feature/action-centered-optimizer` · PR #75
> 조사 기준: `origin/main` `885490f` (PR #72 병합 후)
> 진행 관리: **PR #75 본문의 task list**. 이 문서는 설계와 결정 근거만 담는다.

---

## 1. 목적

Optimize의 선택 단위를 **failure label에서 실제 config action으로** 옮긴다.

```text
현재   라벨 정렬 → 최상위 라벨 1개 → 그 라벨의 처방을 선언 순서대로 적용
목표   모든 활성 라벨이 지지하는 action 생성 → 같은 config 변경을 하나로 통합
       → action끼리 경쟁 → 1개 선택 → 적용 → Eval → 유지/롤백 → 재집계
```

label은 **진단 근거·영향 probe·목표 metric**을 제공하고 실행 순서를 소유하지 않는다.
action은 **우선순위 경쟁·후보값·적용·이력·차단**의 중심이 된다.

### 1.1 이 작업의 성격 — 구조 선투자

이슈 #67은 이 전환을 반대했다. 근거를 코드로 재검증한 결과다.

| #67의 근거 | 재조사 결과 |
| --- | --- |
| ① `cost`/`confidence`가 비어 정렬 재료가 없다 | **여전히 사실** (`cost` 51/51 None, `confidence` 25/25 None) |
| ② 라벨 간 공유 처방이 적다 | **해소됨** — 실행 가능 18개 중 12개가 복수 라벨 지지 |
| ③ A>C>B 인과가 깨진다 | **위험 증가** — B그룹이 ready가 되어 실제로 A와 경쟁 |
| ④ 측정이 병목이다 | **대응 착수** — 선행 PR ① |

> **성능 개선이 아니라 구조 선투자다.** ①이 빈 동안 점수는
> `고유 probe 수 ÷ action 비용`으로 축약되어 현재 `planner._score`와 재료가 같다.
> 선택 결과가 크게 달라지리라 기대해선 안 된다. 지금 하는 이유는 ②가 해소돼 통합
> 실익이 생겼고, `confidence`가 채워질 때 **코드 변경 없이 반영될 자리**를 만들기
> 위해서다. 판단이 틀렸을 경우를 위해 §8 단계 3에 중단 기준을 둔다.

### 1.2 유지해야 하는 안전 원칙

1. 한 번에 config 축 하나만 변경한다.
2. 개선 판정은 사용자 pipeline의 Eval 결과로만 한다.
3. 하한선 위반 또는 개선 마진 미달이면 rollback한다.
4. `graph.py`는 수정하지 않는다.
5. 기존 chunk 후보 경계·prescreener fallback·pass gate·pending finalize를 보존한다.
6. reranker 실행 검증과 precision floor 완화를 보존한다.
7. 실행 불가능 action은 **점수 계산 전에** 제외한다.
8. 같은 probe에서 파생된 여러 label을 중복 투표로 세지 않는다.

---

## 2. 현황

> ⚠️ 아래 수치는 **`origin/main` 머지 시점(#70·#72·#73·#76·#78 반영)** 집계다.
> 초판 이후 네 번 갱신했다. 착수 후에도 main이 움직이면 §2.2의 집계를 다시 돌린다.

전체 라벨 **30개** (ready 19 / draft 7 / manual 4).
실행 가능 action **18개**, 그중 **12개가 복수 라벨 지지**. 차단 **9개**.

| action | 지지 | tier | 재색인 | backend | 비고 |
| --- | --- | --- | --- | --- | --- |
| `chunker.chunk_size:increase` | 3 | A | ✔ | internal | |
| `chunker.chunk_size:decrease` | 3 | A·C | ✔ | internal | 증가와 경쟁 |
| `generation.require_citation:enable` | 3 | B | | rules | |
| `generation.abstention_strict:enable` | 3 | B | | rules | |
| `chunker.chunk_overlap:increase` | 2 | A | ✔ | internal | |
| `chunker.strategy:replace` | 2 | A | ✔ | **internal** | 후보 2개 — #73으로 sweep 지원 |
| `context.compression.enabled:enable` | 2 | C | | rules | **신규 활성화** |
| `generation.temperature:decrease` | 2 | B | | rules | |
| `retriever.mmr:enable` | 2 | A·C | | rules | |
| `retriever.top_k:increase` | 2 | A | | internal | |
| `retriever.top_k:decrease` | 2 | C | | internal | 증가와 경쟁 |
| `reranker.enabled:enable` | 2 | A | | rules | |
| `reranker.enabled:disable` | 1 | A | | rules | enable과 배타(§4.4) |
| `reranker.candidate_count:increase` | 1 | A | | rules | |
| `retriever.hybrid_dense_weight:adjust` | 1 | A | | internal | 방향이 실측으로 결정(§3.2) |
| `retriever.search_type:replace` | 1 | A | | rules | |
| `generation.completeness_mode:enable` | 1 | B | | rules | |
| `generation.restate_question:enable` | 1 | B | | rules | |

**차단된 action 9개** — catalog에 blocked reason과 함께 등록한다.

| 사유 | action | 해제 조건 |
| --- | --- | --- |
| `not_state_mappable` | `query_rewrite`, `adaptive_retrieval`, `answer_checklist_review`, `conflict_resolution_prompt`, `context_ordering`, `noise_filter`, `reranker_model` | mapper 계약 + 소비 노드 추가 |
| `capability_off` | `embedding.model`, `generation.model` | `DEFAULT_CAPABILITIES` 값 변경 (둘 다 "검증된 후보 부재") |

머지 시점에 달라진 것:

- `context.compression.enabled` — `capability_off` → **활성화**
  (`too_long_context`, `context_noise_interference`가 지지)
- `reranker_model:replace` — 차단 목록에 **신규**
  (`retrieval_reranker_demotion`의 두 번째 처방, mapper 미등록)
- `bad_gold_chunk` 라벨 추가(#70) — D그룹 `manual`이라 **action 지형에 영향 없음**

### 2.1 sweep 지원 축은 5개

```python
BACKEND_SUPPORTED_PATHS["internal"] = {
    "retriever.top_k", "retriever.hybrid_dense_weight",
    "chunker.chunk_size", "chunker.chunk_overlap",
    "chunker.strategy",          # ← #73으로 추가됨
}
```

나머지 13개는 `rules` backend로 **1회 적용**된다.

### 2.2 집계표를 손으로 유지하지 않는다

문서가 세 번 낡은 원인이다. 아래를 계산하는 스크립트를 `tools/`에 두고 문서는 그
출력을 참조한다. §7의 catalog 무결성 테스트 입력으로도 쓴다.

```text
ready 라벨 → canonicalize_path 적용 → action key
STATE_MAPPABLE_PATHS / PATH_CAPABILITIES / DEFAULT_CAPABILITIES 통과 여부
action별 지지 label·tier, 같은 축의 경쟁 관계
```

---

## 3. 도메인 모델

### 3.1 ActionDefinition — 실제 config 변경의 정적 정의

```python
@dataclass(frozen=True)
class ActionDefinition:
    key: str
    canonical_path: str
    operation: ActionOperation
    reindex_required: bool
    base_cost: float
    capability: str | None
    conflict_family: str
    description: str
    blocked_reason: str | None = None      # not_state_mappable | capability_off
    fixed_value: Any | None = None
    prerequisites: tuple[str, ...] = ()
```

`base_cost`는 **현행 이분법을 그대로 이관**한다. `confidence`가 빈 상태에서 유일한
차별 재료이므로 근거 없는 새 숫자를 만들지 않는다.

```python
base_cost = 3.0 if reindex_required else 1.0   # 판정 근거는 optimizer.REINDEX_PATHS
```

라벨의 첫 처방이 아니라 **action 자신의** 재색인 여부를 쓴다(초판이 지적한 왜곡의
수정). `score_breakdown`에 `cost_source: "reindex_flag"`를 기록해 나중에 실측 기반으로
바뀔 때 이력에서 구분한다.

> 이 이분법에서는 같은 축의 반대 방향과, 재색인 없는 A·B 그룹 action의 **분모가 같아
> 약분된다.** 그래서 tier(§4.3)와 eligibility(§4.4)가 필요하다 — cost로는 못 가른다.

### 3.2 action key 규칙

```text
<canonical_path>:<operation>
```

| operation | 언제 | 값 처리 | 예 |
| --- | --- | --- | --- |
| `increase` / `decrease` | 방향이 고정 | 후보값으로 | `retriever.top_k:increase` |
| `enable` / `disable` | boolean 축 | 없음 | `reranker.enabled:enable` |
| `replace` | 고정값 교체 | **후보값으로** | `chunker.strategy:replace` |
| `adjust` | 방향·폭이 진단 실측으로 결정 | 후보값 + provenance | `retriever.hybrid_dense_weight:adjust` |

원칙:

- action key에 label이나 기존 prescription ID를 넣지 않는다.
- **고정값을 key에 넣지 않는다.** 값은 후보값으로 넘긴다.
- 한 action은 canonical 축 하나만 소유한다.
- 키 생성 시 반드시 `canonicalize_path`를 거친다.

#### `replace`에서 값을 key에 넣지 않는 이유 — starvation

초판 규칙(`set:<value>`)을 `chunker.strategy`에 적용하면 **그 축이 영원히 선택되지
않는다.**

```text
chunker.strategy:set:recursive_sentence  ← chunking_context_mismatch, retrieval_semantic_mismatch
chunker.strategy:set:markdown_recursive  ← chunking_context_mismatch, retrieval_semantic_mismatch
                                            └ 지지 label 집합이 완전히 동일
```

지지 probe가 같으면 점수가 **수학적으로 항상 같아** §4.4의 보류 규칙에 영구히 걸린다.
`optimizer.py`의 제약 주석이 밝히듯 두 값은 경쟁자가 아니라 **같은 처방의 2후보**다.

```python
"chunker.strategy": {"allowed": ["recursive_sentence", "markdown_recursive"]},
```

정밀도 손실은 없다 — 어떤 값으로 시도했는지는 `ActionAttemptKey`의
`candidate_fingerprint`가 구분한다.

#### `adjust` — 방향이 실측으로 정해지는 축

```python
# rules.py
"retriever.hybrid_dense_weight": "shift_to_favored_channel"
```

Eval이 실측한 `favored_channel`(어느 검색 채널이 gold를 상위에 뒀나)로 방향이 정해진다.
`increase`/`decrease`로 쪼개지 않는 이유는 두 방향이 경쟁 관계가 아니기 때문이다 —
어느 쪽인지는 데이터가 정하고 지지 label도 하나뿐이다.

**추가 구현이 필요 없다.** planner 근거값 계산(`_ground_hybrid_dense_weight`),
mapper 매핑, `DEFAULT_CONSTRAINTS`(`{min:0.1, max:0.9}`),
capability(`hybrid_fusion_weight`), internal sweep 지원까지 이미 존재한다.
`_GROUNDED_ONLY`로 지정돼 있어 **실측 신호가 없거나 채널 투표가 동수면 후보가 만들어지지
않고 자동 제외**된다.

### 3.3 ActionSupport / ActionCandidate

```python
@dataclass
class ActionSupport:                 # 한 label 묶음이 action에 주는 런타임 근거
    action_key: str
    label: str
    group: FailureGroup
    finding_ids: list[str]
    affected_probes: set[str]         # set으로 중복 제거
    confidence: float
    target_metrics: list[str]
    candidate_values: list[Any]
    applies_when: dict[str, Any]
    reason: str
    grounding_metadata: dict[str, Any]


@dataclass
class ActionCandidate:               # 같은 action key를 지지하는 support의 통합 = 선택 단위
    action_key: str
    definition: ActionDefinition
    supports: list[ActionSupport]
    supporting_labels: list[str]
    supporting_probes: set[str]
    opposing_action_keys: list[str]
    search_space: dict[str, list[Any]]
    target_metrics: list[str]
    score: float
    score_breakdown: dict[str, Any]
    status: Literal["ready", "blocked", "conflicted"]
    metadata: dict[str, Any]
```

- 같은 label의 여러 Finding은 먼저 하나의 support로 묶는다.
- preliminary(`confirmed=False`) Finding은 자동 처방에서 제외한다(기존과 동일).
- manual Finding은 투표에 넣지 않고 decision/report에 보존한다.
- `applies_when`을 만족하지 않는 support는 생성하지 않는다.
- 최종적으로 `PrescriptionCandidate`를 제거하고 이것으로 교체한다.

### 3.4 식별자

```python
@dataclass(frozen=True)
class ActionAttemptKey:              # 정확한 전이 — 품질 실패 차단용
    action_key: str
    baseline_fingerprint: str
    candidate_fingerprint: str

@dataclass(frozen=True)
class ActionStudyKey:                # 완료된 sweep 차단용
    action_key: str
    baseline_fingerprint: str
    search_space_fingerprint: str
```

fingerprint 입력: action key + baseline canonical effective config + candidate config +
관련 runtime capability identity.

---

## 4. 선택 알고리즘

```python
def plan(state, exclusions):
    manual, actionable = split_findings(state.report.findings)
    decision = decide_mode(state, actionable, manual)
    if decision.mode != "apply_optimize":
        return None, decision

    grouped    = group_findings_by_label(actionable)
    supports   = build_action_supports(grouped, state)
    candidates = aggregate_action_candidates(supports, state)
    candidates = filter_ineligible_actions(candidates, state, exclusions)   # ← §4.4보다 먼저
    candidates = resolve_action_conflicts(candidates)
    selected   = rank_action_candidates(candidates)[0]
    return build_action_request(selected, state, manual), decision
```

### 4.1 고유 probe 기반 가중 투표

label 수를 그대로 합산하면 같은 probe에서 파생된 여러 label이 여러 표가 된다.
**probe 단위로 한 번만 기여**하게 한다.

```text
probe_support(p, action) = 그 probe에서 action을 지지한 support 중 최대 confidence
coverage_weight(action)  = Σ probe_support(p, action)
action_score             = coverage_weight / action.base_cost
```

`score_breakdown`: `supporting_label_count`, `supporting_probe_count`,
`weighted_probe_support`, `grounded_support_count`, `target_metric_count`,
`cost_source`, `confidence_source`.

label 다양성은 taxonomy label을 많이 만들수록 점수가 커지는 왜곡을 낳으므로 **주 점수가
아닌 tie-breaker**로만 쓴다.

### 4.2 정렬

```text
1. causal_rank 오름차순 (A, C, B)     ← 불변조건
2. action_score 내림차순
3. grounded support 수 내림차순
4. action 비용 오름차순
5. action_key 사전순                  ← 결정성 보장
```

`causal_rank` = 그 action을 지지하는 label 중 가장 높은 우선순위 그룹.

### 4.3 🔒 A > C > B는 불변조건이다

초판은 hard tier를 "초기 권장안"으로 두고 group multiplier로 바꿀 여지를 남겼다.
**B그룹이 ready인 한 선택지로 두면 안 된다.**

```text
[B] generation.require_citation:enable   지지 3 ÷ 비용 1 = 3.0   ← 이긴다
[A] retriever.top_k:increase             지지 2 ÷ 비용 1 = 2.0
```

generation 경로는 재색인이 없어 비용이 `top_k`와 같고, 약분되면 순수 지지 수 싸움이
된다. hard tier가 없으면 **검색이 새는 상태에서 생성 프롬프트를 먼저 손대는 순서
역전이 기본 시나리오**가 된다(garbage-in tuning).

- `causal_rank`가 다르면 점수를 보지 않는다.
- group multiplier는 채택하지 않는다 — 지지가 몰리면 여전히 역전을 허용한다(`10×1 > 2×3`).
- 정책 객체로 분리하되 기본값 교체는 별도 PR과 별도 근거를 요구한다.

### 4.4 ⚠️ 충돌 판정보다 eligibility가 먼저다

순서를 지키지 않으면 **실제로는 경쟁하지 않는 축에 보류 규칙이 걸려 영구 교착**이
발생한다. 이번 검토에서 두 번 발견된 실패 패턴이다.

#### boolean 축은 no-op 필터가 자동 배타한다

```python
# optimizer.py:398
current_value = get_current_value(baseline_config, path)
filtered = [value for value in filtered if value != current_value]
```

| 현재 상태 | `enable` | `disable` | 경쟁 |
| --- | --- | --- | --- |
| 꺼짐 | 유효 | no-op → 제거 | 없음 |
| 켜짐 | no-op → 제거 | 유효 | 없음 |

진단 쪽에서도 배타적이다 — 리랭커가 꺼져 있으면 `retrieval_reranker_demotion`
(리랭커가 gold를 강등시킴) Finding 자체가 생길 수 없다.
**boolean 축에 `conflict_margin_ratio`를 적용하지 않는다.** 적용하면 2:1 지지에서
절대 조건에 걸려 축이 영원히 닫힌다.

#### 방향 축은 근거값 계산이 먼저 차단한다

```python
# planner._ground_chunk_size_candidates
if (direction == "decrease" and target >= current_int) or (
    direction == "increase" and target <= current_int
):
    metadata["status"] = "direction_conflict"
    return None, metadata          # 후보 없음 → 제외
```

gold span 길이 분포(P85)가 방향을 정하므로, 근거가 정상 계산되면 증가·감소가 동시에
링에 오르지 않는다.

#### 충돌 정책이 실제로 필요한 경우

근거 계산이 실패해 방향 키워드 폴백(`×2`/`÷2`)으로 넘어간 때뿐이다
(`missing_gold_spans`, `insufficient_spans`, `invalid_policy`).

```text
1. causal tier가 다르면 높은 tier 우선
2. 같은 tier면 grounded support가 있는 쪽 우선
3. 둘 다 같으면 점수 비교 —
   상대 (우세-열세)/우세 >= 0.20  그리고  절대 probe 차이 >= 2  일 때만 우세 선택
4. 미달이면 그 축을 이번 방문 보류, 다음 순위 축 검토
5. 양쪽 점수와 보류 이유를 report metadata에 남긴다
```

절대 조건이 필요한 이유: 상대 조건만 두면 `3:2`가 33%로 통과하는데, probe 1개 차이는
§5.2의 측정 노이즈에 묻힌다.

### 4.5 후보값 통합

같은 action을 지지해도 label마다 후보값이 다를 수 있다.

1. support별 후보값과 provenance를 보존한다.
2. action 방향에 맞지 않는 값을 제거한다.
3. stable dedupe → constraint 적용 → baseline 변화량이 작은 값부터 정렬.
4. candidate budget(기존 `_MAX_SWEEP_CANDIDATES=3`)까지만 포함한다.
5. 후보가 여러 개면 internal sweep 또는 chunk prescreener가 선택한다.
6. 후보가 하나면 rules backend가 한 번 적용한다.

chunk 후보 정책은 그대로 보존한다 — 정상 intersection이 있으면 단계별 후보군 유지,
없을 때만 absolute boundary clamp, 유효한 span이 있을 때만 prescreener,
context/dependency 부재는 rules fallback, 실제 prescreener 실패는 숨기지 않는다.

### 4.6 실행 가능성 정책 (planner·optimizer 공유)

```text
canonical path 정규화 → state mapping → backend path → pipeline capability
→ runtime capability → baseline prerequisite → numeric/allowed constraint
→ no-op 제거 → 단일 축 보장
```

planner는 통과한 candidate만 점수화하고, optimizer는 실행 직전 같은 정책으로 재검증한다.

특수 조건: reranker candidate count는 reranker enabled + runtime verified 필요.
runtime unavailable은 품질 blacklist로 분류하지 않는다. chunk overlap은 chunk size
비율 제약을 유지한다. multi-axis action은 기본 거부한다.

---

## 5. 합의된 결정

| 항목 | 결정 | 근거 |
| --- | --- | --- |
| action key 형식 | 방향·boolean은 key에, **고정값은 후보값으로** | §3.2 starvation |
| `adjust` operation | 신설 | §3.2 |
| A>C>B hard tier | **불변조건**, multiplier 미채택 | §4.3 |
| conflict margin | 상대 20% + 절대 probe 2, **eligibility 통과 후에만** | §4.4 |
| boolean 축 | 충돌 정책 대상 아님 | §4.4 |
| inverse transition | **정확 전이만 차단** | §5.1 |
| `base_cost` | 현행 이분법 이관 | §3.1 |
| `max_iterations` | 3 → 5 (탐색 깊이 목적) | §5.3 |
| 개선 판정 마진 | composite 2점 = 내부 `0.02` | §5.2 |
| `graph.py` | 무수정 | §5.3 |
| `rules.py` 전면 재작성 | **불필요** | §6 |
| 차단 기능 활성화 | 별도 PR | — |
| `confidence` 생산 | 별도 Eval 작업 | §1.1 |

### 5.1 blacklist와 inverse transition

```text
품질 실패            → exact ActionAttemptKey 차단
완료된 sweep         → ActionStudyKey 차단
capability 미지원    → visit-local exclusion 또는 deferred 기록
새 baseline          → 같은 action 재시도 허용
```

**정확 전이만 차단하고 축 단위 방문 이력은 두지 않는다.** config가 바뀌면 재색인·재평가를
거쳐 새 Finding과 새 점수가 나오므로 같은 축의 재평가가 타당한 경우가 있다.

```text
reranker 켜봄 → 별로 → 롤백 → top_k 고쳐 검색 개선
→ 이제 reranker는 후보 풀이 달라졌으니 효과도 다르다 → 재평가할 가치가 있다
```

축을 닫으면 "A>C>B로 상류를 먼저 고친 뒤 하류를 다시 본다"는 설계 의도와 충돌한다.
무한 반복은 `ActionAttemptKey`가 정확 전이를 막고 후보 소진으로 축이 닫히며,
예산(5회/20방문)과 개선 마진이 추가로 방어한다.

> 현재 `(label, prescription_id)` blacklist는 baseline 무관 영구 차단이라 오히려 더
> 강하다. `ActionAttemptKey` 전환은 차단 **강화가 아니라 baseline별 완화**다.

### 5.2 개선 판정 최소 마진 → **선행 PR ①**

현재 판정은 0.0001만 올라도 유지한다. 노이즈로 오른 값을 개선으로 확정하면 롤백
안전망을 **통과하면서** 왕복이 발생한다.

```python
MIN_IMPROVEMENT_MARGIN = 0.02      # composite 표시 2점. judge와 sweep이 같은 값을 쓴다
```

`internal_adapter`에 `min_delta`가 **이미 완성돼 있고** planner가 값을 넘기지 않아
비활성 상태다. 두 지점 스케일은 모두 0~1 정규화 composite다.

### 5.3 반복 예산

```text
iteration = 실제 새 ActionStudy를 적용한 횟수
  같은 action 내부 sweep은 iteration 하나
  baseline/no-op이면 예산 미소비
  rollback 후 다른 action은 새 iteration
  같은 action도 새 baseline에서 새 study면 새 iteration
```

`max_iterations` 3 → 5. action 단위로 선택이 잘게 쪼개지므로 18개 action을 3회로
탐색하기엔 얕다. **위험 대응이 아니라 탐색 깊이 확보다.**

#### `iteration`은 `graph.py`가 종료 조건으로 읽는다

```python
# graph.py:65  route_after_eval
if state.iteration >= state.max_iterations:
    ...  return "serve"      # 파이프라인 종료
```

`graph.py`를 한 줄도 고치지 않아도 `iteration` 의미를 바꾸면 동작이 달라진다.
**그러나 라우팅 수정은 해법이 아니다** — `graph.py`는 읽기만 하고 증가는 `agent.py`
소관이라, 미소비 경로가 반복되면 가드가 발동하지 않는다. 실제 최종 방어선은
`graph.py`가 모르는 `optimize_visit_count`(agent.py:462)다. **무수정 원칙을 유지한다.**

#### no-op은 루프를 돌지 않는다

```python
# graph.py:83  route_after_optimize
if state.status in ("applied", "rolled_back"):
    return "index"      # config가 바뀐 경우만 루프
return "serve"          # skipped / verified / manual_required → 종료
```

"변경 없는 방문이 무한 반복"은 구조적으로 성립하지 않는다. 최악은 적용·롤백 반복이며
적용마다 예산이 소비되어 10회 내외로 묶인다. `max_optimize_visits`(20)는 도달값이
아니라 최종 안전선이다.

**착수 전 증명할 것**: 미소비 경로 목록, 각 경로의 수렴, visit 소비 여부, 최악 방문 수,
종료 경로 동등성. **핵심 불변조건은 `optimize_visit_count`가 어떤 경로에서도 증가한다는
것.** iteration 미소비는 허용하되 visit 미소비는 허용하지 않는다.

### 5.4 결과 귀속

action이 지지받은 사실과 label이 실제 해결된 사실을 구분한다. 적용 당시
supporting labels/probes를 저장하고, 다음 Eval의 remaining labels/probes와 비교해
before/after confirmed Finding 차집합으로 resolved labels를 계산한다.

---

## 6. 파일별 변경

> **`rules.py` 전면 재작성은 전제가 아니다.** patch 키가 canonical 20 / flat 20으로
> 섞여 있으나 **실행 가능한 flat 키 4개가 전부 `canonicalize_path`로 정규화된다**
> (`top_k`, `chunk_size`, `chunk_overlap`, `embedding_model`). 정규화되지 않는 8개는
> 어차피 `STATE_MAPPABLE_PATHS` 밖이다. catalog가 키 생성 시 `canonicalize_path`를
> 거치면 혼재는 무해하다. rules 정리는 가독성 목적의 선택 작업이다.

| 파일 | 변경 | 핵심 주의사항 |
| --- | --- | --- |
| **`action_catalog.py`** (신규) | `ActionDefinition` 레지스트리, key 유일성·계약 검증 | reindex/cost/capability/conflict family의 단일 진실 원천. 차단 action도 blocked reason과 함께 등록 |
| **`action_aggregator.py`** (신규) | Finding→support→candidate, 점수·충돌·정렬 | Finding을 재진단하지 않는다. 같은 probe 중복 투표 제거. tie에서도 결정적 |
| **`candidate_values.py`** (신규) | planner의 후보값 수학 이관 | **기존 수학을 그대로 옮긴다.** label 하드코딩을 `candidate_strategy` registry로 교체. evidence analysis는 memoize |
| **`eligibility.py`** (신규) | 실행 가능성 정책 공유 | **신설이 아니라 추출.** `optimizer.py`의 `DEFAULT_CONSTRAINTS`/`DEFAULT_CAPABILITIES`/`PATH_CAPABILITIES`/`STATE_MAPPABLE_PATHS`/`BACKEND_SUPPORTED_PATHS`/`REINDEX_PATHS`를 옮긴다. **값을 바꾸지 않는다** |
| `rules.py` | `LABEL_TO_ACTION_RULES`로 전환, patch/cost 중복 제거 | action key·candidate strategy·`applies_when` 중심. target metrics와 manual action은 label 수준 유지 |
| `schemas.py` | 새 dataclass 추가, `OptimizationRequest`/`HistoryItem`/`Report`에 action 필드 | `failure_label`·`related_failure_labels`·`candidates`는 deprecated. dual-read 기간 유지 |
| `planner.py` | 얇은 orchestration으로 축소 | 유지: 분기·preliminary 제외·manual 보존·ID 생성·backend 선택. 제거: label score/ranking, 대표 label 선택, 후보값 수학 |
| `optimizer.py` | 선택된 action 하나만 준비, eligibility 재검증 | 여러 prescription 순회 제거. patch metadata에 attempt/study fingerprint. embedding recreate 안전 플래그·chunk fallback·multi-axis 방어 유지 |
| `agent.py` | **핵심 전환 지점** | `last_failure_label` 비교 제거 → 새 ActionStudy 기준 iteration. `blocked_action_attempts`/`completed_action_studies`. reranker guardrail을 action key 분기로. active study를 단일 축 일반화 |
| `history.py` | pending item이 action/support snapshot 수령 | `last_action_key`, before/after confirmed snapshot, resolved/remaining 계산, fingerprint helper. 판정 수학과 floor는 유지 |
| `reporter.py` | action 합의 중심 설명 | 실제 selected action을 읽는다 — 첫 candidate로 추측하지 않는다 |
| `config_mapper.py` | catalog canonical path와 일치 검증 | 핵심 mapping 유지. hybrid는 boolean이 아닌 canonical `"hybrid"` |
| `adapters/internal_adapter.py` | action key·supporting labels를 trial metadata로 | 대표 label 의존 제거. sweep 알고리즘·단일 축 검증 유지 |
| `adapters/chunk_prescreener.py` | 여러 support의 span/document 통합 | 동일 span dedupe. missing context fallback 유지, 실제 failure 구분 유지 |
| `adapters/ragbuilder_adapter.py` | payload를 action 중심으로 | 대표 label은 설명용 compatibility field로만. strict hybrid 계약·external validation 유지 |
| `core/state.py` | `blocked_action_attempts`, `completed_action_studies` 추가 | 기존 `blacklist`/`completed_prescriptions`와 병행 후 전환. **`graph.py`는 수정하지 않는다** |
| `serve/report_view.py`, `serve/web_api.py` | action key·supporting/resolved label 표시 | 구버전 history fallback 유지 |

### 6.1 reranker guardrail 이관

prescription ID 분기를 action key 분기로 교체한다.

```text
enable_reranker         → reranker.enabled:enable
widen_rerank_candidates → reranker.candidate_count:increase
```

보존: runtime unknown/unavailable은 품질 blacklist 제외, disabled 상태에서 candidate
count 확대 금지, reranker applied 불완전 시 unjudgeable rollback, low-rank 감소와
composite 상승 시 precision floor 완화, 같은 runtime 오류 무한 반복 방지.

---

## 7. 테스트

### 7.1 신규

| 파일 | 필수 항목 |
| --- | --- |
| `test_action_catalog.py` | key 유일성, 동일 path/operation 중복 금지, 모든 label reference가 catalog에 존재, 한 action 한 축, reindex/cost/capability 일치, ready action은 mapping 또는 blocked reason 보유 |
| `test_action_aggregator.py` | 다른 ID의 동일 변경 통합, 같은 probe 1회 집계, preliminary·manual 제외, `applies_when` 불일치 제외, 실행 불가 사전 제외, action 자체 비용 사용, A>C>B tier, 결정적 tie-break, 근소 충돌 보류, grounded 우선, target metrics stable union |
| `test_candidate_values.py` | 기존 planner 후보 수학 이관 + 여러 support union/dedupe, 방향 위배 값 제거, baseline no-op 제거, constraint 후 stable ordering, budget, provenance, chunk 후보 보존, dead-zone clamp, invalid evidence fallback |
| `test_action_history.py` | 구버전 fallback, action fingerprint 직렬화, resolved/remaining 계산 |

### 7.2 Invariant (전 구현 공통)

1. 선택 action은 catalog에 존재하고 eligibility를 통과한다
2. search space는 단일 canonical 축이고 모든 후보값이 constraint 안이다
3. 방향 후보는 baseline 대비 올바른 방향이다
4. 같은 probe는 score에 최대 한 번 기여한다
5. score와 breakdown 합계가 일치한다
6. 동일 입력은 항상 같은 선택을 낸다
7. blocked exact attempt를 재선택하지 않는다
8. 새 baseline/candidate를 과도하게 차단하지 않는다
9. **B그룹 action은 실행 가능한 A/C action이 남아 있는 한 선택되지 않는다**
10. **같은 축의 여러 고정값은 단일 action의 후보로 통합된다** (경쟁하지 않는다)
11. **어떤 축도 "지지가 동일해서" 영구 보류되지 않는다** (starvation 부재)
12. 적용/rollback 후 config는 허용 snapshot이다

### 7.3 회귀 (기존 테스트 통과 유지)

`test_planner` `test_optimizer` `test_optimize_agent` `test_internal_adapter`
`test_ragbuilder_adapter` `test_enable_reranker` `test_chunk_prescreener`
`test_chunk_grounding_integration` `test_chunk_overlap_grounding` `test_evidence_window`
`test_config_mapper` `test_rollback_cache` `test_report_view` `test_serve_api` `test_pipeline`

특히: 기존 chunk 후보/prescreener, gate/pending finalize, reranker guardrail,
rollback cache·reindex 요구, `graph.py` route 무변경.

---

## 8. 구현 단계

> 각 단계의 상세 작업은 **PR #75 본문의 task list**로 관리한다.

| 단계 | 내용 | 상태 |
| --- | --- | --- |
| ~~**0-A**~~ | ~~선행 PR ①② 병합~~ | ✅ #76(마진), #73(sweep 축) |
| ~~**0**~~ | ~~Baseline 고정~~ | ✅ 아래 결과 |
| ~~**1**~~ | ~~schema + action catalog~~ | ✅ `action_catalog.py`, 27 tests |
| ~~**2**~~ | ~~candidate value 분리~~ | ✅ planner 1839 → 791줄 |
| ~~**3**~~ | ~~aggregation shadow mode~~ | ✅ 중단 기준 판정 통과(§8.1) |
| ~~**4**~~ | ~~planner 선택 중심 전환~~ | ✅ 알려진 실패 21건은 5~7에서 해소 |
| ~~**5**~~ | ~~history·blacklist·iteration 전환~~ | ✅ 16건 해소 (§8.2 결과) |
| ~~**6**~~ | ~~reranker·adapter guardrail 이관~~ | ✅ 5건 해소 (§8.3 결과) |
| ~~**7**~~ | ~~reporter·serve 전환~~ | ✅ §8.4 결과 |
| **8** | compatibility 제거·문서 갱신 | §8.5 |

알려진 실패 21건은 전부 해소됐다. 현재 **1066 tests 전부 통과**
(환경 사유 4개 모듈 `test_pipeline`·`test_ragas_eval`·`test_oauth`·`test_eval` 제외).

### 8.1 중단 기준 판정 결과 (단계 3)

시나리오 5개로 legacy 와 action 선택을 비교했다. **5건 중 2건에서 선택이 갈렸고,
그중 1건이 `shared_support_won`** 이라 진행 조건(공유 합산으로 달라진 사례 1건 이상)을
충족했다.

```text
공유<단독 케이스
  legacy : retrieval_low_rank 가 probe 6개로 단독 최고 → 리랭커를 켠다
  action : 두 라벨이 함께 지지하는 top_k 증가를 고른다(probe 합산)
```

함께 관측된 것: 같은 비용 안에서는 `action_key` 사전순으로 갈리므로 **`rules.py` 의
"가벼운 것 먼저" 선언 순서 의도가 사라진다.** 그대로 두기로 했다 — 최종 판정은 Eval
실측이 하고 예산이 5회라 밀린 후보도 결국 시도된다. 선언 순서를 tie-break 에 넣으면
"label 이 실행 순서를 소유하지 않는다"는 전환 목적과 어긋난다.

### 8.2 단계 5 — history·blacklist·iteration 전환

**실패 중인 테스트 9건**(`test_optimize_agent`)이 이 단계의 완료 지표다.

#### 고칠 지점 (`agents/optimize/agent.py`)

| 위치 | 현재 | 바꿀 것 |
| --- | --- | --- |
| `agent.py:565-569` | `last_failure_label` 비교로 `starts_new_label` 판정 | `last_action_key` 비교로 `starts_new_action_study` |
| `agent.py:741` | `starts_new_label` 이면 iteration 증가 | 새 ActionStudy 적용 시 증가 |
| `agent.py:665` | `state.blacklist.add((label, prescription_id))` | `blocked_action_attempts.add(ActionAttemptKey)` |
| `agent.py:162,414,509,924` | `item.failure_labels[0]` / `selected_prescription_id` 로 로그·판정 | `item.action_key` |
| `agent.py:873` | 완료 sweep 을 `(label, prescription_id)` 로 기록 | `completed_action_studies.add(ActionStudyKey)` |

#### `agents/optimize/history.py`

```text
254  failure_labels=[request.failure_label]      → action_key, supporting_labels 스냅샷
257  selected_prescription_id=prescription_id    → action_attempt_key / action_study_key
299  last_failure_label()                        → last_action_key()
```

`create_pending_item` 이 support 스냅샷(지지 label·probe)을 받아야 §5.4 결과 귀속이
가능하다. 판정 수학(`judge`, `check_floor`, 개선 마진)은 **건드리지 않는다.**

#### `core/state.py`

```python
blocked_action_attempts: set = field(default_factory=set)
completed_action_studies: set = field(default_factory=set)
```

기존 `blacklist` / `completed_prescriptions` 와 병행한다. planner 의
`_normalize_exclusions` 가 이미 두 형태를 모두 받으므로, agent 가 새 필드를 채우기
시작해도 선택 경로는 깨지지 않는다.

#### fingerprint

```text
ActionAttemptKey  = (action_key, baseline_fingerprint, candidate_fingerprint)
ActionStudyKey    = (action_key, baseline_fingerprint, search_space_fingerprint)
입력: action key + baseline canonical effective config + candidate config
      + 관련 runtime capability identity
```

**정확 전이만 차단한다**(§5.1). 축 단위 방문 이력은 두지 않는다.

#### 완료 조건

- `test_optimize_agent` 9건 통과
- `test_planner` 7건을 action 기준으로 재작성 (선택 결과 변화는 의도된 것)
- `(label, prescription_id)` 튜플이 실행 제어에 쓰이지 않음
- rollback 후 재색인 요구·visit limit·pending finalize 동작 불변

#### 단계 5 결과 — 계획과 달랐던 것

**뒤집힌 동작 2건.** 둘 다 계획대로 두기로 확인했다.

| 동작 | 이전 | 지금 | 근거 |
| --- | --- | --- | --- |
| 예산 소진 후 같은 라벨의 다음 처방 | iteration 미소비로 계속 적용 | 다른 action = 새 study 라 롤백만 확정하고 종료 | §5.3. 라벨 하나가 예산 밖에서 무한히 config 를 갈아 끼울 수 있던 구멍 |
| 노이즈 우세 시 C를 A보다 먼저 | `_context_noise_precedes_top_k_expansion` 예외 | tier 가 무조건 우선 | §4.3 hard tier 불변조건 |

**계획에 없던 구멍 3건을 함께 막았다.**

1. **단계 4 회귀** — `_build_action_request` 가 `rerank_candidate_policy` 상한을
   `metadata["constraints"]` 로 싣지 않아, 방향 폴백(현재값×2=60)이 정책 상한 50 을
   넘겨 config 에 박혔다. eligibility 와 optimizer 재검증 양쪽에 정책 제약을 넘긴다.
2. **끝난 sweep 의 자기 재탐색** — study key 는 baseline fingerprint 를 담는데
   sweep 승자가 **그 action 자신 때문에** baseline 을 움직여 fingerprint 가 어긋난다.
   다음 방문에 같은 축을 곧바로 다시 훑었다(이미 잰 값을 다시 재는 낭비).
   측정한 후보를 **결과 baseline 기준으로도** 소진 처리해 §5.1 의 "후보 소진으로
   축이 닫힌다"가 실제로 성립하게 했다.
3. **설명 유실** — 실행 가능한 action 이 하나도 없으면 request 가 없어 제외 사유를
   실을 곳이 사라졌다. `OptimizeDecision.metadata` 에
   `rejected_actions`/`deferred_axes` 를 담는다.

**`PrescriptionOrderCharacterization` 교체.** 단계 0 이 예고한 대로 이 클래스는
전환으로 깨졌다. `LabelNoLongerOwnsExecutionOrderTest` 로 교체해 **새 규칙**
(tier → 점수 → grounded → 비용 → key 사전순, 선언 순서 미개입)을 박제했다.
박제를 그냥 두면 "전환이 일어났다"와 "테스트가 깨졌다"를 구분할 수 없다.

### 8.3 단계 6 — reranker·adapter guardrail 이관

**실패 중인 테스트 3건**(`test_enable_reranker`)이 완료 지표다.

#### prescription id 분기를 action key 로 (`agent.py:720, 782, 1237`)

```python
{"enable_reranker", "widen_rerank_candidates"}
→ {"reranker.enabled:enable", "reranker.candidate_count:increase"}
```

세 곳 모두 같은 집합을 쓰므로 상수 하나로 묶는다.

#### 보존해야 하는 동작

- runtime unknown/unavailable 은 **품질 blacklist 로 분류하지 않는다**(deferred 기록)
- reranker 가 꺼진 상태에서 candidate count 확대 금지 (`optimizer` 의 `reranker_disabled`)
- reranker 적용이 불완전하면 `unjudgeable` rollback
- low-rank 감소 + composite 상승 시 precision floor 완화
- 같은 runtime 오류 무한 반복 방지

> ⚠️ 단계 4 에서 planner 의 runtime 판정을 "정보 부재는 통과" 로 바꿨다. 최종 판정이
> optimizer 에 있다는 전제이므로, 이 단계에서 optimizer 의 deferred 경로가 실제로
> 동작하는지 반드시 확인한다.

#### adapter (`internal_adapter`, `ragbuilder_adapter`, `chunk_prescreener`)

- 대표 label 의존 제거 → action key + supporting labels 를 trial metadata 로
- RAGBuilder payload: `failure_label`/`related_failure_labels` → action 중심.
  외부 호환이 필요하면 대표 label 은 설명용 compatibility field 로만 유지
- chunk prescreener: 여러 support 의 span/document 통합, 동일 span dedupe

#### 단계 6 결과 — ⚠️ 의 답

**optimizer 의 deferred 경로는 살아 있다. 그런데 그게 문제가 아니었다.**

runtime 정보가 아직 없는 첫 방문에서는 planner 가 통과시키고 optimizer 가 최종
판정한다 — 그 경로는 정상 동작한다(테스트로 고정). 실제 원인은 반대쪽이었다:
**runtime 상태가 이미 "unavailable" 로 알려졌거나 선행 조건이 미충족이면 planner 가
점수 경쟁 전에 걸러내 deferred 기록이 아예 생기지 않았다.**

판정이 앞단으로 옮겨오면 **보고 책임도 함께 옮겨와야 한다.** planner 의
`rejected_actions` 중 `runtime_capability_unavailable`·`prerequisite_unmet` 을
agent 가 보류로 번역해 리포트에 싣는다. 어느 계층이 걸렀는지는 사용자 관심사가
아니므로 optimizer 가 거른 경우와 보고 형태가 같다.

**소진 표현이 바뀌었다.** no-op 축(예: 리랭커가 이미 켜진 상태의
`reranker.enabled:enable`)은 예전엔 선택된 뒤 optimizer 가 `no_valid_candidate_values`
로 거절하며 blacklist 에 올렸다. 이제는 선택 전에 걸러지고 품질 실패로 기록되지
않는다 — 실행조차 안 된 처방을 비난하지 않는 쪽이 맞다.

### 8.4 단계 7 — reporter·serve 전환

소비처는 7군데다(`reporter.py`, `serve/report_view.py`, `serve/api.py`).

- 대표 label 하나가 아니라 **action + supporting labels** 로 설명
- opposing labels 와 충돌 보류 이유 노출(`request.metadata["deferred_axes"]`)
- score breakdown drill-down (`supporting_probe_count`, `weighted_probe_support`,
  `cost_source`, `confidence_source`)
- resolved / remaining label 구분
- **구버전 history fallback 유지** — 이전 실행의 `selected_prescription_id` 기록도 읽어야 한다
- reporter 는 실제 selected action 을 읽는다. 첫 candidate 로 추측하지 않는다

#### 단계 7 결과

소비처는 3개 파일이었다(`reporter.py`, `serve/report_view.py`, `serve/web_api.py`).

| 지점 | 전환 |
| --- | --- |
| `OptimizationReport` | `action_key`·`supporting`/`opposing`/`resolved`/`remaining_labels`·`score_breakdown`·`deferred_axes` 추가 |
| `reporter._selected_prescription` | 후보 목록 첫 원소 **추측을 제거**. patch metadata 의 출처를 읽는다 |
| `reporter` 요약문 | 대표 라벨 1개 → action + 지지 라벨 전체("N개 라벨이 함께 지지한") |
| `report_view` 차트 점 이름 | 처방 id 표 → **축 이름 + 동작 동사** 조립(`top_k 확대`). catalog 에 action 이 늘어도 축만 등록하면 된다 |
| `report_view` 처방 카드 | `target` 을 지지 라벨 전체로, `resolved`/`remaining` 노출 |
| `web_api` 단계 요약 | `selected_prescription_id` → `action_key` |

**롤백에는 resolved 를 붙이지 않는다.** 설정을 되돌렸으므로 그 개선은 지금 config 에
남아 있지 않다. 지지 라벨 전체가 remaining 이 된다.

**`drill.rows` 에 섞지 않았다.** 그 필드는 `report.html` 의 sweep 막대그래프 전용
(`{k, val, w, win}`)이라 선택 근거를 넣으면 렌더가 깨진다. `drill.notes` 로 분리하고
프론트에 렌더링·CSS 를 추가했다. 펼친 패널은 고정 높이(320px)라 내용이 잘릴 수 있어
420px + 스크롤로 바꿨다.

### 8.5 단계 8 — compatibility 제거

`PrescriptionCandidate` 참조가 남은 곳:

```text
agents/optimize/planner.py      (legacy 파생 생성)
agents/optimize/optimizer.py    (소비)
agents/optimize/schemas.py      (정의)
tests/test_optimizer.py, test_optimize_agent.py, test_ragbuilder_adapter.py
문서 3개 (README / PROGRESS / 이 파일)
```

순서: 소비처 전환 확인 → `OptimizationRequest.candidates` 파생 중단 →
`PrescriptionCandidate` 제거 → `failure_label`·`related_failure_labels` 제거 →
문서 갱신(`AGENTS.md`, `README.md`, `CONTEXT.md`, `PROGRESS.md`,
`OPTIMIZER_IMPLEMENTATION_PLAN.md` 에 superseded 표시).

**저장 state 호환 요구를 먼저 확인한다.** 이전 실행의 `optimization_history` 를 읽어야
하면 구버전 필드 reader 를 남긴다.

### 단계 0 결과 (완료)

| 산출물 | 내용 |
| --- | --- |
| `tools/action_inventory.py` | rules + optimizer 정책에서 실행 가능 action을 집계. §2 현황표의 생성기이자 §2.2에서 제안한 자동화 |
| `tests/test_action_inventory.py` | 집계 결과를 baseline으로 고정 (12 tests) |
| `tests/test_planner_characterization.py` | 전환 전 선택 결과 박제 (13 tests) |

**테스트 baseline** (`origin/main` 머지 후)

```text
960 tests / 약 14초 (단계 0 산출물 25개 포함, 이전 936)
환경 사유 수집 에러 3~4건 — 코드 회귀 아님
  test_pipeline    data/pdf_corpus.json 없음 (실데이터 필요)
  test_ragas_eval  OPENAI_API_KEY 실키 필요
  test_oauth       OAuth 수동 확인 스크립트
  test_eval        ⚠️ flaky — 실행마다 수집 성공/실패가 갈린다(외부 상태 의존)
```

네 파일 모두 `unittest.TestCase`가 아닌 **실행 스크립트**인데 `test_*.py` 이름 때문에
오수집된다. 구현 중 실패가 나면 이 목록과 대조해 **환경 문제인지 코드 회귀인지**
즉시 구분한다.

**동점 입력을 characterization에 넣지 않았다.** 현재 `_group_by_label`은 dict 삽입
순서에 의존하는데 §4.2가 결정적 tie-break를 도입하므로, 동점에서는 선택이 바뀌는 것이
**정상**이다. 그런 입력을 박제하면 단계 1·2의 "기존 동작 변화 없음"과 충돌한다.

- 비동점 입력(`GroupPriority`·`ScoreOrder`·`SearchSpace`·`DecisionMode`) → 전환 후에도 **같은 선택** 요구
- 동점 입력(`TieBreakDeterminism`) → 선택 내용이 아니라 **결정성만** 검증

`PrescriptionOrderCharacterization`은 예외다. "라벨이 처방 순서를 소유한다"는 성질을
박제했는데 **전환으로 사라지는 것이 목표**다. 단계 4에서 action 기준 테스트로 교체하며,
그때 이 클래스가 깨지는 것이 전환이 실제로 일어났다는 증거가 된다.

**단계 3 중단 기준**: shadow mode에서 legacy와 action 선택을 비교해 기록한다.

| 관측 | 조치 |
| --- | --- |
| 선택이 달라진 사례 0건 | 전환 보류 — 구조만 남기고 선택 로직은 legacy 유지 |
| 달라졌으나 전부 tie-break 차이 | 전환 보류 — 실익 없음 확인 |
| 공유 합산으로 달라진 사례 1건 이상 | 계속 진행 |

0건이어도 catalog·eligibility 추출은 유지할 가치가 있다(중복 선언 제거, 정책 단일화).
중단은 **선택 로직 전환에만** 적용한다.

---

## 9. 완료 기준

**구조** — label이 실행 순서를 결정하지 않는다 / config 변경이 catalog에 한 번만 정의된다 /
동일 action은 이름이 달라도 하나로 집계된다 / 실행·history·blacklist identity가 action 기반이다

**안전** — 한 번에 축 하나 / 실행 불가 action은 경쟁 전 제외 / 증가·감소를 임의 선택하지 않는다 /
B그룹이 A/C를 앞지르지 않는다 / 지지가 동일한 축이 영구 보류되지 않는다 /
exact 실패 transition을 재시도하지 않는다 / rollback cache와 reindex 요구 보존 /
절대 visit limit 유지

**품질 판정** — 사용자 pipeline Eval 전에 improved를 확정하지 않는다 / 개선 마진 미달을
개선으로 판정하지 않는다 / composite·floor 기반 keep/rollback 유지 / supporting과 resolved
label 구분 / internal sweep은 action study 하나

**설명 가능성** — score breakdown / supporting label과 고유 probe 수 / opposing label과
충돌 처리 이유 / config before-after와 재색인 여부 / resolved-remaining label

---

## 10. 검증

```bash
python3 -m unittest discover -s tests
```

```bash
python3 -m compileall -q agents core tests && git diff --check
```

외부 API·LLM·Qdrant가 필요한 테스트는 기본 단위 테스트와 분리하고, 실패 시 환경 문제와
코드 회귀를 구분해 기록한다.

---

## 11. 범위 밖

- 차단된 기능의 실제 활성화 (`query_rewrite`, `adaptive_retrieval`, `embedding.model` 등)
- Finding별 `confidence` 생산 (Eval 계약 변경 필요)
- 후보값 수학과 keep/rollback 임계값 변경
- 동적 마진(표준오차 기반) — Eval이 probe별 점수를 노출해야 가능
- `base_cost` 세분화 — `confidence` 생산과 함께

구조 변경의 효과와 신규 기능 활성화의 효과를 따로 검증하기 위한 분리다.
