# 개선 판정 최소 마진 도입 — 구현 계획

> 상태: 구현 전 설계·작업 계획
> 작업 브랜치: `feature/optimize-improvement-margin`
> 분기 기준: `origin/main` (`ae3e46c`)
> 목표: 측정 노이즈로 인한 "가짜 개선"을 유지 판정에서 걸러낸다.
> 관련: 이슈 #67 선결 과제 ②, `ACTION_CENTERED_OPTIMIZER_IMPLEMENTATION_PLAN.md` §5.11

---

## 1. 목적

### 1.1 문제

현재 keep/rollback 판정은 **티끌만큼만 올라도 유지**한다.

```python
# agents/optimize/history.py:143  judge()
if after_score > before_score:      # 0.0001 상승도 "개선"
    return Verdict(keep=True, ...)
```

Eval 점수에는 LLM judge 편차·표본 노이즈가 섞여 있다. 노이즈로 우연히 오른 값을
개선으로 확정하면 다음이 발생한다.

1. **효과 없는 config가 유지된다.** 롤백 안전망은 "점수가 안 올랐을 때" 작동하는데,
   노이즈가 점수를 올려버렸으므로 통과한다.
2. **왕복(oscillation)이 안전망을 통과한다.** 같은 축을 늘렸다 줄였다 해도 각각
   노이즈로 소폭 상승하면 둘 다 "개선"으로 유지된다.
3. **예산이 낭비된다.** `max_iterations`가 3회뿐인데 노이즈 추적에 소모된다.

### 1.2 해결

개선으로 인정할 **최소 상승폭**을 둔다.

```text
after_score > before_score + MARGIN   → 유지
그 외                                  → 롤백
```

마진 값은 **composite 표시 점수 기준 2점**으로 잡았다.
⚠️ 이 값은 잠정치다 — 노이즈 분포를 측정해서 나온 값이 아니다(§8 첫 항목).

---

## 2. 현재 코드 상태 — 사전 조사 결과

### 2.1 ⚠️ sweep 쪽은 이미 구현돼 있다 (중요)

`internal_adapter`에는 **`min_delta`라는 이름으로 동일 기능이 이미 완성돼 있다.**
planner가 값을 넘기지 않아 기본값 `0.0`으로 **비활성 상태**일 뿐이다.

```python
# agents/optimize/adapters/internal_adapter.py:314
def _min_delta(self, request: OptimizationRequest) -> float:
    raw = request.metadata.get("min_delta", 0.0)     # ← 아무도 넣지 않아 항상 0.0
    if not self._is_finite_number(raw) or float(raw) < 0:
        raise ValueError("min_delta는 0 이상의 유한한 숫자여야 합니다.")
    return float(raw)
```

```python
# agents/optimize/adapters/internal_adapter.py:733  _complete()
improvement = self._improvement(best_candidate.score, baseline.score, direction)
improved = improvement > 0 and (
    improvement > min_delta
    or math.isclose(improvement, min_delta, rel_tol=1e-12, abs_tol=1e-12)
)
if not improved:
    best_is_baseline = True
    selected = baseline          # ← baseline으로 되돌림
```

`direction`(minimize/maximize)까지 반영한 `_improvement`가 이미 있고,
baseline 부재 시 방어 로직도 있다.

```python
# internal_adapter.py:715
if baseline is None and min_delta > 0:
    return self._result(request, status="failed",
                        error="min_delta 비교에 필요한 scorable baseline trial이 없습니다.", ...)
```

**따라서 sweep 쪽 작업은 "값을 넘기는 것"뿐이다.** 로직을 새로 만들지 말 것.

### 2.2 judge 쪽은 미구현

```python
# agents/optimize/history.py:143
if after_score > before_score:
    return Verdict(keep=True, ..., reason=f"종합점수 상승 {before_score:.3f}→{after_score:.3f} → 유지")
return Verdict(keep=False, ..., reason=f"종합점수 미상승 ... → 롤백")
```

여기에 마진 개념이 없다.

### 2.3 점수 스케일 — 두 지점 모두 0~1

**표시 점수 2점 = 내부 값 `0.02`** 이다. 혼동하면 마진이 100배 어긋난다.

```python
# history.py:87  _read_score()
total = (report.composite_score or {}).get("total")   # 0~100
if total is not None:
    return float(total) / 100.0                       # → 0~1
```

```python
# planner.py  _report_metrics()
metrics["composite_total"]  = float(composite_total)          # 0~100 (표시용)
metrics["composite_score"]  = float(composite_total) / 100.0  # 0~1  (탐색 objective)
```

`internal_adapter`의 objective는 `request.metadata["primary_metric"]`이며
planner가 항상 `"composite_score"`(0~1)를 넣는다. 즉 **두 지점의 스케일이 같다.**

---

## 3. 변경 계획

> ⚠️ 아래 §3.1·§3.3 예시의 주석에 나오는 *"sweep이 고른 최선이 judge에서 탈락해
> 예산을 날린다"* 는 **현재 배선에서 성립하지 않는다.** sweep 승자는
> `_finish_internal_study`가 그 자리에서 확정해 `history.judge`를 거치지 않는다.
> 임계를 맞춰야 하는 진짜 이유는 §7-A ③ 참조. (계획 당시 서술을 이력으로 남긴다.)

### 3.1 상수 정의 — `agents/optimize/history.py`

두 모듈이 공유해야 하므로 한 곳에만 정의한다. `internal_adapter`가 `history`를
import하지 않으므로, planner가 중계한다(planner는 이미 양쪽을 안다).

```python
# agents/optimize/history.py 상단, _FLOORS 근처

# 개선으로 인정할 최소 상승폭(정규화 composite 0~1 기준).
# 표시 점수 2점 = 0.02. Eval 노이즈(LLM judge 편차·표본 오차) 안에 묻히는
# 상승을 "개선"으로 확정하지 않기 위한 값이다. 잠정치이며 σ 측정으로 재보정해야 한다.
# judge(유지/롤백)와 internal sweep(best 후보 선정)이 같은 값을 써야 한다 —
# 다르면 "sweep이 고른 최선이 judge에서 탈락"해 예산 한 번을 통째로 날린다.
MIN_IMPROVEMENT_MARGIN: float = 0.02
```

### 3.2 judge에 마진 적용 — `agents/optimize/history.py`

```python
# judge() 내부, floor 검사 이후
if after_score >= before_score + MIN_IMPROVEMENT_MARGIN:
    return Verdict(keep=True, ...)
return Verdict(keep=False, ...)
```

**주의사항**

- floor 위반 검사(`check_floor`)는 **마진보다 먼저** 유지한다. 하한선 위반은
  점수와 무관하게 무조건 롤백이다.
- `reason` 문구에 마진을 노출해 사용자가 판정 근거를 알 수 있게 한다.
  예: `"종합점수 상승폭 부족 0.812→0.818 (필요 +0.020) → 롤백"`
- 부동소수 비교는 `>=`와 함께 `math.isclose`를 쓰는 `internal_adapter`와
  동작을 맞춘다(경계값에서 두 모듈의 판정이 갈리면 안 된다).

### 3.3 sweep에 마진 전달 — `agents/optimize/planner.py`

`_build_request()`의 `metadata` dict에 한 줄 추가한다. `"primary_metric"`
바로 아래가 자연스럽다.

```python
metadata: dict[str, Any] = {
    "primary_metric": "composite_score",
    # sweep이 baseline을 이겼다고 판정하는 최소 상승폭. judge와 같은 값을 써야
    # "sweep이 고른 최선이 judge에서 탈락"하는 낭비가 생기지 않는다.
    "min_delta": history.MIN_IMPROVEMENT_MARGIN,
    "study_baseline_config": dict(state.index_config),
    ...
}
```

**import 주의**: planner가 `history`를 import하면 순환 참조가 생기는지 확인할 것.
`history`는 `schemas`와 `core`만 import하므로 문제없을 것으로 보이나, 실패하면
상수를 `agents/optimize/schemas.py`나 별도 `constants.py`로 옮긴다.

### 3.4 활성화 후 부작용 확인 — `internal_adapter`

`min_delta > 0`이 되면 **새 실패 경로가 열린다.**

```python
if baseline is None and min_delta > 0:
    status="failed", stop_reason="missing_scorable_baseline"
```

지금까지 `min_delta`가 0이라 이 경로는 죽어 있었다. 활성화하면 **baseline trial이
평가되지 않은 상황에서 study가 failed로 끝난다.** `agent.py`의 `_fail_active_study`
경로가 이 실패를 어떻게 처리하는지(롤백/blacklist/unjudgeable 분류) 반드시 확인하고,
필요하면 "품질 실패가 아닌 측정 불가"로 분류되게 한다.

---

## 4. 부작용과 완화

### 4.1 조기 종료 위험

마진을 올리면 롤백이 늘고, 롤백은 blacklist를 채운다.

```python
# planner.py  plan()
reason="처방 후보가 모두 블랙리스트에 걸림"   → use_current → serve
```

**마진 ↑ → 롤백 ↑ → 후보 고갈 → 아무 개선 없이 종료**가 가능하다.

**완화**: 사후 검증이 가능하도록 다음을 기록한다.

- 마진 때문에 탈락한 시도 수와 그때의 실제 점수 차이
  (`Verdict.reason` 또는 `history item.metadata`)
- blacklist 고갈로 조기 종료한 횟수

### 4.2 진짜 작은 개선을 놓친다

RAG 파라미터 하나를 바꿔 얻는 효과는 원래 작다. 마진이 노이즈보다 과도하게 크면
실제 개선까지 버린다. 반대로 노이즈보다 작으면 애초에 제 역할을 못 한다.
2점은 잠정치이므로 위 로그로 사후 조정한다.

---

## 5. 테스트 계획

### 5.1 `tests/test_optimize_agent.py` 또는 신규 `tests/test_improvement_margin.py`

judge 단위:

1. 상승폭이 마진 **미만** → `keep=False`, `rollback_reason`에 마진 명시
2. 상승폭이 마진과 **정확히 같음** → `keep=True` (경계 포함)
3. 상승폭이 마진 **초과** → `keep=True`
4. 하한선 위반 + 상승폭 충분 → **`keep=False`** (floor가 우선)
5. 점수 하락 → `keep=False` (기존 동작 유지)
6. `composite` 부재로 `overall_score` 폴백 시에도 같은 마진 적용

### 5.2 `tests/test_internal_adapter.py`

sweep 단위:

1. 후보가 baseline보다 마진 미만으로 나음 → `best_is_baseline=True`, baseline 선택
2. 마진과 정확히 같음 → 후보 선택 (`math.isclose` 경계)
3. `direction="minimize"` 지표에서도 부호가 올바른가
4. baseline trial이 없고 `min_delta > 0` → `status="failed"`,
   `stop_reason="missing_scorable_baseline"`
5. `min_delta`가 metadata에 실제로 전달되는가 (planner 통합)

### 5.3 일관성 테스트 (필수)

**두 모듈이 같은 임계를 쓰는지**를 직접 검증한다. 이 테스트가 이번 작업의 핵심이다.

```text
같은 (before, after) 쌍에 대해
  internal_adapter가 "개선"이라 판정하면
  history.judge도 반드시 keep=True 여야 한다
```

경계값(마진 ± 1e-12)에서도 두 판정이 갈리지 않아야 한다.

### 5.4 회귀 확인

- `tests/test_optimizer.py`
- `tests/test_enable_reranker.py` — reranker guardrail이 마진과 충돌하지 않는지
  (특히 precision floor 완화 경로)
- `tests/test_pipeline.py`

---

## 6. 검증 명령

```bash
python3 -m unittest tests.test_internal_adapter tests.test_optimize_agent tests.test_optimizer
```

```bash
python3 -m unittest tests.test_enable_reranker tests.test_planner tests.test_pipeline
```

```bash
python3 -m compileall -q agents core tests
```

---

## 7. 완료 기준

- [x] 마진 상수가 **한 곳**에만 정의되고 두 모듈이 공유한다
      (`history.MIN_IMPROVEMENT_MARGIN` → planner가 `metadata["min_delta"]`로 중계)
- [x] `history.judge`가 마진 미만 상승을 유지하지 않는다
- [x] `internal_adapter`가 `min_delta`를 planner로부터 실제로 받는다
- [x] 경계값에서 두 모듈의 판정이 일치한다
      (`tests/test_improvement_margin.py::ThresholdConsistencyTest`, 마진 ±1e-12)
- [x] floor 위반이 마진보다 우선한다
- [x] `min_delta > 0` 활성화로 열리는 `missing_scorable_baseline` 경로가
      품질 실패가 아닌 **측정 불가**로 분류된다
      (`_fail_active_study(unjudgeable=True)` → blacklist 대신 `_unjudgeable_exclusions`)
- [x] 마진 탈락 사유가 사용자 리포트에 노출된다
      (`Verdict.reason` → `reporter.build_trial_report` summary)
- [x] 기존 optimize 테스트 전량 통과

부작용 계측(§4.1)은 `history.finalize_item`이 이력 항목에 남긴다 —
`margin_rejected`(마진 때문에 탈락), `score_delta`(그때의 실제 점수 차이),
`improvement_margin`(적용된 마진 값). 조기 종료는 planner의
`reason="처방 후보가 모두 블랙리스트에 걸림"` 결정으로 이미 구분된다.

---

## 7-A. PR #76 리뷰 반영 (구현 후)

리뷰에서 지적된 6건을 모두 확인했고, 반려한 것은 없다.

| 지적 | 확인 방법 | 조치 |
|---|---|---|
| `_relax_reranker_precision_floor`가 옛 기준 | 코드 | `meets_improvement_margin`으로 교체 |
| `pass_threshold_reached`가 min_delta 우회 | 실행 재현 | baseline 있으면 마진 요구 |
| "judge에서 탈락" 근거 불성립 | 코드 | 서술 정정 |
| 주석 ↔ 본문 모순 | PR #66 원문 | 잠정치임을 명시 |
| `unjudgeable` 전환이 더 엄격 | 코드 | 사유별 시도 예산 분리 |
| `improvement_margin` 탈락 시에만 | 자명 | 전 항목 기록 |

**① floor 완화 예외의 점수 관문** (`agent._relax_reranker_precision_floor`)

하한선 위반이 있으면 judge가 floor 판정으로 먼저 반환하므로 마진이 개입하지 않는다.
이 함수의 `after_score > before_score`가 그 경로의 유일한 방어선이었고, 통과하면
`floor_violations=[]`로 판정을 완전히 뒤집는다. 즉 **+0.001 노이즈가 하한선 위반
처방을 되살렸다.** judge와 같은 기준으로 교체했다.

**② `pass_threshold_reached` 조기 반환** (`internal_adapter._complete`)

게이트를 넘은 후보를 min_delta 비교 없이 채택했다. 재현: baseline 89.9 → 후보 90.1
(+0.002, 마진의 1/10)에서 `best_is_baseline=False, improved=None`. `_finish_internal_study`도
마진 없이 `keep=True`로 확정하므로 **노이즈로 게이트를 넘은 config가 그대로 굳었다.**

scorable baseline이 있으면 통과 여부와 무관하게 마진을 요구하도록 고쳤다. baseline이
없거나 후보에 점수가 없으면 비교 근거가 없으므로 현행(통과 채택)을 유지한다.
비교식은 `_meets_min_delta`로 뽑아 두 경로가 같은 식을 쓰게 했다.

**③ 일관성의 진짜 이유**

`_finish_internal_study`가 `pending=False`로 끝내므로 **sweep 승자는 `history.judge`를
거치지 않는다.** "sweep이 고른 최선이 judge에서 탈락해 예산을 날린다"는 서술은 현재
배선에서 성립하지 않는다. 두 임계를 맞춰야 하는 실제 이유는 **두 경로가 각각 독립적으로
"개선했는가"를 판정하고 둘 다 사용자 리포트로 나가기 때문**이다 — 임계가 다르면 같은
점수 변화가 경로에 따라 다르게 보고된다.

**④ 시도 예산** — `blacklist`와 `optimization_history`는 같은 `AgentDoctorState`
필드라 수명이 같다. "새 실행에서 재시도"는 양쪽 모두에 해당하므로 `unjudgeable`
전환의 실익은 수명이 아니라 **분류**(측정 불가를 품질 실패로 기록하지 않음)다.
한편 `_MAX_UNJUDGEABLE_ATTEMPTS=1`이 기존 retryable 경로의 2회를 1회로 줄이고 있었다.
의도한 변경이 아니라 사유별로 분리했다 — 리포트 부재 1회,
`missing_scorable_baseline` 2회(`_MAX_MEASUREMENT_FAILURE_ATTEMPTS`).

## 8. 범위 밖

- **마진 값 자체의 재보정 (최우선 후속)**: `0.02`는 **잠정치이며 노이즈 분포를
  측정해서 나온 값이 아니다.** 유일한 관측 근거는 같은 config 재평가가 82↔78
  (표시 **4점** 폭)로 흔들린 사례 하나뿐이다(PR #66 리뷰). 그게 사실이면 2점은
  노이즈보다 작아 제 역할을 못 한다. **같은 config로 Eval을 N회 반복해 σ를 재고**
  마진을 `k·σ`로 다시 정해야 한다. 보정 관측값은 `finalize_item`이 남기는
  `margin_rejected`·`score_delta`·`improvement_margin`으로 모인다.
  PR #70(grounded-credit)이 composite 분산을 바꿀 수 있으므로 재베이스라인 시점에
  함께 캘리브레이션한다.
- **동적 마진**: 고정값은 노이즈 크기를 *가정*할 뿐 측정하지 않는다. Eval이
  probe별 점수를 노출하면 표준오차(`std/√n`) 기반으로 대체할 수 있으나, Eval과의
  계약 변경이 필요하므로 별도 작업으로 둔다.
- **action 중심 전환**: 이 작업은 그와 독립적이며 먼저 병합돼야 한다.
  (`ACTION_CENTERED_OPTIMIZER_IMPLEMENTATION_PLAN.md` §8.0-A 선행 PR ①)
- **`max_iterations` 상향**: action 전환 후에야 의미가 있으므로 그쪽 PR에 둔다.
- ~~**마진 값 재조정**: 2점은 실측 테스트 결과다.~~ → 사실이 아니다. 위 §8 첫 항목 참조.
