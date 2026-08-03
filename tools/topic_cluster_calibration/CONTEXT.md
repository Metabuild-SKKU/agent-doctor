# topic_cluster 임계값 캘리브레이션

## 배경

`agents/eval/topic_cluster.py` 의 `classify` 는 "검색 실패한 gold 청크들이 임베딩
공간에서 얼마나 뭉쳤나"를 코퍼스 baseline 대비 **비율(ratio)** 로 판정해
`concentrated` / `spread` / `none` / `unmeasured` 로 가른다. 이 신호는 Optimize
planner 가 처방을 가르는 데 쓰인다(뭉침 → 임베딩 교체, 안 뭉침 → 청킹 조정).

판정 경계는 원래 `agents/eval/types.py` 의 두 고정 상수였다(캘리브레이션 전):

```python
TOPIC_CLUSTER_CONCENTRATED_RATIO = 1.3   # ratio >= 이 값 → concentrated
TOPIC_CLUSTER_SPREAD_RATIO       = 1.1   # ratio <= 이 값 → spread
# 그 사이 → none
```

이 캘리브레이션 결과로 경계는 실패 gold 수 N 에 따라 동적으로 바뀐다(아래 실측 결과 참고):
`none 대 = 1.0 ± k·C/sqrt(N)`, `topic_cluster.dynamic_bounds(N)` 가 계산한다.

## 문제 (types.py TODO(eval-캘리브레이션))

두 값은 **캘리브레이션 전 임의값**이다. `none` 구간 폭(1.1~1.3 = 0.2)이 ratio
추정량의 분산보다 좁다. types.py 에 기록된 기존 실측:

| 실패 gold 수 | ratio 중앙값 | stdev |
|---|---|---|
| 10 | 1.04 | 0.93 |
| 20 | 1.00 | 0.50 |
| 100 | 1.06 | 0.21 |

즉 "주제 신호가 전혀 없는"(=`none` 이어야 할) 코퍼스에서도 실패 gold 가 보통
수십 개인 구간에선 stdev(~0.5)가 none 대 폭(0.2)보다 커서, 노이즈만으로도
`spread`/`concentrated` 로 튄다(기존 실측: 60회 중 none 은 7회뿐).

지금은 `spread`/`concentrated` 가 같은 처방(임베딩 교체)이라 그 둘 사이 오분류는
무해하지만, **none ↔ (spread/concentrated) 경계 오분류는 비싼 재색인을 잘못
발동**시킨다. 그래서 PR #65 에서 이 신호의 소비를 유예(관측용)했고, 재개 조건이
바로 이 캘리브레이션이다.

## 목표

실측 ratio 분포를 근거로 두 경계값을 다시 정한다. TODO 가 제시한 두 방향:

- **(a) none 대 폭을 분산에 맞춰 넓힌다** — 예: 중앙값 ±k·stdev
- **(b) 실패 gold 수에 따라 폭을 동적으로 조절** — 표본 적으면 더 관대하게

## 접근 (로컬 스크립트, LLM 0)

`classify` 는 임베딩 코사인만 쓰므로 LLM/외부 의존이 없다. 여러 코퍼스에서
"신호 없음(null)" 상황을 만들어 ratio 분포를 실측한다.

1. **null 분포 수집**: gold 정합이 맞는 코퍼스(`tests/corpus/qa.json`,
   KorQuAD taxonomy 등)의 청크 임베딩에서, 실패 gold 를 **무작위로** 뽑아
   "주제 신호 없음"을 인위적으로 만든다. 실패 gold 수(N)를 바꿔가며
   `classify_detail` 의 ratio 를 여러 번(seed 다르게) 수집.
2. **분포 통계**: N 별 ratio 중앙값·stdev·분위수를 낸다(types.py 표 재현·확장).
3. **경계 제안**: null 분포의 상/하위 분위수(예: p95/p5)를 경계 후보로 제시.
   신호 있는 코퍼스(같은 도메인 gold 만 뽑기)와 대비해 분리도 확인.

**gold 정합 주의(메모리)**: `eval_probes.json` 은 gold_spans=[] 라 recall/gold 가
거짓 0 → ratio 오염. 반드시 gold 정합 코퍼스(`tests/corpus/qa.json`,
run_corpus 경로, KorQuAD)에서만 뽑는다.

## 재사용 자원

- `agents/eval/topic_cluster.py`: `classify_detail`, `stride_sample`, `_valid`,
  `_mean_pairwise_cosine`, `_baseline_cohesion` (수치 계산 전부 여기 있음)
- `TOPIC_CLUSTER_BASELINE_SAMPLE` 등 상수(types.py)
- 결정성: 무작위 표본은 seed 고정(스크립트가 seed 를 인자로 받아 재현 가능하게)

## 산출물

- `tools/topic_cluster_calibration/collect_ratio_distribution.py` — 분포 수집·통계
- 결과 리포트(분포 표 + 경계 제안). 확정 경계값은 types.py 반영.

## 실측 결과 (확정)

코퍼스: 세금가이드 PDF(`tests/corpus/260508…펼침면.pdf`)를 정규 파이프라인으로 색인한
597청크(`chunks.json` 포맷, BGE-M3 1024-dim). types.py 표의 문서 10×50=500청크 기준을 충족.

- **null 분포**(무작위 실패 gold): 중앙값이 N 무관하게 1.00 에 붙고(baseline 편향 없음),
  stdev 는 `C/sqrt(N)` 로 규칙적으로 감소 — `stdev·sqrt(N) ≈ 0.10~0.13`(평균 C≈0.118).
    N=5→0.058  N=10→0.040  N=20→0.026  N=40→0.018  N=100→0.010
- **신호 분포**(같은 주제 인접 블록): 중앙값 N=10→1.19, N=20→1.15, N=40→1.09 로 >1 쏠림.
- **옛 고정 경계(1.1~1.3) 문제**: null 의 99%가 none 밖(대부분 spread)으로 오분류.
- **확정: (b) 동적 조절** — none 대 = `1.0 ± k·C/sqrt(N)`, `C=0.118`, `k=2.33`.
  null 누출(=비싼 재색인 오발) 1~4%, 신호 검출 91~97% 로 균형(k=1.645 는 누출 7~11%).
  `topic_cluster.dynamic_bounds(N)` 가 계산하고, types.py 는 계수 `TOPIC_CLUSTER_NULL_C`
  / `TOPIC_CLUSTER_BOUNDARY_K` 만 둔다.

## 범위 밖 (이 PR 아님)

- 소비 재개(`planner._CONSUME_TOPIC_CLUSTER_SIGNAL = True`) — 임베딩 교체 실행
  (capability 활성화 + 검증 모델 후보)까지 준비돼야 하며 별도 과제.
- 임베딩 교체 capability 자체.
