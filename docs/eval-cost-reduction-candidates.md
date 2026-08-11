# Eval 채점(LLM-as-Judge) 비용 절감 — 검토한 전체 방안

> 상태: **검토 문서** (draft). 2026-08-11 검토에서 다룬 모든 방안을 채택/보류/기각
> 구분 없이 전부 기록한다. 수치 기준: claude-haiku-4-5 심판, 7블록 fused 판정,
> top_k=5 × chunk_size=512, KorQuAD 475청크.

## 기준 실측치

judge 1회:

| 항목 | 값 |
|---|---|
| 입력 토큰 | ~6,340 (프리픽스 2,209 + user ~4,130) |
| 출력 토큰 | ~1,634 |
| 비용 구성 | 입력 47% : **출력 53%** |
| 판정 1회 비용 | ~$0.0138 |
| 100 probe × 통상 6라운드 환산 | ~$11 |

- **입력 분해**: 컨텍스트 48.1% · 고정 프리픽스 41.0% · 구조 4.9% · Q+A+R 6.0%
- **출력 분해**: reason 41.4% · 구조(키·괄호) 23.5% · 중복 문장 16.7% · 고유 문장 10.5% · 질문 생성 7.9%
- `output_config` 스키마 607토큰은 별도로 매 호출 전액 과금(캐시 불가 필드)

---

## A. 출력 축 (비용의 53%)

### A-1. `reason` 필드 제거 — 판정비용 **−21.3% (실측)**, 컨텍스트 지표 +0.05 관대화

- 판정 응답의 `reason` 은 코드 어디서도 읽지 않는다 (`agents/` 전수 grep).
  `diagnose.py` 의 `Finding.reason` 은 자체 생성 문자열로 무관.
- 단 스키마·few-shot 이 reason 을 verdict **앞**에 강제 → CoT 채널로 실재
  (원시 응답 키 순서 실측 확인: 스키마 선언 순서와 생성 순서 완전 일치).

**2차 재검증 (KorQuAD 475청크 · dense 폴백 0 · clean 변형 · A1/A2/B1/B2 각 20 probe):**

| | A(유지) | B(제거) | 실측 |
|---|---|---|---|
| 출력/호출 | 1,634 | 1,096 | **−33.0%** |
| 입력/호출 | 6,338 | 5,943 | −6.2% |
| 판정 비용 | | | **−21.3%** |
| 파싱 실패(max_tokens 절단) | 3/40 | **0/40** | 재요청 비용도 절감 |

품질 (자기일치 바닥선 A-A 0.044·B-B 0.027 대비):

- `faithfulness`·`correctness`: 바닥선 이내 — 효과 없음
- `context_precision` +0.03~0.06 / `context_recall` +0.05~0.06 — **일관된 상향(관대화)**.
  두 쌍 모두 같은 방향, 1차 실험과도 일치. reason 강제가 없으면 attributed=1 을 더 쉽게 준다.
- 부수 발견: B 쪽 자기일치도가 오히려 높다 (kappa 0.918 vs 0.899)

**선택지**: (a) 전면 제거 + `RAGAS_*_MIN` 임계값 재캘리브레이션 /
(b) 관대화가 확인된 `context_verdicts`·`recall_classifications` 만 유지, 나머지 제거(~12–15% 추정).
후보 B-2 도입 시 두 블록이 rule 로 대체되므로 (b)는 자연 소멸. **권장: (b)**.

### A-2. `correctness` 블록 인덱스화 — 총비용 ~7%, 위험 없음

`_fused_correctness_counts` 는 `len(TP)/len(FP)/len(FN)` 만 소비하는데, 모델은
TP/FP/FN 각각에 statement 전문 + reason 을 쓴다. `{"TP": [0, 2], "FP": [1], "FN": []}`
(answer/reference_statements 의 인덱스 배열)로 바꾸면 정보 손실 0.

- 대상: `agents/eval/metrics_ragas.py` `_FUSED_BLOCKS["correctness"]` + `_FUSED_SCHEMA_PROPS`
- 주의: 프롬프트 문자열 스키마 ↔ output_config 스키마 대칭을 `tests/test_ragas_fused.py` 가
  핀으로 잡음 — 두 쪽 동시 수정

### A-3. statement 중복 → 인덱스 참조 — 총비용 ~11%, 위험 낮음

`faithfulness_verdicts`/`recall_classifications` 가 `answer_statements` 문장 전문을
재출력한다(출력의 16.7%, 고유 문장 8개가 출력에 ~24회 등장). `context_verdicts` 가
이미 쓰는 index 참조 패턴으로 통일. index 파손 폴백은 `_fused_context_precision` 에
구현돼 있어 재사용 가능.

### A-4. `RELEVANCY_STRICTNESS` 3→2 — 총비용 ~1.4%

relevancy 생성 질문 3건(출력 7.9%)을 2건으로. 코사인 평균의 표본이 줄어드는 것 외
위험 거의 없음.

### A-5. `EVAL_RAGAS_FUSED_MAX_TOKENS` 조정 — 절감 아님 (주의사항)

출력 상한(4096)은 실사용분만 과금되므로 낮춰도 절감이 아니다. 오히려 reason 유지
상태에서는 절단 → 파싱 실패 → 재요청(refetch)으로 **비용이 늘 수 있다** (실측: A 변형
40회 중 3회 절단). A-1 적용 시 절단이 사라진다.

---

## B. 판정 자체를 줄이거나 대체

### B-1. rule-선별 게이트 — 판정 호출 ~25% 절감, 점수 의미 변경

rule 지표(recall@k·char_f1)가 명백히 통과시킨 probe 는 실제 트랙 RAGAS 를 생략.
현재는 전체 probe 에 실제 트랙이 돈다(오라클 트랙만 실패 한정).

- **단서**: RAGAS 평균이 overall/composite 에 들어가므로 "실패 probe 만의 평균"은
  상향 편향 — 생략 probe 의 점수 대체 규칙을 함께 정의해야 함
- **도입 시점: 성능 검증 단계 종료 후** (검증 중 측정 도구 변경 금지)

### B-2. `context_precision`/`context_recall` 을 gold 기반 rule 로 대체 — ~20% (추정)

두 블록은 파이프라인이 이미 가진 정보(`gold_chunk_ids`, 20건 중 19건 보유)를 LLM 에게
자연어로 되묻는 구조. gold 포함 여부 + `_average_precision` 으로 rule 계산 가능.
블록 7→5로 프리픽스·출력 동반 감소.

- 의미 변경: "의미적으로 유용한가"(LLM) → "gold 를 포함하는가"(rule) — 더 엄격해짐.
  `composite_score ≥ 90` 통과 기준·과거 실행 비교성에 영향
- gold 없는 probe 는 LLM 블록 폴백 필요 (`want_prec` 가드에 붙일 자리 있음)
- A-1 재검증에서 관대화가 확인된 지표가 정확히 이 둘 — 대체 시 관대화 문제도 소멸

### B-3. 전면 rule 전환 — **기각**

생성 원인(B계열 라벨)은 전부 RAGAS 의존(`types.py` tier 사다리). rule 만으론
예비 `generation_failure` 롤업만 남고, planner 는 예비 finding 을 자동 처방에서
제외하므로 **Optimize 의 생성 측 처방이 전부 죽는다**. 어휘 매칭의 원리적 한계도
저장소가 이미 문서화 (`metrics_basic.py`: '사망'⊂'사망하지 않았다' recall=1.0).
비용은 1.5% 수준까지 떨어지나 진단 도구의 정체성을 잃는 거래.

---

## C. 전송·실행 구조

### C-1. Message Batches API — 총비용 **−50%**, 품질 변화 없음

판정은 probe별 독립 + 전체 대기 구조라 배치형 워크로드 그 자체. 50% 할인에
rate limit(100 probe 동기 실행 시 분당 12만+ 입력 토큰) 문제도 함께 해소.

- 대가: `core/llm_clients.py` 배치 제출·폴링 경로, 라운드 지연 분 단위 증가
- **도입 시점: 100 probe 규모 확대 직전** (개발 중엔 동기 호출이 유리)

### C-2. `EVAL_MODE` 라운드 혼용 — 스윕 라운드 판정비 → 0

Optimize 내부 sweep 라운드는 `standard`(rule·재검색만), 최종 판정 라운드만 `deep`.
스위치는 이미 존재. 단 라운드 간 라벨 확정 수준이 달라져 점수 비교성 확인 필요.

### C-3. 라운드 간 probe 단위 판정 캐시 — 처방 종류에 따라 가변

현재 eval 캐시는 스냅샷 전체 단위(`EvalSnapshot`)라 config 가 바뀌면 전부 무효.
생성 측 처방(temperature 등)은 검색 결과가 동일하므로 `(answer, contexts, reference)`
해시 단위 캐시면 검색 불변 probe 의 판정을 재사용 가능. 효과는 Optimize 가 고르는
처방 분포에 좌우.

### C-4. probe 수·반복 상한 조정 — 비용의 지배 변수

비용은 probe 수 × 라운드 수에 선형. `EVAL_TESTSET_SIZE`, `max_iterations`(5),
`max_optimize_visits`(20)가 곱해지는 자리. 절감 기법이 아니라 **규모 설계** 문제 —
최악 시나리오(21라운드)는 통상(6라운드)의 3.5배.

---

## D. 입력 축 — 대부분 기각

### D-1. 프롬프트 캐싱 — **기각 (구조적 불가)**

- 캐시 대상은 system 블록(프리픽스 2,209토큰)뿐인데 haiku 캐시 최소가 4,096 —
  경고의 "이번 입력 6,382"는 요청 전체라 오해 소지 (프리픽스는 그중 35%)
- few-shot 패딩으로 4,096 을 강제 충족 시: 실질 절감 6% (프리픽스 몫 34% × 캐시
  할인 − 패딩 비용). 판정 기준까지 바뀌는 대가
- sonnet-5(캐시 최소 1,024) 전환 시: 캐시가 걸려도 단가 3배가 이겨 **2.6배 손해** (실측)
- 이론 천장 자체가 총비용 14.5% — 캐시가 완벽해도 출력 레버보다 작다

### D-2. 프리픽스 내 스키마 문자열 제거 (anthropic 한정) — 몇 %

anthropic 은 output_config(607토큰)가 디코딩을 강제하므로 프롬프트 안의 스키마
문자열은 이중 전송. 단 openrouter/gemini 는 프롬프트 스키마에만 의존 → provider
분기 필요. 코드 주석도 프롬프트 스키마의 무강제성을 인정.

### D-3. Q/A/R 글자수 제한 — **기각**

Q+A+R 합계가 전체 입력의 6.0%(호출당 323토큰: Q 78 · A 186 · R 58)뿐 — 절반으로
줄여도 총비용 1.4%. 게다가 답변 절단은 faithfulness 를 왜곡(잘린 부분의 환각 미탐),
정답 절단은 recall 상향 편향(잘린 요소가 "요구되지 않은 것"이 됨).

### D-4. 컨텍스트(top_k × chunk_size) 축소 — **기각**

입력의 48.1%로 가장 크지만 **측정 대상 그 자체**. top_k·chunk_size 는 Optimize 가
스윕하는 파라미터라 비용 사유로 고정하면 탐색 공간을 없앤다. 역으로 탐색이 이 값을
키우면 판정비가 비례 상승(top_k 10 × chunk 1024 = 입력 2.2배) — 견적 시 유의.

### D-5. 검색 컨텍스트 압축 재사용 — 기각 + **잠재 버그 메모**

`_compress_contexts_for_question`(generator 내부, 기본 off)은 심판 경로에 안 걸린다.
심판에 적용하는 것은 측정 대상 변경이라 기각. 별개로: **Optimize 가 압축 처방을 켜면**
생성기는 압축본, 심판은 원본을 보게 되어 faithfulness 가 후해지는 비대칭이 생긴다 —
절감과 무관하게 확인 필요.

---

## E. 모델 선택 축

### E-1. deepseek 심판 유지 — 동일 실행 ~$1.4 (haiku 의 1/8)

Claude 전환의 근거는 비용이 아니라 (1) 생성/심판 계열 분리, (2) JSON 스키마 강제
(`.env` 주석). 그 가치가 차액(통상 실행 기준 ~$10)인지가 본질적 질문.

### E-2. 2단 심판 (deepseek 선별 → haiku 실패 확정) — 판정비 ~45% (추정)

전 probe 를 deepseek 으로 채점, 실패 판정만 haiku 재확정. 선별 단계에 self-preference
편향 재유입(생성 모델과 동일 계열이 1차 판정) + 두 심판의 판정 경계 차이로 라벨 분포
변동 가능. 미검토 상태로 기록만.

### E-3. thinking(추론 토큰)과의 교환 — **기각**

reason 과 thinking 은 같은 메커니즘(답 확정 전 토큰 소비)이나 transport 가 판정
호출에서 thinking disabled 를 고정하며 켜는 경로 자체가 없다. reason 이 유일한 사고
채널 — 맞교환 불가. reason 은 항목별 국소·짧게 통제·응답에 남는 형태라 21건 독립
판정에는 전역 사고 블록보다 적합하다는 점도 기록.

### E-4. fable/mythos 계열 — **기각**

thinking 을 끌 수 없어(400) 출력 상한 25K 로 강제 상향 + 사고 토큰 전액 과금.
단가도 $10/$50. 판정용으로 부적합.

---

## F. 이미 구현되어 있는 절감 (현상 유지 확인)

| 항목 | 내용 |
|---|---|
| fused 단일 호출 | 지표 7개를 chat 1회로 통합 — 구 지표별 호출 대비 probe당 23.7→1.6회 |
| 오라클 트랙 실패 한정 | 실패 probe 에만 오라클 답변 생성·판정 |
| EvalSnapshot 캐시 | 동일 config 복귀 시 답변 생성·RAGAS 없이 복원 (LRU 2개) |
| 고정 테스트셋 (`EVAL_PROBE_SOURCE=made`) | probe 재생성 비용 0 + 실행 간 비교성 |
| Index 그래프 LLM off | `auto` 는 OPENAI_API_KEY 만 봄 — **`INDEX_LLM_PROVIDER=openrouter` 를 켜면 청크당 1회(4,330청크 = 라운드당 4,330호출)가 추가되므로 켜지 말 것** |
| 임베딩 | bge-m3 OpenRouter $0.01/1M — 색인 1회 ~$0.02, 전체 실행 $0.1 안팎으로 무시 가능 |

---

## 권장 순서

```
지금:            A-2 (correctness 인덱스화, 위험 0)
              → A-1(b) (reason 부분 제거 — context 블록 2개만 유지)
              → 검증 작업 계속 (측정 도구 동결)
100 probe 직전:  C-1 (배치 API — 비용 반값 + rate limit 해소)
검증 종료 후:     B-1 (rule-선별 게이트) · B-2 (context 지표 rule 대체)
```

조합 추정 (100 probe · 통상 6라운드): $11 → 배치 $5.5 → +A-2 $5.1 → +A-1 ~$3.7–4.2.
