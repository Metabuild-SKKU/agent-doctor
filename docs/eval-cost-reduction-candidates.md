# Eval 채점(LLM-as-Judge) 비용 절감 후보

> 상태: **검토 문서** (draft). 실측·실험 근거와 함께 후보를 우선순위로 정리한다.
> 수치 기준: claude-haiku-4-5 심판, 7블록 fused 판정, top_k=5 × chunk_size=512.

## 기준 실측치

judge 1회(2026-08-11, KorQuAD 유사 규모 입력):

| 항목 | 값 |
|---|---|
| 입력 토큰 | ~6,420 (프리픽스 2,209 + user 4,211) |
| 출력 토큰 | ~1,474 |
| 비용 구성 | 입력 47% : **출력 53%** |
| 판정 1회 비용 | ~$0.0138 |
| 100 probe × 6라운드 환산 | ~$11 |

입력의 세부: 컨텍스트 48.1% · 고정 프리픽스 41.0% · 구조 4.9% · Q+A+R 6.0%.
출력의 세부: reason 41.4% · 구조 23.5% · 중복 문장 16.7% · 고유 문장 10.5% · 질문 생성 7.9%.

## 후보 목록 (우선순위순)

### 1. `correctness` 블록 인덱스화 — 총비용 ~7%, 위험 없음

`_fused_correctness_counts` 는 `len(TP)/len(FP)/len(FN)` 만 소비하는데, 모델은
TP/FP/FN 각각에 statement 전문 + reason 을 쓴다. `{"TP": [0, 2], "FP": [1], "FN": []}`
형태(인덱스 배열)로 바꾸면 정보 손실 0.

- 대상: `agents/eval/metrics_ragas.py` `_FUSED_BLOCKS["correctness"]` + `_FUSED_SCHEMA_PROPS`
- 주의: 프롬프트 문자열 스키마와 output_config 스키마의 대칭을
  `tests/test_ragas_fused.py` 가 핀으로 잡고 있음 — 두 쪽을 같이 수정

### 2. `reason` 필드 제거 — 총비용 ~27%, 재검증 진행 중

- 판정 응답의 `reason` 은 코드 어디서도 읽지 않는다 (`agents/` 전수 grep 확인).
  `diagnose.py` 의 `Finding.reason` 은 자체 생성 문자열로 무관.
- 스키마·few-shot 이 reason 을 verdict **앞**에 강제 → CoT 채널로 실재
  (원시 응답 키 순서 실측으로 확인). 즉 "필요 없다" 가 아니라 "효과가 측정되지 않았다".
- 1차 A/A/B 실험(20 probe): 효과 검출 안 됨 — A-vs-B 차이가 A-vs-A 잡음 바닥선 이내.
  단 한계 3개(키워드 폴백·11청크 코퍼스·프리픽스 reason 잔존)로 재검증 필요.
- **2차 실험(KorQuAD·dense 필수·clean 변형·A1/A2/B1/B2) 실행 중** — 결과로 이 문서 갱신 예정.

### 3. statement 중복 → 인덱스 참조 — 총비용 ~11%, 위험 낮음

`faithfulness_verdicts`/`recall_classifications` 가 `answer_statements` 의 문장 전문을
재출력한다(출력의 16.7%). `context_verdicts` 가 이미 쓰는 index 참조 패턴으로 통일.
index 파손 시 폴백도 `_fused_context_precision` 에 이미 구현돼 있어 재사용 가능.

### 4. Message Batches API — 총비용 −50%, 품질 변화 없음

판정은 probe별 독립 + 전체 대기 구조라 배치형 워크로드 그 자체. 50% 할인에
rate limit 문제(100 probe 동기 실행 시 분당 12만+ 입력 토큰)도 함께 해소.

- 대가: `core/llm_clients.py` 에 배치 제출·폴링 경로 추가, 라운드 지연 분 단위 증가
- **도입 시점: 100 probe 규모 확대 직전** (개발 중엔 동기 호출의 빠른 피드백이 유리)

### 5. rule-선별 게이트 — 판정 호출 ~25% 절감, 점수 의미 변경

rule 지표(recall@k·char_f1)가 명백히 통과시킨 probe 는 실제 트랙 RAGAS 를 생략.
현재는 전체 probe 에 실제 트랙이 돈다(오라클만 실패 한정).

- **단서**: RAGAS 평균이 overall/composite 에 들어가므로 "실패 probe 만의 평균" 은
  편향 — 생략 probe 의 점수 대체 규칙을 함께 정의해야 함
- **도입 시점: 성능 검증 단계 종료 후** (검증 중 측정 도구 변경 금지)

### 6. 프리픽스 내 스키마 문자열 제거 (anthropic 한정) — 몇 %

anthropic 경로는 output_config(607토큰)가 디코딩을 강제하므로 프롬프트 안의 스키마
문자열은 이중 전송. 단 openrouter/gemini 는 프롬프트 스키마에만 의존 → provider 분기 필요.

### 7. `RELEVANCY_STRICTNESS` 3→2 — 총비용 ~1.4%

질문 생성 3건(출력 7.9%)을 2건으로. 작지만 위험 거의 없음.

## 검토 후 기각

| 후보 | 기각 사유 |
|---|---|
| 프롬프트 캐싱 | 프리픽스 2,209 < haiku 캐시 최소 4,096 — 구조적으로 불가. 패딩 강제 시 실질 6%뿐. sonnet-5 전환은 캐시를 얻고도 2.6배 손해 (실측) |
| Q/A/R 글자수 제한 | Q+A+R 합계가 전체 입력의 6.0%(호출당 323토큰)뿐. 답변 절단은 faithfulness 왜곡, 정답 절단은 recall 상향 편향 |
| 컨텍스트(top_k·chunk) 축소 | 입력의 48%지만 **측정 대상 그 자체** — Optimize 탐색 공간이라 비용 사유로 고정 불가 |
| 전면 rule 전환 | 생성 원인(B)은 RAGAS 의존 — rule 만으론 `generation_failure` 롤업만 남아 Optimize 의 생성 측 처방이 전부 죽는다 (`types.py` tier 사다리 주석 참고) |
| thinking 활성화와의 교환 | 판정 транспорт 는 thinking disabled 고정. reason 이 유일한 사고 채널이라 맞교환 불가 |

## 권장 순서

```
지금:            1 (correctness 인덱스화)
              → 2 (reason 제거 — 재검증 결과 확인 후)
              → 검증 작업 계속 (측정 도구 동결)
100 probe 직전:  4 (배치 API)
검증 종료 후:     5 (rule-선별 게이트)
```

조합 추정 (100 probe · 통상 6라운드): $11 → 배치 $5.5 → +인덱스화 $5.1 → +reason $3.7.

## 참고: 대안 축

- deepseek 심판 유지 시 동일 실행 ~$1.4 (haiku 대비 1/8). Claude 를 쓰는 근거는
  비용이 아니라 스키마 강제·생성/심판 계열 분리 (.env 주석 참고).
- 2단 심판(deepseek 선별 → haiku 실패 확정): 판정비 ~45% 절감 가능하나
  선별 단계에 self-preference 편향 재유입 — 미검토.
