# 논문 기반 업그레이드 로드맵

최신 RAG 논문 조사(2026-08-06)와 실측 로그 분석에서 나온 개선 항목 8개.
**순서가 결론이다** — 아래 "왜 이 순서인가"를 먼저 읽을 것.

근거 실행: `output/logs/corpus_20260804_103059.txt` (30문항 · 5반복 · KorQuAD 20문서)

---

## 현황

| # | 항목 | 출처 | 상태 |
|---|---|---|---|
| 1 | 평가 문항 30 → 150 | RAISE | ✅ 문서화 완료 (`.env.example`·`agents/eval/README.md`) |
| 2 | 같은 config 3회 반복해 σ 측정 | RAISE | ✅ `tools/measure_eval_noise.py` — **실측 미실행** |
| 3 | 라벨 1개 추가 검토 (E15) | RAGEC | 🔒 **B그룹 verifier 설계 선행** — E14 는 접음 |
| 4 | **채점을 부분점수로** | RAGChecker | ✅ **PR #122 머지** |
| 5 | 흔들리는 심판 신호가 라벨을 뒤집지 못하게 | 심판 감사 | ✅ **PR #125 머지** |
| 6 | 개선 마진을 통계로 | Noisy but Valid | 🔒 2번 실측 필요 |
| 7 | 처방 선택을 밴딧으로 | AutoRAG-HP | 🔒 4·6 필요 (**근거 하나 철회** — 7번 절 참고) |
| 8 | 라벨 정확도 측정 | Doctor-RAG | 🔒 코퍼스 배관 필요 |

### 재베이스라인은 한 번에

**#114 · #122 · #125 가 전부 점수를 올린다.** 셋이 함께 들어간 뒤 실측을 한 번만 돌리는 게
맞다 — 따로 들어가면 매번 이전 실행과의 비교가 끊긴다.

| PR | 무엇이 점수를 올리나 |
|---|---|
| #114 | 골드 좌표 정정 — 어긋난 골드가 만들던 거짓 실패가 사라진다 |
| #122 | recall 부분점수 — 0 이던 probe 가 부분점수를 받는다 |
| #125 | 예비 골드 오류도 점수 제외 — 제외 범위가 넓어진다 |

---

## 왜 이 순서인가

**지금 종합점수는 노이즈와 구분되지 않는다.** 그래서 무엇을 고쳐도 좋아졌는지 측정할 수 없다.

```
반복 0 : 종합 75   (리랭커 off)
반복 5 : 종합 73   (리랭커 off, rerank_candidates 만 다름)
```

리랭커가 꺼지면 `rerank_candidates` 는 검색에 관여하지 않으므로(판정창에만 쓰임) 두 반복은
기능적으로 같은 config 다. 편차 0.020 = `MIN_IMPROVEMENT_MARGIN`(0.020).

같은 실행에서 살아남은 처방 하나가 `+0.020 ≥ 마진 0.020` 턱걸이였다.

→ **1·4 를 먼저 해야 나머지를 잴 수 있다.** 7(밴딧)은 보상 신호가 노이즈면 학습 자체가 안 된다.

---

## 항목별

### 1. 평가 문항 30 → 150 ✅

RAISE 가 13개 탐색 알고리즘을 벤치마크하며 프록시 크기를 ablation 했고, **100~200 예제에서
안정**된다고 보고한다(그 아래는 시드 간 편차가 큼). 우리는 30이었다.

`KORQUAD_MAX_DOCS` 가 먼저 문서를 자르고 그 안에서 QA 를 세므로 둘을 같이 올려야 한다.
정제본 기준 문서 267 개 안에 QA **179 건**이 들어온다(실측). 즉 150 은 문서를 더 늘리지
않아도 확보된다.

**현재 실행값은 100 이다**(`KORQUAD_QA_LIMIT`). 표본이 아니라 **비용** 때문에 낮춰 잡았다 —
심판 호출이 QA 개수에 선형이라 150 이면 1.5 배다. RAISE 가 말한 안정 구간(100~200)의
하한이라 목적(30 → 세 자리)은 달성했고, 필요하면 179 까지 설정만 바꾸면 된다.

> **문서 수와 QA 수는 비용 성격이 다르다.** 문서를 늘리면 인덱싱(임베딩)이 늘지만
> `index_cache` 가 재사용하므로 **한 번**이고, 채점 비용은 top_k 가 고정이라 안 늘어난다.
> 반면 QA 를 늘리면 **매 라운드** 심판 호출이 늘어난다. 규모를 키울 때는 문서를 먼저,
> QA 는 필요한 만큼만 올리는 게 맞다.

### 2. σ 측정 도구 ✅ (실측 미실행)

`history.max_repeated_measurement_spread()` 가 정상 실행에서 σ 를 줍도록 설계돼 있으나,
롤백이 Eval 을 재실행하지 않고 진단 캐시를 복원해(로그의 "롤백 진단 캐시 복원") 같은 config
가 두 번 측정되는 일이 없다 → 항상 `None`. 그래서 전용 도구를 뒀다.

```powershell
python tools/measure_eval_noise.py -n 3
```

150문항 3회 ≈ 75분 / $0.55. **비용이 드는 이유**: 재려는 흔들림이 LLM 에서 나오므로 LLM 을
실제로 불러야만 관측된다.

**무엇을 얻고 무엇을 못 얻는지 분명히 해둔다.** n=3 은 σ 를 확정하지 못한다 — 표본 3개면
σ 추정의 오차가 σ 자체만 하다. 이 측정이 답하는 것은 **"노이즈가 마진(2점)보다 큰가"** 하나다.

| | |
|---|---|
| 얻는 것 | 노이즈가 마진보다 큰가 · 문항을 150 으로 올린 게 효과가 있었나 |
| 못 얻는 것 | 정확한 σ · "마진을 3.4점으로" 같은 값 |

RAISE 도 시드 3개를 쓴다. 통계적으로 이상적이어서가 아니라 감당 가능한 표준이라서다.

**"롤백 때 재측정하면 σ 가 공짜로 쌓인다"는 계산 착오였다(철회).** 롤백 3건이면 Eval 이
3회 더 붙고, 그게 **실행할 때마다** 반복된다. 전용 측정은 $0.54 를 한 번 쓴다. 한 번 재는
쪽이 싸다.

### 3. 라벨 추가 검토 ⏸ — E14 는 접고 E15 만 남긴다

RAGEC 택소노미(16개)와 우리 31개를 대조한 결과 **이름까지 같은 게 6개**로 외부 검증됐다.
우리에게 없는 것이 둘이었고, 기존 원칙(**처방이 기존 라벨과 다르면 새 라벨, 겹치면 아니다**
— `agents/optimize/rules.py` 의 라벨 도입 합의)으로 판정했다.

**E14 Contextual Misalignment**(답은 맞는데 질문에 답하지 않음) — **도입하지 않는다.**

| 겹침 후보 | 처방 | 판정 |
|---|---|---|
| `generation_partial_answer` | `completeness_prompt` | 다름 — 그쪽은 "덜 답함", E14 는 "다른 걸 답함" |
| `generation_misinterpretation` | `restate_question` | **사실상 동일** — 둘 다 "질문을 못 알아들었으니 다시 진술시킨다" |

**E15 Chronological Inconsistency**(시간 순서를 뒤바꿈) — **지금은 판정할 수 없다.**

> 예: "임진왜란과 정유재란 중 뭐가 먼저인가?" → 근거에 1592년·1597년이 **둘 다 있는데**
> 답변이 순서를 뒤집는다. 각 사실은 근거에 있고(faithfulness 안 낮음) 숫자도 안 틀렸는데
> **관계**만 틀린 경우다.

우리 라벨 셋에 걸쳐 있다.

| 기존 라벨 | 처방 | E15 의 어느 부분을 덮나 | 상태 |
|---|---|---|---|
| `generation_contradiction` | `llm_verification_pass` | 앞뒤가 모순되는 경우 | **draft** |
| `generation_numerical_error` | `enable_calculation_check` | 연도·날짜가 숫자로 틀린 경우 | **draft** |
| `generation_hop_binding_error` | `force_hop_evidence_binding` | 여러 사실을 잘못 엮은 경우 | **draft** |

판정 기준은 하나다 — **"시간축 검증 단계"가 이 셋과 다른 처방인가.**

#### 그런데 지금은 그 비교가 성립하지 않는다

비교 대상 셋이 **전부 `status: "draft"`** 이고, 공교롭게도 **B그룹 draft 전부**가 정확히
이 셋이다(`agents/optimize/rules.py`). 셋 다 같은 벽에 막혀 있다.

```
generation_contradiction     BLOCKER: generation_config 없음 + verifier 노드 자체가 미구현
generation_numerical_error   BLOCKER: generation_config 필드 및 calculation checker 단계 없음
generation_hop_binding_error BLOCKER: generation_config 필드 및 verifier 단계 없음
```

즉 **존재하지 않는 처방과 존재하지 않는 처방을 비교**하는 상황이다. 어느 쪽으로 답해도
근거가 없다. 게다가 셋 다 결국 B그룹 공통 노드(`evidence_mapper`/`generation_verifier`/
`revision`, 설계 초안 단계) 위에 얹힐 예정이라 — **그 노드가 "검증 종류"를 어떻게 나누는지가
정해져야** 시간축이 별도 단계인지 그 안의 한 종류인지 말할 수 있다. 지금 미리 정하면 그
설계를 제약하기만 한다.

#### 그리고 지금은 재볼 수도 없다

현재 평가셋에 시간 순서 질문이 **0건**이다(아래 "KorQuAD 로는 라벨 검증이 불가능하다" 절).
도입해도 발화하지 않으므로 맞는지 틀린지 확인할 방법이 없다 — 이미 31개 중 19개가 같은
이유로 미발화인데 20번째를 더하는 셈이다.

#### 판정이 가능해지는 조건

둘 다 충족돼야 한다.

| | 조건 | 현재 |
|---|---|---|
| ① | `generation_verifier` 노드 설계 확정 → 검증 종류를 어떻게 나누는지 정해짐 | 설계 초안 |
| ② | 평가셋에 시간 순서 질문 존재 → 발화를 실제로 관측 | 0건 |

**①이 사실상 상위 항목이다.** B그룹 draft 3개가 한꺼번에 풀리는 자리이므로, 그때
"시간축을 네 번째 검증 종류로 넣을까"를 자연스럽게 같이 결정하게 된다.

> **E14 와 성격이 다르다.** E14 는 "겹친다"가 **확인된** 판정이고, E15 는 **판정 자체가
> 불가능한** 상태다. 접은 게 아니라 조건부 보류다.

### 4. 채점을 부분점수로 🔵 PR #122

**우리 최대 결함.** `span_recall_at_k` 는 골드 구간을 빈틈없이 덮어야 1점인 이진 판정이라,
정답을 맞힌 실행도 recall=0 이 된다. 실측 **30문항 중 14건(47%)**.

RAGChecker(NeurIPS 2024)는 골드와 응답을 **원자 claim** 으로 쪼개고 entailment 로 대조한다.
recall 이 자연히 연속값이 된다(덮은 claim / 전체 claim). 귀속 원리도 함께 온다.

> 빠진 claim = 검색 결함 · 근거 없는 claim = 생성 결함

부수 효과: `tools/build_clean_qa.py` 가 "정답이 여러 곳이라 모호"로 버린 **1,169건**을
부분점수로는 회수할 수 있다.

#### 설계 갈림길은 없다 — LLM 이 필요 없다

처음엔 "골드 분해를 LLM claim 추출로 할지 규칙 기반으로 할지"가 갈림길이라고 봤는데,
**코드를 보니 갈림길이 아니다.** 두 가지 이유다.

**① 답변 축은 이미 claim 단위로 쪼개져 있다.**

```python
# metrics_ragas.py:1113 — RAGAS 가 이미 계산한다
out_counts = {"answer_correctness_tp": tp,   # 맞은 골드 요소
              "answer_correctness_fp": fp,   # 군더더기
              "answer_correctness_fn": fn}   # 빠뜨린 골드 요소
```

RAGChecker 의 claim 분해와 같은 것이고, 커버리지 `TP/(TP+FN)` 는 `types.blend_answer_score`
에서 이미 쓰고 있다. 여기에 새로 만들 것이 없다.

**② 0점이 나는 곳은 답변이 아니라 검색이고, 그건 문자 좌표 산수다.**

```python
# metrics_basic.py:339-341 — 여기가 원인이다
        if cursor >= end:
            covered += 1          # span 하나를 통째로 덮어야 1
    return covered / len(valid_spans)
```

골드 구간(문자 좌표)을 빈틈없이 덮었는지를 0/1 로 센다. 텍스트를 이해할 필요가 없으므로
**부분점수도 LLM 없이 낼 수 있다.**

```
지금   : 골드 구간을 다 덮었으면 1, 아니면 0
바꾸면 : 덮은 문자 수 / 골드 전체 문자 수
```

결정론적이고 비용 0 이다. 함수 하나(`span_recall_at_k`)를 고치는 일이다.

### 5. 흔들리는 심판 신호가 라벨을 뒤집지 못하게 🔵 PR #125

> 처음엔 "`faithfulness` 를 게이트에서 제외" 라고 적었는데 **그 표현은 틀렸다.** 이 신호에는
> 정당한 역할이 있다. 문제는 신호의 존재가 아니라 **흔들리는 신호 하나가 결정론적 신호 셋을
> 뒤집는 구조**다.

`bad_gold_chunk` 가 붙으려면 조건 4개를 통과해야 한다(`diagnose.py:1414-1422`).

| 조건 | 신호 | 재실행하면 같은 값? |
|---|---|---|
| 오라클 컨텍스트가 있다 | `oracle_answer` | ✅ |
| 골드로는 답이 안 나온다 | `oracle_f1`(글자 비교) | ✅ |
| 실제 답은 맞았다 | `f1_score`(글자 비교) | ✅ |
| **답이 검색 근거에 붙었다** | **`faithfulness`(심판 LLM)** | ❌ |

`probe_qa_4195` 를 5회 전부 추적하면 답도 검색 결과도 같은데 **반복 3에서만** 네 번째가
1.000 → 0.000 으로 튀어 라벨이 통째로 사라졌다. 그 재분류가 처방 `rerank_candidates 20→22`
를 만들었고 KEEP 판정으로 최종 config 에 남았다(`context_precision` 은 0.78 → 0.73 인데도).

**네 번째 조건을 그냥 없애면 안 된다.** 주석이 이유를 적고 있다 — *"답이 검색 근거에 안
붙으면(parametric 등) 골드 단정 불가"*. 모델이 검색 결과가 아니라 자기 기억으로 답했다면
"골드가 틀렸다"고 단정할 수 없다. 정당한 가드다.

#### 채택: A(예비로 강등) — 구현 결과

세 선택지를 놓고 **A** 를 택했다.

| | 방법 | 판정 |
|---|---|---|
| **A** | 라벨을 **없애는** 대신 `confirmed=False`(예비)로 **강등** | **채택** |
| B | 조건을 **글자 기반** grounding 으로 교체 | 보류 — 새 신호를 설계해야 한다(현재 grounding 판정이 전부 `faithfulness` 기반: `_grounded_ok`·`_grounded_verified`·`_retrieval_verified_grounded`) |
| C | 심판을 여러 번 불러 중앙값 | **불가** — 같은 모델 배심원단은 에러 상관 ρ=0.944~0.972 라 평균이 분산을 못 줄인다(arXiv 2607.08535) |

구현하면서 설계가 두 번 좁아졌다. **둘 다 기존 테스트가 잡았다.**

| `faithfulness` | 라벨 | 억제 범위 | 점수 |
|---|---|---|---|
| 문턱 이상 | `bad_gold_chunk` **확정** | 검색·생성·컨텍스트 전부 | 제외 |
| 측정됐고 미달 | `[예비] bad_gold_chunk` | **검색만** | 제외 |
| **미측정**(DEEP 미만) | 라벨 없음 | — | — |

- **미측정은 예비로도 세우지 않는다.** `None` 은 "근거가 없다"가 아니라 "RAGAS 를 안 돌렸다"
  이다. 여기서 예비를 세우면 DEEP 미만 실행 **전체**가 검색 진단을 잃는다
  (`test_critical_findings_do_not_move_reliability` 가 STANDARD 모드에서 깨져 발견).
- **예비는 검색만 막는다.** 오라클이 스스로 근거 없이 답한 경우(`generation_hallucination`)는
  골드가 아니라 **생성기**에 대한 증거라 살려 둔다
  (`test_parametric_does_not_mask_oracle_generation_failure` 가 깨져 범위를 좁혔다).

#### 점수 제외를 예비까지 넓혔다

`report.is_gold_labeling_error` 에서 `confirmed` 요구를 뺐다. 확정만 빼면 **제외 여부가 심판
노이즈를 그대로 탄다** — 같은 probe 가 5회 중 4회는 제외되고 튄 1회만 실패로 집계되면서
종합점수를 흔든다. 우리가 고치려는 게 바로 그 흔들림이다.

제외의 근거는 결정론적인 쪽에 있다 — `_f1_ok` 와 `not _oracle_ok` 로 "평가셋 결함" 판단은
이미 서 있고, `faithfulness` 는 **왜** 그런지만 가른다(그건 확정/예비로 표시된다).

### 6. 개선 마진을 통계로 🔒

2번 실측 후. **다만 2번이 정확한 σ 를 주지 못한다는 점을 전제로 설계해야 한다**(2번 절 참고).
얻는 것은 "노이즈가 마진보다 큰가"라는 예/아니오이지 σ 값이 아니다.

| | 방법 | 비용 |
|---|---|---|
| **A** | 마진 상수를 올린다(관측 폭 기준) | **0** |
| **B** | "회차 평균으로 판정" — 판정 방식 자체를 바꾼다 | **매 실행 3배, 영구** |

B 가 근본이지만 비용이 영구적이다. σ 를 정확히 모르는 상태에서는 A 를 보수적으로
잡는 편(관측 폭의 1.5배 — 통계적 근거가 아니라 정책값이다)이 현실적이다.

Noisy but Valid(arXiv 2601.20913)가 보정셋으로 심판 TPR/FPR 을 추정해 분산 보정 임계값을
만드는 방법을 제시한다. ⚠️ **원문 미확인**(PDF 용량 초과) — 채택 전 직접 읽을 것.

### 7. 처방 선택을 밴딧으로 🔒

`rank_action_candidates` 의 첫 정렬 키가 `_tier_of`(A>C>B 하드 순서)라 **점수를 무조건 이긴다.**

| 반복 | 선택된 것 | 밀린 것 | 결과 |
|---|---|---|---|
| 1 | `reranker.enabled:disable` 1.0 | `abstention_relaxed` 2.0 | 롤백 |
| 3 | `reranker.candidate_count:increase` 1.0 | `abstention_relaxed` 3.0 | 마진 턱걸이 |
| 4 | `reranker.enabled:disable` 1.0 | `restate_question` **4.0** | 롤백 |

5회 중 3회가 이 패턴이고 전부 롤백/턱걸이다. **이 근거는 유효하다** — 티어가 정렬 첫 키인
성질은 지금도 그대로다.

#### 근거 하나는 철회한다 — "같은 처방 두 번 시도"는 이미 고쳐졌다

처음엔 `reranker.enabled:disable` 이 **두 번 시도돼 두 번 다 실패**한 것도 근거로 적었다
(`ActionAttemptKey` 가 baseline 지문을 포함해서 무관한 축 변경이 차단을 풀었다). 그런데
시각을 맞춰보니 **근본 수정이 그 실행 직후에 들어갔다.**

```
corpus_20260804_103059   8/4 10:30   ← 사고가 찍힌 실행(이 문서의 근거 로그)
41feb76 cooldown 추가     8/4 12:08   ← 근본 수정
```

`_rollback_action_cooldown_exclusions` 의 docstring 이 같은 진단을 적고 있다.

> *"`blocked_action_attempts` 는 의도적으로 baseline 이 바뀌면 풀리는 정확한 전이 차단이다.
> 그런데 baseline 지문이 config 전체를 보므로 무관한 축이 조금만 움직여도 풀려, 방금 점수를
> 떨어뜨린 처방이 곧바로 다시 선택된다. 그 의미는 그대로 두고 history 기반 횟수 제한을
> 따로 얹는다."*

즉 **밴딧이 없어도 그 사고는 다시 안 난다.** 이 근거는 빼고, 티어 규칙 하나로 7번을 지탱한다.

(관련: PR #123 이 planner 결과에 안전망을 한 겹 더 두고 있다. 근본 원인이 아니라
defense-in-depth 로 보인다 — 리뷰에서 실제 누수 관측 여부를 질의했다.)

#### 논문 쪽

AutoRAG-HP(EMNLP Findings 2024)의 2단 계층 MAB 는 상위가 **어느 모듈**을, 하위가 **어느 값**을
고른다. 우리 구조와 대응되는데 차이는 상위 결정이 **학습되느냐 고정이냐**다.

밴딧의 값어치는 이제 "같은 처방 반복 방지"가 아니라 **"지지 점수를 실제로 반영한다"** 쪽에
있다. 지금은 티어가 상수라 지지 1.0 이 지지 4.0 을 영구히 이기는데, 밴딧이면 실패가 쌓인
축의 기대값이 내려가 그 순서가 학습으로 뒤집힌다.

### 8. 라벨 정확도 측정 🔒

Doctor-RAG(arXiv 2604.00865)가 라벨별 정확도 표를 싣는다(Format 94.9% ~ Search 62.0%).
**우리는 진단 정확도를 한 번도 재본 적이 없다.**

정답지 후보: [layer6ai-labs/rag-error-classification](https://github.com/layer6ai-labs/rag-error-classification)
(MIT) — 사람이 라벨링한 RAG 에러 **377건**.

⚠️ CSV 에 검색된 청크도 원본 문서도 없다(질문/정답/RAG답변/라벨뿐). 우리 `diagnose()` 는
순위·recall@k·오라클 트랙·RAGAS 가 필요하므로 **"CSV 넣고 채점"은 불가.** 원본
DragonBall 을 받아 우리 파이프라인으로 다시 돌려야 한다. 배관은 남지만 제일 비싼
것(사람 라벨링)은 끝나 있다.

---

## 채택하지 않은 것

**Doctor-RAG 의 `k†`(최초 실패 지점 국소화) — 도입하지 않는다.**

한 probe 에 검색 라벨과 생성 라벨이 같이 붙으면 하류를 감쇠시키자는 제안이었으나, **우리
생성 라벨은 오라클 트랙을 본다**(`_faith_oracle`·`_correctness_counts_oracle` →
`record.oracle_ragas`). 골드를 통째로 쥐여줬는데도 틀렸다는 뜻이라 검색 실패와 **독립적으로**
성립한다. Doctor-RAG 는 같은 판단을 모델에게 물어봐서 하고 진단 정확도가 62~81% 다.

**우리 반사실 실행이 원리적으로 더 단단하므로 유지한다.**

단, 오라클 판정의 유효성 = 골드의 정확성이다. 골드가 엉뚱하면 오라클도 엉뚱한 청크를 받아
실패하고 생성을 무고한다.

#### PR #119 와 헷갈리지 말 것 — 다른 것이다

PR #119(`feat/eval-failure-localization`)가 "Doctor-RAG 참고"라는 제목으로 열려 있으나
**위에서 철회한 것과 다른 작업이다.** 두 가지를 한다.

1. **라벨 → `failure_stage`/`repair_scope` metadata 추가.** 판정 로직을 건드리지 않는
   순수 추가 정보다.
2. **`bad_gold_answer` 면 나머지 진단을 멈춤.** 이것은 "상류 라벨이 하류 라벨을 감쇠"가
   아니라 **"정답셋이 망가졌으면 파이프라인을 진단하지 않는다"** 이고, `bad_gold_chunk` 가
   이미 갖고 있던 배타 규칙(그 docstring — *"경쟁 슬롯의 거짓 원인을 막고 이 하나만 남겨
   검수로 보낸다"*)을 확장한 것이다. 기존 설계와 일관된다.

**조율이 필요한 지점은 따로 있다** — `repair_scope` 가 `rules.py` 의 처방과 같은 정보를
두 번 적는다.

| 라벨 | #119 `repair_scope` | `rules.py` 처방 |
|---|---|---|
| `retrieval_low_rank` | `increase_candidate_pool_or_rerank` | `enable_reranker` |
| `generation_misinterpretation` | `restate_question_then_regenerate` | `restate_question` |
| `generation_partial_answer` | `regenerate_with_completeness_prompt` | `completeness_prompt` |

지금은 Optimize 가 `rules.py` 만 읽고 새 필드에는 소비처가 없다. 둘이 갈라지면 어느 쪽이
정본인지 알 수 없게 되므로, 소비처가 생기기 전에 정본을 정해야 한다.

---

## 부수 확인 — KorQuAD 로는 라벨 검증이 불가능하다

RAGEC 377건과 질문 종류를 대조했다.

| 질문 종류 | RAGEC | 우리 실행 |
|---|---|---|
| 여러 문서 종합 | 120 | 0 |
| 시간 순서 | 59 | 0 |
| 멀티홉 추론 | 50 | 0 |
| 여러 문서 비교 | 47 | 0 |
| 단순 사실 | 33 (9%) | **30 (100%)** |

우리 30문항은 전부 `taxonomy·single`, `qtype=None` 이다. 31개 라벨 중 **19개가 미발화**했는데,
멀티홉 라벨은 멀티홉 질문이 0이면 **원리적으로** 못 나온다.

→ 라벨 유효성 검증이 목표라면 **코퍼스 교체가 선택이 아니라 전제다.** 다만 파이프라인이
도는지 확인하는 용도로는 KorQuAD 로 충분하다. 교체 대상은 "라벨 검증용 코퍼스"이지
"실행 확인용 코퍼스"가 아니다.

---

## 출처

| 약칭 | 논문 |
|---|---|
| Doctor-RAG | [Failure-Aware Repair Framework for Agentic RAG](https://arxiv.org/html/2604.00865) |
| RAGChecker | [Fine-grained Framework for Diagnosing RAG](https://proceedings.neurips.cc/paper_files/paper/2024/file/27245589131d17368cccdfa990cbf16e-Paper-Datasets_and_Benchmarks_Track.pdf) (NeurIPS 2024) |
| RAISE | [RAG Design as an Architecture Search Problem](https://arxiv.org/html/2605.30029) |
| AutoRAG-HP | [Automatic Online Hyper-Parameter Tuning for RAG](https://arxiv.org/abs/2406.19251) (EMNLP Findings 2024) |
| 심판 감사 | [When the Judge Changes, So Does the Measurement](https://arxiv.org/html/2607.08535v1) |
| Noisy but Valid | [Robust Statistical Evaluation of LLMs with Imperfect Judges](https://arxiv.org/pdf/2601.20913) |
| RAGEC | [Classifying and Addressing the Diversity of Errors in RAG](https://aclanthology.org/2026.eacl-long.147/) (EACL 2026) |
| 실패 택소노미 | [A Systematic Taxonomy of Failure Modes in RAG](https://aclanthology.org/2026.trustnlp-main.27.pdf) (TrustNLP 2026) |
