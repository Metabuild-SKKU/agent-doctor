# Eval Agent

Index Agent가 만든 `state.chunks`를 대상으로 RAG 파이프라인 품질을 진단하는 에이전트.
"점수를 내는 것"이 아니라 **RAG가 왜 실패하는지 원인(검색 실패 vs 생성 실패)을 구분**하는 것이 목표.

> STEP1~5 전 구간이 실제로 동작합니다(Probe 자동생성, 검색+생성, 규칙 지표, RAGAS(옵션),
> 원인 판정, 리포트). 남은 확장 지점은 맨 아래 `[구현 포인트]` 참고 — 대부분 "더 정교하게"의
> 문제고 "동작 안 함"은 아니다. STEP2 검색·생성은 공용 모듈 `agents/rag/`(retriever/generator)를
> 사용한다(과거의 임시 파일 `retrieval_temp.py`는 삭제됨).

---

## 역할

```
[Index Agent] → state.chunks
                     ↓
                [Eval Agent]
   Probe 생성 → 검색·생성 → 지표 → 원인 판정 → 리포트
                     ↓
      state.probes, state.report → route_after_eval()
                     ↓
        pass_threshold ? [Serve] : [Optimize]
```

---

## 처리 흐름 (설계 STEP 1~5)

```
STEP1  Probe 생성        probe_gen.py       user_log(최우선) / 지식그래프 기반 RAGAS 4분면
                                            + DataMorgana-lite + 무응답(Held-out·False Premise)
                                            / 그래프가 비면 단일홉 폴백. eval_probes.json 캐시(probe_store.py)
STEP2  검색 + 생성        agents/rag/        retriever.py(벡터 검색, 키워드 폴백) + generator.py(LLM 생성, 추출식 폴백)
STEP3-1 규칙 지표         metrics_basic.py   Recall@k(span 우선), answer_match(짧은 정답 containment·긴 정답 창 F1), Oracle F1, EM
                          metrics_search.py  tier2 측정 — gold 순위 재검색 / BM25 / 코퍼스 조회
STEP3-2 LLM(RAGAS) 진단   metrics_ragas.py   Faithfulness/ContextPrecision/Recall/Relevancy/AnswerCorrectness (DEEP 이상)
STEP4  성공판정 + 원인 판정 diagnose.py        _is_success 게이트 → 실패일 때만 라벨 판정 → Finding(label=처방 라벨)
STEP5  리포트            report.py          overall_score / pass_threshold 산출
```

측정 파일은 **"tier = 파일"** 로 정렬돼 있다. 어떤 자원이 드는 측정인지 파일만 보고 알 수 있다:

| 파일 | tier | 자원 |
|---|---|---|
| `metrics_common.py` | — (인프라) | 진단 모드·자원 컨텍스트(`_ctx`)·memoize(`_cache`) |
| `metrics_basic.py` | tier1 | 없음 — 이미 가진 답변·정답·청크 좌표만 계산(청크 경계 측정 포함) |
| `metrics_search.py` | tier2 | 추가 검색 쿼리(top-N 재검색 / BM25 / 코퍼스 조회) |
| `metrics_ragas.py` | tier3 | LLM(RAGAS) 호출 |

측정 모듈은 '값'만 내고, **임계값 판정과 라벨 부여는 전부 `diagnose.py` 소관**이다.

### 완전 진단 결과 캐시

Eval은 인덱스 fingerprint, 검색 설정, Probe 입력, 진단 모드, 모델/provider 환경을
묶은 키로 `probes`, `report`, `diagnosis_cache`를 함께 저장한다. 롤백으로 동일한
키가 다시 활성화되면 검색·답변 생성·RAGAS·원인 진단을 실행하지 않고 결과를 바로
복원한다.

캐시는 현재 후보와 pending 처방의 baseline을 최대 두 개 보관한다. 후보가 세 개
이상 연속 평가돼도 baseline은 고정하고 중간 후보만 교체하므로, 실패 판정 후 바로
이전의 검증된 진단 결과로 돌아갈 수 있다. 모델, API key 사용 가능 여부, 외부 Probe
파일의 내용 hash·크기·수정 시각 등이 바뀌면 키가 달라져 다시 평가한다.

### 성공/실패 판정 — 실패일 때만 원인을 찾는다

`diagnose()`는 측정을 마친 뒤 `_is_success()`로 probe 단위 성공/실패를 먼저 판정한다.

| 반환 | 뜻 | 처리 |
|---|---|---|
| `True` | 성공 (정답 일치 / 무응답 기대인데 올바르게 기권) | `[]` 반환 — 라벨 판정 안 함 |
| `None` | 판정 불가 (대조할 정답셋 없음) | `[]` 반환 (실패라 단정할 근거가 없다) |
| `False` | 실패 | A/B/C 원인 판정으로 진행 |

판정 기준은 **recall + 정답 혼합 점수**다. recall 은 정답셋이 있는 경로에서만
본다 — 무응답 기대 probe 는 gold 가 없어 `recall_at_k = -1` 이라, 앞에서 보면 올바른 기권까지 실패가 된다.

#### 검색 판정 — 골드 커버리지를 부분점수로 (2026-08-07)

`span_recall_at_k` 가 골드 구간을 **빈틈없이 다 덮어야 1, 아니면 0** 인 이진 판정이었다.
그 판정이 "정답을 맞혔는데 검색 실패로 집계되는" 오탐을 대량으로 만들었다 —
실측(`output/logs/corpus_20260804_103059.txt`) 30문항 중 **14건(47%)** 이 `recall=0` 이었다.

이제 **덮은 문자 수 / 골드 전체 문자 수**를 쓴다. LLM 은 쓰지 않는다 — 좌표 산수라
결정론적이고 비용이 0이다.

> **지표가 1점을 못 주는 게 아니다**(리뷰 지적). 골드와 가장 많이 겹치는 청크를 주면
> 이진 판정으로도 `1.000` 이 나온다 — 위 14건이 0점인 건 검색기가 그 청크를 top-k 에 못
> 넣었기 때문이다. 부분점수의 값어치는 "불가능을 가능하게" 가 아니라 **부분적으로 잘한
> 검색에 기울기를 주는 것**이다. 이진일 땐 골드 청크가 6위든 100위든 똑같이 0 이라
> optimize 가 어느 방향으로 움직여야 나아지는지 알 수 없다.

**span 이 여럿이면 길이로 가중해 합친다(micro average).** span 별 비율을 단순 평균하면
점수가 **청킹 경계에 의존한다** — 같은 골드 500자 중 앞 100자만 검색했을 때, 청커가
100/400 으로 잘랐으면 0.5, 250/250 으로 잘랐으면 0.2 가 된다. 검색 성과는 같은데 값이
갈리는 것이다. `gold_spans` 는 원자적 근거 단위가 아니라 `positive_chunk_ids` 를 좌표로
환산한 값이라 그 경계 자체가 청킹 산물이고, optimize 가 처방으로 바꾸는 축이 바로 청크
전략이다. 가중 평균은 자르는 방식과 무관하게 같은 값을 낸다.

같은 코퍼스(문서 20개)에서 "검색기가 골드와 가장 많이 겹치는 청크 하나를 가져왔다"고
가정하고 채점 방식만 대조한 결과다.

| 골든 QA | 골드 폭(중앙값) | 옛 판정 | 새 판정 | 0점→부분점수 |
|---|---|---|---|---|
| `qa_pairs.jsonl`(원본) | 499자 | **0.105** | **0.725** | 38건 중 **34건(89%)** |
| `qa_pairs_clean.jsonl`(정제본) | 5자 | 1.000 | 1.000 | 0건 |

**정제본에서는 아무것도 바뀌지 않는다.** 골드가 이미 좁아 청크 하나에 들어가기 때문이다.
즉 이 변경은 정제본을 대체하는 게 아니라, 정제가 **버려야 했던 QA**(정답이 여러 곳이라
모호해 제외된 1,169건)를 되살리는 쪽으로 값을 한다. 둘은 상보적이다.

**라벨 배정은 바뀌지 않는다.** 전부 덮으면 정확히 `1.0`, 하나도 못 덮으면 `0.0` 이라
`_recall_ok`(`>= 1`)와 A슬롯 진입(`0 <= recall < 1`)의 참거짓이 뒤집히지 않는다.

> ⚠️ **재베이스라인 필요.** `scoring.reliability_score` 가 recall 을 그대로 곱하므로 부분
> 커버리지 probe 의 신뢰도가 0 에서 부분점수로 오른다. `composite`·`reliability`·
> `mean_recall_at_k` 를 **이 변경 이전 실행과 직접 비교하면 개선으로 오독된다**
> (optimize history 의 before/after 포함). optimize gate 의 `RECALL_FLOOR` 를 통과하는
> 실행도 늘어난다.

#### 정답 판정 — lexical 단독 문턱을 버리고 RAGAS 와 섞는다 (2026-07-30)

lexical char-F1(`answer_match`)은 KorQuAD 추출형 짧은 정답용 지표다. 생성 답변에 쓰면 두 방향으로
어긋난다: gold 가 3~5문장이면 근거·소제목을 갖춘 *맞은* 답변이 precision 감점으로 0.3~0.4 로 깎이고,
gold 가 묻지도 않은 수식어를 하나 더 갖고 있으면 맞은 단답도 0.49 가 된다. 그 답변들이 `f1 < 0.5`
게이트에서 실패로 잡히고, 검색·오라클은 통과했으니 C그룹(`context_noise_interference`,
`chunking_underchunking`)으로 오진돼 optimize 가 엉뚱한 처방(top_k↓·노이즈 필터)을 받았다.
반대로 의미 지표(RAGAS)만 단독 문턱으로 쓰면 판정기 편차에 그대로 노출된다. 그래서 **두 축을 섞은
점수 하나**로 판정한다(`diagnose._answer_ok` / `types.blend_answer_score`).

```
semantic     = max(answer_correctness, gold 커버리지 TP/(TP+FN))   # 커버리지는 faithfulness ≥ 0.7 일 때만
answer_score = 0.4·lexical + 0.6·semantic                          # ANSWER_SEMANTIC_WEIGHT = 0.6
통과          = semantic ≥ 0.3 (ANSWER_SEMANTIC_FLOOR) and answer_score ≥ 0.5 (ANSWER_PASS_THRESHOLD)
```

| 상황 | 판정 |
|---|---|
| 맞은 단답, gold 에 여분 수식어 (`lexical 0.49` · `semantic 0.73`) | `0.63` **통과** (기존엔 실패) |
| 맞은 서술형 장문 (`lexical 0.34` · 커버리지 `1.0`) | `0.74` **통과** |
| 근접 오답·부정문 (`lexical 0.9` · `semantic 0.2`) | 바닥선에서 **실패** |
| 길게 쓴 무관한 답 (`lexical 0.39` · `semantic` 낮음) | **실패** |
| RAGAS 미측정(DEEP 미만·판정기 degrade) | lexical 단독 `F1_PASS_THRESHOLD(0.5)` — 기존 동작 |

- 의미축 바닥선(`ANSWER_SEMANTIC_FLOOR`)이 기존 `answer_correctness` 강등 규칙의 역할을 잇는다
  (부정문·`3월`↔`3일` 차단). 레거시 `ANSWER_CORRECTNESS_MIN` 은 게이트에서 더 쓰지 않는다.
- 의미축에 gold 커버리지를 함께 넣는 이유: `answer_correctness = TP/(TP+0.5(FP+FN))` 의 분모에
  군더더기(FP)가 들어가 verbose 정답을 lexical 과 **같은 방향**으로 깎는다. 커버리지는 FP 를 빼고
  '정답 요소를 다 담았나'만 묻는 recall 축이고, 누락·모순은 FN 으로 분류되므로 근접 오답은 여전히 떨어진다.
- lexical 쪽도 tier1 에서 한 겹 보정한다: `metrics_basic.best_window_char_f1` — 정답 ≥30자 + 답변이
  1.3배 이상 길면 precision 분모를 답변 전체가 아니라 **정답 길이만큼의 창**으로 잡는다(실측: 맞은
  서술형 답변 0.34→0.63, 길게 쓴 무관한 답변 0.29→0.39 로 여전히 미달 — 창 크기가 고정돼 길게 써서
  점수를 버는 경로는 없다).
- `oracle_f1` 도 같은 이유로 깎이므로 `_oracle_ok` 이 오라클 트랙에 동일한 혼합 점수를 쓴다.
- 게이트는 record 의 RAGAS dict 만 읽는다(LLM 재호출 없음) — `report._oracle_accuracy` 가 성공
  probe 에도 `_oracle_ok` 을 부르기 때문에 지켜야 하는 성질이다. 같은 이유로 미측정 성공분을
  통과 처리하는 추론도 `_oracle_ok` 이 아니라 `_oracle_accuracy` 쪽에 있다(`_is_success` 는
  `_faith`·`_abstention_judged` 를 타서 LLM 을 부를 수 있다).
- probe 로그에 `answer=0.63(의미 0.73, 커버리지 0.50)` 이 함께 찍혀, f1 이 낮은데 통과한(또는 f1 이
  높은데 실패한) 근거를 볼 수 있다. Finding 의 `reason` 도 `answer=…(f1 …·의미 …)` 로 기록된다.

**의미축에 넣지 않는 것** (게이트가 승격까지 하므로, 신뢰할 수 없는 성분은 통과를 만든다):

| 제외 대상 | 이유 |
|---|---|
| `answer_correctness_degraded` (유사도 단독 계산) | 한국어는 같은 주제면 사실이 틀려도 코사인이 0.8 대 — 승격에 쓰면 오답이 통과한다. 강등 전용이던 시절엔 안전했던 폴백이다 |
| 커버리지 < `GOLD_COVERAGE_MIN(0.8)` | 부분 답변이 커버리지를 '부분 점수'로 들고 통과하면 `generation_partial_answer` 진단이 사라진다. 누락 감점은 FN 을 분모에 넣는 `answer_correctness` 몫 |
| 근거 없는(faithfulness < 0.7) 커버리지 | 파라미터 기억으로 맞힌 답까지 올리면 `generation_parametric_overreliance` 가 잡을 케이스를 가린다 |

**남는 위험 (튜닝 대상)**

- 가중치 0.4/0.6 · 문턱 0.5 의 기하학: 의미축이 **0.834 이상이면 lexical 0 이어도 통과**하고,
  의미축이 바닥선(0.3)이면 lexical 0.8 을 넘겨야 통과한다. 즉 판정 권한의 60%가 LLM 판정기에 있다.
  판정기를 덜 믿고 싶으면 `ANSWER_SEMANTIC_WEIGHT` 를 낮춘다.
- gold 문장 수가 1~2개인 probe 는 커버리지가 사실상 0/1 이라, 판정기의 TP 오분류 하나가 곧 통과다
  (문턱 0.8 은 그런 probe 에서 사실상 '전부 맞음'만 통과시키지만, 그 판단이 LLM 한 번에 달려 있다).
- gold 답변이 질문이 묻지 않은 수식어까지 담고 있으면 커버리지가 구조적으로 낮게 나온다
  (실측 사례: `f1=0.49` probe 의 커버리지 0.5). 이건 지표가 아니라 **probe gold 품질** 문제라
  STEP1 쪽에서 줄여야 한다 — 커버리지에 하한 veto 를 걸면 그런 정답이 다시 실패한다.
- `_oracle_ok` 도 완화됐으므로 `_generation_failed`(B) 전제가 덜 성립하고 `_context_failed`(C)
  전제는 더 자주 성립한다 — 같은 실패가 B에서 C로 **재분류**될 수 있다(원인 자체는 C 가 더 맞지만,
  라벨 분포·처방 순서가 이전 실행과 달라진다).
- `bad_gold_answer`(사람 검수 큐로 가는 D 라벨)는 `_oracle_ok` 실패가 전제라 **발동이 줄기만 한다**
  — 새 게이트의 통과집합이 옛 게이트의 상위집합이기 때문(전수 격자로 고정:
  `tests/test_answer_match.py::TestGateMonotonicity`). 줄어드는 대부분은 '오라클 답이 gold 와
  의미상 같은데 표현만 달라서 정답셋 오류로 몰리던' 오탐이지만, **gold 가 질문이 묻지 않은 요소까지
  담아 lexical 이 낮던 유형은 이제 통과**해 라벨로 남지 않는다(성공 probe 는 findings 가 비어야
  하므로 라벨을 붙일 수 없다). 그 신호는 리포트 카운터 `semantic_rescued` 가 대신 들고 있다 —
  'lexical 미달인데 의미축으로 통과' 건수 = 정답셋 품질 검수 후보이자 채점 변경의 영향 크기.
  (C 슬롯의 `bad_gold_answer` 는 슬롯 전제가 `_oracle_ok` 를 요구하는데 함수는 그 반대를 요구해
  이번 변경과 무관하게 원래부터 도달 불가다 — 실질 발동 경로는 B 슬롯 하나뿐.)
- 점수 스케일이 전반적으로 올라간다. `composite`/`reliability`/`oracle_accuracy` 를 이 변경 **이전
  실행과 직접 비교하면 개선으로 오독**된다(optimize history 의 before/after 포함) — 재베이스라인 필요.

성공/실패는 별도 필드를 두지 않는다. 이 게이트 덕에 **`findings` 가 비었으면 성공**이 규약이 아니라
구조적 보장이 된다(`_is_success()`/`diagnose()` 의 판정 자체가 그 보장이다).

**주의(종합점수 통일, 2026-07-27)**: 위 보장은 여전히 참이지만, `scoring.reliability_score`가
집계하는 값은 더 이상 "findings 유무의 이진 카운트(통과 probe 수 / 전체)"가 아니다. gold probe는
`recall@k × answer_score`(게이트와 같은 혼합 점수)의 **연속값**으로 신뢰도를 매기고, 이는 `findings`와
디커플링됐다 — 예를 들어 char-F1이 낮아 finding이 붙은 probe도 의미축이 높으면 신뢰도 점수는
부분점수를 받는다. 게이트와 같은 값을 보므로 '통과했는데 신뢰도만 낮게 남아 optimize 탐색이 반대
방향을 가리키는' 어긋남도 없다.
무응답 기대 probe(`answer_exists=False`)만 여전히 findings 유무의 이진값(1/0)을 쓴다.
즉 **화면에 보이는 "신뢰도" 숫자는 "findings 없는 probe의 비율"이 아니라 "probe별 soft 신뢰도의
평균"이다.** 자세한 배경은 `agents/optimize/history.py`의 `judge`/`_read_score` 주석과 PR #47 참고.

실패로 판정되면 전제별로 해당 그룹만 검사한다 — A(검색, `0 <= recall < 1` + `_retrieval_fixable`)
/ B(생성, `_generation_failed`) / C(컨텍스트, `_context_failed`). 슬롯마다 `_pick()`으로
확정(confirmed) 우선 하나씩 채택하고, D(데이터, `corpus_gap` 계열)는 additive로 더 붙는다.

검색으로 고칠 수 없는 실패는 A 슬롯을 아예 닫는다(`_retrieval_fixable`) — gold가 전부 코퍼스
밖이거나 `answer_exists=False` probe인 경우. 안 닫으면 구체 라벨이 self-scope로 다 빠져도 롤업
`retrieval_failure`가 남아 "검색을 고쳐라"가 처방된다.

`corpus_gap` / `corpus_gap_partial_hop`은 A 슬롯과 별개로 additive로 붙는다 — 검색을 고치는 것과
자료를 채우는 것은 처방이 다르다. 누락된 gold 목록은 `metadata["missing_gold_ids"]`로 넘긴다
(optimize는 코퍼스 멤버십을 스스로 구할 수 없다 — `_ctx.corpus_ids`는 Eval 자원).
무응답 기대 probe(`answer_exists=False`)에는 D도 붙지 않는다(채울 자료가 없으므로) —
남는 건 B의 `generation_abstention_failure` 하나다.

기권(답을 안 냄)은 C 슬롯 전제(`_context_failed`)에서 배제된다 — C 라벨들은
전부 "틀린 답을 냈다"를 전제로 인과를 세우는데(노이즈에 이끌림·리랭커가 무관한 청크를 올림·
청크가 커서 노이즈) 기권에는 그 서사가 성립하지 않는다. 유효 근거를 두고 기권한 경우
(recall=1) `_generation_failed`가 B 슬롯을 열어 `generation_wrongful_abstention`이
그 자리를 가져간다 — `generation_abstention_failure`의 정반대 짝이다.

B/C 배타는 두 슬롯이 `_wrongful_abstention_premise` 하나를 공유해서 보장된다(함수 호출
순서가 아니라). 그 전제는 오라클 통과를 요구하지 않는다 — 오라클 답변도 같은 generator 가
만들므로 과다 기권이 심할수록 오라클도 함께 기권해, `_oracle_ok` 를 걸면 "가끔 기권"만
잡히고 "항상 기권"이라는 더 심한 케이스가 빠져나간다.

---

## 입출력 (계약)

```
읽기: state.chunks, state.user_questions, state.index_config, state.iteration,
      state.active_index_key, state.optimization_history
쓰기: state.probes, state.report, state.diagnosis_cache,
      state.eval_cache, state.active_eval_key, state.eval_cache_hit,
      state.status, state.error
```

`DiagnosticReport`(→ `state.report`)의 핵심 필드:

```python
overall_score : float | None   # RAGAS 가중평균(있으면) / 규칙지표 폴백 / 신호없으면 None
pass_threshold: bool           # overall_score >= 0.8 (설계 §7, types.PASS_SCORE_THRESHOLD)
ragas_scores  : dict           # RAGAS 평균 + 규칙지표 평균 + 결과 분포(diagnosed/ok)
oracle_accuracy: float | None  # Oracle 트랙 통과율
findings      : list[Finding]  # 원인 라벨(확정 우선 정렬). 각 Finding 은 label/confirmed 보유
findings_summary: dict         # {mode, total, confirmed, preliminary, confirmed_labels, preliminary_labels}
```

각 `Finding` 은 `label`(진단명)과 **`confirmed`**(현재 모드에서 확정됐는지)를 가진다.
`confirmed=False`(예비)는 *더 깊은 모드에서 확정 가능한 의심 원인*이며, `overall_score`/`pass_threshold` 를
바꾸지 않는다(지표 기반). Optimize 는 `findings_summary.confirmed_labels` 로 확정 원인부터 처방한다.

---

## 환경 변수

| 변수 | 기본 | 설명 |
|------|------|------|
| `EVAL_MODE` | `fast` | **진단 깊이(비용 tier)**: `fast`/`standard`/`deep` 또는 `1`~`3`. `full`/`4` 는 `deep` 으로 접힌다. 아래 표 참고 |
| `EVAL_ENABLE_LLM` | off | `1/true` 면 RAGAS(LLM-as-Judge) 진단 허용 (**+ `EVAL_MODE≥deep` 이어야 실제 실행**) |
| `EVAL_LLM_PROVIDER` | `openai` | LLM 호출 provider 선택: `openai` / `gemini` / `github` / `openrouter` (아래 참고) |
| `OPENAI_API_KEY` | — | provider=openai 일 때 필요 |
| `GEMINI_API_KEY` | — | provider=gemini 일 때 필요(Google AI Studio 무료 티어) |
| `GITHUB_TOKEN` | — | provider=github 일 때 필요(`models:read` 권한 포함된 PAT) |
| `OPENROUTER_API_KEY` | — | provider=openrouter 일 때 필요(유료) |
| `EVAL_JUDGE_MODEL` / `..._GEMINI` / `..._GITHUB` / `..._OPENROUTER` | `gpt-4o` / `gemini-flash-latest` / `openai/gpt-4o` / `openai/gpt-4o` | Probe 질문 생성 + RAGAS 평가(심판) 모델(설계 원칙: 응답≠평가). 답변 생성 모델은 `RAG_*`(→ `agents/rag/generator.py`)가 담당 |
| `EVAL_EMBED_MODEL` / `EVAL_EMBED_MODEL_GEMINI` | `text-embedding-3-small` / `gemini-embedding-001` | Response Relevancy 코사인용 임베딩. github·openrouter 는 임베딩 엔드포인트가 없어 OpenAI 키로 폴백하고, 그것도 없으면 **로컬 BGE-M3**(Index 와 같은 모델, 비용 0)로 계산한다 |
| `QDRANT_URL` / `QDRANT_API_KEY` | `:memory:` | 검색 인덱스 대상 |

> 기본값만으로도(위 키 전부 미설정) **외부 API 없이** 규칙 지표 기반 진단이 동작합니다(폴백 설계).

### LLM Provider — `agents/eval/llm_provider.py`

OpenAI 유료 토큰이 없어도 무료 대체 provider로 STEP1(질문 생성)·STEP3-2(RAGAS
심판·임베딩)를 실제 LLM으로 돌릴 수 있게 하는 브릿지 계층. `chat_json`/`embed_texts`
두 함수로 provider 차이를 감추고, `probe_gen.py`/`metrics_ragas.py`가 전부
이 계층만 호출한다(직접 `from openai import OpenAI` 하지 않음). STEP2 답변 생성은
`agents/rag/generator.py`가 담당하며, 그쪽은 자체 provider 선택 로직(`RAG_LLM_PROVIDER`)을 쓴다.

- **openai**(기본): 정식 OpenAI API.
- **gemini**: Google AI Studio 무료 티어. `google-genai` 패키지 필요(`pip install google-genai`).
  **주의**: 무료 티어가 분당 요청 수 제한(계정별로 다름, 낮으면 5회/분 수준)이 있어 청크 수가
  많으면(질문 생성 N회 + 답변 생성 N회) 429(RESOURCE_EXHAUSTED)가 잦다 — 실패해도 자동으로
  휴리스틱/추출식 폴백으로 넘어가 파이프라인은 안 죽는다.
- **github**: [GitHub Models](https://github.com/marketplace/models) — OpenAI 호환 API를
  `base_url=https://models.github.ai/inference` 로 그대로 재사용. GitHub 개인 액세스 토큰에
  **`models:read` 권한이 반드시 있어야 함**(없으면 401/403). 임베딩 엔드포인트는 제공하지
  않아 `embed_texts`는 provider 무관하게 OpenAI 키로만 동작(없으면 해당 RAGAS 지표만 스킵).
- **openrouter**: [OpenRouter](https://openrouter.ai) — 키 하나로 여러 publisher 모델
  (`openai/…`, `anthropic/…`, `google/…`)을 호출. OpenAI 호환 API를
  `base_url=https://openrouter.ai/api/v1` 로 재사용한다. 다른 셋과 달리 **무료 브릿지가 아니라
  유료**이며, 상시 사용을 전제한 provider 다.
  - 모델명은 반드시 `publisher/model` 형식. 형식이 틀리면 404 → 폴백으로 조용히 강등된다.
  - **심판 모델은 `response_format=json_object` 지원 모델로 고를 것.** 미지원이면 `chat_json`
    파싱이 실패해 `{}` 로 폴백하고, 해당 점수가 결측 처리된다.
  - 임베딩 엔드포인트가 없어(카탈로그 337개 중 임베딩 모델 0개) `embed_texts` 는 github 와
    마찬가지로 OpenAI 키로 폴백하고, 키가 없으면 **로컬 BGE-M3** 로 계산한다(비용 0, 외부 호출 없음).
    결측이 되면 지표 하나가 비는 데서 끝나지 않는다 — `diagnose` 의 `bad_gold_answer` /
    `bad_gold_answer_oracle` 이 `rel` 을 AND 조건으로 요구해 두 라벨이 영구히 침묵하고,
    그 라벨에 걸린 probe 자동 재생성 루프까지 멈춘다.
    임베딩 모델이 바뀌면 코사인 분포도 달라지므로 **API 임베딩 실행과 값을 직접 비교하지 말 것**
    (실행당 1회 안내가 나온다). 모델 로드에 실패해 해시 폴백 상태면 채점에 쓰지 않고 결측으로 둔다.
  - 비용은 단가표 추정이 아니라 **응답이 알려준 실제 과금액**으로 집계된다
    (`core/llm_clients.py` 가 요청에 `usage.include` 를 붙인다).

모델명·무료 티어 한도는 시점에 따라 바뀔 수 있다 — 401/403/404 가 나면 해당 콘솔에서 현재
사용 가능한 모델명을 다시 확인할 것.

---

## 진단 모드 (비용 tier)

라벨(진단명)은 판별에 필요한 **가장 비싼 자원**을 tier 로 갖는다. 사용자가 고른 모드(`EVAL_MODE`)가
그 tier 상한을 정해, 감당 못 하는 라벨은 **예비(`Finding.confirmed=False`)** 로 내보내고 상위 모드에서 확정한다.

**확정(`confirmed=True`)은 그 라벨의 '확정 신호'가 실제로 발동해야 성립한다** — 자원 미실행/미측정이면 예비다
(단순히 `mode>=tier` 라서 확정이 아님). 신호는 자기 tier의 자원을 self-gate 하고(모드 부족 시 `None`),
라벨 함수는 `확정 신호 → (미실행이고 싼 신호 있으면) 예비` 순으로 판정한다. → 거짓확정이 구조적으로 불가.

| 모드 | 값 | 추가 자원 | 확정 가능한 라벨 |
|------|----|----------|-----------------|
| `fast` | 1 | 규칙·기존 지표만(추가 쿼리 없음) | `retrieval_incomplete_enumeration` (gold수 vs top-k 순수 규칙) |
| `standard` | 2 | + 추가 검색 쿼리(top-N 재검색·BM25·코퍼스) | 검색 원인(`low_rank`/`lexical`/`semantic`/`missing_gold`), `corpus_gap(_partial_hop)` |
| `deep` | 3 | + **LLM(RAGAS/AspectCritic)** | 생성 원인(`hallucination`/`partial`/`hop_binding`/`contradiction`), context 원인(`too_long_context`/`lost_in_the_middle`/`underchunking`/`noise_interference`), `bad_gold_answer` |

- **tier4(파이프라인 재실행/ablation)는 없앴다.** context 원인을 재실행으로 확정하던 경로는
  Optimize 가 config 를 바꿔 재실행·검증하는 것과 중복이라 Optimize 로 넘겼다. `Mode.FULL`
  상수도 제거해 `deep` 이 가장 깊은 모드다 — 단 `EVAL_MODE=full`/`4` 는 웹 UI depth 문자열이라
  `deep` 으로 접는다(지우면 `fast` 로 조용히 강등된다).
- **C그룹 라벨은 `deep` 에서 확정된다.** 예비로 둔 이유가 "tier4 가 확정해 줄 것"이었으나 그
  tier 가 없으니, 실측 신호(context 길이·gold 위치=tier1, faithfulness·context_precision=tier3)가
  발동하면 확정으로 낸다. `confirmed` 는 '처방이 통한다'가 아니라 '판별 신호가 측정됐다'는 뜻이라
  이 정의에 맞다. **예외는 `reranker_low_precision` 하나** — 리랭크 전/후 순위를 대조해야
  인과가 서는데 retriever 가 리랭크 전 후보를 남기지 않아 예비로 남는다.
  길이 원인과 리랭커가 함께 성립하면 `_pick` 이 확정 쪽을 채택한다(튜플 순서 비의존).
- **생성 원인은 전부 RAGAS(=deep) 의존** → `deep` 미만이면 하나의 예비 `generation_failure` 로 롤업된다
  (LLM 없이는 hallucination/bad_gold 를 싸게 구분할 수 없다는 정직한 한계).
- **STEP3-2 RAGAS 는 `deep` 이상에서만 실행**된다(`metrics_ragas` 의 DEEP 게이트). `EVAL_ENABLE_LLM` 과 AND 조건.
  실제 트랙은 전 probe, **오라클 트랙은 실패로 판정된 probe 에만** 계산한다(성공 probe 는 진단을
  건너뛰므로 오라클 값의 소비처가 없다).
- **오라클 답변 생성도 같은 게이트를 탄다** — STEP2 는 실제 트랙만 만들고, STEP3 에서 실패 판정을
  세운 뒤 실패 probe 에만 gold context 답변을 생성한다(`_is_success` 는 오라클 필드를 안 읽으므로
  순서가 성립한다). 성공 probe 는 `oracle_answer=None` 으로 남고 probe 로그의 oracle 칸은 `-` 다.
  `report._oracle_accuracy` 는 그 미측정분(`oracle_answer is None` + `findings` 없음)을 통과로
  추론해 분모를 유지하고(성공 전제가 recall=1 이라 gold 는 이미 context 안에 있었다), 실측 표본 수는
  `ragas_scores.oracle_measured` 로 남긴다. `mean_oracle_f1` 과 `oracle_accuracy` 는 **이 변경 이전
  실행과 직접 비교하면 안 된다**(재베이스라인 대상) — 전자는 분모가 실측분으로 좁아졌고, 후자는
  분자가 바뀌었다(예전엔 성공 probe 의 오라클 실패도 감점됐지만 이제 그 몫은 통과로 추론된다).
- STEP3 의 실패 판정은 `agent.run()` 이 진입부에서 `set_mode(mode)` 를 부른 뒤에 돈다. 안 부르면
  전역 tier 가 모듈 기본값(`FAST`)이라 `_grounded_ok`·`_abstained` 의 tier3 축이 통째로 빠져,
  근거성만으로 실패하는 probe 가 STEP3 에선 성공·STEP4 diagnose 에선 실패로 갈린다(그 probe 는
  오라클 답변을 못 받고, 지연 계산으로도 복구되지 않는다).
- tier2 측정(`_gold_ranks`/`_bm25_hits_gold`/`_gold_in_corpus`)은 **구현·배선 완료**돼 있다 —
  `agent.py::run()`이 `set_diag_context(retrieve_fn=..., keyword_fn=..., ragas_fn=...)` 로 자원을
  주입한다. `EVAL_MODE`가 self-gate 를 통과할 만큼(standard 이상) 높아야 실제로 호출된다 —
  "미구현"이 아니라 "모드가 낮아서 미실행"인 경우가 대부분.

### 라벨별 tier 분류 (확정에 필요한 자원)

아래는 각 라벨을 **확정**하는 데 필요한 자원 정리(설계 문서용). tier 값은 `Finding` 에 싣지 않고,
각 측정이 `metrics_*` 에서 자기 자원(tier)을 실행 모드로 self-gate 한다(mode 부족 시 `None`).

| 라벨 | 그룹 | tier | 확정 자원 |
|---|---|:---:|---|
| `retrieval_incomplete_enumeration` | A 검색 | 1 | gold수 vs top-k 순수 규칙 |
| `retrieval_rank_fusion_loss` | A 검색 | 2 | 채널별(dense/BM25) 순위 vs 융합 순위 |
| `retrieval_duplicate_crowding` | A 검색 | 2 | 상위 경쟁청크 중복 분석(재검색 0회) |
| `retrieval_rerank_candidate_miss` | A 검색 | 2 | 리랭크 직전 후보 목록(`pre_rerank_ids`) |
| `retrieval_reranker_demotion` | A 검색 | 2 | 리랭크 직전 후보 목록(`pre_rerank_ids`) |
| `retrieval_low_rank` | A 검색 | 2 | top-N 재검색 (위 넷이 아닌 잔여) |
| `retrieval_lexical_mismatch` | A 검색 | 2 | BM25 조회 |
| `retrieval_semantic_mismatch` | A 검색 | 2 | BM25 + 코퍼스 확인 |
| `retrieval_missing_gold` | A 검색 | 2 | 코퍼스 멤버십 조회 |
| `retrieval_missing_bridge_dependency` | A 검색 | — | 예비만 (decompose 재검색 회복 미측정) |
| `generation_hallucination` | B 생성 | 3 | RAGAS faithfulness |
| `generation_partial_answer` | B 생성 | 3 | RAGAS relevancy |
| `generation_hop_binding_error` | B 생성 | 3 | RAGAS faithfulness(+추론검증) |
| `generation_contradiction` / `numerical_error` / `misinterpretation` | B 생성 | 3 | 추론 실패 모드 단일분류(LLM 1회) |
| `generation_abstention_failure` | B 생성 | 2~3 | 기권했어야 하는데 지어냄(두 갈래는 `metadata.trigger`로 구분 — `no_answer_expected` / `corpus_gap`) |
| `generation_wrongful_abstention` | B 생성 | 1~3 | 근거는 검색됐는데(recall=1) 기권 — C 전제를 닫고 자리를 가져간다. 처방 `relax_abstention` |
| `generation_parametric_overreliance` | B 생성 | 3 | 정답이지만 real faithfulness 낮음 |
| `generation_failure` (롤업) | B 생성 | 3 | DEEP에서 세분화 (항상 예비) |
| `too_long_context` | C context | 3 | context 길이 + 근거 없음, gold 는 양끝 |
| `lost_in_the_middle` | C context | 3 | gold 가 긴 context 중간 + 근거 없음 |
| `context_noise_interference` | C context | 3 | real faithfulness 높음 = 노이즈에 근거 |
| `chunking_underchunking` | A 청킹 | 3 | 근거 밀도 낮음 + context_precision 낮음 |
| `reranker_low_precision` | A 청킹 | 3 | 예비만 (리랭크 전/후 대조 불가 → 인과 미측정) |
| `bad_gold_answer` | D 데이터 | 3 | RAGAS 2지표(진짜 확정은 사람) |
| `corpus_gap` | D 데이터 | 2 | 코퍼스 조회(누락 gold id 는 `metadata.missing_gold_ids`) |
| `corpus_gap_partial_hop` | D 데이터 | 2 | 코퍼스 조회(hop별) |

#### 순위 원인은 단계로 나뉜다

최종 순위는 `채널 검색(dense/BM25) → 융합 → 후보창 → 리랭크 → top_k 컷` 을 거쳐 만들어진다.
"순위가 낮다"는 증상이고, **어느 단계에서 gold 를 잃었는지가 처방을 정한다**.

| 잃은 단계 | 라벨 | 처방 |
|---|---|---|
| 융합 | `retrieval_rank_fusion_loss` | `hybrid_dense_weight` 를 우세 채널 쪽으로 |
| 경쟁 구성 | `retrieval_duplicate_crowding` | MMR (리랭커로는 안 고쳐진다 — 레버 미구현이라 `draft`) |
| 후보창 | `retrieval_rerank_candidate_miss` | `rerank_candidates` 확대 |
| 리랭크(강등) | `retrieval_reranker_demotion` | 리랭커 되돌리기 / 모델 교체 |
| 리랭크(못 올림) | `retrieval_reranker_ineffective` | 모델 교체 (미정 → `draft`) |
| (잔여) | `retrieval_low_rank` | `use_reranker` 켜기 |

순위 라벨의 관할(`_rank_scope`)은 세 갈래의 합집합이다.

| 갈래 | 뜻 | 창 적용 |
|---|---|---|
| `_rankable` | 융합 순위가 top_k 밖 | 적용 (리랭크 계열 처방의 도달 범위) |
| `_rerank_lost` | 리랭크 **전엔 top_k 안**이었는데 결과엔 없음 (강등) | 무관 (융합 순위와 독립) |
| `_rerank_not_lifted` | 후보엔 있었으나 리랭크 전에도 top_k 밖 (못 올림) | 무관 |
| 융합 손실 | 단일 채널은 top_k 안인데 융합이 밀어냄 | **미적용** |

`_rerank_lost` 가 따로 필요한 이유는 wide 재검색이 리랭크를 끄기 때문이다 — 융합이 gold 를 3위에
뒀는데 리랭커가 12위로 떨어뜨렸다면 융합 순위는 top_k 이내라 첫 갈래에 안 잡히지만, 그건
교과서적인 리랭커 강등이다.

융합 손실에 창을 적용하지 않는 이유는 **처방마다 도달 범위가 다르기** 때문이다. 창의 논거는
"리랭커가 닿는 범위"인데 `hybrid_dense_weight` 는 리랭커와 무관하게 어떤 융합 순위든 끌어올릴 수
있다. 게이트는 처방을 따라간다.

**다른 A 라벨과의 배타** — 순위 라벨이 다룰 구간(`_rank_scope`)이면 `lexical_mismatch` ·
`semantic_mismatch` 는 스스로 침묵한다. 두 라벨의 전제가 "dense 가 gold 를 놓쳤다"인데,
창 안에 있다는 건 놓친 게 아니라 순위를 낮게 준 것이라 전제가 사실과 어긋나기 때문이다.
순위 라벨끼리는 파이프라인 앞단이 뿌리다: `융합 손실`·`중복 밀림` > `후보창 밖` > `강등` > 잔여.

`retrieval_missing_gold`(코퍼스 존재만 실측하는 폴백)와 `chunking_*`(맨 뒤 배치)은 예전부터
순서로 갈리는 설계다 — 순위 라벨 분할과 무관하게 그대로 유지된다.

판정 범위는 **도달 가능 창**(`metrics_common.reachable_window`)으로 제한한다 —
`rerank_candidate_policy.max_candidates` 보다 뒤 순위의 gold 는 리랭커를 켜도 후보를 넓혀도
닿지 않으므로 순위 문제가 아니라 표현 문제(`semantic`/`lexical mismatch`)로 인계한다.
이 경계가 없으면 wide-N(=100) 안의 모든 검색 실패가 `low_rank` 하나로 흡수된다.

>  "확정(`confirmed`)"은 처방이 통한다는 뜻이 아니라 **그 원인의 판별 신호가 실제로 측정됐다**는
> 뜻이다. C그룹은 신호가 전부 실측이라 `deep` 에서 확정된다. 예비로 남는 건 판별 신호 자체가
> 없는 둘 — `reranker_low_precision`(리랭크 전/후 순위 미기록)과 `retrieval_missing_bridge_dependency`
> (decompose 재검색 회복 미측정). `bad_gold_answer` 는 tier3까지만 의심 가능하고 진짜 확정은 사람 검수 몫.
>
> Optimize(`planner._split_findings`)는 예비 Finding 을 자동 처방에서 제외한다 — C그룹을 확정으로
> 올린 이유가 이것이다(예비로 두면 `rules.py` 가 ready 여도 처방이 영원히 트리거되지 않는다).
> 남은 예비 2종은 Optimize 쪽에 "예비를 실험 후보로 받아 pending 으로 검증" 경로가 붙어야
> 인수인계가 완성된다(별도 작업).

---

## 테스트

```bash
python tests/check_eval.py       # mock chunks 5개로 STEP1~5 단독 실행 (Index 없이).
                                 # Probe(질문/정답) + STEP2 검색결과·생성답변까지 콘솔에 출력한다.
python tests/check_ragas_eval.py # 실제 OpenAI API로 RAGAS 4지표 실측 검증 (키 없으면 자동 스킵)
```

`check_eval.py`는 `EVAL_LLM_PROVIDER`(`.env`)를 그대로 읽으므로, `github`/`gemini` 로 설정해두면
실제 무료 LLM이 만든 질문·답변이 출력된다(미설정 시 키 없음 폴백 경로로 결정적 동작).

첫 실행 후 프로젝트 루트에 `eval_probes.json`(Probe 캐시, STEP1 참고)이 생긴다 —
`.gitignore`의 `*.json` 에 걸려 커밋되지 않으며, 지워도 다음 실행에서 재생성된다.

### KorQuAD 2.1 데이터셋으로 평가 (`taxonomy` 소스)

사람이 만든 골든 QA 로 진단한다. **별도 드라이버 없이 정규 파이프라인**(`run_local_pipeline.py`
/ `graph.py`)이 설정만으로 돈다. 설정은 두 군데 — corpus 는 Ingest, qa 는 Eval:

| 설정 | 값 | 무엇 |
|------|----|------|
| Ingest (corpus) | `SOURCE_TYPE=korquad`, `SOURCE_URL=data/corpus.jsonl` | 청크를 원문으로 복원해 Document 로 수집(→ Index 가 재청킹) |
| Eval (qa) | `EVAL_PROBE_SOURCE=taxonomy` | `EVAL_TAXONOMY_QA`(기본 data/qa_pairs.jsonl)의 질문·정답·`positive_chunk_ids` 를 Probe 로 로드 |

#### 1) 데이터 파일 두 개를 `data/` 에 둔다

`data/` 는 gitignore 라 직접 준비한다. 스키마·형식은 → **[data/README.md](../../data/README.md)**.

```
data/
├── corpus.jsonl     # {doc_id, chunk_id, title, text, char_start, char_end}
└── qa_pairs.jsonl   # {qa_id, question, answer_text, doc_id, positive_chunk_ids}
```

#### 2) 환경변수 설정

`.env`(권장) 또는 shell 에 넣는다. `.env.example` 참고. korquad 는 아래만으로 충분:

```dotenv
# .env
SOURCE_TYPE=korquad
SOURCE_URL=data/corpus.jsonl
# EVAL_PROBE_SOURCE 는 korquad 면 스크립트가 taxonomy 로 자동 세팅(명시해도 됨)
EVAL_TAXONOMY_QA=data/qa_pairs.jsonl
KORQUAD_MAX_DOCS=20        # 스모크: 앞 20문서만. 전체는 비우거나 0
KORQUAD_QA_LIMIT=50        # 스모크: qa 50개. 전체는 비우거나 0
EVAL_MODE=1               # 1=fast(무비용) … 3=deep(생성·RAGAS, API 비용). full/4 → deep
# EVAL_ENABLE_LLM=1       # RAGAS 켜기(EVAL_MODE>=deep 과 AND). 켜면 API 비용
```

> ⚠️ **`.env` 값이 안 먹으면 shell 환경변수를 의심하라.** `load_dotenv()` 는 `override=False`
> 라 **이미 export 된 환경변수를 .env 로 덮지 않는다**. 예: 셸에 `EVAL_MODE=4` 가 남아 있으면
> `.env` 에 `EVAL_MODE=1` 을 써도 4 로 돈다. → `echo $EVAL_MODE` 로 확인하고 `unset EVAL_MODE`,
> 또는 `EVAL_MODE=1 python …` 처럼 인라인으로 넘긴다.

#### 3) 실행

```bash
python run_local_pipeline.py                        # 위 .env 대로 (korquad 스모크)
python graph.py                                     # 동일 (Serve 까지 — SOURCE_TYPE/URL 동일 계약)
SOURCE_TYPE=file python run_local_pipeline.py       # 기존 hr_policy 데모로 전환
EVAL_MODE=deep EVAL_ENABLE_LLM=1 python run_local_pipeline.py  # 생성·RAGAS 채점(API 비용)
```

#### 동작 원리

어댑터(`agents/eval/datasets/korquad.py`)가 corpus 를 **원문 좌표**로 복원하고, qa 의
`positive_chunk_ids` 를 그 좌표의 `gold_spans` 로 실어 준다. Index 가 자기 전략으로 **재청킹**한
뒤 `_resync_gold_chunk_ids` 가 gold_spans 와 겹치는 현재 청크를 다시 찾아 `gold_chunk_ids` 를
확정한다 → **chunk_size/전략이 바뀌어도(Optimize) gold 가 유지**된다.

- corpus 경로는 **한 곳(`SOURCE_URL`=state.source_url)** 에서만 정한다. Eval 도 gold 좌표
  조회에 같은 파일을 `state.source_url` 로 재사용(별도 corpus env 없음 → 좌표계 어긋남 방지).
- STEP1 흐름: `EVAL_PROBE_SOURCE=taxonomy` → `agent.py` 의 `else` 분기 → `generate_probes()`
  내부 taxonomy 분기 → `_from_taxonomy()`. 코퍼스 버전이 그대로면 캐시 재사용, 바뀌면 재resync.

#### 환경변수 정리

| 환경변수 | 기본 | 설명 |
|----------|------|------|
| `SOURCE_TYPE` / `SOURCE_URL` | `korquad` / `data/corpus.jsonl` | 파이프라인 소스(Ingest·Eval 공용). `run_local_pipeline.py`·`graph.py` 둘 다 읽음 |
| `EVAL_PROBE_SOURCE` | (korquad 면 `taxonomy` 자동) | qa 소스. taxonomy = 외부 골든 QA |
| `EVAL_TAXONOMY_QA` | `data/qa_pairs.jsonl` | taxonomy qa 파일 경로 |
| `KORQUAD_MAX_DOCS` | (스모크 20) | 앞 N개 문서만. 0/미설정=전체. corpus·qa 동일 규칙 |
| `KORQUAD_QA_LIMIT` | (스모크 50) | qa 개수 상한. 0/미설정=전체 |
| `EVAL_MODE` / `EVAL_ENABLE_LLM` | `fast` / off | 진단 깊이·RAGAS. deep 이상+LLM 이면 생성·RAGAS 채점(**API 비용**) |

> 생성 채점(token F1)은 LLM 답변 생성이 있어야 의미가 있다 — LLM을 끄면 추출식 폴백이
> 청크를 통째로 반환해 F1 이 0 에 수렴한다. **검색(recall@k) 지표는 LLM 없이도 유효**하다.

---

## 파일 구조

```
agents/eval/
├── agent.py           # run(state) — STEP1~5 오케스트레이션 + tier2~4 자원 주입(set_diag_context)
├── types.py           # EvalRecord(내부 중간결과) · Mode(tier) · 상수
├── metrics_common.py  # 측정 공통 자원 — 진단 모드(비용 게이트)·자원 컨텍스트(_ctx)·memoize(_cache)
├── probe_gen.py       # STEP1  Probe 생성(user_log/RAGAS 4분면/DataMorgana-lite/무응답)
├── probe_store.py     # STEP1  eval_probes.json 영속화(원문 문서 버전 불변 시 재사용)
├── knowledge_graph.py # STEP1  청크 간 관계 그래프(RAGAS 멀티홉 후보 탐색용, 휴리스틱 전용)
├── llm_provider.py    # LLM 호출 추상화(OpenAI/Gemini/GitHub Models) — probe_gen/metrics_ragas 공용
├── metrics_basic.py   # [tier1] 지표(Recall@k/answer_match/EM) + 청크 경계 측정 — 추가 자원 없음
├── metrics_search.py  # [tier2] 추가 검색 쿼리 측정(gold 순위 재검색/BM25/코퍼스 조회)
├── metrics_ragas.py   # STEP3-2 RAGAS 4지표 + AspectCritic(LLM-as-Judge, 옵션)
├── diagnose.py        # STEP4  16개 라벨 함수(A/B/C/D) → Finding, 브랜치리스 조립
├── report.py          # STEP5  DiagnosticReport 집계(overall_score/pass_threshold/findings_summary)
└── README.md          # 이 파일
```

---

## 주요 확장 지점 `[구현 포인트]`

1. **Probe 자동생성** (`probe_gen.py`, `knowledge_graph.py`, `probe_store.py`) — ✅ 배선 완료.
   `generate_probes()`가 `_allocate_budget()` 비율(75% RAGAS 4분면 / 20% DataMorgana-lite /
   5% 무응답)대로 실제로 섞어 생성하고, `probe_store.py`가 `eval_probes.json`으로 캐시해
   원문 문서가 안 바뀌면 재사용한다. 재청킹 후에도 같은 질문과 `gold_spans`를 유지하고
   현재 청크 기준 `gold_chunk_ids`만 다시 맞춘다(매 Optimize 반복마다 LLM 재호출 방지). 세부:
   - `knowledge_graph.py` — 청크 간 관계 그래프(키워드 Jaccard + 임베딩 코사인, LLM 미사용)와
     `connected_pairs()`로 멀티홉 후보 탐색.
   - `_generate_ragas_probes`(`probe_gen.py`) — 그래프 기반 단일홉(구체/추상) + 멀티홉
     (bridge/comparison/aggregation) 질문 합성(LLM + 휴리스틱 폴백). 그래프가 비면(청크
     부족) `_from_chunks` 단일홉 폴백으로 전체 대체.
   - `_generate_datamorgana_probes` — 거친 스타일(conversational/long/breadth) 조합으로
     단일홉 질문 생성(풀 DataMorgana 대신 최소 버전).
   - `_generate_no_answer_probes`(Held-out·False Premise 절반씩) — `answer_exists=False`,
     `ground_truth=None` probe를 만들어 `_is_success`/`is_abstention` 게이팅이 "정답 없음을
     올바르게 기권"과 "무응답인데 답을 지어내는 생성 실패"를 구분해 진단할 수 있게 함.
   - `_build_doc_position_index`/`_locate_span`/`_resync_gold_chunk_ids` — `gold_spans`(원문
     절대 좌표) 기준으로 재청킹 후에도 `gold_chunk_ids`를 다시 맞추는 유틸. RAGAS와
     DataMorgana 생성기는 LLM의 exact evidence quote를 source chunk 안에서 찾아 좌표를
     채우며, evidence가 없으면 source chunk 좌표로 폴백한다.
   - 남은 일: `state.user_questions` 없이 taxonomy(사람 작성) 소스 자체를 만드는 부분은 미착수.
2. **LLM Provider** (`llm_provider.py`) — ✅ 구현됨. OpenAI 토큰 승인 전 무료 대체용 브릿지.
   `EVAL_LLM_PROVIDER=openai|gemini|github|openrouter` 로 전환, `probe_gen.py`/`metrics_ragas.py`
   가 전부 이 계층만 통해 LLM을 호출한다. 자세한 내용은 위 "LLM Provider" 절 참고.
3. **RAGAS 지표** (`metrics_ragas.py`) — ✅ 구현됨. RAGAS 0.4.3 소스의 프롬프트·예시·조립
   형식을 그대로 옮겨 LLM-as-Judge로 직접 계산(Faithfulness/ContextPrecision/ContextRecall/
   ResponseRelevancy + contradiction AspectCritic). `EVAL_ENABLE_LLM=1`+`EVAL_MODE≥deep`로
   활성화. ragas 라이브러리는 langchain 버전 충돌로 import가 불안정해 미사용 — 환경이
   지원하면 `evaluate_*_track` 내부만 교체해도 결과가 동일하도록 설계.
4. **LLM 생성** (`agents/rag/generator.py`) — STEP2 답변 생성은 공용 RAG 모듈이 담당한다
   (`RAG_LLM_PROVIDER`로 OpenAI/Gemini/GitHub 선택, 키 없으면 추출식 폴백). 프롬프트
   엔지니어링·컨텍스트 랭킹 등 고도화는 rag 모듈 쪽 과제.
5. **Reranker** — Bi-Encoder 후 Cross-Encoder 재정렬(2차 개선). Index 검색이 담당하게 될 영역.
   README 작성 시점 기준 우선순위 낮음.
6. **STEP4 라벨 세트 확장** (`diagnose.py`) — `chunking_context_mismatch`는 구현됨.
   exact `gold_spans`가 현재 청크들의 합집합에는 포함되지만 한 청크에는 온전히 포함되지
   않는 검색 실패를 FAST 모드에서 확정한다(LLM·추가 검색 불필요). 그 밖에 Notion 설계
   문서엔 현재 라벨보다 많은 후보 라벨이 있다.
   `generation_contradiction/misinterpretation/numerical_error/hop_binding_error`는
   `generation_reasoning_failure`(오라클 답변 단일분류)로, `abstention_failure`·
   `parametric_overreliance`·`chunking_over/underchunking`·`reranker_low_precision`도 구현됐다.
   `reranker_low_recall`은 리랭크 전/후 순위 기록이 선행돼야 판별 가능.
7. **임시 검색 제거** — ✅ 완료. `retrieval_temp.py` 는 삭제됐고, `agent.py` 는
   `agents/rag/retriever.py`(`build_retriever`) 검색과 `agents/rag/generator.py`
   (`generate_answer`) 생성을 호출한다.
