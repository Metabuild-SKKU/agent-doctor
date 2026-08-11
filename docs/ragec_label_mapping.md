# RAGEC ↔ 우리 라벨 대조표

로드맵 8-b. [RAGEC](https://github.com/layer6ai-labs/rag-error-classification)(MIT, 사람 라벨 377건)을
우리 진단의 정답지로 쓰기 위한 대조다. **우리 택소노미로 갈아타는 것이 아니다** — RAGEC 은
분류용(현상이 다르면 다른 라벨), 우리는 처방용(**처방이** 다르면 다른 라벨)이라 갈아타면
라벨→처방→config 패치 사슬이 끊긴다. 정답지로만 쓴다.

원본: `annotation/RAGEC_annotations.csv` · 377행 · 컬럼
`query_id, question, answer, query_type, rag_answer, error_stage, error_category`

---

## 먼저 — 정책이 다르다. 이게 대조보다 크다

| | RAGEC | 우리 |
|---|---|---|
| 질의당 라벨 | **1개** (377 query_id / 377행, 중복 0) | 슬롯별 다중(A·B·C·D) |
| 단계 | Chunking · Retrieval · Reranking · Generation (4) | A 검색 · B 생성 · C 컨텍스트 · D 데이터 (4, **경계가 다름**) |
| 생성 판정 | 검색이 실패하면 생성은 **안 본다** | **오라클 트랙**으로 검색과 독립 판정 |

실제 분포가 이 정책을 드러낸다.

```
Chunking 77(20%) · Retrieval 161(43%) · Reranking 46(12%) · Generation 93(25%)
└──────────── 75% 가 생성 이전 단계에서 끝난다 ────────────┘
```

**RAGEC 의 단일 라벨 정책은 사람이 손으로 한 `k†`(최초 실패 지점 국소화)다.** 우리는 그
방식을 로드맵에서 명시적으로 거부했다 — 생성 라벨이 `_faith_oracle`·`oracle_ragas` 를 보므로
"골드를 통째로 쥐여줬는데도 틀렸다" 가 검색 실패와 독립적으로 성립하기 때문이다.

### 채점은 '포함' 으로 한다

우리가 한 질의에 `retrieval_missing_gold` + `generation_hallucination` 을 내고 RAGEC 이 `E4` 만
적어 뒀다면 우리가 틀린 게 아니라 **더 말한 것**이다. top-1 비교는 이 정책 차이를 오차로
집계한다.

| 방식 | 채택 |
|---|---|
| **포함** — RAGEC 라벨이 우리 findings 안에 있나 | ✅ |
| top-1 — 우리 1순위가 그들 라벨과 같나 | ❌ 정책 차이를 오차로 셈 |
| 단계 — 단계만 맞나 | 보조 지표로 병기 |

---

## 대조표

`?` 는 실제 사례를 봐야 갈리는 자리다. 1:N 이 많은 것은 우리가 31개, RAGEC 이 16개라
**우리가 더 잘게 쪼개 처방을 붙였기** 때문이다 — 그 자체가 결함은 아니다.

### Chunking

| RAGEC | 건수 | 우리 라벨 | 판단 |
|---|---|---|---|
| **E1 Overchunking** | 55 | `chunking_overchunking` **또는** `retrieval_incomplete_enumeration` | ⚠️ **갈림** — 아래 참고 |
| E2 Underchunking | **0** | `chunking_underchunking` | 이름 동일. **정답지에 사례 없음** |
| **E3 Context Mismatch** | 22 | `chunking_context_mismatch` | 이름·정의 일치 |

> **E1 이 갈리는 이유.** 사례가 전부 "여러 사건을 열거해야 하는 요약 질문인데 답변이 일부만
> 담았다" 이다. 근거가 여러 청크로 쪼개져서인지(→ 청크 크기를 키운다) 검색 개수가 모자라서인지
> (→ 개수를 늘린다) 처방이 갈린다. RAGEC 은 이걸 Chunking 으로 봤지만 우리 기준으로는
> `retrieval_incomplete_enumeration` 이 맞는 사례가 섞여 있을 수 있다. **55건은 이 표에서 가장 큰
> 단일 항목이라, 여기 판단이 정확도 수치를 좌우한다.**

### Retrieval

| RAGEC | 건수 | 우리 라벨 | 판단 |
|---|---|---|---|
| **E4 Missed Retrieval** | **152** | `retrieval_missing_gold` | ✅ **확정** — `corpus_gap` 은 0건 |
| **E5 Low Relevance** | 6 | `retrieval_low_rank` · `retrieval_semantic_mismatch` | ? |
| **E6 Semantic Drift** | 3 | `retrieval_semantic_mismatch` | 사례가 "인접 주제로 흘렀다" 라 부합 |

> **E4 는 확정했다.** 152건 전부 `ground_truth.doc_ids` 가 가리키는 문서가 DragonBall 코퍼스에
> 존재한다(실측 152/152, 코퍼스 결손 0건, 골드 문서 미지정 0건). 즉 **문서는 있는데 검색이 못
> 가져온 것**이라 `retrieval_missing_gold` 하나로 대응한다. `corpus_gap` 은 이 정답지로 검증할 수
> 없다(D그룹 커버리지 참고).
>
> 사례도 부합한다 — 정답이 "October 2021" 인데 답변이 "January 2021" 사건을 말한다.

### Reranking

| RAGEC | 건수 | 우리 라벨 | 판단 |
|---|---|---|---|
| **E7 Low Recall** | 33 | `retrieval_rerank_candidate_miss` · `retrieval_reranker_demotion` · `retrieval_reranker_ineffective` | ⚠️ 1:3 |
| **E8 Low Precision** | 13 | `reranker_low_precision` | 이름·정의 일치 |

> E7 이 1:3 인 것은 우리가 **리랭커 실패를 원인별로 쪼갰기** 때문이다 — 후보에 아예 없었나
> (후보창을 넓힌다) / 받고도 떨어뜨렸나(되돌린다) / 봤는데 못 올렸나(모델을 바꾼다). 처방이
> 각각 달라 합칠 수 없다. 포함 채점이면 셋 중 하나만 맞아도 통과다.

### Generation

| RAGEC | 건수 | 우리 라벨 | 판단 |
|---|---|---|---|
| **E9 Abstention Failure** | 23 | `generation_abstention_failure` | 이름·정의 일치. 20건이 `ground_truth="Unable to answer"`(그중 19가 E9) |
| **E10 Fabricated Content** | 3 | `generation_hallucination` | 정의 일치 |
| E11 Parametric Overreliance | **0** | `generation_parametric_overreliance` | 이름 동일. **정답지에 사례 없음** |
| **E12 Incomplete Answer** | 33 | `generation_partial_answer` | 정의 일치 |
| **E13 Misinterpretation** | 29 | `generation_misinterpretation` | 이름·정의 일치 |
| **E14 Contextual Misalignment** | 4 | `generation_misinterpretation` (**우리는 별도 라벨을 두지 않음**) | 처방 동일이라 접었다 |
| E15 Chronological Inconsistency | **0** | (도입 결정됨 — 대응 라벨 신설 예정) | **정답지에 사례 없음** |
| **E16 Numerical Error** | 1 | `generation_numerical_error` | 이름 일치. 표본 1건 |

---

## 우리 라벨 기준 커버리지 — 14/31

RAGEC 으로 **검증할 수 있는 것은 14개뿐**이다. 나머지 17개는 정답지에 대응 사례가 없다.

| 그룹 | 검증 가능 | 검증 불가 |
|---|---|---|
| A 검색 | 8/15 | `rank_fusion_loss` · `duplicate_crowding` · `reranker_demotion` · `reranker_ineffective` · `lexical_mismatch` · `incomplete_enumeration` · `missing_bridge_dependency` |
| B 생성 | 5/9 | `contradiction` · `hop_binding_error` · `wrongful_abstention` · `parametric_overreliance` |
| **C 컨텍스트** | **0/3** | `too_long_context` · `lost_in_the_middle` · `context_noise_interference` |
| D 데이터 | 1/4 | `corpus_gap_partial_hop` · `bad_gold_answer` · `bad_gold_chunk` |

**C그룹이 통째로 0인 이유**는 RAGEC 택소노미에 "컨텍스트 구성" 단계가 없기 때문이다(4단계가
Chunking/Retrieval/Reranking/Generation). 우리 C그룹은 근거도 있고 검색도 됐는데 **컨텍스트를
어떻게 조립했느냐**로 틀리는 경우라, 그들의 축에는 자리가 없다.

> 위 분류는 이름·정의 기준 **1차 추정**이다. 배관(8-c) 후 실제 실행에서 어느 라벨이 발화하는지
> 보면 확정된다 — 특히 E1(55건)의 갈림과 E4(152건)의 `corpus_gap` 여부.

### `bad_gold_*` 는 결과 해석에 주의가 필요하다

`bad_gold_answer`·`bad_gold_chunk` 는 "정답지가 틀렸다" 는 라벨이다. RAGEC 은 사람이 검수한
정답지라 그런 사례가 없는 게 당연하다. 그런데 **DragonBall 정답지에도 오류가 있을 수 있으므로**,
우리 파이프라인이 이 라벨을 내면 두 가능성이 갈리지 않는다.

- 우리 진단의 오탐
- 실제로 DragonBall 정답이 틀림

정확도 계산에서 **별도 항목으로 빼고 표본을 사람이 확인**해야 한다.

---

## E15 — 미관측이 도입 반대 근거가 되지 못한다

E15 는 377건 중 0건이다. 처음에는 "시간 순서 질문이 59건이나 실패했는데 하나도 E15 가 아니다"
를 도입 반대 근거로 봤으나, **그 판단은 틀렸다.**

```
시간 순서 질문 실패                59건
  그중 Generation 단계까지 간 것      5건   ← E15 가 나올 수 있었던 전부
     E12 Incomplete Answer          3
     E13 Misinterpretation          2
```

나머지 54건은 검색·청킹에서 끝나 생성 단계를 보지도 못했다(위 단일 라벨 정책). **표본이 5건인
0건은 아무 증거도 아니다.**

E2·E11 도 같은 이유로 0건인데 우리는 그 라벨을 이미 갖고 있고 쓸모 있다고 본다. **0건은 "이
코퍼스 + 이 RAG 시스템에서 안 나왔다" 이지 "그 실패가 없다" 가 아니다.**

→ E15 도입은 이 데이터와 무관하게 판단한다. 다만 **RAGEC 으로는 E15 진단을 검증할 수 없다**는
사실은 기록해 둔다.

---

## 질문 유형 분포 — 코퍼스 교체의 근거

| 질문 유형 | RAGEC 377 | 우리 실행(KorQuAD 30) |
|---|---|---|
| Multi-document Information Integration | 120 | 0 |
| Multi-document Time Sequence | 59 | 0 |
| Multi-hop Reasoning | 50 | 0 |
| Multi-document Comparison | 47 | 0 |
| Factual | 33 | **30 (100%)** |
| Summarization | 30 | 0 |
| Irrelevant Unsolvable | 25 | 0 |
| Summary | 13 | 0 |

우리 31개 중 19개가 미발화하는 이유가 여기 있다 — 멀티홉 질문이 0이면 멀티홉 라벨은
**원리적으로** 못 나온다. **정답지를 얻는 것이자 질문 유형을 얻는 것이다.**

> `Summarization Question`(30)과 `Summary Question`(13)이 따로 존재한다. 같은 개념을 다르게 적은
> 것으로 보이며 데이터 자체의 흠일 수 있다 — 집계 시 합칠지 정해야 한다.

---

## 이 표에서 아직 확정되지 않은 것

대응 자체는 13종 전부 적었지만, **아래 둘은 배관(8-c) 전에는 확정할 수 없다.** 사례마다
갈리는 것이라 표를 보고 정할 수 없고, 실제로 돌려 봐야 안다.

| | 미확정 | 왜 |
|---|---|---|
| **E1 Overchunking (55건)** | `chunking_overchunking` ↔ `retrieval_incomplete_enumeration` | 근거가 청크로 쪼개져서인지 검색 개수가 모자라서인지는 실제 청킹·검색 결과를 봐야 갈린다. **처방이 다르다**(청크 크기 ↔ 검색 개수) |
| **E5 Low Relevance (6건)** | `retrieval_low_rank` ↔ `retrieval_semantic_mismatch` | 순위가 밀린 것인지 표현이 안 맞은 것인지. 표본이 작아 영향은 제한적 |

E1 은 **377건 중 55건(15%)** 으로 단일 항목 2위다. 포함 채점에서는 우리가 둘 중 하나만 내도
통과하므로 **정확도 수치 자체는 안 흔들리지만**, "어느 쪽으로 진단했나" 를 따로 집계해 두면
청킹 처방과 검색 처방 중 무엇이 실제로 맞았는지 볼 수 있다.

나머지 11종은 이름·정의가 일치하거나(E3·E8·E9·E10·E12·E13·E16) 실측으로 확정했다(E4).
E14 는 우리가 별도 라벨을 두지 않기로 한 자리이고, E2·E11·E15 는 사례가 없다.

---

## 다음 (8-c)

```
1. dragonball_docs.jsonl (영어 108개) → corpus.jsonl 형식
2. ground_truth.references(근거 '문장') → 문서에서 위치를 찾아 gold_spans 좌표로
     ← tools/build_clean_qa.py 가 KorQuAD 에 한 일과 같다. 재활용한다
3. RAGEC 377건의 query_id 로 probe 생성
4. 파이프라인 실행 → 우리 라벨 산출
5. 포함 채점 + 단계 채점 병기
```

**조인은 확인했다** — RAGEC `query_id` → DragonBall 영어 질의 **377/377**, 질문 텍스트 100% 일치.

| 출처 | 규모 |
|---|---|
| `dragonball_docs.jsonl` | 영어 108개 · 본문 중앙값 **8,564자**(우리 KorQuAD 8,020자와 동급) |
| `dragonball_queries.jsonl` | 영어 3,108건 · `ground_truth.doc_ids`(골드 문서) · `references`(근거 문장) 2,893건 |
