# 진단 유효성 검증 사용법

우리 진단(32개 라벨)이 실제로 맞는지를 재는 절차입니다. 두 갈래로 나뉩니다.

| | 무엇과 비교 | 커버리지 | 비용 |
|---|---|---|---|
| **A. RAGEC 대조** | 공개 데이터셋의 사람 라벨 377건 | 32개 중 16개 | 실행 1회 |
| **B. 자체 라벨링** | 우리 팀이 **우리 실행 결과**를 보고 붙인 라벨 | 32개 전부 | 실행 1회 + 라벨링 시간 |

**B가 본체입니다.** A는 외부 참고치이고, 왜 그런지는 아래 [두 방식의 차이](#두-방식의-차이--왜-b가-본체인가)에 있습니다.

---

## 한눈에

```powershell
# ① 원본 데이터 준비 (최초 1회)
python tools/build_ragec_dataset.py

# ② 실행 — 진단 + 라벨 시트가 함께 나온다
python tools/run_ragec_validation.py --limit 10     # 먼저 10건으로 배선 확인
python tools/run_ragec_validation.py                # 문제 없으면 전체 377건

# ③ A. RAGEC 대조 (실행 시 자동 출력, 재실행도 가능)
python tools/score_ragec.py --findings output/ragec/findings.jsonl

# ④ B. 자체 라벨링 — label_sheet.json 의 '정답라벨' 칸을 채운 뒤
python tools/score_human_labels.py
```

**전체 377건 1회: 약 100분 / API 비용 약 $5.** 비용의 98%가 진단 단계의 판정용 LLM 호출에서 발생합니다.

---

## ① 원본 데이터 준비

```powershell
python tools/build_ragec_dataset.py
```

RAGEC의 오류 라벨과 DragonBall 원문을 결합해 세 파일을 만듭니다.

| 파일 | 내용 |
|---|---|
| `data/ragec_corpus.jsonl` | 문서 108개 (기업 리포트 40 + 법률 판결문 68) |
| `data/ragec_qa.jsonl` | 질문·정답·정답 근거 위치 |
| `data/ragec_answer_key.jsonl` | 사람이 붙인 오류 라벨 377건 |

`--limit`을 붙여 실행하면 QA만 잘리고 **코퍼스는 항상 108개 전부** 색인됩니다. 검색 난이도를 유지하기 위해서입니다.

---

## ② 실행

```powershell
python tools/run_ragec_validation.py --limit 10
```

Ingest → Index → Eval까지만 돕니다. **Optimize는 일부러 돌리지 않습니다** — config가 바뀌고 재평가가 붙으면 "어느 시점의 진단을 채점하는지"가 흐려집니다.

### 나오는 것

```
output/ragec/findings.jsonl     probe별 우리 진단 + 관측값
output/ragec/label_sheet.json   사람이 채울 시트 (우리 진단 없음)
output/ragec/logs/run_*.log     실행 로그 (파이프라인 스모크 로그와 분리)
콘솔                            RAGEC 대조 결과
```

### 옵션

| 옵션 | 뜻 |
|---|---|
| `--limit N` | probe 상한 (0=전체 377) |
| `--label-sample N` | 라벨 시트 표본 수 (기본 60, 0=전체) |
| `--embed {openrouter,gpu,cpu}` | 임베딩 계산 위치 |
| `--rerank {openrouter,gpu,cpu}` | 리랭크 계산 위치 |

`--embed`는 생략해도 `.env`의 `INDEX_EMBED_PROVIDER`를 따릅니다.

> **메모리 주의.** 로컬 임베딩 모델이 작업자당 약 2GB를 점유합니다. 램 8GB 환경에서는 `EVAL_LLM_CONCURRENCY=2` 이하로 두세요. 이보다 높으면 실행 중 중단됩니다(실측).

---

## ③ A. RAGEC 대조

실행 시 자동으로 나오고, 채점만 다시 돌릴 수도 있습니다.

```powershell
python tools/score_ragec.py --findings output/ragec/findings.jsonl
```

**파이프라인 없이 1초 만에 끝납니다.** 대조표(`RAGEC_TO_OURS`)를 고치고 다시 채점하는 반복이 가능합니다.

```
라벨 포함 정확도  n/m     사람이 붙인 원인을 우리도 짚었나
단계 정확도       n/m     어느 단계에서 실패했는지는 맞췄나

· 우리 파이프라인이 성공한 질문         (제외)
· 검색 단계 라벨인데 우리 검색은 성공     (제외)
· 대응 라벨이 없는 카테고리             (제외)
```

### 제외 항목이 왜 있나

RAGEC 377건은 **그들 시스템이 실패한 질문**입니다. 검색기가 다른 우리는 같은 질문에서 성공하거나, 다른 지점에서 실패할 수 있습니다.

```
qa_id 2205
  RAGEC   E4 Missed Retrieval ("검색이 정답 문서를 못 찾았다")
  우리     recall=1.00 · 정답 청크 2/2 검색
          답변에 "Bali, Paris, and New York" 이 그대로 들어 있는데
          "June 20이라고 명시되진 않았다"며 기권
```

우리 검색은 성공했으니 **E4는 우리에게 성립하지 않습니다.** 라벨이 다른 게 오진의 증거가 아닙니다. 그래서 `recall=1.0`이면서 RAGEC가 검색 계열 라벨을 준 건은 채점에서 뺍니다.

판정 근거는 **측정값(recall)이지 우리 라벨이 아닙니다.** 우리 라벨로 판정하면 "진단이 다르니까 봐준다"가 되어 틀릴 수가 없는 채점이 됩니다. 제외 규칙은 정확도를 올리는 방향으로만 작동하므로 **몇 건을 왜 뺐는지 항상 화면에 표시**됩니다.

---

## ④ B. 자체 라벨링 — 이게 본체입니다

### 절차

1. `output/ragec/label_sheet.json`을 엽니다
2. 각 항목의 상황을 읽고 **`정답라벨`** 칸에 라벨 이름을 적습니다
3. 저장 후 `python tools/score_human_labels.py`

결과는 `output/ragec/scores/human_labels_<시각>.txt`(사람용)와 `.json`(실행 간 비교용)으로
자동 저장됩니다. 콘솔에만 두면 창을 닫는 순간 사라지고, 다시 보려면 라벨링을 다시 해야 합니다.

### 시트 모양

```json
{
  "qa_id": "2205",
  "질문": "What new tour destinations did Grand Adventures add on June 20, 2021?",
  "정답": "Bali, Paris, and New York.",
  "우리_답변": "I cannot answer based on the provided information. The context mentions...",

  "검색_recall": 1.0,
  "정답청크_검색됨": "2/2",
  "정답청크_순위": "chunk_004=1위, chunk_005=2위",
  "검색방식": "dense",
  "리랭커": "disabled",

  "f1": 0.244,
  "종합점수": 0.351,
  "faithfulness": 1.0,

  "정답라벨": ""
}
```

**채우는 칸은 하나뿐입니다.** 라벨 목록과 정의는 [`tests/diagnose_grid/LABELS.md`](../tests/diagnose_grid/LABELS.md)에 있습니다.

### 지켜야 할 세 가지

**① 우리 진단을 보지 마세요.** 시트에는 일부러 빼두었습니다. `findings.jsonl`이나 실행 로그를 먼저 보면 '이게 맞나'가 아니라 '동의하는가'를 판단하게 되어, 검증이 성립하지 않습니다.

**② 진단 규칙을 짜지 않은 사람이 붙이는 게 좋습니다.** 규칙을 아는 사람이 붙이면 상황을 판단하는 게 아니라 규칙을 적용하게 됩니다.

**③ 억지로 고르지 마세요.** 두 가지 탈출구가 있습니다.

| 적는 값 | 뜻 | 이게 많이 나오면 |
|---|---|---|
| `해당없음` | 32개 중 맞는 게 없다 | **택소노미에 구멍이 있다** |
| `판단불가` | 주어진 자료로는 못 정하겠다 | **시트에 실을 정보가 부족하다** |

둘 다 정확도 계산에서 빠지고 따로 집계됩니다. 이 둘이 많이 나오는 것 자체가 의미 있는 결과입니다.

### 표본을 다시 뽑고 싶을 때

파이프라인을 다시 돌리지 마세요(100분 + $5). 시트만 다시 만들면 됩니다.

```powershell
python tools/make_label_sheet.py --limit 80 --seed 1
```

표본은 우리 예측 라벨 기준으로 **층화 추출**됩니다. 무작위로 뽑으면 한 라벨로 쏠려(실측: 10건 중 5건이 `retrieval_low_rank`) 희귀 라벨이 표본에 아예 안 들어옵니다.

### 채점 결과 읽는 법

```
라벨 포함 정확도   사람 라벨이 우리 findings 안에 있으면 맞음
라벨 top-1 정확도  우리가 그걸 첫 번째로 냈나
단계 / 그룹 정확도  더 거친 축 — 라벨은 틀려도 방향은 맞았나

자주 어긋난 쌍 (사람 → 우리)
  3×  retrieval_missing_gold  →  retrieval_low_rank
```

**포함과 top-1을 같이 봐야 합니다.** 포함만 보면 라벨을 남발할수록 점수가 오릅니다.

**맨 아래 혼동 쌍이 가장 쓸모 있습니다.** 어디를 고쳐야 하는지는 여기서 나옵니다.

---

## 두 방식의 차이 — 왜 B가 본체인가

같은 질문을 쓰지만, **정답 라벨을 누가 무엇을 보고 붙였느냐**가 다릅니다.

```
A. RAGEC 대조     우리 관측  +  다른 시스템의 관측을 본 사람 라벨   →  출처 불일치
B. 자체 라벨링     우리 관측  +  그 관측을 본 사람 라벨            →  출처 일치
```

RAGEC 논문([arXiv:2510.13975](https://arxiv.org/abs/2510.13975))의 저자들도 **자기 시스템의 오류를** 손으로 라벨링했습니다. 관측과 라벨의 출처를 맞춘 것이 그 검증이 성립한 이유입니다. 우리가 라벨만 빌려오면 그 성질이 깨집니다.

그래서 결과를 보고할 때 표현을 구분해야 합니다.

```
A →  "RAGEC 사람 라벨과의 일치도 X% (16/32 라벨, 호환 표본에 한함)"
B →  "진단 유효성 X%"
```

### 참고 수치

RAGEC 논문이 자기네 자동 분류기를 사람 라벨과 대조한 결과입니다.

| 단계 | 수치 |
|---|---|
| 답변이 틀렸나 (2지선다) | 92.9% |
| 어느 단계인가 (4지선다) | **57.8%** |
| 어느 유형인가 (16지선다) | **40.3%** |

**세분화될수록 급격히 떨어집니다.** 40%대는 실패가 아니라 이 문제의 현재 수준입니다. 저자들도 "far from perfect"라고 적었습니다. 다만 채점 규칙이 달라 우리 수치와 직접 비교는 못 합니다.

> 참고로 이 논문에는 **라벨러 간 일치도(IAA)가 없습니다.** 저자 본인들만 라벨링했기 때문입니다. 우리가 라벨러 2명을 두고 일부 표본을 겹치게 하면, 논문이 하지 않은 것을 하는 셈입니다 — "사람끼리는 얼마나 맞나"라는 천장을 알 수 있습니다.

---

## ⚠ 한 번의 실행으로는 절반밖에 못 잽니다

**가장 중요한 주의사항입니다.**

우리 라벨은 "처방이 다르면 다른 라벨"이라는 원칙으로 나뉘어 있습니다. 그런데 처방 중 상당수가 **같은 스위치의 반대 방향**입니다.

```
retrieval_low_rank           처방: 리랭커를 켜라     ← 리랭커가 꺼져 있어야 진단됨
retrieval_reranker_demotion  처방: 리랭커를 꺼라     ← 리랭커가 켜져 있어야 진단됨
```

이 둘은 **논리적으로 같은 실행에서 나올 수 없습니다.** 라벨 설계가 잘못된 게 아니라, 스위치가 두 상태를 동시에 가질 수 없어서입니다.

### 시나리오별로 무엇이 관측 가능한가

| 설정 | 관측 가능해지는 라벨 | 관측 불가능해지는 라벨 |
|---|---|---|
| **리랭커 OFF** (현재 기본) | `retrieval_low_rank` (확정) | 리랭커 4형제 전부 |
| **리랭커 ON** | `rerank_candidate_miss`<br>`reranker_demotion`<br>`reranker_ineffective`<br>`reranker_low_precision` | `low_rank`가 예비로 강등 |
| **하이브리드 OFF** (현재 기본) | `retrieval_lexical_mismatch` | `rank_fusion_loss` |
| **하이브리드 ON** | `retrieval_rank_fusion_loss` | `lexical_mismatch` 억제 |
| **MMR OFF** (현재 기본) | `retrieval_duplicate_crowding` | — |
| **청킹 `fixed`** (현재 기본) | `chunking_context_mismatch` | — |
| **청킹 `recursive_sentence`** | — | 청킹 라벨이 줄어듦 |

### 리랭커를 켜야 하는 이유 — 정답지가 근거입니다

정답지 377건의 분포입니다.

```
E7 Low Recall      33건  ┐
E8 Low Precision   13건  ┘  46건 = 12%
```

이 46건에 대응하는 우리 라벨 4개는 **전부 리랭크 실행 기록을 전제**로 합니다. 리랭크 전후 순위를 비교해야 갈리기 때문입니다.

```
정답 청크가 리랭커 후보 목록에 있었나?
├─ 없었다              →  rerank_candidate_miss
└─ 있었다
   └─ 리랭크 '전' 순위가 top_k 안이었나?
      ├─ 안이었다 (밀림)  →  reranker_demotion
      └─ 밖이었다        →  reranker_ineffective
```

리랭커가 꺼져 있으면 **"리랭크 전 순위"라는 것 자체가 존재하지 않습니다.** 비교할 두 시점 중 하나가 없으니 판정이 불가능하고, 이 46건은 자동으로 오답이 됩니다.

게다가 `retrieval_low_rank`가 이 상태에서 **잔여 라벨로 모든 순위 실패를 흡수**합니다. 실측 10건에서 실패 7건 중 5건이 이 라벨 하나였습니다.

```powershell
# 리랭커를 켠 실행 (rerank_candidates도 함께 올려야 함)
python tools/run_ragec_validation.py --rerank openrouter
```

> `rerank_candidates`가 기본 20이라 정답이 25~29위인 경우는 리랭커를 켜도 후보창 밖입니다. 정책 상한(50)까지 올려야 사정권에 들어옵니다.

### 권장 — 최소 2회, 가능하면 3회

```
실행 1  현재 기본값               low_rank · lexical_mismatch · duplicate_crowding
실행 2  리랭커 ON + 후보창 확대     리랭커 4형제
실행 3  하이브리드 ON             rank_fusion_loss
```

**각 실행마다 라벨 시트를 따로 만들어 라벨링해야** A그룹 15개 라벨이 고루 검증됩니다. 한 번만 돌리면 그 설정에서 관측 가능한 라벨만 재고 끝납니다.

시나리오를 바꿀 때는 `core/state.py`의 기본값을 고치지 마세요 — 그건 모든 사람의 baseline과 Optimize 데모를 함께 바꿉니다. 실행 스크립트에서만 덮어쓰고, **로그에 기본값과 다르다는 사실을 남겨야** 나중에 그 수치가 어떤 설정에서 나온 건지 알 수 있습니다.

---

## 이 검증이 답하지 못하는 것

**① RAGEC로는 32개 중 16개만 검증됩니다.** 특히 C그룹(컨텍스트 구성) 3개와 D그룹(데이터) 4개는 정답지에 대응 사례가 **0건**이라 전혀 검증되지 않습니다(E2·E11·E15도 0건). 자체 라벨링(B)이 이 구멍을 메웁니다.

**② 택소노미 자체가 맞는지는 검증되지 않습니다.** 라벨러에게 우리 32개 목록을 주고 고르게 하므로, 검증되는 것은 "이 택소노미 안에서 우리가 제대로 배정하는가"입니다. `해당없음`이 많이 나오면 그게 택소노미 문제의 신호입니다.

**③ 처방이 실제로 효과가 있는지는 별개입니다.** 진단이 맞았다는 더 강한 증거는 "그 진단대로 고쳤더니 실제로 나아졌다"입니다. 이건 처방 전후 점수를 비교해야 하고, 개선폭이 평가 편차보다 큰지 판정할 기준선(σ 실측)이 먼저 필요합니다.

---

## 막히면

| 증상 | 원인 / 조치 |
|---|---|
| STEP3에서 실행이 죽음 | 메모리 부족. `EVAL_LLM_CONCURRENCY=2` 로 낮추기 |
| 답변이 질문과 다른 언어로 나옴 | `RAG_ANSWER_LANGUAGE=match` 확인 (스크립트가 자동 설정) |
| `label_sheet.json`에 지표가 비어 있음 | 구버전 덤프. 파이프라인 재실행 필요 |
| 채점기가 JSON 오류를 냄 | 줄 번호를 알려줍니다. 값은 큰따옴표, 마지막 항목 뒤 쉼표 금지 |
| 정확도가 전부 0 | 실패 probe가 없거나 라벨 표기 오타. `--detail` 출력으로 확인 |
| 실행이 KorQuAD 설정으로 돎 | 스크립트가 환경변수를 고정하므로 발생하지 않아야 함. 로그 첫 줄 확인 |

---

## 관련 문서

- [`docs/ragec_label_mapping.md`](ragec_label_mapping.md) — RAGEC 16개 카테고리 ↔ 우리 32개 라벨 대조표
- [`tests/diagnose_grid/LABELS.md`](../tests/diagnose_grid/LABELS.md) — 라벨 정의 사전 (라벨링할 때 참조)
- [RAGEC 논문](https://arxiv.org/abs/2510.13975) / [코드·데이터](https://github.com/layer6ai-labs/rag-error-classification)
