# 진단 유효성 검증 사용법

우리 진단이 맞는지를 **사람이 라벨링한 정답지**로 재는 방법. 처음 쓰는 사람이 따라 할 수
있게 순서대로 적는다.

- **정답지** [RAGEC](https://github.com/layer6ai-labs/rag-error-classification) — RAG 실패 377건에 사람이 원인을 매긴 데이터(MIT)
- **코퍼스** [DragonBall](https://github.com/OpenBMB/RAGEval) — 그 질문들의 원본 문서(RAGEval 프로젝트)

---

## 한눈에

```
① 원본 3개 파일 내려받기        (1회, 10MB)
② 어댑터로 우리 형식으로 변환    (1회, 수초)
③ 실행 + 채점                   (API 비용)
```

```bash
# ②
python tools/build_ragec_dataset.py \
    --ragec RAGEC_annotations.csv \
    --docs dragonball_docs.jsonl \
    --queries dragonball_queries.jsonl

# ③  먼저 10건으로 배선 확인 → 문제 없으면 전체
python tools/run_ragec_validation.py --limit 10 --embed openrouter
python tools/run_ragec_validation.py --embed openrouter
```

---

## ① 원본 내려받기

`data/` 는 gitignore 라 각자 받는다. 세 파일이 필요하다.

```bash
curl -sLO https://raw.githubusercontent.com/layer6ai-labs/rag-error-classification/main/annotation/RAGEC_annotations.csv
curl -sLO https://raw.githubusercontent.com/OpenBMB/RAGEval/main/dragonball_dataset/dragonball_docs.jsonl
curl -sLO https://raw.githubusercontent.com/OpenBMB/RAGEval/main/dragonball_dataset/dragonball_queries.jsonl
```

| 파일 | 크기 | 내용 |
|---|---|---|
| `RAGEC_annotations.csv` | 400KB | 실패 377건 + 사람이 매긴 원인 라벨 |
| `dragonball_docs.jsonl` | 1.9MB | 원본 문서(영어 108 / 중국어 108) |
| `dragonball_queries.jsonl` | 8.5MB | 질의 6,709건(골드 문서·근거 문장 포함) |

## ② 변환

```bash
python tools/build_ragec_dataset.py \
    --ragec RAGEC_annotations.csv \
    --docs dragonball_docs.jsonl \
    --queries dragonball_queries.jsonl
```

정상 출력:

```
문서 108개 → data/ragec_corpus.jsonl
QA   377건 → data/ragec_qa.jsonl
정답지 377건 → data/ragec_answer_key.jsonl

  gold span 1,511개 (멀티문서 QA 226건)
  답 있는 질문             358
  무응답 질문               19
  근거 좌표화            1,516
  근거 못 찾음               1
  무응답인데 근거 있음(버림)       1
```

**QA 가 377건이 아니면 무언가 어긋난 것이다.** 원본 파일이 최신인지 확인한다.

뒤의 네 줄은 변환 품질이다. `근거 못 찾음 1`·`무응답인데 근거 있음 1` 은 원본 데이터의
흠이라 정상이다(377건 중 각 1건).

### 세 파일이 각각 무엇인가

| 파일 | 누가 읽나 |
|---|---|
| `ragec_corpus.jsonl` | Ingest — 문서를 수집해 Index 가 재청킹 |
| `ragec_qa.jsonl` | Eval — 질문·정답·골드 좌표로 probe 를 만든다 |
| **`ragec_answer_key.jsonl`** | **채점기만.** 파이프라인은 보지 않는다 |

정답 라벨을 파이프라인이 보면 채점이 아니라 커닝이라, 파일을 나눠 뒀다.

## ③ 실행 + 채점

```bash
python tools/run_ragec_validation.py --embed openrouter
```

이 스크립트가 하는 일:

```
환경변수 고정 → Ingest → Index → Eval → findings 덤프 → 채점
```

**환경변수를 안에서 고정한다.** `SOURCE_TYPE`·`SOURCE_URL`·`EVAL_TAXONOMY_QA` 를 손으로
맞추다 하나라도 어긋나면 KorQuAD 설정으로 돌아 **비용만 나가고 채점은 못 한다.**

| 플래그 | 쓸 때 |
|---|---|
| `--limit 10` | 배선 확인. 비용이 거의 없고 전체 경로가 도는지 본다 |
| `--embed openrouter` | 임베딩을 API 로. GPU 가 있으면 `--embed gpu` 가 훨씬 빠르다 |

### 결과

```
output/ragec/findings.jsonl    probe 별 {qa_id, labels, failed}
콘솔                            정확도 표
output/logs/ragec_*.log        실행 로그
```

채점만 다시 하려면(실행 없이):

```bash
python tools/score_ragec.py \
    --findings output/ragec/findings.jsonl \
    --key data/ragec_answer_key.jsonl
```

---

## 결과 읽는 법

```
  라벨 포함 정확도  225/320  (70.3%)
  단계 정확도       247/281  (87.9%)

  카테고리                                  맞음      전체      정확도
  E4 Missed Retrieval                   88     127      69%
  E1 Overchunking                       38      47      81%
  …

  · 우리 파이프라인이 성공한 질문    41건 (진단할 게 없어 제외)
  · 실패했는데 원인을 못 짚음        39건 (오답으로 셈)
  · bad_gold_* 를 낸 probe          12건 (제외 — 사람이 표본 확인 필요)
```

### 무엇을 세고 무엇을 안 세나

| 우리 결과 | 처리 |
|---|---|
| 실패 + 라벨 맞음 | 정답 |
| 실패 + 라벨 틀림 | **오답** — 진짜 진단 오류 |
| **성공** | 제외 |
| 대응 라벨 없음(E15) | 제외 |
| `bad_gold_*` | 제외 |

**"성공 → 제외" 가 중요하다.** RAGEC 377건은 *그들* RAG 시스템이 실패한 질문이다. 검색기·
생성 모델이 다른 우리가 같은 질문에서 성공하는 건 정상이고, 그걸 오답으로 세면 정확도가
진단 품질이 아니라 **"우리가 얼마나 그들과 비슷하게 실패하나"** 를 재게 된다.

**`bad_gold_*` 는 사람이 봐야 한다.** 우리 진단의 오탐일 수도, 실제로 DragonBall 정답이
틀린 것일 수도 있어 자동으로 갈리지 않는다.

### 왜 '포함' 으로 재나

RAGEC 은 **질의당 라벨 1개**이고 검색이 실패하면 생성은 보지 않는다(최초 실패 단계 정책).
우리는 슬롯별로 여러 개를 내고, 생성 라벨은 오라클 트랙을 봐서 검색 실패와 독립적이다.

그래서 우리가 `retrieval_missing_gold` + `generation_hallucination` 을 내고 그쪽이 `E4` 만
적었다면 **틀린 게 아니라 더 말한 것**이다. 그들 라벨이 우리 것 안에 있으면 맞은 것으로 센다.

---

## 이 검증이 답하지 못하는 것

**우리 라벨 31개 중 14개만 잴 수 있다.** 나머지 17개는 정답지에 대응 사례가 없다.

| 그룹 | 검증 가능 |
|---|---|
| A 검색 | 8/15 |
| B 생성 | 5/9 |
| **C 컨텍스트** | **0/3** — RAGEC 4단계에 "컨텍스트 구성" 축이 없다 |
| D 데이터 | 1/4 |

대응 관계와 그 근거는 → [`docs/ragec_label_mapping.md`](ragec_label_mapping.md)

---

## 막히면

| 증상 | 원인 |
|---|---|
| `입력이 없습니다: data/ragec_*.jsonl` | ② 를 안 돌렸다 |
| QA 가 377건이 아님 | 원본 파일이 다르거나 잘렸다. 크기 확인 |
| `근거 못 찾음` 이 많음 | 문서·질의 파일의 버전이 서로 안 맞는다 |
| 첫 실행이 아주 느림 | 임베딩 API 왕복(질의당 1회). `--embed gpu` 로 로컬 전환 |
| 리랭커가 켜져 있고 CPU | cross-encoder 가 질의당 후보 수만큼 추론한다 — CPU 면 매우 느리다 |
| 전체 테스트가 1건 빨감 | `.env` 의 `EVAL_PROBE_SOURCE=taxonomy` 때문 → 이슈 #126 |

---

## 관련 문서

| | |
|---|---|
| [`docs/ragec_label_mapping.md`](ragec_label_mapping.md) | 대조표 — 어느 RAGEC 라벨이 우리 어느 라벨인가, 미확정 2건 |
| `docs/rag_paper_upgrade_roadmap.md` (PR #121) | 로드맵 8번. 이 작업이 왜 필요한가. **아직 main 에 없다** |
| `tests/diagnose_grid/LABELS.md` | 우리 라벨 31개 사전 |
