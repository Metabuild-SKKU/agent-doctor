# data/ — 평가 데이터셋

`data/` 폴더는 **gitignore** 대상(용량 큼)이라 이 README 만 커밋되고 실제 데이터는 각자 준비한다.
KorQuAD 2.1 로 평가하려면 아래 **두 파일**을 여기에 둔다.

```
data/
├── corpus.jsonl     # 코퍼스(문서별 청크) — Ingest 가 원문 복원해 수집
└── qa_pairs.jsonl   # 골든 QA(정답+gold 청크) — Eval 이 taxonomy Probe 로 로드
```

## 파일 스키마 (JSONL — 한 줄에 JSON 하나)

### `corpus.jsonl`

```json
{"doc_id": "doc_a39...", "chunk_id": "doc_a39..._0", "title": "문서 제목",
 "text": "청크 본문 …", "char_start": 0, "char_end": 484}
```

| 필드 | 설명 |
|------|------|
| `doc_id` | 문서 ID (같은 문서의 청크는 같은 값) |
| `chunk_id` | 청크 ID (문서 내 고유) |
| `title` | 문서 제목 |
| `text` | 청크 본문 |
| `char_start` / `char_end` | 원문 내 이 청크의 문자 좌표(문서별 0 기준). **gold 좌표 복원의 기준** |

> Ingest 는 같은 `doc_id` 청크들을 `char_start/end` 좌표에 되붙여 원문 `Document` 로 복원하고,
> Index 가 자기 전략으로 **재청킹**한다. corpus 의 `chunk_id` 는 그대로 쓰이지 않는다(참고용).

### `qa_pairs.jsonl`

```json
{"qa_id": "38824", "question": "파스칼레 소틸레의 스파이크 높이는 몇 cm인가?",
 "answer_text": "332cm", "doc_id": "doc_a39...",
 "positive_chunk_ids": ["doc_a39..._3", "doc_a39..._4"]}
```

| 필드 | 설명 |
|------|------|
| `qa_id` | 질문 ID |
| `question` | 질문 |
| `answer_text` | 정답(→ `Probe.ground_truth`, token F1 채점 기준) |
| `doc_id` | 정답이 있는 문서 ID |
| `positive_chunk_ids` | 정답이 든 corpus 청크 ID들 → 원문 좌표로 변환돼 `gold_spans` 가 됨(Recall@k 기준) |
| `gold_spans` | *(선택)* 골드 좌표를 직접 지정 `[{"doc_id", "start", "end"}]`. **있으면 `positive_chunk_ids` 보다 우선한다** |

### `gold_spans` 를 왜 따로 두나 — 정제본(`qa_pairs_clean.jsonl`)

`positive_chunk_ids` 로 골드를 적으면 **골드가 정답이 아니라 정답이 든 청크 통째**가 된다.
이 데이터셋 실측으로 골드 폭 중앙값 **497자**, 정답 길이 중앙값 **7자** — 70배다.
`span_recall_at_k` 는 골드 구간을 **빈틈없이 덮어야** 1점인 이진 판정(부분 점수 없음)이라,
corpus 청크 경계와 Index 재청킹 경계가 다르면 정답을 맞힌 실행도 `recall=0` 이 된다.

여기에 두 번째 결함이 겹친다. 최초 전처리가 정답 텍스트를 문서에서 **다시 찾는** 방식이라,
표 문서처럼 같은 값이 여러 행에 나오면 앞의 것에 꽂힌다.

```
"파스칼레 소틸레의 스파이크 높이는?" → 정답 "332cm" 가 문서에 8곳
  골드 1188~2028  ← 다른 선수의 332cm
  실제 5268       ← "소틸레"(문서에 1회)가 있는 행
```

실측(`output/logs/corpus_20260804_103059.txt`)에서 30문항 중 **4건(13%)** 이 `answer=1.00`
인데 `recall=0.00` 이었다.

정제본은 **정답이 문서에 정확히 한 번만 나오는 QA 만 남기고** 골드를 그 한 곳으로 좁힌다.

```bash
python tools/build_clean_qa.py              # → data/qa_pairs_clean.jsonl
```

| | 원본 | 정제본 |
|---|---|---|
| 건수 | 1,718 | 654 |
| 골드 폭(중앙값) | 497자 | **6자** |
| 덮어야 할 청크 | 2~3개 | **1개** |

> 좌표 복원은 파이프라인과 **같은 함수**(`korquad._stitch`)를 쓴다. 이어붙이기(`"".join`)로
> 하면 청크가 겹치는 만큼 뒤쪽 좌표가 밀린다 — 이 코퍼스는 1,000개 문서가 전부 겹쳐 있어
> 초기 구현의 골드 549건 중 539건(98%)이 어긋나 있었다(리뷰 지적으로 수정).

원본 파일은 **그대로 둔다.** 쓰려면 Eval 설정만 바꾼다 — `EVAL_TAXONOMY_QA=data/qa_pairs_clean.jsonl`.

## 사용법

파일을 둔 뒤 설정·실행 방법은 → [agents/eval/README.md](../agents/eval/README.md) 의
**“KorQuAD 2.1 데이터셋으로 평가”** 절 참고.
