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

## 사용법

파일을 둔 뒤 설정·실행 방법은 → [agents/eval/README.md](../agents/eval/README.md) 의
**“KorQuAD 2.1 데이터셋으로 평가”** 절 참고.
