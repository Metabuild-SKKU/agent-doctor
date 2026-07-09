# Index Agent

Ingest Agent가 수집한 `Document`를 청킹 → 임베딩 → Qdrant 저장해 검색 가능한 상태로 만드는 에이전트.

---

## 역할

```
[Ingest Agent] → state.documents
                        ↓
                 [Index Agent]
                        ↓
        청킹 → 임베딩 → Qdrant 저장
                        ↓
                 state.chunks  →  [Eval Agent]로 전달
```

---

## 처리 흐름

```
Document.content (전체 텍스트)
    ↓ _chunk_text()
["청크1", "청크2", ...]  (512자 단위, overlap 50)
    ↓ embed()
[[0.12, -0.34, ...], ...]  (384차원 벡터)
    ↓ upsert_chunks()
Qdrant 컬렉션 "agent_doctor"
    ↓
state.chunks (Chunk 리스트, embedding 포함)
```

---

## 출력 데이터 (`state.chunks`)

```python
@dataclass
class Chunk:
    chunk_id    : str               # "{doc_id}_chunk_{index}"
    doc_id      : str               # 원본 문서 ID
    text        : str               # 청크 텍스트
    embedding   : list[float]       # 384차원 벡터 (sentence-transformers)
    sparse_vector: dict | None      # 향후 BGE-M3 hybrid 검색용
    metadata    : dict              # title, source, chunk_index 등
```

---

## 사용 기술

| 항목 | 현재 (테스트) | 향후 (운영) |
|------|-------------|------------|
| 청킹 | 고정 크기 512자, overlap 50 | - |
| 임베딩 모델 | `paraphrase-multilingual-MiniLM-L12-v2` (384차원, 무료) | - |
| 벡터 DB | Qdrant in-memory | Qdrant Cloud |
| 한국어 지원 | 보통 | 우수 (OpenAI) / 최상 (BGE-M3) |



---

## 테스트

### 단독 테스트 (mock Document)

```bash
python tests/test_index.py
```

mock Document 2개로 청킹 + 임베딩 결과 확인. Ingest Agent 없이 실행 가능.

### 파이프라인 테스트 (Notion 연동)

```bash
python tests/test_pipeline.py
```

실제 Notion 페이지 수집 → Index → Serve까지 연결.

---

## 파일 구조

```
agents/index/
├── agent.py          # run(state) — 청킹 + 임베딩 + Qdrant upsert
├── qdrant_store.py   # Qdrant 클라이언트 / upsert / 검색 / 임베딩 공통 모듈
└── README.md         # 이 파일
```

---

## qdrant_store.py 공개 인터페이스

Serve Agent의 `api.py`도 이 모듈을 import해서 사용함.

```python
from agents.index.qdrant_store import build_client, ensure_collection, upsert_chunks, search, embed

build_client(url=":memory:")          # Qdrant 클라이언트 생성
ensure_collection(client)             # 컬렉션 없으면 생성
upsert_chunks(client, chunks)         # Chunk 리스트 저장
search(client, query_vector, top_k)   # 벡터 유사도 검색
embed("텍스트")                       # 텍스트 → 벡터
```
