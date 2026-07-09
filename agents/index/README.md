# Index Module

Ingest가 만든 `Document`를 검증하고, 검색 가능한 `Chunk`와 그래프 산출물로 변환한다.

## 처리 흐름

```text
state.documents
  → 입력 검증·텍스트 정규화
  → 문서/청크 SHA-256 중복 제거
  → Markdown header + 문단 경계 청킹
  → BAAI/bge-m3 dense embedding
  → 선택적 sparse vector 생성
  → Qdrant upsert
  → NetworkX graph + Mermaid/PyVis 출력
  → state.chunks / state.index_artifacts
```

## v1 기본 선택

| 항목 | 기본값 | 변경 방법 |
|---|---|---|
| 청킹 | Markdown 구조 우선 + recursive boundary | `chunk_size`, `chunk_overlap` |
| 크기 | 600자, overlap 80자 | `state.index_config` |
| 임베딩 | `BAAI/bge-m3` (1024차원) | `embedding_model`, `embedding_dimension` |
| Vector DB | Qdrant | `QDRANT_URL`, `QDRANT_API_KEY` |
| 검색 | Dense, top-k 5 | `top_k` |
| Hybrid | 기본 OFF | `use_hybrid=True` |
| Reranker | 기본 OFF | `use_reranker=True` |
| Graph | NetworkX + Mermaid/PyVis | `graph_*` 설정 |

Hybrid와 reranker는 baseline 결과를 먼저 측정한 뒤 Optimize가 켜는 기능이다.

## 파일

```text
agents/index/
├── agent.py          # 검증·중복 제거·청킹·전체 실행
├── qdrant_store.py   # 임베딩·dense/hybrid 검색·reranker·Qdrant
├── graph_index.py    # entity/relation graph와 시각화
└── README.md
```

## 입력·출력 계약

입력:

```python
state.documents: list[Document]
state.index_config: dict
```

출력:

```python
state.chunks: list[Chunk]
state.index_artifacts = {
    "graphml": ".../index_graph.graphml",
    "mermaid": ".../index_graph.md",
    "pyvis": ".../index_graph.html",
    "documents": 2,
    "chunks": 14,
    "reused_embeddings": 0,
}
```

각 Chunk metadata에는 `document_hash`, `chunk_hash`, `index_signature`,
`embedding_model`과 검색 설정이 들어간다. Serve API는 이 값을 읽어 질문도
같은 모델로 임베딩한다.

## Graph 추출

`graph_extraction="auto"`이면 `OPENAI_API_KEY`가 있을 때 LLM JSON extraction을
사용한다. 키가 없거나 호출에 실패하면 keyword extraction으로 폴백한다.

생성 관계:

```text
(Document)-[:contains]->(Chunk)
(Chunk)-[:mentions]->(Entity)
(Entity)-[:related_to]->(Entity)
(Chunk)-[:similar_to]->(Chunk)
```

## 모델을 바꿀 때

문서 임베딩과 검색 임베딩의 모델·차원은 반드시 같아야 한다. 기존 Qdrant
컬렉션과 차원이 다르면 기본적으로 오류를 낸다. 기존 컬렉션 재생성을 명시적으로
허용하려면 다음 값을 사용한다.

```python
state.index_config["recreate_collection_on_dimension_mismatch"] = True
```

운영 데이터가 있는 컬렉션에서는 삭제 영향을 확인한 뒤 사용해야 한다.

## 테스트

```powershell
python -m unittest tests.test_index_unit -v
python tests/test_index.py
```

단위 테스트는 임베딩 모델을 mock 처리하므로 BGE-M3를 다운로드하지 않는다.
