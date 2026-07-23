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
| 청킹 | 교체 가능한 4개 전략 | `chunk_strategy`, `chunk_size`, `chunk_overlap` |
| 크기 | 512자, overlap 50자 | `state.index_config` |
| 임베딩 | `BAAI/bge-m3` (1024차원) | `embedding_model`, `embedding_dimension` |
| Vector DB | Qdrant | `QDRANT_URL`, `QDRANT_API_KEY` |
| 검색 | Dense, top-k 5 | `top_k` |
| Hybrid | 기본 OFF | `use_hybrid=True` |
| Reranker | 기본 OFF | `use_reranker=True` |
| Graph | NetworkX + Mermaid/PyVis | `graph_*` 설정 |

Hybrid와 reranker는 baseline 결과를 먼저 측정한 뒤 Optimize가 켜는 기능이다.

## 롤백 2-slot 캐시

Index는 원문과 청킹·임베딩 설정의 fingerprint를 만들고, 현재 버전과 롤백
기준 버전을 최대 두 개까지 보관한다. Qdrant 컬렉션도 코퍼스 namespace마다
`agent_doctor_<namespace>_slot_0`, `agent_doctor_<namespace>_slot_1`
두 개만 사용한다.

- 새로운 후보는 롤백 기준본이 아닌 반대 슬롯을 교체한다.
- 이전 fingerprint로 돌아오면 저장된 `chunks`, 임베딩, 그래프 산출물을 복원한다.
- 같은 프로세스의 retriever 적재 캐시까지 남아 있으면 Qdrant upsert도 생략한다.
- 적재 캐시가 유실돼도 원격 Qdrant 슬롯이 남아 있으면 upsert 없이 다시 연결한다.
- 슬롯도 사라졌다면 저장된 임베딩으로 복구하므로 재임베딩은 하지 않는다.
- 캐시 사용 여부와 최대 개수는 `rollback_cache_enabled`,
  `rollback_cache_max_versions`로 전달하며 현재 상한은 2다.

캐시는 프로세스 메모리의 `state.index_cache`에 있다. 따라서 프로세스 재시작을
가로지르는 영속 롤백은 보장하지 않지만, 고정 Qdrant 슬롯 덕분에 컬렉션이
버전 수만큼 계속 늘어나지는 않는다.

같은 Qdrant에서 동일 코퍼스를 여러 프로세스가 동시에 튜닝한다면 서로 다른
`qdrant_collection_namespace`를 지정해야 한다. 비어 있으면 입력 source URL/type
(없으면 문서 source 목록)에서 namespace를 파생한다. 업그레이드 전에 쓰던 단일
`agent_doctor` 컬렉션은 데이터
안전을 위해 자동 삭제하지 않으므로, 새 슬롯 검증 후 운영 정책에 맞게 별도로
정리한다.

## 청킹 전략 교체

모든 전략은 동일한 `ChunkDraft(text, section, start, end)` 형태를 반환하므로
이후 임베딩·Qdrant·Eval 코드는 바뀌지 않는다.

```python
state.index_config["chunk_strategy"] = "fixed"
state.index_config["chunk_strategy"] = "markdown"
state.index_config["chunk_strategy"] = "recursive"
state.index_config["chunk_strategy"] = "markdown_recursive"

# 번호 단계로도 선택 가능
state.index_config["chunk_stage"] = 1  # fixed
state.index_config["chunk_stage"] = 2  # recursive
state.index_config["chunk_stage"] = 3  # markdown_recursive
```

| 전략 | 동작 | 용도 |
|---|---|---|
| `fixed` | 일정 글자 수로 자름 | 가장 단순한 baseline |
| `markdown` | 제목/소제목 경계만 사용 | 구조 비교 실험 |
| `recursive` | 문단→문장→공백 경계 사용 | 일반 텍스트 |
| `markdown_recursive` | Markdown 1차 분할 후 긴 섹션 재분할 | 기본값 |

한 번의 Index 실행에서는 한 전략을 선택한다. Eval 결과가 낮으면 Optimize가
`chunk_strategy`를 바꾸고 Index를 재실행할 수 있다.

실험용 청킹 함수를 추가할 때는 `agent.py`의 `register_chunk_strategy()`로 등록한다.
각 함수는 `Document, chunk_size, chunk_overlap`을 받아 `ChunkDraft(text, section, start, end)`
계약을 지키면 된다.

## 도구 교체 지점

Index 전체를 갈아엎지 않고 외부 도구만 바꾸려면 `agent.py`의 `IndexTools`를 사용한다.
기본값은 현재 `qdrant_store.py`와 `graph_index.py` 구현을 그대로 묶는다.

```python
from agents.index.agent import IndexTools, run

state = run(
    state,
    tools=IndexTools(
        build_client=my_vector_client,
        ensure_collection=my_collection_setup,
        delete_document_chunks=my_delete,
        upsert_chunks=my_upsert,
        embed=my_embed,
        count_tokens=my_token_counter,
        build_sparse_vector=my_sparse_encoder,
        build_graph_artifacts=my_graph_builder,
    ),
)
```

운영 설정으로 바꿀 때는 `core/state.py`의 `index_config` 기본값과 `agents/serve/api.py`의
검색 호출부도 같은 provider/모델 계약을 보도록 맞춰야 한다.

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
    "index_cache_hit": False,
    "active_index_key": "...",
    "qdrant_collection_name": "agent_doctor_<namespace>_slot_0",
}
```

각 Chunk에는 원문 기준 `char_span`, `token_count`, `hash`, `parent_id`가
보존된다. metadata에는 `document_hash`, `chunk_hash`, `index_signature`,
`embedding_model`과 검색 설정이 들어간다. Eval은 `char_span`으로 재청킹
후에도 gold 위치를 다시 찾고, Serve API는 저장된 모델 설정으로 질문을
동일한 벡터 공간에 임베딩한다.

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
