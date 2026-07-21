"""RAGBuilder 0.1.6에서 strict hybrid 검색을 구성하는 custom retriever."""
from __future__ import annotations

from typing import Any


class StrictHybridRetriever:
    """항상 dense와 BM25를 함께 사용하는 EnsembleRetriever 팩토리.

    RAGBuilder의 custom retriever 로더는 지정된 class를 생성하고 그 반환값을
    retriever로 사용한다. 이 팩토리는 별도 wrapper 계층을 만들지 않고 LangChain의
    EnsembleRetriever를 직접 반환해 RAGBuilder의 기존 실행 계약을 유지한다.
    """

    def __new__(
        cls,
        vectorstore: Any,
        retriever_k: int = 20,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> Any:
        """surrogate index와 동일한 chunks로 dense+BM25 retriever를 만든다."""

        # optional dependency는 native RAGBuilder 실행 시점에만 불러온다.
        from langchain.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever
        from ragbuilder.core.config_store import ConfigStore

        if vectorstore is None:
            raise ValueError("strict hybrid 구성에 vectorstore가 필요합니다.")
        if retriever_k < 1:
            raise ValueError("retriever_k는 1 이상이어야 합니다.")

        ingest_pipeline = ConfigStore().get_best_data_ingest_pipeline()
        if ingest_pipeline is None:
            raise ValueError("strict hybrid 구성 전에 data ingest를 실행해야 합니다.")

        # RAGBuilder의 native BM25 경로와 동일하게 best ingest pipeline에서 chunks를
        # 생성한다. 따라서 dense index와 lexical retriever가 같은 surrogate corpus와
        # chunking 결과를 사용한다.
        chunks = ingest_pipeline.ingest(status=None)
        if not chunks:
            raise ValueError("strict hybrid BM25를 구성할 chunk가 없습니다.")

        weights = [float(dense_weight), float(bm25_weight)]
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("hybrid retriever weight는 음수가 아니며 합이 0보다 커야 합니다.")
        total = sum(weights)
        normalized_weights = [weight / total for weight in weights]

        dense = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retriever_k},
        )
        lexical = BM25Retriever.from_documents(chunks, k=retriever_k)
        return EnsembleRetriever(
            retrievers=[dense, lexical],
            weights=normalized_weights,
        )
