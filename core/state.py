"""
core/state.py
LangGraph 공유 상태 정의

모든 에이전트가 이 State를 읽고 씀.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from core.schema import Document, Chunk, Probe, DiagnosticReport


@dataclass
class AgentDoctorState:
    """
    파이프라인 전체 공유 상태.

    흐름:
      Ingest  → documents 채움
      Index   → chunks 채움
      Eval    → probes, report 채움
      Optimize → index_config 수정
      Serve   → mcp_endpoint 채움
    """

    # 입력
    source_url: str = ""
    source_type: str = ""                # "notion" | "gdrive" | "file" | "slack"
    user_questions: list[str] = field(default_factory=list)

    # Ingest Agent 결과
    documents: list[Document] = field(default_factory=list)

    # Index Agent 결과
    chunks: list[Chunk] = field(default_factory=list)
    index_config: dict = field(default_factory=lambda: {
        "chunk_size": 600,
        "chunk_overlap": 80,
        "chunk_strategy": "markdown_recursive",
        "embedding_model": "BAAI/bge-m3",
        "embedding_dimension": 1024,
        "deduplicate": True,
        "top_k": 5,
        "use_hybrid": False,
        "hybrid_dense_weight": 0.7,
        "use_reranker": False,
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "graph_enabled": True,
        "graph_extraction": "auto",
        "graph_llm_model": "gpt-4.1-mini",
        "graph_similarity_threshold": 0.9,
        "graph_output_dir": "output/index_graph",
        "recreate_collection_on_dimension_mismatch": False,
    })
    index_artifacts: dict = field(default_factory=dict)

    # Eval Agent 결과
    probes: list[Probe] = field(default_factory=list)
    report: Optional[DiagnosticReport] = None

    # 반복 제어
    iteration: int = 0
    max_iterations: int = 3

    # Serve Agent 결과
    mcp_endpoint: Optional[str] = None

    # 파이프라인 제어
    status: str = "pending"
    error: Optional[str] = None
    current_agent: str = ""
