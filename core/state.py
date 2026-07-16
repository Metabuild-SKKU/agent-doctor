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
        "chunk_size": 512,
        "chunk_overlap": 50,
        "embedding_model": "openai://text-embedding-3-small",
        "use_hybrid": True,
        # 검색 시 가져올 청크 수. Eval(agents/eval/agent.py)이 검색에, Index가 청크
        # metadata 기록에 소비한다. 둘 다 미지정 시 5를 폴백으로 쓰고 있어 같은 값을
        # 명시해 동작을 바꾸지 않으면서 Optimize가 조정할 baseline을 드러낸다.
        "top_k": 5,
    })

    # Eval Agent 결과
    probes: list[Probe] = field(default_factory=list)
    report: Optional[DiagnosticReport] = None
    # 진단 신호 캐시: {probe_id: {signal_name: value}}. 같은 파이프라인 버전 내 재진단 시 재사용.
    # 버전(index_config+코퍼스)이 바뀌면 Eval 진입 시 무효화한다.
    diagnosis_cache: dict = field(default_factory=dict)
    diagnosis_cache_version: str = ""

    # 반복 제어
    iteration: int = 0
    max_iterations: int = 3

    # Optimize Agent 이력/제어
    # optimization_history: 각 처방 시도 기록. 원소는 OptimizationHistoryItem
    #   (agents/optimize/schemas.py). core가 optimize를 import하지 않도록 타입은 느슨하게 둔다.
    # blacklist: 롤백된 (label, prescription_id) 조합. planner가 재시도에서 제외.
    optimization_history: list = field(default_factory=list)
    blacklist: set = field(default_factory=set)
    # optimization_report: 이번 Optimize 방문의 사용자용 처방 리포트.
    #   원소 타입은 OptimizationReport(agents/optimize/schemas.py). core가 optimize를
    #   import하지 않도록 타입은 느슨하게 둔다. Serve가 마지막 방문 리포트를 읽는다.
    optimization_report: Optional[object] = None

    # Serve Agent 결과
    mcp_endpoint: Optional[str] = None

    # 파이프라인 제어
    status: str = "pending"
    error: Optional[str] = None
    current_agent: str = ""
