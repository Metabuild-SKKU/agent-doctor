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
        # Index가 실제 지원하는 모델만 지정할 것 — 미지원 문자열은
        # SentenceTransformer()에 그대로 전달되어 로드가 멈출 수 있음
        "embedding_model": "BAAI/bge-m3",
        # GPU가 있으면 CUDA, 없으면 CPU를 선택한다. 명시적으로 "cpu"/"cuda"도 지정 가능하다.
        "embedding_device": "auto",
        # 대용량 문서의 청크를 한 건씩 추론하지 않고 묶어서 임베딩한다.
        "embedding_batch_size": 16,
        # Eval STEP1 지식그래프는 임베딩 유사도를 GPU 블록 행렬곱으로 계산한다.
        "eval_graph_device": "auto",
        "eval_graph_top_k": 12,
        "eval_graph_batch_size": 512,
        "use_hybrid": True,
        # 검색 시 가져올 청크 수. Eval(agents/eval/agent.py)이 검색에, Index가 청크
        # metadata 기록에 소비한다. 둘 다 미지정 시 5를 폴백으로 쓰고 있어 같은 값을
        # 명시해 동작을 바꾸지 않으면서 Optimize가 조정할 baseline을 드러낸다.
        "top_k": 5,
        # gold span 길이 분포에서 chunk_size 탐색 후보를 만드는 정책.
        # Optimize가 상태를 통해 읽도록 두어 코드 하드코딩 없이 조정할 수 있다.
        "chunk_candidate_policy": {
            "target_quantile": 0.85,
            "margin_ratio": 0.20,
            "rounding_step": 50,
            "path_fractions": [0.33, 0.66, 1.0],
            "candidate_count": 3,
            "min_span_count": 3,
        },
        # 청크 경계에서 잘린 gold span이 한 청크에 다시 들어오도록
        # chunk_overlap 후보를 만드는 정책. 후보의 최종 선택은 실제 청커
        # dry-run 결과(전체 포함률과 중복량)를 사용한다.
        "chunk_overlap_candidate_policy": {
            "target_quantiles": [0.50, 0.85, 0.95],
            "rounding_step": 25,
            "candidate_count": 3,
            "min_crossing_span_count": 1,
            "max_ratio": 0.40,
            "max_overlap": 300,
        },
        # 임베딩 모델 교체로 벡터 차원이 달라졌을 때 Qdrant 컬렉션을 재생성할지 여부.
        # False(기본)면 차원 불일치 시 ensure_collection이 ValueError로 막는다.
        "recreate_collection_on_dimension_mismatch": False,
    })
    # 인덱싱 부산물(청크/문서 수, 그래프 파일 경로, failed_documents 등).
    # 선언 없이 동적 속성으로 쓰면 LangGraph가 노드 간 상태 복사 시 유실할 수 있다.
    index_artifacts: dict = field(default_factory=dict)

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
    # Optimize가 바꾼 설정 때문에 청크/임베딩을 다시 만들 필요가 있는지 표시한다.
    # Index는 False를 한 번 소비하면 기존 인덱스를 유지한 채 Eval로 넘기고 True로 복원한다.
    reindex_required: bool = True

    # Serve Agent 결과
    mcp_endpoint: Optional[str] = None

    # 파이프라인 제어
    status: str = "pending"
    error: Optional[str] = None
    current_agent: str = ""
