"""
core/state.py
LangGraph 공유 상태 정의

모든 에이전트가 이 State를 읽고 씀.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from core.schema import (
    Document,
    Chunk,
    Probe,
    DiagnosticReport,
    IndexSnapshot,
    EvalSnapshot,
)


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
        # 검증 baseline은 dense-only(off)로 둔다 — use_reranker/use_mmr과 같은 취지.
        # hybrid가 켜져 있으면 retrieval_lexical_mismatch 실패 모드가 잘 안 생기고
        # enable_hybrid 처방도 no-op으로 걸러져 "단어 불일치 → hybrid" 개선을 못 보여준다.
        # (retriever가 dense + keyword fallback을 지원해 off로도 안전.)
        "use_hybrid": False,
        # baseline 검색 결과를 먼저 측정한 뒤 retrieval_low_rank 처방이 켠다.
        # 모델명과 후보 수를 상태에 명시해 Eval과 Serve가 같은 reranker 계약을 쓴다.
        "use_reranker": False,
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "rerank_candidates": 20,
        # MMR 다양성 재정렬. 공통 Retriever가 소비한다(retriever.py). 기본 off로 두어
        # 동작을 바꾸지 않으면서 Optimize(enable_mmr)가 조정할 baseline을 드러낸다.
        "use_mmr": False,
        "mmr_lambda": 0.5,
        "mmr_candidates": 20,
        # Index가 optional 검색 모델을 실제 로드·추론해 capability를 생산한다.
        "reranker_preflight": "eager",
        # 검색 시 가져올 청크 수. Eval(agents/eval/agent.py)이 검색에, Index가 청크
        # metadata 기록에 소비한다. 둘 다 미지정 시 5를 폴백으로 쓰고 있어 같은 값을
        # 명시해 동작을 바꾸지 않으면서 Optimize가 조정할 baseline을 드러낸다.
        "top_k": 5,
        # Optional context filtering before answer generation. Off by default.
        "context_compression": False,
        "chunk_strategy": "fixed",
        "embedding_dimension": 1024,
        "deduplicate": True,
        "hybrid_dense_weight": 0.7,
        # gold span 길이 분포에서 chunk_size 탐색 후보를 만드는 정책.
        # Optimize가 상태를 통해 읽도록 두어 코드 하드코딩 없이 조정할 수 있다.
        "chunk_candidate_policy": {
            "target_quantile": 0.85,
            "margin_ratio": 0.20,
            "rounding_step": 50,
            "path_fractions": [0.33, 0.66, 1.0],
            "candidate_count": 3,
            "min_span_count": 3,
            # 한 번의 처방에서 현재 chunk_size의 ±25% 밖으로 움직이지 않는다.
            # evidence 분포의 긴 꼬리나 짧은 answer span 하나가 과격한 재청킹을
            # 일으키지 않도록 하는 탐색 범위 안전장치다.
            "max_step_ratio": 0.25,
            "min_chunk_size": 200,
            # evidence_window_policy.max_chars는 일반 문맥 확장 상한이고 아래 값은
            # margin 적용 후 후보의 절대 상한이다. gold 자체가 window 상한보다
            # 길면 gold 보존이 우선하므로 두 값은 같은 제한을 뜻하지 않는다.
            "max_chunk_size": 1500,
        },
        # gold 정답 문자열을 그대로 재지 않고, 정답을 이해하는 데 필요한
        # 문장·문단·표 행·리스트 항목의 연속 문맥으로 확장하는 정책.
        "evidence_window_policy": {
            "min_chars": 100,
            "max_chars": 1000,
            "heading_max_distance": 200,
            "adjacent_context_blocks": 1,
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
        "graph_enabled": True,
        "graph_extraction": "auto",
        "graph_llm_model": "gpt-4.1-mini",
        "graph_similarity_threshold": 0.9,
        "graph_output_dir": "output/index_graph",
        "corpus_visualization_enabled": True,
        "corpus_visualization_output_dir": "output/corpus_visualization",
        "corpus_visualization_max_points": 500,
        # 임베딩 모델 교체로 벡터 차원이 달라졌을 때 Qdrant 컬렉션을 재생성할지 여부.
        # False(기본)면 차원 불일치 시 ensure_collection이 ValueError로 막는다.
        "recreate_collection_on_dimension_mismatch": False,
        # 롤백 캐시는 현재/직전 버전만 유지한다. 현재 구현은 최대 2개만 지원한다.
        "rollback_cache_enabled": True,
        "rollback_cache_max_versions": 2,
        # 같은 Qdrant를 여러 파이프라인이 공유하면 고유 namespace를 지정한다.
        # 비어 있으면 Index가 문서 ID/source 조합에서 안정적으로 파생한다.
        "qdrant_collection_namespace": "",
        # ── 생성(답변) 설정 ──────────────────────────────────────────
        # Eval/Serve가 generate_answer로 넘기고 Optimize(B그룹)가 조정한다.
        # use_hybrid/use_reranker처럼 index_config에 함께 담되, 재색인은 키별로
        # 결정되므로(REINDEX_PATHS) 이 키들은 재색인을 유발하지 않는다.
        # 기본값은 현재 하드코딩 동작을 그대로 재현한다(Optimize가 손대기 전엔 불변).
        # 실제 프롬프트 문구는 generator가 이 플래그들로 조립한다
        # (rules는 스위치, 소비처가 문구 — use_hybrid와 같은 패턴).
        # 평가 재현성을 위해 기본 0.0(결정적) — 리뷰 합의(#51). 답변 생성이 실행마다
        # 흔들리면 같은 config 재평가가 ±수 점 튀어 유지/롤백이 노이즈에 좌우된다.
        # 트레이드오프: 0.0이면 lower_temperature 처방이 더 내릴 여지가 없어 no-op 이 된다
        # (온기가 필요한 코퍼스가 확인되면 baseline 을 올려 그 처방을 되살릴 수 있다).
        "temperature": 0.0,
        "grounding_strict": True,    # 컨텍스트만 근거로, 없으면 모른다고 (현 프롬프트)
        "require_citation": True,    # 답변 끝 근거 번호 표시 (현 프롬프트)
        "restate_question": False,   # 답변 전 질문 재진술 강제
        "completeness_mode": False,  # 모든 하위 질문에 빠짐없이 답하도록 유도
        "abstention_strict": False,  # 확신 없으면 반드시 기권하도록 강화
        "generation_model": "",      # Tier3(모델 교체)용 자리. 빈값=env/기본 사용
    })
    # 인덱싱 부산물(청크/문서 수, 그래프 파일 경로, failed_documents 등).
    # 선언 없이 동적 속성으로 쓰면 LangGraph가 노드 간 상태 복사 시 유실할 수 있다.
    index_artifacts: dict = field(default_factory=dict)
    # Index Agent가 생산하고 Optimize가 자동 처방 가능 여부를 판단할 때 읽는다.
    runtime_capabilities: dict = field(default_factory=dict)
    # Index Agent 소유. 현재/직전 인덱스 스냅샷을 LRU 순서로 최대 두 개 보관한다.
    index_cache: list[IndexSnapshot] = field(default_factory=list)
    active_index_key: str = ""
    index_cache_hit: bool = False

    # Eval Agent 결과
    probes: list[Probe] = field(default_factory=list)
    report: Optional[DiagnosticReport] = None
    # 진단 신호 캐시: {probe_id: {signal_name: value}}. 같은 파이프라인 버전 내 재진단 시 재사용.
    # 버전(index_config+코퍼스)이 바뀌면 Eval 진입 시 무효화한다.
    diagnosis_cache: dict = field(default_factory=dict)
    diagnosis_cache_version: str = ""
    # Eval Agent 소유. 현재/직전 완전 진단 결과를 LRU 순서로 최대 두 개 보관한다.
    eval_cache: list[EvalSnapshot] = field(default_factory=list)
    active_eval_key: str = ""
    eval_cache_hit: bool = False

    # 반복 제어
    iteration: int = 0
    max_iterations: int = 3
    optimize_visit_count: int = 0
    max_optimize_visits: int = 20

    # Optimize Agent 이력/제어
    # optimization_history: 각 처방 시도 기록. 원소는 OptimizationHistoryItem
    #   (agents/optimize/schemas.py). core가 optimize를 import하지 않도록 타입은 느슨하게 둔다.
    # blacklist: 롤백된 (label, prescription_id) 조합. planner가 재시도에서 제외.
    optimization_history: list = field(default_factory=list)
    blacklist: set = field(default_factory=set)
    # 내부 sweep을 정상 완료한 (label, prescription_id) 조합.
    # 같은 파이프라인 실행에서 이미 탐색을 끝낸 처방을 다시 시작하지 않도록 한다.
    completed_prescriptions: set = field(default_factory=set)
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
