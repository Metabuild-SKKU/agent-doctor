"""
core/schema.py
Agent Doctor v2 공통 데이터 스키마 (Layer 0)

모든 에이전트가 공유하는 데이터 모델.
해당 스키마를 기반으로 에이전트를 구현
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Document:
    """
    수집된 원본 문서 단위.
    Ingest Agent가 생성 → Index Agent가 소비.
    """
    doc_id: str
    source: str                          # 원본 경로/URL
    format: str                          # pdf | md | html | docx | hwp | csv | xlsx
    content: str                         # 전처리된 순수 텍스트
    metadata: dict = field(default_factory=dict)
    # metadata 예시: author, created_at, version, title
    ingested_at: datetime = field(default_factory=datetime.now)


@dataclass
class Chunk:
    """
    Document를 분할한 청크 단위.
    Index Agent가 생성 → Eval/Optimize Agent가 소비.
    """
    chunk_id: str
    doc_id: str
    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    embedding: Optional[list[float]] = None    # dense 벡터
    sparse_vector: Optional[dict] = None       # BM25 sparse
    metadata: dict = field(default_factory=dict)


@dataclass
class Probe:
    """
    진단용 질문 단위.
    핵심: 청크에서 뽑지 않고 외부에서 가져와야 함.
    """
    probe_id: str
    question: str
    source: str                          # "user_log" | "taxonomy" | "llm_generated"
    expected_difficulty: str = "medium"
    answer_exists: Optional[bool] = None
    ground_truth: Optional[str] = None
    # ── Eval Agent 확장 필드 (설계 문서 'Probe 스키마' / 옵셔널·하위호환) ──
    gold_chunk_ids: list[str] = field(default_factory=list)  # [캐시] Recall@k 계산용 정답 청크.
    # gold_spans 기준으로 재계산되는 캐시 — 재청킹(Optimize→Index) 후에는 무효화될 수 있음.
    qtype: Optional[str] = None          # 멀티홉 유형: "bridge" | "comparison" | "aggregation" | None
    metadata: dict = field(default_factory=dict)   # 생성 출처(gen_method/persona/style/length 등)
    gold_doc_id: Optional[str] = None              # 정답이 있는 원본 문서 ID (재청킹에도 안 깨지는 기준)
    gold_char_span: Optional[tuple[int, int]] = None  # 원문 내 정답 위치 (start, end). 단일 대표 span.
    gold_spans: list[dict] = field(default_factory=list)
    # gold_spans 항목 예: {"doc_id": str, "start": int, "end": int}. 멀티홉 probe는 여러 개 가짐.


@dataclass
class Finding:
    """
    진단 결과 단위.
    Eval Agent가 생성 → Optimize Agent가 소비.
    """
    finding_id: str
    # "gap" | "retrieval_failure" | "generation_failure" (agents/eval/diagnose.py::_group_of 가
    # type 과 label 접두어로 A/B/C/D 그룹(처방 순서 정렬용)을 파생: type=="gap"→D,
    # label이 "retrieval_"로 시작→A, "generation_"으로 시작→B, 그 외(컨텍스트 구조 라벨)→C).
    # "contradiction"/"duplicate"/"staleness"는 예약(미사용) — metrics_ragas.py 가 aspect.contradiction
    # 을 계산은 해두지만 diagnose.py 는 아직 소비하지 않는다("나중에 개발" 주석 참고).
    type: str
    severity: str  # "critical" | "warning" | "info"
    description: str
    label: Optional[str] = None          # 세분화 진단명(처방 파일 라벨). Optimize가 label→처방 매핑에 사용
    confirmed: bool = True               # 확정 판정 여부. False=예비(더 높은 진단 모드에서 확정 필요)
    affected_chunks: list[str] = field(default_factory=list)
    affected_probes: list[str] = field(default_factory=list)
    prescription: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class DiagnosticReport:
    """
    Eval Agent의 최종 진단 리포트.
    """
    report_id: str
    findings: list[Finding] = field(default_factory=list)
    findings_summary: dict = field(default_factory=dict)  # 확정/예비·라벨 집계(진단 모드 포함). Optimize 소비용
    ragas_scores: dict = field(default_factory=dict)
    oracle_accuracy: Optional[float] = None
    overall_score: Optional[float] = None
    pass_threshold: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    iteration: int = 1
