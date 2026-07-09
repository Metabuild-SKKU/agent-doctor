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
    char_span: Optional[tuple[int, int]] = None  # (start, end) — 부모 Document.content 기준 위치.
    # 재청킹돼도 안 깨지는 기준. Eval이 gold_char_span과 겹치는 청크를 찾아 gold를 재판정하는 데 씀.
    token_count: Optional[int] = None            # 실제 임베딩 모델 토크나이저 기준 토큰 수
    parent_id: Optional[str] = None              # Small-to-Big(부모 섹션) 확장 대비. 현재는 미사용
    hash: Optional[str] = None                   # sha256(text) 앞부분. 중복 판별/증분 인덱싱용
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
    gold_doc_id: Optional[str] = None            # 정답이 있는 원본 문서 ID (청킹과 무관, 항상 유효)
    gold_char_span: Optional[tuple[int, int]] = None  # 원문 내 정답 위치 — Chunk.char_span과 동일 좌표계
    gold_chunk_ids: list[str] = field(default_factory=list)  # 생성 시점 캐시. 재청킹 후엔 신뢰 금지,
    # gold_char_span으로 매번 재계산할 것
    qtype: Optional[str] = None          # "bridge" | "comparison" | "aggregation" | None


@dataclass
class Finding:
    """
    진단 결과 단위.
    Eval Agent가 생성 → Optimize Agent가 소비.
    """
    finding_id: str
    type: str    # "gap" | "contradiction" | "duplicate" | "staleness" | "retrieval_failure" | "generation_failure"
    severity: str  # "critical" | "warning" | "info"
    description: str
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
    ragas_scores: dict = field(default_factory=dict)
    oracle_accuracy: Optional[float] = None
    overall_score: Optional[float] = None
    pass_threshold: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    iteration: int = 1
