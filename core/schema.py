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
    # ── Eval Agent 확장 필드 (설계 문서 'Probe 스키마' / 옵셔널·하위호환) ──
    gold_chunk_ids: list[str] = field(default_factory=list)  # [캐시] Recall@k 계산용 정답 청크.
    # gold_spans 기준으로 재계산되는 캐시 — 재청킹(Optimize→Index) 후에는 무효화될 수 있음.
    qtype: Optional[str] = None          # 멀티홉 유형: "bridge" | "comparison" | "aggregation" | None
    metadata: dict = field(default_factory=dict)   # 생성 출처(gen_method/persona/style/length 등)
    gold_doc_id: Optional[str] = None              # 정답이 있는 원본 문서 ID (재청킹에도 안 깨지는 기준)
    gold_char_span: Optional[tuple[int, int]] = None  # 원문 내 정답 위치 (start, end). 단일 대표 span.
    gold_spans: list[dict] = field(default_factory=list)
    # gold_spans 항목 예: {"doc_id": str, "start": int, "end": int}. 멀티홉 probe는 여러 개 가짐.


# Finding.metadata 의 신호가 "재려고 했으나 값을 못 냈다"를 말하는 sentinel.
#
# 여기 있는 이유: Eval 과 Optimize 는 서로 import 하지 않고 Finding.metadata 로만 대화한다.
# 그래서 '미측정'의 표현을 한쪽이 문자열로, 다른 쪽이 None 으로 알고 있어도 타입 검사에
# 안 걸리고 조용히 어긋난다(실제로 어긋나 있었다 — topic_cluster 는 "unmeasured" 를
# 실어 보내는데 소비부는 None 만 미측정으로 취급했다). 계약이 사는 이 파일에 정본을 둔다.
#
# 의미 구분이 중요하다: '못 쟀다'(이 sentinel)는 '쟀는데 해당 없다'와 다르다. 전자는 근거가
# 없으니 신호를 무시하고 원래 순차 시도로 돌아가야 하고, 후자는 처방을 고르는 근거가 된다.
# 둘을 뭉개면 못 잰 회차가 오히려 더 단정적으로 처방을 확정한다.
UNMEASURED_SIGNAL = "unmeasured"


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
    # "contradiction"/"duplicate"/"staleness"는 예약(미사용).
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
    # 실패한 검증 질문의 원문 비교 데이터. EvalRecord는 실행 중에만 존재하므로
    # 리포트 페이지가 질문·기대 정답·실제 답변을 보여주려면 여기 보존해야 한다.
    failed_questions: list[dict] = field(default_factory=list)
    findings_summary: dict = field(default_factory=dict)  # 확정/예비·라벨 집계(진단 모드 포함). Optimize 소비용
    ragas_scores: dict = field(default_factory=dict)
    # Eval이 관측한 검색 runtime 실행 결과. 품질 지표와 섞지 않고 Optimize가
    # 처방 실행 유효성을 판단할 때 사용한다.
    runtime_summary: dict = field(default_factory=dict)
    oracle_accuracy: Optional[float] = None
    overall_score: Optional[float] = None
    composite_score: Optional[dict] = None  # 종합점수(0~100)+성분별 점수. agents/eval/scoring.py 산출
    pass_threshold: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    iteration: int = 1


@dataclass
class IndexSnapshot:
    """
    롤백용 인덱스 스냅샷.

    Index Agent만 생성·갱신하며, 현재 인덱스와 직전 인덱스까지 최대 두 개를
    AgentDoctorState에 보관한다.
    """
    cache_key: str
    chunks: list[Chunk] = field(default_factory=list)
    index_artifacts: dict = field(default_factory=dict)
    collection_name: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class EvalSnapshot:
    """
    롤백용 진단 결과 스냅샷.

    Eval Agent가 생성한 Probe·리포트·진단 신호를 함께 보관해, 동일한 파이프라인
    버전으로 돌아왔을 때 답변 생성과 RAGAS 호출 없이 결과를 복원한다.
    """
    cache_key: str
    index_key: str
    probes: list[Probe] = field(default_factory=list)
    report: Optional[DiagnosticReport] = None
    diagnosis_cache: dict = field(default_factory=dict)
    diagnosis_cache_version: str = ""
    created_at: datetime = field(default_factory=datetime.now)
