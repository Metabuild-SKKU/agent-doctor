"""
agents/eval/types.py
Eval Agent 내부 자료구조 · 상수 정의

설계 문서 STEP 1~5의 파이프라인을 흐르는 "한 probe당 중간/최종 결과"를 담는
내부 전용 dataclass(`EvalRecord`)와, 각 단계에서 공통으로 쓰는 상수를 모아둔다.

주의:
  - `core/schema.py`의 공유 스키마(Probe, Finding, DiagnosticReport)는 팀 공용이므로
    평가 과정에서만 필요한 중간 상태(retrieved_context, generated_answer, branch 등)는
    이 `EvalRecord`에 담아 두고 공유 스키마를 오염시키지 않는다.
    (설계 문서 '토의 주제 1. Probe 스키마 확장'과 연결)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.schema import Finding, Probe


# ── 상수 ──────────────────────────────────────────────────────────

# STEP3-1 규칙 지표 임계값
F1_PASS_THRESHOLD = 0.5        # token F1 통과 기준 (설계상 "F1 통과" 수치화)
DEFAULT_TOP_K = 5              # index_config.top_k 미지정 시 검색 개수

# STEP3-2 RAGAS 지표 임계값 (설계 STEP4 표 기준, 낮으면 Finding 생성)
RAGAS_FAITHFULNESS_MIN = 0.7
RAGAS_CONTEXT_PRECISION_MIN = 0.7
RAGAS_CONTEXT_RECALL_MIN = 0.7
RAGAS_RESPONSE_RELEVANCY_MIN = 0.7

# STEP5 최종 판정
# graph.route_after_eval() 이 이 값 기반 pass_threshold 로 Serve/Optimize 를 분기한다.
PASS_SCORE_THRESHOLD = 0.8

# 가중 평균 가중치 (합=1.0). 없는 지표는 build_report 에서 재정규화.
RAGAS_WEIGHTS = {
    "faithfulness": 0.30,
    "context_precision": 0.25,
    "context_recall": 0.25,
    "response_relevancy": 0.20,
}

# STEP1 Probe 생성 믹스 비율 (설계 §STEP1, 합=1.0)
RAGAS_MIX_RATIO = 0.75
DATAMORGANA_MIX_RATIO = 0.20
NO_ANSWER_MIX_RATIO = 0.05

# STEP1 지식그래프 엣지 임계값 (이 이상이면 두 청크를 연결)
KG_ENTITY_OVERLAP_MIN = 0.2     # 키워드/entity Jaccard 유사도
KG_EMBEDDING_SIM_MIN = 0.5      # chunk.embedding 코사인 유사도

# STEP1 시나리오 샘플링 후보 (RAGAS Scenario 파라미터)
PERSONAS = ["신입사원", "실무 담당자"]
QUERY_STYLES = ["web_search", "conversational", "imperative"]
QUERY_LENGTHS = ["short", "medium", "long"]
EVOL_DIRECTIONS = ["depth", "breadth", "reasoning", "conditioning"]

# STEP1 RAGAS 4분면 배분 비율 (합=1.0). 멀티홉 엣지가 없으면 multi_* 몫은
# single_* 쪽으로 접힌다(agents/eval/probe_gen.py::_allocate_ragas_quadrants).
RAGAS_QUADRANT_WEIGHTS = {
    "single_specific": 0.4,
    "single_abstract": 0.3,
    "multi_specific": 0.2,
    "multi_abstract": 0.1,
}
MULTIHOP_SUBTYPES = ["bridge", "comparison", "aggregation"]


class Branch:
    """
    STEP3-1 규칙 지표(recall@k / F1 / oracle_F1)로 결정되는 진단 브랜치.
    설계 문서 STEP3-1 표의 각 행에 대응한다. STEP3-2(LLM 진단)와 STEP4(원인 판정)의
    실행 경로를 이 값으로 분기한다.
    """
    SUCCESS = "success"                                # 성공: 스킵
    RETRIEVAL_FAIL = "retrieval_fail"                  # 검색 실패 (oracle 통과)
    RETRIEVAL_GEN_FAIL = "retrieval_fail_gen_fail"     # 검색 실패 + 생성 실패
    RETRIEVAL_PARTIAL = "retrieval_partial"            # 검색 부분 실패 (oracle 통과)
    RETRIEVAL_PARTIAL_GEN_FAIL = "retrieval_partial_gen_fail"  # 부분 실패 + 생성 실패
    AMBIGUOUS_CONTEXT = "ambiguous_context"            # 애매함, 컨텍스트 원인
    AMBIGUOUS_GEN = "ambiguous_gen"                    # 애매함, 생성 원인
    NO_ANSWER_OK = "no_answer_ok"                      # 무응답 정답: 올바른 기권
    NO_ANSWER_VIOLATION = "no_answer_violation"        # 무응답인데 답을 지어냄


# ── 내부 결과 자료구조 ────────────────────────────────────────────

@dataclass
class EvalRecord:
    """
    probe 1개에 대한 평가 파이프라인 전 과정의 중간·최종 결과.

    STEP2  → retrieved_*, generated_answer, oracle_answer
    STEP3-1 → recall_at_k, f1_score, oracle_f1, branch
    STEP3-2 → ragas, oracle_ragas, aspect
    STEP4  → findings
    """
    probe: Probe

    # STEP2: 검색 + 생성
    retrieved: list[dict] = field(default_factory=list)          # search() 원본 결과
    retrieved_context: list[str] = field(default_factory=list)   # 청크 텍스트만
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    generated_answer: str = ""
    oracle_answer: Optional[str] = None                          # gold context로 생성한 답
    oracle_context: list[str] = field(default_factory=list)      # gold context 텍스트 (oracle 트랙 RAGAS용)

    # STEP3-1: 규칙 지표
    recall_at_k: float = 0.0
    f1_score: float = 0.0
    oracle_f1: float = 0.0
    branch: str = Branch.SUCCESS

    # STEP3-2: LLM(RAGAS) 지표 — 활성화된 경우에만 채워짐
    ragas: dict = field(default_factory=dict)          # 실제 트랙
    oracle_ragas: dict = field(default_factory=dict)   # 오라클 트랙
    aspect: dict = field(default_factory=dict)         # AspectCritic 결과

    # STEP4: 원인 판정
    findings: list[Finding] = field(default_factory=list)
