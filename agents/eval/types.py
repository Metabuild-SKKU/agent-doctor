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

import os
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
# 스케일 주의: overall_score 는 0~1 (RAGAS 가중 평균 또는 규칙 지표 평균). 설계 문서가 언급하는
# "100점 환산"과는 다른 스케일이다 — 100점 환산이 필요한 소비처(대시보드 등)가 생기면 그 표시
# 계층에서 *100 하고, 이 임계값·overall_score 자체는 0~1로 유지한다(report.py 참고).
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


# ── 진단 모드 (비용 tier) ─────────────────────────────────────────
# 사용자가 고르는 '진단 깊이'. 값 = 그 모드가 감당하는 최대 자원 tier.
#   자원 사다리(=tier):  tier1 규칙·기존지표만  <  tier2 추가 검색 쿼리(top-N 재검색·BM25·코퍼스)
#                        <  tier3 LLM·RAGAS  <  tier4 파이프라인 재실행(ablation)
# 라벨은 '판별에 필요한 가장 비싼 자원'이 곧 그 라벨의 confirm tier 다(각 신호가 signals.py 에서 self-gate).
#   mode >= tier 이고 그 확정 신호가 실제 발동 → 확정(confirmed=True), 아니면 예비(confirmed=False).
# 생성 원인(B)은 전부 RAGAS(=DEEP) 의존이라, DEEP 미만에선 하나의 예비 'generation_failure' 로 롤업.
# STEP3-2 RAGAS 자체도 DEEP 이상에서만 실행한다(signals._faith 등 RAGAS 신호의 DEEP 게이트).
class Mode:
    FAST = 1       # 규칙·기존지표만 (추가 쿼리·LLM 없음) — 나열형(enumeration)만 확정
    STANDARD = 2   # + 추가 검색 쿼리(top-N·BM25·코퍼스)   — 검색 원인 대부분 확정
    DEEP = 3       # + LLM(RAGAS/AspectCritic)         — 생성 원인 진단
    FULL = 4       # + 파이프라인 재실행(ablation)      — context 원인 확정


DEFAULT_MODE = Mode.FAST   # 미지정 시 가장 싼 모드

_MODE_ALIASES = {
    "fast": Mode.FAST, "standard": Mode.STANDARD, "deep": Mode.DEEP, "full": Mode.FULL,
    "1": Mode.FAST, "2": Mode.STANDARD, "3": Mode.DEEP, "4": Mode.FULL,
}


def resolve_mode() -> int:
    """진단 모드 결정. 환경변수 EVAL_MODE(fast|standard|deep|full 또는 1~4), 없으면 DEFAULT_MODE."""
    raw = os.getenv("EVAL_MODE", "").strip().lower()
    return _MODE_ALIASES.get(raw, DEFAULT_MODE)


# ── Probe 소스 (STEP1) ────────────────────────────────────────────
# 실사용 질문(user_log) vs 지식그래프 기반 자동 생성(auto) 중 무엇으로 진단할지 강제하는 스위치.
#   auto     : 항상 자동 생성(state.user_questions 무시). GT·gold 포함 → recall/F1/RAGAS 전량 평가.
#   user_log : 항상 user_questions 사용(질문 없으면 auto 폴백). GT 없음 → reference-free RAGAS만.
#   made     : 이미 만들어 둔 Probe(eval_probes.json)를 코퍼스 버전과 무관하게 그대로 재사용
#              (파일 없음/비었으면 auto 로 폴백해 생성 후 저장). LLM 재호출 없이 고정 테스트셋으로 진단.
#   미지정   : 자동 판별 — questions 있으면 user_log, 없으면 auto (기존 동작, 단 버전 일치 시 캐시 재사용).
PROBE_SOURCE_AUTO = "auto"
PROBE_SOURCE_USER_LOG = "user_log"
PROBE_SOURCE_MADE = "made"
#   taxonomy : 외부 사람작성 QA 데이터셋(KorQuAD 등)을 gold 포함 Probe 로 로드.
#              qa 는 EVAL_TAXONOMY_QA 에서, corpus(좌표 조회용)는 state.source_url 에서
#              가져와(=Ingest 와 동일 소스) gold_spans 를 실어 재청킹 후 resync 로 확정한다.
PROBE_SOURCE_TAXONOMY = "taxonomy"


def resolve_llm_concurrency() -> int:
    """LLM 호출 동시 실행 수. 환경변수 EVAL_LLM_CONCURRENCY(기본 4, 최소 1).
    1 이면 완전 순차(병렬화 이전 동작). Gemini 무료 티어처럼 분당 한도(RPM)가
    낮은 provider 는 2~3 권장 — 429 재시도 로그가 잦으면 낮출 것."""
    try:
        return max(1, int(os.getenv("EVAL_LLM_CONCURRENCY", "4")))
    except (TypeError, ValueError):
        return 4


def resolve_probe_source() -> str:
    """Probe 소스 스위치. EVAL_PROBE_SOURCE(auto|user_log|made|taxonomy), 미지정/오타면 "" (자동 판별)."""
    raw = os.getenv("EVAL_PROBE_SOURCE", "").strip().lower()
    valid = (PROBE_SOURCE_AUTO, PROBE_SOURCE_USER_LOG, PROBE_SOURCE_MADE, PROBE_SOURCE_TAXONOMY)
    return raw if raw in valid else ""


def taxonomy_qa_path() -> str:
    """taxonomy 소스 qa 파일 경로(EVAL_TAXONOMY_QA, 기본 data/qa_pairs.jsonl)."""
    return os.getenv("EVAL_TAXONOMY_QA", "data/qa_pairs.jsonl")


def _pos_int_env(name: str) -> Optional[int]:
    """양의 정수 환경변수 → int, 0/비정수/미설정 → None(=제한 없음)."""
    raw = os.getenv(name, "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else None


def korquad_max_docs() -> Optional[int]:
    """KORQUAD_MAX_DOCS — 앞 N개 문서만(None=전체). Ingest·Eval 이 같은 값을 봐야 corpus/qa
    문서 집합이 정합하므로 파싱을 여기 한 곳으로 모은다(양쪽 복붙 방지)."""
    return _pos_int_env("KORQUAD_MAX_DOCS")


def korquad_qa_limit() -> Optional[int]:
    """KORQUAD_QA_LIMIT — qa 개수 상한(None=전체). Eval 전용."""
    return _pos_int_env("KORQUAD_QA_LIMIT")


def llm_eval_enabled() -> bool:
    """STEP3-2 RAGAS(LLM-as-Judge) 진단 활성화 여부. 기본 꺼짐(EVAL_ENABLE_LLM=1/true/yes/on).
    실제 실행은 signals 의 RAGAS 신호(_faith 등)가 `EVAL_MODE≥deep` 게이트와 AND 로 정한다."""
    return os.getenv("EVAL_ENABLE_LLM", "").strip().lower() in ("1", "true", "yes", "on")


# ── 내부 결과 자료구조 ────────────────────────────────────────────

@dataclass
class EvalRecord:
    """
    probe 1개에 대한 평가 파이프라인 전 과정의 중간·최종 결과.

    STEP2  → retrieved_*, generated_answer, oracle_answer
    STEP3-1 → recall_at_k, f1_score, oracle_f1
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

    # STEP3-1: 규칙 지표 (diagnose 가 진입 시 계산·저장)
    recall_at_k: float = 0.0
    f1_score: float = 0.0
    oracle_f1: float = 0.0

    # STEP3-2: LLM(RAGAS) 지표 — diagnose 가 lazy 로 채움
    ragas: dict = field(default_factory=dict)          # 실제 트랙
    oracle_ragas: dict = field(default_factory=dict)   # 오라클 트랙
    aspect: dict = field(default_factory=dict)         # AspectCritic 결과 — generation_contradiction(주석처리) 용 예약, 현 라이브 미사용
    # RAGAS lazy 계산 여부(트랙별). 빈 결과({})여도 '시도함'으로 남겨 같은 트랙 재-LLM호출을 막는다.
    ragas_done: bool = False
    oracle_ragas_done: bool = False

    # STEP4: 원인 판정
    findings: list[Finding] = field(default_factory=list)

    # 진단 신호 memoize 뷰: agent 가 state.diagnosis_cache[probe_id] 를 주입 → 쓰기가 state 로 전파.
    # 비싼 판별 신호(_signal)가 여기 캐시돼 재진단 시 재사용된다.
    signals: dict = field(default_factory=dict)
