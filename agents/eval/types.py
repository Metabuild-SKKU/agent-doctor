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
# lexical 단독 게이트 문턱 — RAGAS 가 없는 모드(DEEP 미만)나 판정기 실패 때만 쓰인다.
# RAGAS 가 측정된 실행에서는 lexical 을 단독 게이트로 쓰지 않는다(아래 혼합 점수 참고).
F1_PASS_THRESHOLD = 0.5
# answer_match 최댓값. degrade 된(불안정한) 심판이 정답을 오답으로 끌어내리는 걸 막는
# 면제선으로만 쓴다(_degraded_near_miss). 통과 문턱(0.5)이 아니라 이 값을 쓰는 이유:
# 0.5~0.9 대의 애매한 근접 오답은 심판이 죽었을 때 보수적으로 강등해야 하고, 최댓값만
# 면제해야 안전망이 헐거워지지 않는다.
#
# 주의 — 이 값이 곧 문자열 완전일치는 아니다. gold 가 정규화 후 10자 이하면 answer_match
# 가 char-recall 경로를 타므로(metrics_basic.answer_match) gold 를 포함하기만 하면 1.0 이
# 나온다: gold '사망' ↔ 답 '사망하지 않았다' = 1.0. 즉 짧은 gold 의 부정문 근접 오답은
# 이 선을 그냥 통과한다. 그래서 면제는 이 값 단독이 아니라 faithfulness 하한과 **함께**
# 걸어야 한다 — 위 예에서 컨텍스트가 '사망'을 말하면 '사망하지 않았다'는 근거에 안 붙어
# faithfulness 가 떨어지고, 그 축이 강등을 살려낸다. 면제선을 넓히거나 faithfulness 조건을
# 떼면 _degraded_near_miss 가 원래 잡으려던 부정문 오답이 그대로 새어 나간다.
F1_EXACT_MATCH = 1.0
DEFAULT_TOP_K = 5              # index_config.top_k 미지정 시 검색 개수

# STEP3-2 RAGAS 지표 임계값 (설계 STEP4 표 기준, 낮으면 Finding 생성)
RAGAS_FAITHFULNESS_MIN = 0.7
RAGAS_CONTEXT_PRECISION_MIN = 0.7
# 검색된 gold 청크 안에서 실제 근거가 차지하는 최소 비율. 이보다 낮으면 청크가 근거보다 훨씬
# 커서 무관한 내용을 함께 싣고 있다고 본다(chunking_underchunking). 숫자 튜닝 필요.
EVIDENCE_DENSITY_MIN = 0.2
RAGAS_CONTEXT_RECALL_MIN = 0.7
RAGAS_RESPONSE_RELEVANCY_MIN = 0.7

# ── 정답 판정 혼합 점수 (lexical × RAGAS 의미, tier3) ──────────────
# lexical(char-F1)은 표면형만 보고 의미(RAGAS)는 판정기 편차가 있어, 어느 하나를 단독
# 임계값으로 쓰면 '맞은 답이 문턱 하나에 걸려 실패'가 계속 생긴다(실측: f1=0.49 로 실패한
# 정답 → C그룹 오진). 그래서 두 축을 가중합한 점수 하나로 판정한다.
#   answer_score = (1-ANSWER_SEMANTIC_WEIGHT)·lexical + ANSWER_SEMANTIC_WEIGHT·semantic
#   semantic     = max(answer_correctness, gold 커버리지(근거 있을 때만))
# 의미축에 커버리지를 함께 넣는 이유: answer_correctness(=TP/(TP+0.5(FP+FN)))는 군더더기(FP)를
# 분모에 넣어 '근거를 덧붙여 길어진 정답'을 깎는데, 커버리지(TP/(TP+FN))는 FP 를 빼고
# '정답 요소를 다 담았나'만 묻기 때문이다(누락·모순은 FN 이라 근접 오답은 여전히 떨어진다).
ANSWER_SEMANTIC_WEIGHT = 0.6
ANSWER_PASS_THRESHOLD = 0.5
# 커버리지를 의미축으로 인정하는 최소치 — '정답 요소를 (거의) 다 담았다'는 구제 경로이지
# 부분 점수 채널이 아니다. 이 문턱이 없으면 gold 절반만 담은 답변이 커버리지 0.5 를 의미축으로
# 들고 통과해, generation_partial_answer(부분 답변) 진단이 조용히 사라진다.
# 미달이면 의미축은 answer_correctness 가 맡는다(그쪽은 FN 을 분모에 넣어 누락을 감점한다).
GOLD_COVERAGE_MIN = 0.8
# 의미축 바닥선 — 이 아래면 lexical 이 높아도 실패. 표면형만 비슷한 근접 오답(부정문·3월↔3일)을
# 거르던 기존 강등(ANSWER_CORRECTNESS_MIN)의 역할을 혼합 점수 체계에서 이어받는다.
ANSWER_SEMANTIC_FLOOR = 0.3
# (레거시) 단독 강등 임계값 — 혼합 점수 도입 후 게이트에서는 쓰지 않는다. 리포트·문서가
# '근접 오답 경계'를 말할 때 참조하는 값으로만 남긴다.
ANSWER_CORRECTNESS_MIN = 0.5

# ── context 구조 라벨(too_long_context / lost_in_the_middle)의 저비용 발동 문턱 ──
# context 총 길이(문자)가 이보다 길면 길이·배치를 의심한다. 임의값 — 실측 후 튜닝 필요.
CONTEXT_CHARS_MAX = 6000
# 검색 결과에서 gold 청크의 상대 위치가 이 구간(경계 포함)이면 '중간에 묻혔다'로 본다.
# 경계 포함이라 top_k=5 의 2번째 순위(위치 0.25)도 중간으로 잡힌다 — 튜닝 대상.
CONTEXT_MIDDLE_BAND = (0.25, 0.75)

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

# STEP1 지식그래프 엣지 임계값
# 절대 임계값(아래 두 값)은 "관련 있음"이 아니라 "코퍼스에서 명백히 무관한 쌍"을 끊는
# 하한 가드로만 쓴다 — 실측(HR 코퍼스 41청크, BGE-M3): 무관한 쌍끼리도 cos 중앙값이
# 0.46 이라 같은 도메인 문서는 통째로 0.4~0.5대에 깔린다. 절대 임계값 하나로는 "관련
# 있음"을 못 가른다. 그래서 관련도 판정은 KG_TOP_K_NEIGHBORS(각 청크 기준 상대 순위)로
# 하고, 이 값들은 top-k 가 억지로 끌어온 먼 이웃을 끊는 바닥선 역할만 한다.
KG_ENTITY_OVERLAP_MIN = 0.2     # 키워드/entity Jaccard 유사도(한국어 토크나이저 한계로 보조 신호)
KG_EMBEDDING_SIM_MIN = 0.5      # chunk.embedding 코사인 유사도(무관 쌍 하한 가드)

# 각 청크를 코사인 상위 이 개수의 이웃하고만 연결한다(멀티홉 후보). 절대 임계값 대신
# 상대 순위로 판정해, 코퍼스 전체가 높/낮게 깔려도 자동 보정된다. 실측(41청크): k=2 면
# 후보 56쌍·cos 하한 0.518 로, 절대 임계값 OR(261쌍·78%가 cos<0.6 노이즈)보다 억지
# 멀티홉이 대폭 줄었다. 예산(멀티홉 30%)을 채우기 충분한 최소값으로 2 를 기본값으로 둔다.
KG_TOP_K_NEIGHBORS = 2

# STEP4 토픽클러스터 신호 임계값 (retrieval_semantic_mismatch 처방 라우팅용)
# 실패한 gold 청크들이 임베딩 공간에서 얼마나 뭉쳤는지를 코퍼스 baseline 대비 비율로 본다.
# 절대 코사인은 코퍼스마다 깔린 수준이 달라(위 KG 주석: 무관 쌍도 cos 0.46) 쓸 수 없어,
# "실패 gold 응집도 / 코퍼스 전체 평균 응집도" 비율로 상대 판정한다(KG_TOP_K 와 같은 철학).
#   ratio >= CONCENTRATED → "concentrated" (특정 도메인만 약함 → 도메인특화 임베딩)
#   ratio <= SPREAD       → "spread"       (전 주제 흩어져 실패 → 임베딩 모델 자체 약함)
#   그 사이               → "none"         (재봤으나 주제 응집 안 보임 → 청킹 조정)
# "못 잰" 경우는 이 임계값과 무관하게 "unmeasured" 로 따로 나간다(topic_cluster.py 참고).
#
# 경계는 고정 상수가 아니라 실패 gold 수 N 에 따라 동적으로 잡는다(캘리브레이션 결과).
#   none 대 = 1.0 ± k·C/sqrt(N)  →  topic_cluster.dynamic_bounds(N) 가 계산한다.
#     ratio >= 1.0 + k·C/sqrt(N) → "concentrated"
#     ratio <= 1.0 - k·C/sqrt(N) → "spread"
#     그 사이                    → "none"
#
# 캘리브레이션 근거(tools/topic_cluster_calibration, 세금가이드 코퍼스 597청크, LLM 0):
#   "주제 신호 없음"(코퍼스 전체에서 실패 gold 를 무작위 추출)의 null ratio 분포는
#   중앙값이 N 무관하게 1.00 에 붙고(baseline 편향 없음), stdev 는 N 이 커질수록
#   규칙적으로 줄어든다 — stdev·sqrt(N) 이 0.10~0.13 로 거의 일정(=c/sqrt(N) 법칙):
#     N=5→stdev0.058  N=10→0.040  N=20→0.026  N=40→0.018  N=100→0.010
#   그래서 고정 폭(옛 1.1~1.3)은 표본 적은 구간에서 null 분산(예: N=10 이면 ±0.06)보다
#   좁아 신호 없는 회차도 spread/concentrated 로 튀고(옛 경계로 null 의 99% 가 none 밖),
#   표본 많은 구간에선 반대로 과하게 넓었다. 폭을 sqrt(N) 로 좁혀 오분류율을 표본 수와
#   무관하게 일정하게 맞춘다(TODO 가 제시한 (b) 동적 조절).
# k=2.33: null 분포를 none 으로 흡수하는 신뢰구간(편측 ~99%). 대조군(같은 주제 인접
#   블록 = '뭉침' 신호)과 대비 시 null 누출(=비싼 재색인 오발) 1~4%, 신호 검출 91~97%
#   로 균형점이었다(k=1.645 는 누출 7~11%, k=2.5 는 누출은 더 주나 검출 이득 없음).
TOPIC_CLUSTER_NULL_C = 0.118        # null stdev·sqrt(N) 의 실측 상수(c/sqrt(N) 법칙)
TOPIC_CLUSTER_BOUNDARY_K = 2.33     # 경계 폭 계수(1.0 ± k·C/sqrt(N)); 클수록 none 대가 넓다
# 평균 응집도를 잴 때 뽑는 청크 표본 수 상한(전량 O(n^2) 회피). baseline·실패 gold 공용.
# 캘리브레이션도 이 상한 안에서 stdev 를 쟀으므로 C 는 이 값과 짝이다(상한 바뀌면 재보정).
TOPIC_CLUSTER_BASELINE_SAMPLE = 100

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
#                        <  tier3 LLM·RAGAS
# 라벨은 '판별에 필요한 가장 비싼 자원'이 곧 그 라벨의 confirm tier 다(각 측정이 metrics_* 에서 self-gate).
#   mode >= tier 이고 그 확정 측정이 실제 발동 → 확정(confirmed=True), 아니면 예비(confirmed=False).
# 생성 원인(B)은 전부 RAGAS(=DEEP) 의존이라, DEEP 미만에선 하나의 예비 'generation_failure' 로 롤업.
# STEP3-2 RAGAS 자체도 DEEP 이상에서만 실행한다(metrics_ragas._faith 등 RAGAS 측정의 DEEP 게이트).
# tier4(파이프라인 재실행/ablation)는 없다 — Optimize 의 config 재실행과 중복이라 그쪽으로 넘겼고,
# DEEP 이 가장 깊은 모드다. context 원인(C)은 실측 신호로 DEEP 에서 확정된다.
class Mode:
    FAST = 1       # 규칙·기존지표만 (추가 쿼리·LLM 없음) — 나열형(enumeration)만 확정
    STANDARD = 2   # + 추가 검색 쿼리(top-N·BM25·코퍼스)   — 검색 원인 대부분 확정
    DEEP = 3       # + LLM(RAGAS/AspectCritic)         — 생성·context 원인 확정


DEFAULT_MODE = Mode.FAST   # 미지정 시 가장 싼 모드

# "full"/"4" 는 DEEP 으로 접는다 — 웹 UI 의 depth 선택지(agents/serve/web_api.py)와 기존
# EVAL_MODE 설정이 쓰는 문자열이라, 지우면 조용히 DEFAULT_MODE(fast)로 강등된다.
_MODE_ALIASES = {
    "fast": Mode.FAST, "standard": Mode.STANDARD, "deep": Mode.DEEP, "full": Mode.DEEP,
    "1": Mode.FAST, "2": Mode.STANDARD, "3": Mode.DEEP, "4": Mode.DEEP,
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
    # 검색 구현의 실행 상태. reranker가 실제로 호출·성공했는지 Eval 리포트에
    # 집계해 Optimize가 "미실행"을 "효과 없음"으로 오판하지 않게 한다.
    retrieval_details: dict = field(default_factory=dict)
    generated_answer: str = ""
    oracle_answer: Optional[str] = None                          # gold context로 생성한 답
    oracle_context: list[str] = field(default_factory=list)      # gold context 텍스트 (oracle 트랙 RAGAS용)

    # STEP3-1: 규칙 지표 (diagnose 가 진입 시 계산·저장)
    recall_at_k: float = 0.0
    # recall_at_k 를 어느 기준으로 쟀나: "span"(gold 원문 좌표 커버리지) | "chunk"(gold 청크 id 포함률).
    # 로그가 둘을 같은 이름으로 찍어서, gold 청크가 검색 목록에 있는데 recall=0 인 줄이
    # 모순처럼 보였다(span 은 좌표를 빈틈없이 덮어야 1점이라 청크 하나만으론 0 이 될 수 있다).
    recall_basis: str = "chunk"
    # 컨텍스트에 넣은 글자 중 골드 근거의 비율(근거 밀도). recall 의 짝 — recall 은 많이
    # 퍼올수록 무조건 오르는데 그 비용을 재는 축이 없었다. 좌표로 못 재면 None(미측정).
    # RAGAS context_precision 과 다른 축이다: 그쪽은 청크 '순서' 품질이라 top_k 를 늘리거나
    # 청크가 문서 전체가 돼도 안 떨어진다. 자세한 대비는 span_precision_at_k 참고.
    span_precision: Optional[float] = None
    f1_score: float = 0.0
    oracle_f1: float = 0.0
    raw_f1_score: float = 0.0
    raw_oracle_f1: float = 0.0
    best_gold_answer_f1: float = 0.0
    best_oracle_gold_answer_f1: float = 0.0
    best_gold_answer: Optional[str] = None
    best_oracle_gold_answer: Optional[str] = None
    gold_answer_variant_count: int = 0
    exact_match: bool = False        # KorQuAD 공식 EM — 리포트 관측용(게이트·overall_score 미반영)

    # STEP3-2: LLM(RAGAS) 지표 — diagnose 가 lazy 로 채움
    ragas: dict = field(default_factory=dict)          # 실제 트랙
    oracle_ragas: dict = field(default_factory=dict)   # 오라클 트랙
    aspect: dict = field(default_factory=dict)         # 실행 단위 LLM 판정(abstention / reasoning_mode) memoize
    # RAGAS lazy 계산 여부(트랙별). 빈 결과({})여도 '시도함'으로 남겨 같은 트랙 재-LLM호출을 막는다.
    ragas_done: bool = False
    oracle_ragas_done: bool = False

    # STEP4: 원인 판정
    # 성공/실패는 별도 필드를 두지 않는다 — findings 가 비었으면 성공(판정 불가 probe 도 findings 가
    # 없으므로 통과로 집계된다). scoring.reliability_score / report 가 이 규약으로 읽는다.
    findings: list[Finding] = field(default_factory=list)

    # 검색축 신뢰도 override — diagnose 가 '라벨 골드는 못 집었지만 답이 정답이고 검색 근거에
    # 붙었고(grounded) 골드도 유효(oracle 통과)'라고 판정하면(다른 유효 근거로 검색이 검증된
    # label-recall miss), recall 대신 이 값(=faithfulness)을 검색축으로 쓴다. 미설정(None)이면
    # scoring 이 recall_at_k 를 그대로 쓴다. pass/fail(findings)과 reliability 를 같은 판정으로
    # 묶어 재청킹 recall 스윙에 둘이 따로 놀지 않게 한다.
    retrieval_axis: Optional[float] = None

    # 진단 신호 memoize 뷰: agent 가 state.diagnosis_cache[probe_id] 를 주입 → 쓰기가 state 로 전파.
    # 비싼 판별 신호(_signal)가 여기 캐시돼 재진단 시 재사용된다.
    signals: dict = field(default_factory=dict)

    # ragas/oracle_ragas dict 의 answer_correctness 를 속성으로 노출(diagnose 정답 강등 판정용).
    # _compute_ragas_real/_oracle 이 채우므로(DEEP+), 그 미만·미측정이면 빈 dict → None.
    # (oracle 쪽은 실패 probe 에서만 채워진다 — 성공 probe 는 진단을 건너뛰므로 None.)
    @property
    def ragas_answer_correctness(self) -> Optional[float]:
        """answer_correctness(답변↔gold 비교) — 실제 트랙. 미측정·DEEP 미만이면 None."""
        v = self.ragas.get("answer_correctness")
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    @property
    def oracle_ragas_answer_correctness(self) -> Optional[float]:
        """answer_correctness(답변↔gold 비교) — 오라클 트랙. 미측정·DEEP 미만이면 None."""
        v = self.oracle_ragas.get("answer_correctness")
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    @property
    def gold_coverage(self) -> Optional[float]:
        """gold 요소 커버리지 TP/(TP+FN) — 실제 트랙. 카운트 미측정(degraded·DEEP 미만)이면 None.

        answer_correctness(TP/(TP+0.5(FP+FN)))와 달리 군더더기(FP)를 분모에서 뺀 recall 축이다.
        '정답 요소를 다 담았나'만 묻기 때문에, 근거를 덧붙여 길어진 정답 답변이 lexical F1·
        answer_correctness 양쪽에서 깎이는 구조적 저평가를 가려낼 수 있다."""
        return _gold_coverage(self.ragas)

    @property
    def oracle_gold_coverage(self) -> Optional[float]:
        """gold 요소 커버리지 TP/(TP+FN) — 오라클 트랙. 미측정이면 None."""
        return _gold_coverage(self.oracle_ragas)

    @property
    def answer_semantic(self) -> Optional[float]:
        """정답 판정 혼합 점수의 '의미축' — 실제 트랙. RAGAS 미측정이면 None(=lexical 단독 판정)."""
        return _semantic_answer(self.ragas)

    @property
    def oracle_answer_semantic(self) -> Optional[float]:
        """정답 판정 혼합 점수의 '의미축' — 오라클 트랙. 미측정이면 None."""
        return _semantic_answer(self.oracle_ragas)

    @property
    def answer_score(self) -> float:
        """정답 판정 점수 — 실제 트랙. lexical(f1_score)과 의미축의 가중합(의미 미측정이면 lexical)."""
        return blend_answer_score(self.f1_score, self.answer_semantic)

    @property
    def oracle_answer_score(self) -> float:
        """정답 판정 점수 — 오라클 트랙."""
        return blend_answer_score(self.oracle_f1, self.oracle_answer_semantic)


def _gold_coverage(ragas: dict) -> Optional[float]:
    """RAGAS 트랙 dict 의 TP/FN 카운트에서 gold 커버리지를 계산. 카운트 없으면 None."""
    if "answer_correctness_fn" not in ragas:
        return None
    tp, fn = ragas["answer_correctness_tp"], ragas["answer_correctness_fn"]
    denom = tp + fn
    return tp / denom if denom > 0 else None


def _float_or_none(value) -> Optional[float]:
    """dict 에서 읽은 값이 실수 지표인지 확인(bool·None·문자열 배제)."""
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _semantic_answer(ragas: dict) -> Optional[float]:
    """의미축 = max(answer_correctness, gold 커버리지) — 쓸 수 있는 게 없으면 None(→ lexical 단독).

    두 성분 모두 '쓸 수 있을 때만' 넣는다. 게이트가 이 값으로 승격까지 하므로, 신뢰할 수
    없는 성분을 넣으면 오답이 통과한다:
      · answer_correctness: factual 분류(TP/FP/FN)가 실패해 임베딩 유사도 단독으로 계산된
        경우(answer_correctness_degraded)는 제외한다. 한국어에서 같은 주제 문장끼리는 사실이
        틀려도 코사인이 0.8 대로 나와, 유사도만으로 승격시키면 오답이 그대로 통과한다.
        (강등만 하던 시절엔 이 폴백이 안전했다 — metrics_ragas._answer_correctness 주석 참고.)
      · 커버리지: GOLD_COVERAGE_MIN 이상 + 답이 검색 context 에 근거(faithfulness)할 때만.
        문턱은 부분 답변이 커버리지를 부분 점수로 들고 통과하는 걸 막고, 근거 조건은 파라미터
        기억으로 맞힌 답(generation_parametric_overreliance 영역)을 의미축으로 올리지 않는다.
    """
    values: list[float] = []
    ac = _float_or_none(ragas.get("answer_correctness"))
    if ac is not None and not ragas.get("answer_correctness_degraded"):
        values.append(ac)
    coverage = _gold_coverage(ragas)
    faith = _float_or_none(ragas.get("faithfulness"))
    if (coverage is not None and coverage >= GOLD_COVERAGE_MIN
            and faith is not None and faith >= RAGAS_FAITHFULNESS_MIN):
        values.append(coverage)
    return max(values) if values else None


def blend_answer_score(lexical: float, semantic: Optional[float]) -> float:
    """lexical(char-F1 계열)과 의미축을 ANSWER_SEMANTIC_WEIGHT 로 가중합. 의미 미측정이면 lexical.

    한 축이라도 단독 문턱으로 쓰면 '맞은 답이 문턱 하나에 걸려 실패'가 반복된다 —
    lexical 은 표현 차이·길이에, 의미축은 판정기 편차에 흔들리기 때문. 두 축을 섞으면
    한쪽의 흔들림이 다른 쪽에 흡수된다(판정: diagnose._answer_ok)."""
    lexical = max(0.0, min(1.0, lexical))
    if semantic is None:
        return lexical
    semantic = max(0.0, min(1.0, semantic))
    return (1 - ANSWER_SEMANTIC_WEIGHT) * lexical + ANSWER_SEMANTIC_WEIGHT * semantic
