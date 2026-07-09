"""
agents/eval/diagnose.py
STEP4: 원인 판정 (Finding 생성)

구조 원칙:
  1. 라벨마다 판정 함수 1개: 각 함수는 자기 라벨의 '판별 신호(원인)'를 
     검사해 맞으면 Finding, 아니면 None 을 돌려준다.
  2. 브랜치마다 해당 함수들을 호출: 각 브랜치(_dx_*)가 설계의 판정 순서표대로 라벨
     함수들을 부른다. '하나만 고르는' 부분은 _pick(원인 체인 스펙), 추가로 붙는 것은 따로 호출.

Finding에 label을 담고 다음단계로 진행된다.

라벨 그룹: A 검색실패 / B 생성실패 / C context구조 / D 데이터.
Finding.type 필드에 라벨 그룹을 담고, Finding.label 필드에 세분화 라벨을 담는다.

진단 모드:
  - diagnose() 진입 시 현재 실행의 모드(_active_mode)를 설정한다.
  - 지정된 진단 모드 이하까지만 확정할수있음.
    모드가 낮다면 -> 예비로 지정.
  - 생성 원인(B)은 전부 RAGAS(DEEP) 의존 → DEEP 미만이면 예비 generation_failure 로 롤업.

"""
from __future__ import annotations

from typing import Optional

from core.schema import Finding
from agents.eval.types import (
    Branch, EvalRecord, DEFAULT_TOP_K,
    RAGAS_FAITHFULNESS_MIN, RAGAS_RESPONSE_RELEVANCY_MIN,
    Mode, DEFAULT_MODE, resolve_mode,
)

# 진단 모드, 단계
_active_mode: int = DEFAULT_MODE

# confirm tier = 라벨을 '확정'하는 데 필요한 가장 비싼 자원. (전체 분류표는 README '진단 모드' 참고)
#   tier1 순수 규칙 · tier2 추가 검색 쿼리 · tier3 LLM/RAGAS · tier4 파이프라인 재실행
_LABEL_TIER = {
    # A 검색 (recall<1)
    "retrieval_incomplete_enumeration":    Mode.FAST,      # tier1: gold수 vs top-k 순수 규칙
    "retrieval_low_rank":                  Mode.STANDARD,  # tier2: top-N 재검색
    "retrieval_lexical_mismatch":          Mode.STANDARD,  # tier2: BM25 조회
    "retrieval_semantic_mismatch":         Mode.STANDARD,  # tier2: BM25 + 코퍼스 확인
    "retrieval_missing_gold":              Mode.STANDARD,  # tier2: 코퍼스 멤버십 조회
    "retrieval_missing_bridge_dependency": Mode.FULL,      # tier4: iterative decompose 재실행
    # B 생성 (oracle 실패 / ambiguous_gen)
    "generation_hallucination":            Mode.DEEP,      # tier3: RAGAS faithfulness
    "generation_partial_answer":           Mode.DEEP,      # tier3: RAGAS relevancy
    "generation_hop_binding_error":        Mode.DEEP,      # tier3: RAGAS faithfulness(+추론검증)
    "generation_contradiction":            Mode.DEEP,      # tier3: AspectCritic(LLM)
    "generation_failure":                  Mode.DEEP,      # tier3: 예비 롤업(DEEP서 세분화)
    # C context (recall=1, f1 실패)
    "too_long_context":                    Mode.FULL,      # tier4: ablation 재실행(축소)
    "lost_in_the_middle":                  Mode.FULL,      # tier4: 재실행(재정렬)
    "context_noise_interference":          Mode.FULL,      # tier4: 재실행(노이즈 제거)
    # D 데이터
    "bad_gold_answer":                     Mode.DEEP,      # tier3: RAGAS 2지표(진짜확정=사람)
    "corpus_gap":                          Mode.STANDARD,  # tier2: 코퍼스 조회
    "corpus_gap_partial_hop":              Mode.STANDARD,  # tier2: 코퍼스 조회(hop별)
}


def _tier_of(label: str) -> int:
    return _LABEL_TIER.get(label, Mode.FAST)


# ══════════════════════════════════════════════════════════════════
#  A그룹: 검색 실패 (Oracle 통과) — retrieval_*
# ══════════════════════════════════════════════════════════════════

def retrieval_low_rank(record: EvalRecord) -> Optional[Finding]:
    """
    gold가 top-N 후보엔 있으나 순위가 낮아 top-k 밖. 처방: 리랭커 추가.
    확정: top-N 재검색에서 gold 발견(tier2). 
    예비: 없음
    """
    if _gold_in_wider_candidates(record) is True:
        return _finding(record, "retrieval_low_rank", "retrieval_failure", confirmed=True)
    return None


def retrieval_lexical_mismatch(record: EvalRecord) -> Optional[Finding]:
    """
    dense는 놓쳤으나 BM25로 잡히는 단어 불일치. 처방: 하이브리드 검색.
    확정: BM25 가 gold 를 잡음(tier2). 
    예비: 없음
    """
    if _bm25_hits_gold(record) is True:
        return _finding(record, "retrieval_lexical_mismatch", "retrieval_failure", confirmed=True)
    return None


def retrieval_semantic_mismatch(record: EvalRecord) -> Optional[Finding]:
    """
    dense·BM25 모두 놓친 의미 연결 실패. 처방: 임베딩/청킹 교체.
    확정: BM25 도 gold 를 못 잡음(tier2). 
    예비: 없음
    """
    if _bm25_hits_gold(record) is False:
        return _finding(record, "retrieval_semantic_mismatch", "retrieval_failure", confirmed=True)
    return None


def retrieval_missing_gold(record: EvalRecord) -> Optional[Finding]:
    """
    gold는 corpus에 있으나 top-k에 없음. 처방: top_k/chunk 조정.
    확정: 코퍼스에 gold 존재(tier2). 
    예비: recall<1(tier1, 코퍼스 확인 전).
    """
    in_corpus = _gold_in_corpus(record)
    if in_corpus is True:
        return _finding(record, "retrieval_missing_gold", "retrieval_failure", confirmed=True)
    if in_corpus is None and record.recall_at_k < 1:
        return _finding(record, "retrieval_missing_gold", "retrieval_failure", confirmed=False)
    return None


def retrieval_missing_bridge_dependency(record: EvalRecord) -> Optional[Finding]:
    """
    멀티홉 연쇄형: 2번째 hop 근거가 1번째 hop에 의존. 처방: iterative_decompose.
    확정: decompose 재실행 시 회복(tier4). 
    예비: 멀티홉+recall<1(tier1).
    """
    recovers = _bridge_decompose_recovers(record)
    if recovers is True:
        return _finding(record, "retrieval_missing_bridge_dependency", "retrieval_failure", confirmed=True)
    if recovers is None and _is_multi_hop(record) and record.recall_at_k < 1:
        return _finding(record, "retrieval_missing_bridge_dependency", "retrieval_failure", confirmed=False)
    return None


def retrieval_incomplete_enumeration(record: EvalRecord) -> Optional[Finding]:
    """
    나열형: 필요한 근거 개수 가변인데 top-k 고정이라 누락. 처방: 동적 top-k/adaptive.
    확정: gold수 vs top-k 순수 규칙(tier1) — 바로 확정.?????????
    """
    if _enumeration_signal(record):
        return _finding(record, "retrieval_incomplete_enumeration", "retrieval_failure", confirmed=True)
    return None


# ══════════════════════════════════════════════════════════════════
#  B그룹: 생성 실패 (Oracle 실패) — generation_*
# ══════════════════════════════════════════════════════════════════

def generation_hop_binding_error(record: EvalRecord) -> Optional[Finding]:
    """
    멀티홉: 각 hop 사실은 맞으나 결합이 틀림(faithfulness 높음). 처방: 단계별 근거 CoT.
    확정: faithfulness 측정되어 높음(tier3). 미측정 None → 안 뜸(→ _pick 롤업).
    """
    faith = _faith(record)
    if _is_multi_hop(record) and faith is not None and faith >= RAGAS_FAITHFULNESS_MIN:
        return _finding(record, "generation_hop_binding_error", "generation_failure", confirmed=True)
    return None


def generation_hallucination(record: EvalRecord) -> Optional[Finding]:
    """
    정답 context가 있는데 지어냄. 처방: 그라운딩 프롬프트+인용.
    확정: faithfulness 측정되어 낮음(tier3). 미측정 None → 안 뜸(→ _pick 롤업).
    """
    faith = _faith(record)
    if faith is not None and faith < RAGAS_FAITHFULNESS_MIN:
        return _finding(record, "generation_hallucination", "generation_failure", confirmed=True)
    return None


def generation_partial_answer(record: EvalRecord) -> Optional[Finding]:
    """
    정답 context가 있는데 일부 요소·조건 누락. 처방: 완결성 프롬프트/checklist.
    확정: relevancy 측정되어 낮음(tier3).
    """
    rel = _rel(record)
    if rel is not None and rel < RAGAS_RESPONSE_RELEVANCY_MIN:
        return _finding(record, "generation_partial_answer", "generation_failure", confirmed=True)
    return None


# 나중에 개발
# def generation_contradiction(record: EvalRecord) -> Optional[Finding]:
#     """
#     답변이 컨텍스트/사실과 모순(AspectCritic, tier3). 다른 라벨과 함께 붙는다.
#     """
#     if record.aspect.get("contradiction") == 1:
#         return _finding(record, "generation_contradiction", "generation_failure", confirmed=True)
#     return None


# ══════════════════════════════════════════════════════════════════
#  C그룹: context 구조 문제
# ══════════════════════════════════════════════════════════════════

def too_long_context(record: EvalRecord) -> Optional[Finding]:
    """
    context가 너무 길어 잡음·과부하로 품질 저하. 처방: top-k 축소/필터링/압축.
    확정: 축소 재실행 시 회복(tier4). 
    예비: 없음
    """
    if _context_shorten_helps(record) is True:
        return _finding(record, "too_long_context", "retrieval_failure", confirmed=True)
    return None


def lost_in_the_middle(record: EvalRecord) -> Optional[Finding]:
    """청크가 긴 context 중간이라 LLM이 참조 못함. 처방: context 재정렬/top-k 축소.
    확정: gold 앞배치 재실행 시 회복(tier4). 싼 예비 신호 없음."""
    if _gold_front_helps(record) is True:
        return _finding(record, "lost_in_the_middle", "retrieval_failure", confirmed=True)
    return None


def context_noise_interference(record: EvalRecord) -> Optional[Finding]:
    """비-gold 청크의 상충 정보에 이끌림(C 잔여 원인). 처방: 노이즈 필터링+프롬프트.
    확정: 노이즈 제거 재실행 시 회복(tier4). 예비: C 잔여이므로 확정 전엔 항상 예비."""
    helps = _noise_removal_helps(record)
    if helps is True:
        return _finding(record, "context_noise_interference", "retrieval_failure", confirmed=True)
    if helps is None:
        return _finding(record, "context_noise_interference", "retrieval_failure", confirmed=False)
    return None


# ══════════════════════════════════════════════════════════════════
#  D그룹: 데이터 문제 (파이프라인 튜닝 불가)
# ══════════════════════════════════════════════════════════════════

def bad_gold_answer(record: EvalRecord) -> Optional[Finding]:
    """정답셋 자체 오류/모호(충실도·관련성 모두 측정 고득점인데 gold만 불일치). 처방: 사람 검수.
    확정(자동): faith·rel 둘 다 측정 고득점(tier3). 진짜 확정은 사람 검수."""
    if _both_high(_faith(record), _rel(record)):
        return _finding(record, "bad_gold_answer", "gap", confirmed=True)
    return None


def corpus_gap(record: EvalRecord) -> Optional[Finding]:
    """필요한 자료가 코퍼스에 없음(단일홉). 처방: 문서 추가 요청.
    확정: 코퍼스에 gold 없음(tier2). 싼 예비 신호 없음."""
    if _gold_in_corpus(record) is False and not _is_multi_hop(record):
        return _finding(record, "corpus_gap", "gap", confirmed=True)
    return None


def corpus_gap_partial_hop(record: EvalRecord) -> Optional[Finding]:
    """멀티홉 중 일부 hop 근거만 코퍼스에 없음. 처방: 해당 hop 문서 추가 요청.
    확정: 코퍼스에 gold 없음(tier2). 싼 예비 신호 없음."""
    if _gold_in_corpus(record) is False and _is_multi_hop(record):
        return _finding(record, "corpus_gap_partial_hop", "gap", confirmed=True)
    return None


# ══════════════════════════════════════════════════════════════════
#  브랜치별 판정 (설계 STEP4 '브랜치 별 판정 순서')
#  - 검색/생성/컨텍스트 원인은 '한 원인'만 채택 → _pick(record, 스펙)
#  - corpus_gap·모순 은 추가로 붙음
# ══════════════════════════════════════════════════════════════════

# 원인 체인 스펙 = (판정함수들, gate, rollup)  — _pick 이 해석한다.
#   gate   : 이 모드 미만이면 체인을 시도조차 안 함(세분화에 그 자원이 필요). None = 게이트 없음.
#   rollup : 세분화 실패(gate 미달/매치 없음) 시 낼 예비 라벨. None = None 반환.
# 생성·컨텍스트 원인은 전부 RAGAS(=DEEP) 의존 → DEEP 미만이면 예비 generation_failure 로 롤업.
_RETRIEVAL_CAUSE = (
    (retrieval_low_rank, retrieval_lexical_mismatch, retrieval_semantic_mismatch, retrieval_missing_gold),
    None, None,
)
_RETRIEVAL_CAUSE_PARTIAL = (
    (retrieval_incomplete_enumeration, retrieval_missing_bridge_dependency,
     retrieval_low_rank, retrieval_lexical_mismatch, retrieval_semantic_mismatch, retrieval_missing_gold),
    None, None,
)
_GENERATION_CAUSE = (
    (bad_gold_answer, generation_hop_binding_error, generation_hallucination, generation_partial_answer),
    Mode.DEEP, "generation_failure",
)
_CONTEXT_CAUSE = (
    (bad_gold_answer, too_long_context, lost_in_the_middle, context_noise_interference),
    Mode.DEEP, "generation_failure",
)


def _branch_retrieval_fail(record: EvalRecord) -> list[Finding]:
    # 검색 실패(Oracle 통과): 검색 원인 1개
    return _collect(
        _pick(record, _RETRIEVAL_CAUSE)
    )


def _branch_retrieval_partial(record: EvalRecord) -> list[Finding]:
    # 검색 부분 실패(Oracle 통과): 나열형/연쇄형 우선 → 일반 검색 원인
    return _collect(
        _pick(record, _RETRIEVAL_CAUSE_PARTIAL)
    )


def _branch_retrieval_gen_fail(record: EvalRecord) -> list[Finding]:
    # 검색 실패 + 생성 실패: (최대 3개의 레이블이 나올수 있음 - 확인 필요)
    return _collect(
        corpus_gap(record),
        _pick(record, _RETRIEVAL_CAUSE),
        _pick(record, _GENERATION_CAUSE),
    )


def _branch_retrieval_partial_gen_fail(record: EvalRecord) -> list[Finding]:
    # 검색 부분 실패 + 생성 실패 (최대 3개의 레이블이 나올수 있음 - 확인 필요)
    return _collect(
        corpus_gap_partial_hop(record),
        _pick(record, _RETRIEVAL_CAUSE_PARTIAL),
        _pick(record, _GENERATION_CAUSE),
    )


def _branch_ambiguous_context(record: EvalRecord) -> list[Finding]:
    # 애매함, 컨텍스트 원인: C 원인
    return _collect(
        _pick(record, _CONTEXT_CAUSE),
    )


def _branch_ambiguous_gen(record: EvalRecord) -> list[Finding]:
    # 애매함, 생성 원인: 생성 원인
    return _collect(
        _pick(record, _GENERATION_CAUSE),
    )


def _branch_no_answer_violation(record: EvalRecord) -> list[Finding]:
    # 무응답인데 답을 지어냄 → 생성 원인????
    return _collect(
        _pick(record, _GENERATION_CAUSE),
    )


_BRANCH_DISPATCH = {
    Branch.RETRIEVAL_FAIL:             _branch_retrieval_fail,
    Branch.RETRIEVAL_PARTIAL:          _branch_retrieval_partial,
    Branch.RETRIEVAL_GEN_FAIL:         _branch_retrieval_gen_fail,
    Branch.RETRIEVAL_PARTIAL_GEN_FAIL: _branch_retrieval_partial_gen_fail,
    Branch.AMBIGUOUS_CONTEXT:          _branch_ambiguous_context,
    Branch.AMBIGUOUS_GEN:              _branch_ambiguous_gen,
    Branch.NO_ANSWER_VIOLATION:        _branch_no_answer_violation,
}

# ── 판정 콤비네이터 ──────────────────────────────────────────────

def _pick(record: EvalRecord, spec) -> Optional[Finding]:
    """원인 체인 스펙 (funcs, gate, rollup) 에서 '한 원인'을 고른다.

      1) 현재 모드에서 확정 가능한(confirmed) 첫 매치를 우선, 없으면 첫 매치(가장 구체적)를 예비로.
         (tier 게이팅 ↔ 구체성 순서 화해: 상위-tier 예비 라벨이 하위-tier 확정 라벨을 가리지 않게.
          예: STANDARD 에서 bridge(FULL·예비)가 missing_gold(STANDARD·확정)를 덮어쓰지 않도록.)
      2) gate 미만 모드면 체인을 시도조차 안 함(세분화에 그 자원이 필요할 때).
      3) 세분화 실패(gate 미달/매치 없음) 시 rollup(예비)로 롤백. rollup 없으면 None.
    """
    funcs, gate, rollup = spec
    if gate is None or _active_mode >= gate:
        first_match = None
        for fn in funcs:
            f = fn(record)
            if f is None:
                continue
            if f.confirmed:
                return f
            if first_match is None:
                first_match = f
        if first_match is not None:
            return first_match
    if rollup is not None:
        return _finding(record, rollup, rollup, confirmed=False)
    return None


def _collect(*items) -> list[Finding]:
    """None 을 걸러 Finding 리스트로."""
    return [f for f in items if f is not None]


def _dedup(findings: list[Finding]) -> list[Finding]:
    seen, out = set(), []
    for f in findings:
        key = f.label
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


# ══════════════════════════════════════════════════════════════════
#  판별 신호 (라벨 함수가 호출) — tier / 사용 자원별 정리
#    각 신호는 tri-state: None(미실행/모름) · True · False.
#    · 비용 게이트: [훅] 첫 줄에서 self-gate (`if _active_mode < <tier>: return None`).
#    · 확정 게이트: 라벨 함수가 신호 발동 여부로 confirmed 를 정함(_finding 에 명시).
#    · memoize: 비싼 [훅]은 _signal 로 record.signals(=state 캐시)에 저장 → 재진단 시 재사용.
# ══════════════════════════════════════════════════════════════════

def _signal(record: EvalRecord, name: str, compute):
    """비싼 판별 신호 memoize. record.signals(=state.diagnosis_cache[probe_id] 뷰)에 있으면 재사용,
    없으면 compute() 계산해 저장. (self-gate 로 걸러진 None 은 호출 전에 return 되어 캐시되지 않음.)"""
    cache = record.signals
    if name not in cache:
        cache[name] = compute()
    return cache[name]


# ── tier1 · 순수 규칙 (자원: 이미 계산된 지표/probe 메타 — 추가 조회 없음) ──
# (recall_at_k < 1 도 tier1 순수 규칙 — missing_gold / bridge 가 라벨 함수에서 직접 사용)

def _is_multi_hop(record: EvalRecord) -> bool:
    """멀티홉 질문 여부(probe.qtype). bridge / hop_binding / corpus_gap_partial_hop 판별."""
    return record.probe.qtype in ("bridge", "comparison", "aggregation")


def _enumeration_signal(record: EvalRecord) -> bool:
    """gold 개수가 top-k(=검색 결과 수)에 근접/초과 → 나열형 누락. incomplete_enumeration 용."""
    k = len(record.retrieved_chunk_ids) or DEFAULT_TOP_K
    gold_n = len(record.probe.gold_chunk_ids)
    return gold_n >= 2 and gold_n >= int(k * 0.8)


# ── tier2 · 추가 검색 쿼리 (자원: top-N 재검색 / BM25 / 코퍼스 조회) — [훅 미구현] ──

def _gold_in_wider_candidates(record: EvalRecord):
    """[구현 포인트·tier2] top-N(예: 100) 재검색 후 gold 존재 여부. retrieval_low_rank 용."""
    if _active_mode < Mode.STANDARD:   # 비용 게이트 (STANDARD 미만이면 재검색 안 함)
        return None
    return _signal(record, "gold_in_wider_candidates", lambda: None)  # 구현 시 lambda→top-N 재검색


def _bm25_hits_gold(record: EvalRecord):
    """[구현 포인트·tier2] BM25 검색 후 gold 존재 여부. lexical(True)/semantic(False) mismatch 용."""
    if _active_mode < Mode.STANDARD:
        return None
    return _signal(record, "bm25_hits_gold", lambda: None)           # 구현 시 lambda→BM25 조회


def _gold_in_corpus(record: EvalRecord):
    """[구현 포인트·tier2] 코퍼스 전체 조회로 gold 존재 여부. True→missing_gold / False→corpus_gap."""
    if _active_mode < Mode.STANDARD:
        return None
    return _signal(record, "gold_in_corpus", lambda: None)           # 구현 시 lambda→코퍼스 조회


# ── tier3 · LLM/RAGAS (자원: STEP3-2 RAGAS 점수 · AspectCritic) ──
# 점수는 record.ragas / record.oracle_ragas 에 이미 실려 있음(있으면). 미측정 = None.
# (record.aspect["contradiction"] 도 tier3 AspectCritic — generation_contradiction 이 직접 사용)

def _use_oracle(record: EvalRecord) -> bool:
    """트랙 선택: 생성 실패 계열은 오라클 트랙 점수로, 컨텍스트 계열은 실제 트랙으로 판정."""
    return record.branch in (
        Branch.RETRIEVAL_GEN_FAIL, Branch.RETRIEVAL_PARTIAL_GEN_FAIL,
        Branch.AMBIGUOUS_GEN, Branch.NO_ANSWER_VIOLATION,
    )


def _faith(record: EvalRecord):
    """faithfulness(충실도). hallucination / hop_binding / bad_gold 용. 모드부족/미측정 None."""
    if _active_mode < Mode.DEEP:       # 비용 게이트 (RAGAS 는 DEEP 이상에서만)
        return None
    src = record.oracle_ragas if _use_oracle(record) else record.ragas
    return src.get("faithfulness")


def _rel(record: EvalRecord):
    """response_relevancy(관련성). partial_answer / bad_gold 용. 모드부족/미측정 None."""
    if _active_mode < Mode.DEEP:
        return None
    src = record.oracle_ragas if _use_oracle(record) else record.ragas
    return src.get("response_relevancy")


def _both_high(faith, rel) -> bool:
    """bad_gold_answer 판정용: 충실도·관련성이 모두 '측정되어' 임계값 이상."""
    if _active_mode < Mode.DEEP:
        return None
    return (faith is not None and faith >= RAGAS_FAITHFULNESS_MIN
            and rel is not None and rel >= RAGAS_RESPONSE_RELEVANCY_MIN)


# ── tier4 · 파이프라인 재실행 (자원: ablation 재실행) — [훅 미구현] ──
# 각 신호는 확정용. 대응 라벨은 싼 예비 신호가 있으면 그걸로 예비를 먼저 낸다
# (bridge=멀티홉+recall, context_noise=C 잔여 항상). 확정은 이 재실행 훅이 True 를 줄 때만.

def _context_shorten_helps(record: EvalRecord):
    """[구현 포인트·tier4] context 축소 재실행 시 정답률 상승 여부. too_long_context 확정용."""
    if _active_mode < Mode.FULL:
        return None
    return _signal(record, "context_shorten_helps", lambda: None)    # 구현 시 lambda→축소 재실행


def _gold_front_helps(record: EvalRecord):
    """[구현 포인트·tier4] gold를 앞쪽 배치 재실행 시 정답률 회복 여부. lost_in_the_middle 확정용."""
    if _active_mode < Mode.FULL:
        return None
    return _signal(record, "gold_front_helps", lambda: None)         # 구현 시 lambda→재정렬 재실행


def _bridge_decompose_recovers(record: EvalRecord):
    """[구현 포인트·tier4] iterative_decompose 재실행 시 hop2 근거 회복 여부. missing_bridge 확정용."""
    if _active_mode < Mode.FULL:
        return None
    return _signal(record, "bridge_decompose_recovers", lambda: None)  # 구현 시 lambda→분해 재검색


def _noise_removal_helps(record: EvalRecord):
    """[구현 포인트·tier4] 비-gold 노이즈 제거 재실행 시 회복 여부. context_noise 확정용."""
    if _active_mode < Mode.FULL:
        return None
    return _signal(record, "noise_removal_helps", lambda: None)      # 구현 시 lambda→노이즈 제거 재실행


# ── Finding 빌더 ─────────────────────────────────────────────────

def _group_of(label: str, ftype: str) -> str:
    """label·ftype 에서 그룹(A/B/C/D)을 파생 — 처방 순서 정렬용."""
    if ftype == "gap":
        return "D"
    if label.startswith("retrieval_"):
        return "A"
    if label.startswith("generation_"):
        return "B"
    return "C"


# 심각도: 구조적/데이터 결함은 critical, 나머지는 warning
_CRITICAL_LABELS = {
    "retrieval_semantic_mismatch", "retrieval_missing_gold",
    "generation_hallucination", "corpus_gap", "corpus_gap_partial_hop",
}


def _severity_of(label: str) -> str:
    if label in _CRITICAL_LABELS:
        return "critical"
    return "warning"


def _finding(record: EvalRecord, label: str, ftype: str, confirmed: bool) -> Finding:
    """라벨 함수 공통 Finding 생성기.

    confirmed 는 라벨 함수가 명시한다 — '확정 신호가 실제로 발동했는지'.
      True  = 확정 신호(그 자원)가 발동해 확정.
      False = 싼 예비 신호로 의심만(확정 자원 미실행) → 상위 모드에서 확정.
    (mode>=tier 자동판정 아님 — 자원 미실행/미측정이면 예비.) tier 는 리포트용 메타.
    """
    probe = record.probe
    group = _group_of(label, ftype)
    tier = _tier_of(label)
    prefix = "" if confirmed else "[예비] "
    return Finding(
        finding_id=f"{probe.probe_id}:{label}",
        type=ftype,
        severity=_severity_of(label),
        description=f"{prefix}[{group}그룹] {label}",
        label=label,
        tier=tier,
        confirmed=confirmed,
        affected_chunks=list(probe.gold_chunk_ids),
        affected_probes=[probe.probe_id],
        metadata={"group": group, "branch": record.branch},
    )


# ── 메인 ─────────────────────────────────────────────────────────

# 처방 순서: D 먼저 → A → C → B, 그다음 심각도.
_GROUP_ORDER = {"D": 0, "A": 1, "C": 2, "B": 3}
_SEV_ORDER = {"critical": 0, "warning": 1, "info": 2}


def diagnose(record: EvalRecord, mode: Optional[int] = None) -> list[Finding]:
    """record 의 브랜치에 맞는 라벨 함수들을 호출해 Finding 리스트를 만든다.

    mode(진단 tier 상한)에 따라 상위-tier 라벨은 '예비(confirmed=False)'로 나가고,
    생성 원인(RAGAS 의존)은 DEEP 미만이면 하나의 예비 generation_failure 로 롤업된다.
    mode 미지정 시 EVAL_MODE 환경변수(기본 FAST)를 사용한다.
    """
    global _active_mode
    _active_mode = mode if mode is not None else resolve_mode()

    if record.branch in (Branch.SUCCESS, Branch.NO_ANSWER_OK):
        return []

    dispatcher = _BRANCH_DISPATCH.get(record.branch)
    findings = list(dispatcher(record)) if dispatcher else []

    findings = _dedup(findings)
    findings.sort(key=lambda f: (
        _GROUP_ORDER.get(f.metadata.get("group"), 9),
        _SEV_ORDER.get(f.severity, 9),
    ))
    return findings