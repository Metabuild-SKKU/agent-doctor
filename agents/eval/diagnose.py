"""
agents/eval/diagnose.py
STEP4: 원인 판정 (Finding 생성)

설계 문서 STEP4 '브랜치 별 판정 순서' + '처방' 파일을 구조 그대로 구현.

구조 원칙:
  1. **라벨마다 판정 함수 1개.**  각 함수는 자기 라벨의 '판별 신호(원인)'를 검사해
     맞으면 Finding, 아니면 None 을 돌려준다. (원인·처방 요약은 각 함수 docstring 참고)
  2. **브랜치마다 해당 함수들을 호출.**  각 브랜치(_dx_*)가 설계의 판정 순서표대로 라벨
     함수들을 부른다. '하나만 고르는' 부분은 _first(첫 매치), 추가로 붙는 것은 따로 호출.

Finding 은 **라벨(진단명)** 만 싣는다. 라벨별 처방·바꿔야 하는 config 는 처방 파일에 정리돼
있으며 Optimize Agent 가 label 로 매핑해 적용한다(진단=Eval, 처방적용=Optimize 분리).

라벨 그룹: A 검색실패 / B 생성실패 / C context구조 / D 데이터. 처방 순서: **D 먼저** → A → C → B.
Finding.type 은 공유 스키마 허용 집합만 사용하고, 세분화 라벨은 Finding.label 필드에 담는다.

[구현 포인트]  일부 판별 신호는 파이프라인 추가 실험이 필요(_bm25_hits_gold,
  _gold_in_wider_candidates, _gold_in_corpus, _context_shorten_helps, _gold_front_helps).
  지금은 None(미확보)을 반환해 각 체인의 '가장 일반적·저비용 라벨'로 안전 기본 판정한다.
"""
from __future__ import annotations

from typing import Optional

from core.schema import Finding
from agents.eval.types import (
    Branch, EvalRecord, DEFAULT_TOP_K,
    RAGAS_FAITHFULNESS_MIN, RAGAS_RESPONSE_RELEVANCY_MIN,
    Mode, DEFAULT_MODE, resolve_mode,
)


# ══════════════════════════════════════════════════════════════════
#  진단 모드 (비용 tier)
#  - diagnose() 진입 시 현재 실행의 자원 상한(_active_mode)을 설정한다.
#    (단일스레드 eval 루프 전제. 병렬화하면 contextvars 로 교체.)
#  - 라벨마다 '확정(confirm) tier' = 판별에 필요한 가장 비싼 자원(_LABEL_TIER).
#    mode >= tier → 확정(confirmed=True), 미만 → 예비(confirmed=False).
#  - 생성 원인(B)은 전부 RAGAS(DEEP) 의존 → DEEP 미만이면 예비 generation_failure 로 롤업.
# ══════════════════════════════════════════════════════════════════

_active_mode: int = DEFAULT_MODE

_LABEL_TIER = {
    # A 검색: BM25/top-N/rank = tier1, 단 missing_gold 는 코퍼스 스캔 = tier2
    "retrieval_low_rank":                  Mode.FAST,
    "retrieval_lexical_mismatch":          Mode.FAST,
    "retrieval_semantic_mismatch":         Mode.FAST,
    "retrieval_incomplete_enumeration":    Mode.FAST,
    "retrieval_missing_gold":              Mode.STANDARD,
    "retrieval_missing_bridge_dependency": Mode.FULL,     # 재실행(iterative decompose)
    # B 생성: RAGAS/LLM = tier3
    "generation_hop_binding_error":        Mode.DEEP,
    "generation_hallucination":            Mode.DEEP,
    "generation_partial_answer":           Mode.DEEP,
    "generation_contradiction":            Mode.DEEP,
    "generation_failure":                  Mode.DEEP,     # 예비 롤업(원인 미상)
    # C context: RAGAS 후보 + 재실행 확정 = tier4
    "too_long_context":                    Mode.FULL,
    "lost_in_the_middle":                  Mode.FULL,
    "context_noise_interference":          Mode.FULL,
    # D 데이터
    "bad_gold_answer":                     Mode.DEEP,     # RAGAS 두 지표(faith·rel)
    "corpus_gap":                          Mode.STANDARD,
    "corpus_gap_partial_hop":              Mode.STANDARD,
    # aux
    "staleness":                           Mode.DEEP,     # AspectCritic
}


def _tier_of(label: str) -> int:
    return _LABEL_TIER.get(label, Mode.FAST)


# ══════════════════════════════════════════════════════════════════
#  A그룹: 검색 실패 (Oracle 통과) — retrieval_*
# ══════════════════════════════════════════════════════════════════

def retrieval_low_rank(record: EvalRecord) -> Optional[Finding]:
    """gold가 top-N 후보엔 있으나 순위가 낮아 top-k 밖. 처방: 리랭커 추가."""
    if _gold_in_wider_candidates(record) is not True:
        return None
    return _finding(record, "retrieval_low_rank", "retrieval_failure")


def retrieval_lexical_mismatch(record: EvalRecord) -> Optional[Finding]:
    """dense는 놓쳤으나 BM25로 잡히는 단어 불일치. 처방: 하이브리드 검색."""
    if _bm25_hits_gold(record) is not True:
        return None
    return _finding(record, "retrieval_lexical_mismatch", "retrieval_failure")


def retrieval_semantic_mismatch(record: EvalRecord) -> Optional[Finding]:
    """dense·BM25 모두 놓친 의미 연결 실패. 처방: 임베딩/청킹 교체."""
    if _bm25_hits_gold(record) is not False:   # None(미확보)/True 면 아님
        return None
    return _finding(record, "retrieval_semantic_mismatch", "retrieval_failure")


def retrieval_missing_gold(record: EvalRecord) -> Optional[Finding]:
    """gold는 corpus에 있으나 top-k에 없음(검색 원인 미상 시 기본 라벨). 처방: top_k/chunk 조정."""
    if not (record.recall_at_k < 1):
        return None
    return _finding(record, "retrieval_missing_gold", "retrieval_failure")


def retrieval_missing_bridge_dependency(record: EvalRecord) -> Optional[Finding]:
    """멀티홉 연쇄형: 2번째 hop 근거가 1번째 hop에 의존. 처방: iterative_decompose."""
    if not (_is_multi_hop(record) and record.recall_at_k < 1):
        return None
    return _finding(record, "retrieval_missing_bridge_dependency", "retrieval_failure")


def retrieval_incomplete_enumeration(record: EvalRecord) -> Optional[Finding]:
    """나열형: 필요한 근거 개수 가변인데 top-k 고정이라 누락. 처방: 동적 top-k/adaptive."""
    if not _enumeration_signal(record):
        return None
    return _finding(record, "retrieval_incomplete_enumeration", "retrieval_failure")


# ══════════════════════════════════════════════════════════════════
#  B그룹: 생성 실패 (Oracle 실패) — generation_*
# ══════════════════════════════════════════════════════════════════

def generation_hop_binding_error(record: EvalRecord) -> Optional[Finding]:
    """멀티홉: 각 hop 사실은 맞으나 결합이 틀림(faithfulness 높음). 처방: 단계별 근거 CoT.
    faithfulness 가 '측정되어' 높을 때만 판정(미측정=None 은 근거 없음 → 판정 안 함)."""
    faith = _faith(record)
    if not (_is_multi_hop(record) and faith is not None and faith >= RAGAS_FAITHFULNESS_MIN):
        return None
    return _finding(record, "generation_hop_binding_error", "generation_failure")


def generation_hallucination(record: EvalRecord) -> Optional[Finding]:
    """정답 context가 있는데 지어냄. 처방: 그라운딩 프롬프트+인용.
    faithfulness 가 '측정되어' 낮을 때만 판정(미측정=None 은 근거 없음 → 판정 안 함).
    RAGAS 미실행 시엔 _gen_cause 가 예비 generation_failure 로 롤백한다."""
    faith = _faith(record)
    if not (faith is not None and faith < RAGAS_FAITHFULNESS_MIN):
        return None
    return _finding(record, "generation_hallucination", "generation_failure")


def generation_partial_answer(record: EvalRecord) -> Optional[Finding]:
    """정답 context가 있는데 일부 요소·조건 누락. 처방: 완결성 프롬프트/checklist."""
    rel = _rel(record)
    if not (rel is not None and rel < RAGAS_RESPONSE_RELEVANCY_MIN):
        return None
    return _finding(record, "generation_partial_answer", "generation_failure")


def generation_contradiction(record: EvalRecord) -> Optional[Finding]:
    """답변이 컨텍스트/사실과 모순(AspectCritic). 다른 라벨과 함께 붙는다."""
    if record.aspect.get("contradiction") != 1:
        return None
    return _finding(record, "generation_contradiction", "generation_failure")


# ══════════════════════════════════════════════════════════════════
#  C그룹: context 구조 문제
# ══════════════════════════════════════════════════════════════════

def too_long_context(record: EvalRecord) -> Optional[Finding]:
    """context가 너무 길어 잡음·과부하로 품질 저하. 처방: top-k 축소/필터링/압축."""
    if _context_shorten_helps(record) is not True:
        return None
    return _finding(record, "too_long_context", "retrieval_failure")


def lost_in_the_middle(record: EvalRecord) -> Optional[Finding]:
    """청크가 긴 context 중간이라 LLM이 참조 못함. 처방: context 재정렬/top-k 축소."""
    if _gold_front_helps(record) is not True:
        return None
    return _finding(record, "lost_in_the_middle", "retrieval_failure")


def context_noise_interference(record: EvalRecord) -> Optional[Finding]:
    """비-gold 청크의 상충 정보에 이끌림(C 잔여 원인). 처방: 노이즈 필터링+프롬프트."""
    return _finding(record, "context_noise_interference", "retrieval_failure")


# ══════════════════════════════════════════════════════════════════
#  D그룹: 데이터 문제 (파이프라인 튜닝 불가)
# ══════════════════════════════════════════════════════════════════

def bad_gold_answer(record: EvalRecord) -> Optional[Finding]:
    """정답셋 자체 오류/모호(충실도·관련성 모두 측정 고득점인데 gold만 불일치). 처방: 사람 검수."""
    if not _both_high(_faith(record), _rel(record)):
        return None
    return _finding(record, "bad_gold_answer", "gap")


def corpus_gap(record: EvalRecord) -> Optional[Finding]:
    """필요한 자료가 코퍼스에 없음(단일홉). 처방: 문서 추가 요청."""
    if _gold_in_corpus(record) is not False or _is_multi_hop(record):
        return None
    return _finding(record, "corpus_gap", "gap")


def corpus_gap_partial_hop(record: EvalRecord) -> Optional[Finding]:
    """멀티홉 중 일부 hop 근거만 코퍼스에 없음. 처방: 해당 hop 문서 추가 요청."""
    if _gold_in_corpus(record) is not False or not _is_multi_hop(record):
        return None
    return _finding(record, "corpus_gap_partial_hop", "gap")


# ── 보조(처방 파일 밖): AspectCritic staleness ────────────────────

def staleness(record: EvalRecord) -> Optional[Finding]:
    """답변/컨텍스트에 오래된 정보(AspectCritic). taxonomy 밖 보조 Finding."""
    if record.aspect.get("staleness") != 1:
        return None
    return _finding(record, "staleness", "staleness")


# ══════════════════════════════════════════════════════════════════
#  브랜치별 판정 (설계 STEP4 '브랜치 별 판정 순서')
#  - 검색 원인/생성 원인/컨텍스트 원인은 '첫 매치'로 하나만 채택(_first)
#  - corpus_gap·모순·staleness 는 추가로 붙음
# ══════════════════════════════════════════════════════════════════

_RETRIEVAL_CAUSE = (
    retrieval_low_rank, retrieval_lexical_mismatch, retrieval_semantic_mismatch, retrieval_missing_gold,
)
_RETRIEVAL_CAUSE_PARTIAL = (
    retrieval_incomplete_enumeration, retrieval_missing_bridge_dependency,
    retrieval_low_rank, retrieval_lexical_mismatch, retrieval_semantic_mismatch, retrieval_missing_gold,
)
_GENERATION_CAUSE = (
    bad_gold_answer, generation_hop_binding_error, generation_hallucination, generation_partial_answer,
)
_CONTEXT_CAUSE = (
    bad_gold_answer, too_long_context, lost_in_the_middle, context_noise_interference,
)


def _dx_retrieval_fail(record: EvalRecord) -> list[Finding]:
    # 검색 실패(Oracle 통과): 검색 원인 1개
    return _collect(_first(record, _RETRIEVAL_CAUSE))


def _dx_retrieval_partial(record: EvalRecord) -> list[Finding]:
    # 검색 부분 실패(Oracle 통과): 나열형/연쇄형 우선 → 일반 검색 원인
    return _collect(_first(record, _RETRIEVAL_CAUSE_PARTIAL))


def _dx_retrieval_gen_fail(record: EvalRecord) -> list[Finding]:
    # 검색 실패 + 생성 실패: 데이터 → 검색 원인 → 생성 원인 → 모순
    return _collect(
        corpus_gap(record),
        _first(record, _RETRIEVAL_CAUSE),
        _gen_cause(record),
        generation_contradiction(record),
    )


def _dx_retrieval_partial_gen_fail(record: EvalRecord) -> list[Finding]:
    # 검색 부분 실패 + 생성 실패
    return _collect(
        corpus_gap_partial_hop(record),
        _first(record, _RETRIEVAL_CAUSE_PARTIAL),
        _gen_cause(record),
        generation_contradiction(record),
    )


def _dx_ambiguous_context(record: EvalRecord) -> list[Finding]:
    # 애매함, 컨텍스트 원인: C 원인 1개 + 모순
    return _collect(
        _context_cause(record),
        generation_contradiction(record),
    )


def _dx_ambiguous_gen(record: EvalRecord) -> list[Finding]:
    # 애매함, 생성 원인: 생성 원인 1개 + 모순
    return _collect(
        _gen_cause(record),
        generation_contradiction(record),
    )


def _dx_no_answer_violation(record: EvalRecord) -> list[Finding]:
    # 무응답인데 답을 지어냄 → 생성 원인(모드 따라 예비 롤업/세분화) + 모순
    #   다른 생성 브랜치와 동일하게 _gen_cause 로 라우팅(DEEP 미만이면 예비 generation_failure).
    return _collect(
        _gen_cause(record),
        generation_contradiction(record),
    )


_BRANCH_DISPATCH = {
    Branch.RETRIEVAL_FAIL:             _dx_retrieval_fail,
    Branch.RETRIEVAL_PARTIAL:          _dx_retrieval_partial,
    Branch.RETRIEVAL_GEN_FAIL:         _dx_retrieval_gen_fail,
    Branch.RETRIEVAL_PARTIAL_GEN_FAIL: _dx_retrieval_partial_gen_fail,
    Branch.AMBIGUOUS_CONTEXT:          _dx_ambiguous_context,
    Branch.AMBIGUOUS_GEN:              _dx_ambiguous_gen,
    Branch.NO_ANSWER_VIOLATION:        _dx_no_answer_violation,
}


# ── 메인 ─────────────────────────────────────────────────────────

# 처방 순서: D 먼저 → A → C → B, 그다음 심각도. (aux 보조는 맨 뒤)
_GROUP_ORDER = {"D": 0, "A": 1, "C": 2, "B": 3, "aux": 8}
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
    _extend(findings, staleness(record))   # taxonomy 밖 보조

    findings = _dedup(findings)
    findings.sort(key=lambda f: (
        _GROUP_ORDER.get(f.metadata.get("group"), 9),
        _SEV_ORDER.get(f.severity, 9),
    ))
    return findings


# ── 판정 콤비네이터 ──────────────────────────────────────────────

def _first(record: EvalRecord, funcs) -> Optional[Finding]:
    """funcs 를 순서대로 호출. 현재 모드에서 '확정 가능한(confirmed)' 첫 매치를 우선 채택하고,
    없으면 첫 매치(가장 구체적)를 예비로 반환한다. 매치 자체가 없으면 None.

    tier 게이팅과 구체성 순서를 화해시키기 위함: 상위-tier(현재 모드에서 확정 불가) 라벨이
    하위-tier 확정 라벨을 가리지 않게 한다. 예) STANDARD 에서 bridge(FULL·예비)가
    missing_gold(STANDARD·확정)를 덮어쓰지 않도록.
    """
    first_match = None
    for fn in funcs:
        f = fn(record)
        if f is None:
            continue
        if f.confirmed:
            return f
        if first_match is None:
            first_match = f
    return first_match


def _gen_cause(record: EvalRecord) -> Optional[Finding]:
    """생성 원인 1개. DEEP 이상이고 RAGAS 근거로 특정되면 그 라벨, 아니면 예비 generation_failure.
    (DEEP 미만=근거 없음, 또는 DEEP인데 RAGAS 미실행/미매치 → 세분화 불가 → 항상 예비 롤업.)"""
    if _active_mode >= Mode.DEEP:
        cause = _first(record, _GENERATION_CAUSE)
        if cause is not None:
            return cause
    return _finding(record, "generation_failure", "generation_failure", confirmed=False)


def _context_cause(record: EvalRecord) -> Optional[Finding]:
    """컨텍스트(C) 원인 1개. C는 RAGAS(후보)+재실행(확정) 의존. DEEP 이상이면 후보 판정,
    아니면(또는 미특정) 예비 generation_failure 로 롤업."""
    if _active_mode >= Mode.DEEP:
        cause = _first(record, _CONTEXT_CAUSE)
        if cause is not None:
            return cause
    return _finding(record, "generation_failure", "generation_failure", confirmed=False)


def _collect(*items) -> list[Finding]:
    """None 을 걸러 Finding 리스트로."""
    return [f for f in items if f is not None]


def _extend(findings: list[Finding], item: Optional[Finding]) -> None:
    if item is not None:
        findings.append(item)


def _dedup(findings: list[Finding]) -> list[Finding]:
    seen, out = set(), []
    for f in findings:
        key = f.label
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


# ── 판별 신호 (available) ────────────────────────────────────────

def _is_multi_hop(record: EvalRecord) -> bool:
    return record.probe.qtype in ("bridge", "comparison", "aggregation")


def _enumeration_signal(record: EvalRecord) -> bool:
    """gold 개수가 top-k(=검색 결과 수)에 근접/초과하면 나열형 누락으로 본다."""
    k = len(record.retrieved_chunk_ids) or DEFAULT_TOP_K
    gold_n = len(record.probe.gold_chunk_ids)
    return gold_n >= 2 and gold_n >= int(k * 0.8)


def _use_oracle(record: EvalRecord) -> bool:
    """생성 실패 계열 브랜치는 오라클 트랙 점수로, 컨텍스트 계열은 실제 트랙으로 판정."""
    return record.branch in (
        Branch.RETRIEVAL_GEN_FAIL, Branch.RETRIEVAL_PARTIAL_GEN_FAIL,
        Branch.AMBIGUOUS_GEN, Branch.NO_ANSWER_VIOLATION,
    )


def _faith(record: EvalRecord):
    src = record.oracle_ragas if _use_oracle(record) else record.ragas
    return src.get("faithfulness")


def _rel(record: EvalRecord):
    src = record.oracle_ragas if _use_oracle(record) else record.ragas
    return src.get("response_relevancy")


def _both_high(faith, rel) -> bool:
    """bad_gold_answer 판정용: 충실도·관련성이 모두 '측정되어' 임계값 이상."""
    return (faith is not None and faith >= RAGAS_FAITHFULNESS_MIN
            and rel is not None and rel >= RAGAS_RESPONSE_RELEVANCY_MIN)


# ── 판별 신호 (미확보 → [구현 포인트] 훅) ────────────────────────
#
# 파이프라인에 추가 실험(재검색/재실행)이 필요한 신호들. 지금은 None(미확보)을 반환하며,
# 신호를 붙이면 True/False 를 돌려주도록 구현한다. 그러면 관련 라벨 함수가 자동으로 살아난다.

def _gold_in_wider_candidates(record: EvalRecord):
    """[구현 포인트] top-N(예: 100) 재검색 후 gold 존재 여부. retrieval_low_rank 용."""
    return None


def _bm25_hits_gold(record: EvalRecord):
    """[구현 포인트] BM25 검색 후 gold 존재 여부. lexical vs semantic mismatch 용."""
    return None


def _gold_in_corpus(record: EvalRecord):
    """[구현 포인트] corpus 전체 조회로 gold 존재 여부. corpus_gap 용."""
    return None


def _context_shorten_helps(record: EvalRecord):
    """[구현 포인트] context 축소 재실행 시 정답률 상승 여부. too_long_context 용."""
    return None


def _gold_front_helps(record: EvalRecord):
    """[구현 포인트] gold를 앞쪽에 배치 재실행 시 정답률 회복 여부. lost_in_the_middle 용."""
    return None


# ── Finding 빌더 ─────────────────────────────────────────────────

def _group_of(label: str, ftype: str) -> str:
    """label·ftype 에서 그룹(A/B/C/D/aux)을 파생 — 처방 순서 정렬용."""
    if ftype == "staleness":
        return "aux"
    if ftype == "gap":
        return "D"
    if label.startswith("retrieval_"):
        return "A"
    if label.startswith("generation_"):
        return "B"
    return "C"


# 심각도: 구조적/데이터 결함은 critical, 보조 안내는 info, 나머지는 warning
_CRITICAL_LABELS = {
    "retrieval_semantic_mismatch", "retrieval_missing_gold",
    "generation_hallucination", "corpus_gap", "corpus_gap_partial_hop",
}
_INFO_LABELS = {"staleness"}


def _severity_of(label: str) -> str:
    if label in _CRITICAL_LABELS:
        return "critical"
    if label in _INFO_LABELS:
        return "info"
    return "warning"


def _finding(record: EvalRecord, label: str, ftype: str,
             confirmed: Optional[bool] = None) -> Finding:
    """라벨 함수 공통 Finding 생성기. 라벨(진단명)만 싣는다.

    tier = 라벨의 확정 자원 tier. confirmed 기본값은 '현재 진단 모드가 그 tier 를 감당하는지'.
    다만 근거 없이 특정만 못 한 롤업(generation_failure)처럼 강제로 예비 처리해야 할 때는
    confirmed=False 를 명시해 override 한다(모드가 높아도 예비 유지).
    확정 못 한 라벨은 예비(confirmed=False)로 내보내 상위 모드에서 확정하도록 남긴다.
    """
    probe = record.probe
    group = _group_of(label, ftype)
    tier = _tier_of(label)
    if confirmed is None:
        confirmed = _active_mode >= tier
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
