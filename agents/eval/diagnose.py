"""
agents/eval/diagnose.py
STEP4: 원인 판정 (Finding 생성)

구조 원칙(브랜치 없음):
  1. 라벨마다 판정 함수 1개: 각 함수는 자기 라벨의 '판별 신호(원인)'를 검사해
     맞으면 Finding, 아니면 None 을 돌려준다. 각 라벨은 자기 싼 전제(recall/f1/oracle)로
     self-scope 하므로, 안 맞는 슬롯은 자연히 빈다.
  2. diagnose() 가 모든 원인 슬롯을 시도: 슬롯당 _pick 으로 '한 원인' 채택(확정 우선),
     corpus_gap 은 additive. 판별 신호·지표 계산은 전부 signals 모듈에서 lazy·memoize.

Finding에 label을 담고 다음단계로 진행된다.

라벨 그룹: A 검색실패 / B 생성실패 / C context구조 / D 데이터.
Finding.type 필드에 라벨 그룹을 담고, Finding.label 필드에 세분화 라벨을 담는다.

진단 모드:
  - 1[FAST], 2[STANDARD], 3[DEEP], 4[FULL]
  - diagnose() 진입 시 signals.set_mode 로 현재 실행 모드를 설정한다.
  - 지정된 진단 모드 이하 tier 까지만 확정할 수 있다(그 위 tier 라벨은 예비).
  - 각 라벨은 확정 신호(비용 큼)·예비 신호(비용 작음, 없는 경우가 많음)를 가진다.
  - 생성 원인(B)은 전부 RAGAS(DEEP) 의존 → DEEP 미만이면 예비 generation_failure 로 롤업한다.

판별 신호·지표·전제·진단 자원(_ctx)·모드 상태는 agents/eval/signals.py 에 존재한다.
여기서는 라벨 함수와 조립(diagnose)만 담당한다.
"""
from __future__ import annotations

from typing import Optional

from core.schema import Finding
from agents.eval.types import (
    EvalRecord, Mode, resolve_mode,
    RAGAS_FAITHFULNESS_MIN, RAGAS_RESPONSE_RELEVANCY_MIN,
)
from agents.eval.signals import (
    set_mode, set_context, _compute_metrics, _no_diagnosis,
    _retrieval_failed, _generation_failed, _context_applicable,
    _is_multi_hop, _enumeration_cache,
    _gold_in_wider_candidates, _gold_ranks, _bm25_hits_gold, _gold_in_corpus,
    _faith, _faith_oracle, _rel, _rel_oracle, _both_high,
    _context_shorten_helps, _gold_front_helps, _noise_removal_helps,
    _bridge_decompose_recovers,
)


# ══════════════════════════════════════════════════════════════════
#  A그룹: 검색 실패 (Oracle 통과) — retrieval_*
# ══════════════════════════════════════════════════════════════════

def retrieval_low_rank(record: EvalRecord) -> Optional[Finding]:
    """
    gold가 top-N 후보엔 있으나 순위가 낮아 top-k 밖.
    확정: top-N 재검색에서 gold 발견(tier2).
    예비: 없음
    """
    if _gold_in_wider_candidates(record) is True:
        return _finding(record, "retrieval_low_rank", "retrieval_failure", confirmed=True)
    return None


def retrieval_lexical_mismatch(record: EvalRecord) -> Optional[Finding]:
    """
    dense는 놓쳤으나 BM25로 잡히는 단어 불일치.
    확정: BM25 가 gold 를 잡음(tier2).
    예비: 없음
    """
    if _bm25_hits_gold(record) is True:
        return _finding(record, "retrieval_lexical_mismatch", "retrieval_failure", confirmed=True)
    return None


def retrieval_semantic_mismatch(record: EvalRecord) -> Optional[Finding]:
    """
    dense·BM25 모두 놓친 의미 연결 실패. (단 gold 가 코퍼스엔 있을 때만 — 없으면 corpus_gap)
    확정: BM25 도 gold 를 못 잡음 + gold 는 코퍼스에 존재(tier2).
    예비: 없음
    """
    if _bm25_hits_gold(record) is False and _gold_in_corpus(record) is not False:
        return _finding(record, "retrieval_semantic_mismatch", "retrieval_failure", confirmed=True)
    return None


def retrieval_missing_gold(record: EvalRecord) -> Optional[Finding]:
    """
    gold는 corpus에 있으나 top-k에 없음.
    확정: 코퍼스에 gold 존재(tier2).
    예비: recall<1(tier1, 코퍼스 확인 전).
    """
    if not _retrieval_failed(record):
        return None

    in_corpus = _gold_in_corpus(record)
    if in_corpus is True:
        return _finding(record, "retrieval_missing_gold", "retrieval_failure", confirmed=True)
    if in_corpus is None:
        return _finding(record, "retrieval_missing_gold", "retrieval_failure", confirmed=False)
    if in_corpus is False:
        return None


def retrieval_missing_bridge_dependency(record: EvalRecord) -> Optional[Finding]:
    """
    멀티홉 연쇄형: 2번째 hop 근거가 1번째 hop에 의존.
    확정: decompose 재실행 시 회복(tier4).
    예비: 멀티홉+recall<1(tier1).
    """
    if not _retrieval_failed(record) or not _is_multi_hop(record):
        return None

    recovers = _bridge_decompose_recovers(record)
    if recovers is True:
        return _finding(record, "retrieval_missing_bridge_dependency", "retrieval_failure", confirmed=True)
    if recovers is None:
        return _finding(record, "retrieval_missing_bridge_dependency", "retrieval_failure", confirmed=False)
    if recovers is False:
        return None


def retrieval_incomplete_enumeration(record: EvalRecord) -> Optional[Finding]:
    """
    나열형: 필요한 근거 개수 가변인데 top-k 고정이라 누락.
    확정: gold수 vs top-k 순수 규칙(tier1) — 바로 확정.?????????
    """
    if not _retrieval_failed(record):
        return None
    if _enumeration_cache(record):
        return _finding(record, "retrieval_incomplete_enumeration", "retrieval_failure", confirmed=True)
    return None


# ══════════════════════════════════════════════════════════════════
#  B그룹: 생성 실패 (Oracle 실패) — generation_*
# ══════════════════════════════════════════════════════════════════

def generation_hop_binding_error(record: EvalRecord) -> Optional[Finding]:
    """
    멀티홉: 각 hop 사실은 맞으나 결합이 틀림(faithfulness 높음).
    확정: faithfulness 높음.
    """
    if not _generation_failed(record):
        return None
    faith = _faith_oracle(record)
    if _is_multi_hop(record) and faith is not None and faith >= RAGAS_FAITHFULNESS_MIN:
        return _finding(record, "generation_hop_binding_error", "generation_failure", confirmed=True)
    return None


def generation_hallucination(record: EvalRecord) -> Optional[Finding]:
    """
    정답 context가 있는데 지어냄.
    확정: faithfulness 낮음.
    """
    if not _generation_failed(record):
        return None
    faith = _faith_oracle(record)
    if faith is not None and faith < RAGAS_FAITHFULNESS_MIN:
        return _finding(record, "generation_hallucination", "generation_failure", confirmed=True)
    return None


def generation_partial_answer(record: EvalRecord) -> Optional[Finding]:
    """
    정답 context가 있는데 일부 요소·조건 누락.
    확정: relevancy 낮음.
    """
    if not _generation_failed(record):
        return None
    rel = _rel_oracle(record)
    if rel is not None and rel < RAGAS_RESPONSE_RELEVANCY_MIN:
        return _finding(record, "generation_partial_answer", "generation_failure", confirmed=True)
    return None


def generation_failure(record: EvalRecord) -> Optional[Finding]:
    """생성 실패 예비 롤업. 생성이 실패(oracle 실패/무응답 위반)했는데 위 세분화 라벨이
    확정 못 했을 때(RAGAS 미실행 등) 예비로 낸다. 생성 슬롯의 마지막 후보."""
    if _generation_failed(record):
        return _finding(record, "generation_failure", "generation_failure", confirmed=False)
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
    context가 너무 길어 잡음·과부하로 품질 저하.
    확정: 축소 재실행 시 회복(tier4).
    예비: 없음
    """
    if not _context_applicable(record):
        return None
    if _context_shorten_helps(record) is True:
        return _finding(record, "too_long_context", "retrieval_failure", confirmed=True)
    return None


def lost_in_the_middle(record: EvalRecord) -> Optional[Finding]:
    """
    청크가 긴 context 중간이라 LLM이 참조 못함.
    확정: gold 앞배치 재실행 시 회복(tier4).
    예비: 없음
    """
    if not _context_applicable(record):
        return None
    if _gold_front_helps(record) is True:
        return _finding(record, "lost_in_the_middle", "retrieval_failure", confirmed=True)
    return None


def context_noise_interference(record: EvalRecord) -> Optional[Finding]:
    """
    비-gold 청크의 상충 정보에 이끌림.
    확정: 노이즈 제거 재실행 시 회복(tier4).
    예비: 확정 전엔 항상 예비.
    """
    if not _context_applicable(record):
        return None
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
    """
    정답셋 자체 오류/모호(충실도·관련성 모두 고득점인데 gold만 불일치) — 실제 트랙(컨텍스트 계열).
    확정(자동): faith·rel 둘 다 측정 고득점(tier3). [진짜 확정은 사람 검수.]
    """
    if not _context_applicable(record):
        return None
    if _both_high(_faith(record), _rel(record)):
        return _finding(record, "bad_gold_answer", "gap", confirmed=True)
    return None


def bad_gold_answer_oracle(record: EvalRecord) -> Optional[Finding]:
    """
    bad_gold_answer 의 오라클 트랙 버전(생성 실패 계열).
    라벨은 동일('bad_gold_answer').
    """
    if not _generation_failed(record):
        return None
    if _both_high(_faith_oracle(record), _rel_oracle(record)):
        return _finding(record, "bad_gold_answer", "gap", confirmed=True)
    return None


def corpus_gap(record: EvalRecord) -> Optional[Finding]:
    """
    필요한 자료가 코퍼스에 없음(단일홉).
    확정: 코퍼스에 gold 없음(tier2).
    예비: 없음
    """
    if not _retrieval_failed(record):
        return None
    if _gold_in_corpus(record) is False and not _is_multi_hop(record):
        return _finding(record, "corpus_gap", "gap", confirmed=True)
    return None


def corpus_gap_partial_hop(record: EvalRecord) -> Optional[Finding]:
    """
    멀티홉 중 일부 hop 근거만 코퍼스에 없음.
    확정: 코퍼스에 gold 없음(tier2).
    예비: 없음"""
    if not _retrieval_failed(record):
        return None
    if _gold_in_corpus(record) is False and _is_multi_hop(record):
        return _finding(record, "corpus_gap_partial_hop", "gap", confirmed=True)
    return None


# ══════════════════════════════════════════════════════════════════
#  원인 슬롯 (브랜치 없음 — 모든 슬롯을 전부 시도)
#    각 라벨이 자기 싼 전제(recall/f1/oracle)로 self-scope 하므로, 안 맞는 슬롯은 자연히 빈다.
#    슬롯당 _pick 으로 '한 원인' 채택(확정 우선). corpus_gap 은 추가로 붙는다(additive).
#    generation_failure(예비 롤업)는 생성 슬롯 맨 뒤 후보.
# ══════════════════════════════════════════════════════════════════

_RETRIEVAL_CAUSE = (
    retrieval_incomplete_enumeration, retrieval_missing_bridge_dependency,
    retrieval_low_rank, retrieval_lexical_mismatch, retrieval_semantic_mismatch, retrieval_missing_gold,
)
_GENERATION_CAUSE = (
    bad_gold_answer_oracle, generation_hop_binding_error, generation_hallucination,
    generation_partial_answer, generation_failure,
)
_CONTEXT_CAUSE = (
    bad_gold_answer, too_long_context, lost_in_the_middle, context_noise_interference,
)


# ── 판정 콤비네이터 ──────────────────────────────────────────────

def _pick(record: EvalRecord, funcs) -> Optional[Finding]:
    """funcs 중 '한 원인'을 고른다: 확정된(confirmed) 첫 라벨 우선, 없으면 첫 매치(예비), 없으면 None.
    라벨이 자기 전제·신호로 self-scope 하므로, 브랜치 없이 모든 슬롯에 던져도 안 맞으면 None 이 난다."""
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


def _collect(*items) -> list[Finding]:
    """None 을 걸러 리스트로 만드는 util."""
    return [f for f in items if f is not None]


def _dedup(findings: list[Finding]) -> list[Finding]:
    """혹시모를 duplicant를 삭제하는 util."""
    seen, out = set(), []
    for f in findings:
        key = f.label
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


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
# !!!!!!!!!!!!!!!!!!!!!!!!!!!! optimize와 논의 필요
_CRITICAL_LABELS = {
    "retrieval_semantic_mismatch", "retrieval_missing_gold",
    "generation_hallucination", "corpus_gap", "corpus_gap_partial_hop",
}
def _severity_of(label: str) -> str:
    if label in _CRITICAL_LABELS:
        return "critical"
    return "warning"


# gold 순위(top-N 재검색)가 top_k 근거값 계산에 쓰이는 라벨.
# 이 라벨의 Finding 에만 gold_ranks 를 실어 planner(_GROUNDED_VALUES)가 개수 대신
# 순위로 top_k 를 산정하게 한다. tier2(STANDARD+) 에서만 순위가 나오고, 그 아래
# 모드에선 gold_ranks 가 안 실려 planner 가 개수 근사로 폴백한다.
# retrieval_low_rank 는 제외: 정석 처방이 리랭커라 top_k 근거값을 planner 가 쓰지
#   않는다(planner._GROUNDED_VALUES 참고). 순위를 실어봤자 안 쓰이는 무효 데이터다.
_RANK_LABELS = {
    "retrieval_incomplete_enumeration",
    "retrieval_missing_gold",
}


def _finding(record: EvalRecord, label: str, ftype: str, confirmed: bool) -> Finding:
    """라벨 함수 공통 Finding 생성기.

    confirmed 는 라벨 함수가 명시한다 — '확정 신호가 실제로 발동했는지'.
      True  = 확정 신호(그 자원)가 발동해 확정.
      False = 예비 신호로 의심만(확정 자원 미실행) → 상위 모드에서 확정.
    (mode>=tier 자동판정 아님 — 자원 미실행/미측정이면 예비.)
    """
    probe = record.probe
    group = _group_of(label, ftype)
    prefix = "" if confirmed else "[예비] "
    metadata: dict = {"group": group}
    if label in _RANK_LABELS:
        # planner 가 top_k 근거값을 계산할 원시 순위(집계는 planner 소관).
        # None(모드·자원 미충족)이면 싣지 않아 planner 가 개수 폴백을 쓰게 둔다.
        ranks = _gold_ranks(record)
        if ranks:
            metadata["gold_ranks"] = ranks
    return Finding(
        finding_id=f"{probe.probe_id}:{label}",
        type=ftype,
        severity=_severity_of(label),
        description=f"{prefix}[{group}그룹] {label}",
        label=label,
        confirmed=confirmed,
        affected_chunks=list(probe.gold_chunk_ids),
        affected_probes=[probe.probe_id],
        metadata=metadata,
    )


# ── 메인 ─────────────────────────────────────────────────────────

# 처방 순서: D → A → C → B, 그다음 심각도순.
_GROUP_ORDER = {"D": 0, "A": 1, "C": 2, "B": 3}
_SEV_ORDER = {"critical": 0, "warning": 1, "info": 2}

def diagnose(record: EvalRecord, mode: Optional[int] = None) -> list[Finding]:
    """
    metric을 계산하고, diagnosis가 필요없는 경우 return한다 (기존 성공 브랜치)
    이후 모든 라벨에 대해 검사한다.
    라벨은 각
    """
    set_mode(mode if mode is not None else resolve_mode())

    _compute_metrics(record)      # 지표(recall/f1/oracle_f1) 계산 → record 반영

    if _no_diagnosis(record):     # 정답셋 없음 / 올바른 무응답 / 성공
        return []

    findings = _collect(
        corpus_gap(record),                    # D 데이터 (additive)
        corpus_gap_partial_hop(record),        # D 데이터 (additive)
        _pick(record, _RETRIEVAL_CAUSE),       # A 검색 원인 1개
        _pick(record, _CONTEXT_CAUSE),         # C 컨텍스트 원인 1개
        _pick(record, _GENERATION_CAUSE),      # B 생성 원인 1개 (generation_failure 롤업 포함)
    )

    findings = _dedup(findings)
    findings.sort(key=lambda f: (
        _GROUP_ORDER.get(f.metadata.get("group"), 9),
        _SEV_ORDER.get(f.severity, 9),
    ))
    return findings
