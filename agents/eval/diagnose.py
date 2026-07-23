"""
agents/eval/diagnose.py
STEP4: 원인 판정 (Finding 생성)

구조 원칙:
  1. 라벨마다 판정 함수 1개: 각 함수는 자기 라벨의 '판별 신호(원인)'를 검사해
     맞으면 Finding, 아니면 None 을 돌려준다. 
  2. diagnose() 가 모든 원인 슬롯을 시도
     슬롯당 _pick 으로 '한 원인' 채택.
     측정값은 전부 metrics_* 모듈에서 lazy·memoize, 임계값 판정은 여기가 담당.
  3. 기본 지표와 RAGAS는 판정 전에 항상 측정한다.

Finding에 label들을 담고 다음단계로 진행한다.

라벨 그룹: A 검색실패 / B 생성실패 / C context구조 / D 데이터.
Finding.type 필드에 라벨 그룹을 담고, Finding.label 필드에 세분화 라벨을 담는다.

진단 모드:
  - 1[FAST], 2[STANDARD], 3[DEEP] (삭제됨)4[FULL]
  - diagnose() 진입 시 metrics_common.set_mode 로 현재 실행 모드를 설정한다.
  - 지정된 진단 모드 이하 tier인 진단만 수행할 수 있다.
  - 파이프라인 재실행(tier4)로 context 원인을 확정하던 경로는 제거됨. 관련 라벨은 예비 Finding 으로만 남긴다.

측정값·진단 자원(_ctx)·모드 상태는 tier 별 측정 파일에 존재한다:
  metrics_common(인프라) / metrics_basic(tier1) / metrics_search(tier2) / metrics_ragas(tier3).
여기서는 임계값 전제 판정, 라벨 함수, 조립(diagnose)을 담당한다.
"""
from __future__ import annotations

from typing import Optional

from core.schema import Finding
from agents.eval.types import (
    DEFAULT_TOP_K, EvalRecord, Mode, resolve_mode,
    F1_PASS_THRESHOLD, ANSWER_CORRECTNESS_MIN,
    RAGAS_FAITHFULNESS_MIN, RAGAS_RESPONSE_RELEVANCY_MIN,
)
from agents.eval.metrics_common import set_mode, set_context
from agents.eval.metrics_basic import (            # tier1
    is_abstention, _compute_metrics, _gold_span_boundary_analysis,
)
from agents.eval.metrics_search import (           # tier2
    _gold_ranks, _bm25_hits_gold, _gold_in_corpus,
)
from agents.eval.metrics_ragas import (            # tier3
    _compute_ragas_real, _compute_ragas_oracle,
    _faith, _faith_oracle, _rel, _rel_oracle,
)


# ══════════════════════════════════════════════════════════════════
#  임계값 판정 함수
# ══════════════════════════════════════════════════════════════════

def _recall_ok(record: EvalRecord) -> bool:
    """검색 성공"""
    return record.recall_at_k >= 1

def _f1_ok(record: EvalRecord) -> bool:
    """
    Response 정답 판정
    1. lexical(f1_score) 임계값 이상
    2-1. llm이 사용 가능하다면, ragas answer_correctness도 임계값 이상이어야함.
    """
    if record.f1_score < F1_PASS_THRESHOLD:
        return False
    
    ac = record.ragas_answer_correctness
    if ac is None:
        return True
    return ac >= ANSWER_CORRECTNESS_MIN

def _oracle_ok(record: EvalRecord) -> bool:
    """"
    Oracle 정답 판정 (위와 동일)
    """
    if record.oracle_f1 < F1_PASS_THRESHOLD:
        return False
    
    ac = record.oracle_ragas_answer_correctness
    if ac is None:
        return True
    return ac >= ANSWER_CORRECTNESS_MIN

def _is_success(record: EvalRecord) -> Optional[bool]:
    """probe 단위 성공/실패 판정 — recall + answer_match(tier1) + RAGAS answer_correctness(tier3).

    True  = 성공 (검색·정답 모두 통과 / 무응답 기대인데 올바르게 기권)
    False = 실패
    None  = 판정 불가 (대조할 정답셋이 없음)

    recall 은 정답셋이 있는 경로에서만 본다 — 무응답 기대 probe 는 gold 가 없어
    recall_at_k = -1 이므로, 앞에서 recall 을 보면 올바른 기권까지 실패가 된다.
    """
    if record.probe.answer_exists is False:
        return is_abstention(record.generated_answer)   # 무응답 기대 → 올바른 기권이 성공(recall 무관)
    if not record.probe.ground_truth:
        return None                                     # 대조할 정답 없음 → 판정 불가
    # 검색 성공(recall=1) + 정답 일치(answer_match, DEEP 이면 ragas answer_correctness 로 강등)
    return _recall_ok(record) and _f1_ok(record)

def _is_multi_hop(record: EvalRecord) -> bool:
    """멀티홉 질문 여부(probe.qtype). bridge / hop_binding / corpus_gap_partial_hop 판별용."""
    return record.probe.qtype in ("bridge", "comparison", "aggregation")

def _enumeration_cache(record: EvalRecord) -> bool:
    """gold 개수가 top-k(=검색 결과 수)에 근접/초과 → 나열형 누락. incomplete_enumeration 용."""
    k = len(record.retrieved_chunk_ids) or DEFAULT_TOP_K
    gold_n = len(record.probe.gold_chunk_ids)
    return gold_n >= 2 and gold_n >= int(k * 0.8)

# ══════════════════════════════════════════════════════════════════
#  A그룹: 검색 실패 (Oracle 통과) — retrieval_*
# ══════════════════════════════════════════════════════════════════

def retrieval_low_rank(record: EvalRecord) -> Optional[Finding]:
    """
    gold가 top-N 후보엔 있으나 순위가 낮아 top-k 밖.
    확정: top-N 재검색에서 gold 발견(tier2).
    """
    ranks = _gold_ranks(record)
    if ranks is None:
        return None
    missed = set(record.probe.gold_chunk_ids) - set(record.retrieved_chunk_ids)
    if missed and any(ranks.get(g) is not None for g in missed):
        return _finding(
            record, "retrieval_low_rank", "retrieval_failure", confirmed=True,
            reason=f"gold_in_wider_candidates=True, recall@k={_v(record.recall_at_k)}",
        )
    return None


def retrieval_lexical_mismatch(record: EvalRecord) -> Optional[Finding]:
    """
    dense는 놓쳤으나 BM25로 잡히는 단어 불일치.
    확정: BM25 가 gold 를 잡음(tier2).
    """
    if _bm25_hits_gold(record) is True:
        return _finding(
            record, "retrieval_lexical_mismatch", "retrieval_failure", confirmed=True,
            reason=f"bm25_hits_gold=True, recall@k={_v(record.recall_at_k)}",
        )
    return None


def retrieval_semantic_mismatch(record: EvalRecord) -> Optional[Finding]:
    """
    dense·BM25 모두 놓친 의미 연결 실패. (단 gold 가 코퍼스엔 있을 때만 — 없으면 corpus_gap)
    확정: BM25 도 gold 를 못 잡음 + gold 는 코퍼스에 존재(tier2).
    """
    in_corpus = _gold_in_corpus(record)
    if _bm25_hits_gold(record) is False and in_corpus is not False:
        return _finding(
            record, "retrieval_semantic_mismatch", "retrieval_failure", confirmed=True,
            reason=f"bm25_hits_gold=False, gold_in_corpus={_v(in_corpus)}, "
                   f"recall@k={_v(record.recall_at_k)}",
        )
    return None


def retrieval_missing_gold(record: EvalRecord) -> Optional[Finding]:
    """
    gold는 corpus에 있으나 top-k에 없음.
    확정: 코퍼스에 gold 존재(tier2).
    """
    in_corpus = _gold_in_corpus(record)
    if in_corpus is True:
        return _finding(
            record, "retrieval_missing_gold", "retrieval_failure", confirmed=True,
            reason=f"gold_in_corpus=True, recall@k={_v(record.recall_at_k)}",
        )
    if in_corpus is None:
        return _finding(
            record, "retrieval_missing_gold", "retrieval_failure", confirmed=False,
            reason=f"gold_in_corpus=-, recall@k={_v(record.recall_at_k)}",
        )
    if in_corpus is False:
        return None


def chunking_context_mismatch(record: EvalRecord) -> Optional[Finding]:
    """정답 근거가 현재 청크 경계에 나뉘어 한 청크에 온전히 없음을 판정한다.

    gold span과 현재 청크의 원문 좌표만 비교해 경계 분할 후보를 찾는다(저비용).
    검색 실패든 답변 실패든, 경계 분할은 '후보 원인'일 뿐 좌표의 동시 발생만으로
    인과를 확정하지 않는다. 실제 원인인지는 optimize 가 청킹 파라미터를 바꿔
    재실행하며 검증하므로, 여기서는 항상 예비 Finding 으로만 남긴다.
    """

    analysis = _gold_span_boundary_analysis(record)
    if not isinstance(analysis, dict) or analysis.get("boundary_split_count", 0) <= 0:
        return None
    if _recall_ok(record) and not _context_failed(record):
        return None
    finding = _finding(
        record, "chunking_context_mismatch", "retrieval_failure", confirmed=False,
        reason=f"boundary_split={analysis.get('boundary_split_count')}, "
               f"recall@k={_v(record.recall_at_k)}, f1={_v(record.f1_score)}",
    )
    finding.metadata["boundary_analysis"] = dict(analysis)
    return finding


def retrieval_missing_bridge_dependency(record: EvalRecord) -> Optional[Finding]:
    """
    멀티홉 연쇄형: 2번째 hop 근거가 1번째 hop에 의존.
    예비: 멀티홉 + 실제 검색 실패(gold 있는데 일부 미검색, 0<=recall<1). 멀티홉이라는 것만으론
    bridge 의존이라 단정 못 한다(low_rank·lexical 등 다른 원인일 수도). 실제 bridge 인지는
    decompose 재검색으로 회복되는지 봐야 하고(제거된 tier4), 그 확정은 optimize 가 위임받는다 → 예비.
    """
    if not _is_multi_hop(record) or not (0 <= record.recall_at_k < 1):
        return None

    return _finding(
        record, "retrieval_missing_bridge_dependency", "retrieval_failure", confirmed=False,
        reason=f"qtype={record.probe.qtype}, recall@k={_v(record.recall_at_k)}",
    )


def retrieval_incomplete_enumeration(record: EvalRecord) -> Optional[Finding]:
    """
    나열형: 필요한 근거 개수 가변인데 top-k 고정이라 누락.
    확정: gold수 vs top-k 순수 규칙(tier1) — 바로 확정.?????????
    """

    if _enumeration_cache(record):
        return _finding(
            record, "retrieval_incomplete_enumeration", "retrieval_failure", confirmed=True,
            reason=f"gold={len(record.probe.gold_chunk_ids)}, top_k={len(record.retrieved_chunk_ids)}, "
                   f"recall@k={_v(record.recall_at_k)}",
        )
    return None

def retrieval_failure(record: EvalRecord) -> Optional[Finding]:
    """검색 실패 롤업"""
    return _finding(
        record, "retrieval_failure", "retrieval_failure", confirmed=False,
        reason=f"oracle_f1={_v(record.oracle_f1)}, f1={_v(record.f1_score)}, "
                f"faithfulness={_v(_faith_oracle(record))}, relevancy={_v(_rel_oracle(record))}",
    )

# ══════════════════════════════════════════════════════════════════
#  B그룹: 생성 실패 (Oracle 실패) — generation_*
# ══════════════════════════════════════════════════════════════════

def _generation_failed(record: EvalRecord) -> bool:
    """생성 실패 전제(B 공통): gold 컨텍스트로도 답이 틀림, 또는 무응답인데 답을 지어냄."""
    if record.oracle_answer is not None and not _oracle_ok(record):
        return True
    if record.probe.answer_exists is False and not is_abstention(record.generated_answer):
        return True
    return False


def generation_no_abstention(record: EvalRecord) -> Optional[Finding]:
    """
    무응답 기대(answer_exists=False) probe인데 기권하지 않고 답을 지어냄.
    확정: answer_exists=False + is_abstention 아님.
    """
    if record.probe.answer_exists is False and not is_abstention(record.generated_answer):
        return _finding(
            record, "generation_no_abstention", "generation_failure", confirmed=True,
            reason=f"answer_exists=False, 기권 아님(f1={_v(record.f1_score)})",
        )
    return None


def generation_hallucination(record: EvalRecord) -> Optional[Finding]:
    """
    정답 context가 있는데 지어냄.
    확정: faithfulness 낮음.
    """
    faith = _faith_oracle(record)
    if faith is not None and faith < RAGAS_FAITHFULNESS_MIN:
        return _finding(
            record, "generation_hallucination", "generation_failure", confirmed=True,
            reason=f"faithfulness={_v(faith)}<{RAGAS_FAITHFULNESS_MIN}, oracle_f1={_v(record.oracle_f1)}",
        )
    return None

def generation_hop_binding_error(record: EvalRecord) -> Optional[Finding]:
    """
    멀티홉: 각 hop 사실은 맞으나 결합이 틀림(faithfulness 높음).
    확정: faithfulness 높음.
    """
    faith = _faith_oracle(record)
    if _is_multi_hop(record) and faith is not None and faith >= RAGAS_FAITHFULNESS_MIN:
        return _finding(
            record, "generation_hop_binding_error", "generation_failure", confirmed=True,
            reason=f"faithfulness={_v(faith)}>={RAGAS_FAITHFULNESS_MIN}, qtype={record.probe.qtype}, "
                   f"oracle_f1={_v(record.oracle_f1)}",
        )
    return None

def generation_partial_answer(record: EvalRecord) -> Optional[Finding]:
    """
    정답 context가 있는데 일부 요소·조건 누락.
    확정: relevancy 낮음.
    """
    rel = _rel_oracle(record)
    if rel is not None and rel < RAGAS_RESPONSE_RELEVANCY_MIN:
        return _finding(
            record, "generation_partial_answer", "generation_failure", confirmed=True,
            reason=f"response_relevancy={_v(rel)}<{RAGAS_RESPONSE_RELEVANCY_MIN}, "
                   f"oracle_f1={_v(record.oracle_f1)}",
        )
    return None


def generation_failure(record: EvalRecord) -> Optional[Finding]:
    """생성 실패 롤업"""
    return _finding(
        record, "generation_failure", "generation_failure", confirmed=False,
        reason=f"oracle_f1={_v(record.oracle_f1)}, f1={_v(record.f1_score)}, "
                f"faithfulness={_v(_faith_oracle(record))}, relevancy={_v(_rel_oracle(record))}",
    )


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

def _context_failed(record: EvalRecord) -> bool:
    """컨텍스트 구조 문제(C) 전제: 검색 성공(recall=1)·생성 가능(oracle 통과)인데 실제 답만 틀림."""
    return _recall_ok(record) and _oracle_ok(record) and not _f1_ok(record)

def too_long_context(record: EvalRecord) -> Optional[Finding]:
    """
    context가 너무 길어 잡음·과부하로 품질 저하.s
    tier4(축소 재실행) 확정 신호가 유일한 발동 경로였으나 optimize 재실행으로 대체됨.
    # TODO(tier4 제거): 예비 발동 조건(저비용 휴리스틱) 재설계 전까지 dormant.
    """
    return None


def lost_in_the_middle(record: EvalRecord) -> Optional[Finding]:
    """
    청크가 긴 context 중간이라 LLM이 참조 못함.
    tier4(gold 앞배치 재실행) 확정 신호가 유일한 발동 경로였으나 optimize 재실행으로 대체됨.
    # TODO(tier4 제거): 예비 발동 조건(저비용 휴리스틱) 재설계 전까지 dormant.
    """
    return None


def context_noise_interference(record: EvalRecord) -> Optional[Finding]:
    """
    비-gold 청크의 상충 정보에 이끌림.
    예비: recall=1·oracle 통과인데 실제 답만 틀림. 노이즈 제거로 회복되는지는
    optimize 가 top_k 축소·리랭커를 적용해 재실행하며 검증한다.
    """
    return None

def context_failure(record: EvalRecord) -> Optional[Finding]:
    """콘텍스트 실패 롤업"""
    return _finding(
        record, "context_failure", "context_failure", confirmed=False,
        reason=f"oracle_f1={_v(record.oracle_f1)}, f1={_v(record.f1_score)}, "
                f"faithfulness={_v(_faith_oracle(record))}, relevancy={_v(_rel_oracle(record))}",
    )


# ══════════════════════════════════════════════════════════════════
#  D그룹: 데이터 문제 (파이프라인 튜닝 불가)
# ══════════════════════════════════════════════════════════════════

def bad_gold_answer(record: EvalRecord) -> Optional[Finding]:
    """
    정답셋 자체 오류/모호
    콘텍스트 실패 계열 (C 그룹에 함께 있음)
    확정(자동): faith·rel 둘 다 측정 고득점(tier3). [진짜 확정은 사람 검수.]
    """
    faith, rel = _faith(record), _rel(record)
    if (faith is not None and faith >= RAGAS_FAITHFULNESS_MIN
        and rel is not None and rel >= RAGAS_RESPONSE_RELEVANCY_MIN):
        return _finding(
            record, "bad_gold_answer", "gap", confirmed=True,
            reason=f"faithfulness={_v(faith)}, response_relevancy={_v(rel)}, f1={_v(record.f1_score)}",
        )
    return None

def bad_gold_answer_oracle(record: EvalRecord) -> Optional[Finding]:
    """
    bad_gold_answer 의 오라클 트랙 버전
    생성 실패 계열 (B 그룹에 함께 있음)
    라벨은 동일('bad_gold_answer').
    """
    faith, rel = _faith_oracle(record), _rel_oracle(record)
    if (faith is not None and faith >= RAGAS_FAITHFULNESS_MIN
        and rel is not None and rel >= RAGAS_RESPONSE_RELEVANCY_MIN):
        return _finding(
            record, "bad_gold_answer", "gap", confirmed=True,
            reason=f"faithfulness(oracle)={_v(faith)}, response_relevancy(oracle)={_v(rel)}, "
                   f"oracle_f1={_v(record.oracle_f1)}",
        )
    return None


def corpus_gap(record: EvalRecord) -> Optional[Finding]:
    """
    필요한 자료가 코퍼스에 없음(단일홉).
    확정: 코퍼스에 gold 없음(tier2).
    """
    if _gold_in_corpus(record) is False and not _is_multi_hop(record):
        return _finding(
            record, "corpus_gap", "gap", confirmed=True,
            reason=f"gold_in_corpus=False, qtype={record.probe.qtype}, recall@k={_v(record.recall_at_k)}",
        )
    return None


def corpus_gap_partial_hop(record: EvalRecord) -> Optional[Finding]:
    """
    멀티홉 중 일부 hop 근거만 코퍼스에 없음.
    확정: 코퍼스에 gold 없음(tier2).
    """
    if _gold_in_corpus(record) is False and _is_multi_hop(record):
        return _finding(
            record, "corpus_gap_partial_hop", "gap", confirmed=True,
            reason=f"gold_in_corpus=False, qtype={record.probe.qtype}, recall@k={_v(record.recall_at_k)}",
        )
    return None


# ══════════════════════════════════════════════════════════════════
#  원인 슬롯 (브랜치 없음 — 모든 슬롯을 전부 시도)
#    각 라벨이 자기 싼 전제(recall/f1/oracle)로 self-scope 하므로, 안 맞는 슬롯은 자연히 빈다.
#    슬롯당 _pick 으로 '한 원인' 채택(확정 우선). corpus_gap 은 추가로 붙는다(additive).
#    generation_failure(예비 롤업)는 생성 슬롯 맨 뒤 후보.
# ══════════════════════════════════════════════════════════════════

_RETRIEVAL_CAUSE = (
    chunking_context_mismatch, retrieval_incomplete_enumeration, retrieval_missing_bridge_dependency,
    retrieval_low_rank, retrieval_lexical_mismatch, retrieval_semantic_mismatch, retrieval_missing_gold,
    retrieval_failure
)
_GENERATION_CAUSE = (
    generation_no_abstention,bad_gold_answer_oracle, generation_hop_binding_error, 
    generation_hallucination, generation_partial_answer,
    generation_failure,
)
# chunking_context_mismatch 는 A·C 양쪽에 등록한다 — 경계 분할은 '검색이 gold 를 통째로
# 못 가져옴'(A)으로도, '검색은 됐는데 잘린 근거로 답이 틀림'(C)으로도 나타난다.
# A 슬롯에만 두면 recall=1 인 경계 분할 실패에서 도달 자체가 불가능하다(_dedup 이 중복 제거).
_CONTEXT_CAUSE = (
    bad_gold_answer, chunking_context_mismatch,
    too_long_context, lost_in_the_middle, context_noise_interference,
    context_failure
)


# ── 판정 콤비네이터 ──────────────────────────────────────────────

def _pick(record: EvalRecord, funcs) -> Optional[Finding]:
    """
    라벨 함수 중 하나 선택 
    1. 확정된 첫 라벨
    2. 없으면 예비 첫 라벨
    3. 없으면 None.
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
    if label == "chunking_context_mismatch":
        return "A"
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


# gold 순위를 함께 저장해야하는 라벨들.
_RANK_LABELS = {
    "retrieval_incomplete_enumeration",
    "retrieval_missing_gold",
}


def _v(x) -> str:
    """reason 문자열용 값 포맷(float 은 소수 2자리, None 은 '-')."""
    if x is None:
        return "-"
    return f"{x:.2f}" if isinstance(x, float) else str(x)


def _finding(record: EvalRecord, label: str, ftype: str, confirmed: bool, reason: str = "") -> Finding:
    """ 라벨 함수 공통 Finding 생성기. """
    probe = record.probe
    group = _group_of(label, ftype)
    prefix = "" if confirmed else "[예비] "
    metadata: dict = {"group": group, "reason": reason}
    if label in _RANK_LABELS:
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
    지표(STEP3-1)와 RAGAS(STEP3-2)를 먼저 전부 측정하고 이후 모든 라벨에 대해 검사한다.

    라벨은 세 분류로 나뉘어, 해당되는 라벨만 검사된다. 
    """
    set_mode(mode if mode is not None else resolve_mode())

    # metric, ragas 진단
    _compute_metrics(record)      # 지표(recall/f1/oracle_f1) 계산 → record 반영
    _compute_ragas_real(record)   # 실제 트랙 RAGAS — 강등 판정 + 리포트 평균용 (DEEP 이상)

    # 성공/실패 판정 — 실패(False)일 때만 원인을 찾는다.
    # None(판정 불가)은 통과로 묶는다: 대조할 정답이 없어 실패라 단정할 근거가 없다.
    # 이 게이트로 '성공 ⇒ findings 없음' 이 규약이 아니라 보장이 된다.
    if _is_success(record) is not False:
        return []

    # 오라클 트랙 RAGAS — 소비처가 B그룹 라벨·_oracle_ok 뿐이라 실패 probe 에서만 지불한다.
    _compute_ragas_oracle(record)

    # 추가 진단
    findings = []
    if 0 <= record.recall_at_k < 1:                     # A: 검색 실패 (gold 있는데 일부 미검색)
        findings.append(_pick(record, _RETRIEVAL_CAUSE))
        findings.append(corpus_gap(record))             # D: 코퍼스에 gold 없음 (additive)
        findings.append(corpus_gap_partial_hop(record))
    if _generation_failed(record):                      # B: 생성 실패
        findings.append(_pick(record, _GENERATION_CAUSE))
    if _context_failed(record):                         # C: context 구조
        findings.append(_pick(record, _CONTEXT_CAUSE))

    findings = _dedup(_collect(*findings))
    findings.sort(key=lambda f: (
        _GROUP_ORDER.get(f.metadata.get("group"), 9),
        _SEV_ORDER.get(f.severity, 9),
    ))
    return findings
