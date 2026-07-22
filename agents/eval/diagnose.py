"""
agents/eval/diagnose.py
STEP4: 원인 판정 (Finding 생성)

구조 원칙:
  1. 라벨마다 판정 함수 1개: 각 함수는 자기 라벨의 '판별 신호(원인)'를 검사해
     맞으면 Finding, 아니면 None 을 돌려준다. 
  2. diagnose() 가 모든 원인 슬롯을 시도
     슬롯당 _pick 으로 '한 원인' 채택.
     측정값은 전부 metrics_* 모듈에서 lazy·memoize, 임계값 판정(전제·강등)은 여기가 담당.
     기본 지표(_compute_metrics)와 RAGAS(_compute_ragas)는 판정 전에 항상 측정한다.

Finding에 label을 담고 다음단계로 진행된다.

라벨 그룹: A 검색실패 / B 생성실패 / C context구조 / D 데이터.
Finding.type 필드에 라벨 그룹을 담고, Finding.label 필드에 세분화 라벨을 담는다.

진단 모드:
  - 1[FAST], 2[STANDARD], 3[DEEP]
  - diagnose() 진입 시 metrics_common.set_mode 로 현재 실행 모드를 설정한다.
  - 지정된 진단 모드 이하 tier 까지만 확정할 수 있다(그 위 tier 라벨은 예비로 지정할수 있다).
  - 각 라벨은 비용이 적은 예비 신호(없는 경우가 많음)를 가질수 있다.
  - 생성 원인(B)은 전부 RAGAS(DEEP) 의존 → DEEP 미만이면 예비 generation_failure 로 롤업한다.
  - 파이프라인 재실행(tier4)로 context 원인을 확정하던 경로는 제거됨 — 그 검증은 optimize 가
    config 를 바꿔 재실행하며 수행한다. 관련 라벨은 예비 Finding 으로만 남긴다.

측정값·진단 자원(_ctx)·모드 상태는 metrics_common / metrics_basic / metrics_ragas / metrics_chunk 에
존재한다. 여기서는 임계값 전제 판정, 라벨 함수, 조립(diagnose)을 담당한다.
"""
from __future__ import annotations

from typing import Optional

from core.schema import Finding
from agents.eval.types import (
    EvalRecord, Mode, resolve_mode,
    F1_PASS_THRESHOLD, ANSWER_CORRECTNESS_MIN,
    RAGAS_FAITHFULNESS_MIN, RAGAS_RESPONSE_RELEVANCY_MIN,
)
from agents.eval.metrics_common import set_mode, set_context
from agents.eval.metrics_basic import (
    is_abstention, _compute_metrics,
    _is_multi_hop, _enumeration_cache,
    _gold_in_wider_candidates, _gold_ranks, _bm25_hits_gold, _gold_in_corpus,
)
from agents.eval.metrics_ragas import (
    _compute_ragas, _faith, _faith_oracle, _rel, _rel_oracle,
    _answer_correctness_value,
)
from agents.eval.metrics_chunk import _gold_span_boundary_analysis


# ══════════════════════════════════════════════════════════════════
#  전제 판정 (측정값 + 임계값 → 성공/실패·슬롯 self-scope)
#    측정은 metrics_* 가, 임계값 판정은 여기가 담당한다.
#    성공/실패는 recall(검색) + answer_match(정답 일치)로 결정한다. DEEP+ 에서는 RAGAS
#    answer_correctness(답변↔gold 비교)가 낮으면 lexical 통과를 '강등'해 근접 오답을 거른다.
#    그 외 RAGAS(faithfulness/relevancy)는 실패 판정 뒤 생성 원인(B) 세분화에만 쓴다.
# ══════════════════════════════════════════════════════════════════

def _recall_ok(record: EvalRecord) -> bool:
    """gold 를 top-k 로 다 가져옴(recall==1). (recall==-1 = gold 없음 → 검색 성공으로 본다.)"""
    return record.recall_at_k >= 1


def _retrieval_failed(record: EvalRecord) -> bool:
    """gold 가 있는데 top-k 로 다 못 가져옴(0 <= recall < 1). 검색 원인(A) 공통 전제."""
    return 0 <= record.recall_at_k < 1


def _answer_ok(record: EvalRecord, track: str, lexical: float) -> bool:
    """정답 판정(레이어드): lexical answer_match 통과가 기본. 단 DEEP+ 에서 gold-비교
    측정(answer_correctness)이 잡히고 문턱 미만이면 '표면형만 비슷한 근접 오답'으로 보고
    강등한다(FAST/STANDARD 는 ac=None → 순수 lexical 판정, 결정적)."""
    if lexical < F1_PASS_THRESHOLD:
        return False
    ac = _answer_correctness_value(record, track)
    return not (ac is not None and ac < ANSWER_CORRECTNESS_MIN)


def _f1_ok(record: EvalRecord) -> bool:
    """실제 답이 정답과 일치. ground_truth 없으면 판정 불가 → False."""
    return bool(record.probe.ground_truth) and _answer_ok(record, "real", record.f1_score)


def _oracle_ok(record: EvalRecord) -> bool:
    """gold 컨텍스트로 생성한 답이 정답과 일치. oracle 답 없으면 False."""
    return record.oracle_answer is not None and _answer_ok(record, "oracle", record.oracle_f1)


def _generation_failed(record: EvalRecord) -> bool:
    """생성 실패 전제(B 공통): gold 컨텍스트로도 답이 틀림, 또는 무응답인데 답을 지어냄."""
    if record.oracle_answer is not None and not _oracle_ok(record):
        return True
    if record.probe.answer_exists is False and not is_abstention(record.generated_answer):
        return True
    return False


def _context_failed(record: EvalRecord) -> bool:
    """컨텍스트 구조 문제(C) 전제: 검색 성공(recall=1)·생성 가능(oracle 통과)인데 실제 답만 틀림."""
    return _recall_ok(record) and _oracle_ok(record) and not _f1_ok(record)


def _both_high(faith, rel) -> bool:
    """bad_gold_answer 판정용: 충실도·관련성이 모두 '측정되어' 임계값 이상.
    (DEEP 미만·미측정이면 faith/rel 이 None 이라 자연히 False.)"""
    return (faith is not None and faith >= RAGAS_FAITHFULNESS_MIN
            and rel is not None and rel >= RAGAS_RESPONSE_RELEVANCY_MIN)


def _no_diagnosis(record: EvalRecord) -> bool:
    """진단 불필요(성공): 올바른 무응답 / 정답셋 없음 / recall·answer_match 통과.

    무응답 기대(answer_exists=False) probe 는 정답셋(ground_truth)이 없더라도 먼저 판정한다 —
    올바르게 회피하면 통과, 답을 지어내면 진단 대상(B그룹 생성실패)이다. 이 순서를 뒤집으면
    'ground_truth 없음 → 무조건 통과'에 걸려 무응답 지어냄이 조용히 통과 처리된다."""
    if record.probe.answer_exists is False:
        return is_abstention(record.generated_answer)
    if not record.probe.ground_truth:
        return True
    return _recall_ok(record) and _f1_ok(record)


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
        return _finding(
            record, "retrieval_low_rank", "retrieval_failure", confirmed=True,
            reason=f"gold_in_wider_candidates=True, recall@k={_v(record.recall_at_k)}",
        )
    return None


def retrieval_lexical_mismatch(record: EvalRecord) -> Optional[Finding]:
    """
    dense는 놓쳤으나 BM25로 잡히는 단어 불일치.
    확정: BM25 가 gold 를 잡음(tier2).
    예비: 없음
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
    예비: 없음
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
    예비: recall<1(tier1, 코퍼스 확인 전).
    """
    if not _retrieval_failed(record):
        return None

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
    if not (_retrieval_failed(record) or _context_failed(record)):
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
    예비: 멀티홉+recall<1(tier1). decompose 재검색으로 회복되는지는 optimize 가
    query_rewrite:"decompose" 를 적용해 재실행하며 검증한다.
    """
    if not _retrieval_failed(record) or not _is_multi_hop(record):
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
    if not _retrieval_failed(record):
        return None
    if _enumeration_cache(record):
        return _finding(
            record, "retrieval_incomplete_enumeration", "retrieval_failure", confirmed=True,
            reason=f"gold={len(record.probe.gold_chunk_ids)}, top_k={len(record.retrieved_chunk_ids)}, "
                   f"recall@k={_v(record.recall_at_k)}",
        )
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
        return _finding(
            record, "generation_hop_binding_error", "generation_failure", confirmed=True,
            reason=f"faithfulness={_v(faith)}>={RAGAS_FAITHFULNESS_MIN}, qtype={record.probe.qtype}, "
                   f"oracle_f1={_v(record.oracle_f1)}",
        )
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
        return _finding(
            record, "generation_hallucination", "generation_failure", confirmed=True,
            reason=f"faithfulness={_v(faith)}<{RAGAS_FAITHFULNESS_MIN}, oracle_f1={_v(record.oracle_f1)}",
        )
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
        return _finding(
            record, "generation_partial_answer", "generation_failure", confirmed=True,
            reason=f"response_relevancy={_v(rel)}<{RAGAS_RESPONSE_RELEVANCY_MIN}, "
                   f"oracle_f1={_v(record.oracle_f1)}",
        )
    return None


def generation_failure(record: EvalRecord) -> Optional[Finding]:
    """생성 실패 예비 롤업. 생성이 실패(oracle 실패/무응답 위반)했는데 위 세분화 라벨이
    확정 못 했을 때(RAGAS 미실행 등) 예비로 낸다. 생성 슬롯의 마지막 후보."""
    if _generation_failed(record):
        return _finding(
            record, "generation_failure", "generation_failure", confirmed=False,
            reason=f"oracle_f1={_v(record.oracle_f1)}, f1={_v(record.f1_score)}, "
                   f"faithfulness={_v(_faith_oracle(record))}, relevancy={_v(_rel_oracle(record))}",
        )
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
    if not _context_failed(record):
        return None
    return _finding(
        record, "context_noise_interference", "retrieval_failure", confirmed=False,
        reason=f"f1={_v(record.f1_score)}, oracle_f1={_v(record.oracle_f1)}",
    )


# ══════════════════════════════════════════════════════════════════
#  D그룹: 데이터 문제 (파이프라인 튜닝 불가)
# ══════════════════════════════════════════════════════════════════

def bad_gold_answer(record: EvalRecord) -> Optional[Finding]:
    """
    정답셋 자체 오류/모호(충실도·관련성 모두 고득점인데 gold만 불일치) — 실제 트랙(컨텍스트 계열).
    확정(자동): faith·rel 둘 다 측정 고득점(tier3). [진짜 확정은 사람 검수.]
    """
    if not _context_failed(record):
        return None
    faith, rel = _faith(record), _rel(record)
    if _both_high(faith, rel):
        return _finding(
            record, "bad_gold_answer", "gap", confirmed=True,
            reason=f"faithfulness={_v(faith)}, response_relevancy={_v(rel)}, f1={_v(record.f1_score)}",
        )
    return None


def bad_gold_answer_oracle(record: EvalRecord) -> Optional[Finding]:
    """
    bad_gold_answer 의 오라클 트랙 버전(생성 실패 계열).
    라벨은 동일('bad_gold_answer').
    """
    if not _generation_failed(record):
        return None
    faith, rel = _faith_oracle(record), _rel_oracle(record)
    if _both_high(faith, rel):
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
    예비: 없음
    """
    if not _retrieval_failed(record):
        return None
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
    예비: 없음"""
    if not _retrieval_failed(record):
        return None
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
    chunking_context_mismatch, retrieval_incomplete_enumeration,
    retrieval_missing_bridge_dependency,
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
    지표(STEP3-1)와 RAGAS(STEP3-2)를 먼저 전부 측정하고, diagnosis가 필요없는 경우 return한다
    (기존 성공 브랜치). 이후 모든 라벨에 대해 검사한다.

    측정은 스킵하지 않는다 — 성공 probe 도 RAGAS 점수를 갖고 리포트 평균에 들어간다.
    (RAGAS 는 DEEP 이상에서만 실행되므로 그 미만 모드의 비용은 그대로다.)
    """
    set_mode(mode if mode is not None else resolve_mode())

    _compute_metrics(record)      # 지표(recall/f1/oracle_f1) 계산 → record 반영
    _compute_ragas(record)        # RAGAS(실제·오라클 트랙) 계산 → record 반영 (DEEP 이상)

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
