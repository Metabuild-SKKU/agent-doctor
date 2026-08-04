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
  - 1[FAST], 2[STANDARD], 3[DEEP] — DEEP 이 가장 깊다(tier4/FULL 은 없앴다).
  - diagnose() 진입 시 metrics_common.set_mode 로 현재 실행 모드를 설정한다.
  - 지정된 진단 모드 이하 tier인 진단만 수행할 수 있다.
  - 파이프라인 재실행으로 확정하던 경로는 Optimize 로 넘겼다. context 원인(C)은 실측 신호로
    DEEP 에서 확정한다 — reranker_low_precision 만 인과 미측정이라 예비로 남는다.

측정값·진단 자원(_ctx)·모드 상태는 tier 별 측정 파일에 존재한다:
  metrics_common(인프라) / metrics_basic(tier1) / metrics_search(tier2) / metrics_ragas(tier3).
여기서는 임계값 전제 판정, 라벨 함수, 조립(diagnose)을 담당한다.
"""
from __future__ import annotations

from typing import Optional

from core.schema import Finding
from agents.eval.types import (
    DEFAULT_TOP_K, EvalRecord, Mode, resolve_mode,
    F1_PASS_THRESHOLD, F1_EXACT_MATCH, ANSWER_PASS_THRESHOLD, ANSWER_SEMANTIC_FLOOR,
    ANSWER_CORRECTNESS_MIN, EVIDENCE_DENSITY_MIN,
    CONTEXT_CHARS_MAX, CONTEXT_MIDDLE_BAND,
    RAGAS_FAITHFULNESS_MIN, RAGAS_RESPONSE_RELEVANCY_MIN, RAGAS_CONTEXT_PRECISION_MIN,
)
from agents.eval.metrics_common import (
    set_mode, set_context, active_mode, missed_gold_ids,
    candidate_window, reachable_window,
)
from agents.eval.metrics_basic import (            # tier1
    is_abstention, _compute_metrics, _gold_span_boundary_analysis,
    _gold_chunk_evidence_density, _oversized_gold_spans,
    _context_char_total, _gold_position_band,
)
from agents.eval.metrics_search import (           # tier2
    _gold_ranks, _bm25_hits_gold, _gold_in_corpus, _missed_gold_in_corpus,
    _gold_absent_from_corpus, _gold_corpus_membership,
    _gold_dense_ranks, _gold_lexical_ranks, _gold_pre_rerank_ranks,
    _redundancy_above_gold, _rerank_promoted_ids,
)
from agents.eval.metrics_ragas import (            # tier3
    _compute_ragas_real, _compute_ragas_oracle, _abstention_judged, _reasoning_mode_oracle,
    _correctness_counts_oracle, _faith, _faith_oracle, _rel, _rel_oracle, _ctx_precision,
)


# ══════════════════════════════════════════════════════════════════
#  임계값 판정 함수
# ══════════════════════════════════════════════════════════════════

def _recall_ok(record: EvalRecord) -> bool:
    """검색 성공"""
    return record.recall_at_k >= 1

def _degraded_near_miss(record: EvalRecord, *, oracle: bool) -> bool:
    """유사도 단독으로 계산된 answer_correctness 가 근접 오답 문턱 미만인가 — 강등 전용 신호.

    degraded(=factual TP/FP/FN 분류 실패) 값은 의미축(승격)에서 빼지만, 강등에서까지 빼면
    'degrade 는 판정을 느슨하게 만들지 않는다'는 규약(metrics_ragas._answer_correctness 주석)이
    깨진다 — 실제로 lexical 0.5·degraded ac 0.0 이 옛 게이트에선 실패, 새 게이트에선 통과였다.
    승격은 막고 강등만 남겨, 판정기가 죽었을 때 게이트가 헐거워지지 않게 한다.

    단 어휘 F1 이 완전일치(gold 완전 포함, F1_EXACT_MATCH)면 강등하지 않는다 — 이 강등의 대상은
    '어휘~0.5 의 애매한 근접 오답'인데(위 예시), 완전일치인(=애매하지 않은) 정답까지 끌어내리면
    degrade 된(불안정한) 심판이 정답을 오답으로 뒤집는다. 그러면 recall=1·oracle 통과와 겹쳐
    C 전제(_context_failed)가 열려 context_noise_interference 로 오진되고, 심판 degrade 의
    비결정성 때문에 같은 답이 반복마다 통과/실패를 오간다(실측: probe_qa_26360 반복2 ❌ vs 반복3 ✅).
    면제선을 통과 문턱(0.5)이 아니라 완전일치로 둔 이유는, 0.5~0.9 대의 애매한 근접 오답은 심판이
    죽었을 때 여전히 보수적으로 강등해야 하기 때문이다(그 경계는 안전망으로 유지). 완전일치만
    면제하므로 승격이 아니라 'degrade 가 정답을 끌어내리는 것'만 막는다 — 게이트는 안 헐거워진다.

    추가로 '검색 근거에 붙었나(faithfulness)'까지 요구한다 — 완전일치라도 부정문 오답
    ('X 가 아니다')은 gold 를 글자 그대로 담아 어휘가 1.0 이 되지만 문맥과 충돌해 근거성이 낮다.
    이 강등이 원래 (심판이 살아있을 때 ANSWER_SEMANTIC_FLOOR 가) 걸러주던 게 바로 그 부정문
    오답이라, 심판이 degrade 된 실행에서 근거성 가드가 그 몫을 이어받아 면제 구멍을 좁힌다.
    근거성 미측정(None)이면 면제하지 않아 기존 강등으로 흐른다(보수적)."""
    ragas = record.oracle_ragas if oracle else record.ragas
    if not ragas.get("answer_correctness_degraded"):
        return False
    lexical = record.oracle_f1 if oracle else record.f1_score
    faith = _faith_oracle(record) if oracle else _faith(record)
    if (lexical is not None and lexical >= F1_EXACT_MATCH
            and faith is not None and faith >= RAGAS_FAITHFULNESS_MIN):
        return False
    ac = record.oracle_ragas_answer_correctness if oracle else record.ragas_answer_correctness
    return ac is not None and ac < ANSWER_CORRECTNESS_MIN


def _answer_ok(record: EvalRecord, *, oracle: bool) -> bool:
    """정답 판정 — lexical 과 RAGAS 의미축을 섞은 한 점수(answer_score)로 판정한다.

    lexical 단독 게이트를 버린 이유: char-F1 은 표현 차이·길이에 흔들려서(gold 가 묻지 않은
    수식어를 하나 더 갖고 있으면 맞은 답도 0.49) 문턱 하나로 정답을 가를 수 없다. 실제로
    문턱 바로 아래로 떨어진 정답들이 실패로 잡히고, 검색·오라클은 통과했으니 C그룹
    (context_noise_interference·chunking_underchunking)으로 오진돼 optimize 가 엉뚱한
    처방을 받았다. 반대로 의미축(RAGAS)만 쓰면 판정기 편차에 그대로 노출된다 —
    그래서 가중합(types.blend_answer_score)으로 서로의 흔들림을 흡수시킨다.

    · 판정기 degrade + 근접 오답 문턱 미만(_degraded_near_miss): 실패 — 승격은 못 하지만
      강등은 하는 값이라, 판정기가 죽었을 때 게이트가 헐거워지지 않게 막는다.
    · 의미축 미측정(DEEP 미만·판정기 degrade): lexical 단독 F1_PASS_THRESHOLD (기존 동작).
    · 의미축 < ANSWER_SEMANTIC_FLOOR: lexical 이 아무리 높아도 실패 — 표면형만 비슷한
      근접 오답(부정문·'3월'↔'3일')을 거르던 강등 규칙을 이 바닥선이 이어받는다.
    · 그 밖: answer_score >= ANSWER_PASS_THRESHOLD.

    통과집합은 옛 게이트(lexical 단독 + ac 강등)의 상위집합이다 — 전수 격자로 '옛 통과 →
    새 실패' 0건을 확인했다(tests/test_answer_match.py::TestGateMonotonicity). 그래서
    _oracle_ok 실패를 전제로 하는 라벨(bad_gold_answer·B그룹)은 발동이 줄기만 하고 늘지 않는다.

    비용: 두 트랙 모두 record dict 만 읽는다(LLM 트리거 없음). _oracle_ok 은
    report._oracle_accuracy 가 성공 probe 에도 부르므로 이 성질이 지켜져야 한다.
    이 성질은 _degraded_near_miss 의 검사 순서가 지탱한다 — 그 안의 면제선이 _faith 를
    보지만, 그 전에 answer_correctness_degraded 가 아니면 곧장 False 로 빠진다. degrade
    는 이미 끝난 RAGAS 실행의 플래그라 여기서 새 호출이 나지 않는다. 순서를 바꾸거나
    면제선을 앞으로 옮기면 이 계약이 깨진다.
    """
    if _degraded_near_miss(record, oracle=oracle):
        return False
    lexical = record.oracle_f1 if oracle else record.f1_score
    semantic = record.oracle_answer_semantic if oracle else record.answer_semantic
    if semantic is None:
        return lexical >= F1_PASS_THRESHOLD
    if semantic < ANSWER_SEMANTIC_FLOOR:
        return False
    score = record.oracle_answer_score if oracle else record.answer_score
    return score >= ANSWER_PASS_THRESHOLD


def _f1_ok(record: EvalRecord) -> bool:
    """실제 트랙 정답 판정(이름은 호출부 호환 유지 — 실제 기준은 혼합 점수 _answer_ok)."""
    return _answer_ok(record, oracle=False)

def _oracle_ok(record: EvalRecord) -> bool:
    """오라클 트랙 정답 판정 (기준 동일)."""
    return _answer_ok(record, oracle=True)

def _abstained(record: EvalRecord) -> bool:
    """기권 판정 — DEEP+ 는 AspectCritic, 미만·미측정은 마커 휴리스틱(tier1).

    빈 답변은 기권이 아니다 — LLM 오류·타임아웃으로 아무것도 못 받은 것을 '올바른 기권'으로
    통과시키면 _is_success 가 성공으로 집계해 생성 장애가 진단에서 사라진다.
    (판정 대상도 아니라 LLM 호출도 생략한다.)
    """
    if not (record.generated_answer or "").strip():
        return False
    judged = _abstention_judged(record)
    if judged is not None:
        return judged
    return is_abstention(record.generated_answer)


def _grounded_ok(record: EvalRecord) -> bool:
    """실제 답이 검색 context 에 근거하나(real faithfulness). 미측정(DEEP 미만)은 통과로 본다."""
    faith = _faith(record)
    return faith is None or faith >= RAGAS_FAITHFULNESS_MIN


def _grounded_verified(record: EvalRecord) -> bool:
    """실제 답이 검색 근거에 붙었다고 '실측으로 확인'됐나 — faithfulness 가 측정되고 문턱 이상.

    _grounded_ok 와 반대로 미측정(None)은 False 로 본다: '근거 없음이 반증 안 됨'(_grounded_ok)이
    아니라 '근거 있음이 확인됨'을 요구한다. 검색축에 크레딧을 줄지 판단하는 자리라, 검증 안 된
    grounding 에 크레딧을 주면 parametric(근거 없이 맞힌 답)까지 검색 성공으로 오인한다."""
    faith = _faith(record)
    return faith is not None and faith >= RAGAS_FAITHFULNESS_MIN


def _retrieval_verified_grounded(record: EvalRecord) -> bool:
    """라벨 골드는 top-k 에 못 들었지만(recall<1) 검색이 '다른 유효 근거'로 정답을 뒷받침한 경우.

    전제: 답이 정답(_f1_ok) ∧ 답이 검색 근거에 붙음(_grounded_verified) ∧ 골드도 유효(_oracle_ok).
      · _oracle_ok 로 골드 유효성을 확인 → 골드가 틀린 경우(bad_gold_chunk)와 분리된다.
      · _grounded_verified 로 parametric(근거 없이 맞힌 답)을 배제 → 진짜 검색 실패는 실패로 남는다.
    이 경우 검색은 라벨 골드 청크 하나를 놓쳤을 뿐 정답 근거는 실제로 제공했으므로, recall 스윙만으로
    실패/성공이 뒤집히지 않게 성공으로 처리한다(검색축 크레딧은 faithfulness).
    호출 전 _compute_ragas_oracle 이 돌아 _oracle_ok 가 유효해야 한다(diagnose 순서 참조)."""
    return (not _recall_ok(record) and _f1_ok(record)
            and _oracle_ok(record) and _grounded_verified(record))


def _parametric_overreliance(record: EvalRecord) -> bool:
    """정답인데 검색 context 에 근거가 없음 = 모델 파라미터 기억으로 답함.

    gold 가 실제로 검색된 경우(recall=1)만 본다 — 미검색이면 근거를 못 쓴 게 당연하고
    그건 A그룹(검색 실패) 몫이다.
    """
    return (record.probe.answer_exists is not False
            and bool(record.probe.ground_truth)
            and _recall_ok(record) and _f1_ok(record)
            and not _grounded_ok(record))


def _is_success(record: EvalRecord) -> Optional[bool]:
    """probe 단위 성공/실패 판정 — recall + 정답 혼합 점수(_answer_ok: lexical tier1 × 의미 tier3).

    True  = 성공 (검색·정답 모두 통과 / 무응답 기대인데 올바르게 기권)
    False = 실패
    None  = 판정 불가 (대조할 정답셋이 없음)

    recall 은 정답셋이 있는 경로에서만 본다 — 무응답 기대 probe 는 gold 가 없어
    recall_at_k = -1 이므로, 앞에서 recall 을 보면 올바른 기권까지 실패가 된다.

    근거성(_grounded_ok)도 본다 — 정답이어도 context 에 근거가 없으면 파라미터 기억으로 맞힌
    것이라 검색이 검증되지 않은 상태다(generation_parametric_overreliance). DEEP 미만은
    faithfulness 미측정이라 통과로 처리돼 기존 동작과 같다.
    """
    if record.probe.answer_exists is False:
        return _abstained(record)                       # 무응답 기대 → 올바른 기권이 성공(recall 무관)
    if not record.probe.ground_truth:
        return None                                     # 대조할 정답 없음 → 판정 불가
    # 검색 성공(recall=1) + 정답 일치(혼합 점수 — DEEP 이면 RAGAS 의미축이 함께 들어간다)
    return _recall_ok(record) and _f1_ok(record) and _grounded_ok(record)

def _is_multi_hop(record: EvalRecord) -> bool:
    """멀티홉 질문 여부(probe.qtype). bridge / hop_binding / corpus_gap_partial_hop 판별용."""
    return record.probe.qtype in ("bridge", "comparison", "aggregation")

def _enumeration_pressure(record: EvalRecord) -> Optional[bool]:
    """근거 개수가 top-k 슬롯에 근접/초과하나(나열형 압박). 개수는 청킹 불변량인 span 수로.
    True=span 압박 / None=span 없음(legacy)→청크 수 폴백 / False=압박 없음."""
    k = len(record.retrieved_chunk_ids) or DEFAULT_TOP_K
    threshold = int(k * 0.8)
    spans = record.probe.gold_spans
    if spans:
        return bool(len(spans) >= 2 and len(spans) >= threshold)
    gold_n = len(record.probe.gold_chunk_ids)          # legacy: 청크 수 폴백
    return None if (gold_n >= 2 and gold_n >= threshold) else False


def _enumeration_recoverable_by_top_k(record: EvalRecord) -> bool:
    """놓친 gold 가 wide-N 안에 있나(top_k↑ 로 회복 가능). 전부 밖이면 semantic 영역.
    tier2 미측정(None)이면 보수적으로 True."""
    ranks = _gold_ranks(record)
    if ranks is None:
        return True
    return any(ranks.get(g) is not None for g in missed_gold_ids(record))

# ══════════════════════════════════════════════════════════════════
#  A그룹: 검색 실패 (Oracle 통과) — retrieval_*
# ══════════════════════════════════════════════════════════════════

def _retrieval_fixable(record: EvalRecord) -> bool:
    """검색으로 고칠 수 있는 실패인가 — A 슬롯 진입 전제.

    recall<1 만으로 슬롯을 열면 구체 라벨들은 self-scope 로 빠져도 롤업(retrieval_failure)이
    무조건 붙어 '검색을 고쳐라'가 남는다. 아래 둘은 검색 결함이 아니다:
    1) gold 가 전부 코퍼스 밖 → 자료를 채워야 한다(corpus_gap 몫). '하나라도 없음'이 아니라
       '전부 없음'으로 본다 — 부분 gap 의 코퍼스에 있는 몫은 실제로 검색이 놓친 것이다.
    2) answer_exists=False probe → gold 는 답의 근거가 아니라 전제를 반박하는 청크라
       (false premise, probe_gen 참조) recall 자체가 성립하지 않는다.
    """
    if record.probe.answer_exists is False:
        return False
    return _gold_absent_from_corpus(record) is not True


# ── 순위 원인 공통 기반 ──────────────────────────────────────────
#  최종 순위는 `채널 검색 → 융합 → 후보창 → 리랭크 → top_k 컷` 을 거쳐 만들어진다.
#  '순위가 낮다'는 증상이라, 어느 단계에서 gold 를 잃었는지가 처방을 정한다.
#  아래 헬퍼들이 그 단계 귀속을 위한 공통 전제를 만든다.

def _reranked(record: EvalRecord) -> bool:
    """이 검색에 리랭크가 실제로 적용됐나(설정만 켜진 게 아니라 성공적으로 돌았나)."""
    return bool(record.retrieval_details.get("reranked"))


def _ranked_beyond_top_k(record: EvalRecord) -> dict[str, int]:
    """놓친 gold 중 융합 순위가 top_k 밖인 것 {gold_id: rank}. 순위 라벨 공통 전제.

    순위가 top_k 이내로 나오면 '순위가 낮아 밖'과 모순이라 제외한다 — 그건 검색 비결정성이나
    인덱스 변경이지 순위 문제가 아니다(결정적 리트리버면 발생하지 않는다).
    """
    ranks = _gold_ranks(record)
    if ranks is None:
        return {}
    top_k = len(record.retrieved_chunk_ids)
    if top_k <= 0:
        return {}                        # 검색 0건 → 순위 문제가 아니라 검색 장애
    return {g: ranks[g] for g in missed_gold_ids(record)
            if ranks.get(g) is not None and ranks[g] > top_k}


def _rankable(record: EvalRecord) -> dict[str, int]:
    """융합 순위로 다룰 gold — top_k 밖 + 도달 가능 창(reachable_window) 안쪽.

    창 밖은 리랭커를 켜도 후보를 넓혀도 닿지 않아 순위 문제가 아니다 → 표현 문제
    (semantic/lexical mismatch)로 인계한다. 이 경계가 없으면 wide_n(=100) 안의 모든
    검색 실패가 순위 라벨로 흡수된다.

    ⚠ 리랭크 단계에서 잃은 gold 는 여기 안 잡힐 수 있다(_rerank_lost 참고) — 순위 라벨의
    전체 관할은 이 둘의 합집합(_rank_scope)이다.
    """
    window = reachable_window()
    return {g: r for g, r in _ranked_beyond_top_k(record).items() if r <= window}


def _rerank_lost(record: EvalRecord) -> dict[str, int]:
    """리랭크 **전엔 top_k 안**이었는데 결과엔 없는 놓친 gold {gold_id: pre_rerank 순위}.
    = 리랭커가 실제로 떨어뜨린 몫(강등).

    융합 순위가 top_k **이내**여도 대상이다. 이게 _rankable 과 갈리는 지점이자, wide 재검색에서
    리랭크를 끈 것의 직접적 귀결이다:

      융합이 gold 를 3위에 뒀는데(top_k=5 안) 리랭커가 12위로 떨어뜨려 놓친 경우
      → 교과서적인 리랭커 강등인데, 융합 순위 3 은 'top_k 밖'이 아니라 _rankable 에 안 잡힌다.

    예전엔 재검색도 리랭크를 태워서 이 경우가 '재검색도 top_k 밖'으로 보였다. 이제 융합 순위는
    리랭크 이전 값이라 그 은폐가 사라졌고, 대신 '후보엔 있었는데 결과엔 없다'가 리랭크 단계
    손실의 직접 증거가 된다 — 순위 대조가 필요 없다.

    이 집합을 안 보면 강등이 침묵할 뿐 아니라, lexical/semantic 의 양보 게이트도 안 걸려
    "dense 가 gold 를 놓쳤다"는 거짓 전제로 임베딩·청킹 처방이 나간다(리랭커를 끄면 낫는데).

    단 '후보 목록에 있었다'만으로는 강등이 아니다. 리랭크 **전에도 top_k 밖**이던 gold 는
    리랭커가 떨어뜨린 게 아니라 끌어올리는 데 실패한 것이다(_rerank_not_lifted). 두 경우는
    처방이 다르다 — 강등은 롤백(disable_reranker)이 유효하지만, 못 끌어올린 경우는 롤백해도
    융합 순위가 그대로 top_k 밖이라 개선 가능성이 0이다.
    """
    if not _rerank_cut_attributable(record):
        return {}
    pre = _gold_pre_rerank_ranks(record) or {}
    top_k = len(record.retrieved_chunk_ids)
    return {g: pre[g] for g in missed_gold_ids(record)
            if pre.get(g) is not None and pre[g] <= top_k}


def _rerank_not_lifted(record: EvalRecord) -> dict[str, int]:
    """리랭커 후보에 있었지만 **리랭크 전에도 top_k 밖**이던 놓친 gold {gold_id: pre 순위}.

    리랭커가 떨어뜨린 게 아니라 올려야 할 것을 못 올린 경우다. 롤백은 여기서 확정 무효다 —
    되돌리면 융합 순위가 그대로 쓰이는데 그 순위가 애초에 top_k 밖이라 gold 는 여전히 누락된다.
    남는 레버는 리랭커 모델 교체이며, 그 처방이 열리기 전까지는 리포트 전용이다(rules.py draft).
    """
    if not _rerank_cut_attributable(record):
        return {}
    pre = _gold_pre_rerank_ranks(record) or {}
    top_k = len(record.retrieved_chunk_ids)
    return {g: pre[g] for g in missed_gold_ids(record)
            if pre.get(g) is not None and pre[g] > top_k}


def _rank_scope(record: EvalRecord) -> set[str]:
    """순위 라벨 전체의 관할(gold id 집합) — 세 갈래의 합집합.

      _rankable            : 융합 순위가 top_k 밖 + 도달 가능 창 안 (리랭크 계열 라벨의 관할)
      _rerank_lost         : 리랭크 전엔 top_k 안이었는데 결과엔 없음 = 강등 (융합 순위 무관)
      _rerank_not_lifted   : 후보엔 있었으나 리랭크 전에도 top_k 밖 = 못 끌어올림
      _rerank_window_missed: 후보 목록에 아예 없었음 (_rankable 의 부분집합)
      융합 손실             : 단일 채널이 top_k 안에 뒀는데 융합이 밀어냄 (창 무관 — 아래 참고)

    lexical/semantic mismatch 는 이 관할이 비어 있을 때만 발동한다(전제가 'dense 가 놓침'인데,
    여기 잡혔다는 건 dense 가 gold 를 실제로 올려놨다는 뜻이라 전제가 깨진다).

    순위 값이 아니라 id 집합을 돌려준다 — 세 갈래의 순위가 서로 다른 단계의 값(pre_rerank vs
    융합)이라 한 dict 에 섞으면 소비처가 같은 의미로 오해하기 쉽다. 여기 쓰임은 '비었나'뿐이다.
    """
    scope = (set(_rerank_lost(record)) | set(_rerank_not_lifted(record))
             | set(_rankable(record)))
    advantage = _channel_advantage(record)
    if advantage is not None:
        scope.add(advantage[1])
    return scope


def _rerank_stage_visible(record: EvalRecord) -> bool:
    """리랭크 단계를 실측으로 들여다볼 수 있나 — 적용됐고 후보 목록이 기록됐나.

    거짓이면 리랭크 단계는 미측정이고, 순위 원인은 잔여 라벨(low_rank)이 맡는다.

    MMR 여부는 여기서 보지 않는다 — 이 신호가 뒷받침하는 사실은 '리랭커 입력 목록이 무엇이었나'
    이고, MMR 은 리랭크 **이후** 단계라 그 사실을 바꿀 수 없기 때문이다. 최종 컷의 귀속이
    필요한 라벨만 _rerank_cut_attributable 을 따로 본다.
    """
    return _reranked(record) and _gold_pre_rerank_ranks(record) is not None


def _rerank_cut_attributable(record: EvalRecord) -> bool:
    """최종 top_k 컷을 리랭커에게 귀속시킬 수 있나.

    리랭커와 MMR 이 둘 다 켜지면 최종 선택을 MMR 이 맡는다(리랭커는 후보풀 순서만 바꾼다,
    agents/rag/retriever.py). 그러면 후보에 있던 gold 가 결과에 없어도 떨어뜨린 주체가
    리랭커라는 보장이 없어 '강등'·'못 끌어올림'을 단정할 수 없다.

    반면 '후보 목록에 아예 없었다'(candidate_miss)는 리랭크 이전 사실이라 MMR 과 무관하다 —
    그 라벨까지 이 게이트로 막으면, MMR 을 켜는 순간 후보창 문제가 처방을 못 받게 된다.
    """
    if record.retrieval_details.get("mmr_applied"):
        return False
    return _rerank_stage_visible(record)


def _rerank_window_missed(record: EvalRecord) -> dict[str, int]:
    """리랭커 후보 목록에 아예 없던 대상 gold {gold_id: 융합 순위}.

    리랭크 이전 사실이라 MMR 적용 여부와 무관하다(_rerank_cut_attributable 참고).
    """
    if not _rerank_stage_visible(record):
        return {}
    pre = _gold_pre_rerank_ranks(record) or {}
    return {g: r for g, r in _rankable(record).items() if pre.get(g) is None}


def _channel_advantage(record: EvalRecord):
    """융합 순위는 top_k 밖인데 단일 채널은 top_k 안에 뒀나(= 융합이 깎았다).

    반환 (channel, gold_id, channel_rank, fused_rank) / 없거나 미측정이면 None.
    하이브리드가 실제로 쓰인 검색에서만 본다 — dense 단일 모드에서 BM25 가 gold 를 잡는 건
    융합 손실이 아니라 애초에 그 채널이 파이프라인에 없는 것이라 lexical_mismatch 몫이다.

    ⚠ 여기만 도달 가능 창을 적용하지 않는다(_rankable 이 아니라 _ranked_beyond_top_k 를 쓴다).
    창의 논거는 '리랭커 처방이 닿는 범위'인데, 이 라벨의 처방(hybrid_dense_weight)은 리랭커와
    무관하게 어떤 융합 순위든 끌어올릴 수 있기 때문이다 — 처방마다 도달 범위가 다르니 게이트도
    처방을 따라간다. 창을 걸면 dense 1위/융합 60위 같은 극단적 융합 손실에서 이 라벨이 침묵하고,
    semantic_mismatch 가 'dense 가 놓쳤다'는 거짓 전제로 임베딩 교체를 처방하게 된다.
    """
    if record.retrieval_details.get("search_mode") != "hybrid":
        return None
    targets = _ranked_beyond_top_k(record)
    if not targets:
        return None
    top_k = len(record.retrieved_chunk_ids)
    dense = _gold_dense_ranks(record) or {}
    lexical = _gold_lexical_ranks(record) or {}
    best = None
    for gold_id, fused in targets.items():
        for channel, ranks in (("dense", dense), ("lexical", lexical)):
            rank = ranks.get(gold_id)
            if rank is not None and rank <= top_k and (best is None or rank < best[2]):
                best = (channel, gold_id, rank, fused)
    return best


def _crowding_recovery(record: EvalRecord):
    """중복 청크를 접으면 gold 가 top_k 안까지 올라오나.

    반환 (gold_id, analysis) / 회복 안 되거나 미측정이면 None.
    """
    analyses = _redundancy_above_gold(record)
    if not analyses:
        return None
    top_k = len(record.retrieved_chunk_ids)
    for gold_id, analysis in sorted(analyses.items(), key=lambda kv: kv[1]["rank"]):
        if analysis["redundant"] > 0 and analysis["projected_rank"] <= top_k:
            return gold_id, analysis
    return None


def _upstream_rank_cause(record: EvalRecord) -> bool:
    """순위를 잃은 더 앞단 원인이 실측됐나 — 밴드 라벨들의 공통 반대 게이트.

    융합 손실·중복 밀림은 단계(후보창/리랭크)와 무관하게 성립할 수 있고, 성립하면 그쪽이
    뿌리다. 예: 융합이 gold 를 40위로 밀어 후보창(20) 밖으로 내보냈다면 '후보창을 넓혀라'가
    아니라 '융합 가중치를 고쳐라'가 맞다. 튜플 순서가 아니라 신호로 배타를 세운다
    (신호는 memoize 되므로 이 게이트에 추가 검색 비용이 없다).
    """
    return _channel_advantage(record) is not None or _crowding_recovery(record) is not None


def _rank_reason(record: EvalRecord, targets: dict[str, int]) -> str:
    """순위 라벨 공통 reason 접두 — 대상 gold 의 순위와 top_k."""
    ranked = ", ".join(f"{g}:{r}" for g, r in sorted(targets.items(), key=lambda kv: kv[1]))
    return (f"missed_gold_ranks=[{ranked}] > top_k={len(record.retrieved_chunk_ids)}, "
            f"recall@k({record.recall_basis})={_v(record.recall_at_k)}")


def retrieval_rank_fusion_loss(record: EvalRecord) -> Optional[Finding]:
    """
    하이브리드 융합이 단일 채널의 상위 순위를 깎아 gold 를 top_k 밖으로 밀어냄.
    확정: 검색이 hybrid + 어느 한 채널(dense/BM25)은 gold 를 top_k 안에 뒀는데 융합 순위는 밖(tier2).

    처방이 리랭커가 아니라 융합 가중치(hybrid_dense_weight)라서 따로 가른다 — 한 채널이 이미
    정답을 상위에 두고 있으면, 비싼 cross-encoder 를 새로 태우기 전에 가중치를 그 채널 쪽으로
    옮기는 게 싸고 정확하다. 어느 채널이 유리한지는 metadata['favored_channel'] 로 넘긴다
    (planner 가 후보 가중치를 계산할 방향 근거).
    """
    advantage = _channel_advantage(record)
    if advantage is None:
        return None
    channel, gold_id, channel_rank, fused_rank = advantage
    finding = _finding(
        record, "retrieval_rank_fusion_loss", "retrieval_failure", confirmed=True,
        reason=f"{channel}_rank={channel_rank}<=top_k={len(record.retrieved_chunk_ids)} "
               f"인데 fused_rank={fused_rank}({gold_id}), recall@k({record.recall_basis})={_v(record.recall_at_k)}",
    )
    finding.metadata["favored_channel"] = channel
    finding.metadata["channel_ranks"] = {
        "dense": _gold_dense_ranks(record) or {},
        "lexical": _gold_lexical_ranks(record) or {},
        "fused": _gold_ranks(record) or {},
    }
    return finding


def retrieval_duplicate_crowding(record: EvalRecord) -> Optional[Finding]:
    """
    상위 슬롯을 중복 청크가 차지해 gold 가 밀려남.
    확정: gold 위 비-gold 중 near-duplicate 잉여분을 접으면 예상 순위가 top_k 이내(tier2 기하).

    순위 원인 중 유일하게 리랭커로 안 고쳐져서 따로 가른다 — cross-encoder 는 중복 청크를
    상위에 그대로 둔다(각각이 실제로 질문과 관련 있으니 점수가 높다). 처방은 중복 제거·MMR 이다.
    회복 가능성을 순위표에서 직접 계산하므로(재검색 0회) 확정으로 낸다.

    융합 손실이 실측되면 양보한다 — 둘 다 성립할 수 있는데(하이브리드 + 채널 우세 + 중복 회복)
    융합이 파이프라인 앞단이라 뿌리다. 이 쌍만 튜플 순서로 갈리던 것을 신호로 세운 것이다.
    """
    if _channel_advantage(record) is not None:
        return None                      # 융합 손실이 앞단 → 그쪽이 뿌리
    recovery = _crowding_recovery(record)
    if recovery is None:
        return None
    gold_id, analysis = recovery
    finding = _finding(
        record, "retrieval_duplicate_crowding", "retrieval_failure", confirmed=True,
        reason=f"rank={analysis['rank']}({gold_id}), 중복 경쟁청크={analysis['redundant']} → "
               f"중복 제거 시 순위={analysis['projected_rank']}<=top_k="
               f"{len(record.retrieved_chunk_ids)}, recall@k({record.recall_basis})={_v(record.recall_at_k)}",
    )
    finding.metadata["crowding_analysis"] = {
        g: dict(a) for g, a in (_redundancy_above_gold(record) or {}).items()
    }
    return finding


def retrieval_rerank_candidate_miss(record: EvalRecord) -> Optional[Finding]:
    """
    gold 가 리랭커 후보창 밖이라 리랭커가 보지도 못함.
    확정: 리랭크 적용됨 + gold 가 pre_rerank 후보 목록에 없음(비용 0 측정).

    순위 대조가 아니라 '리랭커가 실제로 받은 후보 목록에 없었다'로 직접 확인한다.
    처방은 후보창 확대뿐이다 — 목록에 없던 것은 아무리 잘 정렬해도 올라오지 않는다.

    앞단 원인(융합 손실·중복 밀림)이 실측되면 그쪽이 뿌리라 양보한다 — 융합이 밀어내 창 밖으로
    나간 것을 '창을 넓혀라'로 처방하면 원인을 두고 증상을 키우는 셈이 된다.
    """
    if _upstream_rank_cause(record):
        return None
    targets = _rerank_window_missed(record)
    if not targets:
        return None
    finding = _finding(
        record, "retrieval_rerank_candidate_miss", "retrieval_failure", confirmed=True,
        reason=f"{_rank_reason(record, targets)}, pre_rerank_ids에 없음"
               f"(candidate_window={candidate_window()})",
    )
    finding.metadata["candidate_window"] = candidate_window()
    return finding


def retrieval_reranker_ineffective(record: EvalRecord) -> Optional[Finding]:
    """
    리랭커가 gold 를 후보로 봤지만 top_k 안으로 끌어올리지 못함.
    확정: 리랭크 적용됨 + gold 가 후보 목록 안 + **리랭크 전에도 top_k 밖**(비용 0 측정).

    강등(retrieval_reranker_demotion)과 반드시 갈라야 한다 — 여기선 리랭커가 떨어뜨린 게
    아니라 올리는 데 실패한 것이라, 강등의 정석 처방인 롤백(disable_reranker)이 **확정 무효**다.
    되돌리면 융합 순위가 그대로 쓰이는데 그 순위가 애초에 top_k 밖이라 gold 는 여전히 누락된다.
    개선 가능성이 0인 처방에 iteration 을 쓰지 않으려면 판정 단계에서 갈라야 한다.

    남는 레버는 리랭커 모델 교체뿐인데 후보가 미정이라(rules.py BLOCKER) 지금은 리포트 전용이다.
    리랭커를 한 번 켠 뒤의 주류 경로라 커버리지 영향이 있다 — 모델 교체가 열리면 ready 로 올린다.
    """
    if not _rerank_cut_attributable(record) or _upstream_rank_cause(record):
        return None
    if _rerank_window_missed(record):
        return None                      # 후보창 밖 gold 가 함께 있음 → candidate_miss 가 앞단
    targets = _rerank_not_lifted(record)
    if not targets:
        return None
    seen = ", ".join(f"{g}:{r}" for g, r in sorted(targets.items(), key=lambda kv: kv[1]))
    finding = _finding(
        record, "retrieval_reranker_ineffective", "retrieval_failure", confirmed=True,
        reason=f"pre_rerank_ranks=[{seen}] — 리랭크 전에도 top_k="
               f"{len(record.retrieved_chunk_ids)} 밖(강등 아님, 못 끌어올림), "
               f"reranker_status={record.retrieval_details.get('reranker_status')}, "
               f"recall@k({record.recall_basis})={_v(record.recall_at_k)}",
    )
    finding.metadata["pre_rerank_ranks"] = dict(targets)
    return finding


def retrieval_reranker_demotion(record: EvalRecord) -> Optional[Finding]:
    """
    리랭커가 gold 를 후보로 보고도 top_k 밖으로 떨어뜨림.
    확정: 리랭크 적용됨 + gold 가 pre_rerank 후보 목록 안 + **리랭크 전엔 top_k 안** + 최종 top_k 밖.

    '리랭크 전엔 top_k 안'이 이 라벨의 핵심 전제다. 그게 없으면 리랭커가 올리는 데 실패했을
    뿐인 케이스(retrieval_reranker_ineffective)까지 '떨어뜨렸다'로 확정하고, 롤백이라는
    확정 무효 처방을 내게 된다.

    후보창을 넓히는 처방이 여기선 무효라서 candidate_miss 와 반드시 갈라야 한다 — gold 는 이미
    창 안에 있었고 리랭커가 매긴 점수가 낮았을 뿐이라, 창을 넓히면 gold 아래로 후보만 더 들어온다.
    처방은 리랭커 되돌리기·모델 교체이며, 우리가 켠 리랭커가 역효과라는 뜻이므로
    optimize 의 롤백 신호이기도 하다.

    pre_rerank_ids 가 없으면(옛 계약 retriever) candidate_miss 와 구분이 불가해 침묵한다.

    놓친 gold 가 여럿이라 '후보창 밖'과 '강등'이 한 probe 에 같이 있으면 candidate_miss 에
    양보한다 — 후보 선정이 리랭크보다 앞단이고, 이 코드베이스는 앞단 원인을 뿌리로 본다
    (융합 손실이 리랭크 단계 라벨보다 앞서는 것과 같은 기준). 양보하지 않으면 둘 다 확정으로
    서서 튜플 순서로만 갈린다.
    """
    if not _rerank_cut_attributable(record) or _upstream_rank_cause(record):
        return None
    if _rerank_window_missed(record):
        return None                      # 후보창 밖 gold 가 함께 있음 → candidate_miss 가 앞단
    targets = _rerank_lost(record)
    if not targets:
        return None
    seen = ", ".join(f"{g}:{r}" for g, r in sorted(targets.items(), key=lambda kv: kv[1]))
    fused = _gold_ranks(record) or {}
    fused_note = ", ".join(f"{g}:{fused.get(g)}" for g in sorted(targets))
    finding = _finding(
        record, "retrieval_reranker_demotion", "retrieval_failure", confirmed=True,
        reason=f"pre_rerank_ranks=[{seen}](리랭크 전 top_k 안) 인데 최종 top_k="
               f"{len(record.retrieved_chunk_ids)} 밖, fused_ranks=[{fused_note}], "
               f"reranker_status={record.retrieval_details.get('reranker_status')}, "
               f"recall@k({record.recall_basis})={_v(record.recall_at_k)}",
    )
    finding.metadata["pre_rerank_ranks"] = dict(targets)
    return finding


def retrieval_low_rank(record: EvalRecord) -> Optional[Finding]:
    """
    gold가 후보엔 있으나 순위가 낮아 top-k 밖 — 리랭크로 되돌릴 순수 정렬 오류.
    확정: 놓친 gold 가 융합 재검색에서 top_k 뒤·도달 가능 창 안 순위로 발견 + 리랭크 미적용(tier2).

    단계 귀속이 끝나고 남는 잔여 라벨이다. 리랭커가 아직 안 켜진 구간이라 처방은 '켜기' 하나이고,
    여기가 원래 이 라벨이 뜻하던 케이스다. 아래가 실측되면 그쪽이 가져간다:
      융합 손실 / 중복 밀림 (앞단 원인) · 후보창 밖 / 리랭커 강등 (리랭크 단계)
    배타는 튜플 순서가 아니라 각자의 신호로 선다.

    리랭크는 돌았는데 단계를 귀속할 수 없는 구성 — 후보 목록 미기록(옛 계약 retriever)이거나
    MMR 이 최종 컷을 맡은 경우 — 에서는 어느 단계에서 잃었는지 알 수 없다 → 예비로 낸다.
    이미 켜진 리랭커에 '켜라'를 처방하지 않기 위해서다(planner 는 예비를 자동 처방에서 뺀다).
    """
    targets = _rankable(record)
    if (not targets or _upstream_rank_cause(record)
            or _rerank_cut_attributable(record) or _rerank_window_missed(record)):
        return None
    confirmed = not _reranked(record)
    return _finding(
        record, "retrieval_low_rank", "retrieval_failure", confirmed=confirmed,
        reason=f"{_rank_reason(record, targets)}, reranked={bool(_reranked(record))}"
               f"{'' if confirmed else ', pre_rerank_ids=-(단계 귀속 불가)'}",
    )


def retrieval_lexical_mismatch(record: EvalRecord) -> Optional[Finding]:
    """
    dense는 놓쳤으나 BM25로 잡히는 단어 불일치.
    확정: BM25 가 gold 를 잡음 + 순위 라벨이 다룰 구간이 아님(tier2).

    양보 기준이 '융합 wide-N 후보에 있나'가 아니라 '도달 가능 창 안에 있나'다 — 창 밖(예:
    BM25 1위인데 융합 80위)은 리랭커로 못 고치므로 순위 문제가 아니라 어휘 불일치이고,
    처방도 하이브리드 활성화가 맞다. 예전 기준으로는 이 케이스를 low_rank 가 가져가서
    닿지도 않는 리랭커를 처방했다.
    하이브리드가 이미 켜진 채로 창 안에서 밀린 경우는 retrieval_rank_fusion_loss 영역이다.
    놓친 gold 가 여럿이어도 그중 하나만 BM25 에 잡히면 probe 전체가 이 라벨이다(슬롯당 1원인).
    """
    if _bm25_hits_gold(record) is not True:
        return None
    if _rank_scope(record):
        return None                      # 순위 라벨이 다룰 구간 → 그쪽에 양보
    return _finding(
        record, "retrieval_lexical_mismatch", "retrieval_failure", confirmed=True,
        reason=f"bm25_hits_gold=True, dense_missed=True, recall@k({record.recall_basis})={_v(record.recall_at_k)}",
    )


def retrieval_semantic_mismatch(record: EvalRecord) -> Optional[Finding]:
    """
    dense·BM25 모두 놓친 의미 연결 실패. (단 gold 가 코퍼스엔 있을 때만 — 없으면 corpus_gap)
    확정: BM25 도 gold 를 못 잡음 + 놓친 gold 중 코퍼스에 있는 게 있음(tier2).
    멤버십은 gold 별로 본다 — probe 단위 bool 로 접으면 혼합 코퍼스(놓친 gold 가 {있음+없음})
    에서 라벨이 통째로 소실된다. 코퍼스에 없는 몫은 corpus_gap 이 additive 로 함께 붙는다.
    코퍼스 멤버십 미측정(None)은 corpus_gap 과 구분 불가라 예비(missing_gold 와 동일 기준).
    qtype=bridge 는 bridge 의존과 구분 불가라 양보(원 질문으론 hop2 를 원래 못 찾음).

    도달 가능 창 안에 gold 가 있으면 양보한다(lexical_mismatch 와 같은 기준) — 그 경우
    dense 는 gold 를 '놓친' 게 아니라 순위를 낮게 준 것이라 이 라벨의 전제가 사실과 어긋난다.
    게이트가 없으면 순위 라벨과 이 라벨이 둘 다 확정으로 서서 튜플 순서로만 갈린다.
    """
    if record.probe.qtype == "bridge":
        return None                      # bridge 의존과 구분 불가 → bridge 에 양보
    if _rank_scope(record):
        return None                      # 순위 라벨이 다룰 구간 → 그쪽에 양보
    if _bm25_hits_gold(record) is not False:
        return None
    in_corpus = _missed_gold_in_corpus(record)
    if in_corpus is None:
        return _finding(
            record, "retrieval_semantic_mismatch", "retrieval_failure", confirmed=False,
            reason=f"bm25_hits_gold=False, missed_gold_in_corpus=-, recall@k({record.recall_basis})={_v(record.recall_at_k)}",
        )
    if in_corpus:
        return _finding(
            record, "retrieval_semantic_mismatch", "retrieval_failure", confirmed=True,
            reason=f"bm25_hits_gold=False, "
                   f"missed_gold_in_corpus={len(in_corpus)}/{len(missed_gold_ids(record))}, "
                   f"recall@k({record.recall_basis})={_v(record.recall_at_k)}",
        )
    return None                          # 놓친 gold 가 전부 코퍼스 밖 → corpus_gap 영역


def retrieval_missing_gold(record: EvalRecord) -> Optional[Finding]:
    """
    gold는 corpus에 있으나 top-k에 없음.
    확정: 놓친 gold 중 코퍼스에 있는 게 있음(tier2, semantic 과 동일하게 gold 별 멤버십).
    [폴백] 메커니즘(순위/어휘/의미)은 못 밝히고 코퍼스 존재만 실측 — 자원 다 주입된 런타임에선
    앞 라벨들이 선점하고, 자원 빠진 구성에서만 이 라벨이 잡는다.
    qtype=bridge 는 bridge 의존과 구분 불가라 양보(semantic 과 동일 기준).
    """
    if record.probe.qtype == "bridge":
        return None                      # bridge 의존과 구분 불가 → bridge 에 양보
    if not missed_gold_ids(record):
        return None                      # 놓친 gold 청크가 없음 → 'top-k 에 없다'가 성립 안 함
    in_corpus = _missed_gold_in_corpus(record)
    if in_corpus is None:
        return _finding(
            record, "retrieval_missing_gold", "retrieval_failure", confirmed=False,
            reason=f"missed_gold_in_corpus=-, recall@k({record.recall_basis})={_v(record.recall_at_k)}",
        )
    if in_corpus:
        return _finding(
            record, "retrieval_missing_gold", "retrieval_failure", confirmed=True,
            reason=f"missed_gold_in_corpus={len(in_corpus)}/{len(missed_gold_ids(record))}, "
                   f"recall@k({record.recall_basis})={_v(record.recall_at_k)}",
        )
    return None                          # 놓친 gold 가 전부 코퍼스 밖 → corpus_gap 영역


def _has_oversized_gold_span(record: EvalRecord) -> bool:
    """gold span 이 최장 청크보다 길다 = 겹침으로는 원리적으로 한 청크에 못 담는다."""
    analysis = _oversized_gold_spans(record)
    return bool(analysis and analysis.get("oversized_count", 0) > 0)


def chunking_overchunking(record: EvalRecord) -> Optional[Finding]:
    """
    청크가 근거보다 작아 gold span 이 한 청크에 담기지 못함.
    확정: gold span 길이 > 최장 청크 길이(tier1 기하).

    담김 가능 조건이 L <= chunk_size 라, L 이 그보다 크면 겹침을 늘려도 절대 담기지 않는다.
    그래서 처방이 overlap 이 아니라 chunk_size 증가이고, chunking_context_mismatch 와 배타다
    (그쪽은 겹침으로 회복 가능한 경계 분할 — 회복 가능성은 optimize 가 시뮬레이션으로 판정).
    """
    if not _has_oversized_gold_span(record):
        return None
    if _recall_ok(record) and not _context_failed(record):
        return None
    analysis = _oversized_gold_spans(record)
    finding = _finding(
        record, "chunking_overchunking", "retrieval_failure", confirmed=True,
        reason=f"max_span={analysis['max_span_len']}>max_chunk={analysis['max_chunk_len']}, "
               f"oversized={analysis['oversized_count']}, recall@k({record.recall_basis})={_v(record.recall_at_k)}",
    )
    finding.metadata["oversized_analysis"] = dict(analysis)
    return finding


def chunking_context_mismatch(record: EvalRecord) -> Optional[Finding]:
    """정답 근거가 현재 청크 경계에 나뉘어 한 청크에 온전히 없음을 판정한다.

    gold span과 현재 청크의 원문 좌표만 비교해 경계 분할을 잡는다(저비용·결정적, LLM 없음).

    확정(tier1): 경계 분할이 실측되면 confirmed. 이 코드베이스의 confirmed 는 '처방이 통한다'가
    아니라 '그 원인의 판별 신호가 실제로 측정됐다'는 뜻이다 — retrieval_low_rank(gold 가 wide
    후보에 있음)·retrieval_missing_gold(코퍼스에 있음)와 같은 기준이며, 그 라벨들도 처방 효과를
    증명하진 않는다. planner 가 처방 후보를 만들 때 쓰는 근거도 같은 기하 정보다
    (candidate_grounding.source = gold_span_boundary_geometry).

    단 _RETRIEVAL_CAUSE 안에서는 맨 뒤에 둔다 — 실측된 다른 검색 원인(enumeration/low_rank 등)이
    있으면 그쪽이 먼저 채택되고, 경계 분할은 달리 설명이 없을 때 채택된다.
    """

    if _has_oversized_gold_span(record):
        return None                      # 겹침으로 못 담는 길이 → overchunking 영역
    analysis = _gold_span_boundary_analysis(record)
    if not isinstance(analysis, dict) or analysis.get("boundary_split_count", 0) <= 0:
        return None
    if _recall_ok(record) and not _context_failed(record):
        return None
    # 반대 게이트: span 개수 압박이면 슬롯 부족이 지배 → enumeration 에 양보.
    if (missed_gold_ids(record) and _enumeration_pressure(record) is True
            and _enumeration_recoverable_by_top_k(record)):
        return None
    finding = _finding(
        record, "chunking_context_mismatch", "retrieval_failure", confirmed=True,
        reason=f"boundary_split={analysis.get('boundary_split_count')}, "
               f"recall@k({record.recall_basis})={_v(record.recall_at_k)}, {_answer_reason(record)}",
    )
    finding.metadata["boundary_analysis"] = dict(analysis)
    return finding


def retrieval_missing_bridge_dependency(record: EvalRecord) -> Optional[Finding]:
    """
    연쇄형(bridge): hop2 근거가 hop1 답에 의존해 원 질문 검색으론 못 찾음.
    예비: hop 의존을 확정할 신호(decompose 재검색 회복)가 없어 optimize 가 위임받는다.
    comparison/aggregation 은 hop 간 독립이라 제외(나열형은 enumeration 담당).
    low_rank·lexical 확정은 원 질문으로 잡힌다는 실측이라 bridge 를 반증 → 그쪽이 우선.
    처방(enable_query_decomposition)은 rules.py draft — query_rewrite/max_hops 스키마 미합의 BLOCKER.
    """
    if record.probe.qtype != "bridge" or not (0 <= record.recall_at_k < 1):
        return None
    if not missed_gold_ids(record):
        return None                      # 놓친 hop 근거가 없음 → bridge 의존을 의심할 근거 없음

    return _finding(
        record, "retrieval_missing_bridge_dependency", "retrieval_failure", confirmed=False,
        reason=f"qtype={record.probe.qtype}, recall@k({record.recall_basis})={_v(record.recall_at_k)}",
    )


def retrieval_incomplete_enumeration(record: EvalRecord) -> Optional[Finding]:
    """
    나열형(aggregation): 필요한 근거 개수가 가변인데 top-k 고정이라 일부 누락.
    확정: span 개수 압박(청킹 불변량) + qtype=aggregation + 놓친 gold 가 wide-N 안.
    개수를 gold_chunk_ids 로 세면 세밀 청킹이 부풀려 chunking 을 나열형으로 오진 → span 수로.
    압박 없으면 chunking 에 양보(반대 게이트). qtype None·legacy 는 예비.
    """
    missed = missed_gold_ids(record)
    if not missed:
        return None                      # 놓친 gold 없음 → 개수 부족 누락 아님
    if not record.retrieved_chunk_ids:
        return None                      # 검색 0건(장애) → 슬롯 부족이 아니라 롤업 영역

    pressure = _enumeration_pressure(record)
    if pressure is False:
        return None                      # 개수 압박 없음 → chunking 등 다른 원인
    if not _enumeration_recoverable_by_top_k(record):
        return None                      # 놓친 gold 전부 wide-N 밖 → top_k↑ 무효, semantic 영역

    confirmed = pressure is True and record.probe.qtype == "aggregation"
    return _finding(
        record, "retrieval_incomplete_enumeration", "retrieval_failure", confirmed=confirmed,
        reason=f"spans={len(record.probe.gold_spans)}, gold_chunks={len(record.probe.gold_chunk_ids)}, "
               f"top_k={len(record.retrieved_chunk_ids)}, qtype={record.probe.qtype}, "
               f"recall@k({record.recall_basis})={_v(record.recall_at_k)}",
    )

def retrieval_failure(record: EvalRecord) -> Optional[Finding]:
    """검색 실패 롤업"""
    return _finding(
        record, "retrieval_failure", "retrieval_failure", confirmed=False,
        reason=_rollup_reason(record),
    )

# ══════════════════════════════════════════════════════════════════
#  B그룹: 생성 실패 (Oracle 실패) — generation_*
# ══════════════════════════════════════════════════════════════════

def _generation_failed(record: EvalRecord) -> bool:
    """생성 실패 전제(B 공통): gold 컨텍스트로도 답이 틀림, 또는 무응답인데 답을 지어냄.

    parametric_overreliance 는 여기 넣지 않는다 — 전제가 '답이 맞음'이라 다른 B 원인과 경쟁
    관계가 아니고, _pick 에 섞으면 확정으로 먼저 뽑혀 같은 probe 의 오라클 생성 실패를 가린다.
    corpus_gap 처럼 슬롯 밖 additive 로 붙인다.
    """
    if record.oracle_answer is not None and not _oracle_ok(record):
        return True
    if record.probe.answer_exists is False and not _abstained(record):
        return True
    if _wrongful_abstention_premise(record):    # 유효 근거 두고 기권 → 생성측 과다 기권
        return True
    return False


def _wrongful_abstention_premise(record: EvalRecord) -> bool:
    """근거가 검색됐는데 기권했나 — 과다 기권(B) 전제.

    검색 성공(recall=1) = 답의 근거가 실제로 top-k 에 있었는데 모델이 '제공된 정보로는 알 수
    없습니다'로 회피한 경우다. 기권이 옳은 상황(_expects_abstention: 무응답 기대·코퍼스 결손)은
    제외한다 — 그건 generation_abstention_failure 의 정반대 짝이다.

    _oracle_ok 는 전제에 넣지 않는다(리뷰 High): 오라클 답변도 같은 generator·같은
    index_config 로 생성되므로(eval/agent.py), 과다 기권이 체계적이면 오라클도 함께 기권해
    oracle_f1 이 낮아진다. 그걸 요구하면 '가끔 기권'만 잡고 '항상 기권'(더 심각한 쪽)은
    놓치는 역설이 생긴다 — 이 라벨이 잡으려던 상황이 정확히 그쪽이다.

    비용 순서: 싼 지표(recall·f1)로 먼저 거르고, 마커 휴리스틱(is_abstention)이 걸린
    뒤에만 판정기(_abstained → AspectCritic, record 단위 memoize)를 부른다. not _f1_ok 는
    의미상 무해하고(기권 답이 정답 판정을 통과할 일은 없다) 정답 경로의 LLM 호출을 없앤다."""
    if not _recall_ok(record) or _f1_ok(record):
        return False
    if _expects_abstention(record):
        return False
    if not is_abstention(record.generated_answer):
        return False                     # 싼 게이트 — 마커가 없으면 판정기까지 가지 않는다
    return _abstained(record)


def _reasoning_mode(record: EvalRecord) -> Optional[str]:
    """오라클 답변의 추론 실패 모드(LLM 단일분류, DEEP+). 미측정이면 None.

    '근거는 있는데 답이 틀린'(faithfulness 문턱 이상) 경우에만 부른다 — 근거 자체가 없으면
    hallucination 이 이미 결정하므로 분류기를 부를 이유가 없다(LLM 1회 절약).
    단일홉의 hop_binding 은 misinterpretation 으로 흡수한다 — 엮을 hop 이 없으니 결합 오류가
    성립하지 않고, 실제로는 관계·조건을 잘못 읽은 것이다(안 그러면 롤업으로 버려진다).
    """
    faith = _faith_oracle(record)
    if faith is None or faith < RAGAS_FAITHFULNESS_MIN:
        return None
    mode = _reasoning_mode_oracle(record)
    if mode == "hop_binding" and not _is_multi_hop(record):
        return "misinterpretation"
    return mode


# 분류기가 지목하는 모드 → 라벨. 'other'만 여기 없다(구체적 원인 지목이 아님).
_REASONING_LABELS = {
    "contradiction": "generation_contradiction",
    "numerical_error": "generation_numerical_error",
    "misinterpretation": "generation_misinterpretation",
    "hop_binding": "generation_hop_binding_error",
}


def _reasoning_failure_identified(record: EvalRecord) -> bool:
    """분류기가 구체적 추론 실패를 지목했나."""
    return _reasoning_mode(record) in _REASONING_LABELS


def _hop_binding_counts_hit(record: EvalRecord) -> bool:
    """카운트가 결합 오류를 지목하나 — FN=0(요소 누락 없음) & FP>0(근거 없는 주장).

    전제(근거는 있었나 = faith≥문턱)는 포함하지 않는다 — 호출부가 이미 세운 뒤에 부른다.
    근거 없는 답(hallucination 영역)에도 참이 될 수 있으니 전제 없이 재사용하지 말 것.
    """
    if not _is_multi_hop(record):
        return False
    counts = _correctness_counts_oracle(record)
    if counts is None:
        return False
    _tp, fp, fn = counts
    return fn == 0 and fp > 0


def _reasoning_failure_evidence(record: EvalRecord) -> bool:
    """추론 실패가 실측됐나 — 분류기 지목, 또는 분류기 미측정 시 카운트 폴백.
    bad_gold_answer 주장을 반증한다.

    카운트 폴백까지 보는 이유: 분류기 미측정이면 폴백이 결합 오류를 확정으로 내는데,
    bad_gold_answer_oracle 이 튜플상 앞이라 그 확정을 선점했다(순서 대신 함수 자체로 배타).

    단 폴백은 분류기가 미측정일 때만 본다. 분류기가 실제로 돌아 'other'(구체적 실패 없음)를
    냈다면 그게 카운트 휴리스틱보다 강한 측정이고, reasoning_failure 도 그 경우엔 침묵한다 —
    여기서 양보까지 하면 반증만 당한 채 아무 라벨도 안 남아 슬롯이 롤업으로 강등된다.
    """
    if _reasoning_failure_identified(record):
        return True
    return _reasoning_mode(record) is None and _hop_binding_counts_hit(record)


def _expects_abstention(record: EvalRecord) -> bool:
    """기권했어야 하는 상황인가 — 답의 근거가 애초에 없는 두 경우.

    1) 무응답 기대 probe(answer_exists=False): 원래 답이 없는 질문 (tier1)
    2) gold 가 하나도 코퍼스에 없음 + 답도 틀림: 답은 있으나 우리 코퍼스에 근거가 없음 (tier2)

    2번은 '전부 없음'(_gold_absent_from_corpus)이라야 한다 — '하나라도 없음'으로 잡으면
    부분 gap(일부 gold 는 코퍼스에 있고 검색까지 됨)에서 실제 근거에 기반한 부분 답변을
    환각으로 확정하게 되고, 같은 probe 에 함께 붙는 A 라벨(검색을 고쳐라)·
    corpus_gap_partial_hop(자료를 채워라)과 처방이 정면으로 모순된다.
    정답을 맞힌 2번은 뺀다 — 근거 없이 맞힌 건 parametric_overreliance 소관이다.
    (싼 판정만 쓴다 — 여기서 걸러야 _abstained 의 AspectCritic 호출이 해당 probe 에만 든다.)
    """
    if record.probe.answer_exists is False:
        return True
    return _gold_absent_from_corpus(record) is True and not _f1_ok(record)


def generation_abstention_failure(record: EvalRecord) -> Optional[Finding]:
    """
    기권했어야 하는 질문에 기권하지 않고 답을 지어냄.
    확정: _expects_abstention + 기권 아님(DEEP+ AspectCritic / 미만은 마커 휴리스틱).

    두 상황(무응답 기대 / 코퍼스에 근거 없음)을 한 라벨로 둔다 — 실패 행동이 '근거 없이
    단정했다'로 같고 처방도 같아서(기권 프롬프트 강화·인용 요구), 나눠도 optimize 가
    다르게 처방할 게 없다. 어느 쪽이 발동했는지는 reason 으로 구분한다.
    corpus_gap 쪽은 D 라벨(자료 보강)이 함께 붙는다 — 자료를 채우는 것과 별개로
    '근거가 없을 때 어떻게 행동해야 하는가'를 이 라벨이 짚는다.
    (라벨은 optimize/rules.py 의 처방 키와 일치시킨다 — generation_abstention_failure)

    어느 쪽이 발동했는지는 metadata['trigger'] 로도 남긴다 — reason 문자열만으로는
    reporter·optimize 가 '자료를 채우면 사라질 기권 실패'와 그렇지 않은 것을 파싱 없이 못 가른다.
    """
    if not (record.generated_answer or "").strip():
        return None                      # 빈 답변은 지어낸 게 아니라 생성 실패 → 롤업 몫
    if not _expects_abstention(record) or _abstained(record):
        return None
    judge = "aspect_critic" if _abstention_judged(record) is not None else "heuristic"
    no_answer_expected = record.probe.answer_exists is False
    trigger = ("answer_exists=False" if no_answer_expected
               else f"gold_in_corpus=False, {_answer_reason(record)}(오답)")
    finding = _finding(
        record, "generation_abstention_failure", "generation_failure", confirmed=True,
        reason=f"{trigger}, 기권 아님({judge})",
    )
    finding.metadata["trigger"] = "no_answer_expected" if no_answer_expected else "corpus_gap"
    return finding


def generation_wrongful_abstention(record: EvalRecord) -> Optional[Finding]:
    """근거는 검색됐는데 잘못 기권함 — generation_abstention_failure 의 정반대 짝.

    확정: 검색 성공(recall=1)인데 실제 답은 기권('제공된 정보로는 알 수 없습니다'). 답의 근거가
    실제로 top-k 에 있었는데 모델이 회피한 것이라, 검색·컨텍스트 구조가 아니라 생성측 과다
    기권이다. 처방은 노이즈필터/MMR 이 아니라 기권 완화(relax_abstention).

    오라클 통과를 요구하지 않는다 — 오라클도 같은 generator·설정으로 생성돼 과다 기권이
    체계적이면 함께 기권하므로, 요구하면 정작 심한 케이스를 놓친다(전제 함수 주석 참고).

    기권 답은 주장이 없어 faithfulness=1 로 C 게이트(context_noise_interference·
    reranker_low_precision)를 trivially 통과해 오라벨됐었다(실측: probe_qa_42204 반복1 →
    reranker_low_precision, 반복2·3 → context_noise_interference). _context_failed 가 같은
    전제로 기권을 제외하고 이 라벨이 B 슬롯에서 진실한 원인을 짚는다."""
    if not _wrongful_abstention_premise(record):
        return None
    judge = "aspect_critic" if _abstention_judged(record) is not None else "heuristic"
    return _finding(
        record, "generation_wrongful_abstention", "generation_failure", confirmed=True,
        reason=f"recall@k={_v(record.recall_at_k)}(근거 검색됨), "
               f"oracle_f1={_v(record.oracle_f1)}, 기권함({judge}), {_answer_reason(record)}",
    )


def generation_parametric_overreliance(record: EvalRecord) -> Optional[Finding]:
    """
    정답이지만 검색 context 에 근거가 없음 — 모델 파라미터 기억에 의존.
    확정: gold 검색됨(recall=1) + 정답 + real faithfulness 낮음(tier3).
    답이 맞아 사용자 눈엔 성공이지만 검색이 검증되지 않은 상태라, 코퍼스가 바뀌면 조용히 깨진다.
    """
    if not _parametric_overreliance(record):
        return None
    return _finding(
        record, "generation_parametric_overreliance", "generation_failure", confirmed=True,
        reason=f"faithfulness={_v(_faith(record))}<{RAGAS_FAITHFULNESS_MIN}(근거 없음), "
               f"{_answer_reason(record)}(정답), recall@k({record.recall_basis})={_v(record.recall_at_k)}",
    )


def generation_hallucination(record: EvalRecord) -> Optional[Finding]:
    """
    정답 context가 있는데 지어냄.
    확정: faithfulness 낮음(= 답변이 gold context 어디에도 근거 없음).
    문턱 이상인데 답이 틀린 경우는 근거는 있으나 잘못 쓴 것이라 generation_reasoning_failure 소관.
    """
    faith = _faith_oracle(record)
    if faith is not None and faith < RAGAS_FAITHFULNESS_MIN:
        return _finding(
            record, "generation_hallucination", "generation_failure", confirmed=True,
            reason=f"faithfulness={_v(faith)}<{RAGAS_FAITHFULNESS_MIN}, oracle_f1={_v(record.oracle_f1)}",
        )
    return None


def generation_reasoning_failure(record: EvalRecord) -> Optional[Finding]:
    """
    근거는 있으나 답이 틀린 네 원인(모순/수치/해석/결합)을 LLM 단일분류 1회로 가른다.
    확정: 분류기가 넷 중 하나를 지목(tier3, DEEP+ 전용).
    분류기 미측정이면 결합 오류만 카운트로 폴백(_hop_binding_from_counts) — 나머지 셋은
    카운트로 구분이 안 된다. 'other'는 구체적 지목이 아니라 침묵(롤업 몫).

    함수 하나에 라벨 넷을 두는 건 '라벨마다 함수 1개' 원칙의 의도적 예외다 — 측정이 하나뿐이고
    분류기가 한 값만 돌려줘 넷이 구조적으로 배타라, 함수를 쪼개면 순서에 의미가 없는 항목만
    늘고 '독립 신호 여러 개'라는 잘못된 인상을 준다.
    """
    faith = _faith_oracle(record)
    if faith is None or faith < RAGAS_FAITHFULNESS_MIN:
        return None                      # 근거 자체가 없음 → hallucination 영역
    mode = _reasoning_mode(record)
    if mode is None:
        return _hop_binding_from_counts(record, faith)
    label = _REASONING_LABELS.get(mode)
    if label is None:
        return None                      # 'other'
    return _finding(
        record, label, "generation_failure", confirmed=True,
        reason=f"reasoning_mode={mode}, faithfulness={_v(faith)}(근거는 있음), "
               f"qtype={record.probe.qtype}, oracle_f1={_v(record.oracle_f1)}",
    )

def _hop_binding_from_counts(record: EvalRecord, faith) -> Optional[Finding]:
    """분류기 미측정 시의 안전망 — 카운트로 결합 오류만 판정한다.

    FN=0 이 '결합' 신호다(필요한 gold 요소가 다 있는데 답이 틀렸으면 남는 설명은 잘못 엮었다는
    것뿐이고, 그 주장이 FP 로 잡힌다). 나머지 셋(모순/수치/해석)은 카운트로 구분할 수 없다.
    """
    if not _is_multi_hop(record):
        return None
    counts = _correctness_counts_oracle(record)
    if counts is None:
        return _finding(
            record, "generation_hop_binding_error", "generation_failure", confirmed=False,
            reason=f"reasoning_mode=-, correctness_counts=-, faithfulness={_v(faith)}, "
                   f"qtype={record.probe.qtype}, oracle_f1={_v(record.oracle_f1)}",
        )
    tp, fp, fn = counts
    if _hop_binding_counts_hit(record):
        return _finding(
            record, "generation_hop_binding_error", "generation_failure", confirmed=True,
            reason=f"reasoning_mode=-, missing={fn}(요소 누락 없음), unsupported={fp}, tp={tp}, "
                   f"faithfulness={_v(faith)}, qtype={record.probe.qtype}, "
                   f"oracle_f1={_v(record.oracle_f1)}",
        )
    return None                          # FN>0 이면 요소 누락 → partial_answer 영역

def generation_partial_answer(record: EvalRecord) -> Optional[Finding]:
    """
    정답 context가 있는데 일부 요소·조건 누락.
    확정: gold 문장 중 답변에 없는 것(FN)이 있고, 맞은 것(TP)도 있음 = 부분 답변.
    FN=0(누락 없음)·TP=0(전부 누락, '부분'이 아님)은 침묵.
    카운트 미측정이면 relevancy 로 폴백하되 예비 — relevancy 는 누락이 아니라 on-topic 여부를
    재고 회피성·빈 답변에 0 을 줘서 확정 근거로 약하다.
    """
    counts = _correctness_counts_oracle(record)
    if counts is not None:
        tp, _fp, fn = counts
        if fn > 0 and tp > 0:
            return _finding(
                record, "generation_partial_answer", "generation_failure", confirmed=True,
                reason=f"missing={fn}/{tp + fn}(gold 요소), tp={tp}, oracle_f1={_v(record.oracle_f1)}",
            )
        return None

    rel = _rel_oracle(record)
    if rel is not None and rel < RAGAS_RESPONSE_RELEVANCY_MIN:
        return _finding(
            record, "generation_partial_answer", "generation_failure", confirmed=False,
            reason=f"correctness_counts=-, response_relevancy={_v(rel)}<{RAGAS_RESPONSE_RELEVANCY_MIN}, "
                   f"oracle_f1={_v(record.oracle_f1)}",
        )
    return None


def generation_failure(record: EvalRecord) -> Optional[Finding]:
    """생성 실패 롤업"""
    return _finding(
        record, "generation_failure", "generation_failure", confirmed=False,
        reason=_rollup_reason(record),
    )


# ══════════════════════════════════════════════════════════════════
#  C그룹: context 구조 문제
# ══════════════════════════════════════════════════════════════════

def _context_failed(record: EvalRecord) -> bool:
    """컨텍스트 구조 문제(C) 전제: 검색 성공(recall=1)·생성 가능(oracle 통과)인데 실제 답만 틀림.

    기권은 제외한다 — 유효 근거를 두고 기권한 건 컨텍스트 '구조'(노이즈·길이·배치)로 답이
    틀린 게 아니라 생성측 과다 기권이다(generation_wrongful_abstention, B). 제외하지 않으면
    기권 답이 주장 없음 → faithfulness=1 로 context_noise_interference·reranker_low_precision
    게이트를 trivially 통과해 오라벨되고, 노이즈필터/MMR 같은 엉뚱한 처방을 부른다.

    배제 술어는 B 라벨과 같은 _wrongful_abstention_premise 를 쓴다 — 두 슬롯이 한 신호로
    갈려야 '한쪽이 가져가면 다른 쪽은 안 가져간다'가 보장되고, 그 안의 마커 선필터
    (is_abstention) 덕에 C 후보마다 AspectCritic 을 부르지 않는다(리뷰 Medium)."""
    return (_recall_ok(record) and _oracle_ok(record)
            and not _f1_ok(record) and not _wrongful_abstention_premise(record))

def _context_ungrounded(record: EvalRecord) -> bool:
    """C 전제 + 실제 답이 gold·노이즈 어디에도 근거 없음(real faithfulness 낮음).

    faithfulness 높은 쪽(노이즈 청크에 근거함)은 context_noise_interference 담당이라,
    이 전제가 그 반대편 — 청크는 다 있는데 길이·배치 때문에 gold 를 못 쓴 경우 — 를 가른다.
    노이즈가 청크 '안'이면 chunking_underchunking 영역이라 함께 배제한다(C그룹 3자 배타).
    """
    if not _context_failed(record) or _chunk_noise_heavy(record):
        return False
    faith = _faith(record)
    return faith is not None and faith < RAGAS_FAITHFULNESS_MIN


def _gold_in_middle_band(record: EvalRecord) -> Optional[bool]:
    """검색 결과 안 gold 가 양끝이 아니라 중간 밴드에 있나. 위치 미측정이면 None.

    미측정을 False(=양끝)로 접으면 안 된다 — 위치는 chunk-id 대조인데 C 전제의 recall 은
    span 기준이라, 재청킹으로 id 가 어긋난 recall=1 케이스(missed_gold_ids 독스트링 참고)나
    검색 결과 3건 미만에서 위치가 안 잡힌다. 그때 too_long_context 가 전부 흡수하면
    처방이 갈린다(재배치 vs 길이 축소). 미측정이면 둘 다 침묵시킨다.
    """
    position = _gold_position_band(record)
    if position is None:
        return None
    return CONTEXT_MIDDLE_BAND[0] <= position <= CONTEXT_MIDDLE_BAND[1]


def too_long_context(record: EvalRecord) -> Optional[Finding]:
    """
    context가 너무 길어 잡음·과부하로 품질 저하.
    확정: C 전제 + 답에 근거 없음(faith 낮음) + context 길이 문턱 초과 + gold 는 양끝.
    gold 가 중간이면 lost_in_the_middle 영역 — 위치로 함수 자체 배타(튜플 순서 비의존).
    위치 미측정(None)이면 배치를 못 가르므로 침묵한다.
    """
    if not _context_ungrounded(record):
        return None
    total = _context_char_total(record)
    if total < CONTEXT_CHARS_MAX or _gold_in_middle_band(record) is not False:
        return None
    return _finding(
        record, "too_long_context", "context_failure", confirmed=True,
        reason=f"context_chars={total}>={CONTEXT_CHARS_MAX}, "
               f"faithfulness={_v(_faith(record))}<{RAGAS_FAITHFULNESS_MIN}(근거 없음), "
               f"recall@k({record.recall_basis})={_v(record.recall_at_k)}",
    )


def lost_in_the_middle(record: EvalRecord) -> Optional[Finding]:
    """
    청크가 긴 context 중간이라 LLM이 참조 못함.
    확정: C 전제 + 답에 근거 없음 + gold 가 중간 밴드 + context 길이 문턱 초과.
    길이 조건을 함께 두는 건 이 현상 자체가 긴 context 에서만 성립하기 때문이다 —
    짧은 context 에서 근거를 못 쓴 건 배치 문제가 아니라 생성측 이탈(롤업)이다.
    """
    if not _context_ungrounded(record) or _gold_in_middle_band(record) is not True:
        return None
    total = _context_char_total(record)
    if total < CONTEXT_CHARS_MAX:
        return None
    return _finding(
        record, "lost_in_the_middle", "context_failure", confirmed=True,
        reason=f"gold_position={_v(_gold_position_band(record))}(중간), context_chars={total}, "
               f"faithfulness={_v(_faith(record))}<{RAGAS_FAITHFULNESS_MIN}(근거 없음), "
               f"recall@k({record.recall_basis})={_v(record.recall_at_k)}",
    )


def _chunk_noise_heavy(record: EvalRecord) -> bool:
    """청크 '안'이 노이즈로 채워졌나 — 근거 밀도가 낮고 context_precision 도 낮음.
    청크 '사이' 노이즈(비-gold 청크)와 가르는 신호다."""
    density = _gold_chunk_evidence_density(record)
    precision = _ctx_precision(record)
    return (density is not None and density < EVIDENCE_DENSITY_MIN
            and precision is not None and precision < RAGAS_CONTEXT_PRECISION_MIN)


def chunking_underchunking(record: EvalRecord) -> Optional[Finding]:
    """
    청크가 근거보다 훨씬 커서 무관한 내용까지 함께 딸려옴.
    확정: C 전제 + gold 청크 내부 근거 밀도 낮음 + context_precision 낮음.

    청크 '사이' 노이즈(context_noise_interference)가 아니라 청크 '안'의 노이즈다 —
    gold 를 담은 청크만 분모로 삼아 재므로 top_k·리랭커 문제와 섞이지 않는다.
    """
    if not _context_failed(record) or not _chunk_noise_heavy(record):
        return None
    return _finding(
        record, "chunking_underchunking", "retrieval_failure", confirmed=True,
        reason=f"evidence_density={_v(_gold_chunk_evidence_density(record))}<{EVIDENCE_DENSITY_MIN}, "
               f"context_precision={_v(_ctx_precision(record))}<{RAGAS_CONTEXT_PRECISION_MIN}, "
               f"recall@k({record.recall_basis})={_v(record.recall_at_k)}",
    )


def reranker_low_precision(record: EvalRecord) -> Optional[Finding]:
    """
    리랭커가 무관한 청크를 상위로 올림.
    확정: C 전제 + 리랭크가 실제 적용됨 + context_precision 낮음 + 청크 안 노이즈는 아님
          + **리랭크가 top_k 구성을 실제로 바꿨음**(_rerank_promoted_ids).

    마지막 조건이 이 라벨을 예비에서 확정으로 올린다. 예전엔 '리랭크를 거친 결과의 정밀도가
    낮다'까지만 말했다 — 리랭커가 원인이라는 증거가 아니었다(원래 검색이 나빴을 수도 있다).
    retriever 가 pre_rerank_ids 를 싣게 된 뒤로 전/후 대조가 가능해져(PR #51 이 미뤄둔 조건),
    리랭크가 새로 밀어 올린 청크가 있을 때만 리랭커에 책임을 묻는다.

    구성이 안 바뀌었으면(promoted 가 빈 리스트) 아예 발행하지 않는다 — 리랭크 전에도 같은
    청크들이 들어 있었으므로 리랭커를 바꿔봐야 이 실패는 그대로다. 전/후 기록이 없는 옛
    결과(promoted is None)는 판정할 근거가 없으니 종전대로 예비로 남긴다.
    """
    if not _context_failed(record):
        return None
    if not record.retrieval_details.get("reranked"):
        return None
    if _chunk_noise_heavy(record):
        return None                      # 노이즈가 청크 안 → underchunking 영역
    precision = _ctx_precision(record)
    if precision is None or precision >= RAGAS_CONTEXT_PRECISION_MIN:
        return None
    promoted = _rerank_promoted_ids(record)
    if promoted is not None and not promoted:
        return None                      # 리랭커가 top_k 구성을 안 바꿈 → 원인 아님
    confirmed = promoted is not None
    evidence = (f"리랭크 승격 {len(promoted)}개" if confirmed
                else "리랭크 전/후 기록 없음(예비)")
    finding = _finding(
        record, "reranker_low_precision", "retrieval_failure", confirmed=confirmed,
        reason=f"reranked=True, context_precision={_v(precision)}<{RAGAS_CONTEXT_PRECISION_MIN}, "
               f"{evidence}, recall@k({record.recall_basis})={_v(record.recall_at_k)}",
    )
    if confirmed:
        finding.metadata["rerank_promoted_ids"] = list(promoted)
    return finding


def context_noise_interference(record: EvalRecord) -> Optional[Finding]:
    """
    비-gold 청크의 상충 정보에 이끌림.
    확정: C 전제(recall=1·oracle 통과·실제 답 틀림) + 실제 답이 검색 context 에는 근거 있음
          (real faithfulness 높음) → gold 아닌 청크에 근거했다는 뜻.

    faithfulness 는 retrieved_context(gold+노이즈) 기준이라, 노이즈 청크의 정보를 가져다 쓰면
    '근거 있음'으로 높게 나온다. 낮은 쪽은 gold·노이즈 어디에도 없는 생성측 이탈이라 다른 원인이다.
    처방(enable_noise_filter/mmr)은 rules.py draft — filtering/MMR/reranker 필드 합의 미완.

    기권은 _context_failed 가 이미 배제한다 — 기권 답은 주장이 없어 faithfulness=1 로 이 게이트를
    trivially 통과하지만 '노이즈에 이끌림'이 아니라 과다 기권(generation_wrongful_abstention)이다.
    """
    if not _context_failed(record):
        return None
    if _chunk_noise_heavy(record):
        return None                      # 노이즈가 청크 안 → chunking_underchunking 영역
    faith = _faith(record)
    if faith is None or faith < RAGAS_FAITHFULNESS_MIN:
        return None
    return _finding(
        record, "context_noise_interference", "context_failure", confirmed=True,
        reason=f"faithfulness={_v(faith)}>={RAGAS_FAITHFULNESS_MIN}(검색 context 엔 근거 있음), "
               f"recall@k({record.recall_basis})={_v(record.recall_at_k)}, {_answer_reason(record)}",
    )

def context_failure(record: EvalRecord) -> Optional[Finding]:
    """콘텍스트 실패 롤업"""
    return _finding(
        record, "context_failure", "context_failure", confirmed=False,
        reason=_rollup_reason(record),
    )


# ══════════════════════════════════════════════════════════════════
#  D그룹: 데이터 문제 (파이프라인 튜닝 불가)
# ══════════════════════════════════════════════════════════════════

def bad_gold_answer(record: EvalRecord) -> Optional[Finding]:
    """
    정답셋 자체 오류/모호
    콘텍스트 실패 계열 (C 그룹에 함께 있음)
    확정(자동): faith·rel 둘 다 측정 고득점(tier3). [진짜 확정은 사람 검수.]

    단 oracle 이 통과했으면 침묵한다 — gold context 로는 정답을 맞혔다는 뜻이라 '정답셋이
    틀렸다'가 반증된다. 이때 faith·rel 고득점은 '실제 답이 gold 아닌 청크에 근거했다'는
    신호이므로 context_noise_interference 가 가져간다.
    (C 슬롯 전제 _context_failed 가 oracle 통과를 요구하므로 이 트랙에선 사실상 발동하지 않는다.
     오라클 실패 케이스는 B 슬롯의 bad_gold_answer_oracle 이 맡는다.)
    """
    if _oracle_ok(record):
        return None
    faith, rel = _faith(record), _rel(record)
    if (faith is not None and faith >= RAGAS_FAITHFULNESS_MIN
        and rel is not None and rel >= RAGAS_RESPONSE_RELEVANCY_MIN):
        return _finding(
            record, "bad_gold_answer", "gap", confirmed=True,
            reason=f"faithfulness={_v(faith)}, response_relevancy={_v(rel)}, {_answer_reason(record)}",
        )
    return None

def bad_gold_answer_oracle(record: EvalRecord) -> Optional[Finding]:
    """
    bad_gold_answer 의 오라클 트랙 버전
    생성 실패 계열 (B 그룹에 함께 있음)
    라벨은 동일('bad_gold_answer').
    추론 실패가 실측되면(분류기 지목 또는 카운트 폴백) '답은 맞는데 정답셋이 틀렸다'가
    반증된다 → 그쪽에 양보.
    """
    if _reasoning_failure_evidence(record):
        return None
    faith, rel = _faith_oracle(record), _rel_oracle(record)
    if (faith is not None and faith >= RAGAS_FAITHFULNESS_MIN
        and rel is not None and rel >= RAGAS_RESPONSE_RELEVANCY_MIN):
        return _finding(
            record, "bad_gold_answer", "gap", confirmed=True,
            reason=f"faithfulness(oracle)={_v(faith)}, response_relevancy(oracle)={_v(rel)}, "
                   f"oracle_f1={_v(record.oracle_f1)}",
        )
    return None


def bad_gold_chunk(record: EvalRecord) -> Optional[Finding]:
    """골드 '청크' 라벨 오류 — 실제 답은 맞는데(f1 통과) 골드 청크로는 못 맞춘다(oracle 실패).

    정답셋의 정답 '텍스트'는 맞지만, 근거로 지정된 '청크'(positive_chunk_ids→gold_spans)가
    엉뚱한 곳을 가리킨 경우다. 표 문서에서 같은 값(예: 스파이크 높이 332cm)이 여러 행에 나와
    정답 값 매칭이 다른 개체의 행에 꽂히는 식으로 생긴다. 파이프라인은 실제 근거를 찾아
    정답을 냈고(real faithfulness 높음 = 검색된 청크에 근거함), 골드 청크만으로는 답이 안
    나오므로(not _oracle_ok), 원인은 검색·생성이 아니라 골드 라벨이다.

    bad_gold_answer(정답 '텍스트' 오류)와 구분된다 — 그쪽은 오라클 답이 gold 에 충실한데
    ground_truth 와 어긋나는 경우고(정답이 틀림), 이쪽은 실제 답이 맞는데 gold 청크가 그 답을
    못 담은 경우다(청크가 틀림). 판별 열쇠는 '실제 답이 맞았나(_f1_ok)'다.

    이 신호가 뜨면 diagnose 가 경쟁 슬롯의 거짓 원인(retrieval_low_rank·generation_*)을 막고
    이 하나만 남겨 검수로 보낸다. 점수에서는 거짓 실패로 제외한다(report.is_gold_labeling_error).
    """
    if record.oracle_answer is None:
        return None                      # 골드 컨텍스트 없음(코퍼스 결손 등) → 청크 오라벨로 단정 불가
    if _oracle_ok(record):
        return None                      # 골드로 답이 나오면 골드 청크는 정상
    if not _f1_ok(record):
        return None                      # 실제 답도 틀리면 골드 문제로 단정 못 함(진짜 실패 영역)
    faith = _faith(record)
    if faith is None or faith < RAGAS_FAITHFULNESS_MIN:
        return None                      # 답이 검색 근거에 안 붙으면(parametric 등) 골드 단정 불가
    return _finding(
        record, "bad_gold_chunk", "gap", confirmed=True,
        reason=f"f1={_v(record.f1_score)}(실제 답 정답)·oracle_f1={_v(record.oracle_f1)}(골드론 실패)"
               f", faithfulness={_v(faith)}(검색 근거 있음) → 골드 청크 오라벨(정답 텍스트는 정상)",
    )


def _corpus_membership_ratio(record: EvalRecord) -> str:
    """코퍼스에 있는 gold 비율을 'n/N' 로. 멤버십이 per-gold 라 'False'(=하나라도 없음)만
    적으면 부분 gap 을 '전부 없음'으로 오독하게 된다(A 라벨의 missed_gold_in_corpus 와 같은 형식)."""
    membership = _gold_corpus_membership(record) or {}
    return f"{sum(1 for present in membership.values() if present)}/{len(membership)}"


def _corpus_gap_premise(record: EvalRecord) -> bool:
    """자료 결손(D) 공통 전제: gold 중 코퍼스에 없는 것이 있음.

    answer_exists=False probe 는 뺀다 — 답이 애초에 없으니 채울 자료도 없고,
    '관련 문서를 추가 수집하라'(rules.py manual_action)가 거짓 처방이 된다.
    """
    if record.probe.answer_exists is False:
        return False
    return _gold_in_corpus(record) is False


def _gold_absent_ids(record: EvalRecord) -> list[str]:
    """코퍼스에 없는 gold id 목록. 미측정이면 빈 리스트.

    `missed_gold_ids`(검색이 못 가져온 몫)와 다르다 — 이건 코퍼스 자체에 없는 몫이다.
    """
    membership = _gold_corpus_membership(record) or {}
    return [g for g, present in membership.items() if not present]


def _gap_finding(record: EvalRecord, label: str) -> Finding:
    """corpus_gap 계열 공통 Finding — reason(비율) + 누락 gold id.

    비율만으론 '코퍼스 어디가 빈지'를 리포트에 못 적는데, optimize 는 멤버십을 스스로
    구할 수 없다(`_ctx.corpus_ids` 는 Eval 자원) → id 목록을 metadata 로 넘긴다.
    """
    finding = _finding(
        record, label, "gap", confirmed=True,
        reason=f"gold_in_corpus={_corpus_membership_ratio(record)}, "
               f"qtype={record.probe.qtype}, recall@k({record.recall_basis})={_v(record.recall_at_k)}",
    )
    finding.metadata["missing_gold_ids"] = _gold_absent_ids(record)
    return finding


def corpus_gap(record: EvalRecord) -> Optional[Finding]:
    """
    필요한 자료가 코퍼스에 없음(단일홉).
    확정: gold 중 코퍼스에 없는 것이 있음(tier2).
    """
    if _corpus_gap_premise(record) and not _is_multi_hop(record):
        return _gap_finding(record, "corpus_gap")
    return None


def corpus_gap_partial_hop(record: EvalRecord) -> Optional[Finding]:
    """
    멀티홉 중 일부 hop 근거만 코퍼스에 없음.
    확정: gold 중 코퍼스에 없는 것이 있음(tier2).
    """
    if _corpus_gap_premise(record) and _is_multi_hop(record):
        return _gap_finding(record, "corpus_gap_partial_hop")
    return None


# ══════════════════════════════════════════════════════════════════
#  원인 슬롯 (브랜치 없음 — 모든 슬롯을 전부 시도)
#    각 라벨이 자기 싼 전제(recall/f1/oracle)로 self-scope 하므로, 안 맞는 슬롯은 자연히 빈다.
#    슬롯당 _pick 으로 '한 원인' 채택(확정 우선). corpus_gap 은 추가로 붙는다(additive).
#    generation_failure(예비 롤업)는 생성 슬롯 맨 뒤 후보.
# ══════════════════════════════════════════════════════════════════

# 순위 원인 4형제(fusion_loss / duplicate_crowding / candidate_miss / demotion)와 잔여 low_rank 는
# 각자의 신호로 배타가 서 있어 여기 순서에 기대지 않는다(전수 확인: tests/test_diagnose_labels).
# 앞에 두는 건 읽는 순서를 파이프라인 단계 순서와 맞추기 위해서다.
_RETRIEVAL_CAUSE = (
    retrieval_incomplete_enumeration, retrieval_missing_bridge_dependency,
    retrieval_rank_fusion_loss, retrieval_duplicate_crowding,
    retrieval_rerank_candidate_miss, retrieval_reranker_demotion,
    retrieval_reranker_ineffective, retrieval_low_rank,
    retrieval_lexical_mismatch, retrieval_semantic_mismatch, retrieval_missing_gold,
    # chunking 은 확정이지만 맨 뒤 — 실측된 다른 검색 원인이 있으면 그쪽을 먼저 채택한다.
    chunking_overchunking, chunking_context_mismatch,
    retrieval_failure
)
# parametric_overreliance 는 여기 없다 — 슬롯 밖 additive(diagnose 참조).
_GENERATION_CAUSE = (
    generation_wrongful_abstention,
    generation_abstention_failure, bad_gold_answer_oracle,
    # reasoning_failure 한 함수가 라벨 4개를 낸다
    # (contradiction/numerical_error/misinterpretation/hop_binding_error).
    generation_reasoning_failure,
    generation_hallucination, generation_partial_answer,
    generation_failure,
)
# chunking_context_mismatch 는 A·C 양쪽에 등록한다 — 경계 분할은 '검색이 gold 를 통째로
# 못 가져옴'(A)으로도, '검색은 됐는데 잘린 근거로 답이 틀림'(C)으로도 나타난다.
# A 슬롯에만 두면 recall=1 인 경계 분할 실패에서 도달 자체가 불가능하다(_dedup 이 중복 제거).
# 노이즈가 '청크 안'이면 underchunking, '청크 사이'면 reranker/noise_interference —
# _chunk_noise_heavy 로 배타가 서서 순서에 기대지 않는다.
# 단 reranker_low_precision 과 too_long_context/lost_in_the_middle 은 함께 성립할 수 있다
# (리랭커가 gold 를 중간으로 밀면 둘 다 참). 이때는 _pick 이 확정을 먼저 뽑아 그쪽이 채택된다 —
# reranker 는 인과(리랭크 전/후 대조)가 미측정이라 예비로 남기 때문이고, 순서가 아니라
# confirmed 로 갈린다. 다른 확정이 없을 때만 reranker 가 슬롯을 가져간다.
_CONTEXT_CAUSE = (
    bad_gold_answer, chunking_overchunking, chunking_context_mismatch,
    chunking_underchunking, reranker_low_precision,
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
    if label.startswith("chunking_") or label.startswith("reranker_"):
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
    "generation_hallucination", "generation_abstention_failure",   # 근거 없는데 지어냄 = 환각
    "generation_contradiction",                                    # 문맥과 정면 충돌 = 사실 오류
    "corpus_gap", "corpus_gap_partial_hop",
}
def _severity_of(label: str) -> str:
    if label in _CRITICAL_LABELS:
        return "critical"
    return "warning"


# gold 순위를 함께 저장해야하는 라벨들.
# candidate_miss 는 planner 가 이 순위로 후보창 목표값(무릎)을 계산한다 — top_k 근거값을
# 순위에서 뽑는 두 라벨과 같은 구조다.
_RANK_LABELS = {
    "retrieval_incomplete_enumeration",
    "retrieval_missing_gold",
    "retrieval_rerank_candidate_miss",
}


def _v(x) -> str:
    """reason 문자열용 값 포맷(float 은 소수 3자리, None 은 '-').

    3자리인 이유: 2자리면 임계값 비교가 자기모순처럼 읽힌다 — 실제 0.699 인 값이
    'context_precision=0.70<0.7' 로 찍혀 '같은 값인데 왜 실패?' 로 오독된다.
    """
    if x is None:
        return "-"
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def _answer_reason(record: EvalRecord) -> str:
    """reason 문자열용 정답 판정 근거 — 'answer=0.63(f1 0.49·의미 0.73)'.

    f1 만 적으면 오독을 부른다: 판정은 f1 단독이 아니라 혼합 점수(_answer_ok)라, f1 이 문턱
    아래인데 통과한(또는 f1 이 문턱 위인데 실패한) 라벨의 근거가 로그에서 사라진다.
    의미축 미측정(DEEP 미만)이면 lexical 단독 판정이므로 f1 만 적는다.

    단 의미축이 '심판 degrade' 로 빠진 경우는 그 값을 드러낸다 — degrade + 낮은 ac 는
    _degraded_near_miss 로 판정을 오답으로 뒤집는데(어휘 f1 이 높아도), 그 신호가 안 보이면
    'f1 완벽인데 실패'가 로그로 설명되지 않는다(실측: probe_qa_26360 계열). 미측정이 degrade
    때문인지(ac_degraded) 단순 저모드인지(f1 만)를 가른다. 실제로 그 강등이 걸렸는지까지
    적어야 '값은 낮은데 왜 통과했나(또는 그 반대)'를 한 줄로 읽을 수 있다."""
    semantic = record.answer_semantic
    if semantic is not None:
        return (f"answer={_v(record.answer_score)}"
                f"(f1 {_v(record.f1_score)}·의미 {_v(semantic)})")
    if record.ragas.get("answer_correctness_degraded"):
        demoted = "→강등" if _degraded_near_miss(record, oracle=False) else ""
        return (f"f1={_v(record.f1_score)}·의미측정실패"
                f"(ac_degraded={_v(record.ragas_answer_correctness)}{demoted})")
    return f"f1={_v(record.f1_score)}"


def _rollup_reason(record: EvalRecord) -> str:
    """롤업(그룹만 알고 원인은 모름) 공통 reason.

    구체 원인을 하나도 못 고른 이유는 대개 자원 부족이라, 아래 지표가 전부 '-' 로 비는
    저모드에서는 값만 봐선 '왜 롤업인지'를 알 수 없다 → 실행 모드를 함께 남긴다.
    """
    return (f"구체 원인 미실측(mode={active_mode()}), "
            f"oracle_f1={_v(record.oracle_f1)}, {_answer_reason(record)}, "
            f"faithfulness={_v(_faith_oracle(record))}, relevancy={_v(_rel_oracle(record))}")


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

    # 아래 두 단락(골드 오라벨·검증된 label-recall miss)은 골드가 코퍼스에 있을 때만 탄다.
    # gold 가 코퍼스에 없으면(_corpus_gap_premise) 없는 청크를 '재지정'할 수도(bad_gold_chunk),
    # '다른 유효 근거로 검증됐다'고 볼 수도(label-recall miss) 없다 — 단락하면 additive 인
    # corpus_gap/corpus_gap_partial_hop(자료 결손·누락 gold 표기)이 통째로 사라진다(리뷰 지적).
    # 이 경우 정상 경로로 흘려 corpus_gap 이 그 사실을 보고하게 양보한다.
    if not _corpus_gap_premise(record):
        # 골드 청크 오라벨: 실제 답은 맞는데(f1 통과) 골드 청크로는 못 맞춘 경우 — 검색·생성이
        # 아니라 골드 라벨이 틀린 것이다. 경쟁 슬롯의 거짓 원인(retrieval_low_rank·generation_*)을
        # 막고 이 하나만 남겨 검수로 보낸다(점수에서는 report 가 거짓 실패로 제외).
        chunk_mislabel = bad_gold_chunk(record)
        if chunk_mislabel is not None:
            return [chunk_mislabel]

        # 검증된 label-recall miss: 라벨 골드는 못 집었지만 답이 정답·검색 근거에 붙고 골드도
        # 유효하면, 검색은 다른 유효 근거로 정답을 뒷받침한 것이라 실패가 아니다. recall 스윙
        # (재청킹)만으로 pass/fail 이 뒤집히지 않게 성공 처리하고, 검색축 크레딧(faithfulness)을
        # record 에 남겨 reliability 가 같은 판정을 쓰게 한다(parametric·골드오류는 위에서 걸러짐).
        if _retrieval_verified_grounded(record):
            record.retrieval_axis = _faith(record)
            return []

    # 추가 진단
    findings = []
    if 0 <= record.recall_at_k < 1:                     # A: 검색 실패 (gold 있는데 일부 미검색)
        if _retrieval_fixable(record):                  # 코퍼스 밖·무응답 기대는 검색 몫이 아니다
            findings.append(_pick(record, _RETRIEVAL_CAUSE))
        findings.append(corpus_gap(record))             # D: 코퍼스에 gold 없음 (additive)
        findings.append(corpus_gap_partial_hop(record))
    if _generation_failed(record):                      # B: 생성 실패
        findings.append(_pick(record, _GENERATION_CAUSE))
    if _context_failed(record):                         # C: context 구조
        findings.append(_pick(record, _CONTEXT_CAUSE))
    # B: 정답이지만 근거 없음 (additive) — 전제가 '답이 맞음'이라 위 원인들과 경쟁하지 않는다.
    findings.append(generation_parametric_overreliance(record))
    # B: 기권 실패 (additive) — corpus_gap probe 는 gold context 가 없어 오라클 트랙이 안 돌고
    # _generation_failed 가 안 켜져 B 슬롯에 도달하지 못한다(슬롯 경유분은 _dedup 이 접는다).
    findings.append(generation_abstention_failure(record))

    findings = _dedup(_collect(*findings))
    findings.sort(key=lambda f: (
        _GROUP_ORDER.get(f.metadata.get("group"), 9),
        _SEV_ORDER.get(f.severity, 9),
    ))
    return findings
