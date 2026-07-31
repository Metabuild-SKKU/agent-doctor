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
    F1_PASS_THRESHOLD, ANSWER_PASS_THRESHOLD, ANSWER_SEMANTIC_FLOOR,
    ANSWER_CORRECTNESS_MIN, EVIDENCE_DENSITY_MIN,
    CONTEXT_CHARS_MAX, CONTEXT_MIDDLE_BAND,
    RAGAS_FAITHFULNESS_MIN, RAGAS_RESPONSE_RELEVANCY_MIN, RAGAS_CONTEXT_PRECISION_MIN,
)
from agents.eval.metrics_common import set_mode, set_context, active_mode, missed_gold_ids
from agents.eval.metrics_basic import (            # tier1
    is_abstention, _compute_metrics, _gold_span_boundary_analysis,
    _gold_chunk_evidence_density, _oversized_gold_spans,
    _context_char_total, _gold_position_band,
)
from agents.eval.metrics_search import (           # tier2
    _gold_ranks, _bm25_hits_gold, _gold_in_corpus, _missed_gold_in_corpus,
    _gold_absent_from_corpus, _gold_corpus_membership,
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
    """
    ragas = record.oracle_ragas if oracle else record.ragas
    if not ragas.get("answer_correctness_degraded"):
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


def retrieval_low_rank(record: EvalRecord) -> Optional[Finding]:
    """
    gold가 top-N 후보엔 있으나 순위가 낮아 top-k 밖.
    확정: 놓친 gold 가 wide-N 재검색에서 top_k 보다 뒤 순위로 발견(tier2).

    순위가 top_k 이내로 나오면 '순위가 낮아 밖'과 모순이라 제외한다 — 그건 검색 비결정성이나
    인덱스 변경이지 순위 문제가 아니다(결정적 리트리버면 발생하지 않는다).
    """
    ranks = _gold_ranks(record)
    if ranks is None:
        return None
    top_k = len(record.retrieved_chunk_ids)
    if top_k <= 0:
        return None                      # 검색 0건 → 순위 문제가 아니라 검색 장애
    beyond = {g: ranks[g] for g in missed_gold_ids(record)
              if ranks.get(g) is not None and ranks[g] > top_k}
    if not beyond:
        return None
    ranked = ", ".join(f"{g}:{r}" for g, r in sorted(beyond.items(), key=lambda kv: kv[1]))
    return _finding(
        record, "retrieval_low_rank", "retrieval_failure", confirmed=True,
        reason=f"missed_gold_ranks=[{ranked}] > top_k={top_k}, recall@k={_v(record.recall_at_k)}",
    )


def retrieval_lexical_mismatch(record: EvalRecord) -> Optional[Finding]:
    """
    dense는 놓쳤으나 BM25로 잡히는 단어 불일치.
    확정: BM25 가 gold 를 잡음 + dense wide-N 도 그 gold 를 못 잡음(tier2).
    dense wide-N 에 있으면 low_rank 영역 — 튜플 순서 대신 함수 자체로 배타(ranks memoize 공유).
    놓친 gold 가 여럿이어도 그중 하나만 BM25 에 잡히면 probe 전체가 이 라벨이다(슬롯당 1원인).
    남은 semantic 실패는 이번 처방 적용 후 다음 iteration 에서 다시 잡힌다.
    """
    if _bm25_hits_gold(record) is not True:
        return None
    ranks = _gold_ranks(record) or {}
    if any(ranks.get(g) is not None for g in missed_gold_ids(record)):
        return None                      # dense wide-N 에 있음 → low_rank
    return _finding(
        record, "retrieval_lexical_mismatch", "retrieval_failure", confirmed=True,
        reason=f"bm25_hits_gold=True, dense_missed=True, recall@k={_v(record.recall_at_k)}",
    )


def retrieval_semantic_mismatch(record: EvalRecord) -> Optional[Finding]:
    """
    dense·BM25 모두 놓친 의미 연결 실패. (단 gold 가 코퍼스엔 있을 때만 — 없으면 corpus_gap)
    확정: BM25 도 gold 를 못 잡음 + 놓친 gold 중 코퍼스에 있는 게 있음(tier2).
    멤버십은 gold 별로 본다 — probe 단위 bool 로 접으면 혼합 코퍼스(놓친 gold 가 {있음+없음})
    에서 라벨이 통째로 소실된다. 코퍼스에 없는 몫은 corpus_gap 이 additive 로 함께 붙는다.
    코퍼스 멤버십 미측정(None)은 corpus_gap 과 구분 불가라 예비(missing_gold 와 동일 기준).
    qtype=bridge 는 bridge 의존과 구분 불가라 양보(원 질문으론 hop2 를 원래 못 찾음).
    """
    if record.probe.qtype == "bridge":
        return None                      # bridge 의존과 구분 불가 → bridge 에 양보
    if _bm25_hits_gold(record) is not False:
        return None
    in_corpus = _missed_gold_in_corpus(record)
    if in_corpus is None:
        return _finding(
            record, "retrieval_semantic_mismatch", "retrieval_failure", confirmed=False,
            reason=f"bm25_hits_gold=False, missed_gold_in_corpus=-, recall@k={_v(record.recall_at_k)}",
        )
    if in_corpus:
        return _finding(
            record, "retrieval_semantic_mismatch", "retrieval_failure", confirmed=True,
            reason=f"bm25_hits_gold=False, "
                   f"missed_gold_in_corpus={len(in_corpus)}/{len(missed_gold_ids(record))}, "
                   f"recall@k={_v(record.recall_at_k)}",
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
            reason=f"missed_gold_in_corpus=-, recall@k={_v(record.recall_at_k)}",
        )
    if in_corpus:
        return _finding(
            record, "retrieval_missing_gold", "retrieval_failure", confirmed=True,
            reason=f"missed_gold_in_corpus={len(in_corpus)}/{len(missed_gold_ids(record))}, "
                   f"recall@k={_v(record.recall_at_k)}",
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
               f"oversized={analysis['oversized_count']}, recall@k={_v(record.recall_at_k)}",
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
               f"recall@k={_v(record.recall_at_k)}, {_answer_reason(record)}",
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
        reason=f"qtype={record.probe.qtype}, recall@k={_v(record.recall_at_k)}",
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
               f"recall@k={_v(record.recall_at_k)}",
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
    return False


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
               f"{_answer_reason(record)}(정답), recall@k={_v(record.recall_at_k)}",
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
    """컨텍스트 구조 문제(C) 전제: 검색 성공(recall=1)·생성 가능(oracle 통과)인데 실제 답만 틀림."""
    return _recall_ok(record) and _oracle_ok(record) and not _f1_ok(record)

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
               f"recall@k={_v(record.recall_at_k)}",
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
               f"recall@k={_v(record.recall_at_k)}",
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
               f"recall@k={_v(record.recall_at_k)}",
    )


def reranker_low_precision(record: EvalRecord) -> Optional[Finding]:
    """
    리랭커가 무관한 청크를 상위로 올림.
    예비: C 전제 + 리랭크가 실제 적용됨 + context_precision 낮음 + 청크 안 노이즈는 아님.

    C그룹에서 유일하게 예비로 남는 라벨이다 — 확정하려면 리랭크 전/후 순위를 대조해야 하는데
    retriever 가 리랭크 전 후보를 남기지 않는다(search_with_details 가 results 를 덮어씀).
    그래서 '리랭크를 거친 결과의 정밀도가 낮다'까지만 말한다(리랭커가 원인이라는 증거는 아니다 —
    원래 검색이 나빴을 수도 있다). 전/후 기록을 남기면 그때 확정으로 올릴 수 있다(별도 PR).
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
    return _finding(
        record, "reranker_low_precision", "retrieval_failure", confirmed=False,
        reason=f"reranked=True, context_precision={_v(precision)}<{RAGAS_CONTEXT_PRECISION_MIN}, "
               f"recall@k={_v(record.recall_at_k)}",
    )


def context_noise_interference(record: EvalRecord) -> Optional[Finding]:
    """
    비-gold 청크의 상충 정보에 이끌림.
    확정: C 전제(recall=1·oracle 통과·실제 답 틀림) + 실제 답이 검색 context 에는 근거 있음
          (real faithfulness 높음) → gold 아닌 청크에 근거했다는 뜻.

    faithfulness 는 retrieved_context(gold+노이즈) 기준이라, 노이즈 청크의 정보를 가져다 쓰면
    '근거 있음'으로 높게 나온다. 낮은 쪽은 gold·노이즈 어디에도 없는 생성측 이탈이라 다른 원인이다.
    처방(enable_noise_filter/mmr)은 rules.py draft — filtering/MMR/reranker 필드 합의 미완.
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
               f"recall@k={_v(record.recall_at_k)}, {_answer_reason(record)}",
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
               f"qtype={record.probe.qtype}, recall@k={_v(record.recall_at_k)}",
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

_RETRIEVAL_CAUSE = (
    retrieval_incomplete_enumeration, retrieval_missing_bridge_dependency,
    retrieval_low_rank, retrieval_lexical_mismatch, retrieval_semantic_mismatch, retrieval_missing_gold,
    # chunking 은 확정이지만 맨 뒤 — 실측된 다른 검색 원인이 있으면 그쪽을 먼저 채택한다.
    chunking_overchunking, chunking_context_mismatch,
    retrieval_failure
)
# parametric_overreliance 는 여기 없다 — 슬롯 밖 additive(diagnose 참조).
_GENERATION_CAUSE = (
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
_RANK_LABELS = {
    "retrieval_incomplete_enumeration",
    "retrieval_missing_gold",
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
    의미축 미측정(DEEP 미만)이면 lexical 단독 판정이므로 f1 만 적는다."""
    semantic = record.answer_semantic
    if semantic is None:
        return f"f1={_v(record.f1_score)}"
    return (f"answer={_v(record.answer_score)}"
            f"(f1 {_v(record.f1_score)}·의미 {_v(semantic)})")


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
