"""
agents/eval/signals.py
판별 신호 레이어 — diagnose 의 라벨 함수가 호출하는 '신호·지표·전제' 계산을 모아둔다.

각 신호는 tri-state: None(미실행/모름) · True · False.
  · 비용 게이트: 비싼 신호는 첫 줄에서 self-gate (`if _active_mode < <tier>: return None`).
  · memoize: 비싼 신호는 _cache 로 record.signals(=state 캐시)에 저장 → 재진단 시 재사용.
  · 진단 계산(지표·RAGAS·재검색·재실행)은 전부 여기서 lazy 로 계산된다.

tier / 사용 자원:
  tier1 순수 규칙 · tier2 추가 검색 쿼리(top-N 재검색·BM25·코퍼스) · tier3 LLM/RAGAS · tier4 파이프라인 재실행.

diagnose() 가 진입 시 set_mode(mode) 로 현재 실행 모드를 설정하고, 이 모듈의 신호들이 그 값을
읽어 self-gate 한다. (단일스레드 eval 루프 전제. 병렬화하면 contextvars 로 교체.)
"""
from __future__ import annotations

from agents.eval.types import (
    Mode, EvalRecord, DEFAULT_TOP_K, F1_PASS_THRESHOLD,
    RAGAS_FAITHFULNESS_MIN, RAGAS_RESPONSE_RELEVANCY_MIN,
)
from agents.eval.metrics import recall_at_k, token_f1, is_abstention   # STEP3-1 지표(diagnose 이관)


# ── 진단 모드 (현재 실행의 tier 상한) — diagnose() 가 set_mode 로 설정 ──
_active_mode: int = Mode.FAST


def set_mode(mode: int) -> None:
    """diagnose() 진입 시 현재 실행 모드를 설정. 이하 tier 신호까지만 확정 가능(그 위는 예비)."""
    global _active_mode
    _active_mode = mode


def active_mode() -> int:
    """현재 실행 모드(신호 self-gate 기준). 필요 시 외부에서 조회용."""
    return _active_mode


# ── 진단 자원 컨텍스트 (tier2~4 훅이 쓸 검색·재생성 자원 — agent 가 set_context 로 주입) ──

class _Ctx:
    """
    tier2/tier4 판별 훅(재검색·코퍼스 조회·재생성)이 쓰는 자원. agent 가 set_context 로 주입한다.
    2단계: RAG/index module에서 값 및 함수들을 가져와야한다!!!!!!!!!!!
    """
    client = None
    chunks: list = []
    corpus_ids: frozenset = frozenset()
    retrieve_fn = None       # (client, chunks, question, top_n) -> list[{"chunk_id",...}]
    keyword_fn = None        # (chunks, query, top_n) -> list[{"chunk_id",...}]
    generate_fn = None       # (question, contexts) -> str   (tier4 ablation 재생성)
    ragas_fn = None          # (record, track) -> dict  track: "real"|"oracle"  (tier3 RAGAS lazy)
    wide_n: int = 100        # top-N 재검색·BM25 후보 크기


_ctx = _Ctx()


def set_context(client=None, chunks=None, retrieve_fn=None, keyword_fn=None,
                generate_fn=None, ragas_fn=None, wide_n=100):
    """tier2~4 판별 훅이 쓸 자원 주입. agent.run 이 진단 전 1회 호출.
    미주입이면 해당 훅은 자원 없음으로 None(=미확보) 반환."""
    _ctx.client = client
    _ctx.chunks = chunks or []
    _ctx.corpus_ids = frozenset(c.chunk_id for c in _ctx.chunks)
    _ctx.retrieve_fn = retrieve_fn
    _ctx.keyword_fn = keyword_fn
    _ctx.generate_fn = generate_fn
    _ctx.ragas_fn = ragas_fn
    _ctx.wide_n = wide_n


# ── memoize ──────────────────────────────────────────────────────

def _cache(record: EvalRecord, name: str, compute):
    """ 판별 신호 memoize를 위한 함수.

     1) record.signals(=state.diagnosis_cache[probe_id] 뷰)에 있으면 재사용,
     2) 없으면 compute() 계산해 저장.
     """
    cache = record.signals
    if name not in cache:
        cache[name] = compute()
    return cache[name]


# ── STEP3-1 규칙 지표 (diagnose 진입 시 계산·저장) ────────────────

def _compute_metrics(record: EvalRecord) -> None:
    """규칙 지표(recall/f1/oracle_f1)를 record 에 계산·저장. (agent STEP3-1 이관, diagnose 진입 시 1회.)
    전제 헬퍼·report 가 record.recall_at_k / f1_score / oracle_f1 로 읽는다."""
    gt = record.probe.ground_truth
    record.recall_at_k = recall_at_k(record.probe.gold_chunk_ids, record.retrieved_chunk_ids)
    record.f1_score = token_f1(record.generated_answer, gt) if gt else 0.0
    record.oracle_f1 = token_f1(record.oracle_answer, gt) if (gt and record.oracle_answer) else 0.0


# ── 전제 신호 (브랜치 대체 — 각 슬롯이 언제 적용되는지) ───────────

def _recall_ok(record: EvalRecord) -> bool:
    """recall_at_k를 검사해서 threshold를 넘기는지 검사. 매우 간단한데 일관성을 위해 따로 분리"""
    return record.recall_at_k >= 1


def _retrieval_failed(record: EvalRecord) -> bool:
    """gold 가 있는데 top-k 로 다 못 가져옴(0 <= recall < 1). 검색 원인(A) 공통 전제.
    (recall == -1 = gold 없음 → 검색 실패 아님.)"""
    return 0 <= record.recall_at_k < 1


def _f1_ok(record: EvalRecord) -> bool:
    """실제 답이 정답과 일치(token_f1 통과). ground_truth 없으면 판정 불가 → False."""
    return bool(record.probe.ground_truth) and record.f1_score >= F1_PASS_THRESHOLD


def _oracle_ok(record: EvalRecord) -> bool:
    """gold 컨텍스트로 생성한 답이 정답과 일치(oracle_f1 통과). oracle 답 없으면 False."""
    return record.oracle_answer is not None and record.oracle_f1 >= F1_PASS_THRESHOLD


def _generation_failed(record: EvalRecord) -> bool:
    """순수 생성 실패 전제(B그룹 공통, 브랜치 대신):
    gold 컨텍스트로도 답이 틀림(oracle 실패), 또는 무응답인데 답을 지어냄."""
    if record.oracle_answer is not None and not _oracle_ok(record):
        return True
    if record.probe.answer_exists is False and not is_abstention(record.generated_answer):
        return True
    return False


def _context_applicable(record: EvalRecord) -> bool:
    """컨텍스트 구조 문제(C) 전제: 검색 성공(recall=1)·생성 가능(oracle 통과)인데 실제 답만 틀림."""
    return _recall_ok(record) and _oracle_ok(record) and not _f1_ok(record)


def _no_diagnosis(record: EvalRecord) -> bool:
    """진단 불필요(= 예전 Branch.SUCCESS/NO_ANSWER_OK): 정답셋 없음 / 올바른 무응답 / 성공."""
    if not record.probe.ground_truth:
        return True
    if record.probe.answer_exists is False and is_abstention(record.generated_answer):
        return True
    return _recall_ok(record) and _f1_ok(record)


# ── tier1 · 순수 규칙 (자원: 이미 계산된 지표/probe 메타 — 추가 조회 없음) ──
# (recall_at_k < 1 도 tier1 순수 규칙 — missing_gold / bridge 가 라벨 함수에서 직접 사용)

def _is_multi_hop(record: EvalRecord) -> bool:
    """멀티홉 질문 여부(probe.qtype). bridge / hop_binding / corpus_gap_partial_hop 판별."""
    return record.probe.qtype in ("bridge", "comparison", "aggregation")


def _enumeration_cache(record: EvalRecord) -> bool:
    """gold 개수가 top-k(=검색 결과 수)에 근접/초과 → 나열형 누락. incomplete_enumeration 용."""
    k = len(record.retrieved_chunk_ids) or DEFAULT_TOP_K
    gold_n = len(record.probe.gold_chunk_ids)
    return gold_n >= 2 and gold_n >= int(k * 0.8)


# ── tier2 · 추가 검색 쿼리 (자원: top-N 재검색 / BM25 / 코퍼스 조회) — set_context 로 주입 ──

def _gold_in_wider_candidates(record: EvalRecord):
    """
    top-N 재검색에서, top-k 가 놓친 gold 가 넓은 후보엔 있나 확인.
    retrieval_low_rank 확정용.
    True=놓친 gold 찾음 / False=후보에도 없음 / None=자원·모드 미충족.
    """
    if _active_mode < Mode.STANDARD or _ctx.retrieve_fn is None:
        return None

    def compute():
        missed = set(record.probe.gold_chunk_ids) - set(record.retrieved_chunk_ids) # 차집합 - 놓친 골드
        if not missed:
            return None
        hits = _ctx.retrieve_fn(_ctx.client, _ctx.chunks, record.probe.question, _ctx.wide_n) # 넓은 n으로 검색
        wide_ids = {h.get("chunk_id") for h in hits}
        return bool(missed & wide_ids) # 교집합 - 하나라도 찾았으면 True.

    return _cache(record, "gold_in_wider_candidates", compute)


def _bm25_hits_gold(record: EvalRecord):
    """
    키워드(BM25) 검색이 dense top-k 가 놓친 gold 를 잡나.
    lexical(True)/semantic(False) mismatch 용.
    True=키워드로 잡힘(단어 불일치) / False=키워드도 놓침(의미 불일치) / None=자원·모드 미충족.
    """
    if _active_mode < Mode.STANDARD or _ctx.keyword_fn is None:
        return None

    def compute():
        missed = set(record.probe.gold_chunk_ids) - set(record.retrieved_chunk_ids)
        if not missed:
            return None
        hits = _ctx.keyword_fn(_ctx.chunks, record.probe.question, _ctx.wide_n) # 위와 같으나 검색 함수만 다름
        kw_ids = {h.get("chunk_id") for h in hits}
        return bool(missed & kw_ids)
    return _cache(record, "bm25_hits_gold", compute)


def _gold_in_corpus(record: EvalRecord):
    """gold 가 코퍼스에 존재하나(멤버십 조회). True→missing_gold / False→corpus_gap.
    gold 전부 존재 True / 하나라도 없으면 False / gold·자원 없으면 None."""
    if _active_mode < Mode.STANDARD or not _ctx.corpus_ids:
        return None

    def compute():
        golds = record.probe.gold_chunk_ids
        if not golds:
            return None
        return all(g in _ctx.corpus_ids for g in golds) # 코퍼스 전체와 대조
    return _cache(record, "gold_in_corpus", compute)


# ── tier3 · LLM/RAGAS (자원: set_context.ragas_fn 로 lazy 계산) ──
#   실제 트랙  = record.ragas       (검색결과 컨텍스트로 생성한 답)
#   오라클 트랙 = record.oracle_ragas (gold 컨텍스트로 생성한 답)
# 생성 원인(hallucination/hop_binding/partial)은 항상 오라클, bad_gold만 각 트랙 사용.

def _ensure_ragas(record: EvalRecord, track: str):
    """트랙 RAGAS 점수를 record 에 lazy 계산·저장(트랙별 1회만). agent 가 미리 안 돌리고
    diagnose 가 필요할 때 _ctx.ragas_fn 으로 계산 → '진단 계산은 전부 diagnose 안'.
    빈 결과({})여도 *_done 플래그로 '시도함'을 기록해 같은 트랙 재-LLM호출(수 번의 LLM콜)을 막는다."""
    if _ctx.ragas_fn is None:
        return
    if track == "oracle":
        if not record.oracle_ragas_done:
            record.oracle_ragas_done = True
            record.oracle_ragas = _ctx.ragas_fn(record, "oracle") or {}
    elif not record.ragas_done:
        record.ragas_done = True
        record.ragas = _ctx.ragas_fn(record, "real") or {}


def _faith(record: EvalRecord):
    """faithfulness(충실도) — 실제 트랙 (lazy)."""
    if _active_mode < Mode.DEEP:       # 비용 게이트
        return None
    _ensure_ragas(record, "real")
    return record.ragas.get("faithfulness")


def _faith_oracle(record: EvalRecord):
    """faithfulness(충실도) — 오라클 트랙 (lazy)."""
    if _active_mode < Mode.DEEP:
        return None
    _ensure_ragas(record, "oracle")
    return record.oracle_ragas.get("faithfulness")


def _rel(record: EvalRecord):
    """response_relevancy(관련성) — 실제 트랙 (lazy)."""
    if _active_mode < Mode.DEEP:
        return None
    _ensure_ragas(record, "real")
    return record.ragas.get("response_relevancy")


def _rel_oracle(record: EvalRecord):
    """response_relevancy(관련성) — 오라클 트랙 (lazy)."""
    if _active_mode < Mode.DEEP:
        return None
    _ensure_ragas(record, "oracle")
    return record.oracle_ragas.get("response_relevancy")


def _both_high(faith, rel) -> bool:
    """bad_gold_answer 판정용: 충실도·관련성이 모두 '측정되어' 임계값 이상."""
    if _active_mode < Mode.DEEP:
        return None
    return (faith is not None and faith >= RAGAS_FAITHFULNESS_MIN
            and rel is not None and rel >= RAGAS_RESPONSE_RELEVANCY_MIN)


# ── tier4 · 파이프라인 재실행 (자원: ablation 재생성/재검색) — set_context 로 주입 ──
# context 를 수정(축소/재정렬/노이즈제거)해 재생성한 답의 token_f1 이 baseline 보다 오르면
# 그 수정이 '원인을 제거'한 것 → 확정. bridge 는 1차 근거로 질의를 확장해 재검색.
_ABLATION_MARGIN = 0.1   # baseline(record.f1_score) 대비 이 이상 개선돼야 '도움됨(True)'


def _ablation_helps(record: EvalRecord, contexts: list):
    """수정된 contexts 로 재생성한 답의 token_f1 이 baseline 대비 _ABLATION_MARGIN 이상 개선되나.
    True=개선(그 수정이 원인 제거) / False=미개선 / None=생성함수·정답 없음."""
    gen = _ctx.generate_fn
    gt = record.probe.ground_truth
    if gen is None or not gt or not contexts:
        return None
    new_f1 = token_f1(gen(record.probe.question, contexts), gt)
    return new_f1 >= record.f1_score + _ABLATION_MARGIN


def _context_shorten_helps(record: EvalRecord):
    """[tier4] context 를 절반으로 줄여 재생성 시 f1 개선되나. too_long_context 확정용."""
    if _active_mode < Mode.FULL or _ctx.generate_fn is None:
        return None

    def compute():
        ctx = record.retrieved_context
        if len(ctx) <= 2:
            return None                        # 이미 짧음 → 축소 무의미
        return _ablation_helps(record, ctx[:len(ctx) // 2])
    return _cache(record, "context_shorten_helps", compute)


def _gold_front_helps(record: EvalRecord):
    """[tier4] gold 청크를 맨 앞으로 재정렬 후 재생성 시 f1 회복되나. lost_in_the_middle 확정용."""
    if _active_mode < Mode.FULL or _ctx.generate_fn is None:
        return None

    def compute():
        golds = set(record.probe.gold_chunk_ids)
        pairs = list(zip(record.retrieved_chunk_ids, record.retrieved_context))
        front = [t for i, t in pairs if i in golds]
        rest = [t for i, t in pairs if i not in golds]
        if not front or not rest:
            return None                        # gold 없거나 재정렬 무의미
        reordered = front + rest
        if reordered == record.retrieved_context:
            return None                        # 이미 gold 가 앞
        return _ablation_helps(record, reordered)
    return _cache(record, "gold_front_helps", compute)


def _noise_removal_helps(record: EvalRecord):
    """[tier4] 비-gold(노이즈) 청크 제거 시 f1 회복되나. context_noise 확정용.
    C그룹은 recall=1 이라 gold-only == oracle context → 이미 계산된 oracle_f1 재사용(재생성 불필요)."""
    if _active_mode < Mode.FULL:
        return None

    def compute():
        if not record.probe.ground_truth or record.oracle_answer is None:
            return None
        golds = set(record.probe.gold_chunk_ids)
        if not any(i not in golds for i in record.retrieved_chunk_ids):
            return None                        # 제거할 노이즈(비-gold)가 없음
        return record.oracle_f1 >= record.f1_score + _ABLATION_MARGIN
    return _cache(record, "noise_removal_helps", compute)


def _bridge_decompose_recovers(record: EvalRecord):
    """[tier4] 1차 근거로 질의를 확장(연쇄)해 재검색 시 놓친 gold 를 회복하나. missing_bridge 확정용.
    단 plain 재검색(low_rank)으로 이미 잡히면 bridge 아님 → False."""
    if _active_mode < Mode.FULL or _ctx.retrieve_fn is None:
        return None

    def compute():
        missed = set(record.probe.gold_chunk_ids) - set(record.retrieved_chunk_ids)
        if not missed:
            return None
        if _gold_in_wider_candidates(record) is True:
            return False                       # plain 재검색으로 잡힘 → low_rank 소관
        expanded = record.probe.question + " " + " ".join(record.retrieved_context[:2])
        hits = _ctx.retrieve_fn(_ctx.client, _ctx.chunks, expanded, _ctx.wide_n)
        hop2_ids = {h.get("chunk_id") for h in hits}
        return bool(missed & hop2_ids)
    return _cache(record, "bridge_decompose_recovers", compute)
