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
읽어 self-gate 한다. (단일스레드 전제 — STEP2 병렬화(agent.py) 이후에도 diagnose 는
Phase C 순차 구간에서만 실행되므로 유효하다. probe 전체를 스레드화하려면 _active_mode 를
contextvars 로 교체해야 한다.)
"""
from __future__ import annotations

from agents.eval.types import (
    Mode, EvalRecord, DEFAULT_TOP_K, F1_PASS_THRESHOLD,
    RAGAS_FAITHFULNESS_MIN, RAGAS_RESPONSE_RELEVANCY_MIN,
)
from agents.eval.metrics import (
    recall_at_k,
    span_recall_at_k,
    token_f1,
    is_abstention,
)


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


# ── STEP3 지표 (diagnose 진입 시 계산·저장 — 판정 전, 스킵 없음) ───

def _compute_metrics(record: EvalRecord) -> None:
    """규칙 지표(recall/f1/oracle_f1)를 record 에 계산·저장. (agent STEP3-1 이관, diagnose 진입 시 1회.)
    전제 헬퍼·report 가 record.recall_at_k / f1_score / oracle_f1 로 읽는다."""
    gt = record.probe.ground_truth
    span_recall = span_recall_at_k(
        record.probe.gold_spans,
        record.retrieved_chunk_ids,
        _ctx.chunks,
    )
    record.recall_at_k = (
        span_recall
        if span_recall is not None
        else recall_at_k(record.probe.gold_chunk_ids, record.retrieved_chunk_ids)
    )
    record.f1_score = token_f1(record.generated_answer, gt) if gt else 0.0
    record.oracle_f1 = token_f1(record.oracle_answer, gt) if (gt and record.oracle_answer) else 0.0


def _compute_ragas(record: EvalRecord) -> None:
    """RAGAS 점수(실제·오라클 트랙)를 record 에 계산·저장. (STEP3-2, diagnose 진입 시 1회.)

    _compute_metrics 와 같은 자리에서 항상 돌린다 — 진단이 필요 없는 probe(성공·정답셋 없음·
    올바른 무응답)도 faithfulness/response_relevancy 를 갖게 된다. 예전엔 라벨 함수가 필요할 때만
    lazy 로 불러서, report 의 RAGAS 평균이 '진단이 돌아간 실패 probe'만의 평균이었다.

    비용 게이트는 DEEP 유지 — 그 미만 모드에선 LLM 을 한 번도 부르지 않는다.
    이후 _faith/_rel 등의 lazy 호출은 *_done 플래그에 걸려 재호출되지 않는다."""
    if _active_mode < Mode.DEEP:
        return
    _ensure_ragas(record, "real")
    _ensure_ragas(record, "oracle")


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
    """진단 불필요(= 예전 Branch.SUCCESS/NO_ANSWER_OK): 올바른 무응답 / 정답셋 없음 / 성공.

    무응답 기대(answer_exists=False) probe 는 정답셋(ground_truth)이 없더라도 먼저 판정한다 —
    올바르게 회피하면 통과, 답을 지어내면 진단 대상(B그룹 생성실패)이다. 이 순서를 뒤집으면
    'ground_truth 없음 → 무조건 통과'에 걸려 무응답 지어냄이 조용히 통과 처리된다."""
    if record.probe.answer_exists is False:
        return is_abstention(record.generated_answer)
    if not record.probe.ground_truth:
        return True
    return _recall_ok(record) and _f1_ok(record)


def _chunk_char_span(chunk) -> tuple[int, int] | None:
    """현재 청크의 원문 절대좌표를 안전하게 읽는다."""

    raw = getattr(chunk, "char_span", None)
    if raw is None and isinstance(getattr(chunk, "metadata", None), dict):
        raw = chunk.metadata.get("char_span")
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != 2
        or isinstance(raw[0], bool)
        or isinstance(raw[1], bool)
        or not isinstance(raw[0], int)
        or not isinstance(raw[1], int)
        or raw[0] < 0
        or raw[1] <= raw[0]
    ):
        return None
    return raw[0], raw[1]


def _exact_probe_gold_spans(record: EvalRecord) -> list[dict]:
    """경계 진단에 사용할 exact gold span만 고른다.

    chunk_fallback은 기존 청크 전체를 정답 위치로 대신 기록한 값이라 경계가
    잘렸는지 판정할 근거가 될 수 없다. 품질 메타데이터가 없는 기존 Probe는
    하위 호환을 위해 exact로 취급한다.
    """

    grounding = record.probe.metadata.get("span_grounding", {})
    if not isinstance(grounding, dict):
        grounding = {}
    raw_qualities = grounding.get("span_qualities")
    qualities = raw_qualities if isinstance(raw_qualities, list) else []
    status = grounding.get("status")
    spans: list[dict] = []
    for index, span in enumerate(record.probe.gold_spans):
        if not isinstance(span, dict):
            continue
        doc_id = span.get("doc_id")
        start = span.get("start")
        end = span.get("end")
        if (
            not isinstance(doc_id, str)
            or isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            continue
        quality = qualities[index] if index < len(qualities) else None
        if quality == "chunk_fallback" or (
            quality is None and status in {"chunk_fallback", "partial"}
        ):
            continue
        spans.append({"doc_id": doc_id, "start": start, "end": end})
    return spans


def _gold_span_boundary_analysis(record: EvalRecord):
    """gold span이 현재 인접 청크 경계에 나뉘었는지 저비용으로 분석한다.

    LLM이나 추가 검색을 호출하지 않고, Eval이 이미 가진 원문 절대좌표만 쓴다.
    한 청크가 span 전체를 포함하면 정상이고, 그렇지 않지만 현재 청크들의 합집합이
    span 전체를 덮으면 경계 분할로 본다. 좌표가 없는 환경은 미확정(None)이다.
    """

    if not _ctx.chunks:
        return None

    def compute():
        spans = _exact_probe_gold_spans(record)
        if not spans:
            return None
        chunks_by_doc: dict[str, list[tuple[int, int]]] = {}
        for chunk in _ctx.chunks:
            position = _chunk_char_span(chunk)
            doc_id = getattr(chunk, "doc_id", None)
            if position is None or not isinstance(doc_id, str):
                continue
            chunks_by_doc.setdefault(doc_id, []).append(position)
        for positions in chunks_by_doc.values():
            positions.sort()

        contained_count = 0
        split_count = 0
        uncovered_count = 0
        for span in spans:
            start, end = span["start"], span["end"]
            positions = chunks_by_doc.get(span["doc_id"], [])
            if any(c_start <= start and c_end >= end for c_start, c_end in positions):
                contained_count += 1
                continue

            intersections = sorted(
                (max(start, c_start), min(end, c_end))
                for c_start, c_end in positions
                if c_start < end and c_end > start
            )
            cursor = start
            for covered_start, covered_end in intersections:
                if covered_start > cursor:
                    break
                cursor = max(cursor, covered_end)
                if cursor >= end:
                    break
            if intersections and cursor >= end:
                split_count += 1
            else:
                uncovered_count += 1

        return {
            "span_count": len(spans),
            "contained_count": contained_count,
            "boundary_split_count": split_count,
            "uncovered_count": uncovered_count,
        }

    return _cache(record, "gold_span_boundary_analysis", compute)


def _merged_span_text(span: dict, chunks: list) -> str | None:
    """검색된 청크 조각을 원문 좌표 순서로 이어 gold span 텍스트를 복원한다."""

    start, end = span["start"], span["end"]
    pieces: list[tuple[int, int, str]] = []
    for chunk in chunks:
        if getattr(chunk, "doc_id", None) != span["doc_id"]:
            continue
        position = _chunk_char_span(chunk)
        text = getattr(chunk, "text", None)
        if position is None or not isinstance(text, str):
            continue
        chunk_start, chunk_end = position
        if len(text) != chunk_end - chunk_start:
            # 좌표와 텍스트 길이가 다르면 안전하게 재구성할 수 없다.
            continue
        piece_start = max(start, chunk_start)
        piece_end = min(end, chunk_end)
        if piece_start < piece_end:
            pieces.append((
                piece_start,
                piece_end,
                text[piece_start - chunk_start:piece_end - chunk_start],
            ))

    cursor = start
    merged: list[str] = []
    while cursor < end:
        eligible = [piece for piece in pieces if piece[0] <= cursor < piece[1]]
        if not eligible:
            return None
        piece_start, piece_end, text = max(eligible, key=lambda piece: piece[1])
        offset = cursor - piece_start
        take = piece_end - cursor
        merged.append(text[offset:offset + take])
        cursor = piece_end
    return "".join(merged)


def _boundary_merged_context(record: EvalRecord) -> list[str] | None:
    """검색 context에서 분할 gold 조각만 하나의 연속 문맥으로 교체한다."""

    retrieved_by_id = {
        chunk.chunk_id: chunk
        for chunk in _ctx.chunks
        if chunk.chunk_id in set(record.retrieved_chunk_ids)
    }
    retrieved_chunks = list(retrieved_by_id.values())
    merged_texts: list[str] = []
    affected_ids: set[str] = set()
    for span in _exact_probe_gold_spans(record):
        positions = [
            (_chunk_char_span(chunk), chunk)
            for chunk in retrieved_chunks
            if chunk.doc_id == span["doc_id"]
        ]
        if any(
            position is not None
            and position[0] <= span["start"]
            and position[1] >= span["end"]
            for position, _chunk in positions
        ):
            continue
        merged = _merged_span_text(span, retrieved_chunks)
        if not merged:
            return None
        merged_texts.append(merged)
        for position, chunk in positions:
            if (
                position is not None
                and position[0] < span["end"]
                and position[1] > span["start"]
            ):
                affected_ids.add(chunk.chunk_id)
    if not merged_texts or not affected_ids:
        return None

    affected_indices = [
        index
        for index, chunk_id in enumerate(record.retrieved_chunk_ids)
        if chunk_id in affected_ids
    ]
    if not affected_indices:
        return None
    first = min(affected_indices)
    contexts: list[str] = []
    for index, (chunk_id, context) in enumerate(
        zip(record.retrieved_chunk_ids, record.retrieved_context)
    ):
        if index == first:
            contexts.extend(merged_texts)
        if chunk_id not in affected_ids:
            contexts.append(context)
    return contexts


def _boundary_merge_helps(record: EvalRecord):
    """[tier4] 분할된 gold 조각만 이어 붙이면 실제 답변 품질이 회복되는지 확인한다."""

    if _active_mode < Mode.FULL or _ctx.generate_fn is None:
        return None

    def compute():
        contexts = _boundary_merged_context(record)
        if not contexts:
            return None
        return _ablation_helps(record, contexts)

    return _cache(record, "boundary_merge_helps", compute)


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

def _wide_hits(record: EvalRecord):
    """top-N(wide_n) 재검색 결과(순위 내림차순 정렬)를 probe 당 1회만 계산·공유.

    같은 질문의 wide 재검색은 gold_in_wider_candidates(존재 여부)와 gold_ranks(순위)가
    함께 필요로 한다. 검색 1회를 memoize 로 공유해 tier2 비용을 중복 지불하지 않는다.
    True 결과가 아니라 원본 hits(list[dict{"chunk_id",...}])를 그대로 캐시한다.
    None=자원·모드 미충족.
    """
    if _active_mode < Mode.STANDARD or _ctx.retrieve_fn is None:
        return None

    def compute():
        return _ctx.retrieve_fn(
            _ctx.client, _ctx.chunks, record.probe.question, _ctx.wide_n
        )

    return _cache(record, "wide_hits", compute)


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
        hits = _wide_hits(record) # 넓은 n으로 검색(memoize 공유)
        wide_ids = {h.get("chunk_id") for h in hits}
        return bool(missed & wide_ids) # 교집합 - 하나라도 찾았으면 True.

    return _cache(record, "gold_in_wider_candidates", compute)


def _gold_ranks(record: EvalRecord):
    """probe 의 각 gold 청크가 wide_n 재검색에서 몇 위인지(1-based) 매핑.

    planner 가 top_k 근거값을 계산할 원시 순위 측정치다(집계·후보화는 planner 소관).
    "gold 가 5개니 top_k=5" 같은 개수 추정과 달리, "가장 늦게 나오는 gold 가 20위면
    top_k 는 최소 20" 이라는 실측을 준다(multi-hop/나열형에서 개수 ≪ 순위).

    반환: {gold_id: rank}  rank 는 1-based, wide_n 밖이면 None(=top_k 로 도달 불가).
          gold 없음 → None / 모드·자원 미충족 → None.
    """
    if _active_mode < Mode.STANDARD or _ctx.retrieve_fn is None:
        return None

    def compute():
        golds = record.probe.gold_chunk_ids
        if not golds:
            return None
        hits = _wide_hits(record)
        if hits is None:
            return None
        order = {h.get("chunk_id"): i + 1 for i, h in enumerate(hits)}  # 1-based 순위
        return {g: order.get(g) for g in golds}  # wide_n 밖이면 None

    return _cache(record, "gold_ranks", compute)


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
    """트랙 RAGAS 점수를 record 에 계산·저장(트랙별 1회만). 실제로는 diagnose 진입 시
    _compute_ragas 가 두 트랙을 먼저 채우고, 아래 신호들의 호출은 그 결과를 재사용한다.
    빈 결과({})여도 *_done 플래그로 '시도함'을 기록해 같은 트랙 재-LLM호출(수 번의 LLM콜)을 막는다.
    (oracle 답이 없으면 _ctx.ragas_fn 이 {} 를 돌려준다.)"""
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
