"""
agents/eval/metrics_search.py
[tier2] 추가 검색 쿼리가 필요한 측정을 모은 파일.

자원: top-N 재검색(dense) / BM25 키워드 검색 / 코퍼스 멤버십 조회.
전부 metrics_common.set_context 로 주입된 자원을 쓰고, `active_mode() < STANDARD` 면 측정하지
않고 None 을 돌려준다(비용 게이트). 결과는 _cache 로 probe 당 1회만 계산한다.

여기는 '측정'만 한다 — 임계값 판정과 라벨 부여는 diagnose 소관이다.
"""
from __future__ import annotations

from agents.eval.types import Mode, EvalRecord
from agents.eval.metrics_common import _ctx, _cache, active_mode, missed_gold_ids


def _gold_ranks(record: EvalRecord):
    """probe 의 각 gold 청크가 wide_n 재검색에서 몇 위인지(1-based) 매핑. tier2 측정의 단일 소스.

    top-N(wide_n) 재검색을 probe 당 1회만 돌려(memoize) 순위표를 만든다.
    '놓친 gold 가 넓은 후보에 있나'(low_rank 판정)도 이 순위 맵에서 파생 — 재검색 1회를 공유한다.

    planner 가 top_k 근거값을 계산할 원시 순위 측정치다(집계·후보화는 planner 소관).
    "gold 가 5개니 top_k=5" 같은 개수 추정과 달리, "가장 늦게 나오는 gold 가 20위면
    top_k 는 최소 20" 이라는 실측을 준다(multi-hop/나열형에서 개수 ≪ 순위).

    반환: {gold_id: rank}  rank 는 1-based, wide_n 밖이면 None(=top_k 로 도달 불가).
          gold 없음 → None / 모드·자원 미충족 → None.
    """
    if active_mode() < Mode.STANDARD or _ctx.retrieve_fn is None:
        return None

    def compute():
        golds = record.probe.gold_chunk_ids
        if not golds:
            return None
        hits = _ctx.retrieve_fn(
            _ctx.client, _ctx.chunks, record.probe.question, _ctx.wide_n
        )
        order = {h.get("chunk_id"): i + 1 for i, h in enumerate(hits)}  # 1-based 순위
        return {g: order.get(g) for g in golds}  # wide_n 밖이면 None

    return _cache(record, "gold_ranks", compute)


def _bm25_hits_gold(record: EvalRecord):
    """키워드(BM25) 검색이 dense top-k 가 놓친 gold 를 잡나. lexical/semantic mismatch 용.
    대상은 코퍼스에 있는 놓친 gold 만 — 코퍼스에 없는 걸 BM25 가 못 잡는 건 '의미 불일치'가
    아니라 corpus_gap 이다(멤버십 미측정이면 놓친 gold 전부).
    True=키워드로 잡힘(단어 불일치) / False=키워드도 놓침(의미 불일치) / None=자원·모드 미충족."""
    if active_mode() < Mode.STANDARD or _ctx.keyword_fn is None:
        return None

    def compute():
        missed = _missed_gold_in_corpus(record)
        if missed is None:
            missed = missed_gold_ids(record)
        if not missed:
            return None
        hits = _ctx.keyword_fn(_ctx.chunks, record.probe.question, _ctx.wide_n)  # 위와 같으나 검색 함수만 다름
        kw_ids = {h.get("chunk_id") for h in hits}
        return bool(missed & kw_ids)

    return _cache(record, "bm25_hits_gold", compute)


def _gold_corpus_membership(record: EvalRecord):
    """gold 청크별 코퍼스 존재 여부 {gold_id: bool}. gold·자원 없으면 None.

    probe 단위 bool 하나로 접으면 혼합 코퍼스(놓친 gold 가 {있음+없음})에서 검색 라벨이
    통째로 소실된다 — 코퍼스에 있는데 못 찾은 gold 까지 corpus_gap 으로 넘어가기 때문이다.
    """
    if active_mode() < Mode.STANDARD or not _ctx.corpus_ids:
        return None

    def compute():
        golds = record.probe.gold_chunk_ids
        if not golds:
            return None
        return {g: g in _ctx.corpus_ids for g in golds}  # 코퍼스 전체와 대조

    return _cache(record, "gold_corpus_membership", compute)


def _gold_in_corpus(record: EvalRecord):
    """gold 가 전부 코퍼스에 존재하나. corpus_gap(하나라도 없음) 판정용.
    전부 존재 True / 하나라도 없으면 False / gold·자원 없으면 None."""
    membership = _gold_corpus_membership(record)
    return None if membership is None else all(membership.values())


def _gold_absent_from_corpus(record: EvalRecord):
    """gold 가 하나도 코퍼스에 없나 — '답의 근거가 애초에 없다'의 실측. 미측정이면 None.

    _gold_in_corpus 의 부정이 아니다: 그쪽은 '하나라도 없음'이라 부분 gap(일부 gold 는
    코퍼스에 있고 검색까지 된 상태)도 False 가 된다. 근거가 실제로 있는 답변을
    '근거 없이 지어냄'으로 몰지 않으려면 전부 없음을 따로 물어야 한다.
    """
    membership = _gold_corpus_membership(record)
    return None if membership is None else not any(membership.values())


def _missed_gold_in_corpus(record: EvalRecord):
    """놓친 gold 중 코퍼스에 존재하는 것 — 검색 원인(A) 라벨의 실제 근거.
    코퍼스에 없는 gold 는 검색으로 못 고치니 corpus_gap 몫이라 뺀다. 멤버십 미측정이면 None."""
    membership = _gold_corpus_membership(record)
    if membership is None:
        return None
    return {g for g in missed_gold_ids(record) if membership.get(g)}
