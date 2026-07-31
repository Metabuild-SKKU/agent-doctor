"""
agents/eval/metrics_search.py
[tier2] 추가 검색 쿼리가 필요한 측정을 모은 파일.

자원: top-N 재검색(융합) / dense 단일 채널 재검색 / BM25 키워드 검색 / 코퍼스 멤버십 조회.
전부 metrics_common.set_context 로 주입된 자원을 쓰고, `active_mode() < STANDARD` 면 측정하지
않고 None 을 돌려준다(비용 게이트). 결과는 _cache 로 probe 당 1회만 계산한다.

여기는 '측정'만 한다 — 임계값 판정과 라벨 부여는 diagnose 소관이다.

[단계별 순위]
  최종 순위는 `채널 검색(dense/BM25) → 융합 → 후보창 → 리랭크 → top_k 컷` 을 거쳐 만들어진다.
  "순위가 낮다"는 증상이고, 어느 단계에서 gold 를 잃었는지가 처방을 정한다. 그래서 단계마다
  gold 순위를 따로 잰다:
    _gold_dense_ranks    : dense 단일 채널
    _gold_lexical_ranks  : BM25 단일 채널
    _gold_ranks          : 융합(리랭크 이전) — 기존 단일 소스, 의미는 그대로
    _gold_pre_rerank_ranks / retrieved_chunk_ids : 후보창·최종 컷(프로덕션 검색에서 그대로 옴)
"""
from __future__ import annotations

from agents.eval.types import Mode, EvalRecord
from agents.eval.metrics_common import (
    _ctx, _cache, active_mode, _missed_gold_ids, reachable_window,
)


def _rank_map(hits, golds) -> dict:
    """검색 결과(순서 있는 hit 목록)에서 gold 별 1-based 순위 맵을 만든다. 밖이면 None."""
    order = {h.get("chunk_id"): i + 1 for i, h in enumerate(hits)}
    return {g: order.get(g) for g in golds}


def _wide_hit_ids(record: EvalRecord):
    """융합 wide_n 재검색 결과의 chunk_id 순서 목록. 순위·경쟁청크 분석의 단일 소스.

    재검색 1회를 _gold_ranks(순위)와 _redundancy_above_gold(중복 밀림)가 공유한다.
    """
    if active_mode() < Mode.STANDARD or _ctx.retrieve_fn is None:
        return None

    def compute():
        hits = _ctx.retrieve_fn(
            _ctx.client, _ctx.chunks, record.probe.question, _ctx.wide_n
        )
        return [h.get("chunk_id") for h in hits if h.get("chunk_id")]

    return _cache(record, "wide_hit_ids", compute)


def _gold_ranks(record: EvalRecord):
    """probe 의 각 gold 청크가 wide_n 재검색에서 몇 위인지(1-based) 매핑. tier2 측정의 단일 소스.

    top-N(wide_n) 재검색을 probe 당 1회만 돌려(memoize) 순위표를 만든다.
    '놓친 gold 가 넓은 후보에 있나'(low_rank 판정)도 이 순위 맵에서 파생 — 재검색 1회를 공유한다.

    ⚠ 이 순위는 **리랭크 이전(융합) 순위**여야 한다. retrieve_fn 이 프로덕션과 같은 리랭크를
    태우면 wide_n(=100) 개를 리랭크한 순서가 나오는데, 실제 검색은 rerank_candidates(=20) 개만
    리랭크하므로 서로 다른 함수가 된다. 그러면 '후보창 밖이라 리랭커가 못 봤다'와 '보고도
    떨어뜨렸다'를 가르는 기준값 자체가 오염된다. agent.run 이 리랭크를 끈 검색을 주입한다.

    planner 가 top_k 근거값을 계산할 원시 순위 측정치다(집계·후보화는 planner 소관).
    "gold 가 5개니 top_k=5" 같은 개수 추정과 달리, "가장 늦게 나오는 gold 가 20위면
    top_k 는 최소 20" 이라는 실측을 준다(multi-hop/나열형에서 개수 ≪ 순위).

    반환: {gold_id: rank}  rank 는 1-based, wide_n 밖이면 None(=top_k 로 도달 불가).
          gold 없음 → None / 모드·자원 미충족 → None.
    """
    golds = record.probe.gold_chunk_ids
    if not golds:
        return None
    hit_ids = _wide_hit_ids(record)
    if hit_ids is None:
        return None
    return _cache(
        record, "gold_ranks",
        lambda: _rank_map([{"chunk_id": cid} for cid in hit_ids], golds),
    )


def _gold_dense_ranks(record: EvalRecord):
    """dense 단일 채널 wide_n 재검색에서의 gold 순위. 융합 손실 판정용.

    융합(dense+BM25 가중합) 순위와 대조해 "한 채널은 상위에 뒀는데 융합이 떨어뜨렸다"를
    가른다. dense_fn 미주입(기존 배선·테스트)이면 None 이라 그 라벨이 스스로 침묵한다.
    """
    if active_mode() < Mode.STANDARD or _ctx.dense_fn is None:
        return None

    def compute():
        golds = record.probe.gold_chunk_ids
        if not golds:
            return None
        hits = _ctx.dense_fn(
            _ctx.client, _ctx.chunks, record.probe.question, _ctx.wide_n
        )
        return _rank_map(hits, golds)

    return _cache(record, "gold_dense_ranks", compute)


def _gold_lexical_ranks(record: EvalRecord):
    """BM25 단일 채널 wide_n 검색에서의 gold 순위. 융합 손실 판정 + bm25 적중 판정의 원천."""
    if active_mode() < Mode.STANDARD or _ctx.keyword_fn is None:
        return None

    def compute():
        golds = record.probe.gold_chunk_ids
        if not golds:
            return None
        hits = _ctx.keyword_fn(_ctx.chunks, record.probe.question, _ctx.wide_n)
        return _rank_map(hits, golds)

    return _cache(record, "gold_lexical_ranks", compute)


def _gold_pre_rerank_ranks(record: EvalRecord):
    """프로덕션 검색의 **리랭크 직전** 후보 순서에서의 gold 순위. 재검색 비용 0.

    retriever.search_with_details 가 실어 보내는 pre_rerank_ids 에서 그대로 파생한다.
    이 목록은 rerank_candidates 개로 잘려 있으므로, 여기에 없다 = 리랭커 입력에 없었다
    (후보창 밖)는 뜻이다. 옛 계약의 retriever(테스트 주입 등)는 키가 없어 None.
    """
    ids = record.retrieval_details.get("pre_rerank_ids")
    if not isinstance(ids, list):
        return None
    golds = record.probe.gold_chunk_ids
    if not golds:
        return None
    return _rank_map([{"chunk_id": cid} for cid in ids], golds)


def _bm25_hits_gold(record: EvalRecord):
    """키워드(BM25) 검색이 dense top-k 가 놓친 gold 를 잡나. lexical/semantic mismatch 용.
    대상은 코퍼스에 있는 놓친 gold 만 — 코퍼스에 없는 걸 BM25 가 못 잡는 건 '의미 불일치'가
    아니라 corpus_gap 이다(멤버십 미측정이면 놓친 gold 전부).
    True=키워드로 잡힘(단어 불일치) / False=키워드도 놓침(의미 불일치) / None=자원·모드 미충족.

    순위 맵(_gold_lexical_ranks)에서 파생한다 — 같은 BM25 검색을 두 번 돌리지 않는다.
    """
    if active_mode() < Mode.STANDARD or _ctx.keyword_fn is None:
        return None

    def compute():
        missed = _missed_gold_in_corpus(record)
        if missed is None:
            missed = _missed_gold_ids(record)
        if not missed:
            return None
        ranks = _gold_lexical_ranks(record) or {}
        return any(ranks.get(g) is not None for g in missed)

    return _cache(record, "bm25_hits_gold", compute)


# ── 경쟁 청크 구성 분석 (중복 밀림) ──────────────────────────────

def _shingles(text: str) -> frozenset:
    """문자 3-gram 집합. 한국어는 공백 토큰이 신뢰도가 낮아 문자 단위로 본다."""
    normalized = "".join((text or "").split())
    if len(normalized) < 3:
        return frozenset({normalized}) if normalized else frozenset()
    return frozenset(normalized[i:i + 3] for i in range(len(normalized) - 2))


_NEAR_DUPLICATE_JACCARD = 0.8


def _near_duplicate(left: dict, right: dict) -> bool:
    """두 청크가 사실상 같은 내용인가 — 좌표 겹침 비율 또는 문자 3-gram Jaccard.

    좌표 규칙은 '겹치나'가 아니라 '거의 같은 구간인가'로 본다. chunk_overlap 청킹에서
    인접 청크는 **설계상 항상** 겹치기 때문이다(기본값 512/50 이면 짧은 쪽의 약 10%).
    '1자라도 겹치면 중복'으로 잡으면 한 문서의 연속 청크가 통째로 잉여로 접혀
    projected_rank 가 과대 회복되고, duplicate_crowding 이 오확정된다.
    그러면 _upstream_rank_cause 때문에 실행 가능한 순위 라벨까지 전부 침묵한다.

    임계는 Jaccard 와 같은 값을 쓴다 — 두 규칙이 같은 질문("거의 같은 내용인가")에
    답하므로 문턱도 같아야 한다.
    """
    if (left.get("doc_id") and left.get("doc_id") == right.get("doc_id")
            and left.get("span") and right.get("span")):
        l_start, l_end = left["span"]
        r_start, r_end = right["span"]
        overlap = min(l_end, r_end) - max(l_start, r_start)
        shorter = min(l_end - l_start, r_end - r_start)
        if shorter > 0 and overlap / shorter >= _NEAR_DUPLICATE_JACCARD:
            return True
    l_grams, r_grams = left.get("grams"), right.get("grams")
    if not l_grams or not r_grams:
        return False
    union = len(l_grams | r_grams)
    if union == 0:
        return False
    return len(l_grams & r_grams) / union >= _NEAR_DUPLICATE_JACCARD


def _chunk_view(chunk_id: str) -> dict:
    """중복 판정에 필요한 최소 정보만 뽑는다(_ctx.chunk_by_id 인덱스 조회).

    인덱스를 쓰는 이유: 이 함수는 probe 마다 후보 수십 개에 대해 불린다. 선형 탐색이면
    (후보 수 × 코퍼스 크기 × probe 수)가 되어 코퍼스가 커질수록 진단이 급격히 느려진다.
    """
    chunk = _ctx.chunk_by_id.get(chunk_id)
    if chunk is None:
        return {"chunk_id": chunk_id, "doc_id": "", "span": None, "grams": frozenset()}
    return {
        "chunk_id": chunk_id,
        "doc_id": getattr(chunk, "doc_id", ""),
        "span": getattr(chunk, "char_span", None),
        "grams": _shingles(getattr(chunk, "text", "")),
    }


def _redundancy_above_gold(record: EvalRecord):
    """놓친 gold 위에 쌓인 '중복 경쟁 청크' 수와 중복 제거 시 예상 순위.

    반환: {gold_id: {"rank", "redundant", "projected_rank"}} / 미측정이면 None.
      redundant      = gold 보다 위에 있는 비-gold 중 서로 near-duplicate 인 것들의 잉여분
                       (그룹마다 대표 1개는 남긴다 — 중복 제거는 그룹을 없애는 게 아니다)
      projected_rank = rank - redundant (중복을 접었을 때 gold 가 올라갈 순위)

    이 신호가 필요한 이유: 중복 밀림은 순위 원인 중 유일하게 리랭커로 안 고쳐진다.
    cross-encoder 는 중복 청크를 상위에 그대로 둔다(각각이 실제로 질문과 관련 있으니까).
    처방이 갈리므로(MMR/중복 제거 vs 리랭커) 신호도 따로 잰다.

    비용: 도달 가능 창(reachable_window, 기본 50) 안쪽만 본다 — 그 밖은 순위 라벨의 관할이 아니다.
    """
    hit_ids = _wide_hit_ids(record)
    if hit_ids is None:
        return None

    def compute():
        ranks = _gold_ranks(record) or {}
        window = reachable_window()
        top_k = len(record.retrieved_chunk_ids)
        if top_k <= 0:
            return {}
        golds = set(record.probe.gold_chunk_ids)
        # 하한(top_k)이 상한만큼 중요하다. 이 신호의 주장은 "중복이 gold 를 top_k 밖으로
        # 밀어냈다"인데, 융합 순위가 이미 top_k 이내인 gold 는 융합 단계에서 밀린 적이 없다.
        # 하한을 빼면 리랭커가 강등시킨 gold(융합 순위가 top_k 이내로 잡힌다 — wide 재검색이
        # 리랭크를 끄므로)까지 대상이 되어, 위에 중복 한 쌍만 있으면 projected_rank <= top_k 가
        # 자명하게 성립한다. 그러면 crowding 이 확정으로 서서 실행 가능한 reranker_demotion 을
        # 침묵시키고, 정작 자신은 처방이 없어 그 probe 가 아무 처방도 못 받는다.
        targets = {
            g: r for g in _missed_gold_ids(record)
            if (r := ranks.get(g)) is not None and top_k < r <= window
        }
        if not targets:
            return {}

        # 위에서부터 훑으며 near-duplicate 그룹을 만든다(대표는 먼저 나온 것).
        # 순위는 반드시 **전체 목록 기준**으로 담는다 — gold 를 뺀 목록의 인덱스를 쓰면
        # 위에 낀 gold 만큼 앞으로 당겨져, 어떤 gold 보다 아래에 있는 중복이 '위에 있다'로
        # 오집계된다(그 gold 는 접어도 안 올라오는데 회복 가능으로 잡힌다).
        # gold 는 경쟁 청크가 아니므로 그룹핑에서만 빼고, 순위 좌표계는 건드리지 않는다.
        deepest = max(targets.values())
        representatives: list[dict] = []
        redundant_ranks: list[int] = []          # 잉여 청크가 놓인 전체 순위(1-based)
        for full_rank, chunk_id in enumerate(hit_ids[:deepest - 1], start=1):
            if chunk_id in golds:
                continue
            view = _chunk_view(chunk_id)
            if any(_near_duplicate(view, rep) for rep in representatives):
                redundant_ranks.append(full_rank)
            else:
                representatives.append(view)

        out = {}
        for gold_id, rank in targets.items():
            redundant = sum(1 for r in redundant_ranks if r < rank)
            out[gold_id] = {
                "rank": rank,
                "redundant": redundant,
                "projected_rank": rank - redundant,
            }
        return out

    return _cache(record, "redundancy_above_gold", compute)


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
    return {g for g in _missed_gold_ids(record) if membership.get(g)}
