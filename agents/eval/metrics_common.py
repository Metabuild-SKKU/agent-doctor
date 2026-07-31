"""
agents/eval/metrics_common.py
[tier 없음 · 인프라] 측정 레이어 공통 자원 — 진단 모드(비용 게이트 기준)·자원 컨텍스트(_ctx)·memoize(_cache).

tier 별 측정 파일(metrics_basic=tier1 / metrics_search=tier2 / metrics_ragas=tier3)이 공유한다.
diagnose() 가 진입 시 set_mode / set_context 로 설정·주입한다.

주의: _active_mode 는 재바인딩되는 int 이므로 다른 모듈에서 `from ... import _active_mode`
하면 정지 바인딩이 된다(set_mode 후에도 옛 값을 봄). 반드시 active_mode() 로 조회할 것.
_ctx 는 속성만 변이되는 싱글턴이라 import 로 공유해도 안전하다(set_context 가 재바인딩 안 함).
"""
from __future__ import annotations

from agents.eval.types import Mode, EvalRecord


# index_config["rerank_candidates"] 기본값과 같은 값. Eval 이 config 를 못 받은
# 경로(테스트·외부 주입)에서도 도달 가능 창을 정의하기 위한 폴백이다.
DEFAULT_RERANK_CANDIDATES = 20

# 리랭커에 태울 후보 수의 상한(index_config["rerank_candidate_policy"]["max_candidates"]).
# 후보를 넓히는 처방이 갈 수 있는 최대치이자, '순위 문제'로 다룰 순위의 상한이다.
DEFAULT_MAX_RERANK_CANDIDATES = 50


# ── 진단 모드 (현재 실행의 tier 상한) — diagnose() 가 set_mode 로 설정 ──
_active_mode: int = Mode.FAST


def set_mode(mode: int) -> None:
    """diagnose() 진입 시 현재 실행 모드를 설정. 이하 tier 측정까지만 확보 가능(그 위는 None)."""
    global _active_mode
    _active_mode = mode


def active_mode() -> int:
    """현재 실행 모드(측정 self-gate 기준). 측정 함수가 이 값으로 비용 게이트한다."""
    return _active_mode


# ── 진단 자원 컨텍스트 (tier2~3 측정이 쓸 검색·RAGAS 자원 — agent 가 set_context 로 주입) ──

class _Ctx:
    """
    tier2/tier3 측정(재검색·코퍼스 조회·RAGAS)이 쓰는 자원. agent 가 set_context 로 주입한다.
    2단계: RAG/index module에서 값 및 함수들을 가져와야한다!!!!!!!!!!!
    """
    client = None
    chunks: list = []
    corpus_ids: frozenset = frozenset()
    chunk_by_id: dict = {}   # 청크 단건 조회용 인덱스(중복 밀림 분석이 후보마다 훑는다)
    retrieve_fn = None       # (client, chunks, question, top_n) -> list[{"chunk_id",...}]
                             #   ⚠ 리랭크 이전(융합) 순위를 돌려줘야 한다 — reachable_window 참고
    dense_fn = None          # (client, chunks, question, top_n) -> list[{"chunk_id",...}]  dense 단일 채널
    keyword_fn = None        # (chunks, query, top_n) -> list[{"chunk_id",...}]
    ragas_fn = None          # (record, track) -> dict  track: "real"|"oracle"  (tier3 RAGAS lazy)
    wide_n: int = 100        # top-N 재검색·BM25 후보 크기
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES          # 현재 리랭커 후보창
    max_rerank_candidates: int = DEFAULT_MAX_RERANK_CANDIDATES  # 후보창을 넓힐 수 있는 상한


_ctx = _Ctx()


def set_context(client=None, chunks=None, retrieve_fn=None, keyword_fn=None,
                ragas_fn=None, wide_n=100, dense_fn=None,
                rerank_candidates=DEFAULT_RERANK_CANDIDATES,
                max_rerank_candidates=DEFAULT_MAX_RERANK_CANDIDATES):
    """tier2~3 측정이 쓸 자원 주입. agent.run 이 진단 전 1회 호출.
    미주입이면 해당 측정은 자원 없음으로 None(=미확보) 반환."""
    _ctx.client = client
    _ctx.chunks = chunks or []
    _ctx.corpus_ids = frozenset(c.chunk_id for c in _ctx.chunks)
    _ctx.chunk_by_id = {c.chunk_id: c for c in _ctx.chunks}
    _ctx.retrieve_fn = retrieve_fn
    _ctx.dense_fn = dense_fn
    _ctx.keyword_fn = keyword_fn
    _ctx.ragas_fn = ragas_fn
    _ctx.wide_n = wide_n
    _ctx.rerank_candidates = (
        int(rerank_candidates) if rerank_candidates else DEFAULT_RERANK_CANDIDATES
    )
    _ctx.max_rerank_candidates = (
        int(max_rerank_candidates) if max_rerank_candidates
        else DEFAULT_MAX_RERANK_CANDIDATES
    )


def candidate_window() -> int:
    """현재 리랭커가 입력으로 받는 후보 수. 순위 밴드를 가르는 안쪽 경계."""
    return _ctx.rerank_candidates or DEFAULT_RERANK_CANDIDATES


def reachable_window() -> int:
    """순위 라벨이 다룰 순위 상한 — '처방이 실제로 닿는 범위'.

    검색 순위 원인(low_rank 계열)의 처방은 전부 '리랭커에 태워 다시 정렬한다'로 귀결된다.
    리랭커는 후보 N 개만 입력으로 받으므로, 후보를 넓힐 수 있는 최대치보다 뒤 순위의 gold 는
    리랭커를 켜든 후보를 넓히든 top_k 안으로 들어올 수 없다. 그 구간은 '순위가 낮다'가 아니라
    '표현이 안 맞는다'이고, 처방도 임베딩·청킹·하이브리드 쪽이다
    (retrieval_semantic_mismatch / retrieval_lexical_mismatch 소관).

    경계를 임의 배수(top_k×N)가 아니라 config 에서 파생하는 게 요점이다 — 정책값
    (rerank_candidate_policy.max_candidates)을 올리면 판정 창도 같이 넓어져,
    진단 쪽에 따로 튜닝할 상수가 남지 않는다.

    안쪽 경계(candidate_window)와 함께 순위를 세 밴드로 가른다:
      (top_k, 후보창]        → 리랭크 단계 문제 (리랭커 off 면 low_rank, on 이면 강등)
      (후보창, 도달 상한]    → 후보창 밖 (창을 넓혀야 닿음)
      (도달 상한, ...)       → 순위 문제 아님 (표현 문제로 인계)
    """
    return max(candidate_window(),
               _ctx.max_rerank_candidates or DEFAULT_MAX_RERANK_CANDIDATES)


# ── 검색 원인 라벨 공통 기준 (tier 없음 · 자원 불필요) ──────────────

def _missed_gold_ids(record: EvalRecord) -> set[str]:
    """top-k 가 놓친 gold 청크 id 집합 — 검색 원인(A) 라벨들의 공통 근거.

    recall_at_k 는 gold_spans(원문 좌표) 기준이라 '정답 구간이 청크 경계에 잘려 덜 덮였다'
    까지 실패로 센다. 반면 검색 원인 라벨(enumeration/bridge/low_rank/lexical/semantic/
    missing_gold)은 전부 '어떤 gold 청크를 놓쳤나'를 근거로 삼는다.

    두 단위가 어긋나는 경우가 있다 — gold 청크는 전부 검색됐는데 span 은 부분만 덮인 상황
    (gold_chunk_ids 는 span 에서 파생된 캐시라 재청킹 후 경계가 달라지면 생긴다). 이때
    '놓친 청크'가 없으므로 위 라벨들은 근거가 없다. 그런데도 발동하면 "gold 가 top-k 에
    없다" 같은 사실과 반대되는 주장을 confirmed 로 내게 된다.

    그래서 이 집합이 비면 chunk-id 기반 검색 라벨은 전부 스스로 빠지고, 좌표 기반인
    chunking_context_mismatch(경계 분할)가 그 자리를 가져간다 — 실제 원인에 맞는 라벨이다.
    """
    return set(record.probe.gold_chunk_ids) - set(record.retrieved_chunk_ids)


# ── memoize ──────────────────────────────────────────────────────

def _cache(record: EvalRecord, name: str, compute):
    """측정값 memoize.

    1) record.signals(=state.diagnosis_cache[probe_id] 뷰)에 있으면 재사용,
    2) 없으면 compute() 계산해 저장.
    """
    cache = record.signals
    if name not in cache:
        cache[name] = compute()
    return cache[name]
