"""
agents/eval/metrics_common.py
측정 레이어 공통 자원 — 진단 모드(비용 게이트 기준)·자원 컨텍스트(_ctx)·memoize(_cache).

metrics_basic / metrics_chunk / metrics_ragas 가 공유한다. diagnose() 가 진입 시
set_mode / set_context 로 설정·주입한다.

주의: _active_mode 는 재바인딩되는 int 이므로 다른 모듈에서 `from ... import _active_mode`
하면 정지 바인딩이 된다(set_mode 후에도 옛 값을 봄). 반드시 active_mode() 로 조회할 것.
_ctx 는 속성만 변이되는 싱글턴이라 import 로 공유해도 안전하다(set_context 가 재바인딩 안 함).
"""
from __future__ import annotations

from agents.eval.types import Mode, EvalRecord


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
    retrieve_fn = None       # (client, chunks, question, top_n) -> list[{"chunk_id",...}]
    keyword_fn = None        # (chunks, query, top_n) -> list[{"chunk_id",...}]
    ragas_fn = None          # (record, track) -> dict  track: "real"|"oracle"  (tier3 RAGAS lazy)
    wide_n: int = 100        # top-N 재검색·BM25 후보 크기


_ctx = _Ctx()


def set_context(client=None, chunks=None, retrieve_fn=None, keyword_fn=None,
                ragas_fn=None, wide_n=100):
    """tier2~3 측정이 쓸 자원 주입. agent.run 이 진단 전 1회 호출.
    미주입이면 해당 측정은 자원 없음으로 None(=미확보) 반환."""
    _ctx.client = client
    _ctx.chunks = chunks or []
    _ctx.corpus_ids = frozenset(c.chunk_id for c in _ctx.chunks)
    _ctx.retrieve_fn = retrieve_fn
    _ctx.keyword_fn = keyword_fn
    _ctx.ragas_fn = ragas_fn
    _ctx.wide_n = wide_n


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
