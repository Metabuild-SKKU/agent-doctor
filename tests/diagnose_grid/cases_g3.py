"""
tests/diagnose_grid/cases_g3.py
G3 청킹 부분격자 — overchunking / context_mismatch / underchunking / incomplete_enumeration.

축: K1~K3(문서 길이·청크 크기·겹침) · P1(gold span 좌표·개수) · P2(span_grounding)
    · Q1(top-k) · J5(context_precision)
"""
from tests.diagnose_grid.builder import Answer, Case, Doc


CASES = [
    # ── C 경로(recall=1, oracle 통과, 실제 답 틀림)에서 청킹 라벨 3종 ──
    Case(
        id="g3_overchunking",
        docs=[Doc("d1", 1200)], chunk_size=200, chunk_overlap=0,
        gold_spans=[("d1", 0, 700)],            # 길이 700 > 최장 청크 200
        retrieved=[0, 1, 2, 3],                 # gold 전부 검색됨 → recall=1
        answer=Answer.WRONG, oracle_answer=Answer.GOLD_FULL,
        assert_derived={"recall_at_k": 1.0, "oversized_count": ">0",
                        "f1_score": "<0.5", "oracle_f1": ">=0.5"},
        expect={"A": "chunking_overchunking"},
    ),
    Case(
        id="g3_context_mismatch",
        docs=[Doc("d1", 1200)], chunk_size=200, chunk_overlap=0,
        gold_spans=[("d1", 350, 450)],          # 청크1(200-400)·청크2(400-600) 경계에 걸침
        retrieved=[1, 2, 0],
        answer=Answer.WRONG, oracle_answer=Answer.GOLD_FULL,
        assert_derived={"recall_at_k": 1.0, "boundary_split": ">0", "oversized_count": 0},
        expect={"A": "chunking_context_mismatch"},
    ),
    Case(
        id="g3_underchunking",
        docs=[Doc("d1", 1500)], chunk_size=500, chunk_overlap=0,
        gold_spans=[("d1", 100, 150)],          # 근거 50자 / 청크 500자 → 밀도 0.1
        retrieved=[0, 1, 2],
        answer=Answer.WRONG, oracle_answer=Answer.GOLD_FULL,
        judge_real={"context_precision": 0.4},
        assert_derived={"recall_at_k": 1.0, "evidence_density": "<0.2",
                        "boundary_split": 0, "oversized_count": 0},
        expect={"A": "chunking_underchunking"},
    ),

    # ── A 경로(recall<1) 나열형 ──
    Case(
        id="g3_incomplete_enumeration",
        docs=[Doc("d1", 1200)], chunk_size=200, chunk_overlap=0,
        gold_spans=[("d1", 0, 100), ("d1", 400, 500)],   # 근거 2개, top_k 2슬롯
        qtype="aggregation",
        retrieved=[0, 5],                        # 청크2(400-600) 놓침
        wide_ranking=[0, 5, 3, 2, 1, 4],         # 놓친 gold 는 wide 4위 → top_k↑ 로 회복 가능
        assert_derived={"recall_at_k": "<1", "missed_count": 1},
        expect={"A": "retrieval_incomplete_enumeration"},
    ),

    # ── A 경로(recall<1)에서 청킹 라벨은 앞선 검색 원인에 선점당한다 ──
    # oversized 가 실측돼도 A 슬롯은 순위·코퍼스 라벨이 가져간다. 아래 셋이 그 경계다.
    Case(
        id="g3_a_path_taken_by_low_rank",
        docs=[Doc("d1", 1200)], chunk_size=200, chunk_overlap=0,
        gold_spans=[("d1", 0, 700)],
        retrieved=[0, 1],                        # gold 청크 2·3 놓침 → recall<1
        wide_ranking=[0, 1, 4, 5, 2, 3],         # 놓친 gold 가 도달 가능 창 안 → 순위 문제
        answer=Answer.WRONG, oracle_answer=Answer.GOLD_FULL,
        assert_derived={"recall_at_k": "<1", "oversized_count": ">0"},
        expect={"A": "retrieval_low_rank"},
    ),
    Case(
        id="g3_a_path_taken_by_missing_gold",
        docs=[Doc("d1", 1200)], chunk_size=200, chunk_overlap=0,
        gold_spans=[("d1", 0, 700)],
        retrieved=[0, 1],
        wide_ranking=None,                       # 순위 신호 없음 → low_rank 침묵
        answer=Answer.WRONG, oracle_answer=Answer.GOLD_FULL,
        assert_derived={"recall_at_k": "<1", "oversized_count": ">0"},
        expect={"A": "retrieval_missing_gold"},
    ),
    Case(
        # A 슬롯에서 청킹 라벨이 채택되는 유일한 경로: qtype=bridge.
        # bridge 는 semantic_mismatch·missing_gold 를 침묵시키고 자신은 예비로 남으므로,
        # _pick 의 '확정 우선'이 확정인 청킹 라벨을 뽑는다.
        id="g3_a_path_reached_via_bridge",
        docs=[Doc("d1", 1200)], chunk_size=200, chunk_overlap=0,
        gold_spans=[("d1", 0, 700)], qtype="bridge",
        retrieved=[0, 1],
        wide_ranking=None,
        answer=Answer.WRONG, oracle_answer=Answer.GOLD_FULL,
        assert_derived={"recall_at_k": "<1", "oversized_count": ">0"},
        expect={"A": "chunking_overchunking"},
    ),

    # ── P2: span_grounding 이 chunk_fallback 이면 청킹 신호가 통째로 꺼진다 ──
    Case(
        id="g3_span_grounding_fallback",
        docs=[Doc("d1", 1200)], chunk_size=200, chunk_overlap=0,
        gold_spans=[("d1", 350, 450)],
        span_grounding="chunk_fallback",
        retrieved=[1, 2, 0],
        answer=Answer.WRONG, oracle_answer=Answer.GOLD_FULL,
        assert_derived={"boundary_split": None, "oversized_count": None},
        expect={"C": "context_failure"},          # 청킹 신호 없음 → C 슬롯 롤업
    ),
]
