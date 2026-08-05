"""
tests/diagnose_grid/cases_g3.py
G3 청킹 부분격자 — overchunking / context_mismatch / underchunking / incomplete_enumeration.

축: K1~K3(문서 길이·청크 크기·겹침) · P1(gold span 좌표·개수) · P2(span_grounding)
    · Q1(top-k) · J5(context_precision)

Case 는 판정에 관여하는 축을 전부 명시해야 한다(기본값 없음). 안 적힌 값이 라벨을 가르는
일을 막으려는 것이다 — answer 하나만 바꿔도 C 슬롯 전제가 무너져 청킹 라벨이 통째로 사라진다.
"""
from tests.diagnose_grid.builder import Answer, Case, Doc

GT = "1972년 12월 27일에 제7차 개정 헌법이 공포되었다"

# C 경로 답변 시나리오 — 실제 답은 틀리고 골드 컨텍스트로는 맞는다.
# 이게 있어야 _context_failed(recall=1 · oracle 통과 · 실제 답 틀림)가 선다.
_WRONG_BUT_ORACLE_OK = dict(answer=Answer.WRONG, oracle_answer=Answer.GOLD_FULL)


CASES = [
    # ── C 경로(recall=1, oracle 통과, 실제 답 틀림)에서 청킹 라벨 3종 ──
    Case(
        id="g3_overchunking",
        docs=[Doc("d1", 1200)], chunk_strategy="fixed", chunk_size=200, chunk_overlap=0,
        gold_spans=[("d1", 0, 700)],            # 길이 700 > 최장 청크 200
        span_grounding=None, ground_truth=GT, qtype=None, answer_exists=None,
        retrieved=[0, 1, 2, 3],                 # gold 전부 검색됨 → recall=1
        search_mode="dense", reranked=False, mmr_applied=False,
        **_WRONG_BUT_ORACLE_OK,
        assert_derived={"recall_at_k": 1.0, "oversized_count": ">0",
                        "f1_score": "<0.5", "oracle_f1": ">=0.5"},
        expect={"A": "chunking_overchunking"},
    ),
    Case(
        id="g3_context_mismatch",
        docs=[Doc("d1", 1200)], chunk_strategy="fixed", chunk_size=200, chunk_overlap=0,
        gold_spans=[("d1", 350, 450)],          # 청크1(200-400)·청크2(400-600) 경계에 걸침
        span_grounding=None, ground_truth=GT, qtype=None, answer_exists=None,
        retrieved=[1, 2, 0],
        search_mode="dense", reranked=False, mmr_applied=False,
        **_WRONG_BUT_ORACLE_OK,
        assert_derived={"recall_at_k": 1.0, "boundary_split": ">0", "oversized_count": 0},
        expect={"A": "chunking_context_mismatch"},
    ),
    Case(
        id="g3_underchunking",
        docs=[Doc("d1", 1500)], chunk_strategy="fixed", chunk_size=500, chunk_overlap=0,
        gold_spans=[("d1", 100, 150)],          # 근거 50자 / 청크 500자 → 밀도 0.1
        span_grounding=None, ground_truth=GT, qtype=None, answer_exists=None,
        retrieved=[0, 1, 2],
        search_mode="dense", reranked=False, mmr_applied=False,
        **_WRONG_BUT_ORACLE_OK,
        judge_real={"context_precision": 0.4},
        assert_derived={"recall_at_k": 1.0, "evidence_density": "<0.2",
                        "boundary_split": 0, "oversized_count": 0},
        expect={"A": "chunking_underchunking"},
    ),

    # ── A 경로(recall<1) 나열형 ──
    Case(
        id="g3_incomplete_enumeration",
        docs=[Doc("d1", 1200)], chunk_strategy="fixed", chunk_size=200, chunk_overlap=0,
        gold_spans=[("d1", 0, 100), ("d1", 400, 500)],   # 근거 2개, top_k 2슬롯
        span_grounding=None, ground_truth=GT, qtype="aggregation", answer_exists=None,
        retrieved=[0, 5],                        # 청크2(400-600) 놓침
        wide_ranking=[0, 5, 3, 2, 1, 4],         # 놓친 gold 는 wide 4위 → top_k↑ 로 회복 가능
        search_mode="dense", reranked=False, mmr_applied=False,
        **_WRONG_BUT_ORACLE_OK,
        assert_derived={"recall_at_k": "<1", "missed_count": 1},
        expect={"A": "retrieval_incomplete_enumeration"},
    ),

    # ── A 경로(recall<1)에서 청킹 라벨은 앞선 검색 원인에 선점당한다 ──
    # oversized 가 실측돼도 A 슬롯은 순위·코퍼스 라벨이 가져간다. 아래 셋이 그 경계다.
    Case(
        id="g3_a_path_taken_by_low_rank",
        docs=[Doc("d1", 1200)], chunk_strategy="fixed", chunk_size=200, chunk_overlap=0,
        gold_spans=[("d1", 0, 700)],
        span_grounding=None, ground_truth=GT, qtype=None, answer_exists=None,
        retrieved=[0, 1],                        # gold 청크 2·3 놓침 → recall<1
        wide_ranking=[0, 1, 4, 5, 2, 3],         # 놓친 gold 가 도달 가능 창 안 → 순위 문제
        search_mode="dense", reranked=False, mmr_applied=False,
        **_WRONG_BUT_ORACLE_OK,
        assert_derived={"recall_at_k": "<1", "oversized_count": ">0"},
        expect={"A": "retrieval_low_rank"},
    ),
    Case(
        id="g3_a_path_taken_by_missing_gold",
        docs=[Doc("d1", 1200)], chunk_strategy="fixed", chunk_size=200, chunk_overlap=0,
        gold_spans=[("d1", 0, 700)],
        span_grounding=None, ground_truth=GT, qtype=None, answer_exists=None,
        retrieved=[0, 1],
        wide_ranking=None,                       # 순위 신호 없음 → low_rank 침묵
        search_mode="dense", reranked=False, mmr_applied=False,
        **_WRONG_BUT_ORACLE_OK,
        assert_derived={"recall_at_k": "<1", "oversized_count": ">0"},
        expect={"A": "retrieval_missing_gold"},
    ),
    Case(
        # A 슬롯에서 청킹 라벨이 채택되는 유일한 경로: qtype=bridge.
        # bridge 는 semantic_mismatch·missing_gold 를 침묵시키고 자신은 예비로 남으므로,
        # _pick 의 '확정 우선'이 확정인 청킹 라벨을 뽑는다.
        id="g3_a_path_reached_via_bridge",
        docs=[Doc("d1", 1200)], chunk_strategy="fixed", chunk_size=200, chunk_overlap=0,
        gold_spans=[("d1", 0, 700)],
        span_grounding=None, ground_truth=GT, qtype="bridge", answer_exists=None,
        retrieved=[0, 1],
        wide_ranking=None,
        search_mode="dense", reranked=False, mmr_applied=False,
        **_WRONG_BUT_ORACLE_OK,
        assert_derived={"recall_at_k": "<1", "oversized_count": ">0"},
        expect={"A": "chunking_overchunking"},
    ),

    # ── 섹션 경계의 좌표 틈 — gold 청크를 다 집었는데도 recall=0 이 된다 ──
    # markdown_recursive 는 섹션마다 따로 자르고 청크 앞뒤 공백을 떼므로(Index _trimmed_slice)
    # 섹션 경계에 좌표 틈이 남는다(실측: sample_docs/hr_policy.md 21청크 중 20건).
    # gold span 이 그 틈을 지나면 span_recall 이 '빈틈없이 덮기'에 실패해 0 을 낸다 —
    # 검색은 gold 청크를 전부 가져왔는데도. 그러면 A 슬롯이 열리지만 missed_gold 가 비어
    # 구체 라벨이 전부 침묵하고, 롤업(retrieval_failure)만 남아 '검색을 고쳐라'가 나간다.
    # (이슈 #100)
    Case(
        id="g3_section_gap_zeroes_recall",
        docs=[Doc("d1", text="# 가\n\n" + "정" * 120 + "\n\n# 나\n\n" + "책" * 120 + "\n")],
        chunk_strategy="markdown_recursive", chunk_size=512, chunk_overlap=50,
        gold_spans=[("d1", 100, 140)],           # 청크0 끝(125)과 청크1 시작(127) 사이를 지난다
        span_grounding=None, ground_truth=GT, qtype=None, answer_exists=None,
        retrieved=[0, 1],                        # gold 청크는 전부 검색됨
        search_mode="dense", reranked=False, mmr_applied=False,
        **_WRONG_BUT_ORACLE_OK,
        assert_derived={"recall_at_k": 0.0, "uncovered": 1, "missed_count": 0},
        expect={"A": "retrieval_failure"},
    ),
    Case(
        # 대조 — 같은 문서·전략인데 span 이 틈을 안 지나면 recall=1 로 정상 판정된다.
        id="g3_section_gap_control",
        docs=[Doc("d1", text="# 가\n\n" + "정" * 120 + "\n\n# 나\n\n" + "책" * 120 + "\n")],
        chunk_strategy="markdown_recursive", chunk_size=512, chunk_overlap=50,
        gold_spans=[("d1", 60, 100)],
        span_grounding=None, ground_truth=GT, qtype=None, answer_exists=None,
        retrieved=[0, 1],
        search_mode="dense", reranked=False, mmr_applied=False,
        **_WRONG_BUT_ORACLE_OK,
        assert_derived={"recall_at_k": 1.0, "uncovered": 0},
        expect={"C": "context_failure"},
    ),

    # ── P2: span_grounding 이 chunk_fallback 이면 청킹 신호가 통째로 꺼진다 ──
    Case(
        id="g3_span_grounding_fallback",
        docs=[Doc("d1", 1200)], chunk_strategy="fixed", chunk_size=200, chunk_overlap=0,
        gold_spans=[("d1", 350, 450)],
        span_grounding="chunk_fallback", ground_truth=GT, qtype=None, answer_exists=None,
        retrieved=[1, 2, 0],
        search_mode="dense", reranked=False, mmr_applied=False,
        **_WRONG_BUT_ORACLE_OK,
        assert_derived={"boundary_split": None, "oversized_count": None},
        expect={"C": "context_failure"},          # 청킹 신호 없음 → C 슬롯 롤업
    ),
]
