"""
tests/test_diagnose_labels.py
diagnose 라벨 함수의 경계(edge case) 고정.

각 라벨은 '자기 판별 신호가 실제로 발동했는지'로 스스로를 self-scope 한다. 이 파일은
라벨마다 발동/미발동 경계를 못 박아, 신호나 임계값을 바꿀 때 어떤 라벨의 도달 범위가
움직이는지 드러나게 한다.

라벨 함수는 record 필드만 읽으므로 _compute_metrics 를 거치지 않고 recall/f1/RAGAS 를
직접 주입한다(= 지표 계산이 아니라 '판정'만 검증). tier2 자원은 set_context 로 가짜
retrieve_fn/keyword_fn 을 주입해 흉내낸다.

주의: 여기 고정된 동작 중 일부는 '설계 논의 중'으로 표시돼 있다. 그 테스트는 옳음을
      주장하는 게 아니라 현행 동작을 기록해, 바꿀 때 조용히 지나가지 않게 한다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import Probe, Chunk
from agents.eval import metrics_common, diagnose
from agents.eval.types import (
    EvalRecord, Mode,
    F1_PASS_THRESHOLD, RAGAS_FAITHFULNESS_MIN, RAGAS_RESPONSE_RELEVANCY_MIN,
)


# ── 픽스처 ────────────────────────────────────────────────────────

def _record(
    gold_ids=("g_a",), retrieved_ids=("g_a",), *,
    recall=1.0, f1=1.0, oracle_f1=1.0, qtype=None,
    answer_exists=None, ground_truth="정답", answer="답변", oracle_answer="오라클 답변",
    faith=None, rel=None, faith_oracle=None, rel_oracle=None,
    gold_spans=None,
):
    """라벨 함수가 읽는 필드만 채운 EvalRecord. RAGAS 는 *_done 을 세워 LLM 경로를 막는다."""
    probe = Probe(
        probe_id="p1", question="질문", source="taxonomy",
        gold_chunk_ids=list(gold_ids), qtype=qtype,
        answer_exists=answer_exists, ground_truth=ground_truth,
        gold_spans=list(gold_spans or []),
    )
    rec = EvalRecord(
        probe=probe,
        retrieved_chunk_ids=list(retrieved_ids),
        generated_answer=answer,
        oracle_answer=oracle_answer,
    )
    rec.recall_at_k = recall
    rec.f1_score = f1
    rec.oracle_f1 = oracle_f1
    if faith is not None or rel is not None:
        rec.ragas = {"faithfulness": faith, "response_relevancy": rel}
    rec.ragas_done = True
    if faith_oracle is not None or rel_oracle is not None:
        rec.oracle_ragas = {"faithfulness": faith_oracle, "response_relevancy": rel_oracle}
    rec.oracle_ragas_done = True
    return rec


class _FakeRetriever:
    def __init__(self, ranked_ids):
        self.ranked_ids = ranked_ids

    def __call__(self, *args, **kwargs):
        top_n = args[-1] if args else kwargs.get("top_n", 100)
        return [{"chunk_id": cid} for cid in self.ranked_ids[:top_n]]


class _FakeKeyword:
    def __init__(self, hit_ids):
        self.hit_ids = hit_ids

    def __call__(self, *args, **kwargs):
        return [{"chunk_id": cid} for cid in self.hit_ids]


class _DiagnoseTestBase(unittest.TestCase):
    """기본은 tier2 자원 없는 STANDARD. 코퍼스에는 g_a·g_b 가 있다."""

    CORPUS = ("g_a", "g_b", "g_c")

    def setUp(self):
        metrics_common.set_mode(Mode.STANDARD)
        self._chunks = [Chunk(c, "d1", "본문", char_span=(i * 100, (i + 1) * 100))
                        for i, c in enumerate(self.CORPUS)]
        metrics_common.set_context(chunks=self._chunks)

    def tearDown(self):
        metrics_common.set_context()
        metrics_common.set_mode(Mode.FAST)

    def _with(self, *, retrieve=None, keyword=None):
        metrics_common.set_context(
            chunks=self._chunks,
            retrieve_fn=_FakeRetriever(retrieve) if retrieve is not None else None,
            keyword_fn=_FakeKeyword(keyword) if keyword is not None else None,
        )


# ══════════════════════════════════════════════════════════════════
#  공통 전제: 놓친 gold 청크가 없으면 chunk-id 기반 검색 라벨은 발동 금지
#  (recall 은 gold_spans 기준이라 '구간이 덜 덮임'까지 실패로 세는데,
#   그 상황에서 놓친 청크는 없을 수 있다 — Fix 1)
# ══════════════════════════════════════════════════════════════════

class MissedGoldGuardTest(_DiagnoseTestBase):
    def test_missed_is_empty_when_all_gold_retrieved(self):
        rec = _record(("g_a", "g_b"), ("g_a", "g_b", "x"), recall=0.5)
        self.assertEqual(metrics_common._missed_gold_ids(rec), set())

    def test_missed_lists_only_unretrieved_gold(self):
        rec = _record(("g_a", "g_b"), ("g_a", "x"), recall=0.5)
        self.assertEqual(metrics_common._missed_gold_ids(rec), {"g_b"})

    def test_missing_gold_silent_when_nothing_missed(self):
        """수정 전에는 'gold 가 top-k 에 없다'를 confirmed·critical 로 주장했다."""
        rec = _record(("g_a", "g_b"), ("g_a", "g_b"), recall=0.5)
        self.assertIsNone(diagnose.retrieval_missing_gold(rec))

    def test_enumeration_silent_when_nothing_missed(self):
        rec = _record(("g_a", "g_b"), ("g_a", "g_b"), recall=0.5)
        self.assertTrue(diagnose._enumeration_cache(rec))       # 개수 전제는 성립
        self.assertIsNone(diagnose.retrieval_incomplete_enumeration(rec))

    def test_bridge_silent_when_nothing_missed(self):
        rec = _record(("g_a", "g_b"), ("g_a", "g_b"), recall=0.5, qtype="bridge")
        self.assertIsNone(diagnose.retrieval_missing_bridge_dependency(rec))

    def test_low_rank_silent_when_nothing_missed(self):
        self._with(retrieve=["g_a", "g_b"])
        rec = _record(("g_a", "g_b"), ("g_a", "g_b"), recall=0.5)
        self.assertIsNone(diagnose.retrieval_low_rank(rec))


# ══════════════════════════════════════════════════════════════════
#  A그룹: 검색 실패
# ══════════════════════════════════════════════════════════════════

class RetrievalLowRankTest(_DiagnoseTestBase):
    def test_none_without_tier2_resource(self):
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5)
        self.assertIsNone(diagnose.retrieval_low_rank(rec))     # retrieve_fn 미주입

    def test_confirmed_when_missed_gold_sits_in_wider_candidates(self):
        self._with(retrieve=["g_a", "x", "y", "g_b"])           # g_b 는 4위
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5)
        finding = diagnose.retrieval_low_rank(rec)
        self.assertIsNotNone(finding)
        self.assertTrue(finding.confirmed)
        self.assertEqual(finding.severity, "warning")           # critical 아님

    def test_none_when_missed_gold_absent_even_from_wide_search(self):
        self._with(retrieve=["g_a", "x", "y"])                  # g_b 가 wide 밖 → rank None
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5)
        self.assertIsNone(diagnose.retrieval_low_rank(rec))


class RetrievalMismatchTest(_DiagnoseTestBase):
    """lexical(BM25 잡음) / semantic(BM25도 놓침) 는 배타적이며 bm25 신호로 갈린다."""

    def test_lexical_confirmed_when_keyword_search_catches_missed_gold(self):
        self._with(keyword=["g_b"])
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5)
        finding = diagnose.retrieval_lexical_mismatch(rec)
        self.assertIsNotNone(finding)
        self.assertTrue(finding.confirmed)
        self.assertIsNone(diagnose.retrieval_semantic_mismatch(rec))   # 배타

    def test_semantic_confirmed_when_keyword_also_misses_but_gold_in_corpus(self):
        self._with(keyword=["zzz"])
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5)
        finding = diagnose.retrieval_semantic_mismatch(rec)
        self.assertIsNotNone(finding)
        self.assertTrue(finding.confirmed)
        self.assertEqual(finding.severity, "critical")
        self.assertIsNone(diagnose.retrieval_lexical_mismatch(rec))    # 배타

    def test_semantic_silent_when_gold_absent_from_corpus(self):
        """코퍼스에 없으면 semantic 이 아니라 corpus_gap 영역."""
        self._with(keyword=["zzz"])
        rec = _record(("g_a", "unknown"), ("g_a",), recall=0.5)
        self.assertIsNone(diagnose.retrieval_semantic_mismatch(rec))

    def test_both_silent_without_keyword_resource(self):
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5)     # keyword_fn 미주입 → None
        self.assertIsNone(diagnose.retrieval_lexical_mismatch(rec))
        self.assertIsNone(diagnose.retrieval_semantic_mismatch(rec))

    def test_lexical_silent_when_dense_wide_search_also_has_gold(self):
        """BM25 로 잡혀도 dense wide-N 후보에 있으면(순위만 낮음) low_rank 영역 — lexical 아님."""
        self._with(retrieve=["g_a", "x", "y", "g_b"], keyword=["g_b"])   # g_b 는 dense 4위 + BM25
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5)
        self.assertIsNone(diagnose.retrieval_lexical_mismatch(rec))
        self.assertTrue(diagnose.retrieval_low_rank(rec).confirmed)      # 배타 상대는 low_rank

    def test_semantic_preliminary_when_corpus_membership_unknown(self):
        """BM25 는 놓쳤으나 코퍼스 멤버십 미측정(None) → 확정 못 하고 예비(missing_gold 와 동일)."""
        metrics_common.set_context(chunks=[], keyword_fn=_FakeKeyword(["zzz"]))   # corpus_ids 빔 → None
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5)
        finding = diagnose.retrieval_semantic_mismatch(rec)
        self.assertIsNotNone(finding)
        self.assertFalse(finding.confirmed)


class RetrievalMissingGoldTest(_DiagnoseTestBase):
    def test_confirmed_when_gold_in_corpus(self):
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5)
        finding = diagnose.retrieval_missing_gold(rec)
        self.assertTrue(finding.confirmed)
        self.assertEqual(finding.severity, "critical")

    def test_preliminary_when_corpus_membership_unknown(self):
        metrics_common.set_mode(Mode.FAST)                      # tier2 미도달 → in_corpus None
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5)
        finding = diagnose.retrieval_missing_gold(rec)
        self.assertIsNotNone(finding)
        self.assertFalse(finding.confirmed)

    def test_silent_when_gold_absent_from_corpus(self):
        rec = _record(("g_a", "unknown"), ("g_a",), recall=0.5)
        self.assertIsNone(diagnose.retrieval_missing_gold(rec))

    def test_carries_gold_ranks_for_planner_when_measured(self):
        self._with(retrieve=["g_a", "x", "y", "g_b"])
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5)
        finding = diagnose.retrieval_missing_gold(rec)
        self.assertEqual(finding.metadata["gold_ranks"], {"g_a": 1, "g_b": 4})


class RetrievalEnumerationTest(_DiagnoseTestBase):
    """[설계 논의 중] 현행은 gold 개수 vs 실행 시점 top-k 비교만 본다.

    아래 두 테스트는 옳음을 주장하지 않는다 — qtype 을 안 보고, top_k 를 키우면
    임계가 함께 올라 라벨이 꺼지는 현행 동작을 기록해둔 것이다.
    """

    def test_single_gold_never_fires(self):
        rec = _record(("g_a",), ("x",), recall=0.0)
        self.assertIsNone(diagnose.retrieval_incomplete_enumeration(rec))

    def test_confirmed_when_gold_count_reaches_top_k(self):
        rec = _record(("g_a", "g_b", "g_c"), ("g_a", "x", "y"), recall=0.33)
        finding = diagnose.retrieval_incomplete_enumeration(rec)
        self.assertIsNotNone(finding)
        self.assertTrue(finding.confirmed)

    def test_fires_regardless_of_qtype(self):
        """[설계 논의 중] 나열형(aggregation) 전용이 아니라 comparison 에도 붙는다."""
        rec = _record(("g_a", "g_b", "g_c"), ("g_a", "x", "y"),
                      recall=0.33, qtype="comparison")
        self.assertIsNotNone(diagnose.retrieval_incomplete_enumeration(rec))

    def test_larger_top_k_raises_the_threshold_and_silences_label(self):
        """[설계 논의 중] 처방(top_k 증가)이 진단을 끈다 — gold 3개, top-k 10 → 미발동."""
        rec = _record(("g_a", "g_b", "g_c"), ["g_a"] + [f"x{i}" for i in range(9)],
                      recall=0.33)
        self.assertIsNone(diagnose.retrieval_incomplete_enumeration(rec))


class RetrievalBridgeTest(_DiagnoseTestBase):
    def test_preliminary_for_multi_hop_partial_recall(self):
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5, qtype="bridge")
        finding = diagnose.retrieval_missing_bridge_dependency(rec)
        self.assertIsNotNone(finding)
        self.assertFalse(finding.confirmed)          # tier4 제거 → 확정 불가

    def test_silent_for_single_hop(self):
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5, qtype=None)
        self.assertIsNone(diagnose.retrieval_missing_bridge_dependency(rec))

    def test_silent_when_recall_is_complete(self):
        rec = _record(("g_a", "g_b"), ("g_a", "g_b"), recall=1.0, qtype="bridge")
        self.assertIsNone(diagnose.retrieval_missing_bridge_dependency(rec))

    def test_silent_when_no_gold_exists(self):
        rec = _record((), (), recall=-1.0, qtype="bridge")
        self.assertIsNone(diagnose.retrieval_missing_bridge_dependency(rec))


class RetrievalRollupTest(_DiagnoseTestBase):
    def test_rollup_always_yields_preliminary_finding(self):
        """_RETRIEVAL_CAUSE 맨 뒤 — 세부 원인을 못 고를 때 슬롯이 비지 않게 한다."""
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5)
        finding = diagnose.retrieval_failure(rec)
        self.assertIsNotNone(finding)
        self.assertFalse(finding.confirmed)


# ══════════════════════════════════════════════════════════════════
#  B그룹: 생성 실패 (오라클 트랙 RAGAS 기반 · DEEP+)
# ══════════════════════════════════════════════════════════════════

class GenerationLabelTest(_DiagnoseTestBase):
    def setUp(self):
        super().setUp()
        metrics_common.set_mode(Mode.DEEP)          # B그룹은 RAGAS 필요

    def test_no_abstention_confirmed_when_model_answers_unanswerable(self):
        rec = _record(answer_exists=False, ground_truth=None, answer="지어낸 답")
        finding = diagnose.generation_no_abstention(rec)
        self.assertIsNotNone(finding)
        self.assertTrue(finding.confirmed)

    def test_no_abstention_silent_when_model_correctly_abstains(self):
        rec = _record(answer_exists=False, ground_truth=None,
                      answer="제공된 정보로는 알 수 없습니다")
        self.assertIsNone(diagnose.generation_no_abstention(rec))

    def test_hallucination_confirmed_below_faithfulness_threshold(self):
        rec = _record(oracle_f1=0.1, faith_oracle=RAGAS_FAITHFULNESS_MIN - 0.01)
        finding = diagnose.generation_hallucination(rec)
        self.assertTrue(finding.confirmed)
        self.assertEqual(finding.severity, "critical")

    def test_hallucination_silent_exactly_at_threshold(self):
        rec = _record(oracle_f1=0.1, faith_oracle=RAGAS_FAITHFULNESS_MIN)
        self.assertIsNone(diagnose.generation_hallucination(rec))

    def test_hallucination_silent_when_ragas_missing(self):
        metrics_common.set_mode(Mode.STANDARD)      # DEEP 미만 → faith None
        rec = _record(oracle_f1=0.1, faith_oracle=0.1)
        self.assertIsNone(diagnose.generation_hallucination(rec))

    def test_hop_binding_confirmed_for_multi_hop_with_high_faithfulness(self):
        rec = _record(oracle_f1=0.1, qtype="bridge", faith_oracle=0.9)
        finding = diagnose.generation_hop_binding_error(rec)
        self.assertTrue(finding.confirmed)

    def test_hop_binding_silent_for_single_hop(self):
        rec = _record(oracle_f1=0.1, qtype=None, faith_oracle=0.9)
        self.assertIsNone(diagnose.generation_hop_binding_error(rec))

    def test_hop_binding_ignores_relevancy(self):
        """[설계 논의 중] rules.py 는 이 라벨의 target_metric 을 answer_relevancy 로
        잡아뒀는데 판정은 faithfulness 만 본다 — 관련성이 바닥이어도 발동한다."""
        rec = _record(oracle_f1=0.1, qtype="bridge", faith_oracle=1.0, rel_oracle=0.0)
        self.assertIsNotNone(diagnose.generation_hop_binding_error(rec))

    def test_partial_answer_confirmed_below_relevancy_threshold(self):
        rec = _record(oracle_f1=0.1, rel_oracle=RAGAS_RESPONSE_RELEVANCY_MIN - 0.01)
        self.assertTrue(diagnose.generation_partial_answer(rec).confirmed)

    def test_partial_answer_silent_exactly_at_threshold(self):
        rec = _record(oracle_f1=0.1, rel_oracle=RAGAS_RESPONSE_RELEVANCY_MIN)
        self.assertIsNone(diagnose.generation_partial_answer(rec))

    def test_generation_failed_premise_requires_oracle_miss(self):
        self.assertTrue(diagnose._generation_failed(_record(oracle_f1=0.1)))
        self.assertFalse(diagnose._generation_failed(_record(oracle_f1=1.0)))


# ══════════════════════════════════════════════════════════════════
#  C그룹: context 구조 (tier4 제거로 3개는 dormant)
# ══════════════════════════════════════════════════════════════════

class ContextLabelTest(_DiagnoseTestBase):
    def setUp(self):
        super().setUp()
        metrics_common.set_mode(Mode.DEEP)

    def test_context_failed_premise_is_retrieval_ok_but_answer_wrong(self):
        self.assertTrue(diagnose._context_failed(
            _record(recall=1.0, oracle_f1=1.0, f1=F1_PASS_THRESHOLD - 0.01)))
        self.assertFalse(diagnose._context_failed(
            _record(recall=0.5, oracle_f1=1.0, f1=0.1)))       # 검색 실패는 A그룹

    def test_tier4_labels_are_dormant(self):
        """확정 신호(재실행)가 제거돼 optimize 재실행으로 대체됨 — 항상 None."""
        rec = _record(recall=1.0, oracle_f1=1.0, f1=0.1)
        self.assertIsNone(diagnose.too_long_context(rec))
        self.assertIsNone(diagnose.lost_in_the_middle(rec))
        self.assertIsNone(diagnose.context_noise_interference(rec))

    def test_bad_gold_answer_confirmed_when_both_ragas_high(self):
        rec = _record(recall=1.0, oracle_f1=1.0, f1=0.1, faith=0.8, rel=0.9)
        finding = diagnose.bad_gold_answer(rec)
        self.assertIsNotNone(finding)
        self.assertEqual(finding.type, "gap")                  # D그룹으로 분류
        self.assertEqual(finding.metadata["group"], "D")

    def test_bad_gold_answer_silent_when_only_one_metric_high(self):
        rec = _record(recall=1.0, oracle_f1=1.0, f1=0.1, faith=0.9, rel=0.1)
        self.assertIsNone(diagnose.bad_gold_answer(rec))

    def test_bad_gold_answer_silent_without_ragas(self):
        metrics_common.set_mode(Mode.STANDARD)
        rec = _record(recall=1.0, oracle_f1=1.0, f1=0.1, faith=0.9, rel=0.9)
        self.assertIsNone(diagnose.bad_gold_answer(rec))


# ══════════════════════════════════════════════════════════════════
#  D그룹: 데이터 결손
# ══════════════════════════════════════════════════════════════════

class GapLabelTest(_DiagnoseTestBase):
    def test_corpus_gap_for_single_hop_when_gold_absent(self):
        rec = _record(("unknown",), ("x",), recall=0.0, qtype=None)
        finding = diagnose.corpus_gap(rec)
        self.assertTrue(finding.confirmed)
        self.assertEqual(finding.severity, "critical")
        self.assertIsNone(diagnose.corpus_gap_partial_hop(rec))        # 배타

    def test_corpus_gap_partial_hop_for_multi_hop_when_gold_absent(self):
        rec = _record(("unknown",), ("x",), recall=0.0, qtype="bridge")
        self.assertTrue(diagnose.corpus_gap_partial_hop(rec).confirmed)
        self.assertIsNone(diagnose.corpus_gap(rec))                    # 배타

    def test_both_silent_when_gold_present_in_corpus(self):
        rec = _record(("g_a",), ("x",), recall=0.0)
        self.assertIsNone(diagnose.corpus_gap(rec))
        self.assertIsNone(diagnose.corpus_gap_partial_hop(rec))


# ══════════════════════════════════════════════════════════════════
#  조립: 성공 게이트 · _pick 우선순위 · 정렬
# ══════════════════════════════════════════════════════════════════

class AssemblyTest(_DiagnoseTestBase):
    def test_success_gate_passes_correct_abstention(self):
        rec = _record(answer_exists=False, ground_truth=None,
                      answer="제공된 정보로는 알 수 없습니다")
        self.assertIs(diagnose._is_success(rec), True)

    def test_success_gate_undecidable_without_ground_truth(self):
        rec = _record(ground_truth=None)
        self.assertIsNone(diagnose._is_success(rec))

    def test_success_gate_requires_both_recall_and_answer(self):
        self.assertTrue(diagnose._is_success(_record(recall=1.0, f1=1.0)))
        self.assertFalse(diagnose._is_success(_record(recall=0.5, f1=1.0)))
        self.assertFalse(diagnose._is_success(_record(recall=1.0, f1=0.1)))

    def test_pick_prefers_confirmed_over_earlier_preliminary(self):
        """순서가 앞서도 예비는 밀린다 — 뒤에 확정이 있으면 그쪽을 채택한다."""
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5, qtype="bridge")
        # 앞자리가 실제로 '예비'를 내는지부터 못 박는다(None 이면 우선순위 검증이 무의미).
        earlier = diagnose.retrieval_missing_bridge_dependency(rec)
        self.assertIsNotNone(earlier)
        self.assertFalse(earlier.confirmed)

        picked = diagnose._pick(rec, (
            diagnose.retrieval_missing_bridge_dependency,   # 예비 (앞)
            diagnose.retrieval_missing_gold,                # 확정 (뒤)
            diagnose.retrieval_failure,                     # 예비 롤업
        ))
        self.assertEqual(picked.label, "retrieval_missing_gold")
        self.assertTrue(picked.confirmed)

    def test_pick_falls_back_to_first_preliminary(self):
        metrics_common.set_mode(Mode.FAST)
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5)
        picked = diagnose._pick(rec, (diagnose.retrieval_low_rank,   # None
                                      diagnose.retrieval_failure))   # 예비
        self.assertEqual(picked.label, "retrieval_failure")
        self.assertFalse(picked.confirmed)

    def test_dedup_keeps_first_occurrence_per_label(self):
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5)
        dup = [diagnose.retrieval_missing_gold(rec), diagnose.retrieval_missing_gold(rec)]
        self.assertEqual(len(diagnose._dedup(dup)), 1)

    def test_group_derivation_matches_prescription_order(self):
        self.assertEqual(diagnose._group_of("corpus_gap", "gap"), "D")
        self.assertEqual(diagnose._group_of("retrieval_low_rank", "retrieval_failure"), "A")
        self.assertEqual(diagnose._group_of("chunking_context_mismatch", "retrieval_failure"), "A")
        self.assertEqual(diagnose._group_of("generation_hallucination", "generation_failure"), "B")
        self.assertEqual(diagnose._group_of("context_failure", "context_failure"), "C")


if __name__ == "__main__":
    unittest.main(verbosity=2)
