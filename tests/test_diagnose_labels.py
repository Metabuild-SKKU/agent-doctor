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
from agents.eval import metrics_common, metrics_search, diagnose
from agents.eval.types import (
    EvalRecord, Mode,
    F1_PASS_THRESHOLD, RAGAS_FAITHFULNESS_MIN, RAGAS_RESPONSE_RELEVANCY_MIN,
    CONTEXT_CHARS_MAX,
)


# ── 픽스처 ────────────────────────────────────────────────────────

def _record(
    gold_ids=("g_a",), retrieved_ids=("g_a",), *,
    recall=1.0, f1=1.0, oracle_f1=1.0, qtype=None,
    answer_exists=None, ground_truth="정답", answer="답변", oracle_answer="오라클 답변",
    faith=None, rel=None, faith_oracle=None, rel_oracle=None,
    gold_spans=None, counts_oracle=None,
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
    if counts_oracle is not None:                       # (tp, fp, fn)
        tp, fp, fn = counts_oracle
        rec.oracle_ragas.update({"answer_correctness_tp": tp,
                                 "answer_correctness_fp": fp,
                                 "answer_correctness_fn": fn})
    rec.oracle_ragas_done = True
    return rec


def _spans(n, doc="d1"):
    """개수만 필요한 곳에 쓰는 더미 gold_spans n개(좌표는 겹치지 않는 유효값)."""
    return [{"doc_id": doc, "start": i * 10, "end": i * 10 + 5} for i in range(n)]


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


class _FakeAbstentionJudge:
    """abstention 트랙만 응답하는 가짜 ragas_fn(호출 횟수 기록)."""

    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = []

    def __call__(self, record, track):
        self.calls.append(track)
        return {"abstention": self.verdict} if track == "abstention" else {}


class _FakeReasoningJudge:
    """reasoning_mode 트랙만 응답하는 가짜 ragas_fn(호출 횟수 기록)."""

    def __init__(self, mode):
        self.mode = mode
        self.calls = []

    def __call__(self, record, track):
        self.calls.append(track)
        if track != "reasoning_mode":
            return {}
        return {"reasoning_mode": self.mode} if self.mode is not None else {}


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

    def _with(self, *, retrieve=None, keyword=None, ragas=None):
        metrics_common.set_context(
            chunks=self._chunks,
            retrieve_fn=_FakeRetriever(retrieve) if retrieve is not None else None,
            keyword_fn=_FakeKeyword(keyword) if keyword is not None else None,
            ragas_fn=ragas,
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
        self.assertIsNot(diagnose._enumeration_pressure(rec), False)     # 개수 전제는 성립(legacy)
        self.assertIsNone(diagnose.retrieval_incomplete_enumeration(rec))   # 놓친 gold 없음 → 침묵

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

    def test_silent_when_missed_gold_ranks_within_top_k(self):
        """재검색이 top_k 이내 순위를 내면 '순위가 낮아 밖'과 모순 → 검색 비결정성이지 순위 문제 아님."""
        self._with(retrieve=["g_a", "g_b", "x"])                # g_b 는 2위인데 top_k=3
        rec = _record(("g_a", "g_b"), ("g_a", "x", "y"), recall=0.5)
        self.assertIsNone(diagnose.retrieval_low_rank(rec))

    def test_reason_carries_measured_ranks(self):
        self._with(retrieve=["g_a", "x", "y", "g_b"])           # g_b 는 4위, top_k=1
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5)
        self.assertIn("g_b:4", diagnose.retrieval_low_rank(rec).metadata["reason"])

    def test_metadata_marks_candidate_miss_when_gold_is_outside_rerank_window(self):
        self._with(retrieve=["g_a", "x", "y", "g_b"])
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5)
        rec.retrieval_details = {
            "reranker_enabled": True,
            "reranked": True,
            "rerank_candidates": 2,
        }

        finding = diagnose.retrieval_low_rank(rec)

        self.assertEqual(finding.metadata["low_rank_cause"], "candidate_miss")
        self.assertEqual(
            finding.metadata["suggested_prescription"],
            "widen_rerank_candidates",
        )
        self.assertEqual(finding.metadata["lowest_missed_gold_rank"], 4)

    def test_metadata_marks_reranker_insufficient_when_gold_was_reranked(self):
        self._with(retrieve=["g_a", "x", "y", "g_b"])
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5)
        rec.retrieval_details = {
            "reranker_enabled": True,
            "reranked": True,
            "rerank_candidates": 10,
        }

        finding = diagnose.retrieval_low_rank(rec)

        self.assertEqual(
            finding.metadata["low_rank_cause"],
            "reranker_applied_but_insufficient",
        )
        self.assertEqual(
            finding.metadata["suggested_prescription"],
            "widen_rerank_candidates",
        )

    def test_metadata_marks_lexical_overlap_when_bm25_also_hits_gold(self):
        self._with(retrieve=["g_a", "x", "y", "g_b"], keyword=["g_b"])
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5)

        finding = diagnose.retrieval_low_rank(rec)

        self.assertEqual(finding.metadata["low_rank_cause"], "lexical_overlap_low_rank")
        self.assertEqual(
            finding.metadata["suggested_prescription"],
            "enable_reranker",
        )

    def test_silent_when_retrieval_returned_nothing(self):
        self._with(retrieve=["g_a", "x", "g_b"])
        rec = _record(("g_a", "g_b"), (), recall=0.0)
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

    def test_semantic_confirmed_for_mixed_corpus_gold(self):
        """놓친 gold 가 {코퍼스 있음+없음} 혼합 — 코퍼스에 있는 몫은 검색 실패로 남아야 한다.
        (예전엔 probe 단위 bool 이라 라벨이 통째로 소실되고 corpus_gap 만 남았다.)"""
        self._with(keyword=["zzz"])
        rec = _record(("g_a", "unknown"), ("x",), recall=0.0)
        finding = diagnose.retrieval_semantic_mismatch(rec)
        self.assertIsNotNone(finding)
        self.assertTrue(finding.confirmed)
        self.assertIn("missed_gold_in_corpus=1/2", finding.metadata["reason"])

    def test_semantic_silent_when_every_missed_gold_is_outside_corpus(self):
        self._with(keyword=["zzz"])
        rec = _record(("unknown", "unknown2"), ("x",), recall=0.0)
        self.assertIsNone(diagnose.retrieval_semantic_mismatch(rec))

    def test_bm25_target_excludes_gold_outside_corpus(self):
        """코퍼스 밖 gold 를 BM25 가 잡는 건 lexical 근거가 아니다 — corpus_gap 몫."""
        self._with(keyword=["unknown"])
        rec = _record(("g_a", "unknown"), ("x",), recall=0.0)
        self.assertIs(metrics_search._bm25_hits_gold(rec), False)
        self.assertIsNone(diagnose.retrieval_lexical_mismatch(rec))

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

    def test_confirmed_for_mixed_corpus_gold(self):
        """코퍼스에 있는데 못 찾은 gold 가 하나라도 있으면 검색 실패다(나머지는 corpus_gap)."""
        rec = _record(("g_a", "unknown"), ("x",), recall=0.0)
        finding = diagnose.retrieval_missing_gold(rec)
        self.assertTrue(finding.confirmed)
        self.assertIn("missed_gold_in_corpus=1/2", finding.metadata["reason"])

    def test_silent_when_every_missed_gold_is_outside_corpus(self):
        rec = _record(("unknown", "unknown2"), ("x",), recall=0.0)
        self.assertIsNone(diagnose.retrieval_missing_gold(rec))

    def test_carries_gold_ranks_for_planner_when_measured(self):
        self._with(retrieve=["g_a", "x", "y", "g_b"])
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5)
        finding = diagnose.retrieval_missing_gold(rec)
        self.assertEqual(finding.metadata["gold_ranks"], {"g_a": 1, "g_b": 4})


class RetrievalEnumerationTest(_DiagnoseTestBase):
    """개수 압박은 청킹 불변량인 gold_spans 수로 센다(gold_chunk_ids 아님).
    확정 = span 압박 + qtype=aggregation. None qtype·legacy·wide-N 밖은 예비/침묵."""

    def test_single_span_never_fires(self):
        rec = _record(("g_a",), ("x",), recall=0.0, qtype="aggregation", gold_spans=_spans(1))
        self.assertIsNone(diagnose.retrieval_incomplete_enumeration(rec))

    def test_confirmed_when_aggregation_and_span_pressure(self):
        rec = _record(("g_a", "g_b", "g_c"), ("g_a", "x", "y"),
                      recall=0.33, qtype="aggregation", gold_spans=_spans(3))   # k=3, span 3 압박
        finding = diagnose.retrieval_incomplete_enumeration(rec)
        self.assertIsNotNone(finding)
        self.assertTrue(finding.confirmed)

    def test_preliminary_when_qtype_untagged(self):
        """span 압박은 성립하나 qtype 미태깅(None) → 확정 못 하고 예비."""
        rec = _record(("g_a", "g_b", "g_c"), ("g_a", "x", "y"),
                      recall=0.33, qtype=None, gold_spans=_spans(3))
        finding = diagnose.retrieval_incomplete_enumeration(rec)
        self.assertIsNotNone(finding)
        self.assertFalse(finding.confirmed)

    def test_preliminary_when_spans_absent_legacy(self):
        """gold_spans 미제공(legacy) → 청크 수 폴백, aggregation 이어도 예비."""
        rec = _record(("g_a", "g_b", "g_c"), ("g_a", "x", "y"),
                      recall=0.33, qtype="aggregation")               # gold_spans 없음
        finding = diagnose.retrieval_incomplete_enumeration(rec)
        self.assertIsNotNone(finding)
        self.assertFalse(finding.confirmed)

    def test_silent_when_chunk_count_inflated_but_few_spans(self):
        """gold_chunk_ids 는 많아도(세밀 청킹) 근거 span 은 1개 → 나열형 아님, 침묵(→chunking)."""
        rec = _record(("g_a", "g_b", "g_c"), ("g_a", "x", "y"),
                      recall=0.33, qtype="aggregation", gold_spans=_spans(1))
        self.assertIsNone(diagnose.retrieval_incomplete_enumeration(rec))

    def test_silent_when_all_missed_outside_wide_search(self):
        """놓친 gold 가 wide-N 밖이면 top_k↑ 무효 → semantic 영역, 침묵."""
        self._with(retrieve=["g_a", "p", "q"])                       # 놓친 g_b·g_c 가 wide 밖
        rec = _record(("g_a", "g_b", "g_c"), ("g_a", "x", "y"),
                      recall=0.33, qtype="aggregation", gold_spans=_spans(3))
        self.assertIsNone(diagnose.retrieval_incomplete_enumeration(rec))

    def test_silent_when_retrieval_returned_nothing(self):
        """검색 0건(장애)은 슬롯 부족이 아니라 롤업 영역 → 침묵."""
        rec = _record(("g_a", "g_b", "g_c"), (),
                      recall=0.0, qtype="aggregation", gold_spans=_spans(3))
        self.assertIsNone(diagnose.retrieval_incomplete_enumeration(rec))

    def test_larger_top_k_raises_the_threshold_and_silences_label(self):
        """top_k 를 키우면 임계도 오른다 — span 3, top-k 10 → 압박 해소, 미발동."""
        rec = _record(("g_a", "g_b", "g_c"), ["g_a"] + [f"x{i}" for i in range(9)],
                      recall=0.33, qtype="aggregation", gold_spans=_spans(3))
        self.assertIsNone(diagnose.retrieval_incomplete_enumeration(rec))


class ChunkingOverchunkingTest(_DiagnoseTestBase):
    """span 이 최장 청크보다 길면 겹침으로는 절대 못 담는다 — overlap 처방이 무효인 구간."""

    def _rec(self, span_len, chunk_len=100):
        chunks = [Chunk(f"g_{i}", "d1", "본문", char_span=(i * chunk_len, (i + 1) * chunk_len))
                  for i in range(4)]
        metrics_common.set_context(chunks=chunks)
        return _record(("g_0", "g_1"), ("g_0",), recall=0.5,
                       gold_spans=[{"doc_id": "d1", "start": 0, "end": span_len}])

    def test_confirmed_when_span_longer_than_any_chunk(self):
        finding = diagnose.chunking_overchunking(self._rec(span_len=250))   # 청크 100
        self.assertIsNotNone(finding)
        self.assertTrue(finding.confirmed)                  # 기하 사실이라 확정
        self.assertEqual(finding.metadata["group"], "A")
        self.assertEqual(finding.metadata["oversized_analysis"]["oversized_count"], 1)

    def test_silent_when_span_fits_in_a_chunk(self):
        self.assertIsNone(diagnose.chunking_overchunking(self._rec(span_len=60)))

    def test_context_mismatch_yields_to_overchunking(self):
        """겹침으로 못 담는 길이면 overlap 처방 라벨은 양보한다.
        (_RETRIEVAL_CAUSE 전체로는 실측된 다른 검색 원인이 먼저 채택되므로 — 설계대로 —
         두 chunking 라벨만 놓고 배타를 확인한다.)"""
        rec = self._rec(span_len=250)
        self.assertIsNone(diagnose.chunking_context_mismatch(rec))
        picked = diagnose._pick(rec, (diagnose.chunking_overchunking,
                                      diagnose.chunking_context_mismatch))
        self.assertEqual(picked.label, "chunking_overchunking")

    def test_context_mismatch_still_owns_recoverable_split(self):
        """청크에 담기는 길이의 경계 분할은 그대로 chunking_context_mismatch."""
        chunks = [Chunk(f"g_{i}", "d1", "본문", char_span=(i * 100, (i + 1) * 100))
                  for i in range(4)]
        metrics_common.set_context(chunks=chunks)
        rec = _record(("g_0", "g_1"), ("g_0",), recall=0.5,
                      gold_spans=[{"doc_id": "d1", "start": 80, "end": 120}])   # 40자, 경계 걸침
        self.assertIsNone(diagnose.chunking_overchunking(rec))
        self.assertTrue(diagnose.chunking_context_mismatch(rec).confirmed)


class ChunkingGateTest(_DiagnoseTestBase):
    """chunking 은 enumeration 과 반대 게이트로 배타 — span 압박이면 나열형에 양보.
    base 청크 좌표: g_a(0,100)·g_b(100,200)·g_c(200,300) in d1."""

    def test_fires_on_boundary_split_without_span_pressure(self):
        """경계 분할 span 1개(압박 없음) → chunking 확정."""
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5,
                      gold_spans=[{"doc_id": "d1", "start": 80, "end": 120}])   # g_a·g_b 경계
        finding = diagnose.chunking_context_mismatch(rec)
        self.assertIsNotNone(finding)
        self.assertTrue(finding.confirmed)
        self.assertIsNone(diagnose.retrieval_incomplete_enumeration(rec))       # 압박 없음

    def test_yields_to_enumeration_under_span_pressure(self):
        """경계 분할이 있어도 span 개수 압박이면 슬롯 부족이 지배 → chunking 침묵, enumeration 이 가져감."""
        rec = _record(("g_a", "g_b", "g_c"), ("g_a", "x", "y"), recall=0.33, qtype="aggregation",
                      gold_spans=[{"doc_id": "d1", "start": 80, "end": 120},
                                  {"doc_id": "d1", "start": 180, "end": 220}])  # 2 경계 분할, k=3
        self.assertIsNone(diagnose.chunking_context_mismatch(rec))
        self.assertTrue(diagnose.retrieval_incomplete_enumeration(rec).confirmed)


class RetrievalBridgeTest(_DiagnoseTestBase):
    def test_preliminary_for_multi_hop_partial_recall(self):
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5, qtype="bridge")
        finding = diagnose.retrieval_missing_bridge_dependency(rec)
        self.assertIsNotNone(finding)
        self.assertFalse(finding.confirmed)          # hop 의존 판별 신호 미측정 → 확정 불가

    def test_silent_for_single_hop(self):
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5, qtype=None)
        self.assertIsNone(diagnose.retrieval_missing_bridge_dependency(rec))

    def test_silent_for_independent_hop_types(self):
        """comparison/aggregation 은 hop 간 독립 → bridge 의존 아님."""
        for qtype in ("comparison", "aggregation"):
            rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5, qtype=qtype)
            self.assertIsNone(diagnose.retrieval_missing_bridge_dependency(rec))

    def test_bridge_probe_wins_slot_over_semantic_and_missing_gold(self):
        """bridge qtype 이면 semantic·missing_gold(양립 증거)는 양보, bridge 예비가 슬롯 채택."""
        self._with(keyword=["zzz"])                              # BM25 도 놓침
        rec = _record(("g_a", "g_b"), ("g_a", "x", "y", "z", "w"),   # k=5 → 개수 압박 없음
                      recall=0.5, qtype="bridge")
        self.assertIsNone(diagnose.retrieval_semantic_mismatch(rec))
        self.assertIsNone(diagnose.retrieval_missing_gold(rec))
        picked = diagnose._pick(rec, diagnose._RETRIEVAL_CAUSE)
        self.assertEqual(picked.label, "retrieval_missing_bridge_dependency")

    def test_lexical_still_beats_bridge_when_bm25_catches_gold(self):
        """BM25 로 잡히면 원 질문으로 회복 가능 = bridge 반증 → lexical 확정이 우선."""
        self._with(keyword=["g_b"])
        rec = _record(("g_a", "g_b"), ("g_a", "x", "y", "z", "w"),
                      recall=0.5, qtype="bridge")
        picked = diagnose._pick(rec, diagnose._RETRIEVAL_CAUSE)
        self.assertEqual(picked.label, "retrieval_lexical_mismatch")
        self.assertTrue(picked.confirmed)

    def test_silent_when_recall_is_complete(self):
        rec = _record(("g_a", "g_b"), ("g_a", "g_b"), recall=1.0, qtype="bridge")
        self.assertIsNone(diagnose.retrieval_missing_bridge_dependency(rec))

    def test_silent_when_no_gold_exists(self):
        rec = _record((), (), recall=-1.0, qtype="bridge")
        self.assertIsNone(diagnose.retrieval_missing_bridge_dependency(rec))


class RollupTest(_DiagnoseTestBase):
    """롤업 3종 — 슬롯 맨 뒤·항상 예비. 구체 라벨이 하나라도 서면 밀려난다."""

    ROLLUPS = (
        (diagnose.retrieval_failure, "retrieval_failure", "A"),
        (diagnose.generation_failure, "generation_failure", "B"),
        (diagnose.context_failure, "context_failure", "C"),
    )

    def test_rollups_are_always_preliminary(self):
        """_pick 맨 뒤 — 세부 원인을 못 고를 때 슬롯이 비지 않게 한다."""
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5)
        for fn, label, group in self.ROLLUPS:
            with self.subTest(label=label):
                finding = fn(rec)
                self.assertFalse(finding.confirmed)
                self.assertEqual(finding.label, label)
                self.assertEqual(finding.metadata["group"], group)

    def test_rollup_reason_records_mode_when_metrics_are_blank(self):
        """저모드에선 지표가 전부 '-' 라, 왜 롤업인지는 실행 모드로만 알 수 있다."""
        metrics_common.set_mode(Mode.FAST)
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5)
        reason = diagnose.retrieval_failure(rec).metadata["reason"]
        self.assertIn(f"mode={Mode.FAST}", reason)
        self.assertIn("faithfulness=-", reason)

    def test_rollup_yields_to_any_specific_cause(self):
        """구체 라벨이 예비여도 롤업보다 앞이라 채택된다(롤업은 최후)."""
        metrics_common.set_mode(Mode.FAST)                     # tier2 없음 → 구체 라벨도 예비
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5)
        self.assertNotEqual(diagnose._pick(rec, diagnose._RETRIEVAL_CAUSE).label,
                            "retrieval_failure")

    def test_rollups_have_no_prescription_and_stay_report_only(self):
        """롤업은 rules.py 에 항목이 없어 optimize 가 건너뛴다 — 리포트 전용이라는 계약."""
        from agents.optimize import rules
        for _fn, label, _group in self.ROLLUPS:
            with self.subTest(label=label):
                self.assertIsNone(rules.get_rule(label))
                self.assertFalse(rules.is_actionable(label))


# ══════════════════════════════════════════════════════════════════
#  A 슬롯 진입 전제 — recall<1 만으로는 열지 않는다
#    구체 라벨은 self-scope 로 빠져도 롤업이 무조건 붙어 '검색을 고쳐라'가 남기 때문에,
#    검색으로 고칠 수 없는 실패는 슬롯 자체를 닫아야 한다.
# ══════════════════════════════════════════════════════════════════

class RetrievalSlotGateTest(_DiagnoseTestBase):
    def _a_labels(self, rec):
        return {f.label for f in diagnose.diagnose(rec, Mode.STANDARD)
                if f.metadata["group"] == "A"}

    def test_closed_when_every_gold_is_outside_corpus(self):
        """자료가 없는 걸 검색이 고칠 수는 없다 — corpus_gap 만 남아야 한다."""
        rec = _record(("unknown",), ("x",), recall=0.0, answer="엉뚱한 말")
        self.assertFalse(diagnose._retrieval_fixable(rec))
        self.assertEqual(self._a_labels(rec), set())

    def test_open_on_partial_gap(self):
        """코퍼스에 있는 몫은 실제로 검색이 놓친 것 — '전부 없음'이 아니면 닫지 않는다."""
        rec = _record(("g_a", "unknown"), ("x",), recall=0.0, answer="엉뚱한 말")
        self.assertTrue(diagnose._retrieval_fixable(rec))
        self.assertTrue(self._a_labels(rec))

    def test_closed_for_false_premise_probe(self):
        """answer_exists=False 의 gold 는 답의 근거가 아니라 전제를 반박하는 청크라
        recall 이 성립하지 않는다 — 그걸 미검색으로 세면 무응답 probe 에 검색 처방이 붙는다."""
        rec = _record(("g_a",), ("x",), recall=0.0, answer_exists=False,
                      ground_truth=None, answer="지어낸 답")
        self.assertFalse(diagnose._retrieval_fixable(rec))
        self.assertEqual(self._a_labels(rec), set())

    def test_open_when_gold_is_in_corpus(self):
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5)
        self.assertTrue(diagnose._retrieval_fixable(rec))


# ══════════════════════════════════════════════════════════════════
#  B그룹: 생성 실패 (오라클 트랙 RAGAS 기반 · DEEP+)
# ══════════════════════════════════════════════════════════════════

class GenerationLabelTest(_DiagnoseTestBase):
    def setUp(self):
        super().setUp()
        metrics_common.set_mode(Mode.DEEP)          # B그룹은 RAGAS 필요

    def test_no_abstention_confirmed_when_model_answers_unanswerable(self):
        rec = _record(answer_exists=False, ground_truth=None, answer="지어낸 답")
        finding = diagnose.generation_abstention_failure(rec)
        self.assertIsNotNone(finding)
        self.assertTrue(finding.confirmed)

    def test_no_abstention_silent_when_model_correctly_abstains(self):
        rec = _record(answer_exists=False, ground_truth=None,
                      answer="제공된 정보로는 알 수 없습니다")
        self.assertIsNone(diagnose.generation_abstention_failure(rec))

    def test_no_abstention_is_critical(self):
        rec = _record(answer_exists=False, ground_truth=None, answer="지어낸 답")
        self.assertEqual(diagnose.generation_abstention_failure(rec).severity, "critical")

    def test_aspect_critic_overrides_heuristic_miss_at_deep(self):
        """마커가 없어 휴리스틱은 '기권 아님'이라 보지만, AspectCritic 이 기권으로 판정 → 침묵."""
        judge = _FakeAbstentionJudge(1)
        self._with(ragas=judge)
        rec = _record(answer_exists=False, ground_truth=None,
                      answer="그 부분은 문서에서 다루지 않습니다")   # 마커 미포함
        self.assertTrue(diagnose.is_abstention(rec.generated_answer) is False)
        self.assertIsNone(diagnose.generation_abstention_failure(rec))
        self.assertIn("abstention", judge.calls)

    def test_aspect_critic_catches_marker_false_positive_at_deep(self):
        """마커('알 수 없')를 품었지만 실제론 답을 지어낸 케이스 — AspectCritic 이 잡아낸다."""
        judge = _FakeAbstentionJudge(0)
        self._with(ragas=judge)
        rec = _record(answer_exists=False, ground_truth=None,
                      answer="정확히는 알 수 없지만 답은 42입니다")
        self.assertTrue(diagnose.is_abstention(rec.generated_answer))   # 휴리스틱은 기권으로 오판
        finding = diagnose.generation_abstention_failure(rec)
        self.assertIsNotNone(finding)
        self.assertTrue(finding.confirmed)
        self.assertIn("aspect_critic", finding.metadata["reason"])

    def test_heuristic_used_and_no_llm_call_below_deep(self):
        judge = _FakeAbstentionJudge(1)
        self._with(ragas=judge)
        metrics_common.set_mode(Mode.STANDARD)                  # DEEP 미만 → 판정 호출 없음
        rec = _record(answer_exists=False, ground_truth=None, answer="지어낸 답")
        finding = diagnose.generation_abstention_failure(rec)
        self.assertTrue(finding.confirmed)
        self.assertIn("heuristic", finding.metadata["reason"])
        self.assertEqual(judge.calls, [])

    def test_aspect_critic_is_memoized_per_run_not_in_signals(self):
        """판정은 answer 의존이라 record.aspect(실행 단위)에 memoize — signals 로 새면
        index_config 가 그대로인 재실행에서 옛 답변의 판정을 재사용하게 된다."""
        judge = _FakeAbstentionJudge(0)
        self._with(ragas=judge)
        rec = _record(answer_exists=False, ground_truth=None, answer="지어낸 답")
        for _ in range(3):
            diagnose.generation_abstention_failure(rec)
        self.assertEqual(judge.calls.count("abstention"), 1)         # 3회 호출 → LLM 1회
        self.assertEqual(rec.aspect["abstention"], False)
        self.assertNotIn("abstention_judged", rec.signals)           # signals 오염 없음

    def test_aspect_critic_failure_is_not_retried(self):
        """ragas_fn 이 {} 폴백이어도 같은 실행에선 재호출하지 않고 휴리스틱으로 간다."""
        judge = _FakeAbstentionJudge(None)                           # {"abstention": None}
        self._with(ragas=judge)
        rec = _record(answer_exists=False, ground_truth=None, answer="지어낸 답")
        for _ in range(3):
            finding = diagnose.generation_abstention_failure(rec)
        self.assertEqual(judge.calls.count("abstention"), 1)
        self.assertTrue(finding.confirmed)                           # 휴리스틱 폴백으로 확정
        self.assertIn("heuristic", finding.metadata["reason"])

    def test_empty_answer_is_not_abstention(self):
        """빈 답변(LLM 오류·타임아웃)을 '올바른 기권'으로 통과시키면 생성 장애가 성공으로 집계된다."""
        judge = _FakeAbstentionJudge(1)
        self._with(ragas=judge)
        rec = _record(answer_exists=False, ground_truth=None, answer="")
        self.assertFalse(diagnose._abstained(rec))
        self.assertIs(diagnose._is_success(rec), False)          # 성공으로 새지 않는다
        self.assertTrue(diagnose._generation_failed(rec))        # B슬롯은 열린다
        self.assertEqual(judge.calls, [])                        # 판정 대상 아님 → LLM 미호출

    def test_empty_answer_not_labelled_as_fabrication(self):
        """빈 답변은 '지어냄'이 아니다 — 라벨 대신 롤업으로 간다."""
        rec = _record(answer_exists=False, ground_truth=None, answer="")
        self.assertIsNone(diagnose.generation_abstention_failure(rec))
        picked = diagnose._pick(rec, diagnose._GENERATION_CAUSE)
        self.assertEqual(picked.label, "generation_failure")
        self.assertFalse(picked.confirmed)

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

    def test_reasoning_modes_map_to_their_labels(self):
        """분류기 한 번 호출로 세 라벨이 갈린다 — 함수는 하나, 라벨은 셋."""
        for mode, label in (("contradiction", "generation_contradiction"),
                            ("numerical_error", "generation_numerical_error"),
                            ("misinterpretation", "generation_misinterpretation")):
            self._with(ragas=_FakeReasoningJudge(mode))
            rec = _record(oracle_f1=0.1, faith_oracle=0.9, rel_oracle=0.9)
            finding = diagnose.generation_reasoning_failure(rec)
            self.assertIsNotNone(finding, mode)
            self.assertEqual(finding.label, label)
            self.assertTrue(finding.confirmed)

    def test_hop_binding_mode_maps_to_its_label_for_multi_hop(self):
        self._with(ragas=_FakeReasoningJudge("hop_binding"))
        rec = _record(oracle_f1=0.1, qtype="bridge", faith_oracle=0.9, rel_oracle=0.9)
        finding = diagnose.generation_reasoning_failure(rec)
        self.assertEqual(finding.label, "generation_hop_binding_error")
        self.assertTrue(finding.confirmed)

    def test_single_hop_binding_absorbed_into_misinterpretation(self):
        """단일홉엔 엮을 hop 이 없다 — 롤업으로 버리지 않고 관계 오독으로 흡수한다."""
        self._with(ragas=_FakeReasoningJudge("hop_binding"))
        rec = _record(oracle_f1=0.1, qtype=None, faith_oracle=0.9, rel_oracle=0.9)
        finding = diagnose.generation_reasoning_failure(rec)
        self.assertEqual(finding.label, "generation_misinterpretation")
        self.assertTrue(finding.confirmed)

    def test_reasoning_failure_silent_for_other(self):
        """'other'는 구체적 원인 지목이 아니라 롤업 몫."""
        self._with(ragas=_FakeReasoningJudge("other"))
        rec = _record(oracle_f1=0.1, faith_oracle=0.9, rel_oracle=0.9)
        self.assertIsNone(diagnose.generation_reasoning_failure(rec))

    def test_classifier_not_called_when_faithfulness_already_confirms(self):
        """faith 낮으면 hallucination 이 결정 — 분류기 호출 안 함(LLM 1회 절약)."""
        judge = _FakeReasoningJudge("contradiction")
        self._with(ragas=judge)
        rec = _record(oracle_f1=0.1, faith_oracle=RAGAS_FAITHFULNESS_MIN - 0.01)
        self.assertTrue(diagnose.generation_hallucination(rec).confirmed)
        self.assertIsNone(diagnose.generation_reasoning_failure(rec))
        self.assertEqual(judge.calls, [])

    def test_reasoning_failure_beats_bad_gold_answer_in_slot(self):
        """구체적 추론 실패가 지목되면 '정답셋이 틀렸다'가 반증된다."""
        self._with(ragas=_FakeReasoningJudge("contradiction"))
        rec = _record(oracle_f1=0.1, faith_oracle=0.9, rel_oracle=0.9)
        self.assertIsNone(diagnose.bad_gold_answer_oracle(rec))
        picked = diagnose._pick(rec, diagnose._GENERATION_CAUSE)
        self.assertEqual(picked.label, "generation_contradiction")
        self.assertEqual(picked.severity, "critical")

    def test_classifier_overrides_count_based_hop_binding(self):
        """카운트상 결합 오류 조건이어도 분류기 지목이 우선한다(폴백은 미측정 때만)."""
        self._with(ragas=_FakeReasoningJudge("numerical_error"))
        rec = _record(oracle_f1=0.1, qtype="bridge", faith_oracle=0.9,
                      counts_oracle=(3, 2, 0))            # 카운트상으론 hop_binding 조건
        self.assertEqual(diagnose.generation_reasoning_failure(rec).label,
                         "generation_numerical_error")

    def test_classifier_confirms_hop_binding_without_counts(self):
        """분류기가 hop_binding 을 지목하면 카운트 없이도 확정된다."""
        self._with(ragas=_FakeReasoningJudge("hop_binding"))
        rec = _record(oracle_f1=0.1, qtype="bridge", faith_oracle=0.9)
        finding = diagnose.generation_reasoning_failure(rec)
        self.assertTrue(finding.confirmed)
        self.assertIn("reasoning_mode=hop_binding", finding.metadata["reason"])

    def test_other_mode_keeps_bad_gold_answer(self):
        """'other' 는 구체적 원인 지목이 아니라 기존 판정을 그대로 둔다."""
        self._with(ragas=_FakeReasoningJudge("other"))
        rec = _record(oracle_f1=0.1, faith_oracle=0.9, rel_oracle=0.9)
        self.assertIsNotNone(diagnose.bad_gold_answer_oracle(rec))

    def test_count_fallback_beats_bad_gold_answer_in_slot(self):
        """분류기 미측정이어도 카운트가 결합 오류를 지목하면 '정답셋 오류'는 반증된다.
        (예전엔 bad_gold_answer_oracle 이 튜플상 앞이라 이 확정을 선점했다.)"""
        rec = _record(oracle_f1=0.1, qtype="bridge", faith_oracle=0.9, rel_oracle=0.9,
                      counts_oracle=(3, 2, 0))
        self.assertIsNone(diagnose.bad_gold_answer_oracle(rec))
        picked = diagnose._pick(rec, diagnose._GENERATION_CAUSE)
        self.assertEqual(picked.label, "generation_hop_binding_error")
        self.assertTrue(picked.confirmed)

    def test_count_fallback_keeps_bad_gold_when_counts_unmeasured(self):
        """카운트도 없으면 결합 오류는 예비뿐이라 확정 bad_gold_answer 를 밀어내지 않는다."""
        rec = _record(oracle_f1=0.1, qtype="bridge", faith_oracle=0.9, rel_oracle=0.9)
        self.assertIsNotNone(diagnose.bad_gold_answer_oracle(rec))
        self.assertEqual(diagnose._pick(rec, diagnose._GENERATION_CAUSE).label,
                         "bad_gold_answer")

    def test_measured_other_is_not_overridden_by_counts(self):
        """분류기가 실제로 돌아 '구체적 실패 없음'이라고 했으면, 약한 카운트 휴리스틱이
        그 판정을 뒤엎으면 안 된다 — 양보만 하고 라벨은 아무것도 안 나와 롤업으로 강등됐다."""
        self._with(ragas=_FakeReasoningJudge("other"))
        rec = _record(oracle_f1=0.1, qtype="bridge", faith_oracle=0.9, rel_oracle=0.9,
                      counts_oracle=(3, 2, 0))              # 카운트상으론 hop_binding 조건
        self.assertIsNone(diagnose.generation_reasoning_failure(rec))   # 'other' → 침묵
        self.assertIsNotNone(diagnose.bad_gold_answer_oracle(rec))      # 반증당하지 않는다
        self.assertEqual(diagnose._pick(rec, diagnose._GENERATION_CAUSE).label,
                         "bad_gold_answer")                 # 롤업으로 안 떨어진다

    def test_counts_fallback_still_wins_when_classifier_unmeasured(self):
        rec = _record(oracle_f1=0.1, qtype="bridge", faith_oracle=0.9, rel_oracle=0.9,
                      counts_oracle=(3, 2, 0))              # 분류기 자원 미주입
        self.assertIsNone(diagnose.bad_gold_answer_oracle(rec))
        self.assertEqual(diagnose._pick(rec, diagnose._GENERATION_CAUSE).label,
                         "generation_hop_binding_error")

    def test_classifier_memoized_once(self):
        judge = _FakeReasoningJudge("contradiction")
        self._with(ragas=judge)
        rec = _record(oracle_f1=0.1, faith_oracle=0.9, rel_oracle=0.9)
        for _ in range(3):
            diagnose._pick(rec, diagnose._GENERATION_CAUSE)
        self.assertEqual(judge.calls.count("reasoning_mode"), 1)

    def test_unknown_mode_is_treated_as_unmeasured(self):
        """분류기가 규정 외 값을 내면 미측정으로 떨어뜨린다(오라벨 방지)."""
        self._with(ragas=_FakeReasoningJudge("garbage_value"))
        rec = _record(oracle_f1=0.1, faith_oracle=0.9, rel_oracle=0.9)
        self.assertIsNone(diagnose.generation_reasoning_failure(rec))
        self.assertIsNotNone(diagnose.bad_gold_answer_oracle(rec))

    def test_hop_binding_needs_high_faithfulness(self):
        """근거가 약하면(faith<문턱) 결합 오류가 아니라 hallucination 영역."""
        rec = _record(oracle_f1=0.1, qtype="bridge",
                      faith_oracle=RAGAS_FAITHFULNESS_MIN - 0.01, counts_oracle=(3, 2, 0))
        self.assertIsNone(diagnose.generation_reasoning_failure(rec))

    def test_hop_binding_silent_for_single_hop(self):
        rec = _record(oracle_f1=0.1, qtype=None, faith_oracle=0.9, counts_oracle=(3, 2, 0))
        self.assertIsNone(diagnose.generation_reasoning_failure(rec))

    def test_hop_binding_confirmed_when_nothing_missing_but_unsupported_claim(self):
        """FN=0(요소 다 있음) + FP>0(근거 없는 주장) = 잘못 엮음 → 결합 오류."""
        rec = _record(oracle_f1=0.1, qtype="bridge", faith_oracle=0.9, counts_oracle=(3, 2, 0))
        finding = diagnose.generation_reasoning_failure(rec)
        self.assertTrue(finding.confirmed)
        self.assertIn("missing=0", finding.metadata["reason"])

    def test_hop_binding_yields_to_partial_answer_when_elements_missing(self):
        """FN>0 은 요소 누락 → 결합 오류가 아니라 부분 답변. 카운트로 배타(순서 비의존)."""
        rec = _record(oracle_f1=0.1, qtype="bridge", faith_oracle=0.9, counts_oracle=(2, 1, 2))
        self.assertIsNone(diagnose.generation_reasoning_failure(rec))
        picked = diagnose._pick(rec, diagnose._GENERATION_CAUSE)
        self.assertEqual(picked.label, "generation_partial_answer")   # 멀티홉도 이제 도달 가능
        self.assertTrue(picked.confirmed)

    def test_hop_binding_silent_when_no_unsupported_claim(self):
        """FN=0·FP=0 이면 잘못 엮었다는 근거가 없다 → 침묵."""
        rec = _record(oracle_f1=0.1, qtype="bridge", faith_oracle=0.9, counts_oracle=(3, 0, 0))
        self.assertIsNone(diagnose.generation_reasoning_failure(rec))

    def test_hop_binding_preliminary_when_counts_unmeasured(self):
        """카운트 없으면 faithfulness 만으론 '결합'을 특정 못 한다 → 예비."""
        rec = _record(oracle_f1=0.1, qtype="bridge", faith_oracle=1.0, rel_oracle=0.0)
        finding = diagnose.generation_reasoning_failure(rec)
        self.assertIsNotNone(finding)
        self.assertFalse(finding.confirmed)

    def test_partial_answer_confirmed_when_some_gold_elements_missing(self):
        """TP>0·FN>0 = 일부는 맞고 일부는 누락 → 정확히 '부분 답변'."""
        rec = _record(oracle_f1=0.1, counts_oracle=(2, 0, 2))
        finding = diagnose.generation_partial_answer(rec)
        self.assertTrue(finding.confirmed)
        self.assertIn("missing=2/4", finding.metadata["reason"])

    def test_partial_answer_silent_when_nothing_missing(self):
        rec = _record(oracle_f1=0.1, counts_oracle=(3, 1, 0))            # FN=0
        self.assertIsNone(diagnose.generation_partial_answer(rec))

    def test_partial_answer_silent_when_everything_missing(self):
        """TP=0 은 전부 누락 = '부분'이 아니다 → 다른 라벨/롤업 영역."""
        rec = _record(oracle_f1=0.1, counts_oracle=(0, 2, 3))
        self.assertIsNone(diagnose.generation_partial_answer(rec))

    def test_partial_answer_counts_win_over_relevancy(self):
        """카운트가 있으면 relevancy 는 보지 않는다 — 누락 없으면 rel 이 낮아도 침묵."""
        rec = _record(oracle_f1=0.1, counts_oracle=(3, 0, 0),
                      rel_oracle=RAGAS_RESPONSE_RELEVANCY_MIN - 0.01)
        self.assertIsNone(diagnose.generation_partial_answer(rec))

    def test_partial_answer_preliminary_when_counts_unmeasured(self):
        """degraded(카운트 없음) → relevancy 폴백이지만 확정은 못 한다."""
        rec = _record(oracle_f1=0.1, rel_oracle=RAGAS_RESPONSE_RELEVANCY_MIN - 0.01)
        finding = diagnose.generation_partial_answer(rec)
        self.assertIsNotNone(finding)
        self.assertFalse(finding.confirmed)

    def test_partial_answer_silent_exactly_at_threshold(self):
        rec = _record(oracle_f1=0.1, rel_oracle=RAGAS_RESPONSE_RELEVANCY_MIN)
        self.assertIsNone(diagnose.generation_partial_answer(rec))

    def test_parametric_overreliance_confirmed_when_correct_but_ungrounded(self):
        """정답이어도 검색 context 에 근거가 없으면 파라미터 기억으로 맞힌 것."""
        rec = _record(recall=1.0, f1=1.0, oracle_f1=1.0,
                      faith=RAGAS_FAITHFULNESS_MIN - 0.01)
        finding = diagnose.generation_parametric_overreliance(rec)
        self.assertIsNotNone(finding)
        self.assertTrue(finding.confirmed)

    def test_parametric_overreliance_silent_when_grounded(self):
        rec = _record(recall=1.0, f1=1.0, oracle_f1=1.0, faith=0.9)
        self.assertIsNone(diagnose.generation_parametric_overreliance(rec))

    def test_parametric_overreliance_silent_when_answer_wrong(self):
        """답이 틀렸으면 근거 없음은 환각 쪽 — '맞았지만 근거 없음'이 이 라벨의 전제다."""
        rec = _record(recall=1.0, f1=0.1, oracle_f1=1.0, faith=0.1)
        self.assertIsNone(diagnose.generation_parametric_overreliance(rec))

    def test_parametric_overreliance_silent_when_gold_not_retrieved(self):
        """미검색이면 근거를 못 쓴 게 당연 → A그룹 몫."""
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5, f1=1.0, oracle_f1=1.0, faith=0.1)
        self.assertIsNone(diagnose.generation_parametric_overreliance(rec))

    def test_parametric_overreliance_silent_below_deep(self):
        metrics_common.set_mode(Mode.STANDARD)                # faithfulness 미측정
        rec = _record(recall=1.0, f1=1.0, oracle_f1=1.0, faith=0.1)
        self.assertIsNone(diagnose.generation_parametric_overreliance(rec))
        self.assertIsNot(diagnose._is_success(rec), False)     # 기존 동작 유지(성공)

    def test_success_gate_fails_ungrounded_correct_answer(self):
        rec = _record(recall=1.0, f1=1.0, oracle_f1=1.0,
                      faith=RAGAS_FAITHFULNESS_MIN - 0.01)
        self.assertIs(diagnose._is_success(rec), False)
        self.assertFalse(diagnose._generation_failed(rec))     # 경쟁 원인 아님 → 슬롯은 안 연다

    def test_diagnose_emits_finding_for_ungrounded_correct_answer(self):
        """성공게이트만 바꾸고 발동 경로가 없으면 findings 가 비어 규약상 도로 성공이 된다.
        슬롯 밖 additive 로 붙어 실제 라벨이 나와야 한다.
        (diagnose 는 _compute_metrics 로 f1 을 답변 문자열에서 다시 계산하므로 문자열을 맞춘다.)"""
        rec = _record(answer="정답", oracle_answer="정답",
                      faith=RAGAS_FAITHFULNESS_MIN - 0.01)
        findings = diagnose.diagnose(rec, Mode.DEEP)
        self.assertEqual([f.label for f in findings], ["generation_parametric_overreliance"])

    def test_parametric_does_not_mask_oracle_generation_failure(self):
        """실제 답은 기억으로 맞히고 오라클 답은 깨진 probe — 두 원인이 다 남아야 한다.
        (_GENERATION_CAUSE 에 넣으면 _pick 이 하나만 뽑아 오라클 실패가 가려진다.)"""
        rec = _record(answer="정답", oracle_answer="전혀 다른 소리",
                      faith=RAGAS_FAITHFULNESS_MIN - 0.01,
                      faith_oracle=RAGAS_FAITHFULNESS_MIN - 0.01)
        labels = {f.label for f in diagnose.diagnose(rec, Mode.DEEP)}
        self.assertIn("generation_parametric_overreliance", labels)
        self.assertIn("generation_hallucination", labels)

    def test_generation_failed_premise_requires_oracle_miss(self):
        self.assertTrue(diagnose._generation_failed(_record(oracle_f1=0.1)))
        self.assertFalse(diagnose._generation_failed(_record(oracle_f1=1.0)))


# ══════════════════════════════════════════════════════════════════
#  C그룹: context 구조
#    faith 높음(노이즈에 근거) / faith 낮음(어디에도 근거 없음)으로 갈리고,
#    후자는 길이·gold 배치로 too_long_context / lost_in_the_middle 이 나뉜다.
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

    def _long_context(self, *, gold_at, n=5, faith=RAGAS_FAITHFULNESS_MIN - 0.01,
                      chars=CONTEXT_CHARS_MAX, precision=None, spans=None):
        """검색 결과 n건 중 gold_at 위치에 gold 를 둔 record. 총 context 길이는 chars."""
        ids = [f"x{i}" for i in range(n)]
        ids[gold_at] = "g_a"
        rec = _record(("g_a",), tuple(ids), recall=1.0, oracle_f1=1.0, f1=0.1,
                      gold_spans=spans)
        rec.retrieved_context = ["가" * (chars // n)] * n
        rec.ragas = {"faithfulness": faith}
        if precision is not None:
            rec.ragas["context_precision"] = precision
        rec.ragas_done = True
        return rec

    def test_lost_in_the_middle_confirmed_when_gold_buried(self):
        """gold 는 검색됐는데 답이 어디에도 근거하지 않고, gold 가 긴 context 한가운데 있다.

        확정이다 — 위치·길이·faithfulness 가 전부 실측이라(confirmed 는 '처방이 통한다'가 아니라
        '판별 신호가 측정됐다'는 뜻) 더 올라갈 tier 도 없다.
        """
        rec = self._long_context(gold_at=2)                    # 상대 위치 0.5
        finding = diagnose.lost_in_the_middle(rec)
        self.assertIsNotNone(finding)
        self.assertTrue(finding.confirmed)
        self.assertEqual(finding.metadata["group"], "C")
        self.assertIsNone(diagnose.too_long_context(rec))      # 위치로 배타

    def test_too_long_context_confirmed_when_gold_at_edge(self):
        """gold 가 맨 앞인데도 못 썼다 — 배치가 아니라 길이·과부하 쪽."""
        rec = self._long_context(gold_at=0)
        finding = diagnose.too_long_context(rec)
        self.assertIsNotNone(finding)
        self.assertTrue(finding.confirmed)
        self.assertIsNone(diagnose.lost_in_the_middle(rec))    # 배타

    def test_both_silent_when_context_is_short(self):
        """짧은 context 에서 근거를 못 쓴 건 길이·배치 문제가 아니라 생성측 이탈 → 롤업."""
        rec = self._long_context(gold_at=2, chars=1000)
        self.assertIsNone(diagnose.lost_in_the_middle(rec))
        self.assertIsNone(diagnose.too_long_context(rec))

    def test_both_silent_when_answer_is_grounded(self):
        """faith 높음 = 노이즈 청크에 근거함 → context_noise_interference 영역."""
        rec = self._long_context(gold_at=2, faith=0.9)
        self.assertIsNone(diagnose.lost_in_the_middle(rec))
        self.assertIsNone(diagnose.too_long_context(rec))
        self.assertIsNotNone(diagnose.context_noise_interference(rec))

    def test_both_silent_below_deep(self):
        """faithfulness 는 tier3 — DEEP 미만이면 갈릴 근거가 없다."""
        metrics_common.set_mode(Mode.STANDARD)
        rec = self._long_context(gold_at=2)
        self.assertIsNone(diagnose.lost_in_the_middle(rec))
        self.assertIsNone(diagnose.too_long_context(rec))

    def test_both_silent_when_gold_position_is_unmeasurable(self):
        """gold id 드리프트(재청킹 후 recall 은 span 기준으로 1인데 chunk-id 는 안 맞음) —
        위치를 못 재면 too_long_context 가 전부 흡수해선 안 된다(처방이 재배치와 갈린다)."""
        rec = self._long_context(gold_at=0)
        rec.retrieved_chunk_ids = [f"drifted{i}" for i in range(5)]   # gold id 가 하나도 안 맞음
        self.assertIsNone(diagnose._gold_in_middle_band(rec))
        self.assertIsNone(diagnose.too_long_context(rec))
        self.assertIsNone(diagnose.lost_in_the_middle(rec))

    def test_both_silent_when_results_too_few_to_have_a_middle(self):
        """결과가 3건 미만이면 앞·중간·뒤가 안 갈린다 — 같은 이유로 둘 다 침묵."""
        rec = self._long_context(gold_at=0, n=2)
        self.assertIsNone(diagnose._gold_in_middle_band(rec))
        self.assertIsNone(diagnose.too_long_context(rec))
        self.assertIsNone(diagnose.lost_in_the_middle(rec))

    def test_yield_to_underchunking_when_noise_is_inside_chunk(self):
        """노이즈가 청크 '안'이면 길이·배치가 아니라 청크 크기 문제다(C그룹 3자 배타)."""
        rec = self._long_context(gold_at=2, precision=0.1,
                                 spans=[{"doc_id": "d1", "start": 0, "end": 10}])
        self.assertIsNone(diagnose.lost_in_the_middle(rec))
        self.assertEqual(diagnose._pick(rec, diagnose._CONTEXT_CAUSE).label,
                         "chunking_underchunking")

    def test_noise_interference_confirmed_when_answer_grounded_but_wrong(self):
        """faithfulness 는 retrieved_context(gold+노이즈) 기준 — 근거는 있는데 답이 틀리면
        gold 아닌 청크에 근거했다는 뜻이다."""
        rec = _record(recall=1.0, oracle_f1=1.0, f1=0.1, faith=0.9, rel=0.9)
        finding = diagnose.context_noise_interference(rec)
        self.assertIsNotNone(finding)
        self.assertTrue(finding.confirmed)
        self.assertEqual(finding.metadata["group"], "C")

    def test_noise_interference_silent_when_answer_ungrounded(self):
        """faithfulness 낮음 = gold·노이즈 어디에도 근거 없음 → 노이즈에 이끌린 게 아니다."""
        rec = _record(recall=1.0, oracle_f1=1.0, f1=0.1,
                      faith=RAGAS_FAITHFULNESS_MIN - 0.01, rel=0.9)
        self.assertIsNone(diagnose.context_noise_interference(rec))

    def test_noise_interference_silent_without_ragas(self):
        metrics_common.set_mode(Mode.STANDARD)                 # DEEP 미만 → faith None
        rec = _record(recall=1.0, oracle_f1=1.0, f1=0.1, faith=0.9, rel=0.9)
        self.assertIsNone(diagnose.context_noise_interference(rec))

    def test_noise_interference_silent_when_context_premise_unmet(self):
        rec = _record(recall=0.5, oracle_f1=1.0, f1=0.1, faith=0.9)   # 검색 실패는 A그룹
        self.assertIsNone(diagnose.context_noise_interference(rec))

    def test_noise_interference_wins_context_slot_over_bad_gold_answer(self):
        """oracle 통과가 '정답셋 오류'를 반증하므로 bad_gold_answer 는 양보한다."""
        rec = _record(recall=1.0, oracle_f1=1.0, f1=0.1, faith=0.9, rel=0.9)
        self.assertIsNone(diagnose.bad_gold_answer(rec))
        picked = diagnose._pick(rec, diagnose._CONTEXT_CAUSE)
        self.assertEqual(picked.label, "context_noise_interference")

    def _chunky(self, *, span_len, chunk_len=100, precision=0.1, reranked=False):
        """gold 청크 하나 안에 span_len 만큼만 근거가 있는 record. 밀도 = span_len/chunk_len."""
        chunks = [Chunk("g_a", "d1", "본문", char_span=(0, chunk_len))]
        metrics_common.set_context(chunks=chunks)
        rec = _record(("g_a",), ("g_a",), recall=1.0, oracle_f1=1.0, f1=0.1,
                      gold_spans=[{"doc_id": "d1", "start": 0, "end": span_len}])
        rec.ragas = {"context_precision": precision, "faithfulness": 0.9}
        rec.ragas_done = True
        rec.retrieval_details = {"reranked": reranked}
        return rec

    def test_underchunking_confirmed_when_evidence_buried_in_big_chunk(self):
        """근거가 청크의 10% 뿐 + precision 낮음 → 청크가 근거보다 큼."""
        rec = self._chunky(span_len=10)
        finding = diagnose.chunking_underchunking(rec)
        self.assertIsNotNone(finding)
        self.assertTrue(finding.confirmed)
        self.assertEqual(finding.metadata["group"], "A")    # 청킹은 전부 A그룹

    def test_underchunking_silent_when_evidence_dense(self):
        rec = self._chunky(span_len=80)                     # 밀도 0.8
        self.assertIsNone(diagnose.chunking_underchunking(rec))

    def test_underchunking_silent_when_precision_ok(self):
        rec = self._chunky(span_len=10, precision=0.9)
        self.assertIsNone(diagnose.chunking_underchunking(rec))

    def test_underchunking_beats_noise_interference(self):
        """노이즈가 청크 '안'이면 청크 '사이' 라벨은 양보한다."""
        rec = self._chunky(span_len=10)
        self.assertIsNone(diagnose.context_noise_interference(rec))
        picked = diagnose._pick(rec, diagnose._CONTEXT_CAUSE)
        self.assertEqual(picked.label, "chunking_underchunking")

    def test_reranker_low_precision_preliminary_when_reranked_and_imprecise(self):
        rec = self._chunky(span_len=80, reranked=True)      # 밀도는 높음(청크 안 문제 아님)
        finding = diagnose.reranker_low_precision(rec)
        self.assertIsNotNone(finding)
        self.assertFalse(finding.confirmed)                 # 전/후 순위 비교 불가 → C그룹 유일 예비
        self.assertEqual(finding.metadata["group"], "A")

    def test_confirmed_length_cause_beats_preliminary_reranker(self):
        """리랭커와 길이 원인이 함께 성립하면 확정 쪽을 채택한다 — 순서가 아니라 confirmed 로 갈린다.

        예전엔 셋 다 예비라 튜플 순서(리랭커 우선)가 결정했다. 이제 리랭커만 예비로 남아
        (인과 미측정) _pick 이 확정된 길이 원인을 먼저 뽑는다.
        """
        rec = self._long_context(gold_at=0, precision=0.1)   # gold 양끝 + precision 낮음
        rec.retrieval_details = {"reranked": True}
        self.assertTrue(diagnose.too_long_context(rec).confirmed)
        self.assertFalse(diagnose.reranker_low_precision(rec).confirmed)
        self.assertEqual(diagnose._pick(rec, diagnose._CONTEXT_CAUSE).label, "too_long_context")

    def test_reranker_owns_slot_when_no_confirmed_cause(self):
        """길이·배치가 성립하지 않으면(짧은 context) 예비 리랭커가 슬롯을 가져간다."""
        rec = self._long_context(gold_at=0, precision=0.1, chars=1000)
        rec.retrieval_details = {"reranked": True}
        self.assertIsNone(diagnose.too_long_context(rec))
        self.assertEqual(diagnose._pick(rec, diagnose._CONTEXT_CAUSE).label,
                         "reranker_low_precision")

    def test_reranker_silent_when_not_reranked(self):
        rec = self._chunky(span_len=80, reranked=False)
        self.assertIsNone(diagnose.reranker_low_precision(rec))

    def test_reranker_yields_to_underchunking_when_noise_inside_chunk(self):
        rec = self._chunky(span_len=10, reranked=True)      # 밀도 낮음 → 청크 안 문제
        self.assertIsNone(diagnose.reranker_low_precision(rec))
        picked = diagnose._pick(rec, diagnose._CONTEXT_CAUSE)
        self.assertEqual(picked.label, "chunking_underchunking")

    def test_new_labels_silent_below_deep(self):
        """context_precision 은 tier3 — DEEP 미만에선 둘 다 침묵."""
        metrics_common.set_mode(Mode.STANDARD)
        rec = self._chunky(span_len=10, reranked=True)
        self.assertIsNone(diagnose.chunking_underchunking(rec))
        self.assertIsNone(diagnose.reranker_low_precision(rec))

    def test_bad_gold_answer_confirmed_when_oracle_also_fails(self):
        """oracle 이 못 맞힌 경우에만 '정답셋 오류' 주장이 성립한다."""
        rec = _record(recall=1.0, oracle_f1=0.1, f1=0.1, faith=0.8, rel=0.9)
        finding = diagnose.bad_gold_answer(rec)
        self.assertIsNotNone(finding)
        self.assertEqual(finding.type, "gap")                  # D그룹으로 분류
        self.assertEqual(finding.metadata["group"], "D")

    def test_bad_gold_answer_silent_when_only_one_metric_high(self):
        rec = _record(recall=1.0, oracle_f1=0.1, f1=0.1, faith=0.9, rel=0.1)
        self.assertIsNone(diagnose.bad_gold_answer(rec))

    def test_bad_gold_answer_silent_without_ragas(self):
        metrics_common.set_mode(Mode.STANDARD)
        rec = _record(recall=1.0, oracle_f1=0.1, f1=0.1, faith=0.9, rel=0.9)
        self.assertIsNone(diagnose.bad_gold_answer(rec))

    def test_low_f1_numeric_grounded_answer_still_requires_answer_correctness(self):
        rec = _record(
            recall=1.0,
            oracle_f1=0.2,
            f1=0.29,
            ground_truth="asset_total 49,157,964,024 and liability_total 32,695,236,480",
            answer="The report says asset_total is 49,157,964,024 and liability_total is 32,695,236,480.",
            faith=1.0,
            rel=0.97,
        )
        self.assertFalse(diagnose._f1_ok(rec))
        self.assertIsNotNone(diagnose.bad_gold_answer(rec))

    def test_low_f1_numeric_answer_still_bad_gold_when_not_grounded_enough(self):
        rec = _record(
            recall=1.0,
            oracle_f1=0.2,
            f1=0.29,
            ground_truth="asset_total 49,157,964,024 and liability_total 32,695,236,480",
            answer="The report says asset_total is 49,157,964,024 and liability_total is 32,695,236,480.",
            faith=0.8,
            rel=0.97,
        )
        self.assertFalse(diagnose._f1_ok(rec))
        self.assertIsNotNone(diagnose.bad_gold_answer(rec))


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

    def _gap(self, *, answer="지어낸 답", f1=0.1):
        """gold 가 코퍼스에 없는 probe(= corpus_gap 전제)."""
        return _record(("unknown",), ("x",), recall=0.0, f1=f1, answer=answer)

    def test_abstention_failure_on_gap_confirmed_when_model_makes_something_up(self):
        """corpus_gap 은 '자료를 채워라'까지만 말한다 — 기권했어야 하는데 지어낸 건 별개 문제."""
        finding = diagnose.generation_abstention_failure(self._gap())
        self.assertIsNotNone(finding)
        self.assertTrue(finding.confirmed)
        self.assertEqual(finding.severity, "critical")
        self.assertEqual(finding.metadata["group"], "B")

    def test_abstention_failure_on_gap_silent_when_model_abstains(self):
        """올바른 동작 — 근거가 없다고 말했으면 생성 실패가 아니다."""
        self.assertIsNone(diagnose.generation_abstention_failure(
            self._gap(answer="제공된 정보로는 알 수 없습니다")))

    def test_abstention_failure_on_gap_silent_when_answer_is_correct(self):
        """근거 없이 맞힌 건 parametric_overreliance 영역."""
        self.assertIsNone(diagnose.generation_abstention_failure(self._gap(f1=1.0)))

    def test_abstention_failure_on_gap_silent_when_answer_is_empty(self):
        """빈 답변은 지어낸 게 아니라 생성 실패 → 롤업 몫."""
        self.assertIsNone(diagnose.generation_abstention_failure(self._gap(answer="")))

    def test_abstention_failure_silent_on_partial_gap(self):
        """gold 하나라도 코퍼스에 있으면 근거 있는 부분 답변일 수 있다 — 환각으로 몰면 안 된다."""
        rec = _record(("g_a", "unknown"), ("g_a",), recall=0.5, f1=0.1, qtype="bridge")
        self.assertIsNone(diagnose.generation_abstention_failure(rec))

    def test_abstention_failure_on_gap_silent_when_gold_is_in_corpus(self):
        rec = _record(("g_a",), ("x",), recall=0.0, f1=0.1)
        self.assertIsNone(diagnose.generation_abstention_failure(rec))

    def test_abstention_judge_not_called_when_abstention_is_not_expected(self):
        """근거가 있는 probe 까지 AspectCritic 을 부르면 실패 probe 마다 LLM 1회가 는다."""
        metrics_common.set_mode(Mode.DEEP)
        judge = _FakeAbstentionJudge(False)
        self._with(ragas=judge)
        rec = _record(("g_a",), ("x",), recall=0.0, f1=0.1)      # gold 는 코퍼스에 있음
        self.assertIsNone(diagnose.generation_abstention_failure(rec))
        self.assertNotIn("abstention", judge.calls)

    def test_abstention_failure_on_gap_silent_without_tier2(self):
        """코퍼스 멤버십은 tier2 — 미측정이면 gap 전제를 못 세운다."""
        metrics_common.set_mode(Mode.FAST)
        self.assertIsNone(diagnose.generation_abstention_failure(self._gap()))

    def test_gap_emits_both_corpus_gap_and_abstention_label(self):
        """자료 보강(D)과 기권 동작(B)은 처방이 달라 둘 다 남아야 한다.
        (B 슬롯은 오라클 답이 없어 안 열리므로 additive 경로로만 도달한다.)"""
        labels = {f.label for f in diagnose.diagnose(self._gap(), Mode.STANDARD)}
        self.assertIn("corpus_gap", labels)
        self.assertIn("generation_abstention_failure", labels)

    def test_mixed_corpus_emits_both_retrieval_and_gap_labels(self):
        """혼합 코퍼스 — 코퍼스에 있는 몫은 A 라벨, 없는 몫은 D 라벨로 함께 남는다.

        기권 라벨은 붙으면 안 된다: 근거 일부는 실제로 코퍼스에 있으니 '기권했어야 했다'가
        같은 record 의 A 라벨(검색을 고쳐라)과 정면으로 모순되는 처방이 된다.
        """
        # 답이 틀려야 기권 분기까지 실제로 도달한다(_compute_metrics 가 f1 을 다시 계산하므로
        # 여기서 f1 을 주입하지 않고 답변 자체를 틀리게 둔다). 이게 없으면 assertNotIn 이 공허하다.
        rec = _record(("g_a", "unknown"), ("x",), recall=0.0, answer="엉뚱한 말")
        labels = {f.label for f in diagnose.diagnose(rec, Mode.STANDARD)}
        self.assertFalse(diagnose._f1_ok(rec))                 # 기권 분기 도달 전제
        self.assertIn("retrieval_missing_gold", labels)
        self.assertIn("corpus_gap", labels)
        self.assertNotIn("generation_abstention_failure", labels)

    def test_missing_gold_ids_lists_only_absent_gold(self):
        """비율(n/N)만으론 '코퍼스 어디가 빈지'를 못 적는다 — optimize 는 멤버십을 스스로 못 구한다."""
        rec = _record(("g_a", "unknown"), ("x",), recall=0.0, qtype="bridge")
        finding = diagnose.corpus_gap_partial_hop(rec)
        self.assertEqual(finding.metadata["missing_gold_ids"], ["unknown"])

    def test_missing_gold_ids_covers_every_gold_on_full_gap(self):
        rec = _record(("unknown", "unknown2"), ("x",), recall=0.0)
        self.assertEqual(set(diagnose.corpus_gap(rec).metadata["missing_gold_ids"]),
                         {"unknown", "unknown2"})

    def test_missing_gold_ids_empty_when_membership_unmeasured(self):
        """멤버십은 tier2 — 미측정이면 라벨 자체가 안 서므로 키를 소비할 일도 없다."""
        metrics_common.set_mode(Mode.FAST)
        self.assertIsNone(diagnose.corpus_gap(_record(("unknown",), ("x",), recall=0.0)))

    def test_critical_findings_do_not_move_reliability(self):
        """critical 이 additive 로 늘어도 점수축은 안 움직인다 — findings 는 gold 대조 probe 의
        신뢰도 식(recall × 답변축)에 들어가지 않는다(무응답 기대 probe 만 findings 유무로 1/0).

        recall=0.5 · 정답 일치로 잡아 값이 0 이 아닌 지점에서 본다 — gap probe(recall=0)로
        비교하면 0 == 0 이라 findings 무관함을 증명하지 못한다.
        """
        from agents.eval import scoring
        # gold 를 둘 다 코퍼스에 둬야 missing_gold(critical)가 확정으로 선다 — 코퍼스 밖이면
        # 그 라벨이 self-scope 로 빠지고 예비 enumeration 만 남아 critical 이 사라진다.
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5, answer="정답")
        rec.findings = diagnose.diagnose(rec, Mode.STANDARD)
        self.assertTrue(any(f.severity == "critical" for f in rec.findings))
        self.assertEqual(scoring._probe_reliability(rec), 0.5)   # 0.5(recall) × 1.0(정답)

    def test_abstention_label_is_additive_not_a_score_penalty(self):
        """기권 실패가 붙는 corpus_gap probe 의 신뢰도 0 은 recall=0 탓이다 — 라벨 탓이 아니다."""
        from agents.eval import scoring
        rec = self._gap()
        rec.findings = diagnose.diagnose(rec, Mode.STANDARD)
        self.assertIn("generation_abstention_failure", {f.label for f in rec.findings})
        self.assertEqual(rec.recall_at_k, 0.0)
        self.assertEqual(scoring._probe_reliability(rec), 0.0)

    def test_corpus_gap_reason_shows_membership_ratio(self):
        """'gold_in_corpus=False' 만 적으면 부분 gap 을 '전부 없음'으로 오독한다."""
        rec = _record(("g_a", "unknown"), ("x",), recall=0.0)
        self.assertIn("gold_in_corpus=1/2", diagnose.corpus_gap(rec).metadata["reason"])

    def test_both_silent_when_gold_present_in_corpus(self):
        rec = _record(("g_a",), ("x",), recall=0.0)
        self.assertIsNone(diagnose.corpus_gap(rec))
        self.assertIsNone(diagnose.corpus_gap_partial_hop(rec))

    def test_both_silent_for_no_answer_probe(self):
        """답이 애초에 없는 질문엔 채울 자료도 없다 — '문서를 추가 수집하라'가 거짓 처방이 된다."""
        for qtype in (None, "bridge"):
            with self.subTest(qtype=qtype):
                rec = _record(("unknown",), ("x",), recall=0.0, qtype=qtype,
                              answer_exists=False, ground_truth=None)
                self.assertIsNone(diagnose.corpus_gap(rec))
                self.assertIsNone(diagnose.corpus_gap_partial_hop(rec))

    def test_trigger_metadata_separates_the_two_causes(self):
        """처방은 같지만 '자료를 채우면 사라질 기권 실패'인지는 갈라야 한다 — reason 파싱 없이."""
        gap = diagnose.generation_abstention_failure(self._gap())
        self.assertEqual(gap.metadata["trigger"], "corpus_gap")

        no_answer = _record((), ("x",), recall=-1.0, answer_exists=False,
                            ground_truth=None, answer="지어낸 답")
        self.assertEqual(diagnose.generation_abstention_failure(no_answer).metadata["trigger"],
                         "no_answer_expected")

    def test_no_answer_probe_emits_only_the_abstention_label(self):
        """무응답 기대 probe 는 D(자료)·A(검색) 어디에도 걸리지 않는다 — 고칠 건 생성 동작뿐."""
        rec = _record(("g_a",), ("x",), recall=0.0, answer_exists=False,
                      ground_truth=None, answer="지어낸 답")
        self.assertEqual({f.label for f in diagnose.diagnose(rec, Mode.STANDARD)},
                         {"generation_abstention_failure"})


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
        """순서가 앞서도 예비는 밀린다 — 뒤에 확정이 있으면 그쪽을 채택한다.
        (missing_gold 는 bridge 에 양보하게 바뀌어, bridge 를 반증하는 lexical 로 검증한다.)"""
        self._with(keyword=["g_b"])                              # BM25 가 놓친 gold 를 잡음
        rec = _record(("g_a", "g_b"), ("g_a",), recall=0.5, qtype="bridge")
        # 앞자리가 실제로 '예비'를 내는지부터 못 박는다(None 이면 우선순위 검증이 무의미).
        earlier = diagnose.retrieval_missing_bridge_dependency(rec)
        self.assertIsNotNone(earlier)
        self.assertFalse(earlier.confirmed)

        picked = diagnose._pick(rec, (
            diagnose.retrieval_missing_bridge_dependency,   # 예비 (앞)
            diagnose.retrieval_lexical_mismatch,            # 확정 (뒤)
            diagnose.retrieval_failure,                     # 예비 롤업
        ))
        self.assertEqual(picked.label, "retrieval_lexical_mismatch")
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

    def test_full_mode_folds_into_deep(self):
        """tier4 는 없앴다 — 다만 'full' 은 웹 UI depth 문자열이라 DEEP 으로 접어야 한다.
        지우면 EVAL_MODE=full 이 DEFAULT_MODE(fast)로 조용히 강등된다."""
        from agents.eval import types
        self.assertFalse(hasattr(Mode, "FULL"))
        for raw in ("full", "4"):
            with self.subTest(raw=raw):
                os.environ["EVAL_MODE"] = raw
                try:
                    self.assertEqual(types.resolve_mode(), Mode.DEEP)
                finally:
                    os.environ.pop("EVAL_MODE", None)

    def test_group_derivation_matches_prescription_order(self):
        self.assertEqual(diagnose._group_of("corpus_gap", "gap"), "D")
        self.assertEqual(diagnose._group_of("retrieval_low_rank", "retrieval_failure"), "A")
        self.assertEqual(diagnose._group_of("chunking_context_mismatch", "retrieval_failure"), "A")
        self.assertEqual(diagnose._group_of("generation_hallucination", "generation_failure"), "B")
        self.assertEqual(diagnose._group_of("context_failure", "context_failure"), "C")


if __name__ == "__main__":
    unittest.main(verbosity=2)
