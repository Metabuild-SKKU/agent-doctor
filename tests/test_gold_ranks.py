"""
tests/test_gold_ranks.py
gold 순위(gold_rank) 노출 검증 — 2단계.

목적: top_k 근거값을 '개수'가 아니라 '순위'로 산정하기 위해, Eval 이 wide 재검색에서
gold 청크의 순위를 재어 Finding.metadata 로 넘긴다. 그 배선을 세 층에서 검증한다.
  1. signals._gold_ranks    : wide 재검색 결과에서 순위(1-based)를 뽑는다.
  2. signals._wide_hits      : low_rank·순위 계산이 재검색 1회를 memoize 로 공유한다.
  3. diagnose._finding       : 대상 라벨에만 gold_ranks 를 싣는다(모드 게이팅 존중).

qdrant 등 무거운 자원 없이, set_context 로 가짜 retrieve_fn 을 주입해 tier2 를 흉내낸다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import Probe
from agents.eval import signals, diagnose
from agents.eval.types import EvalRecord, Mode


def _record(gold_ids, retrieved_ids, question="질문", qtype=None):
    probe = Probe(
        probe_id="p1", question=question, source="taxonomy",
        gold_chunk_ids=list(gold_ids), qtype=qtype,
    )
    return EvalRecord(probe=probe, retrieved_chunk_ids=list(retrieved_ids))


class _FakeRetriever:
    """넓은 순위표를 순위대로 돌려주는 가짜 retrieve_fn. 호출 횟수를 센다."""

    def __init__(self, ranked_ids):
        self.ranked_ids = ranked_ids   # 1위부터 순서대로
        self.calls = 0

    def __call__(self, client, chunks, question, top_n):
        self.calls += 1
        return [{"chunk_id": cid} for cid in self.ranked_ids[:top_n]]


class GoldRanksTest(unittest.TestCase):
    def setUp(self):
        signals.set_mode(Mode.STANDARD)

    def tearDown(self):
        signals.set_context()  # 주입 자원 초기화
        signals.set_mode(Mode.FAST)

    def test_ranks_are_one_based_positions_in_wide_search(self):
        # gold g_c 는 top-k(2개)가 놓쳤고 wide 에서 13위. 개수(3)가 아니라 순위를 재야 한다.
        retriever = _FakeRetriever(
            ["g_a", "x", "g_b"] + [f"n{i}" for i in range(9)] + ["g_c"]
        )  # g_a=1, g_b=3, g_c=13
        signals.set_context(retrieve_fn=retriever, chunks=[])
        record = _record(["g_a", "g_b", "g_c"], ["g_a", "g_b"])

        ranks = signals._gold_ranks(record)
        self.assertEqual(ranks, {"g_a": 1, "g_b": 3, "g_c": 13})

    def test_gold_beyond_wide_n_is_none(self):
        # wide 결과(3개) 밖의 gold 는 None → top_k 로 도달 불가 신호.
        retriever = _FakeRetriever(["g_a", "x", "y"])
        signals.set_context(retrieve_fn=retriever, chunks=[])
        record = _record(["g_a", "g_far"], ["g_a"])

        ranks = signals._gold_ranks(record)
        self.assertEqual(ranks, {"g_a": 1, "g_far": None})

    def test_fast_mode_yields_no_ranks(self):
        # tier2 미달(FAST) → 재검색 없이 None (planner 는 개수 폴백).
        retriever = _FakeRetriever(["g_a"])
        signals.set_context(retrieve_fn=retriever, chunks=[])
        signals.set_mode(Mode.FAST)
        record = _record(["g_a"], [])

        self.assertIsNone(signals._gold_ranks(record))
        self.assertEqual(retriever.calls, 0)

    def test_wide_search_shared_between_signals(self):
        # low_rank(존재 여부)와 순위 계산이 재검색 1회를 공유해야 한다(memoize).
        retriever = _FakeRetriever(["g_a", "x", "g_b"])
        signals.set_context(retrieve_fn=retriever, chunks=[])
        record = _record(["g_a", "g_b"], ["g_a"])  # g_b 놓침

        self.assertTrue(signals._gold_in_wider_candidates(record))
        signals._gold_ranks(record)
        self.assertEqual(retriever.calls, 1)  # 두 신호가 wide 검색을 한 번만 돌림


class FindingMetadataTest(unittest.TestCase):
    """diagnose._finding 이 대상 라벨에만 순위를 싣는지."""

    def setUp(self):
        signals.set_mode(Mode.STANDARD)
        retriever = _FakeRetriever(["g_a", "x", "g_b"])  # g_a=1, g_b=3
        signals.set_context(retrieve_fn=retriever, chunks=[])
        self.record = _record(["g_a", "g_b"], ["g_a"])

    def tearDown(self):
        signals.set_context()
        signals.set_mode(Mode.FAST)

    def test_rank_label_carries_gold_ranks(self):
        f = diagnose._finding(
            self.record, "retrieval_incomplete_enumeration",
            "retrieval_failure", confirmed=True,
        )
        self.assertEqual(f.metadata["gold_ranks"], {"g_a": 1, "g_b": 3})

    def test_non_rank_label_omits_gold_ranks(self):
        f = diagnose._finding(
            self.record, "context_noise_interference",
            "retrieval_failure", confirmed=True,
        )
        self.assertNotIn("gold_ranks", f.metadata)


if __name__ == "__main__":
    unittest.main()
