"""
tests/test_topic_cluster.py
topic_cluster.classify 단위 테스트 — 실패 gold 응집도의 baseline 대비 상대 판정.

concentrated(특정 도메인 약함) / spread(모델 자체 약함) / none(판정 불가·청크 희석)을
mock 임베딩으로 검증한다. 절대 코사인이 아니라 baseline 대비 비율이라, 코퍼스가 전반적으로
뭉쳐 있어도(같은 도메인) 상대적으로 더 뭉친 실패만 concentrated 로 잡혀야 한다.
"""
import math
import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.eval import topic_cluster as tc


class TopicClusterClassifyTest(unittest.TestCase):
    def test_concentrated_when_failures_cluster_tighter_than_corpus(self):
        # 실패 gold 는 거의 같은 방향(뭉침), 코퍼스는 축마다 흩어짐 → concentrated
        failed = [[1, 0, 0], [0.98, 0.1, 0], [0.97, 0.15, 0.05]]
        corpus = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]]
        self.assertEqual(tc.classify(failed, corpus), tc.CONCENTRATED)

    def test_spread_when_failures_as_scattered_as_corpus(self):
        # 실패 gold 가 코퍼스만큼 흩어짐(직교) → spread (모델 자체 약함)
        failed = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        corpus = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]]
        self.assertEqual(tc.classify(failed, corpus), tc.SPREAD)

    def test_none_when_single_failed_gold(self):
        # 쌍이 없으면 응집도 계산 불가 → none
        self.assertEqual(tc.classify([[1, 0, 0]], [[1, 0, 0], [0, 1, 0]]), tc.NONE)

    def test_none_when_failed_embeddings_missing(self):
        # 실패 gold 임베딩이 전부 비었으면(fallback/미부착) → none
        self.assertEqual(tc.classify([[], []], [[1, 0, 0], [0, 1, 0]]), tc.NONE)

    def test_none_when_corpus_baseline_unmeasurable(self):
        # baseline 을 못 재면(코퍼스 쌍 부족) → none
        self.assertEqual(tc.classify([[1, 0, 0], [0, 1, 0]], [[1, 0, 0]]), tc.NONE)

    def test_absolute_high_cosine_corpus_still_relative(self):
        # 코퍼스가 전반적으로 뭉쳐 있어도(같은 도메인, 절대 cos 높음) 실패가 코퍼스만큼만
        # 뭉쳤으면 concentrated 가 아니어야 한다 — 절대값이 아니라 baseline 대비 비율이므로.
        base = [1.0, 0.0]
        near = [math.cos(0.2), math.sin(0.2)]   # 서로 가까운 방향들
        corpus = [base, near, [math.cos(0.15), math.sin(0.15)], [math.cos(0.25), math.sin(0.25)]]
        failed = [base, near]                    # 코퍼스와 같은 수준의 응집
        self.assertNotEqual(tc.classify(failed, corpus), tc.CONCENTRATED)


if __name__ == "__main__":
    unittest.main()
