"""
tests/test_topic_cluster.py
topic_cluster.classify 단위 테스트 — 실패 gold 응집도의 baseline 대비 상대 판정.

concentrated(특정 도메인 약함) / spread(모델 자체 약함) / none(쟀으나 응집 없음 → 청크
희석) / unmeasured(못 잼 → fallback)를 mock 임베딩으로 검증한다. 절대 코사인이 아니라
baseline 대비 비율이라, 코퍼스가 전반적으로 뭉쳐 있어도(같은 도메인) 상대적으로 더 뭉친
실패만 concentrated 로 잡혀야 한다.
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

    def test_unmeasured_when_single_failed_gold(self):
        # 쌍이 없으면 응집도 계산 불가 → unmeasured(none 아님: 근거 없이 청킹 처방 확정 금지)
        self.assertEqual(tc.classify([[1, 0, 0]], [[1, 0, 0], [0, 1, 0]]), tc.UNMEASURED)

    def test_unmeasured_when_failed_embeddings_missing(self):
        # 실패 gold 임베딩이 전부 비었으면(fallback/미부착) → unmeasured
        self.assertEqual(tc.classify([[], []], [[1, 0, 0], [0, 1, 0]]), tc.UNMEASURED)

    def test_unmeasured_when_corpus_baseline_unmeasurable(self):
        # baseline 을 못 재면(코퍼스 쌍 부족) → unmeasured
        self.assertEqual(tc.classify([[1, 0, 0], [0, 1, 0]], [[1, 0, 0]]), tc.UNMEASURED)

    def test_unmeasured_when_baseline_non_positive(self):
        # baseline 이 음수면(코퍼스가 서로 등질) 나눌 때 부호가 뒤집혀 무조건 spread 가 된다.
        # 실패는 완전히 뭉쳐 있으므로(cos=1) spread 로 새면 명백한 오판 → unmeasured 로 막는다.
        corpus = [[1, 0], [-1, 0]]              # 평균 쌍별 코사인 = -1
        failed = [[1, 0], [1, 0]]               # 완전히 뭉침
        self.assertEqual(tc.classify(failed, corpus), tc.UNMEASURED)

    def test_none_when_ratio_between_thresholds(self):
        # 쟀는데 코퍼스와 비슷한 수준(ratio ≈ 1.2) → none(임베딩 문제 아님 → 청킹 조정).
        # 판정 불가(unmeasured)와 구분돼야 한다.
        base = [1.0, 0.0]
        corpus = [base, [math.cos(0.9), math.sin(0.9)], [math.cos(1.8), math.sin(1.8)]]
        failed = [base, [math.cos(1.15), math.sin(1.15)]]     # ratio ≈ 1.21
        self.assertEqual(tc.classify(failed, corpus), tc.NONE)

    def test_baseline_sample_is_unbiased_by_document_order(self):
        """head-N 표본 편향 회귀 — chunk 순서는 문서 순서라 앞을 자르면 단일 주제가 잡힌다.

        앞쪽 문서만 응집돼 있고 뒤는 흩어진 코퍼스에서, 실패가 그 앞쪽 주제에 몰린 상황.
        head-100 을 baseline 으로 쓰면 baseline 이 부풀려져 concentrated 를 놓친다.
        """
        # 앞 100청크 = 한 방향(응집), 나머지 200청크 = 여러 방향(흩어짐)
        tight = [[math.cos(a * 0.0005), math.sin(a * 0.0005)] for a in range(100)]
        scattered = [[math.cos(a * 0.03), math.sin(a * 0.03)] for a in range(200)]
        corpus = tight + scattered
        failed = tight[:10]                      # 실패가 응집 주제에 몰림

        self.assertEqual(tc.classify(failed, corpus), tc.CONCENTRATED)

        # stride 표본이 head 표본보다 낮은(=편향 없는) baseline 을 내야 한다
        head_baseline = tc._mean_pairwise_cosine(corpus[:100])
        stride_baseline = tc._mean_pairwise_cosine(tc.stride_sample(corpus))
        self.assertLess(stride_baseline, head_baseline)

    def test_sample_covers_full_range_at_every_corpus_size(self):
        """정수 stride 붕괴 회귀 — n < 2*limit 이면 몫이 1 로 뭉개져 head 표본이 된다.

        n=150·limit=100 에서 `v[::150//100]` = `v[::1][:100]` = 앞 100개다. 그 구간이
        곧 head 편향이 되살아나는 지점이라, 크기별로 전 구간을 훑는지 직접 고정한다.
        """
        for n in (101, 150, 199, 200, 299, 300, 1000):
            sample = tc.stride_sample(list(range(n)), limit=100)
            self.assertEqual(len(sample), 100, f"n={n}")
            self.assertEqual(len(set(sample)), 100, f"n={n} 중복 표본")
            # 마지막 표본이 코퍼스 끝 근처에서 나와야 한다(앞쪽에 갇히면 편향).
            self.assertGreater((sample[-1] + 1) / n, 0.98, f"n={n} 앞쪽 편향")

    def test_concentrated_survives_stride_collapse_corpus_size(self):
        # 위 붕괴가 실제 판정을 뒤집는지 — 앞 100청크가 한 문서(응집), 뒤 50이 흩어짐.
        # 정수 stride 였을 때 baseline 0.9998(실제 0.527)로 부풀어 spread 로 오판했다.
        tight = [[math.cos(a * 0.0005), math.sin(a * 0.0005)] for a in range(100)]
        scattered = [[math.cos(a * 0.06), math.sin(a * 0.06)] for a in range(50)]
        corpus = tight + scattered
        self.assertEqual(tc.classify(tight[:10], corpus), tc.CONCENTRATED)

    def test_stride_sample_is_deterministic_and_capped(self):
        vectors = [[float(i), 0.0] for i in range(1000)]
        first = tc.stride_sample(vectors, limit=100)
        self.assertEqual(first, tc.stride_sample(vectors, limit=100))
        self.assertLessEqual(len(first), 100)
        # 전 구간에 걸쳐야 한다 — 앞 100개 안에 갇히면 안 됨
        self.assertGreater(first[-1][0], 100)
        # 상한보다 작으면 그대로 통과
        self.assertEqual(tc.stride_sample(vectors[:50], limit=100), vectors[:50])

    def test_absolute_high_cosine_corpus_still_relative(self):
        # 코퍼스가 전반적으로 뭉쳐 있어도(같은 도메인, 절대 cos 높음) 실패가 코퍼스만큼만
        # 뭉쳤으면 concentrated 가 아니어야 한다 — 절대값이 아니라 baseline 대비 비율이므로.
        base = [1.0, 0.0]
        near = [math.cos(0.2), math.sin(0.2)]   # 서로 가까운 방향들
        corpus = [base, near, [math.cos(0.15), math.sin(0.15)], [math.cos(0.25), math.sin(0.25)]]
        failed = [base, near]                    # 코퍼스와 같은 수준의 응집
        self.assertNotEqual(tc.classify(failed, corpus), tc.CONCENTRATED)


class TopicClusterClassifyDetailTest(unittest.TestCase):
    """classify_detail — 버킷뿐 아니라 캘리브레이션 근거 수치를 함께 돌려주는지.

    소비 유예(관측용) 동안 임계값 재보정 근거를 finding.metadata 에 쌓으려면, 판정에
    쓴 ratio·baseline·응집도·표본 수가 휘발되지 않고 반환돼야 한다(리뷰 Medium).
    """

    def test_classify_matches_detail_bucket(self):
        # classify 는 classify_detail 의 얇은 래퍼여야 한다 — 두 버킷이 항상 같다.
        cases = [
            ([[1, 0, 0], [0.98, 0.1, 0], [0.97, 0.15, 0.05]],
             [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]]),
            ([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
             [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]]),
            ([[1, 0, 0]], [[1, 0, 0], [0, 1, 0]]),           # unmeasured
        ]
        for failed, corpus in cases:
            self.assertEqual(tc.classify(failed, corpus),
                             tc.classify_detail(failed, corpus).bucket)

    def test_detail_carries_ratio_and_sample_sizes(self):
        # 판정된 회차는 ratio·baseline·응집도와 유효 표본 수를 다 실어야 한다.
        failed = [[1, 0, 0], [0.98, 0.1, 0], [0.97, 0.15, 0.05]]
        corpus = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]]
        r = tc.classify_detail(failed, corpus)
        self.assertEqual(r.bucket, tc.CONCENTRATED)
        self.assertIsNotNone(r.ratio)
        self.assertIsNotNone(r.baseline)
        self.assertIsNotNone(r.failed_cohesion)
        # ratio 가 임계값과 정합해야 한다(판정 근거 재현성).
        self.assertGreaterEqual(r.ratio, tc.TOPIC_CLUSTER_CONCENTRATED_RATIO)
        self.assertAlmostEqual(r.ratio, r.failed_cohesion / r.baseline)
        self.assertEqual(r.failed_sample_size, 3)
        self.assertEqual(r.corpus_sample_size, 5)

    def test_detail_counts_only_valid_samples(self):
        # 표본 수는 _valid 통과분(응집도 계산 실입력)만 센다 — 영벡터/None 은 빠진다.
        failed = [[1, 0, 0], [0.98, 0.1, 0], [], None, [0.0, 0.0, 0.0]]
        corpus = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        r = tc.classify_detail(failed, corpus)
        self.assertEqual(r.failed_sample_size, 2)   # 영벡터·빈·None 3개 제외

    def test_unmeasured_detail_keeps_sample_size(self):
        # 못 잰 회차도 표본 수는 남아, unmeasured 원인(2개 미만 vs baseline 불가)이 구분된다.
        r = tc.classify_detail([[1, 0, 0]], [[1, 0, 0], [0, 1, 0]])
        self.assertEqual(r.bucket, tc.UNMEASURED)
        self.assertIsNone(r.ratio)
        self.assertEqual(r.failed_sample_size, 1)    # 유효 1개 → 쌍 없음 → unmeasured


if __name__ == "__main__":
    unittest.main()
