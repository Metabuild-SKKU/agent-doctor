"""
tools/topic_cluster_calibration/collect_ratio_distribution.py
topic_cluster.classify 의 ratio 분포를 실측해 임계값 캘리브레이션 근거를 만든다.

목적(CONTEXT.md 참고): TOPIC_CLUSTER_CONCENTRATED_RATIO / SPREAD_RATIO 두 경계가
캘리브레이션 전 임의값이라, "신호 없음(none)" 상황의 ratio 분포를 실측해 none 대
폭을 분산에 맞춰 다시 정한다.

접근:
  - gold 정합이 맞는 코퍼스의 청크 임베딩을 로드한다(gold 깨진 eval_probes.json 금지).
  - 실패 gold 를 무작위로 뽑아 "주제 신호 없음"을 인위적으로 만든다(seed 고정 = 재현).
  - 실패 gold 수 N 을 바꿔가며 classify_detail 의 ratio 를 반복 수집 → N 별 분포.

비용: 코사인만 쓴다(LLM 0). classify_detail 이 표본 상한(stride_sample)을 걸어
O(n^2) 폭발을 막으므로 반복 수집이 저렴하다.

실행:
    PYTHONPATH=. python tools/topic_cluster_calibration/collect_ratio_distribution.py

주의: 아직 draft — 코퍼스 임베딩 로더(_load_corpus_embeddings)와 통계/리포트
출력은 다음 커밋에서 채운다. 지금은 수집 루프의 골격과 재사용 지점만 배선한다.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.eval import topic_cluster as tc

# 결정성: 무작위 표본은 seed 고정으로 재현 가능하게 한다(probe_gen 전반의 결정성 원칙).
DEFAULT_SEED = 20260731
# 실패 gold 수 그리드 — types.py 표(10/20/100)를 포함해 실측 구간을 넓힌다.
DEFAULT_FAILED_N_GRID = (5, 10, 20, 40, 100)
# N 마다 서로 다른 무작위 실패 표본을 몇 번 뽑아 분포를 낼지.
DEFAULT_TRIALS_PER_N = 200


@dataclass
class RatioSample:
    """한 번의 무작위 추출에서 나온 classify_detail 결과 요약."""
    failed_n: int          # 이번 추출에서 '실패 gold' 로 뽑은 벡터 수(요청값)
    bucket: str
    ratio: float | None
    failed_sample_size: int    # _valid 통과 후 실제 응집도 계산에 들어간 수


def _load_corpus_embeddings(corpus_ref: str) -> list[list[float]]:
    """gold 정합 코퍼스의 청크 dense 임베딩 리스트를 반환한다.

    TODO(다음 커밋): tests/corpus/qa.json + 재적재 경로, 또는 KorQuAD taxonomy
    (agents/eval/datasets/korquad.reconstruct_documents → Index)로 청크를 만들고
    Chunk.embedding 을 뽑는다. gold 깨진 eval_probes.json 은 절대 쓰지 않는다.
    지금은 배선만 — 실제 로딩은 미구현이라 명시적으로 막아둔다.
    """
    raise NotImplementedError(
        "코퍼스 임베딩 로더는 아직 미구현 — 다음 커밋에서 gold 정합 코퍼스"
        "(tests/corpus/qa.json / KorQuAD taxonomy)에서 Chunk.embedding 을 뽑는다."
    )


def sample_null_ratios(
    corpus_vecs: list[list[float]],
    failed_n_grid=DEFAULT_FAILED_N_GRID,
    trials_per_n: int = DEFAULT_TRIALS_PER_N,
    seed: int = DEFAULT_SEED,
) -> list[RatioSample]:
    """'신호 없음' 상황의 ratio 분포를 수집한다.

    실패 gold 를 코퍼스 전체에서 무작위로 뽑으면(특정 주제로 안 몰면) 정답은
    none 이어야 한다 — 여기서 나오는 ratio 의 산포가 곧 노이즈 폭이고, 경계는
    이 폭보다 넓어야 한다.

    TODO(다음 커밋): seed 고정 무작위 추출(random.Random(seed))로 failed 표본을
    뽑아 tc.classify_detail(failed, corpus_vecs) 를 호출하고 RatioSample 로 모은다.
    """
    raise NotImplementedError("수집 루프는 다음 커밋 — 골격만 배선.")


def summarize(samples: list[RatioSample]) -> None:
    """N 별 ratio 중앙값·stdev·분위수를 출력하고 경계 후보를 제안한다.

    TODO(다음 커밋): failed_n 별로 묶어 median/stdev/p5/p95 를 내고, none 분포의
    p95/p5 를 CONCENTRATED/SPREAD 경계 후보로 제시한다. 신호 있는 표본(같은 주제
    gold 만 뽑기)과 대비해 분리도도 함께 낸다.
    """
    raise NotImplementedError("통계·리포트는 다음 커밋 — 골격만 배선.")


def main() -> int:
    parser = argparse.ArgumentParser(description="topic_cluster ratio 분포 수집")
    parser.add_argument("--corpus", default="tests/corpus/qa.json",
                        help="gold 정합 코퍼스 참조(기본: tests/corpus/qa.json)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS_PER_N)
    args = parser.parse_args()

    corpus_vecs = _load_corpus_embeddings(args.corpus)
    samples = sample_null_ratios(corpus_vecs, trials_per_n=args.trials, seed=args.seed)
    summarize(samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
