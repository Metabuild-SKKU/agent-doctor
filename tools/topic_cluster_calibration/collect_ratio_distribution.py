"""
tools/topic_cluster_calibration/collect_ratio_distribution.py
topic_cluster.classify 의 ratio 분포를 실측해 임계값 캘리브레이션 근거를 만든다.

목적(CONTEXT.md 참고): 옛 고정 경계(CONCENTRATED_RATIO=1.3 / SPREAD_RATIO=1.1)가
캘리브레이션 전 임의값이라, "신호 없음(none)" 상황의 ratio 분포를 실측해 none 대
폭을 분산에 맞춰 다시 정했다. 결과: 경계는 실패 gold 수 N 에 따라 동적으로
(1.0 ± k·C/sqrt(N)) 잡는다 — topic_cluster.dynamic_bounds(N), types.py 계수 참고.
이 스크립트는 그 확정 근거를 재현·검증하는 자리다.

접근:
  - gold 정합이 맞는 코퍼스의 청크 임베딩을 로드한다(gold 깨진 eval_probes.json 금지).
  - 실패 gold 를 무작위로 뽑아 "주제 신호 없음"을 인위적으로 만든다(seed 고정 = 재현).
  - 실패 gold 수 N 을 바꿔가며 classify_detail 의 ratio 를 반복 수집 → N 별 분포.

비용: 코사인만 쓴다(LLM 0). classify_detail 이 표본 상한(stride_sample)을 걸어
O(n^2) 폭발을 막으므로 반복 수집이 저렴하다.

입력: run_corpus/serve 가 남긴 chunks.json — gold 정합 코퍼스 PDF 를 정규
파이프라인으로 한 번 색인하면 Chunk.embedding(BGE-M3)이 실려 있어, 재색인·모델
다운로드 없이 이 파일만 읽는다.

실행:
    PYTHONPATH=. python tools/topic_cluster_calibration/collect_ratio_distribution.py
    PYTHONPATH=. python tools/topic_cluster_calibration/collect_ratio_distribution.py \
        --corpus chunks.json --trials 200 --seed 20260731

재보정 주의: 리포트의 버킷 분포(none 비율)가 현재 동적 경계의 흡수력이고, 분위수
표는 참고다. 단, 실패 gold 표본이 코퍼스의 절반을 넘으면 baseline(코퍼스 전체)과
겹쳐 ratio 가 1.0 에 구조적으로 붙으므로(자기 비교), 그런 N 은 리포트가 '(상한)'으로
표시한다. 계수를 다시 정할 때는 충분히 큰 코퍼스(types.py 표는 문서 10×50=500청크
기준)에서 뽑아야 한다 — 작은 코퍼스 결과로 계수를 바꾸지 말 것.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.eval import topic_cluster as tc

# run_corpus / serve 가 남기는 색인 산출물. gold 정합 코퍼스 PDF 를 한 번 색인하면
# 여기에 Chunk.embedding(BGE-M3 1024-dim)이 그대로 실린다 — 재색인·모델 다운로드 없이
# 이 파일만 읽어 ratio 분포를 뽑는다. eval_probes.json(gold_spans=[])은 쓰지 않는다.
DEFAULT_CORPUS = "chunks.json"

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

    입력은 run_corpus/serve 가 남긴 chunks.json — gold 정합 코퍼스 PDF 를 정규
    파이프라인(Ingest→Index)으로 한 번 색인하면 Chunk.embedding 이 그대로 직렬화돼
    있다. 그래서 여기선 BGE-M3 재다운로드나 재색인 없이 파일만 읽는다.

    포맷은 serve/agent.py 의 직렬화와 같다: 청크 dict 의 리스트(또는 {"chunks":[...]}).
    각 청크의 "embedding" 이 1024-dim dense 벡터다. _valid(=None/영벡터 배제)는
    호출부(classify_detail)가 걸므로 여기서 미리 거르지 않는다 — 그래야 표본 슬롯을
    영벡터가 먼저 먹는 순서 뒤집힘(topic_cluster 주석)이 재현되지 않는다.

    gold 깨진 eval_probes.json(gold_spans=[])은 절대 쓰지 않는다 — ratio 오염.
    """
    with open(corpus_ref, encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"] if isinstance(data, dict) else data
    vecs = [c.get("embedding") for c in chunks]
    present = [v for v in vecs if v]
    if not present:
        raise ValueError(
            f"{corpus_ref} 에 임베딩이 없다 — run_corpus 로 gold 정합 코퍼스를 한 번 "
            "색인해 Chunk.embedding 이 실린 chunks.json 을 만든 뒤 다시 실행하라."
        )
    return present


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

    표본은 복원 없이(random.sample) 뽑는다 — 실제 실패 gold 는 서로 다른 청크라,
    복원 추출로 같은 벡터를 중복시키면 응집도가 인위로 부풀어 ratio 가 왜곡된다.
    그래서 요청 N 은 코퍼스 크기로 상한된다(N>corpus 는 뽑을 수 없다). 요청값과 실제
    표본 수(failed_sample_size)를 둘 다 남겨, 상한에 걸린 회차를 리포트가 드러낸다.

    seed 는 회차마다 (seed, N, trial) 로 갈라 결정성을 유지한다 — 같은 seed 로 다시
    돌리면 같은 분포가 나온다(probe_gen 전반의 결정성 원칙).

    성능: baseline(코퍼스 전체 응집도)은 코퍼스가 고정이라 trial 내내 상수다. trial
    마다 classify_detail 을 부르면 이 baseline(stride 후 O(100^2) 코사인)을 매번 다시
    계산해 1000회면 수십 분이 든다. 그래서 baseline·stride 표본을 여기서 한 번만 잡고
    (tc.stride_sample / tc._mean_pairwise_cosine 을 그대로 호출 — 규칙을 복제하지 않는다),
    trial 루프는 실패표본 응집도만 새로 잰다. 판정 임계값도 tc 의 상수를 그대로 읽어
    topic_cluster 와 어긋나지 않는다.
    """
    corpus_size = len(corpus_vecs)
    # baseline 은 한 번만 — classify_detail 과 같은 순서(_valid → stride → 쌍별 코사인)로.
    corpus_usable = tc.stride_sample([v for v in corpus_vecs if tc._valid(v)])
    baseline = tc._mean_pairwise_cosine(corpus_usable)
    if baseline is None or baseline <= 0:
        raise ValueError(
            f"코퍼스 baseline 을 잴 수 없다(baseline={baseline}) — 유효 임베딩이 2개 "
            "미만이거나 코퍼스가 서로 등질이다. classify_detail 이면 unmeasured 로 떨어진다."
        )

    samples: list[RatioSample] = []
    for failed_n in failed_n_grid:
        draw = min(failed_n, corpus_size)
        if draw < 2:
            # 2개 미만이면 응집도 계산 불가 → classify_detail 이면 unmeasured. 수집 의미 없음.
            continue
        for trial in range(trials_per_n):
            # (seed, N, trial) 을 하나의 int 로 접어 회차마다 독립·재현 가능하게.
            rng = random.Random(hash((seed, failed_n, trial)) & 0x7FFFFFFF)
            failed = rng.sample(corpus_vecs, draw)
            # 실패표본도 classify_detail 과 같은 순서로 정규화한 뒤 응집도만 새로 잰다.
            failed_usable = tc.stride_sample([v for v in failed if tc._valid(v)])
            failed_cohesion = tc._mean_pairwise_cosine(failed_usable)
            if failed_cohesion is None:
                bucket, ratio = tc.UNMEASURED, None
            else:
                ratio = failed_cohesion / baseline
                # 경계는 tc.dynamic_bounds(N) 로 — 스크립트가 규칙을 복제하지 않는다.
                spread_hi, concentrated_lo = tc.dynamic_bounds(len(failed_usable))
                if ratio >= concentrated_lo:
                    bucket = tc.CONCENTRATED
                elif ratio <= spread_hi:
                    bucket = tc.SPREAD
                else:
                    bucket = tc.NONE
            samples.append(RatioSample(
                failed_n=failed_n,
                bucket=bucket,
                ratio=ratio,
                failed_sample_size=len(failed_usable),
            ))
    return samples


def _quantile(sorted_xs: list[float], q: float) -> float:
    """정렬된 표본의 q 분위수(선형 보간). statistics.quantiles 는 표본이 적으면
    던지므로, 캘리브레이션 리포트가 작은 코퍼스에서도 죽지 않게 직접 계산한다."""
    if not sorted_xs:
        return float("nan")
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = q * (len(sorted_xs) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 >= len(sorted_xs):
        return sorted_xs[-1]
    return sorted_xs[lo] * (1 - frac) + sorted_xs[lo + 1] * frac


def summarize(samples: list[RatioSample]) -> None:
    """N 별 ratio 중앙값·stdev·분위수를 출력하고 경계 후보를 제안한다.

    null 분포(주제 신호 없음)에서 나온 ratio 의 산포가 곧 노이즈 폭이다. 경계는 이
    폭을 덮어야 신호 없는 회차가 spread/concentrated 로 안 튄다. p5/p95 를 SPREAD/
    CONCENTRATED 경계 후보로 제시한다(전체 null 분포 기준 — 실운영은 여러 N 이 섞임).
    """
    if not samples:
        print("표본이 없다 — 코퍼스가 너무 작거나(N<2) 임베딩이 비었다.")
        return

    ratios_all = [s.ratio for s in samples if s.ratio is not None]

    print(f"\n{'N(요청)':>8} {'N(실표본)':>9} {'회차':>5} {'중앙값':>8} "
          f"{'stdev':>7} {'p5':>7} {'p95':>7}  버킷분포(none/spread/conc/unmeas)")
    print("-" * 92)
    for failed_n in sorted({s.failed_n for s in samples}):
        grp = [s for s in samples if s.failed_n == failed_n]
        rs = sorted(s.ratio for s in grp if s.ratio is not None)
        actual = statistics.median(s.failed_sample_size for s in grp)
        counts = {"none": 0, "spread": 0, "concentrated": 0, "unmeasured": 0}
        for s in grp:
            counts[s.bucket] = counts.get(s.bucket, 0) + 1
        med = f"{statistics.median(rs):.3f}" if rs else "n/a"
        sd = f"{statistics.stdev(rs):.3f}" if len(rs) >= 2 else "n/a"
        p5 = f"{_quantile(rs, 0.05):.3f}" if rs else "n/a"
        p95 = f"{_quantile(rs, 0.95):.3f}" if rs else "n/a"
        capped = " (상한)" if actual < failed_n else ""
        print(f"{failed_n:>8} {int(actual):>9}{'':1} {len(grp):>5} {med:>8} "
              f"{sd:>7} {p5:>7} {p95:>7}  "
              f"{counts['none']}/{counts['spread']}/"
              f"{counts['concentrated']}/{counts['unmeasured']}{capped}")

    if ratios_all:
        allsorted = sorted(ratios_all)
        p5 = _quantile(allsorted, 0.05)
        p95 = _quantile(allsorted, 0.95)
        print("\n=== null 분포 전체 ===")
        print(f"  표본 {len(ratios_all)}개  중앙값={statistics.median(allsorted):.3f}  "
              f"stdev={statistics.stdev(allsorted):.3f}"
              if len(allsorted) >= 2 else f"  표본 {len(allsorted)}개")
        print(f"  분위수    p5 = {p5:.3f}   p95 = {p95:.3f}  "
              f"(고정 경계였다면 이 근처가 후보)")
        # 경계는 동적(tc.dynamic_bounds) 이라 N 마다 다르다 — 계수와, 위 N별 표의
        # 버킷 분포(none 비율)가 곧 현재 경계의 흡수력이다.
        print(f"  동적 경계  1.0 ± {tc.TOPIC_CLUSTER_BOUNDARY_K}·"
              f"{tc.TOPIC_CLUSTER_NULL_C}/sqrt(N)  "
              f"(k={tc.TOPIC_CLUSTER_BOUNDARY_K}, C={tc.TOPIC_CLUSTER_NULL_C})")
        none_total = sum(1 for s in samples if s.bucket == tc.NONE)
        print(f"  현재 동적 경계로 null 표본 중 none 비율: "
              f"{none_total}/{len(samples)} = {none_total / len(samples):.1%}  "
              f"(높을수록 노이즈를 none 으로 잘 흡수)")


def main() -> int:
    parser = argparse.ArgumentParser(description="topic_cluster ratio 분포 수집")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS,
                        help="gold 정합 코퍼스 색인 산출물(기본: chunks.json, "
                             "run_corpus/serve 가 Chunk.embedding 을 실어 남긴 파일)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS_PER_N)
    args = parser.parse_args()

    corpus_vecs = _load_corpus_embeddings(args.corpus)
    print(f"코퍼스: {args.corpus} — 유효 임베딩 {len(corpus_vecs)}개, seed={args.seed}, "
          f"trials/N={args.trials}")
    # 실패 gold 표본이 코퍼스의 상당 비율을 차지하면 baseline(코퍼스 전체)과 겹쳐
    # ratio 가 구조적으로 1.0 에 붙는다(자기 자신과 비교). 그런 N 의 분포는 노이즈
    # 폭이 아니라 상한 아티팩트다 — 캘리브레이션 근거로 쓰기 전에 반드시 인지해야 한다.
    biggest_useful = max((n for n in DEFAULT_FAILED_N_GRID
                          if n <= len(corpus_vecs) // 2), default=None)
    if biggest_useful is None:
        print("  주의: 코퍼스가 너무 작다(유효 임베딩 부족) — 어떤 N 도 코퍼스의 절반을\n"
              "        넘지 않게 잡을 수 없다. 이 코퍼스로 낸 경계는 신뢰하지 말 것.\n"
              "        더 큰 gold 정합 코퍼스를 색인해(run_corpus) chunks.json 을 키운 뒤\n"
              "        재실행하라(수백 청크 이상 권장 — types.py 표는 문서 10×50청크 기준).")
    samples = sample_null_ratios(corpus_vecs, trials_per_n=args.trials, seed=args.seed)
    summarize(samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
