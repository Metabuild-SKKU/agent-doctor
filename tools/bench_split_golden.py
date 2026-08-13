"""
tools/bench_split_golden.py
벤치마크용 골든셋 분할 — 100건을 train/test 로 한 번 갈라 고정한다.

왜 필요한가: 벤치마크에서 우리 파이프라인이 골든셋 전체로 튜닝하고 **같은** 골든셋으로
채점하면 과적합이다("답을 보고 시험 본 점수"). 비교 상대인 AutoRAG 는 train/test 를
나눠 쓰므로, 그대로 붙이면 조건이 달라 어느 쪽이 이기든 결론을 못 쓴다.
그래서 양쪽 모두 train 으로만 최적화하고 test 로만 채점한다.

제품 코드는 건드리지 않는다: 파이프라인은 "골든셋 파일 하나"를 받으므로, 어떤 파일을
주느냐만 바꾸면 된다. 고객 시나리오에서는 골든셋 전체로 튜닝하는 게 정상이다
(맞춰야 할 목표이지 시험지가 아니다) — 분할은 발표용 실험 설계에만 필요하다.

왜 test 를 50건이나 두나: agents/eval/replay.py 가 골든셋 권장 크기를 50~150건으로
안내한다. test 가 그 하한을 밑돌면 두 시스템의 점수 차이가 노이즈에 묻혀 "표본이
너무 적다"는 반론에 답할 수 없다. 벤치마크에서는 튜닝 재료보다 최종 점수의 신뢰도가
우선이라 50/50 을 기본값으로 둔다.

왜 무작위인가: 질문 파일의 앞뒤로 주제가 몰려 있을 수 있어(원본 문서 순서대로 만들면
그렇게 된다) 앞에서 N개를 자르면 train/test 의 주제 분포가 갈린다.

왜 시드를 고정하나: 분할이 바뀌면 이전에 측정한 점수와 비교할 수 없다. 그래서 시드를
manifest 에 남기고, 이미 만들어진 분할은 --force 없이 덮어쓰지 않는다.

사용법:
    python -m tools.bench_split_golden                      # 기본 50/50, bench/ 로
    python -m tools.bench_split_golden --test=40            # 60/40
    python -m tools.bench_split_golden --force              # 기존 분할 덮어쓰기(주의)
"""
from __future__ import annotations

import argparse
import json
import os
import random

DEFAULT_INPUT = "tools/tax_guide_questions.json"
DEFAULT_OUTDIR = "bench"
# 분할을 재현할 수 있게 박아둔 값. 바꾸면 이전 측정과 비교가 끊긴다.
DEFAULT_SEED = 20260813
DEFAULT_TEST = 50

# replay.py 의 GOLDEN_RECOMMENDED_MAX 짝 — 권장 하한. test 가 이보다 작으면 경고한다.
GOLDEN_RECOMMENDED_MIN = 50


def load_golden(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list) or not items:
        raise SystemExit(f"[split] 골든셋이 비었거나 리스트가 아닙니다: {path}")
    return items


def check_quality(items: list[dict]) -> list[str]:
    """분할 전에 골든셋 자체의 결손을 잡는다. 분할 후에 발견하면 어느 쪽에 몇 건이
    들어갔는지까지 따져야 해서 되돌리기가 번거롭다."""
    problems = []
    missing_q = sum(1 for x in items if not str(x.get("question") or "").strip())
    missing_gt = sum(1 for x in items if not str(x.get("ground_truth") or "").strip())
    missing_gc = sum(1 for x in items if not x.get("gold_contexts"))
    dup = len(items) - len({str(x.get("question")) for x in items})
    if missing_q:
        problems.append(f"question 결손 {missing_q}건")
    if missing_gt:
        problems.append(f"ground_truth 결손 {missing_gt}건 — 정답 대조 지표가 미측정된다")
    if missing_gc:
        problems.append(f"gold_contexts 결손 {missing_gc}건 — 검색축 판정이 침묵한다")
    if dup:
        problems.append(f"질문 중복 {dup}건 — 같은 질문이 train/test 양쪽에 갈리면 누수다")
    return problems


def split(items: list[dict], test_size: int, seed: int) -> tuple[list[dict], list[dict]]:
    """무작위 분할. 원본 리스트는 건드리지 않는다(호출자가 다시 쓸 수 있게)."""
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    return shuffled[test_size:], shuffled[:test_size]


def write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(description="벤치마크용 골든셋 train/test 분할")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--test", type=int, default=DEFAULT_TEST, help="test 건수(나머지가 train)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--force", action="store_true", help="기존 분할을 덮어쓴다")
    args = ap.parse_args()

    items = load_golden(args.input)
    if not 0 < args.test < len(items):
        raise SystemExit(f"[split] --test 는 1~{len(items) - 1} 사이여야 합니다(현재 {args.test})")

    problems = check_quality(items)
    if problems:
        print("[split] 골든셋 점검에서 문제가 발견됐습니다:")
        for p in problems:
            print(f"  ! {p}")
        print("  (분할은 계속 진행합니다 — 위 항목은 측정 결과 해석에 영향을 줍니다)")

    train, test = split(items, args.test, args.seed)

    train_path = os.path.join(args.outdir, f"golden_train_{len(train)}.json")
    test_path = os.path.join(args.outdir, f"golden_test_{len(test)}.json")
    manifest_path = os.path.join(args.outdir, "golden_split_manifest.json")

    # 이미 분할이 있으면 조용히 덮지 않는다. 분할이 바뀌면 그 전에 측정한 점수가 전부
    # 무효가 되는데, 리포트만 봐서는 분할이 바뀐 사실을 알 수 없다.
    existing = [p for p in (train_path, test_path, manifest_path) if os.path.exists(p)]
    if existing and not args.force:
        print("[split] 이미 분할이 존재합니다:")
        for p in existing:
            print(f"  · {p}")
        print("  분할을 바꾸면 이전에 측정한 점수와 비교할 수 없습니다.")
        print("  그래도 다시 나누려면: --force")
        return 2

    if len(test) < GOLDEN_RECOMMENDED_MIN:
        print(f"[split] 경고: test {len(test)}건은 권장 하한({GOLDEN_RECOMMENDED_MIN})보다 적습니다."
              " 두 시스템의 점수 차이가 노이즈에 묻힐 수 있습니다.")

    write_json(train_path, train)
    write_json(test_path, test)
    # manifest 는 "이 점수가 어떤 분할에서 나왔나"를 나중에 증명하는 근거다. 질문 원문을
    # 그대로 실어 두면 파일이 사라져도 분할을 복원할 수 있다.
    write_json(manifest_path, {
        "source": args.input,
        "seed": args.seed,
        "total": len(items),
        "train_count": len(train),
        "test_count": len(test),
        "train_questions": [x["question"] for x in train],
        "test_questions": [x["question"] for x in test],
    })

    print(f"[split] {args.input} {len(items)}건 → train {len(train)} / test {len(test)} (seed={args.seed})")
    print(f"  · {train_path}")
    print(f"  · {test_path}")
    print(f"  · {manifest_path}")
    print()
    print("다음 단계:")
    print(f"  · 우리 파이프라인 최적화 → {train_path} 만 사용")
    print(f"  · AutoRAG 최적화        → {train_path} 를 qa.parquet 로 변환")
    print(f"  · 최종 채점(양쪽 모두)  → {test_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
