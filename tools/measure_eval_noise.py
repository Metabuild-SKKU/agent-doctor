"""
tools/measure_eval_noise.py
같은 config 로 Eval 을 N 번 돌려 재측정 편차(σ)를 잰다 — 개선 마진이 노이즈보다 큰지 확인.

## 왜 필요한가

Optimize 는 처방 적용 후 종합점수가 MIN_IMPROVEMENT_MARGIN(0.02 = 2점) 이상 오르면
유지, 아니면 롤백한다. 그 마진은 "노이즈로 우연히 오른 점수를 개선으로 보지 않는다"가
목적인데, **노이즈가 실제로 얼마인지 측정된 적이 없다.** 지금 0.02 는 잠정치다.

history.max_repeated_measurement_spread() 가 정상 실행에서 σ 를 공짜로 줍도록 설계돼
있으나, 롤백이 Eval 을 재실행하지 않고 진단 캐시를 복원하므로(로그의 "롤백 진단 캐시
복원") 같은 config 가 두 번 측정되는 일이 없어 항상 None 이 나온다. 그래서 전용 측정이
필요하다.

실측 근거(output/logs/corpus_20260804_103059.txt, 30문항):
    반복 0: 종합 75  (리랭커 off)
    반복 5: 종합 73  (리랭커 off — 리랭커가 꺼져 있으면 rerank_candidates 는 검색에
                      관여하지 않으므로 기능적으로 같은 config)
    → 편차 0.020 = 마진 0.020. 개선과 노이즈가 구분되지 않는다.

n=2 관측이라 σ 의 하한일 뿐이다. 이 스크립트가 제대로 된 표본을 만든다.

## 무엇을 재는가

Ingest·Index 를 한 번만 하고 Eval 만 N 회 반복한다(인덱싱 재실행 낭비 제거).
Eval 은 taxonomy 소스에서 Probe 캐시를 우회하고 state.report 를 매번 초기화하므로
재실행이 안전하다(agents/eval/agent.py).

    · 종합점수(composite) · 품질 · 신뢰도 · overall 의 회차별 값과 폭
    · 라벨 분포의 회차별 변동 — 점수가 같아도 라벨이 흔들리면 처방 선택이 흔들린다

## 사용법

    python tools/measure_eval_noise.py                # .env 설정 그대로 3회
    python tools/measure_eval_noise.py -n 5
    python tools/measure_eval_noise.py -n 3 --qa-limit 30    # 문항수 영향 비교용

규모는 .env 의 KORQUAD_MAX_DOCS / KORQUAD_QA_LIMIT / EVAL_TAXONOMY_QA 를 따른다.
LLM 을 실제로 호출하므로 비용이 든다(150문항 1회 ≈ 25분).
"""
from __future__ import annotations

import argparse
import collections
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

from core.console import force_utf8_stdio
force_utf8_stdio()


def _label_counts(report) -> dict[str, int]:
    """라벨별 finding 수. findings_summary 형태에 의존하지 않도록 findings 에서 직접 센다."""
    counts: collections.Counter = collections.Counter()
    for finding in getattr(report, "findings", None) or []:
        label = getattr(finding, "label", None)
        if label:
            counts[label] += 1
    return dict(counts)


def _components(report) -> dict[str, float | None]:
    """composite_score dict → {'quality': 86, '신뢰도키': 66} 형태로 평탄화."""
    composite = getattr(report, "composite_score", None) or {}
    return {c["key"]: c.get("score") for c in composite.get("components", [])}


def _spread(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return (max(vals) - min(vals)) if len(vals) > 1 else None


def run_once(state, index_run, eval_run):
    """Index(설정만 반영) → Eval 1회. 새 report 를 단 state 를 돌려준다."""
    state = index_run(state)
    if state.error:
        raise RuntimeError(f"Index 오류: {state.error}")
    state = eval_run(state)
    if state.error:
        raise RuntimeError(f"Eval 오류: {state.error}")
    return state


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("-n", "--runs", type=int, default=3, help="Eval 반복 횟수(기본 3)")
    ap.add_argument("--qa-limit", type=int, default=None,
                    help="KORQUAD_QA_LIMIT 임시 덮어쓰기(문항수 영향 비교용)")
    args = ap.parse_args()

    if args.qa_limit:
        os.environ["KORQUAD_QA_LIMIT"] = str(args.qa_limit)

    from core.run_logger import setup_run_logging
    log_path = setup_run_logging(prefix="eval_noise")

    from core.state import AgentDoctorState
    from agents.ingest.agent import run as ingest_run
    from agents.index.agent import run as index_run
    from agents.eval.agent import run as eval_run
    from agents.optimize.history import MIN_IMPROVEMENT_MARGIN

    source_type = os.getenv("SOURCE_TYPE", "korquad").strip().lower()
    if source_type == "korquad":
        os.environ.setdefault("EVAL_PROBE_SOURCE", "taxonomy")

    print(f"[log] {log_path}")
    print(f"  소스   : {source_type} / {os.getenv('SOURCE_URL', 'data/corpus.jsonl')}")
    print(f"  QA     : {os.getenv('EVAL_TAXONOMY_QA', 'data/qa_pairs.jsonl')}")
    print(f"  제한   : 문서 {os.getenv('KORQUAD_MAX_DOCS') or '전체'} / "
          f"QA {os.getenv('KORQUAD_QA_LIMIT') or '전체'}")
    print(f"  진단   : EVAL_MODE={os.getenv('EVAL_MODE', '-')} "
          f"LLM={os.getenv('EVAL_ENABLE_LLM', '-')}")
    print(f"  반복   : {args.runs}회 (config 변경 없음)")

    state = AgentDoctorState()
    state.source_type = source_type
    state.source_url = os.getenv("SOURCE_URL", "data/corpus.jsonl")

    print("\n=== Ingest (1회) ===")
    state = ingest_run(state)
    if state.error:
        print(f"[중단] Ingest 오류: {state.error}")
        return 1

    rows: list[dict] = []
    for i in range(1, args.runs + 1):
        print(f"\n=== Eval {i}/{args.runs} ===")
        started = time.time()
        state = run_once(state, index_run, eval_run)
        report = state.report
        comp = (getattr(report, "composite_score", None) or {}).get("total")
        rows.append({
            "composite": comp,
            "overall": getattr(report, "overall_score", None),
            "components": _components(report),
            "labels": _label_counts(report),
            "seconds": time.time() - started,
        })
        print(f"  → 종합 {comp} · overall {getattr(report, 'overall_score', None)} "
              f"({rows[-1]['seconds']:.0f}초)")

    _print_summary(rows, MIN_IMPROVEMENT_MARGIN)
    return 0


def _print_summary(rows: list[dict], margin: float) -> None:
    print("\n" + "=" * 60)
    print("  재측정 편차")
    print("=" * 60)

    comps = [r["composite"] for r in rows]
    print("\n종합점수(0~100)")
    print("  회차별: " + ", ".join(str(c) for c in comps))
    spread = _spread(comps)
    if spread is None:
        print("  → 측정값이 부족합니다(1회 이하).")
        return

    mean = statistics.mean(c for c in comps if c is not None)
    stdev = statistics.stdev([c for c in comps if c is not None]) if len(comps) > 2 else None
    print(f"  평균 {mean:.1f} · 폭 {spread:.1f}점"
          + (f" · 표준편차 {stdev:.2f}" if stdev is not None else ""))

    # 마진은 0~1 스케일, composite 은 0~100 → 같은 축으로 맞춰 비교한다.
    margin_pts = margin * 100
    print(f"\n개선 마진 {margin_pts:.1f}점 (MIN_IMPROVEMENT_MARGIN={margin:.3f})")
    if spread >= margin_pts:
        print(f"  ⚠ 편차 {spread:.1f} ≥ 마진 {margin_pts:.1f} — "
              "'개선'과 노이즈가 구분되지 않습니다. 마진 재보정 필요.")
        print(f"     권고: 마진 ≥ {max(margin_pts, spread * 1.5):.1f}점 "
              "(관측 폭의 1.5배) 또는 회차 평균으로 판정.")
    else:
        print(f"  ✓ 편차 {spread:.1f} < 마진 {margin_pts:.1f} — 마진이 노이즈를 덮습니다.")

    keys = [k for k in (rows[0]["components"] or {})]
    if keys:
        print("\n성분별")
        for key in keys:
            vals = [r["components"].get(key) for r in rows]
            s = _spread(vals)
            print(f"  {key:<12} " + ", ".join(str(v) for v in vals)
                  + (f"   폭 {s:.1f}" if s is not None else ""))

    print("\n라벨 분포 (회차별 건수)")
    all_labels = sorted({l for r in rows for l in r["labels"]})
    if not all_labels:
        print("  (finding 없음)")
        return
    width = max(len(l) for l in all_labels)
    unstable = 0
    for label in all_labels:
        vals = [r["labels"].get(label, 0) for r in rows]
        flag = ""
        if min(vals) == 0 < max(vals):
            flag = "  ← 회차에 따라 나타났다 사라짐"
            unstable += 1
        elif max(vals) - min(vals) > 0:
            flag = "  ← 건수 흔들림"
        print(f"  {label:<{width}}  " + " ".join(f"{v:>3}" for v in vals) + flag)
    if unstable:
        print(f"\n  ⚠ 라벨 {unstable}개가 회차에 따라 나타났다 사라집니다 — "
              "같은 config 인데 처방 후보가 달라진다는 뜻입니다.")


if __name__ == "__main__":
    raise SystemExit(main())
