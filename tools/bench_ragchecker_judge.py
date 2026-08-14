# -*- coding: utf-8 -*-
"""RAGChecker 결과 → 사전 등록된 판정 기준으로 "잡았다/놓쳤다" 채점.

★ 사전 등록 (2026-08-13, RAGChecker 실행 전에 작성·고정) ★
이 파일의 문턱값은 RAGChecker 를 한 번도 실행하기 전에 정해 노션
벤치마킹 §8 에 기록했다. 결과를 본 뒤 문턱을 바꾸면 p-해킹이다 —
수정하려면 노션에 사유와 함께 수정 이력을 남겨야 한다.

판정 규칙 (지표는 0~100 스케일, RAGChecker 기본 출력 단위):
  결함 로그의 1차 지표가 ext_none(무결함 대조군) 대비 지정 방향으로
  15pt 이상 벗어나면 "잡았다".
    starve      → retriever.claim_recall      ≤ none − 15
    offtopic    → retriever.context_precision ≤ none − 15
    hallucinate → generator.hallucination     ≥ none + 15
  Δ=15pt 근거: 주입 결함은 극단적(top_k=1, 엉뚱한 질문 검색, 기권 금지)
  이라 진짜 신호는 이보다 훨씬 크게 나와야 정상이고, LLM 심판의
  실행 간 요동(±1~2pt 관측, §13-1)보다는 충분히 크다.

  ext_none 오탐 검사 (절대 건강 범위):
    claim_recall ≥ 60, context_precision ≥ 60, hallucination ≤ 15.
  하나라도 벗어나면 → 해당 축 오탐(또는 기준선 불능)으로 기록한다.

  원인 특정(localization): 잡힌 로그에서 세 축의 이탈 폭(Δpt)을 모두
  구해, 최대 이탈 축이 주입한 원인 축과 일치해야 "특정 성공".
  (예: starve 에서 hallucination 이 claim_recall 보다 크게 움직이면
   "이상은 알았지만 원인은 헛짚음"으로 기록.)

우리 진단기 쪽 판정(같은 시점에 사전 등록, 노션 §8):
  해당 ext_ 라벨 소견이 6문항 중 ≥3건이면 "잡았다".
  hallucinate 는 gold_contexts 가 없어 예비(confirmed=False) 판정까지
  인정하되 표기한다. ext_none 은 critical confirmed 소견 0건이어야 통과.

사용 (RAGChecker 실행이 끝난 뒤):
    python tools/bench_ragchecker_judge.py
입력: bench/out/ragchecker/output_{none,starve,hallucinate,offtopic}.json
      (ragchecker 가 metrics 를 채워 저장한 파일)
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "bench" / "out" / "ragchecker"

DELTA_PT = 15.0  # 사전 등록 — 수정 금지(수정 시 노션에 이력 필수)

# 결함 → (지표 그룹, 지표 이름, 방향). 방향 -1 = 낮아져야 잡은 것, +1 = 높아져야.
PRIMARY = {
    "starve": ("retriever", "claim_recall", -1),
    "offtopic": ("retriever", "context_precision", -1),
    "hallucinate": ("generator", "hallucination", +1),
}

# ext_none 절대 건강 범위 (벗어나면 오탐/기준선 불능)
NONE_HEALTHY = {
    ("retriever", "claim_recall"): (">=", 60.0),
    ("retriever", "context_precision"): (">=", 60.0),
    ("generator", "hallucination"): ("<=", 15.0),
}


def _load_metrics(defect: str) -> dict:
    path = OUT_DIR / f"output_{defect}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics") or {}
    if not metrics:
        raise SystemExit(f"{path.name}: metrics 없음 — ragchecker 실행이 안 됐거나 파일이 잘못됨")
    return metrics


def _get(metrics: dict, group: str, name: str) -> float:
    # ragchecker 출력 키는 "retriever_metrics" / "generator_metrics" / "overall_metrics"
    for key in (f"{group}_metrics", group):
        if key in metrics and name in metrics[key]:
            val = float(metrics[key][name])
            return val * 100.0 if val <= 1.5 else val  # 0~1 스케일 방어적 정규화
    raise SystemExit(f"지표 {group}.{name} 를 결과에서 찾지 못함 — 키 이름 확인 필요")


def main() -> None:
    global OUT_DIR
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help=f"입출력 디렉터리 (기본: {OUT_DIR})")
    args = ap.parse_args()
    if args.dir:
        OUT_DIR = Path(args.dir).resolve()

    base = _load_metrics("none")

    print("=== ext_none 오탐 검사 (절대 건강 범위) ===")
    baseline_ok = True
    for (group, name), (op, limit) in NONE_HEALTHY.items():
        val = _get(base, group, name)
        ok = (val >= limit) if op == ">=" else (val <= limit)
        baseline_ok &= ok
        print(f"  {group}.{name} = {val:.1f}  (기준 {op} {limit})  {'통과' if ok else '★ 오탐/기준선 불능'}")

    print(f"\n=== 결함 판정 (Δ={DELTA_PT}pt, ext_none 대비) ===")
    table = []
    for defect, (group, name, sign) in PRIMARY.items():
        m = _load_metrics(defect)
        deltas = {}  # 세 축 전부의 이탈 폭 — 원인 특정 판정용
        for d2, (g2, n2, s2) in PRIMARY.items():
            deltas[d2] = s2 * (_get(m, g2, n2) - _get(base, g2, n2))
        primary_delta = deltas[defect]
        caught = primary_delta >= DELTA_PT
        localized = caught and defect == max(deltas, key=deltas.get)
        val, ref = _get(m, group, name), _get(base, group, name)
        table.append((defect, f"{group}.{name}", val, ref, primary_delta, caught, localized))
        print(f"  {defect:12s} {group}.{name} = {val:.1f} (none {ref:.1f}, Δ{primary_delta:+.1f}) "
              f"→ {'잡음' if caught else '놓침'}"
              + (f" · 특정 {'성공' if localized else '실패(최대 이탈 축=' + max(deltas, key=deltas.get) + ')'}" if caught else ""))

    summary = {
        "preregistered": {"delta_pt": DELTA_PT, "none_healthy": {f"{g}.{n}": f"{op} {v}" for (g, n), (op, v) in NONE_HEALTHY.items()}},
        "baseline_ok": baseline_ok,
        "verdicts": [
            {"defect": d, "metric": mname, "value": round(v, 1), "none": round(r, 1),
             "delta": round(dl, 1), "caught": c, "localized": lz}
            for d, mname, v, r, dl, c, lz in table
        ],
    }
    out = OUT_DIR / "judge_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
