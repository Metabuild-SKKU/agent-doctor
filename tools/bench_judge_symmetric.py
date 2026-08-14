# -*- coding: utf-8 -*-
"""벤치마크 #2 대칭 재판정 (v2 수정 이력, 2026-08-14).

★ 수정 이력 — v1(bench_ragchecker_judge.py, 2026-08-13 사전 등록)과의 관계 ★
v1 은 RAGChecker 에 "코퍼스 평균 지표 15pt 이동"(관대), 우리에 "문항별 소견
>=3/6건"(엄격)을 요구해 잣대 높이가 달랐다. 데이터를 본 뒤의 변경이므로
v1 판정은 그대로 두고(judge_summary.json / ours_verdicts.json), 여기서는
**양쪽에 동일한 규칙**을 적용한 v2 를 별도 산출한다. 발표에는 v1 을 먼저
보여주고 v2 를 "수정 이력"으로 병기한다 — 사유: 잣대 비대칭 (노션 §8).

v2 공통 규칙 (양쪽 동일):
  결함 로그의 1차 지표 평균이 무결함(none) 대비 지정 방향으로 15pt 이상
  이동하면 "잡았다". 1차 지표는 서로 대응하는 축으로 짝지었다:
                     RAGChecker              우리(replay RAGAS)
    starve        →  claim_recall ↓          context recall ↓
    offtopic      →  context_precision ↓     context precision ↓
    hallucinate   →  hallucination ↑         faithfulness ↓
  none 건강 범위(오탐 검사)도 동일 구조: 검색 재현/정밀 >= 60, 환각축은
  RAGChecker hallucination <= 15 / 우리 faithfulness >= 85.
  원인 특정: 세 축 이탈 폭 중 최대 축 = 주입 축 (동률이면 "동률" 표기).

입력: bench/out/ragchecker/output_*.json (RAGChecker),
      bench/out/ragchecker/ours_*.log (replay 표준 출력 — RAGAS 평균 파싱)
산출: bench/out/ragchecker/judge_symmetric_v2.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "bench" / "out" / "ragchecker"

DELTA_PT = 15.0
DEFECTS = ["starve", "offtopic", "hallucinate"]

# (그룹, 지표, 방향) — 방향 -1: 낮아지면 결함 신호, +1: 높아지면 결함 신호
RC_AXES = {
    "starve": ("retriever", "claim_recall", -1),
    "offtopic": ("retriever", "context_precision", -1),
    "hallucinate": ("generator", "hallucination", +1),
}
OURS_AXES = {
    "starve": ("context_recall", -1),
    "offtopic": ("context_precision", -1),
    "hallucinate": ("faithfulness", -1),
}
# replay 로그의 출력 라인 → 지표 이름
OURS_LINE_MAP = {
    "faithfulness": re.compile(r"^- faithfulness\(환각 없음\)\s*:\s*([\d.]+|미측정)"),
    "context_precision": re.compile(r"^- context precision\(검색 정밀\):\s*([\d.]+|미측정)"),
    "context_recall": re.compile(r"^- context recall\(검색 재현\)\s*:\s*([\d.]+|미측정)"),
}


def load_rc(defect: str) -> dict:
    data = json.loads((OUT_DIR / f"output_{defect}.json").read_text(encoding="utf-8"))
    return data["metrics"]


def rc_get(metrics: dict, group: str, name: str) -> float:
    return float(metrics[f"{group}_metrics"][name])


def load_ours(defect: str) -> dict:
    """replay 로그에서 RAGAS 평균(0~1)을 파싱해 0~100 으로 돌려준다. 미측정은 None."""
    text = (OUT_DIR / f"ours_{defect}.log").read_text(encoding="utf-8")
    vals: dict = {}
    for line in text.splitlines():
        for name, rx in OURS_LINE_MAP.items():
            m = rx.match(line.strip())
            if m:
                vals[name] = None if m.group(1) == "미측정" else float(m.group(1)) * 100.0
    missing = [n for n in OURS_LINE_MAP if n not in vals]
    if missing:
        raise SystemExit(f"ours_{defect}.log 에서 지표 누락: {missing}")
    return vals


def judge_side(name: str, base: dict, get, axes: dict) -> dict:
    verdicts = []
    for defect in DEFECTS:
        cur = get(defect)
        deltas = {}
        for d2, ax in axes.items():
            sign = ax[-1]
            b, c = _ax(base, ax), _ax(cur, ax)
            deltas[d2] = None if (b is None or c is None) else sign * (c - b)
        primary = deltas[defect]
        caught = primary is not None and primary >= DELTA_PT
        measured = {k: v for k, v in deltas.items() if v is not None}
        top = max(measured, key=measured.get) if measured else None
        ties = [k for k, v in measured.items() if top and abs(v - measured[top]) < 1e-9]
        localized = None
        if caught:
            localized = "동률" if len(ties) > 1 and defect in ties else (defect == top)
        verdicts.append({
            "defect": defect,
            "delta_primary": None if primary is None else round(primary, 1),
            "deltas": {k: (None if v is None else round(v, 1)) for k, v in deltas.items()},
            "caught": caught,
            "localized": localized,
        })
    return {"side": name, "verdicts": verdicts}


def _ax(metrics: dict, ax: tuple):
    if len(ax) == 3:
        return rc_get(metrics, ax[0], ax[1])
    return metrics[ax[0]]


def main() -> None:
    global OUT_DIR
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help=f"입출력 디렉터리 (기본: {OUT_DIR})")
    args = ap.parse_args()
    if args.dir:
        OUT_DIR = Path(args.dir).resolve()

    rc_base = load_rc("none")
    ours_base = load_ours("none")

    rc = judge_side("ragchecker", rc_base, load_rc, RC_AXES)
    ours = judge_side("ours", ours_base, load_ours, OURS_AXES)

    # none 오탐 검사 (동일 구조)
    health = {
        "ragchecker": {
            "claim_recall>=60": rc_get(rc_base, "retriever", "claim_recall") >= 60,
            "context_precision>=60": rc_get(rc_base, "retriever", "context_precision") >= 60,
            "hallucination<=15": rc_get(rc_base, "generator", "hallucination") <= 15,
        },
        "ours": {
            "context_recall>=60": ours_base["context_recall"] >= 60,
            "context_precision>=60": ours_base["context_precision"] >= 60,
            "faithfulness>=85": ours_base["faithfulness"] >= 85,
        },
    }

    result = {
        "amendment": "v2 (2026-08-14) — v1 잣대 비대칭 수정. v1 결과는 judge_summary.json / ours_verdicts.json 에 보존.",
        "rule": f"1차 지표 평균이 none 대비 {DELTA_PT}pt 이상 지정 방향 이동 (양쪽 동일)",
        "none_health": health,
        "sides": [rc, ours],
    }
    out = OUT_DIR / "judge_symmetric_v2.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    for side in (rc, ours):
        print(f"=== {side['side']} ===")
        for v in side["verdicts"]:
            loc = {True: "특정 성공", False: "특정 실패", None: "", "동률": "특정 동률(검색축 내)"}[v["localized"]]
            print(f"  {v['defect']:12s} Δ{v['delta_primary']} → {'잡음' if v['caught'] else '놓침'} {loc}"
                  f"   (전축 Δ {v['deltas']})")
    print("none 오탐 검사:", json.dumps(health, ensure_ascii=False))
    print(f"→ {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
