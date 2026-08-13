"""
tools/bench_compare.py
벤치마크 채점·비교 — 여러 triad 로그를 **같은 골든셋·같은 replay 경로**로 재고 표로 낸다.

왜 도구로 만드나: CLI(agents.eval.replay)는 사람이 읽는 출력이라 비교표를 만들려면 매번
파싱해야 하고, 실행마다 RAGAS 를 다시 호출한다. 여기서는 진단을 한 번씩만 돌려 점수·소견을
JSON 으로 남긴다 — 발표 자료(막대 그래프·소견 대조표)가 이 파일 하나에서 나온다.

공정성의 근거가 이 파일에 있다: 모든 로그가 같은 함수(diagnose_external_log)에 같은
골든셋으로 들어간다. 시스템마다 다른 채점 경로를 쓰면 "점수 차이가 시스템 차이냐 측정
차이냐"를 구분할 수 없다.

사용법:
    python -m tools.bench_compare \
        autorag=bench/out/autorag.jsonl \
        ours_baseline=bench/out/ours_baseline.jsonl \
        [ours_tuned=bench/out/ours_tuned.jsonl]
"""
from __future__ import annotations

import json
import os
import sys

DEFAULT_GOLDEN = "bench/golden_test_50.json"
DEFAULT_OUT = "bench/out/comparison.json"


def score_one(label: str, path: str, golden: str) -> dict:
    from agents.eval.replay import diagnose_external_log

    print(f"\n=== {label} · {path} ===")
    report, cap, errors = diagnose_external_log(path, golden_path=golden)
    if report is None:
        return {"label": label, "path": path, "error": "진단 불가", "tier": cap.get("tier")}

    scores = report.ragas_scores or {}
    composite = report.composite_score or {}
    # 소견은 라벨별 확정/예비 건수로 압축한다 — 표에 그대로 들어가는 형태.
    findings: dict = {}
    for f in report.findings:
        slot = findings.setdefault(f.label, {"severity": f.severity, "확정": 0, "예비": 0})
        slot["확정" if f.confirmed else "예비"] += 1

    return {
        "label": label,
        "path": path,
        "records": cap.get("diagnosed") or cap.get("records"),
        "tier": cap.get("tier"),
        "composite_total": composite.get("total"),
        "components": {c["label"]: c["score"] for c in composite.get("components", [])},
        "overall_score": report.overall_score,
        "metrics": {
            "faithfulness": scores.get("faithfulness"),
            "answer_relevancy": scores.get("response_relevancy"),
            "context_precision": scores.get("context_precision"),
            "context_recall": scores.get("context_recall"),
            "rule_char_f1": scores.get("mean_f1"),
        },
        "findings": findings,
    }


def _fmt(v, width: int = 6) -> str:
    return f"{v:.3f}".rjust(width) if isinstance(v, (int, float)) else "미측정".rjust(width)


def print_table(results: list[dict]) -> None:
    ok = [r for r in results if "error" not in r]
    if not ok:
        return
    labels = [r["label"] for r in ok]
    w = max(len(x) for x in labels) + 2

    print("\n" + "=" * (16 + w * len(ok)))
    print("벤치마크 비교 — 동일 골든셋 · 동일 replay 경로")
    print("=" * (16 + w * len(ok)))

    def row(name: str, values: list) -> None:
        print(f"{name:<22}" + "".join(_fmt(v, w).rjust(w) for v in values))

    print(f"{'':<22}" + "".join(x.rjust(w) for x in labels))
    print("-" * (22 + w * len(ok)))
    print(f"{'종합점수 (0~100)':<22}" + "".join(
        (str(r["composite_total"]) if r["composite_total"] is not None else "-").rjust(w) for r in ok))
    print(f"{'  품질':<22}" + "".join(
        str(r["components"].get("품질", "-")).rjust(w) for r in ok))
    print(f"{'  신뢰도':<22}" + "".join(
        str(r["components"].get("신뢰도", "-")).rjust(w) for r in ok))
    print("-" * (22 + w * len(ok)))
    row("overall_score", [r["overall_score"] for r in ok])
    for key, name in (("faithfulness", "환각 없음"),
                      ("answer_relevancy", "동문서답 없음"),
                      ("context_precision", "검색 정밀"),
                      ("context_recall", "검색 재현"),
                      ("rule_char_f1", "규칙 char F1")):
        row(name, [r["metrics"][key] for r in ok])

    # 소견 대조 — 점수만으로는 "왜 그런지"가 안 보인다. 이 표가 발표의 핵심이다.
    all_labels = sorted({k for r in ok for k in r["findings"]})
    if all_labels:
        print("-" * (22 + w * len(ok)))
        print("소견(확정 건수)")
        for lab in all_labels:
            cells = [str(r["findings"].get(lab, {}).get("확정", 0)).rjust(w) for r in ok]
            print(f"  {lab:<20}" + "".join(cells))
    print("=" * (22 + w * len(ok)))


def main(argv: list[str]) -> int:
    from core.console import force_utf8_stdio
    force_utf8_stdio()
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass

    # 리플레이는 RAGAS 가 유일한 작업이라 끄면 생성축 라벨이 통째로 죽는다(replay.py 주석).
    os.environ["EVAL_ENABLE_LLM"] = "1"
    if os.getenv("EVAL_MODE", "").strip().lower() not in ("deep", "full", "3", "4"):
        os.environ["EVAL_MODE"] = "deep"

    golden = DEFAULT_GOLDEN
    out = DEFAULT_OUT
    pairs = []
    for arg in argv:
        if arg.startswith("--golden="):
            golden = arg.split("=", 1)[1]
        elif arg.startswith("--out="):
            out = arg.split("=", 1)[1]
        elif "=" in arg:
            label, path = arg.split("=", 1)
            pairs.append((label, path))
    if not pairs:
        print(__doc__)
        return 2

    missing = [p for _, p in pairs if not os.path.exists(p)]
    if missing:
        print("[compare] 로그 파일이 없습니다: " + ", ".join(missing))
        return 2

    results = [score_one(label, path, golden) for label, path in pairs]
    print_table(results)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"golden": golden, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n[compare] → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
