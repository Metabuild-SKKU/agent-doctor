"""
tools/bench_question_table.py
문항별 전수 대조표 — 시스템들이 같은 질문을 각각 어떻게 처리했는지 한 줄에 놓는다.

왜 필요한가: 진단서(report.json)와 실행 로그는 **실패한 문항만** 남긴다
(run_replay_report._log_questions 는 report.failed_questions 만 돈다). "왜 저 점수였나"를
되짚기엔 그걸로 충분하지만, **두 시스템을 비교**하려면 성공한 문항도 있어야 한다 —
"A는 맞히고 B는 틀린 문항"이 발표에서 가장 쓸모 있는 사례인데, 실패 목록만으로는
한쪽만 보이기 때문이다.

재료를 합친다:
  · 골든셋      질문 · 정답 (채점 기준)
  · triad 로그  각 시스템의 실제 답변 · 검색 컨텍스트
  · report.json 실패 문항과 그 라벨 (여기 없는 문항 = 통과)

산출:
  bench/out/question_table.json  전 문항 × 전 시스템 (기계용)
  bench/out/question_table.md    사람이 읽는 표 + 갈린 문항 하이라이트

사용법:
    python -m tools.bench_question_table \
        "AutoRAG+리랭커=bench/out/report_autorag_rr" \
        "우리+리랭커=bench/out/report_ours_rr"

  값은 진단서 폴더 경로다(report.json 과 *.jsonl 이 그 안에 있어야 한다).
  --out=bench/out/question_table_li  로 짝마다 다른 파일에 쓴다(확장자는 붙지 않는다).
"""
from __future__ import annotations

import json
import os
import sys
import glob

DEFAULT_GOLDEN = "bench/golden_test_50.json"
OUT_JSON = "bench/out/question_table.json"
OUT_MD = "bench/out/question_table.md"


def _norm(text: str) -> str:
    """질문 매칭용 정규화. 공백·문장부호 차이로 못 붙는 일을 막는다
    (qa_merge.normalize_question 과 같은 취지 — 여기서는 의존을 줄이려 최소만)."""
    return "".join(str(text or "").split())


def load_system(label: str, folder: str) -> dict:
    """진단서 폴더 → {질문: {answer, contexts, label, diagnosis}}."""
    report_path = os.path.join(folder, "report.json")
    if not os.path.exists(report_path):
        sys.exit(f"[table] report.json 이 없습니다: {report_path}")
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    # 실패 문항만 들어 있다 — 여기 없으면 통과로 본다.
    failed = {_norm(q.get("q")): q for q in report.get("qas") or []}

    logs = [p for p in glob.glob(os.path.join(folder, "*.jsonl"))]
    if len(logs) != 1:
        sys.exit(f"[table] {folder} 에 triad 로그(.jsonl)가 {len(logs)}개입니다 — 1개여야 합니다.")
    rows: dict[str, dict] = {}
    with open(logs[0], encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = _norm(rec["question"])
            fail = failed.get(key)
            rows[key] = {
                "answer": rec.get("answer", ""),
                "n_ctx": len(rec.get("contexts") or []),
                "passed": fail is None,
                "label": (fail or {}).get("label", ""),
                "diagnosis": (fail or {}).get("diagnosis", ""),
            }
    return {"label": label, "folder": folder, "log": os.path.basename(logs[0]), "rows": rows}


def main(argv: list[str]) -> int:
    from core.console import force_utf8_stdio
    force_utf8_stdio()

    # 산출 경로를 바꿀 수 있어야 한다. 기본 경로로만 쓰면 비교 짝을 바꿔 돌릴 때마다
    # 앞서 만든 표(발표에 쓰는 커밋된 산출물)를 말없이 덮어쓴다.
    global OUT_JSON, OUT_MD
    golden_path = DEFAULT_GOLDEN
    pairs = []
    for arg in argv:
        if arg.startswith("--golden="):
            golden_path = arg.split("=", 1)[1]
        elif arg.startswith("--out="):
            stem = arg.split("=", 1)[1]
            stem = stem[:-5] if stem.endswith(".json") else stem
            OUT_JSON, OUT_MD = f"{stem}.json", f"{stem}.md"
        elif "=" in arg:
            label, folder = arg.split("=", 1)
            pairs.append((label, folder))
    if len(pairs) < 2:
        print(__doc__)
        return 2

    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)
    systems = [load_system(label, folder) for label, folder in pairs]

    table = []
    for item in golden:
        key = _norm(item["question"])
        row = {"question": item["question"], "gold": item.get("ground_truth", ""), "systems": {}}
        for s in systems:
            cell = s["rows"].get(key)
            if cell is None:      # 로그에 없는 질문 = 그 시스템이 답하지 않은 문항
                cell = {"answer": "", "n_ctx": 0, "passed": None,
                        "label": "(로그에 없음)", "diagnosis": ""}
            row["systems"][s["label"]] = cell
        table.append(row)

    labels = [s["label"] for s in systems]
    os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"golden": golden_path, "systems": [
            {k: s[k] for k in ("label", "folder", "log")} for s in systems],
            "questions": table}, f, ensure_ascii=False, indent=2)

    # 요약 — 통과 수와 "갈린 문항"이 핵심이다. 둘 다 맞히거나 둘 다 틀린 문항은
    # 비교에 정보가 없고, 갈린 문항이 곧 발표 사례 후보다.
    print(f"\n골든셋 {len(table)}문항 · 시스템 {len(systems)}개\n")
    for lab in labels:
        ok = sum(1 for r in table if r["systems"][lab]["passed"])
        print(f"  {lab:24} 통과 {ok:2}/{len(table)}")

    split = [r for r in table
             if len({bool(r["systems"][l]["passed"]) for l in labels}) > 1]
    print(f"\n갈린 문항: {len(split)}개 (한쪽만 통과)")

    # 시스템별 전 문항 상세 — replay_run.log 는 실패만 찍으므로(run_replay_report.
    # _log_questions 가 failed_questions 만 돈다) 성공까지 포함한 판본을 따로 남긴다.
    # 두 시스템을 문항 단위로 비교하려면 "A는 맞히고 B는 틀린" 문항이 필요한데,
    # 실패 목록만으로는 한쪽만 보이기 때문이다. 형식은 replay_run.log 와 맞춘다.
    for s in systems:
        detail = [f"─── 문항별 결과 (전 {len(table)}건) ─ {s['label']}",
                  f"    로그 {s['log']} · 골든셋 {os.path.basename(golden_path)}", ""]
        for i, r in enumerate(table, 1):
            c = r["systems"][s["label"]]
            mark = "✅" if c["passed"] else f"❌ {c['label']}"
            detail.append(f"  [{i}/{len(table)}] {mark}")
            detail.append(f"    Q: {r['question']}")
            detail.append(f"    기대: {r['gold']}")
            detail.append(f"    실제: {c['answer']}")
            detail.append(f"    컨텍스트: {c['n_ctx']}개")
            if c["diagnosis"]:
                detail.append(f"    → {c['diagnosis']}")
            detail.append("")
        ok = sum(1 for r in table if r["systems"][s["label"]]["passed"])
        detail.append(f"─── 합계: 통과 {ok} / 실패 {len(table) - ok} / 전체 {len(table)}")
        path = os.path.join(s["folder"], "questions_all.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(detail) + "\n")
        print(f"  → {path}")

    lines = ["# 문항별 전수 대조표", "",
             f"골든셋 `{golden_path}` · {len(table)}문항", ""]
    for lab, s in zip(labels, systems):
        ok = sum(1 for r in table if r["systems"][lab]["passed"])
        lines.append(f"- **{lab}** — 통과 {ok}/{len(table)} · 로그 `{s['log']}`")
    lines += ["", f"## 갈린 문항 {len(split)}개 (발표 사례 후보)", ""]
    for i, r in enumerate(split, 1):
        lines.append(f"### {i}. {r['question']}")
        lines.append(f"- **정답**: {r['gold']}")
        for lab in labels:
            c = r["systems"][lab]
            mark = "✅" if c["passed"] else "❌"
            lines.append(f"- {mark} **{lab}** (ctx {c['n_ctx']}): {c['answer'][:150]}")
            if c["diagnosis"]:
                lines.append(f"  - 진단: {c['label']} — {c['diagnosis']}")
        lines.append("")
    lines += ["## 전 문항", "", "| # | 질문 | 정답 | " + " | ".join(labels) + " |",
              "|---|---|---|" + "---|" * len(labels)]
    for i, r in enumerate(table, 1):
        cells = []
        for lab in labels:
            c = r["systems"][lab]
            cells.append("✅" if c["passed"] else f"❌ {c['label']}")
        q = r["question"][:44]
        g = str(r["gold"])[:26]
        lines.append(f"| {i} | {q} | {g} | " + " | ".join(cells) + " |")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  → {OUT_JSON}\n  → {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
