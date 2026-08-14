# -*- coding: utf-8 -*-
"""결함 주입 로그(triad JSONL) → RAGChecker 입력 JSON 변환.

벤치마크 #2 (노션 벤치마킹 §8): 원인을 아는 로그 4종
(none / starve / hallucinate / offtopic)을 RAGChecker(amazon-science)와
우리 replay 진단기에 똑같이 먹여 "원인을 누가 맞히나"를 비교한다.

이 스크립트는 LLM을 호출하지 않는다 — 포맷 변환만 한다.
실제 채점(ragchecker 실행, extractor/checker LLM 필요)은 별도 단계다.

RAGChecker 입력 스키마 (ragchecker.container.RAGResults):
    {"results": [{"query_id": str, "query": str, "gt_answer": str,
                  "response": str,
                  "retrieved_context": [{"doc_id": str, "text": str}]}]}

hallucinate 로그는 설계상 코퍼스에 답이 없는 질문(GAP_QUESTIONS)이라
ground_truth 가 없다. RAGChecker 는 gt_answer 가 필수이므로, 이 로그의
정답은 기권 문장(ABSTAIN_GT)으로 고정한다. 이는 "정답 없는 질문의 올바른
행동은 기권"이라는 사전 등록된 설계 결정이다 — 실행 결과를 보고 바꾸지 않는다.

사용:
    python tools/bench_ragchecker_convert.py            # 4종 전부
    python tools/bench_ragchecker_convert.py --defect starve
산출: bench/out/ragchecker/input_{defect}.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "external_rag"
OUT_DIR = REPO_ROOT / "bench" / "out" / "ragchecker"

DEFECTS = ["none", "starve", "hallucinate", "offtopic"]

# hallucinate 전용 정답(기권 문장) — 사전 등록, 결과 확인 후 수정 금지.
ABSTAIN_GT = "제공된 자료에서 확인할 수 없습니다."


def convert_one(defect: str) -> dict:
    src = Path(FIXTURE_DIR) / f"ext_{defect}.jsonl"
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]

    results = []
    missing_gt = 0
    for i, row in enumerate(rows):
        gt = row.get("ground_truth")
        if not gt:
            missing_gt += 1
            gt = ABSTAIN_GT
        results.append({
            "query_id": f"{defect}_{i:03d}",
            "query": row["question"],
            "gt_answer": gt,
            "response": row["answer"],
            "retrieved_context": [
                {"doc_id": ctx.get("chunk_id", f"{defect}_{i:03d}_ctx{ctx.get('rank', j)}"),
                 "text": ctx["text"]}
                for j, ctx in enumerate(row.get("contexts", []))
            ],
        })

    if defect != "hallucinate" and missing_gt:
        raise SystemExit(
            f"ext_{defect}: ground_truth 결손 {missing_gt}건 — hallucinate 외에는 결손이 없어야 한다")

    return {"results": results, "_missing_gt_filled_with_abstain": missing_gt}


def main() -> None:
    global FIXTURE_DIR, OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--defect", choices=DEFECTS, help="하나만 변환 (기본: 전부)")
    ap.add_argument("--src-dir", default=None, help=f"fixture 디렉터리 (기본: {FIXTURE_DIR})")
    ap.add_argument("--out-dir", default=None, help=f"출력 디렉터리 (기본: {OUT_DIR})")
    args = ap.parse_args()
    if args.src_dir:
        FIXTURE_DIR = Path(args.src_dir).resolve()
    if args.out_dir:
        OUT_DIR = Path(args.out_dir).resolve()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for defect in ([args.defect] if args.defect else DEFECTS):
        payload = convert_one(defect)
        filled = payload.pop("_missing_gt_filled_with_abstain")
        out = OUT_DIR / f"input_{defect}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        note = f" (gt_answer 기권 문장 대체 {filled}건)" if filled else ""
        print(f"{out.relative_to(REPO_ROOT)}: {len(payload['results'])}건{note}")


if __name__ == "__main__":
    main()
