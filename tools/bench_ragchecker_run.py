# -*- coding: utf-8 -*-
"""RAGChecker 채점 실행기 — input_{defect}.json → output_{defect}.json.

.venv-ragchecker 의 파이썬으로 실행해야 한다 (본체 .venv 에는 ragchecker 없음):
    .venv-ragchecker/Scripts/python.exe -X utf8 tools/bench_ragchecker_run.py [--defect none]

extractor/checker LLM 은 litellm 경유 OpenRouter(gpt-4o-mini)를 쓴다 —
API 키는 리포지토리 표준대로 .env 의 OPENROUTER_API_KEY 를 dotenv 로 읽는다.
판정 문턱은 여기 없다 — 채점 결과 해석은 bench_ragchecker_judge.py (사전 등록).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "bench" / "out" / "ragchecker"
DEFECTS = ["none", "starve", "hallucinate", "offtopic"]

# 같은 모델을 추출기/검사기 양쪽에 쓴다. gpt-4o-mini: 저렴 + AutoRAG 벤치(§13)의
# generator 와 동일 계열이라 심판 품질 시비를 줄인다.
MODEL = "openrouter/openai/gpt-4o-mini"


def main() -> None:
    global OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--defect", choices=DEFECTS, help="하나만 채점 (기본: 전부)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dir", default=None, help=f"입출력 디렉터리 (기본: {OUT_DIR})")
    args = ap.parse_args()
    if args.dir:
        OUT_DIR = Path(args.dir).resolve()

    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(".env 에 OPENROUTER_API_KEY 가 없다")

    # claude-sonnet-5 등 reasoning 모델은 temperature=1 만 받는데 refchecker 가
    # temperature=1e-05 를 하드코딩한다 — litellm 이 미지원 파라미터를 버리게 한다.
    import litellm
    litellm.drop_params = True

    from ragchecker import RAGResults, RAGChecker  # noqa: E402 — venv 확인 후 임포트

    evaluator = RAGChecker(
        extractor_name=args.model,
        checker_name=args.model,
        batch_size_extractor=8,
        batch_size_checker=8,
    )
    for defect in ([args.defect] if args.defect else DEFECTS):
        src = OUT_DIR / f"input_{defect}.json"
        dst = OUT_DIR / f"output_{defect}.json"
        rag_results = RAGResults.from_json(src.read_text(encoding="utf-8"))
        print(f"=== {defect}: {len(rag_results.results)}건 채점 (모델 {args.model}) ===", flush=True)
        evaluator.evaluate(rag_results, "all_metrics")
        dst.write_text(rag_results.to_json(indent=2), encoding="utf-8")
        print(f"→ {dst.relative_to(REPO_ROOT)}")
        print(rag_results.metrics)


if __name__ == "__main__":
    main()
