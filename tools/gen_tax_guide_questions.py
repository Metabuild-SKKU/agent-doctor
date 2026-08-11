"""
tools/gen_tax_guide_questions.py
data/pdf_corpus.json 본문을 잘라 LLM으로 골든 QA(question/ground_truth/gold_contexts)를
목표 개수만큼 뽑아 tools/tax_guide_questions.json 에 저장한다.

100개를 손으로 쓰기엔 비현실적이라 자동화한다. 대신 "지어낸 정답"이 섞이면 골든셋
자체가 틀린 채점 기준이 되므로, gold_context는 반드시 원문에서 그대로 발췌하게 하고
LLM 이 낸 gold_context가 실제로 그 청크의 부분 문자열인지 검증해서 통과한 것만 쓴다
(make_external_rag.py 의 인라인 골든셋과 같은 스키마 — 그대로 재사용 가능).

사용법:
    python -m tools.gen_tax_guide_questions --target=100
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from core.llm_clients import openai_chat, OPENROUTER_BASE_URL  # noqa: E402

DEFAULT_CORPUS = "data/pdf_corpus.json"
DEFAULT_OUT = "tools/tax_guide_questions.json"
# 표지·목차는 사실 진술이 아니라 QA로 뽑기 부적합 — 본문 시작 지점부터 사용한다
# (make_external_rag.py 검증 중 확인한 실측 오프셋: TOC가 대략 4만자까지).
SKIP_PREFIX_CHARS = 40000
CHUNK_CHARS = 1400
GEN_MODEL = os.getenv("EXT_RAG_QA_GEN_MODEL", "openai/gpt-4o-mini")

SYSTEM = (
    "당신은 세금 안내 책자에서 사실 확인 질문(QA)을 만드는 도구입니다. "
    "반드시 주어진 본문에 명시적으로 나온 사실만 사용하세요. 지어내지 마세요."
)

PROMPT_TMPL = """아래는 국세청 세금가이드 책자의 한 부분입니다. 이 본문에서 확인 가능한
구체적 사실(숫자·기한·요건·%) 하나를 골라 질문 1개를 만드세요.

요구사항:
- question: 본문 내용을 아는 사람이 답할 수 있는 명확한 질문
- ground_truth: 간결한 정답 (본문에 있는 표현 그대로, 숫자/단위 포함)
- gold_context: 정답의 근거가 되는 본문 속 문장을 "원문 그대로" 발췌 (요약/수정 금지, 부분 문자열이어야 함)

본문에 위 조건을 만족하는 사실이 없으면 {{"skip": true}} 만 반환하세요.

본문:
---
{chunk}
---

JSON으로만 답하세요: {{"question": "...", "ground_truth": "...", "gold_context": "..."}}
"""


def load_chunks(path: str, size: int, skip: int) -> list[str]:
    with open(path, encoding="utf-8") as f:
        docs = json.load(f)
    chunks: list[str] = []
    for doc in docs:
        text = doc["text"][skip:]
        for i in range(0, len(text), size):
            piece = text[i:i + size].strip()
            if len(piece) > 200:
                chunks.append(piece)
    return chunks


def gen_one(chunk: str, api_key: str) -> dict | None:
    raw = openai_chat(
        SYSTEM, PROMPT_TMPL.format(chunk=chunk), GEN_MODEL,
        json_mode=True, api_key=api_key, base_url=OPENROUTER_BASE_URL,
        max_output_tokens=500, tag="qa-gen",
    )
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if obj.get("skip"):
        return None
    q, gt, gc = obj.get("question"), obj.get("ground_truth"), obj.get("gold_context")
    if not (q and gt and gc):
        return None
    if gc.strip() not in chunk:
        return None  # 원문에 없는 인용 — 검증 실패, 버린다
    return {"question": q.strip(), "ground_truth": gt.strip(), "gold_contexts": [gc.strip()]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--target", type=int, default=100)
    args = ap.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("[gen-qa] OPENROUTER_API_KEY 가 없습니다 — .env 를 확인하세요.")

    chunks = load_chunks(args.corpus, CHUNK_CHARS, SKIP_PREFIX_CHARS)
    print(f"[gen-qa] 후보 청크 {len(chunks)}개, 목표 {args.target}개")

    results: list[dict] = []
    seen_q: set[str] = set()
    for i, chunk in enumerate(chunks):
        if len(results) >= args.target:
            break
        item = gen_one(chunk, api_key)
        if item and item["question"] not in seen_q:
            seen_q.add(item["question"])
            results.append(item)
            print(f"  [{len(results)}/{args.target}] {item['question'][:50]}")
        if (i + 1) % 20 == 0:
            print(f"  ({i + 1}/{len(chunks)}청크 처리, 유효 {len(results)}건)")

    if len(results) < args.target:
        print(f"[gen-qa] 경고: 청크를 모두 소진해 {len(results)}건만 확보했습니다"
              f" (목표 {args.target})")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[gen-qa] {len(results)}건 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
