"""
tools/make_our_rag.py
우리 RAG 실행 로그 덤퍼 — tools/make_external_rag.py 의 **대칭짝**.

왜 필요한가: 벤치마크에서 두 시스템을 같은 자로 재려면 우리 결과도 외부 로그와 같은
형식(스키마 v1 triad)으로 나와야 한다. AutoRAG 나 LlamaIndex 의 RAG 를 우리 Eval
STEP2 안에 넣을 방법이 구조적으로 없기 때문에(STEP2 는 자기가 검색·생성한다), 크로스
도구 비교는 replay 경로가 유일한 공통분모다. 그 경로에 우리 쪽을 태우는 입구가 이 파일이다.

make_external_rag.py 와 **일부러 같은 모양**으로 짰다 — 같은 CLI, 같은 출력 스키마.
두 파일을 나란히 놓았을 때 "질문도 형식도 같고 다른 건 스택뿐"이 눈으로 보여야
비교가 공정하다는 주장이 선다.

기본 파이프라인을 대체하지 않는다: 우리 RAG 를 만들고 최적화하는 건 여전히
Ingest→Index→Eval→Optimize 루프다. 이 도구는 그렇게 만들어진 설정으로 test 문항에
답해 로그를 남기는, 채점 직전 단계일 뿐이다.

생성 LLM 을 고정하는 이유: 모델이 다르면 시스템이 아니라 **모델을 비교**하게 된다.
AutoRAG 쪽 config 와 같은 값(openai/gpt-4o-mini, OpenRouter)으로 맞춘다.

사용법:
    python -m tools.make_our_rag --stage=baseline
    python -m tools.make_our_rag --stage=tuned --index-config=bench/out/tuned_index_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

DEFAULT_CORPUS = "data/pdf_corpus.json"
DEFAULT_QUESTIONS = "bench/golden_test_50.json"
DEFAULT_OUTDIR = "bench/out"

# 양쪽 시스템이 같은 모델을 쓰게 고정한다(bench/autorag_config.yaml 과 같은 값).
# 환경에 이미 값이 있으면 존중하지 않고 덮는다 — 벤치마크는 재현이 우선이고,
# .env 가 사람마다 다르면 같은 명령이 다른 결과를 낸다.
BENCH_LLM_PROVIDER = "openrouter"
BENCH_LLM_MODEL = "openai/gpt-4o-mini"


def load_corpus(path: str) -> list[dict]:
    if not os.path.exists(path):
        sys.exit(f"[our-rag] 코퍼스가 없습니다: {path}\n"
                 f"  data/ 는 gitignore 대상입니다 — PDF 에서 먼저 만드세요:\n"
                 f"  python -m tools.pdf_to_corpus bench/<문서>.pdf")
    with open(path, encoding="utf-8") as f:
        docs = json.load(f)
    if not isinstance(docs, list) or not docs:
        sys.exit(f"[our-rag] 코퍼스가 비었거나 리스트가 아닙니다: {path}")
    return docs


def load_questions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return [q for q in items if str(q.get("question") or "").strip()]


def load_index_config(stage: str, path: str | None) -> dict:
    """stage 에 맞는 index_config 를 만든다.

    tuned 인데 설정 파일이 없으면 **거부한다**. 조용히 기본값으로 떨어지면 baseline 과
    똑같은 로그가 나오는데, 파일만 봐서는 그 사실을 알 수 없다 — 0회차와 최종이 같은
    점수로 찍혀도 "튜닝 효과가 없었다"로 오독하게 된다."""
    from core.state import AgentDoctorState

    config = dict(AgentDoctorState().index_config)   # 0회차 = core/state.py 기본값
    if stage == "baseline":
        if path:
            sys.exit("[our-rag] --stage=baseline 에는 --index-config 를 줄 수 없습니다"
                     " (0회차는 정의상 기본 설정입니다).")
        return config
    if not path:
        sys.exit("[our-rag] --stage=tuned 에는 --index-config 가 필요합니다.\n"
                 "  파이프라인이 최적화를 끝낸 뒤 state.index_config 를 JSON 으로 떨궈 주세요.\n"
                 "  (없으면 기본값으로 돌아 baseline 과 같은 로그가 나옵니다 — 그래서 거부합니다.)")
    with open(path, encoding="utf-8") as f:
        tuned = json.load(f)
    if not isinstance(tuned, dict) or not tuned:
        sys.exit(f"[our-rag] index_config 가 비었거나 dict 가 아닙니다: {path}")
    config.update(tuned)
    return config


def build_pipeline(docs: list[dict], config: dict):
    """우리 Index 에이전트로 색인하고 공용 Retriever 를 만든다."""
    from core.schema import Document
    from core.state import AgentDoctorState
    from agents.index.agent import run as index_run
    from agents.rag.retriever import build_retriever

    state = AgentDoctorState()
    state.index_config = dict(config)
    state.documents = [
        Document(doc_id=d["id"], source=d.get("source", ""), format="pdf", content=d["text"])
        for d in docs
    ]
    state = index_run(state)
    if state.status == "error":
        sys.exit(f"[our-rag] 색인 실패: {state.error}")
    print(f"[our-rag] 청크 {len(state.chunks)}개")
    return build_retriever(state.chunks, config=state.index_config), state


def run(questions: list[dict], retriever, config: dict) -> list[dict]:
    """질문을 돌려 스키마 v1 로그 레코드를 만든다(make_external_rag.run 과 같은 형태)."""
    from agents.rag.generator import generate_answer

    top_k = int(config.get("top_k", 5) or 5)
    records = []
    for i, q in enumerate(questions, 1):
        question = q["question"]
        t0 = time.time()
        hits = retriever.search(question, top_k=top_k)
        contexts = [{
            "text": h.get("text", ""),
            "chunk_id": h.get("chunk_id"),
            "score": float(h["score"]) if h.get("score") is not None else None,
            "rank": rank,
            "source_doc": h.get("doc_id"),
        } for rank, h in enumerate(hits, 1)]
        answer = generate_answer(
            question, [c["text"] for c in contexts], config=config,
            provider=BENCH_LLM_PROVIDER, model=BENCH_LLM_MODEL,
        )
        latency = int((time.time() - t0) * 1000)

        rec = {
            "question": question,
            "contexts": contexts,
            "answer": (answer or "").strip(),
            "config": {
                "top_k": top_k,
                "chunk_size": config.get("chunk_size"),
                "chunk_strategy": config.get("chunk_strategy"),
                "embedding_model": config.get("embedding_model"),
                "llm_model": BENCH_LLM_MODEL,
                "use_hybrid": config.get("use_hybrid"),
                "use_reranker": config.get("use_reranker"),
            },
            "latency_ms": latency,
        }
        # 골든셋 계열은 있을 때만 싣는다(make_external_rag 와 같은 규칙 — falsy 정답을
        # "없음"으로 오인하지 않도록 is not None 으로 판정).
        if q.get("ground_truth") is not None:
            rec["ground_truth"] = q["ground_truth"]
        if q.get("gold_contexts") is not None:
            rec["gold_contexts"] = q["gold_contexts"]

        records.append(rec)
        print(f"  [{i}/{len(questions)}] {question[:34]:34} ctx={len(contexts)} {latency}ms")
    return records


def main() -> int:
    from core.console import force_utf8_stdio
    force_utf8_stdio()

    ap = argparse.ArgumentParser(description="우리 RAG 실행 로그 덤퍼 (벤치마크용)")
    ap.add_argument("--stage", default="baseline", choices=["baseline", "tuned"],
                    help="baseline=0회차(기본 설정) / tuned=최적화 후")
    ap.add_argument("--index-config", default=None,
                    help="tuned 일 때 적용할 index_config JSON")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--questions", default=DEFAULT_QUESTIONS,
                    help=f"기본: {DEFAULT_QUESTIONS} (채점 전용 test 세트)")
    ap.add_argument("--out", default=None, help=f"기본: {DEFAULT_OUTDIR}/ours_<stage>.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="질문 수 제한(0=전체)")
    args = ap.parse_args()

    # 모델 고정은 환경변수로 건다 — generator 가 provider/model 인자를 못 받는 경로로
    # 폴백하더라도 같은 모델을 쓰도록.
    os.environ["RAG_LLM_PROVIDER"] = BENCH_LLM_PROVIDER
    os.environ["RAG_OPENROUTER_MODEL"] = BENCH_LLM_MODEL

    docs = load_corpus(args.corpus)
    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[:args.limit]
    config = load_index_config(args.stage, args.index_config)
    out = args.out or f"{DEFAULT_OUTDIR}/ours_{args.stage}.jsonl"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print(f"[our-rag] stage={args.stage} docs={len(docs)} questions={len(questions)}")
    print(f"[our-rag] chunk={config.get('chunk_strategy')} "
          f"{config.get('chunk_size')}/{config.get('chunk_overlap')} "
          f"top_k={config.get('top_k')} hybrid={config.get('use_hybrid')} "
          f"reranker={config.get('use_reranker')}")
    print(f"[our-rag] llm={BENCH_LLM_MODEL} ({BENCH_LLM_PROVIDER}) — AutoRAG 와 동일 고정")

    retriever, _state = build_pipeline(docs, config)
    records = run(questions, retriever, config)

    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[our-rag] {len(records)}건 → {out}")
    print(f"[our-rag] 다음: python -m agents.eval.replay {out}"
          f" --golden={DEFAULT_QUESTIONS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
