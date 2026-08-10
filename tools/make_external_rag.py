"""
tools/make_external_rag.py
외부 RAG 시뮬레이터 — "남의 RAG" 실행 로그를 스키마 v1 JSONL 로 뽑는다.

왜 필요한가: agents/eval/replay.py 는 '다른 팀이 만든 RAG 를 그 실행 로그로 진단'
하는 경로인데, 지금까지 검증은 손으로 만든 로그로만 했다. 진짜 RAG 가 자연스럽게
내는 실패(검색 굶주림·환각)를 ext_ 소견이 실제로 짚는지는 확인되지 않았다.

왜 LlamaIndex 인가: 검증에 필요한 건 '좋은 RAG' 가 아니라 **우리가 안 만든 RAG** 다.
LlamaIndex 는 자체 청킹(SentenceSplitter)·자체 노드 ID(UUID)·자체 프롬프트를 쓰므로
우리 인덱스와 네임스페이스가 갈린다 — 청크 ID 정합 불가라는 실제 조건이 재현된다.
우리 qdrant_store 를 재활용하면 그 마찰이 사라져 검증이 무의미해진다.

결함 주입(--defect)은 원인이 알려진 로그를 만들기 위한 것이다. 정답을 아는 로그가
있어야 "진단기가 맞혔나"를 판정할 수 있다(진단기를 진단하는 대조군).

    none         정상 — 대조군
    starve       top_k=1 + 청크 축소로 검색을 굶긴다 → 근거 부족 유도
    hallucinate  컨텍스트를 무시하고 지어내라고 지시 → 환각 유도
    offtopic     엉뚱한 질문의 컨텍스트를 붙인다 → 검색 무관 유도

사용법:
    python -m tools.make_external_rag --defect=none --out=tests/fixtures/external_rag/ext_none.jsonl
    python -m tools.make_external_rag --defect=starve --limit=5

임베딩·LLM 은 .env 의 OpenRouter 키를 쓴다(INDEX_EMBED_PROVIDER 와 무관하게
이 도구는 자기 스택으로 돈다 — '남의 RAG' 이므로 우리 설정을 따르지 않는다).
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
DEFAULT_QUESTIONS = "tools/external_rag_questions.json"
# 뽑은 로그는 커밋되는 테스트 자산이라 tests/fixtures 로 간다. 최상위 logs/ 를 쓰면
# 런타임 산출물 경로(output/logs — core/run_logger.py)와 이름이 겹쳐 성격이 헷갈린다.
FIXTURE_DIR = "tests/fixtures/external_rag"

# OpenRouter 는 OpenAI 호환 API 라 llama-index-embeddings-openai / llms-openai 를
# 그대로 쓴다(추가 패키지 불필요 — PR #88 이 core/llm_clients 에서 쓴 것과 같은 원리).
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
EMBED_MODEL = os.getenv("EXT_RAG_EMBED_MODEL", "baai/bge-m3")
LLM_MODEL = os.getenv("EXT_RAG_LLM_MODEL", "anthropic/claude-sonnet-4.5")

# 결함별 설정. chunk_size 까지 흔드는 이유: top_k 만 줄이면 '적게 가져왔다'일 뿐이고,
# 청크를 잘게 쪼개야 근거가 조각나 실제 검색 굶주림에 가까워진다.
DEFECTS = {
    "none":        {"top_k": 5, "chunk_size": 512, "chunk_overlap": 50},
    "starve":      {"top_k": 1, "chunk_size": 128, "chunk_overlap": 0},
    "hallucinate": {"top_k": 5, "chunk_size": 512, "chunk_overlap": 50},
    "offtopic":    {"top_k": 5, "chunk_size": 512, "chunk_overlap": 50},
}

# 환각 유도 프롬프트. "자료를 무시하라"는 지시만으로는 환각이 안 난다 — 실측 결과
# claude-sonnet-4.5 는 지시를 거부하고 컨텍스트대로 정확히 답했다(정답률 6/6).
# 그래서 검색 쪽을 어긋나게 해서(offtopic 과 같은 방식) 근거를 없앤 뒤, 기권만 막는다.
# 근거가 없는데 답해야 하니 지어낼 수밖에 없는 구조 — 프롬프트 순종에 의존하지 않는다.
HALLUCINATE_TMPL = (
    "당신은 사내 규정 안내 챗봇입니다. 아래 참고 자료를 바탕으로 질문에 답하세요. "
    "사용자는 즉답을 원하므로 '자료에 없다', '확인이 필요하다'는 답변은 금지입니다. "
    "구체적인 숫자와 조건을 포함해 자신 있게 답하세요.\n\n"
    "참고 자료:\n{context_str}\n\n질문: {query_str}\n답변: "
)


def _require(module: str, pip_name: str):
    try:
        __import__(module)
    except ImportError:
        sys.exit(f"[ext-rag] {module} 가 없습니다 — pip install {pip_name}")


def load_corpus(path: str) -> list[dict]:
    if not os.path.exists(path):
        # data/ 는 gitignore 대상이라 저장소를 받은 사람에겐 코퍼스가 없다. 로그를 새로
        # 뽑을 때만 필요하므로, 진단만 해볼 사람에게 fixture 경로를 함께 알려준다.
        sys.exit(f"[ext-rag] 코퍼스가 없습니다: {path}\n"
                 f"  data/ 는 gitignore 대상입니다(data/README.md 참고).\n"
                 f"  · 다른 코퍼스로 뽑기: --corpus <경로>\n"
                 f"  · 진단만 돌리기:     python -m agents.eval.replay {FIXTURE_DIR}/ext_none.jsonl")
    with open(path, encoding="utf-8") as f:
        docs = json.load(f)
    if not isinstance(docs, list) or not docs:
        sys.exit(f"[ext-rag] 코퍼스가 비었거나 리스트가 아닙니다: {path}")
    return docs


def load_questions(path: str) -> list[dict]:
    """질문셋(+정답·근거). 골든셋 계열 필드는 선택 — 없으면 로그만 뽑는다."""
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return [q for q in items if str(q.get("question") or "").strip()]


def build_index(docs: list[dict], defect: str):
    """LlamaIndex 로 '남의 RAG' 를 구성한다. 우리 모듈은 일절 쓰지 않는다."""
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai_like import OpenAILike

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        sys.exit("[ext-rag] OPENROUTER_API_KEY 가 없습니다 — .env 를 확인하세요.")

    cfg = DEFECTS[defect]
    # bge-m3 는 OpenAI 카탈로그에 없는 모델이라 차원을 명시해야 SDK 가 검증을 통과한다.
    Settings.embed_model = OpenAIEmbedding(
        model_name=EMBED_MODEL, api_base=OPENROUTER_BASE, api_key=key, dimensions=1024)
    # OpenAI 클래스가 아니라 OpenAILike 를 쓴다 — 전자는 모델명을 자기 카탈로그에서
    # 찾아 컨텍스트 길이를 정하므로 OpenRouter 의 "publisher/model" 이름에 ValueError 로
    # 죽는다(생성자 인자로도 못 막는다: metadata 프로퍼티가 매번 조회한다).
    Settings.llm = OpenAILike(
        model=LLM_MODEL, api_base=OPENROUTER_BASE, api_key=key,
        context_window=int(os.getenv("EXT_RAG_CONTEXT_WINDOW", "200000")),
        is_chat_model=True,
        temperature=1.0 if defect == "hallucinate" else 0.0)
    Settings.node_parser = SentenceSplitter(
        chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"])

    li_docs = [Document(text=d["text"], metadata={"source": d.get("source") or d.get("id")})
               for d in docs]
    index = VectorStoreIndex.from_documents(li_docs)

    kwargs = {"similarity_top_k": cfg["top_k"]}
    if defect == "hallucinate":
        from llama_index.core import PromptTemplate
        kwargs["text_qa_template"] = PromptTemplate(HALLUCINATE_TMPL)
    return index.as_query_engine(**kwargs), cfg


def run(questions: list[dict], engine, cfg: dict, defect: str) -> list[dict]:
    """질문을 돌려 스키마 v1 로그 레코드를 만든다."""
    records = []
    for i, q in enumerate(questions, 1):
        question = q["question"]
        # 엉뚱한 질문으로 검색해 무관한 컨텍스트를 붙인다(검색 실패 재현).
        # offtopic 은 그 자체가 목적이고, hallucinate 는 근거를 없애 지어내게 만드는 수단이다
        # (기권 금지 프롬프트와 조합) — 프롬프트 지시만으로는 환각이 재현되지 않았다.
        misroute = defect in ("offtopic", "hallucinate")
        search_q = questions[i % len(questions)]["question"] if misroute else question

        t0 = time.time()
        resp = engine.query(search_q)
        latency = int((time.time() - t0) * 1000)

        contexts = [{
            "text": n.node.get_content(),
            "chunk_id": n.node.node_id,          # LlamaIndex UUID — 우리와 네임스페이스 다름
            "score": float(n.score) if n.score is not None else None,
            "rank": rank,
            "source_doc": n.node.metadata.get("source"),
        } for rank, n in enumerate(resp.source_nodes, 1)]

        rec = {
            "question": question,
            "contexts": contexts,
            "answer": str(resp).strip(),
            "config": {"top_k": cfg["top_k"], "chunk_size": cfg["chunk_size"],
                       "embedding_model": EMBED_MODEL, "llm_model": LLM_MODEL,
                       "use_reranker": False},
            "latency_ms": latency,
        }
        # 골든셋 계열은 있을 때만 싣는다(시험지 필드 — 로그엔 원래 없는 값).
        if q.get("ground_truth"):
            rec["ground_truth"] = q["ground_truth"]
        if q.get("gold_contexts"):
            rec["gold_contexts"] = q["gold_contexts"]

        records.append(rec)
        print(f"  [{i}/{len(questions)}] {question[:34]:34} ctx={len(contexts)} {latency}ms")
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="외부 RAG 시뮬레이터 (LlamaIndex)")
    ap.add_argument("--defect", default="none", choices=sorted(DEFECTS),
                    help="주입할 결함 (기본 none = 정상 대조군)")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--questions", default=DEFAULT_QUESTIONS)
    ap.add_argument("--out", default=None,
                    help=f"기본: {FIXTURE_DIR}/ext_<defect>.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="질문 수 제한(0=전체)")
    ap.add_argument("--no-gold", action="store_true",
                    help="골든셋 필드를 빼고 뽑는다(triad 전용 tier 재현)")
    args = ap.parse_args()

    _require("llama_index.core", "llama-index")
    _require("llama_index.embeddings.openai", "llama-index-embeddings-openai")

    docs = load_corpus(args.corpus)
    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[:args.limit]
    if args.no_gold:
        questions = [{"question": q["question"]} for q in questions]

    out = args.out or f"{FIXTURE_DIR}/ext_{args.defect}.jsonl"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print(f"[ext-rag] defect={args.defect} docs={len(docs)} questions={len(questions)}")
    print(f"[ext-rag] embed={EMBED_MODEL} llm={LLM_MODEL} (OpenRouter)")
    engine, cfg = build_index(docs, args.defect)
    records = run(questions, engine, cfg, args.defect)

    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ext-rag] {len(records)}건 → {out}")
    print(f"[ext-rag] 다음: python -m agents.eval.replay {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
