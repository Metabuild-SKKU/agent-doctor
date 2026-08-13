"""
tools/make_autorag_log.py
AutoRAG 최적 파이프라인의 실행 로그 덤퍼 — make_our_rag.py / make_external_rag.py 의 세 번째 짝.

왜 API 를 거치나: AutoRAG 가 찾은 최적 파이프라인은 trial 폴더 안에 설정으로만 있고,
그걸 실제로 돌리는 방법은 `autorag run_api` 로 띄워 질의하는 것이다. 우리가 AutoRAG 의
내부를 재구현하면 "우리가 해석한 AutoRAG"를 재는 셈이라 비교가 성립하지 않는다.

세 덤퍼가 같은 스키마 v1 로 뱉는 이유: 채점을 agents.eval.replay 한 곳에서 하기 위해서다.
크로스 도구 비교에서 replay 가 유일한 공통분모다 — 우리 Eval STEP2 는 자기가 검색·생성하므로
남이 만든 답변을 받을 자리가 없다.

서버 띄우기(Windows 주의):
    PYTHONUTF8=1 autorag run_api --trial_dir bench/out/autorag_project/0 --host 127.0.0.1 --port 8100

  PYTHONUTF8=1 이 필요한 이유: run_api 가 trial 의 config.yaml 을 encoding 지정 없이 열어
  Windows 기본 코드페이지(cp949)로 디코드한다. 한국어 프롬프트나 em dash 가 들어 있으면
  UnicodeDecodeError 로 죽는다(실측: config.yaml 18번째 바이트의 '—').

사용법:
    python -m tools.make_autorag_log
    python -m tools.make_autorag_log --limit=3      # 연결 확인용
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_QUESTIONS = "bench/golden_test_50.json"
DEFAULT_URL = "http://127.0.0.1:8100/v1/run"
DEFAULT_OUT = "bench/out/autorag.jsonl"

# AutoRAG config 에 고정한 값과 같아야 한다(bench/autorag_config.yaml).
# 로그의 config 필드는 진단기가 권고 카드에 "현재값"을 실을 때 쓴다.
BENCH_LLM_MODEL = "openai/gpt-4o-mini"


def load_questions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return [q for q in items if str(q.get("question") or "").strip()]


def ask(url: str, question: str, timeout: int) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps({"query": question}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def to_record(question: str, payload: dict, latency_ms: int, golden: dict) -> dict:
    """API 응답 → 스키마 v1 레코드. 다른 두 덤퍼와 필드 구성을 맞춘다."""
    passages = payload.get("retrieved_passage") or []
    contexts = [{
        "text": p.get("content", ""),
        # AutoRAG 의 doc_id 는 우리가 corpus.parquet 에 넣은 chunk_00000 형식 그대로다
        # (make_autorag_dataset 이 매긴 ID) — 그래서 우리 청크와 대조가 가능하다.
        "chunk_id": p.get("doc_id"),
        "score": float(p["score"]) if p.get("score") is not None else None,
        "rank": rank,
        "source_doc": p.get("filepath"),
    } for rank, p in enumerate(passages, 1)]

    rec = {
        "question": question,
        "contexts": contexts,
        "answer": str(payload.get("result") or "").strip(),
        "config": {"llm_model": BENCH_LLM_MODEL, "pipeline": "autorag_best"},
        "latency_ms": latency_ms,
    }
    # 골든셋 계열은 있을 때만(다른 두 덤퍼와 같은 규칙 — falsy 정답을 '없음'으로 오인하지 않게).
    if golden.get("ground_truth") is not None:
        rec["ground_truth"] = golden["ground_truth"]
    if golden.get("gold_contexts") is not None:
        rec["gold_contexts"] = golden["gold_contexts"]
    return rec


def main() -> int:
    from core.console import force_utf8_stdio
    force_utf8_stdio()

    ap = argparse.ArgumentParser(description="AutoRAG 최적 파이프라인 로그 덤퍼")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--questions", default=DEFAULT_QUESTIONS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[:args.limit]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"[autorag-log] {args.url} · 질문 {len(questions)}건")
    records, failures = [], []
    for i, q in enumerate(questions, 1):
        question = q["question"]
        t0 = time.time()
        try:
            payload = ask(args.url, question, args.timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # 한 건 실패로 전체를 버리지 않는다. 다만 조용히 넘기면 "50건 중 몇 건으로 잰
            # 점수인가"를 리포트만 봐서는 알 수 없으므로 끝에 반드시 집계한다.
            failures.append((question, str(exc)))
            print(f"  [{i}/{len(questions)}] 실패: {exc}")
            continue
        latency = int((time.time() - t0) * 1000)
        rec = to_record(question, payload, latency, q)
        records.append(rec)
        print(f"  [{i}/{len(questions)}] {question[:34]:34} ctx={len(rec['contexts'])} {latency}ms")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[autorag-log] {len(records)}건 → {args.out}")
    if failures:
        print(f"[autorag-log] ! 실패 {len(failures)}건 — 채점 표본이 그만큼 줄어듭니다:")
        for q, err in failures[:5]:
            print(f"    · {q[:44]} ({err})")
        # 표본이 갈리면 두 시스템을 같은 자로 잰다는 전제가 깨진다.
        print("    두 시스템의 문항 수가 달라지므로, 실패가 있으면 원인을 고치고 다시 뽑을 것.")
        return 1
    print(f"[autorag-log] 다음: python -m agents.eval.replay {args.out}"
          f" --golden={args.questions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
