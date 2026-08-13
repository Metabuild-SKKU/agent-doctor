"""
tools/bench_golden_to_probes.py
벤치마크 골든셋(question/ground_truth/gold_contexts) → Eval Probe 스토어(qa.json).

왜 필요한가: 우리 파이프라인을 벤치마크 train 세트로 최적화하려면 Eval 이 그 질문들을
써야 한다. Eval 은 Probe 를 EVAL_PROBE_STORE 경로에서 읽으므로(probe_store), 골든셋을
Probe 형식으로 옮겨 놓으면 파이프라인 코드를 건드리지 않고 시험지를 갈아끼울 수 있다.

gold_contexts(텍스트) → gold_spans(문자 좌표)로 바꾸는 이유:
Eval 의 검색 지표(recall@k)는 gold 를 **좌표**로 들고 있다가 재청킹할 때마다 현재 청크에
resync 한다(agents/eval/agent.py). 그래서 청킹 파라미터가 바뀌어도 같은 시험지로 채점된다 —
Optimize 가 청크 크기를 흔들며 탐색할 수 있는 근거가 이것이다. 텍스트만 넘기면 이 resync
가 성립하지 않는다.

좌표는 코퍼스 원문에서 직접 찾는다(korquad.py 가 KorQuAD 좌표를 싣는 것과 같은 취지).
gold_contexts 는 원문에서 그대로 발췌된 것이라 부분문자열 검색으로 정확히 잡힌다
(실측: 100/100). 못 찾은 항목은 조용히 버리지 않고 집계해서 알린다.

source 는 "user_log" 로 둔다 — probe 외부성 원칙(청크에서 뽑지 않는다)을 만족하는 값이며,
이 골든셋은 실제로 원문에서 사람이/LLM 이 만들어 검증한 외부 시험지다.

사용법:
    python -m tools.bench_golden_to_probes --golden=bench/out/golden_train_49.json \
        --out=bench/out/probes_train.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

DEFAULT_CORPUS = "data/pdf_corpus.json"
DEFAULT_GOLDEN = "bench/out/golden_train_49.json"
DEFAULT_OUT = "bench/out/probes_train.json"


def load_corpus_text(path: str) -> tuple[str, str]:
    """코퍼스 → (doc_id, 원문). 문서가 여럿이면 좌표계가 갈리므로 거부한다."""
    with open(path, encoding="utf-8") as f:
        docs = json.load(f)
    if not docs:
        sys.exit(f"[probes] 코퍼스가 비었습니다: {path}")
    if len(docs) > 1:
        sys.exit(f"[probes] 문서가 {len(docs)}개입니다. gold_spans 좌표는 문서 단위라"
                 f" 현재 변환기는 단일 문서만 지원합니다.")
    return docs[0]["id"], docs[0]["text"]


def build_probes(golden: list[dict], doc_id: str, text: str) -> tuple[list[dict], list[dict]]:
    """골든 항목 → Probe dict. 반환: (성립한 probe, 좌표를 못 찾은 항목)."""
    probes, missed = [], []
    for i, item in enumerate(golden):
        spans = []
        for gold in item.get("gold_contexts") or []:
            start = text.find(gold)
            if start < 0:
                continue
            spans.append({"doc_id": doc_id, "start": start, "end": start + len(gold)})
        if not spans:
            missed.append(item)
            continue
        probes.append({
            "probe_id": f"bench_{i:04d}",
            "question": item["question"],
            # 청크에서 뽑지 않은 외부 시험지 — probe 외부성 원칙을 만족한다.
            "source": "user_log",
            "expected_difficulty": None,
            "answer_exists": True,
            "ground_truth": item.get("ground_truth"),
            "gold_chunk_ids": [],      # 재청킹 때마다 Eval 이 좌표로부터 다시 채운다
            "qtype": None,
            "metadata": {"gold_contexts": item.get("gold_contexts") or []},
            "gold_doc_id": doc_id,
            "gold_char_span": (spans[0]["start"], spans[0]["end"]),
            "gold_spans": spans,
        })
    return probes, missed


def main() -> int:
    from core.console import force_utf8_stdio
    force_utf8_stdio()

    ap = argparse.ArgumentParser(description="벤치마크 골든셋 → Eval Probe 스토어")
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    with open(args.golden, encoding="utf-8") as f:
        golden = json.load(f)
    doc_id, text = load_corpus_text(args.corpus)
    probes, missed = build_probes(golden, doc_id, text)

    if not probes:
        sys.exit("[probes] 좌표를 찾은 항목이 하나도 없습니다 — 골든셋과 코퍼스가"
                 " 다른 문서에서 나온 게 아닌지 확인하세요.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        # probe_store.save_probes 와 같은 형식({"version", "probes"}).
        json.dump({"version": "bench-1", "probes": probes}, f, ensure_ascii=False, indent=2)

    print(f"[probes] {args.golden} {len(golden)}건 → probe {len(probes)}건 · {args.out}")
    if missed:
        # 시험지가 줄어든 사실은 반드시 드러낸다 — 두 시스템의 문항 수가 갈리면
        # "같은 재료로 붙였다"는 전제가 깨진다.
        print(f"[probes] ! 좌표를 못 찾아 제외 {len(missed)}건:")
        for m in missed[:5]:
            print(f"    · {m['question'][:44]}")
        return 1
    print("[probes] 다음: EVAL_PROBE_STORE 로 이 파일을 가리켜 파이프라인을 돌리세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
