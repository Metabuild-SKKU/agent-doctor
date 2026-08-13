"""
tools/bench_make_autorag_dataset.py
벤치마크 #1 — 우리 코퍼스·골든셋을 AutoRAG 입력(corpus.parquet / qa.parquet)으로 변환.

왜 우리 청킹을 쓰나: AutoRAG 의 evaluate 는 검색·리랭크·생성을 최적화하고 **청킹은 고정
입력**으로 받는다(corpus.parquet 이 이미 청크다). 그래서 누가 청킹할지는 우리가 정해야
하는데, 양쪽에 같은 청크를 주어야 "같은 검색 대상"이 되어 점수 차이를 검색·생성 탓으로
귀속할 수 있다. 우리 0회차 설정(core/state.py 의 index_config: fixed 512/50)을 쓴다.
  → 이 선택으로 "우리만 청킹까지 튜닝하는" 비대칭이 생긴다. 숨기지 말고 슬라이드에
     명시할 것("AutoRAG는 검색·리랭크·생성을, 우리는 청킹까지 최적화").

retrieval_gt 가 왜 까다로운가: AutoRAG 는 정답 근거를 **corpus 의 doc_id 참조**로 받는데
우리 골든셋의 gold_contexts 는 **원문 텍스트 발췌**다. 그래서 각 발췌가 어느 청크에
들어갔는지 찾아 doc_id 로 바꿔야 한다(Eval 의 gold_spans → gold_chunk_ids resync 와 같은 문제).
  · 한 발췌가 여러 청크에 걸리면(오버랩·반복 상투구) 2D 리스트의 내부 OR 조건으로 넣는다
    — 그중 아무거나 찾으면 정답이다.
  · 어느 청크에도 통째로 안 들어가면(청크 경계에 걸친 긴 발췌) 그 문항은 버린다.
    실측 1건(train, 169자, 표가 뒤엉킨 원문)뿐이고, 채점용 test 50건은 전량 매칭된다.
    비운 채로 넘기지 않는 이유: 양쪽 시스템이 정확히 같은 재료를 받아야 하고, 빈
    retrieval_gt 는 AutoRAG 지표에서 정의되지 않은 값이 된다.

사용법:
    python -m tools.bench_make_autorag_dataset
    python -m tools.bench_make_autorag_dataset --chunk-size=512 --chunk-overlap=50
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

CORPUS_JSON = "data/pdf_corpus.json"
TRAIN_JSON = "bench/golden_train_50.json"
TEST_JSON = "bench/golden_test_50.json"
OUTDIR = "bench/out"

# 0회차 baseline — core/state.py 의 index_config 기본값과 같은 값을 쓴다.
# 여기서 갈라지면 "AutoRAG 가 받은 청크"와 "우리 0회차가 쓴 청크"가 달라져
# 같은 검색 대상이라는 전제가 깨진다.
DEFAULT_STRATEGY = "fixed"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50


def build_chunks(path: str, strategy: str, size: int, overlap: int) -> list[dict]:
    """코퍼스 JSON → AutoRAG corpus 행 목록. 우리 Index 에이전트의 청킹을 그대로 쓴다."""
    from core.schema import Document
    from agents.index.agent import CHUNK_STRATEGIES

    if strategy not in CHUNK_STRATEGIES:
        raise SystemExit(f"[autorag] 모르는 청킹 전략: {strategy} "
                         f"(가능: {', '.join(CHUNK_STRATEGIES)})")
    with open(path, encoding="utf-8") as f:
        docs = json.load(f)
    if not docs:
        raise SystemExit(f"[autorag] 코퍼스가 비었습니다: {path}")

    # metadata.last_modified_datetime 은 AutoRAG 필수 필드다. 실행할 때마다 값이 바뀌면
    # 같은 입력인데 corpus.parquet 이 달라지므로, 코퍼스 파일의 수정 시각을 쓴다(재현성).
    mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)

    rows: list[dict] = []
    for doc in docs:
        document = Document(doc_id=doc["id"], source=doc.get("source", ""),
                            format="pdf", content=doc["text"])
        for draft in CHUNK_STRATEGIES[strategy](document, size, overlap):
            rows.append({
                "doc_id": f"chunk_{len(rows):05d}",
                "contents": draft.text,
                "path": doc.get("source", ""),
                "start_end_idx": (draft.start, draft.end),
                "metadata": {"last_modified_datetime": mtime},
            })
    return rows


def to_qa_rows(golden: list[dict], chunks: list[dict], prefix: str) -> tuple[list[dict], list[dict]]:
    """골든셋 → AutoRAG qa 행 목록. 반환: (성립한 행, 버려진 항목).

    retrieval_gt 는 2D 리스트다. 내부 리스트가 OR 조건이므로, 한 발췌가 여러 청크에
    들어갔으면 그 doc_id 를 한 내부 리스트에 모두 넣는다(어느 걸 찾아도 정답)."""
    rows, dropped = [], []
    for i, item in enumerate(golden):
        golds = item.get("gold_contexts") or []
        # 발췌 하나당 내부 리스트 하나 = "이 발췌를 담은 청크 중 아무거나"(OR).
        # 발췌가 여럿이면 바깥 리스트에 나란히 놓여 AND 로 읽힌다.
        gt: list[list[str]] = []
        for gold in golds:
            ids = [c["doc_id"] for c in chunks if gold in c["contents"]]
            if ids:
                gt.append(ids)
        if not gt:
            dropped.append(item)
            continue
        rows.append({
            "qid": f"{prefix}_{i:04d}",
            "query": item["question"],
            "retrieval_gt": gt,
            "generation_gt": [item["ground_truth"]],
        })
    return rows, dropped


def write_parquet(rows: list[dict], path: str) -> None:
    import pandas as pd

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="AutoRAG 입력 데이터셋 생성")
    ap.add_argument("--corpus", default=CORPUS_JSON)
    ap.add_argument("--train", default=TRAIN_JSON)
    ap.add_argument("--test", default=TEST_JSON)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--strategy", default=DEFAULT_STRATEGY)
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    args = ap.parse_args()

    print(f"[autorag] 청킹: {args.strategy} {args.chunk_size}/{args.chunk_overlap}")
    chunks = build_chunks(args.corpus, args.strategy, args.chunk_size, args.chunk_overlap)
    avg = sum(len(c["contents"]) for c in chunks) // max(1, len(chunks))
    print(f"[autorag] 코퍼스 {args.corpus} → 청크 {len(chunks)}개 (평균 {avg}자)")

    corpus_path = os.path.join(args.outdir, "corpus.parquet")
    write_parquet(chunks, corpus_path)
    print(f"  · {corpus_path}")

    # 제외된 문항은 벤치마크의 재료를 줄이는 결정이라 조용히 넘어가면 안 된다 —
    # 어느 문항이 왜 빠졌는지 남겨야 나중에 "50건이라며 49건이네"를 설명할 수 있다.
    manifest: dict = {"chunk_strategy": args.strategy, "chunk_size": args.chunk_size,
                      "chunk_overlap": args.chunk_overlap, "chunks": len(chunks),
                      "splits": {}}
    for name, path in (("train", args.train), ("test", args.test)):
        with open(path, encoding="utf-8") as f:
            golden = json.load(f)
        rows, dropped = to_qa_rows(golden, chunks, name)
        out = os.path.join(args.outdir, f"qa_{name}.parquet")
        write_parquet(rows, out)

        # 우리 파이프라인이 쓸 같은 재료. AutoRAG 만 49건으로 줄면 우리가 문항 하나를 더
        # 받게 되어 "같은 조건"이 아니다. 골든셋 원본은 건드리지 않고 벤치마크용 사본만 남긴다.
        kept = {d["question"] for d in dropped}
        if kept:
            mirror = os.path.join(args.outdir, f"golden_{name}_{len(rows)}.json")
            with open(mirror, "w", encoding="utf-8") as f:
                json.dump([g for g in golden if g["question"] not in kept],
                          f, ensure_ascii=False, indent=2)
            print(f"  · {mirror}  (우리 파이프라인용 — 같은 {len(rows)}건)")

        multi = sum(1 for r in rows for inner in r["retrieval_gt"] if len(inner) > 1)
        print(f"[autorag] {name}: 골든 {len(golden)}건 → qa {len(rows)}건"
              f" (제외 {len(dropped)}건 · 복수 청크 매칭 {multi}건)")
        print(f"  · {out}")
        for d in dropped:
            gold = (d.get("gold_contexts") or [""])[0]
            print(f"    ! 제외: {d['question'][:40]}… (근거 {len(gold)}자 — 청크 경계에 걸림)")
        manifest["splits"][name] = {
            "golden": len(golden), "qa_rows": len(rows),
            "dropped": [d["question"] for d in dropped],
        }

    manifest_path = os.path.join(args.outdir, "autorag_dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    print(f"  · {manifest_path}")
    print()
    print("다음: .venv-autorag 에서")
    print(f"  autorag evaluate --config <config.yaml>"
          f" --qa_data_path {args.outdir}/qa_train.parquet"
          f" --corpus_data_path {args.outdir}/corpus.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
