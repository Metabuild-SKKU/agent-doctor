"""
agents/eval/datasets/korquad.py
KorQuAD 2.1(전처리본) 로더 — Agent Doctor 파이프라인 입력으로 변환.

data/ 전처리본 스키마:
    corpus.jsonl  : {doc_id, chunk_id, title, text, char_start, char_end}   ← 이미 청킹됨
    qa_pairs.jsonl: {qa_id, question, answer_text, doc_id, positive_chunk_ids}

설계상 이 데이터셋은 사람이 만든 골든 QA 이므로 Probe.source="taxonomy" 로 넣는다
(신뢰도: user_log > taxonomy > llm_generated). 핵심은 두 가지 매핑이다:

    corpus.chunk_id          → Chunk.chunk_id       (그대로)
    qa.positive_chunk_ids    → Probe.gold_chunk_ids (그대로)

corpus 가 이미 청킹돼 있고 qa 의 gold 가 그 chunk_id 를 직접 가리키므로,
Ingest/Index 의 재청킹이나 gold_spans→gold_chunk_ids resync 없이 정답 청크가
정확히 맞는다. 그래서 이 로더는 코퍼스를 Document 가 아니라 Chunk 로 직접 만들어
Index 청킹을 건너뛴다.

[한계] 청크 경계가 고정이라 Optimize 의 chunk_size/overlap 계열 처방은 이 데이터셋에
적용되지 않는다(검색/생성/리랭킹 계열 진단·처방은 그대로 유효).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core.schema import Chunk, Probe

DEFAULT_CORPUS = "data/corpus.jsonl"
DEFAULT_QA = "data/qa_pairs.jsonl"


def _iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _chunk_index(chunk_id: str) -> int:
    """'doc_xxx_3' → 3 (chunk_index). 재청킹 유틸이 청크 순번으로 쓰는 값."""
    tail = (chunk_id or "").rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def _to_chunk(o: dict) -> Chunk:
    text = o.get("text", "") or ""
    return Chunk(
        chunk_id=o["chunk_id"],
        doc_id=o["doc_id"],
        text=text,
        char_span=(o.get("char_start"), o.get("char_end")),
        hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        metadata={"title": o.get("title", ""), "chunk_index": _chunk_index(o["chunk_id"])},
    )


def _to_probe(o: dict) -> Probe:
    return Probe(
        probe_id=f"probe_qa_{o['qa_id']}",
        question=o["question"],
        source="taxonomy",
        expected_difficulty="medium",
        answer_exists=True,
        ground_truth=o.get("answer_text"),
        gold_chunk_ids=list(o.get("positive_chunk_ids", []) or []),
        gold_doc_id=o.get("doc_id"),
        qtype=None,
        metadata={"qa_id": str(o.get("qa_id")), "dataset": "korquad2.1"},
    )


def load_corpus_chunks(path: str = DEFAULT_CORPUS, *, doc_ids=None) -> list[Chunk]:
    """corpus.jsonl → Chunk 리스트. doc_ids 를 주면 그 문서들만 적재."""
    keep = set(doc_ids) if doc_ids is not None else None
    return [_to_chunk(o) for o in _iter_jsonl(path)
            if keep is None or o["doc_id"] in keep]


def load_qa_probes(path: str = DEFAULT_QA, *, limit=None, doc_ids=None) -> list[Probe]:
    """qa_pairs.jsonl → taxonomy Probe 리스트(정답+gold 청크 포함). limit=앞에서 N개."""
    keep = set(doc_ids) if doc_ids is not None else None
    probes: list[Probe] = []
    for o in _iter_jsonl(path):
        if keep is not None and o["doc_id"] not in keep:
            continue
        probes.append(_to_probe(o))
        if limit is not None and len(probes) >= limit:
            break
    return probes


def load_dataset(
    corpus_path: str = DEFAULT_CORPUS,
    qa_path: str = DEFAULT_QA,
    *,
    qa_limit: int | None = None,
    distractor_docs: int = 0,
) -> tuple[list[Chunk], list[Probe]]:
    """(chunks, probes) 반환.

    1) qa 를 앞에서부터 qa_limit 개 고른다(None=전체).
    2) 그 qa 들의 gold 문서는 코퍼스에 반드시 포함(정답 검색 가능).
    3) distractor_docs > 0 : gold 외 문서를 그만큼 더 섞는다(검색 난이도↑).
       distractor_docs < 0 : 전체 코퍼스 사용.
       distractor_docs = 0 : gold 문서만(검색이 쉬워 검색 진단은 약함).
    4) gold_chunk_ids 가 로드된 청크에 실제로 있는지 검증 후 경고.
    """
    probes = load_qa_probes(qa_path, limit=qa_limit)
    gold_docs = {p.gold_doc_id for p in probes if p.gold_doc_id}

    chunks: list[Chunk] = []
    picked: set[str] = set()
    for o in _iter_jsonl(corpus_path):
        did = o["doc_id"]
        if did in gold_docs or distractor_docs < 0 or did in picked:
            chunks.append(_to_chunk(o))
        elif len(picked) < distractor_docs:
            picked.add(did)
            chunks.append(_to_chunk(o))

    _warn_missing_gold(chunks, probes)
    return chunks, probes


def _warn_missing_gold(chunks: list[Chunk], probes: list[Probe]) -> None:
    ids = {c.chunk_id for c in chunks}
    missing = [p.probe_id for p in probes
               if p.gold_chunk_ids and not any(g in ids for g in p.gold_chunk_ids)]
    if missing:
        print(f"[KorQuAD] 경고: gold 청크가 코퍼스에 없는 probe {len(missing)}개 "
              f"(예: {missing[:3]}) — 이 probe 는 recall 0 으로 집계된다")
