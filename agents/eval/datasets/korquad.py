"""
agents/eval/datasets/korquad.py
KorQuAD 2.1(전처리본, data/) 어댑터 — 정규 파이프라인 입력으로 변환.

data/ 스키마:
    corpus.jsonl  : {doc_id, chunk_id, title, text, char_start, char_end}  (문서별 청크)
    qa_pairs.jsonl: {qa_id, question, answer_text, doc_id, positive_chunk_ids}

이 데이터셋은 사람이 만든 골든 QA 이므로 Probe.source="taxonomy" 로 쓴다
(신뢰도 user_log > taxonomy > llm_generated).

- reconstruct_documents(): corpus 청크를 doc_id 별로 원문 좌표(char_start/end)에 되붙여
  Document 로 복원한다 → Ingest 가 이걸 수집하고 Index 가 자기 전략으로 재청킹한다.
- load_taxonomy_probes(): qa 를 taxonomy Probe 로 만들되, positive_chunk_ids 의 원문
  좌표(corpus 에서 조회)를 gold_spans 로 실어 준다 → Eval 이 재청킹된 현재 청크에
  맞춰 _resync_gold_chunk_ids 로 gold_chunk_ids 를 다시 잡는다(청킹 전략이 바뀌어도 유지).

두 함수의 좌표계는 동일하다: _stitch 가 text 를 char_start 위치에 그대로 놓으므로,
복원된 Document.content 의 좌표 == corpus 의 char_start/end == gold_spans 좌표.

max_docs 는 두 함수가 같은 규칙(corpus 등장 순서 앞 N개 doc)으로 제한해 corpus/qa 가
같은 문서 집합을 보도록 맞춘다(소규모 스모크용).
"""
from __future__ import annotations

import json
import os

from core.schema import Document, Probe

DEFAULT_CORPUS = "data/corpus.jsonl"
DEFAULT_QA = "data/qa_pairs.jsonl"


def _iter_jsonl(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"KorQuAD 파일이 없습니다: {path} — data/README.md 를 참고해 파일을 배치하세요.")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _selected_doc_ids(corpus_path: str, max_docs):
    """corpus 등장 순서로 앞 max_docs 개 doc_id 집합. max_docs 없음/<=0 → None(전체)."""
    if not max_docs or max_docs <= 0:
        return None
    picked: list[str] = []
    seen: set[str] = set()
    for o in _iter_jsonl(corpus_path):
        did = o["doc_id"]
        if did not in seen:
            seen.add(did)
            picked.append(did)
            if len(picked) >= max_docs:
                break
    return set(picked)


def _stitch(spans: list[tuple[int, int, str]]) -> str:
    """(start, end, text) 조각들을 원문 좌표에 채워 하나의 문자열로 복원.
    겹치는 구간은 같은 글자로 덮어써지고, 빈 구간은 공백으로 남는다."""
    if not spans:
        return ""
    total = max(max(e for _, e, _ in spans),
                max(s + len(t) for s, _, t in spans))
    buf = [" "] * total
    for start, _end, text in spans:
        for i, ch in enumerate(text):
            buf[start + i] = ch
    return "".join(buf)


def reconstruct_documents(corpus_path: str = DEFAULT_CORPUS, *, max_docs=None) -> list[Document]:
    """corpus.jsonl → Document 리스트(doc_id 당 1개)."""
    keep = _selected_doc_ids(corpus_path, max_docs)
    by_doc: dict[str, dict] = {}
    for o in _iter_jsonl(corpus_path):
        did = o["doc_id"]
        if keep is not None and did not in keep:
            continue
        d = by_doc.setdefault(did, {"title": o.get("title", ""), "spans": []})
        d["spans"].append((int(o.get("char_start", 0)),
                           int(o.get("char_end", 0)),
                           o.get("text", "") or ""))

    docs: list[Document] = []
    for did, d in by_doc.items():
        docs.append(Document(
            doc_id=did,
            source=f"korquad:{corpus_path}",
            format="txt",  # 순수 텍스트 → Index 청킹이 substring 을 보존해 resync 가 안전
            content=_stitch(d["spans"]),
            metadata={"title": d["title"], "dataset": "korquad2.1",
                      "chunk_count": len(d["spans"])},
        ))
    return docs


def _chunk_span_index(corpus_path: str, keep) -> dict[tuple[str, str], tuple[int, int]]:
    """(doc_id, chunk_id) → (start, end). keep(집합/None)로 문서 제한.
    chunk_id 는 스키마상 '문서 내' 고유일 뿐이라, 서로 다른 문서가 같은 chunk_id 를 쓰면
    단일 키로는 뒤 문서가 앞 문서 좌표를 덮어쓴다 → doc_id 를 함께 복합 키로 쓴다."""
    idx: dict[tuple[str, str], tuple[int, int]] = {}
    for o in _iter_jsonl(corpus_path):
        did = o["doc_id"]
        if keep is not None and did not in keep:
            continue
        idx[(did, o["chunk_id"])] = (int(o.get("char_start", 0)), int(o.get("char_end", 0)))
    return idx


def load_taxonomy_probes(qa_path: str = DEFAULT_QA, corpus_path: str = DEFAULT_CORPUS,
                         *, limit=None, max_docs=None) -> list[Probe]:
    """qa_pairs.jsonl → taxonomy Probe(gold_spans 포함). 재청킹 후 resync 로 gold 확정."""
    keep = _selected_doc_ids(corpus_path, max_docs)
    span_of = _chunk_span_index(corpus_path, keep)

    probes: list[Probe] = []
    for o in _iter_jsonl(qa_path):
        did = o.get("doc_id")
        if keep is not None and did not in keep:
            continue
        gold_spans = []
        for cid in (o.get("positive_chunk_ids") or []):
            hit = span_of.get((did, cid))   # qa 의 doc_id 로 그 문서의 청크만 조회
            if hit:
                s, e = hit
                gold_spans.append({"doc_id": did, "start": s, "end": e})
        probes.append(Probe(
            probe_id=f"probe_qa_{o['qa_id']}",
            question=o["question"],
            source="taxonomy",
            expected_difficulty="medium",
            answer_exists=True,
            ground_truth=o.get("answer_text"),
            gold_chunk_ids=[],           # resync 가 현재 청크 기준으로 채운다
            gold_doc_id=did,
            gold_spans=gold_spans,
            qtype=None,
            metadata={"qa_id": str(o.get("qa_id")), "dataset": "korquad2.1"},
        ))
        if limit is not None and len(probes) >= limit:
            break
    return probes
