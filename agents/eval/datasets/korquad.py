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


def _as_bool(value, *, default: bool) -> bool:
    """JSONL 의 진위값을 읽는다. 미지정(None)이면 default.

    문자열도 받는다 — 손으로 만든 데이터셋이 "false" 를 문자열로 싣는 일이 흔하고,
    파이썬에서 비어 있지 않은 문자열은 전부 참이라 조용히 뒤집힌다.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no", ""}
    return bool(value)


# core/schema.Probe.qtype 가 받는 값. 그 밖은 조용히 무시한다 — 오타 하나가 진단 경로를
# 바꾸는 것보다 '멀티홉 표시 없음'(None)으로 떨어지는 쪽이 안전하다.
_QTYPES = frozenset({"bridge", "comparison", "aggregation"})


def _as_qtype(value) -> str | None:
    """JSONL 의 질문 유형을 Probe.qtype 으로 읽는다. 모르는 값이면 None.

    이 값이 진단 세 곳을 연다: 나열형 확정(incomplete_enumeration), 멀티홉 판정
    (_is_multi_hop → hop_binding·corpus_gap_partial_hop), bridge 전용 경로. 그래서
    임의 문자열을 통과시키면 안 된다.
    """
    if not isinstance(value, str):
        return None
    key = value.strip().lower()
    return key if key in _QTYPES else None


def _gold_spans_of(qa: dict, doc_id: str, span_of: dict) -> list[dict]:
    """골드 좌표. 명시 gold_spans 가 있으면 그걸 쓰고, 없으면 청크 id 로 환산한다.

    청크 id 환산은 골드를 **청크 통째**(중앙값 497자)로 넓힌다. 정답은 중앙값 7자라
    70배 넓은 구간을 span_recall 이 빈틈없이 덮어야 1점을 주게 되고, corpus 청크 경계와
    Index 재청킹 경계가 달라 정답을 맞힌 실행도 recall=0 이 된다(실측 13%).

    그래서 정답 위치를 아는 QA 는 좌표를 직접 싣는다(tools/build_clean_qa.py). 명시
    좌표를 우선하되 환산 경로는 그대로 둔다 — 기존 qa_pairs.jsonl 이 계속 돌아야 한다.
    """
    explicit = qa.get("gold_spans")
    if isinstance(explicit, list) and explicit:
        spans = []
        for span in explicit:
            if not isinstance(span, dict):
                continue
            start, end = span.get("start"), span.get("end")
            if (isinstance(start, int) and not isinstance(start, bool)
                    and isinstance(end, int) and not isinstance(end, bool)
                    and end > start >= 0):
                spans.append({"doc_id": span.get("doc_id") or doc_id,
                              "start": start, "end": end})
        if spans:
            return spans
    return [
        {"doc_id": doc_id, "start": hit[0], "end": hit[1]}
        for cid in (qa.get("positive_chunk_ids") or [])
        if (hit := span_of.get((doc_id, cid))) is not None
    ]


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
        gold_spans = _gold_spans_of(o, did, span_of)
        probes.append(Probe(
            probe_id=f"probe_qa_{o['qa_id']}",
            question=o["question"],
            source="taxonomy",
            expected_difficulty="medium",
            # 기본 True — KorQuAD 는 전부 답이 있는 질문이라 기존 파일이 그대로 돈다.
            # 무응답(답할 수 없는) 질문을 담은 데이터셋은 이 필드를 false 로 실어야 한다.
            # 없으면 무응답 probe 가 '답이 있는데 못 맞힌 것'으로 잘못 채점되고,
            # generation_abstention_failure·generation_wrongful_abstention 이 발화하지 않는다
            # (diagnose 가 `answer_exists is False` 로 그 경로를 연다).
            answer_exists=_as_bool(o.get("answer_exists"), default=True),
            ground_truth=o.get("answer_text"),
            gold_chunk_ids=[],           # resync 가 현재 청크 기준으로 채운다
            gold_doc_id=did,
            gold_spans=gold_spans,
            # KorQuAD 는 단일홉이라 없는 게 맞고(기본 None), 멀티홉 유형이 표시된
            # 데이터셋은 이 필드를 실어야 한다. 없으면 진단이 '이 질문에 근거가 몇 개
            # 필요한지' 를 모르는 채로 판정한다 — retrieval_incomplete_enumeration 이
            # 예비로 강등돼 확정 라벨(retrieval_low_rank)에 슬롯을 뺏기고,
            # generation_hop_binding_error 의 카운트 폴백도 열리지 않는다.
            # (실측: DragonBall 18건 라벨링에서 사람이 나열형이라 본 5건을 전부 놓쳤다)
            qtype=_as_qtype(o.get("qtype")),
            metadata={"qa_id": str(o.get("qa_id")), "dataset": "korquad2.1"},
        ))
        if limit is not None and len(probes) >= limit:
            break
    return probes
