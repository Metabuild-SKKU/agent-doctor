"""
tests/diagnose_grid/export.py
Case → JSONL 내보내기. 진단 파이프라인(tests/test_eval_diagnosis_pipeline.py)이 소비한다.

원칙: 계산되는 값은 안 싣는다.
  · recall/f1/oracle_f1 — 청크 좌표·답변 텍스트에서 파생되므로 러너가 계산한다.
    데이터셋에 박아두면 청킹이 바뀌었을 때 옛 값이 남아 좌표와 어긋난다.
  · ragas/oracle_ragas/aspect — 심판 LLM 출력이라 계산으로 못 만든다. 실제로 부르면
    실행마다 흔들려 골든 비교가 깨지므로 이건 싣는다(metrics 아래 유지 — 러너 스키마 호환).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Iterable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests.diagnose_grid.builder import (
    Case, build_chunks, gold_chunk_ids, _answer_text, doc_text,
)


def _chunk_entry(chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "text": chunk.text,
        "char_span": list(chunk.char_span),
    }


def _ranking(indices, chunks) -> list[dict]:
    return [{"chunk_id": chunks[i].chunk_id} for i in indices] if indices else []


def to_dict(case: Case) -> dict:
    """케이스 1건 → 러너가 읽는 dict. 지표는 안 싣는다(러너가 계산)."""
    chunks = build_chunks(case)
    corpus = [c for i, c in enumerate(chunks) if i not in set(case.corpus_exclude)]
    gold_ids = gold_chunk_ids(case, chunks)
    by_id = {c.chunk_id: c for c in chunks}

    judge: dict = {}
    if case.judge_real:
        judge["ragas"] = dict(case.judge_real)
    if case.judge_oracle:
        judge["oracle_ragas"] = dict(case.judge_oracle)
    aspect = {}
    if case.judge_abstention is not None:
        aspect["abstention"] = case.judge_abstention
    if case.judge_reasoning_mode is not None:
        aspect["reasoning_mode"] = case.judge_reasoning_mode
    if aspect:
        judge["aspect"] = aspect

    out = {
        "case_id": case.id,
        "mode": "deep",
        "config": {
            "top_k": len(case.retrieved),
            "wide_n": 100,
            "rerank_candidates": 20,
            "max_rerank_candidates": 50,
        },
        "qar": {
            "question": "이 문서에서 묻는 사실은 무엇인가",
            "gold_answer": case.ground_truth,
            "rag_answer": _answer_text(case.answer, case.ground_truth) or "",
        },
        "gold": {
            "chunk_ids": gold_ids,
            "answer": case.ground_truth,
            "spans": [{"doc_id": d, "start": s, "end": e} for d, s, e in case.gold_spans],
            "chunks": [_chunk_entry(by_id[g]) for g in gold_ids if g in by_id],
        },
        "corpus_chunks": [_chunk_entry(c) for c in corpus],
        "retrieved": [_chunk_entry(chunks[i]) for i in case.retrieved],
        "oracle_answer": _answer_text(case.oracle_answer, case.ground_truth),
        "qtype": case.qtype,
        "answer_exists": case.answer_exists,
        "retrieval_details": {
            "search_mode": case.search_mode,
            "reranked": case.reranked,
            "mmr_applied": case.mmr_applied,
            **({"pre_rerank_ids": [chunks[i].chunk_id for i in case.pre_rerank]}
               if case.pre_rerank is not None else {}),
        },
        "expected_labels": sorted(
            label
            for group in case.expect.values()
            for label in ([group] if isinstance(group, str) else (group or []))
        ),
        # 케이스 구성이 의도대로인지 러너가 확인할 값(파생 검증). 지표가 아니다.
        "assert_derived": dict(case.assert_derived),
    }
    if case.span_grounding is not None:
        out["gold"]["span_grounding"] = case.span_grounding
    if case.wide_ranking is not None:
        out["retriever_candidates"] = _ranking(case.wide_ranking, chunks)
    if case.dense_ranking is not None:
        out["dense_candidates"] = _ranking(case.dense_ranking, chunks)
    if case.lexical_ranking is not None:
        out["keyword_candidates"] = _ranking(case.lexical_ranking, chunks)
    if judge:
        out["metrics"] = judge          # 심판 값만 — recall/f1 은 러너가 계산
    # 격자 재생용 원본 서술. 러너는 안 읽지만, 케이스를 다시 만들 때 필요하다.
    out["_source_case"] = {
        "chunk_strategy": case.chunk_strategy,
        "chunk_size": case.chunk_size,
        "chunk_overlap": case.chunk_overlap,
        "docs": [{"id": d.id, "length": d.length, "space_every": d.space_every,
                  "text": d.text} for d in case.docs],
    }
    return out


def write_jsonl(cases: Iterable[Case], path: str) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(to_dict(case), ensure_ascii=False) + "\n")
            n += 1
    return n


if __name__ == "__main__":
    from tests.diagnose_grid.cases_g3 import CASES
    target = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/diagnose_grid_g3.jsonl"
    os.makedirs(os.path.dirname(target), exist_ok=True)
    print(f"{write_jsonl(CASES, target)}건 → {target}")
