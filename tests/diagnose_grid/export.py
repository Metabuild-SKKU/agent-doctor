"""
tests/diagnose_grid/export.py
Case → JSONL 내보내기. 진단 파이프라인(tests/test_eval_diagnosis_pipeline.py)이 소비한다.

build() 가 만든 EvalRecord 를 직렬화한다 — Case 를 레코드로 옮기는 경로를 하나로 묶기
위해서다. 예전에는 to_dict() 가 Case 필드를 직접 읽어 두 경로가 갈렸고, KorQuAD 전환이
build() 에만 반영되면서 내보낸 JSONL 의 question·ground_truth·gold_spans 가 비었다.

원칙: 계산되는 값은 안 싣는다.
  · recall/f1/oracle_f1 — 청크 좌표·답변 텍스트에서 파생되므로 러너가 계산한다.
    데이터셋에 박아두면 청킹이 바뀌었을 때 옛 값이 남아 좌표와 어긋난다.
  · ragas/oracle_ragas/aspect — 심판 LLM 출력이라 계산으로 못 만든다. 케이스가 값을 적은
    경우만 싣는다(metrics 아래 — 러너 스키마 호환).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Iterable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests.diagnose_grid.builder import Case, build


def _chunk_entry(chunk) -> dict:
    """청크 1개 → JSONL 항목. 판정에 쓰이는 좌표는 전부 싣는다.

    original_char_span 을 빠뜨리면 러너가 복원한 청크는 그 필드가 None 이라
    _chunk_coverage_span 이 char_span 으로 떨어진다 — 트림 틈이 되살아나 #108 이전
    판정이 나온다. 실제로 그랬다(섹션 경계 케이스가 retrieval_failure 로 떨어짐).
    """
    entry = {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "text": chunk.text,
        "section": chunk.section,
        "char_span": list(chunk.char_span),
    }
    if chunk.original_char_span is not None:
        entry["original_char_span"] = list(chunk.original_char_span)
    if chunk.duplicate_spans:
        entry["duplicate_spans"] = [list(s) for s in chunk.duplicate_spans]
    return entry


_JUDGE_LITERAL_FIELDS = ("judge_real", "judge_oracle", "judge_abstention", "judge_reasoning_mode")


def _reject_judge_literals(case: Case) -> None:
    """케이스에 손으로 적은 심판 값을 막는다.

    러너는 ragas·oracle_ragas·aspect 를 fixture 리터럴로 받는 걸 명시적으로 금지하고
    (test_eval_diagnosis_pipeline.JUDGE_METRIC_KEYS), 대신 requires_llm +
    EVAL_DIAGNOSIS_USE_LLM 으로 실제 심판을 태운다. export 가 metrics 아래에 실어 보내면
    러너의 계약 테스트에서 깨지는데, 원인이 두 파일 떨어져 있어 추적이 오래 걸린다.
    여기서 즉시 막아 그 왕복을 없앤다.
    """
    used = [f for f in _JUDGE_LITERAL_FIELDS if getattr(case, f) not in (None, {}, [])]
    if used:
        raise NotImplementedError(
            f"심판 값을 케이스에 적는 경로는 지원하지 않는다({case.id}: {used}). "
            "러너가 ragas/oracle_ragas/aspect 리터럴을 금지한다 — 심판이 필요하면 "
            "needs_judge 를 달아 requires_llm 경로로 실제 심판을 태울 것.")


def _gold_excluded(gold_ids, excluded: set) -> bool:
    """gold 청크가 코퍼스에서 빠졌나. 러너의 gold.in_corpus 가 불리언이라 부분 제외는 못 싣는다."""
    hit = [g for g in gold_ids if g in excluded]
    if hit and len(hit) < len(gold_ids):
        raise NotImplementedError(
            f"gold 청크를 일부만 코퍼스에서 빼는 케이스는 러너가 표현하지 못한다"
            f"(gold.in_corpus 가 불리언) — 빠진 gold: {hit}")
    return bool(hit)


def to_dict(case: Case) -> dict:
    """케이스 1건 → 러너가 읽는 dict. 지표는 안 싣는다(러너가 계산)."""
    if case.compute_ragas:
        raise NotImplementedError(
            "compute_ragas 는 JSONL 경로에서 지원하지 않는다 — 심판 호출은 러너가 맡는다. "
            f"({case.id})")
    _reject_judge_literals(case)

    record, chunks = build(case)
    chunks_by_id = {c.chunk_id: c for c in chunks}
    excluded = {chunks[i].chunk_id for i in case.corpus_exclude}
    corpus = [c for c in chunks if c.chunk_id not in excluded]

    probe = record.probe
    # 코퍼스에서 뺀 청크는 gold.chunks·retrieved 에도 실으면 안 된다 — 러너의
    # _fixture_chunks 가 세 목록의 합집합을 코퍼스로 삼아서, 여기 남기면 제외가 무효가 된다.
    gold_entries = [_chunk_entry(chunks_by_id[g]) for g in probe.gold_chunk_ids
                    if g in chunks_by_id and g not in excluded]
    retrieved_entries = [_chunk_entry(chunks_by_id[cid]) for cid in record.retrieved_chunk_ids
                         if cid in chunks_by_id and cid not in excluded]

    # metrics 는 싣지 않는다 — 규칙 지표는 러너가 계산하고, 심판 값은 리터럴 자체를
    # _reject_judge_literals 가 막는다. 둘 다 러너의 금지 목록과 같은 방향이다.
    out = {
        "case_id": case.id,
        "mode": "deep",
        "situation": case.situation,
        "config": {
            "top_k": len(record.retrieved_chunk_ids),
            "wide_n": 100,
            "rerank_candidates": 20,
            "max_rerank_candidates": 50,
        },
        "qar": {
            "question": probe.question,
            "gold_answer": probe.ground_truth,
            "rag_answer": record.generated_answer,
        },
        "gold": {
            "chunk_ids": list(probe.gold_chunk_ids),
            "answer": probe.ground_truth,
            "spans": list(probe.gold_spans),
            "chunks": gold_entries,
            # 러너는 in_corpus 가 참이면 gold.chunk_ids 중 코퍼스에 없는 것을 더미로 만들어
            # 채운다. 제외한 gold 가 그렇게 되살아나면 corpus_gap 이 안 열린다.
            "in_corpus": not _gold_excluded(probe.gold_chunk_ids, excluded),
        },
        "corpus_chunks": [_chunk_entry(c) for c in corpus],
        "retrieved": retrieved_entries,
        "oracle_answer": record.oracle_answer,
        "qtype": probe.qtype,
        "answer_exists": probe.answer_exists,
        "retrieval_details": dict(record.retrieval_details),
        "expected_labels": sorted(
            label
            for group in case.expect.values()
            for label in ([group] if isinstance(group, str) else (group or []))
        ),
        # 케이스 구성이 의도대로인지 러너가 확인할 값(파생 검증). 지표가 아니다.
        "assert_derived": dict(case.assert_derived),
    }
    if probe.metadata:
        out["probe_metadata"] = dict(probe.metadata)
    if case.wide_ranking is not None:
        out["retriever_candidates"] = [{"chunk_id": chunks[i].chunk_id} for i in case.wide_ranking]
    if case.dense_ranking is not None:
        out["dense_candidates"] = [{"chunk_id": chunks[i].chunk_id} for i in case.dense_ranking]
    if case.lexical_ranking is not None:
        out["keyword_candidates"] = [{"chunk_id": chunks[i].chunk_id} for i in case.lexical_ranking]
    if case.known_gap:
        out["known_gap"] = case.known_gap
    if case.known_gap_labels is not None:
        out["known_gap_labels"] = list(case.known_gap_labels)
    if case.needs_judge:
        # 러너의 심판 게이트에 그대로 얹는다 — 격자가 따로 판단하지 않는다.
        # requires_llm + EVAL_DIAGNOSIS_USE_LLM 이 켜지고 키가 있으면 실제로 돈다.
        out["needs_judge"] = case.needs_judge      # 사람이 읽는 사유
        out["requires_llm"] = True                 # 러너가 읽는 게이트
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
    directory = os.path.dirname(target)
    if directory:
        os.makedirs(directory, exist_ok=True)
    print(f"{write_jsonl(CASES, target)}건 → {target}")
