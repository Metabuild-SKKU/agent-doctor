"""
tests/diagnose_grid/runner.py
케이스 실행·채점.

1. 케이스를 빌드해 diagnose() 를 정상 경로로 돌린다.
2. assert_derived 로 '의도한 파생값이 실제로 나왔나'를 먼저 본다 — 어긋나면 라벨 비교 전에
   케이스 구성 오류로 보고한다(기대 라벨이 틀린 게 아니다).
3. 그룹(A/B/C/D)별 라벨 집합을 expect 와 비교한다.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.eval import diagnose as dx
from agents.eval.metrics_basic import (
    _gold_span_boundary_analysis, _gold_chunk_evidence_density,
    _oversized_gold_spans, _context_char_total, _gold_position_band,
)
from agents.eval.metrics_search import _gold_ranks, _gold_pre_rerank_ranks, _bm25_hits_gold
from agents.eval.metrics_ragas import _faith, _faith_oracle, _rel, _rel_oracle, _ctx_precision
from agents.eval.types import Mode
from tests.diagnose_grid.builder import Case, build


def _derived(record) -> dict:
    """assert_derived 로 확인할 수 있는 파생값 스냅샷."""
    boundary = _gold_span_boundary_analysis(record) or {}
    oversized = _oversized_gold_spans(record) or {}
    ranks = _gold_ranks(record) or {}
    pre = _gold_pre_rerank_ranks(record) or {}
    missed = set(record.probe.gold_chunk_ids) - set(record.retrieved_chunk_ids)
    return {
        "recall_at_k": record.recall_at_k,
        "f1_score": record.f1_score,
        "oracle_f1": record.oracle_f1,
        "boundary_split": boundary.get("boundary_split_count"),
        "contained": boundary.get("contained_count"),
        "uncovered": boundary.get("uncovered_count"),
        "oversized_count": oversized.get("oversized_count"),
        "max_chunk_len": oversized.get("max_chunk_len"),
        "max_span_len": oversized.get("max_span_len"),
        "evidence_density": _gold_chunk_evidence_density(record),
        "context_chars": _context_char_total(record),
        "position_band": _gold_position_band(record),
        "bm25_hits_gold": _bm25_hits_gold(record),
        "gold_ranks": ranks,
        "missed_gold_ranks": {g: ranks.get(g) for g in missed},
        "missed_pre_rerank_ranks": {g: pre.get(g) for g in missed},
        "gold_count": len(record.probe.gold_chunk_ids),
        "missed_count": len(missed),
        # 심판 값 — compute_ragas 케이스에서 계산 결과가 의도한 구간에 들었는지 본다.
        # diagnose 가 이미 트랙을 채운 뒤라 여기서 새 LLM 호출은 나지 않는다(ragas_done).
        "faithfulness": _faith(record),
        "oracle_faithfulness": _faith_oracle(record),
        "response_relevancy": _rel(record),
        "oracle_response_relevancy": _rel_oracle(record),
        "context_precision": _ctx_precision(record),
    }


def _matches(actual, spec) -> bool:
    """spec 이 '<1', '>0', '>=0.5' 면 비교 연산, 아니면 동등 비교."""
    if isinstance(spec, str) and spec[:1] in "<>=" and spec not in ("=",):
        for op in (">=", "<=", "==", "<", ">", "="):
            if spec.startswith(op):
                bound = float(spec[len(op):])
                if actual is None:
                    return False
                if op == ">=":
                    return actual >= bound
                if op == "<=":
                    return actual <= bound
                if op in ("==", "="):
                    return actual == bound
                if op == "<":
                    return actual < bound
                return actual > bound
    return actual == spec


def _by_group(findings) -> dict:
    out = {"A": [], "B": [], "C": [], "D": []}
    for f in findings:
        out.get(f.metadata.get("group"), out["C"]).append(f.label)
    return out


def _normalize(expect) -> dict:
    out = {"A": [], "B": [], "C": [], "D": []}
    for group in out:
        value = expect.get(group)
        if value is None:
            continue
        out[group] = [value] if isinstance(value, str) else list(value)
    return out


@dataclass
class Result:
    case_id: str
    ok: bool
    kind: str                       # "pass" | "label_mismatch" | "case_error"
    expected: dict = field(default_factory=dict)
    actual: dict = field(default_factory=dict)
    derived: dict = field(default_factory=dict)
    bad_asserts: list = field(default_factory=list)
    confirmed: dict = field(default_factory=dict)


def run_case(case: Case) -> Result:
    record, _chunks = build(case)
    findings = dx.diagnose(record, mode=Mode.DEEP)
    derived = _derived(record)

    bad = [(key, spec, derived.get(key))
           for key, spec in case.assert_derived.items()
           if not _matches(derived.get(key), spec)]
    if bad:
        return Result(case.id, False, "case_error", derived=derived, bad_asserts=bad,
                      actual=_by_group(findings))

    actual = _by_group(findings)
    expected = _normalize(case.expect)
    ok = all(sorted(actual[g]) == sorted(expected[g]) for g in expected)
    return Result(
        case.id, ok, "pass" if ok else "label_mismatch",
        expected=expected, actual=actual, derived=derived,
        confirmed={f.label: f.confirmed for f in findings},
    )


def run_all(cases: list[Case]) -> list[Result]:
    return [run_case(c) for c in cases]


def report(results: list[Result]) -> str:
    lines = []
    passed = sum(1 for r in results if r.ok)
    for r in results:
        if r.ok:
            lines.append(f"[PASS] {r.case_id}")
            continue
        if r.kind == "case_error":
            lines.append(f"[CASE] {r.case_id} — 의도한 파생값이 안 나옴")
            for key, spec, actual in r.bad_asserts:
                lines.append(f"         {key}: 기대 {spec} / 실제 {actual}")
            lines.append(f"         실제 라벨: {r.actual}")
            continue
        lines.append(f"[FAIL] {r.case_id}")
        for group in ("A", "B", "C", "D"):
            if sorted(r.expected.get(group, [])) != sorted(r.actual.get(group, [])):
                lines.append(f"         {group}: 기대 {r.expected.get(group)} / 실제 {r.actual.get(group)}")
        lines.append(f"         파생: recall={r.derived['recall_at_k']} f1={r.derived['f1_score']:.3f} "
                     f"oracle_f1={r.derived['oracle_f1']:.3f} "
                     f"split={r.derived['boundary_split']} oversized={r.derived['oversized_count']} "
                     f"density={r.derived['evidence_density']}")
    lines.append(f"\n{passed}/{len(results)} pass")
    return "\n".join(lines)
