from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import Chunk, Probe
from agents.eval import diagnose, metrics_common
from agents.eval.types import EvalRecord, Mode


FIXTURE_PATH = Path(__file__).with_name("fixtures") / "eval_diagnosis_cases.jsonl"
MEASURED_METRIC_KEYS = {
    "recall_at_k",
    "recall_basis",
    "f1_score",
    "oracle_f1",
    "raw_f1_score",
    "raw_oracle_f1",
    "exact_match",
}


class _FakeRetriever:
    def __init__(self, results: list[dict]):
        self.results = [dict(item) for item in results]

    def __call__(self, *args, **kwargs):
        top_n = kwargs.get("top_n")
        if top_n is None and args:
            top_n = args[-1]
        return self.results[: int(top_n or len(self.results))]


def _mode(value) -> int:
    if isinstance(value, int):
        return value
    return {
        "fast": Mode.FAST,
        "standard": Mode.STANDARD,
        "deep": Mode.DEEP,
        "full": Mode.DEEP,
    }.get(str(value or "deep").lower(), Mode.DEEP)


def _load_jsonl(path: Path) -> list[dict]:
    cases: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            case.setdefault("_source", f"{path}:{line_no}")
            cases.append(case)
    return cases


def _case_paths() -> list[Path]:
    paths = [FIXTURE_PATH]
    extra = os.getenv("EVAL_DIAGNOSIS_CASES")
    if extra:
        paths.append(Path(extra))
    return paths


def _load_cases() -> list[dict]:
    cases: list[dict] = []
    for path in _case_paths():
        if not path.exists():
            raise AssertionError(f"fixture file not found: {path}")
        cases.extend(_load_jsonl(path))
    return cases


def _chunk_from_entry(entry, index: int) -> Chunk:
    if isinstance(entry, str):
        entry = {"chunk_id": entry}
    chunk_id = entry["chunk_id"]
    text = entry.get("text") or f"fixture text for {chunk_id}"
    span = entry.get("char_span")
    if span is None:
        span = (index * 100, index * 100 + max(1, len(text)))
    return Chunk(
        chunk_id=chunk_id,
        doc_id=entry.get("doc_id", "fixture_doc"),
        text=text,
        page=entry.get("page"),
        section=entry.get("section"),
        char_span=tuple(span),
    )


def _fixture_chunks(case: dict) -> list[Chunk]:
    seen: dict[str, Chunk] = {}
    entries: list = []
    entries.extend(case.get("corpus_chunks", []))
    entries.extend(case.get("retriever_candidates", []))
    entries.extend(case.get("retrieved", []))
    entries.extend(case.get("gold", {}).get("chunks", []))

    for idx, entry in enumerate(entries):
        chunk = _chunk_from_entry(entry, idx)
        seen.setdefault(chunk.chunk_id, chunk)

    gold = case.get("gold", {})
    if gold.get("in_corpus", True):
        for chunk_id in gold.get("chunk_ids", []):
            if chunk_id not in seen:
                seen[chunk_id] = _chunk_from_entry(chunk_id, len(seen))

    return list(seen.values())


def _ids(entries: list) -> list[str]:
    out: list[str] = []
    for entry in entries:
        out.append(entry if isinstance(entry, str) else entry["chunk_id"])
    return out


def _expected_labels(case: dict) -> list[str]:
    if "expected_labels" in case:
        return list(case["expected_labels"])
    label = case.get("expected_label")
    return [] if label in (None, "", []) else [label]


def _metric_overrides(case: dict) -> dict:
    metrics = case.get("metrics") or {}
    return {key: metrics[key] for key in MEASURED_METRIC_KEYS if key in metrics}


def _apply_measured_metrics(record: EvalRecord, metrics: dict) -> None:
    for key in MEASURED_METRIC_KEYS:
        if key in metrics:
            setattr(record, key, metrics[key])
    record.raw_f1_score = metrics.get("raw_f1_score", record.f1_score)
    record.raw_oracle_f1 = metrics.get("raw_oracle_f1", record.oracle_f1)


def _record_from_case(case: dict) -> EvalRecord:
    qar = case.get("qar", {})
    gold = case.get("gold", {})
    retrieved = case.get("retrieved", [])
    retrieved_ids = _ids(retrieved)
    gold_ids = list(gold.get("chunk_ids", []))

    probe = Probe(
        probe_id=case.get("case_id", "fixture_case"),
        question=qar.get("question") or case.get("question", ""),
        source=case.get("source", "fixture"),
        answer_exists=case.get("answer_exists", gold.get("answer_exists")),
        ground_truth=gold.get("answer") or qar.get("gold_answer"),
        gold_chunk_ids=gold_ids,
        qtype=case.get("qtype") or gold.get("qtype"),
        gold_spans=list(gold.get("spans", [])),
    )
    record = EvalRecord(
        probe=probe,
        retrieved=[dict(item) if isinstance(item, dict) else {"chunk_id": item} for item in retrieved],
        retrieved_chunk_ids=retrieved_ids,
        retrieved_context=[
            item.get("text", "") for item in retrieved if isinstance(item, dict)
        ],
        generated_answer=qar.get("rag_answer") or case.get("generated_answer", ""),
        oracle_answer=case.get("oracle_answer", qar.get("oracle_answer", gold.get("answer"))),
        oracle_context=[
            item.get("text", "") for item in gold.get("chunks", []) if isinstance(item, dict)
        ],
    )

    metrics = case.get("metrics") or {}
    overrides = _metric_overrides(case)
    if overrides:
        _apply_measured_metrics(record, overrides)
    if "ragas" in metrics:
        record.ragas = dict(metrics["ragas"])
        record.ragas_done = True
    if "oracle_ragas" in metrics:
        record.oracle_ragas = dict(metrics["oracle_ragas"])
        record.oracle_ragas_done = True
    if "aspect" in metrics or "aspect" in case:
        record.aspect = dict(metrics.get("aspect", case.get("aspect", {})))
    record.retrieval_details = dict(case.get("retrieval_details", {}))
    return record


def _run_case(case: dict) -> list[str]:
    config = case.get("config", {})
    metrics_common.set_context(
        chunks=_fixture_chunks(case),
        retrieve_fn=_FakeRetriever(case.get("retriever_candidates", [])),
        dense_fn=_FakeRetriever(case.get("dense_candidates", [])),
        keyword_fn=_FakeRetriever(case.get("keyword_candidates", [])),
        wide_n=int(config.get("wide_n", 100)),
        rerank_candidates=int(config.get("rerank_candidates", 20)),
        max_rerank_candidates=int(config.get("max_rerank_candidates", 50)),
    )
    record = _record_from_case(case)
    overrides = _metric_overrides(case)
    if not overrides:
        findings = diagnose.diagnose(record, mode=_mode(case.get("mode")))
    else:
        with patch.object(
            diagnose,
            "_compute_metrics",
            side_effect=lambda rec: _apply_measured_metrics(rec, overrides),
        ):
            findings = diagnose.diagnose(record, mode=_mode(case.get("mode")))
    return [finding.label for finding in findings]


class EvalDiagnosisPipelineFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = _load_cases()

    def tearDown(self):
        metrics_common.set_context()
        metrics_common.set_mode(Mode.FAST)

    def test_fixture_cases_match_expected_labels(self):
        self.assertTrue(self.cases, "at least one diagnosis fixture is required")
        for case in self.cases:
            with self.subTest(case=case.get("case_id"), source=case.get("_source")):
                self.assertEqual(sorted(_expected_labels(case)), sorted(_run_case(case)))

    def test_fixture_cases_carry_review_inputs(self):
        for case in self.cases:
            with self.subTest(case=case.get("case_id"), source=case.get("_source")):
                self.assertIn("config", case)
                self.assertIn("qar", case)
                self.assertIn("gold", case)
                self.assertIn("retrieved", case)
                self.assertIn("expected_labels", case)

                top_k = case.get("config", {}).get("top_k")
                if top_k is not None:
                    self.assertEqual(int(top_k), len(case.get("retrieved", [])))

    def test_metricless_fixture_cases_are_allowed(self):
        metricless = [case for case in self.cases if not _metric_overrides(case)]
        self.assertTrue(metricless, "at least one fixture should let the runner compute metrics")
        for case in metricless:
            with self.subTest(case=case.get("case_id"), source=case.get("_source")):
                self.assertEqual(sorted(_expected_labels(case)), sorted(_run_case(case)))


if __name__ == "__main__":
    unittest.main()
