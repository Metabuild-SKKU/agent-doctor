"""
진단 fixture 러너.

JSONL 한 줄을 하나의 QAR/gold/retrieval 케이스로 읽어 diagnose 라벨을 검증한다.
이 테스트는 새 파이프라인을 만들지 않고, Eval agent가 쓰는 `_prepare_record()`와
`diagnose()`를 그대로 호출한다. 규칙 지표(recall/f1/oracle_f1)는 fixture 숫자를 믿지 않고
항상 텍스트·청크 좌표에서 계산한다. RAGAS/AspectCritic처럼 LLM judge가 필요한 케이스는
fixture에 점수를 박지 않고, `EVAL_DIAGNOSIS_USE_LLM=1`일 때 실제 judge로 계산한다.
"""

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
from agents.eval.agent import _prepare_record
from agents.eval.metrics_ragas import (
    evaluate_abstention,
    evaluate_oracle_track,
    evaluate_real_track,
    evaluate_reasoning_mode,
    _judge as _ragas_judge,
)
from agents.eval.types import EvalRecord, Mode


FIXTURE_PATH = Path(__file__).with_name("fixtures") / "eval_diagnosis_cases.jsonl"
RULE_METRIC_KEYS = {
    "recall_at_k",
    "recall_basis",
    "f1_score",
    "oracle_f1",
    "raw_f1_score",
    "raw_oracle_f1",
    "exact_match",
}
JUDGE_METRIC_KEYS = {"ragas", "oracle_ragas", "aspect"}
TRUE_VALUES = {"1", "true", "yes", "on"}


class _FakeRetriever:
    def __init__(self, results: list[dict]):
        self.results = [dict(item) for item in results]

    def __call__(self, *args, **kwargs):
        top_n = kwargs.get("top_n")
        if top_n is None and args:
            top_n = args[-1]
        return self.results[: int(top_n or len(self.results))]


class _StubRetriever:
    def __init__(self, results: list[dict], details: dict | None = None):
        self.results = [dict(item) for item in results]
        self.details = dict(details or {})

    def search_with_details(self, question: str, top_k: int = 5):
        payload = {key: value for key, value in self.details.items() if key != "results"}
        payload["results"] = self.results[: int(top_k or len(self.results))]
        return payload

    def search(self, question: str, top_k: int = 5):
        return self.results[: int(top_k or len(self.results))]


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
    paths: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        try:
            key = str(path.expanduser().resolve())
        except OSError:
            key = str(path.expanduser().absolute())
        if key in seen:
            return
        seen.add(key)
        paths.append(path)

    add(FIXTURE_PATH)
    extra = os.getenv("EVAL_DIAGNOSIS_CASES")
    if extra:
        add(Path(extra))
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
    # 커버리지 판정 전용 좌표. 없으면 char_span 으로 떨어져 트림 틈이 되살아난다(#100/#108).
    original = entry.get("original_char_span")
    duplicates = entry.get("duplicate_spans")
    return Chunk(
        chunk_id=chunk_id,
        doc_id=entry.get("doc_id", "fixture_doc"),
        text=text,
        page=entry.get("page"),
        section=entry.get("section"),
        char_span=tuple(span),
        original_char_span=tuple(original) if original else None,
        duplicate_spans=[list(s) for s in duplicates] if duplicates else [],
    )


def _fixture_chunks(case: dict) -> list[Chunk]:
    seen: dict[str, Chunk] = {}
    entries: list = []
    entries.extend(case.get("corpus_chunks", []))
    entries.extend(case.get("retriever_candidates", []))
    entries.extend(case.get("dense_candidates", []))
    entries.extend(case.get("keyword_candidates", []))
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


def _expected_labels(case: dict) -> list[str]:
    return list(case["expected_labels"])


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUE_VALUES


def _case_requires_llm(case: dict) -> bool:
    return bool(case.get("requires_llm"))


def _llm_fixture_enabled() -> bool:
    return _env_flag("EVAL_DIAGNOSIS_USE_LLM")


def _llm_required() -> bool:
    return _env_flag("EVAL_DIAGNOSIS_REQUIRE_LLM")


def _llm_unavailable_reason(case: dict) -> str | None:
    if not _case_requires_llm(case):
        return None
    if not _llm_fixture_enabled():
        return "requires_llm case; set EVAL_DIAGNOSIS_USE_LLM=1 to run judge-backed label check"
    if _ragas_judge() is None:
        message = (
            "requires_llm case but no Eval judge key is configured "
            "(set EVAL_LLM_PROVIDER and the provider API key)"
        )
        if _llm_required():
            raise AssertionError(message)
        return message
    return None


def _llm_ragas_track(record: EvalRecord, track: str) -> dict:
    judge = _ragas_judge()
    if judge is None:
        return {}
    if track == "oracle":
        return evaluate_oracle_track(record, judge) if record.oracle_answer is not None else {}
    if track == "abstention":
        return evaluate_abstention(record, judge)
    if track == "reasoning_mode":
        return evaluate_reasoning_mode(record, judge) if record.oracle_answer is not None else {}
    return evaluate_real_track(record, judge)


def _probe_metadata(case: dict) -> dict:
    metadata: dict = {}
    nested_probe = case.get("probe")
    if isinstance(nested_probe, dict) and isinstance(nested_probe.get("metadata"), dict):
        metadata.update(nested_probe["metadata"])
    if isinstance(case.get("probe_metadata"), dict):
        metadata.update(case["probe_metadata"])
    return metadata


def _span_tuple(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return (int(value[0]), int(value[1]))


def _optional_retriever(case: dict, key: str):
    if key not in case:
        return None
    return _FakeRetriever(case.get(key) or [])


def _record_from_case(case: dict) -> EvalRecord:
    qar = case.get("qar", {})
    gold = case.get("gold", {})
    retrieved = case.get("retrieved", [])
    gold_ids = list(gold.get("chunk_ids", []))
    config = case.get("config", {})
    top_k = int(config.get("top_k", len(retrieved) or 5))

    probe = Probe(
        probe_id=case.get("case_id", "fixture_case"),
        question=qar.get("question") or case.get("question", ""),
        source=case.get("source", "fixture"),
        answer_exists=case.get("answer_exists", gold.get("answer_exists")),
        ground_truth=gold.get("answer") or qar.get("gold_answer"),
        gold_chunk_ids=gold_ids,
        qtype=case.get("qtype") or gold.get("qtype"),
        gold_spans=list(gold.get("spans", [])),
        metadata=_probe_metadata(case),
        gold_doc_id=gold.get("doc_id") or case.get("gold_doc_id"),
        gold_char_span=_span_tuple(gold.get("char_span") or case.get("gold_char_span")),
    )
    chunk_text = {chunk.chunk_id: chunk.text for chunk in _fixture_chunks(case)}
    record = _prepare_record(
        probe,
        _StubRetriever(retrieved, case.get("retrieval_details", {})),
        chunk_text,
        top_k,
        {},
    )

    record.generated_answer = qar.get("rag_answer") or case.get("generated_answer", "")
    # oracle_answer 를 생략하면 gold answer 를 그대로 쓴다. 이 경우 오라클 트랙은 "gold 기준
    # 재답변"을 흉내 내는 계약이라 만점에 가까워질 수 있으므로, bad_gold/generation 계열처럼
    # 오라클 결과가 라벨을 좌우하는 fixture 는 oracle_answer 를 명시해야 한다.
    record.oracle_answer = case.get("oracle_answer", qar.get("oracle_answer", gold.get("answer")))
    if "oracle_context" in case:
        record.oracle_context = list(case["oracle_context"])
    return record


def _run_case(case: dict) -> list[str]:
    config = case.get("config", {})
    use_llm = _case_requires_llm(case) and _llm_fixture_enabled() and _ragas_judge() is not None
    metrics_common.set_context(
        chunks=_fixture_chunks(case),
        retrieve_fn=_optional_retriever(case, "retriever_candidates"),
        dense_fn=_optional_retriever(case, "dense_candidates"),
        keyword_fn=_optional_retriever(case, "keyword_candidates"),
        ragas_fn=_llm_ragas_track if use_llm else None,
        wide_n=int(config.get("wide_n", 100)),
        rerank_candidates=int(config.get("rerank_candidates", 20)),
        max_rerank_candidates=int(config.get("max_rerank_candidates", 50)),
    )
    record = _record_from_case(case)
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
        evaluated = 0
        evaluated_with_expected_label = 0
        for case in self.cases:
            with self.subTest(case=case.get("case_id"), source=case.get("_source")):
                reason = _llm_unavailable_reason(case)
                if reason:
                    self.skipTest(reason)
                evaluated += 1
                expected = _expected_labels(case)
                if expected:
                    evaluated_with_expected_label += 1
                self.assertEqual(sorted(expected), sorted(_run_case(case)))
        self.assertGreater(
            evaluated,
            0,
            "at least one non-LLM or judge-enabled diagnosis fixture must be evaluated",
        )
        self.assertGreater(
            evaluated_with_expected_label,
            0,
            "at least one evaluated fixture must assert a concrete diagnosis label",
        )

    def test_fixture_cases_carry_review_inputs(self):
        for case in self.cases:
            with self.subTest(case=case.get("case_id"), source=case.get("_source")):
                self.assertIn("config", case)
                self.assertIn("qar", case)
                self.assertIn("gold", case)
                self.assertIn("retrieved", case)
                self.assertIn("expected_labels", case)
                self.assertNotIn("expected_label", case)

                metrics = case.get("metrics") or {}
                rule_overrides = RULE_METRIC_KEYS & set(metrics)
                self.assertFalse(
                    rule_overrides,
                    "recall/f1/oracle_f1/exact_match/recall_basis are computed "
                    f"from QAR, gold spans, and retrieved chunks; remove {sorted(rule_overrides)}",
                )
                judge_overrides = JUDGE_METRIC_KEYS & set(metrics)
                self.assertFalse(
                    judge_overrides,
                    "ragas/oracle_ragas/aspect must be produced by the Eval judge, "
                    f"not fixture literals; remove {sorted(judge_overrides)}",
                )
                for key in JUDGE_METRIC_KEYS:
                    self.assertNotIn(
                        key,
                        case,
                        f"{key} belongs to the judge output path; do not store it "
                        "as a top-level fixture value",
                    )

                # top_k는 retriever cut에만 쓰이고 diagnose는 len(retrieved_chunk_ids)를 본다.
                # 그래서 fixture가 top_k를 적으면 retrieved 길이와 맞춰 두어야 한다.
                top_k = case.get("config", {}).get("top_k")
                if top_k is not None:
                    self.assertEqual(int(top_k), len(case.get("retrieved", [])))

    def test_case_ids_are_unique(self):
        ids = [case.get("case_id") for case in self.cases]
        self.assertEqual(len(ids), len(set(ids)))

    def test_case_paths_deduplicate_default_fixture(self):
        with patch.dict(os.environ, {"EVAL_DIAGNOSIS_CASES": str(FIXTURE_PATH)}):
            self.assertEqual(_case_paths(), [FIXTURE_PATH])

    def test_probe_metadata_is_carried_into_eval_record(self):
        case = {
            "case_id": "span_grounding_contract",
            "qar": {"question": "q"},
            "gold": {},
            "retrieved": [],
            "probe_metadata": {
                "span_grounding": {
                    "status": "exact",
                    "span_qualities": ["exact"],
                }
            },
        }

        record = _record_from_case(case)

        self.assertEqual(record.probe.metadata["span_grounding"]["status"], "exact")
        self.assertEqual(
            record.probe.metadata["span_grounding"]["span_qualities"],
            ["exact"],
        )

    def test_record_builder_uses_eval_prepare_record_contract(self):
        case = {
            "case_id": "prepare_record_contract",
            "config": {"top_k": 1},
            "qar": {
                "question": "q",
                "gold_answer": "gold",
                "rag_answer": "generated",
            },
            "gold": {
                "chunk_ids": ["gold_chunk"],
                "answer": "gold",
                "chunks": [{"chunk_id": "gold_chunk", "text": "gold context"}],
            },
            "retrieved": [{"chunk_id": "hit", "text": "retrieved context"}],
            "retrieval_details": {
                "search_mode": "hybrid",
                "reranked": True,
                "reranker_status": "applied",
                "pre_rerank_ids": ["gold_chunk", "hit"],
                "mmr_applied": False,
            },
            "expected_labels": [],
        }

        record = _record_from_case(case)

        self.assertEqual(record.retrieved_chunk_ids, ["hit"])
        self.assertEqual(record.retrieved_context, ["retrieved context"])
        self.assertEqual(record.oracle_context, ["gold context"])
        self.assertTrue(record.retrieval_details["reranked"])
        self.assertEqual(record.retrieval_details["search_mode"], "hybrid")

    def test_missing_keyword_candidates_keeps_keyword_channel_unmeasured(self):
        case = {
            "case_id": "keyword_unmeasured_contract",
            "mode": "deep",
            "config": {"top_k": 1, "wide_n": 20},
            "qar": {
                "question": "Which fact is required?",
                "gold_answer": "The gold fact is required.",
                "rag_answer": "The retrieved fact is different.",
            },
            "gold": {
                "chunk_ids": ["gold_chunk"],
                "answer": "The gold fact is required.",
            },
            "retrieved": [
                {"chunk_id": "retrieved_chunk", "text": "The retrieved fact is different."}
            ],
            "retriever_candidates": [
                {"chunk_id": "retrieved_chunk", "text": "The retrieved fact is different."}
            ],
            "expected_labels": ["retrieval_missing_gold"],
        }

        labels = _run_case(case)

        self.assertIsNone(metrics_common._ctx.keyword_fn)
        self.assertNotIn("retrieval_semantic_mismatch", labels)

    def test_empty_keyword_candidates_measure_keyword_miss_when_explicit(self):
        case = {
            "case_id": "keyword_empty_contract",
            "mode": "deep",
            "config": {"top_k": 1, "wide_n": 20},
            "qar": {
                "question": "Which fact is required?",
                "gold_answer": "The gold fact is required.",
                "rag_answer": "The retrieved fact is different.",
            },
            "gold": {
                "chunk_ids": ["gold_chunk"],
                "answer": "The gold fact is required.",
            },
            "retrieved": [
                {"chunk_id": "retrieved_chunk", "text": "The retrieved fact is different."}
            ],
            "retriever_candidates": [
                {"chunk_id": "retrieved_chunk", "text": "The retrieved fact is different."}
            ],
            "keyword_candidates": [],
        }

        _run_case(case)

        self.assertIsNotNone(metrics_common._ctx.keyword_fn)

    def test_llm_ragas_track_delegates_to_eval_judge_functions(self):
        case = {
            "case_id": "llm_delegate_contract",
            "qar": {"question": "q", "rag_answer": "answer"},
            "gold": {"answer": "gold"},
            "retrieved": [{"chunk_id": "hit", "text": "context"}],
            "expected_labels": [],
        }
        record = _record_from_case(case)

        with (
            patch(__name__ + "._ragas_judge", return_value=object()),
            patch(
                __name__ + ".evaluate_real_track",
                return_value={"faithfulness": 1.0},
            ) as real,
        ):
            result = _llm_ragas_track(record, "real")

        self.assertEqual({"faithfulness": 1.0}, result)
        real.assert_called_once()


if __name__ == "__main__":
    unittest.main()
