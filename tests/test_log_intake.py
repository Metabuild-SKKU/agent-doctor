import json
import os
import sys
import tempfile
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.eval.log_intake import (
    TIER_NONE,
    TIER_QA_ONLY,
    TIER_TRIAD,
    assess_capability,
    load_external_log,
    parse_record,
)


def _write_jsonl(path: str, lines: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write((line if isinstance(line, str) else json.dumps(line, ensure_ascii=False)) + "\n")


class ParseRecordTest(unittest.TestCase):
    def test_full_schema_line_is_normalized(self):
        rec = parse_record({
            "question": "공제 한도는?",
            "contexts": [
                {"text": "청크 원문", "chunk_id": "doc3_c12", "score": 0.83, "rank": 1,
                 "source_doc": "세금가이드.pdf"},
                "문자열 컨텍스트도 허용",
            ],
            "answer": "700만원입니다",
            "config": {"top_k": 5, "chunk_size": 512},
            "feedback": "thumbs_down",
            "latency_ms": 1840,
            "timestamp": "2026-08-06T14:02:11",
        })

        self.assertEqual(rec.question, "공제 한도는?")
        self.assertEqual(len(rec.contexts), 2)
        self.assertEqual(rec.contexts[0]["chunk_id"], "doc3_c12")
        self.assertAlmostEqual(rec.contexts[0]["score"], 0.83)
        # 문자열 항목도 같은 키 구조로 통일된다
        self.assertEqual(rec.contexts[1]["text"], "문자열 컨텍스트도 허용")
        self.assertIsNone(rec.contexts[1]["score"])
        self.assertEqual(rec.config["top_k"], 5)
        self.assertEqual(rec.feedback, "thumbs_down")
        self.assertEqual(rec.latency_ms, 1840)

    def test_missing_required_fields_raise(self):
        with self.assertRaises(ValueError):
            parse_record({"question": "질문만 있음", "contexts": []})
        with self.assertRaises(ValueError):
            parse_record({"question": "   ", "answer": "공백 질문"})
        with self.assertRaises(ValueError):
            parse_record(["오브젝트가", "아님"])

    def test_empty_context_entries_are_dropped(self):
        rec = parse_record({
            "question": "q", "answer": "a",
            "contexts": ["", {"text": "  "}, {"chunk_id": "id만 있음"}, "유효"],
        })
        self.assertEqual([c["text"] for c in rec.contexts], ["유효"])


class LoadExternalLogTest(unittest.TestCase):
    def test_broken_lines_are_skipped_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "log.jsonl")
            _write_jsonl(path, [
                {"question": "q1", "contexts": ["c"], "answer": "a1"},
                "이건 JSON이 아님 {",
                {"question": "q2"},          # answer 누락
                "",                           # 빈 줄은 오류 아님
                {"question": "q3", "contexts": ["c"], "answer": "a3"},
            ])

            records, errors = load_external_log(path)

        self.assertEqual([r.question for r in records], ["q1", "q3"])
        self.assertEqual(len(errors), 2)
        self.assertIn("2행", errors[0])
        self.assertIn("3행", errors[1])

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_external_log(os.path.join("없는", "경로", "log.jsonl"))


class AssessCapabilityTest(unittest.TestCase):
    def _record(self, **kw):
        base = {"question": "q", "answer": "a"}
        base.update(kw)
        return parse_record(base)

    def test_empty_log_is_tier_none(self):
        cap = assess_capability([])
        self.assertEqual(cap["tier"], TIER_NONE)

    def test_qa_only_log_limits_diagnosis(self):
        cap = assess_capability([self._record() for _ in range(4)])
        self.assertEqual(cap["tier"], TIER_QA_ONLY)
        self.assertEqual(cap["with_contexts"], 0)

    def test_triad_requires_contexts_in_majority(self):
        with_ctx = [self._record(contexts=["근거"]) for _ in range(3)]
        without = [self._record() for _ in range(3)]
        self.assertEqual(assess_capability(with_ctx + without)["tier"], TIER_TRIAD)
        # 절반 미만이면 표본 편향 — triad 로 안 쳐준다
        self.assertEqual(assess_capability(with_ctx[:2] + without + [self._record()])["tier"],
                         TIER_QA_ONLY)

    def test_config_and_feedback_are_counted(self):
        records = [
            self._record(contexts=[{"text": "근거", "score": 0.9}],
                         config={"top_k": 5}, feedback="thumbs_down"),
            self._record(contexts=["근거"]),
        ]
        cap = assess_capability(records)
        self.assertEqual(cap["tier"], TIER_TRIAD)
        self.assertEqual(cap["with_scores"], 1)
        self.assertEqual(cap["with_config"], 1)
        self.assertEqual(cap["with_feedback"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
