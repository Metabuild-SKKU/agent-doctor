# -*- coding: utf-8 -*-
"""QA셋↔로그 병합 유틸(agents/eval/qa_merge.py) 단위 테스트."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.eval.qa_merge import load_qa_set, merge_qa_into_log, normalize_question


def _write(td, name, text):
    path = os.path.join(td, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


LOG_LINES = "\n".join([
    json.dumps({"question": "공제 한도는?", "answer": "700만원입니다",
                "contexts": ["ctx"]}, ensure_ascii=False),
    json.dumps({"question": "실업률은  얼마인가", "answer": "3.1%",
                "contexts": ["ctx"]}, ensure_ascii=False),
    "깨진 줄 {{{",
])


class TestNormalizeQuestion(unittest.TestCase):
    def test_punctuation_whitespace_case(self):
        self.assertEqual(normalize_question("공제  한도는? "),
                         normalize_question("공제 한도는"))
        self.assertEqual(normalize_question("What IS it?"),
                         normalize_question("what is it"))


class TestLoadQaSet(unittest.TestCase):
    def test_json_array_with_key_variants(self):
        with tempfile.TemporaryDirectory() as td:
            qa = _write(td, "qa.json", json.dumps([
                {"q": "공제 한도는", "answers": ["700만원", "칠백만원"],
                 "context": "연금저축 세액공제 한도는 연 700만원이다."},
            ], ensure_ascii=False))
            qa_map, errors = load_qa_set(qa)
        self.assertEqual(errors, [])
        entry = qa_map[normalize_question("공제 한도는?")]
        self.assertEqual(entry["ground_truth"], "700만원")     # 복수 정답은 첫 항목
        self.assertEqual(len(entry["gold_contexts"]), 1)

    def test_jsonl_format(self):
        with tempfile.TemporaryDirectory() as td:
            qa = _write(td, "qa.jsonl", "\n".join([
                json.dumps({"question": "실업률은 얼마인가?", "ground_truth": "3.1%"},
                           ensure_ascii=False),
                json.dumps({"no_question": 1}),
            ]))
            qa_map, errors = load_qa_set(qa)
        self.assertIn(normalize_question("실업률은 얼마인가"), qa_map)
        self.assertEqual(len(errors), 1)                       # 질문 없는 항목 집계


class TestMerge(unittest.TestCase):
    def _merge(self, qa_payload):
        with tempfile.TemporaryDirectory() as td:
            log = _write(td, "log.jsonl", LOG_LINES)
            qa = _write(td, "qa.json", json.dumps(qa_payload, ensure_ascii=False))
            out = os.path.join(td, "merged.jsonl")
            stats = merge_qa_into_log(log, qa, out)
            with open(out, encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        return stats, lines

    def test_fills_matching_records(self):
        stats, lines = self._merge([
            {"question": "공제 한도는", "ground_truth": "700만원",
             "gold_contexts": ["연금저축 세액공제 한도는 연 700만원이다."]},
            {"question": "없는 질문?", "ground_truth": "x"},
        ])
        self.assertEqual(stats["matched"], 1)
        self.assertEqual(stats["filled_ground_truth"], 1)
        self.assertEqual(stats["filled_gold_contexts"], 1)
        self.assertEqual(stats["qa_unmatched"], 1)
        merged = json.loads(lines[0])
        self.assertEqual(merged["ground_truth"], "700만원")
        self.assertEqual(len(merged["gold_contexts"]), 1)

    def test_does_not_overwrite_existing_ground_truth(self):
        with tempfile.TemporaryDirectory() as td:
            log = _write(td, "log.jsonl", json.dumps(
                {"question": "공제 한도는?", "answer": "a", "contexts": ["c"],
                 "ground_truth": "로그가 준 값"}, ensure_ascii=False))
            qa = _write(td, "qa.json", json.dumps(
                [{"question": "공제 한도는", "ground_truth": "다른 값"}],
                ensure_ascii=False))
            out = os.path.join(td, "m.jsonl")
            stats = merge_qa_into_log(log, qa, out)
            with open(out, encoding="utf-8") as f:
                merged = json.loads(f.read().strip())
        self.assertEqual(merged["ground_truth"], "로그가 준 값")   # 로그 값 유지
        self.assertEqual(stats["conflicts"], 1)
        self.assertEqual(stats["filled_ground_truth"], 0)

    def test_gold_contexts_conflict_is_counted(self):
        # GT 와 같은 규칙 - 불일치를 조용히 버리면 병합 리포트가 거짓말이 된다
        with tempfile.TemporaryDirectory() as td:
            log = _write(td, "log.jsonl", json.dumps(
                {"question": "공제 한도는?", "answer": "a", "contexts": ["c"],
                 "gold_contexts": ["로그가 준 근거"]}, ensure_ascii=False))
            qa = _write(td, "qa.json", json.dumps(
                [{"question": "공제 한도는", "gold_contexts": ["다른 근거"]}],
                ensure_ascii=False))
            out = os.path.join(td, "m.jsonl")
            stats = merge_qa_into_log(log, qa, out)
            with open(out, encoding="utf-8") as f:
                merged = json.loads(f.read().strip())
        self.assertEqual(merged["gold_contexts"], ["로그가 준 근거"])
        self.assertEqual(stats["conflicts"], 1)

    def test_duplicate_qa_question_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            qa = _write(td, "qa.json", json.dumps([
                {"question": "공제 한도는?", "ground_truth": "700만원"},
                {"question": "공제 한도는", "ground_truth": "800만원"},
            ], ensure_ascii=False))
            qa_map, errors = load_qa_set(qa)
        self.assertEqual(len(qa_map), 1)
        self.assertTrue(any("중복 질문" in e for e in errors))
        # 마지막 항목 우선 규약
        self.assertEqual(qa_map[normalize_question("공제 한도는")]["ground_truth"], "800만원")

    def test_broken_lines_pass_through(self):
        stats, lines = self._merge([{"question": "공제 한도는", "ground_truth": "x"}])
        self.assertEqual(lines[-1], "깨진 줄 {{{")               # 원문 그대로 통과
        self.assertEqual(stats["log_lines"], 3)

    def test_unmatched_log_records_unchanged(self):
        _, lines = self._merge([{"question": "공제 한도는", "ground_truth": "x"}])
        untouched = json.loads(lines[1])
        self.assertNotIn("ground_truth", untouched)


if __name__ == "__main__":
    unittest.main()
