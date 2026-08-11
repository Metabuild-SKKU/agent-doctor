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

    def test_csv_format(self):
        # 골든셋을 엑셀에서 CSV로 내보낸 케이스(BOM 포함). 헤더가 곧 키라 관용 규칙이 통한다
        with tempfile.TemporaryDirectory() as td:
            qa = os.path.join(td, "qa.csv")
            with open(qa, "w", encoding="utf-8-sig", newline="") as f:
                f.write("question,answer,context\n")
                f.write("공제 한도는?,700만원,연금저축 한도는 연 700만원이다.\n")
                f.write("빈 정답 행,,\n")
            qa_map, errors = load_qa_set(qa)
        entry = qa_map[normalize_question("공제 한도는")]
        self.assertEqual(entry["ground_truth"], "700만원")
        self.assertEqual(entry["gold_contexts"], ["연금저축 한도는 연 700만원이다."])
        # 정답이 빈 행도 질문은 있으므로 적재되되 값은 None
        self.assertIsNone(qa_map[normalize_question("빈 정답 행")]["ground_truth"])

    def test_cp949_csv_does_not_crash(self):
        """한국어 Windows 엑셀의 기본 'CSV(쉼표로 분리)' 저장은 cp949 - utf-8-sig 로만
        열면 UnicodeDecodeError 가 diagnose_external_log 까지 그대로 전파돼 CLI 가 죽는다."""
        with tempfile.TemporaryDirectory() as td:
            qa = os.path.join(td, "qa.csv")
            with open(qa, "w", encoding="cp949", newline="") as f:
                f.write("question,ground_truth\n")
                f.write("공제 한도는?,700만원\n")
            qa_map, errors = load_qa_set(qa)
        entry = qa_map[normalize_question("공제 한도는")]
        self.assertEqual(entry["ground_truth"], "700만원")

    def test_korquad_style_answers_dict_extracts_text(self):
        """answers=[{"text":...,"answer_start":...}] - dict 를 str() 통짜 변환하면
        "{'text': '700만원', ...}" 이 정답이 돼 F1/correctness 가 조용히 붕괴한다."""
        with tempfile.TemporaryDirectory() as td:
            qa = _write(td, "qa.json", json.dumps([
                {"question": "공제 한도는", "answers": [{"text": "700만원", "answer_start": 10}]},
            ], ensure_ascii=False))
            qa_map, errors = load_qa_set(qa)
        entry = qa_map[normalize_question("공제 한도는?")]
        self.assertEqual(entry["ground_truth"], "700만원")

    def test_xlsx_format(self):
        openpyxl = __import__("openpyxl")
        with tempfile.TemporaryDirectory() as td:
            qa = os.path.join(td, "qa.xlsx")
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.append(["question", "ground_truth", "gold_contexts"])
            ws.append(["실업률은 얼마인가?", "3.1%", "2025년 실업률은 3.1%다."])
            wb.save(qa)
            qa_map, errors = load_qa_set(qa)
        self.assertEqual(errors, [])
        entry = qa_map[normalize_question("실업률은 얼마인가")]
        self.assertEqual(entry["ground_truth"], "3.1%")
        self.assertEqual(entry["gold_contexts"], ["2025년 실업률은 3.1%다."])

    def test_jsonl_content_with_json_extension(self):
        # 상대가 .json 확장자로 JSONL을 주는 실무 케이스 → 재시도로 살린다
        with tempfile.TemporaryDirectory() as td:
            qa = _write(td, "qa.json", "\n".join([
                json.dumps({"question": "q1", "ground_truth": "a1"}, ensure_ascii=False),
                json.dumps({"question": "q2", "ground_truth": "a2"}, ensure_ascii=False),
            ]))
            qa_map, errors = load_qa_set(qa)
        self.assertEqual(len(qa_map), 2)
        self.assertEqual(errors, [])

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
