"""
tests/test_ragec_runner.py
run_ragec_validation 의 구간 실행(--offset/--append) — 파이프라인 없이 도는 부분만.

377건 1회가 약 100분·$5 인데 findings 는 Eval 이 전부 끝난 뒤에만 쓰였다(#145). 구간으로
나눠 돌리고 병합하는 경로가 앞 구간 결과를 지우지 않는지 고정한다.
"""
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.run_ragec_validation import _describe_range, merge_findings, overwrite_guard


class MergeFindingsTest(unittest.TestCase):
    def test_new_probes_are_appended_and_old_ones_kept(self):
        merged = merge_findings([{"qa_id": "1", "labels": ["a"]}],
                                [{"qa_id": "2", "labels": ["b"]}])
        self.assertEqual([r["qa_id"] for r in merged], ["1", "2"])

    def test_rerun_segment_replaces_the_same_qa_id_in_place(self):
        """죽은 구간을 같은 --offset 으로 다시 돌리면 최신 결과가 이겨야 한다."""
        merged = merge_findings([{"qa_id": "1", "labels": ["old"]}, {"qa_id": "2"}],
                                [{"qa_id": "1", "labels": ["new"]}])
        self.assertEqual(merged, [{"qa_id": "1", "labels": ["new"]}, {"qa_id": "2"}])

    def test_qa_id_types_do_not_split_rows(self):
        merged = merge_findings([{"qa_id": 1, "labels": ["old"]}], [{"qa_id": "1", "labels": ["new"]}])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["labels"], ["new"])


class OverwriteGuardTest(unittest.TestCase):
    """구간 실행인데 --append 가 없으면 **돌기 전에** 멈춘다 — 100분 뒤에 알면 늦다."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self._tmp.name) / "findings.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def test_offset_without_append_over_existing_findings_is_refused(self):
        self.path.write_text("{}\n", encoding="utf-8")
        self.assertIsNotNone(overwrite_guard(100, False, self.path))

    def test_append_is_allowed(self):
        self.path.write_text("{}\n", encoding="utf-8")
        self.assertIsNone(overwrite_guard(100, True, self.path))

    def test_first_segment_and_fresh_start_are_allowed(self):
        self.assertIsNone(overwrite_guard(100, False, self.path))      # 파일이 없다
        self.path.write_text("{}\n", encoding="utf-8")
        self.assertIsNone(overwrite_guard(0, False, self.path))        # 처음부터 = 의도된 덮어쓰기


class RangeDescriptionTest(unittest.TestCase):
    def test_ranges_read_as_positions(self):
        self.assertEqual(_describe_range(0, 0), "전체")
        self.assertEqual(_describe_range(0, 10), "상한 10")
        self.assertEqual(_describe_range(100, 100), "101~200번째")
        self.assertEqual(_describe_range(200, 0), "201번째부터 끝까지")


if __name__ == "__main__":
    unittest.main()
