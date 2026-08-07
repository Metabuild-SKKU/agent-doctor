"""
tests/test_diagnose_grid.py
격자 데이터셋을 진단 파이프라인에 물려 돌린다.

JSONL 을 커밋하지 않고 setUp 에서 생성한다 — 케이스 정본은 파이썬이고 JSONL 은 파생물이라,
커밋해두면 케이스를 고친 뒤 재생성을 잊었을 때 옛 데이터로 초록불이 난다.

known_gap 케이스는 실패를 기대한다. 진단이 고쳐져서 통과하기 시작하면 이 테스트가 알려준다
(그때 케이스에서 known_gap 을 지우면 된다).
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.diagnose_grid.cases_g3 import CASES
from tests.diagnose_grid.export import write_jsonl

try:
    from tests.test_eval_diagnosis_pipeline import _expected_labels, _load_jsonl, _run_case
    PIPELINE = True
except ImportError:                                   # #99 머지 전
    PIPELINE = False


@unittest.skipUnless(PIPELINE, "진단 파이프라인(#99)이 필요하다")
class DiagnoseGridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        path = os.path.join(cls._dir.name, "diagnose_grid_g3.jsonl")
        write_jsonl(CASES, path)
        cls.rows = _load_jsonl(__import__("pathlib").Path(path))
        cls.gap = {c.id: c.known_gap for c in CASES if c.known_gap}
        cls.needs_judge = {c.id: c.needs_judge for c in CASES if c.needs_judge}

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def test_labels_match_expectation(self):
        for row in self.rows:
            case_id = row["case_id"]
            with self.subTest(case=case_id):
                if case_id in self.needs_judge:
                    self.skipTest(self.needs_judge[case_id])
                actual = sorted(_run_case(row))
                expected = sorted(_expected_labels(row))
                if case_id in self.gap:
                    self.assertNotEqual(
                        expected, actual,
                        f"known_gap 인데 통과했다 — 진단이 고쳐졌으면 케이스에서 known_gap 을 "
                        f"지울 것: {self.gap[case_id]}")
                else:
                    self.assertEqual(expected, actual)

    def test_cases_carry_situation(self):
        """상황 서술이 있어야 기대 라벨의 근거를 검수할 수 있다."""
        for row in self.rows:
            with self.subTest(case=row["case_id"]):
                self.assertTrue((row.get("situation") or "").strip())


if __name__ == "__main__":
    unittest.main()
