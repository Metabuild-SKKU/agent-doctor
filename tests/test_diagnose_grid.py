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
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.eval import metrics_common
from agents.eval.types import Mode
from tests.diagnose_grid.builder import korquad_available
from tests.diagnose_grid.cases_g3 import CASES
from tests.diagnose_grid.export import write_jsonl

try:
    from tests.test_eval_diagnosis_pipeline import _expected_labels, _load_jsonl, _run_case
    PIPELINE = True
except ImportError:                                   # #99 머지 전
    PIPELINE = False


# 케이스 문서를 KorQuAD 에서 가져오므로 data/ 가 있어야 빌드된다. 없으면 setUpClass 가
# FileNotFoundError 로 죽어 스위트 전체가 error 로 뜬다 — data/ 는 gitignore 대상이라
# (data/README.md) 없는 게 정상이므로 skip 으로 다룬다.
@unittest.skipUnless(PIPELINE, "진단 파이프라인(#99)이 필요하다")
@unittest.skipUnless(korquad_available(), "KorQuAD 데이터셋이 필요하다 — data/README.md 참고")
class DiagnoseGridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        path = os.path.join(cls._dir.name, "diagnose_grid_g3.jsonl")
        write_jsonl(CASES, path)
        cls.rows = _load_jsonl(pathlib.Path(path))
        cls.gap = {c.id: c.known_gap for c in CASES if c.known_gap}
        cls.needs_judge = {c.id: c.needs_judge for c in CASES if c.needs_judge}

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()
        # 전역 측정 컨텍스트를 되돌린다. write_jsonl(build) 과 _run_case 가 set_context·
        # set_mode 를 채워두는데, 남기면 뒤에 도는 테스트의 판정이 바뀐다 — 이 저장소가
        # 이미 한 번 당한 실패다(tests/test_suite_hygiene.py 의 corpus_ids 오염 사례).
        metrics_common.set_context()
        metrics_common.set_mode(Mode.FAST)

    def test_labels_match_expectation(self):
        for row in self.rows:
            case_id = row["case_id"]
            with self.subTest(case=case_id):
                if case_id in self.needs_judge:
                    self.skipTest(self.needs_judge[case_id])
                actual = sorted(_run_case(row))
                expected = sorted(_expected_labels(row))
                if case_id not in self.gap:
                    self.assertEqual(expected, actual)
                    continue

                # known_gap: 아직 기대를 못 맞춘다. 두 방향을 다 건다 —
                #   (1) 여전히 기대와 다른가  → 고쳐지면 알려준다
                #   (2) '지금 내는 라벨'이 그대로인가 → 다른 방식으로 틀려도(회귀) 잡는다
                self.assertNotEqual(
                    expected, actual,
                    f"known_gap 인데 통과했다 — 진단이 고쳐졌으면 케이스에서 known_gap 을 "
                    f"지울 것: {self.gap[case_id]}")
                pinned = row.get("known_gap_labels")
                self.assertIsNotNone(
                    pinned,
                    f"known_gap 케이스는 known_gap_labels 로 현재 라벨을 고정해야 한다 "
                    f"(지금 값: {actual})")
                self.assertEqual(
                    sorted(pinned), actual,
                    f"known_gap 케이스의 현재 라벨이 바뀌었다. 진단 동작이 달라졌다는 뜻이니 "
                    f"의도한 변화면 known_gap_labels 를 {actual} 로 갱신할 것")

    def test_cases_carry_situation(self):
        """상황 서술이 있어야 기대 라벨의 근거를 검수할 수 있다."""
        for row in self.rows:
            with self.subTest(case=row["case_id"]):
                self.assertTrue((row.get("situation") or "").strip())


if __name__ == "__main__":
    unittest.main()
