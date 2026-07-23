"""에이전트 공통 시간 측정 로그 계약 테스트."""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from core.timing import StageTimer


class StageTimerTests(unittest.TestCase):
    def test_stage_and_total_are_logged(self):
        output = io.StringIO()

        with redirect_stdout(output):
            timer = StageTimer("Test")
            with timer.measure("단계"):
                pass
            timer.finish()

        log = output.getvalue()
        self.assertIn("[Test] 시간 | 단계:", log)
        self.assertIn("[Test] 시간 | 전체:", log)

    def test_total_is_logged_only_once(self):
        output = io.StringIO()

        with redirect_stdout(output):
            timer = StageTimer("Test")
            timer.finish()
            timer.finish()

        self.assertEqual(output.getvalue().count("시간 | 전체"), 1)


if __name__ == "__main__":
    unittest.main()
