"""
tests/test_progress_eta.py
진행률 ETA 억제(eta=False)의 계약 고정.

anthropic 배치처럼 전 항목이 한꺼번에 끝나는 fan-out 에서는 첫 완료 기준의 평균 속도
외삽이 수십 배로 틀린다(실측: 1/100 시점 '남은 약 285m' → 실제 4.7분, #131).
eta=False 는 그 추정 문구만 끄고 경과·백분율은 유지해야 한다.
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import progress


def _emit_line(eta: bool) -> str:
    reporter = progress.Progress("라벨", 10, min_interval_sec=0.0, eta=eta)
    buf = io.StringIO()
    with redirect_stdout(buf):
        reporter.tick()          # 1/10 — 진행 줄 1개
    return buf.getvalue()


class ProgressEtaTest(unittest.TestCase):
    def test_eta_off_drops_remaining_but_keeps_elapsed(self):
        line = _emit_line(eta=False)
        self.assertIn("1/10", line)
        self.assertIn("경과", line)
        self.assertNotIn("남은", line)

    def test_eta_on_keeps_remaining_estimate(self):
        self.assertIn("남은", _emit_line(eta=True))

    def test_parallel_map_passes_eta_through(self):
        """parallel_map(eta=False) 이 리포터까지 전달되는지 — 배선이 끊기면 배치 실행의
        ETA 억제가 조용히 무효가 된다."""
        import time
        from unittest.mock import patch
        from core.parallel import parallel_map
        buf = io.StringIO()
        env = {"PROGRESS_LOG": "1", "PROGRESS_MIN_INTERVAL_SEC": "0.001"}
        with patch.dict(os.environ, env), redirect_stdout(buf):
            # 항목당 소요가 최소 간격(1ms)을 넘어야 진행 줄이 나간다 — 즉시 끝나면
            # '침묵했으면 끝까지 침묵' 규약대로 아무것도 안 찍힌다.
            parallel_map(lambda x: time.sleep(0.005), list(range(20)), 1,
                         label="L", eta=False)
        out = buf.getvalue()
        self.assertIn("L", out)
        self.assertNotIn("남은", out)


if __name__ == "__main__":
    unittest.main()
