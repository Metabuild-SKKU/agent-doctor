"""처방 효과 측정 도구의 무과금 dry-run 계약."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools import measure_prescription_effects as measure


class PrescriptionEffectsDryRunTests(unittest.TestCase):
    def test_uses_corrected_sigma_scale(self):
        self.assertEqual(measure.SIGMA_DELTA_AT_100_DISPLAY, 1.20)

    def test_dry_run_forces_cpu_into_every_worker_command(self):
        calls = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        with tempfile.TemporaryDirectory() as tmp, \
             patch("dotenv.load_dotenv"), \
             patch.object(subprocess, "run", side_effect=fake_run), \
             patch.object(sys, "argv", [
                 "measure_prescription_effects.py",
                 "--axis", "top_k=10",
                 "--repeat", "1",
                 "--dry-run",
                 "--run",
                 "--outdir", str(Path(tmp) / "effects"),
             ]):
            self.assertEqual(measure.main(), 0)

        self.assertEqual(len(calls), 2)  # baseline + top_k cell
        for command in calls:
            self.assertIn("--dry-run", command)
            embed_at = command.index("--embed")
            self.assertEqual(command[embed_at + 1], "cpu")

    def test_stub_closes_probe_generation_key_gate(self):
        import agents.eval.agent as eval_agent
        from agents.eval import llm_provider

        old_generate = eval_agent.generate_answer
        old_has_key = llm_provider.has_key
        try:
            measure._stub_generation()
            self.assertFalse(llm_provider.has_key())
            self.assertTrue(eval_agent.generate_answer("Q", ["context"]).startswith("[dry-run]"))
        finally:
            eval_agent.generate_answer = old_generate
            llm_provider.has_key = old_has_key


if __name__ == "__main__":
    unittest.main()
