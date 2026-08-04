"""`import core` 만으로 stdout 인코딩이 UTF-8 로 확정되는지 검증.

cp949 환경에서 '—' 같은 문자를 print 하면 UnicodeEncodeError 가 나고, 그게 포괄
except 에 잡혀 정상 실행이 error 로 뒤집힌다. core/__init__.py 의 방어가 사라지면
여기서 잡힌다. 현재 프로세스 stdout 을 건드리지 않도록 서브프로세스로 확인한다.
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _run(code: str) -> subprocess.CompletedProcess:
    # errors="replace": 자식은 PYTHONIOENCODING=cp949 라 stderr 에 cp949 바이트를 낸다.
    # Python 3.11+ 의 traceback 은 실패한 소스 줄(한글 포함)을 그대로 stderr 에 echo 하므로,
    # 부모가 utf-8 로만 디코딩하면 UnicodeDecodeError 로 stderr 가 None 이 돼 판정이 깨진다.
    # 판정 대상 문자열("UnicodeEncodeError" 등)은 ASCII 라 대체문자로 바뀌어도 온전히 남는다.
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        env={**os.environ, "PYTHONIOENCODING": "cp949"},
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


class CoreImportFixesEncodingTest(unittest.TestCase):

    def test_import_core_sets_utf8_stdout(self):
        out = _run("import core, sys; print(sys.stdout.encoding)")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip().lower(), "utf-8")

    def test_em_dash_print_survives_cp949_env(self):
        out = _run("import core; print('진단 — 완료')")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("—", out.stdout)

    def test_baseline_without_core_would_fail(self):
        # 방어가 없으면 실제로 깨지는 환경인지 확인 — 이 전제가 무너지면 위 두 테스트가 무의미하다.
        out = _run("print('진단 — 완료')")
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("UnicodeEncodeError", out.stderr)


if __name__ == "__main__":
    unittest.main()
