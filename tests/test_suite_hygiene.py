"""
tests/test_suite_hygiene.py
`tests/test_*.py` 가 전부 실제 테스트를 담고 있는지 고정한다.

왜 필요한가:
    pytest 는 `test_*.py` 를 **import 해서** 수집한다. 그래서 최상위에 실행 코드가 있는
    수동 점검 스크립트를 그 이름으로 두면, 수집만 해도 파이프라인이 통째로 돌아간다.

    실제로 그랬다. `tests/test_eval.py` 는 테스트 함수가 0 개인 스크립트인데 이름 때문에
    수집돼, import 시점에 BGE-M3 를 내려받고 Eval 을 실행했다(전체 스위트 494초 중 대부분).
    비용도 문제지만 **전역 상태를 남기는 게 더 나쁘다** — 그 실행이 채운
    `metrics_search._ctx.corpus_ids` 때문에 뒤에 도는 `test_bad_gold_chunk` 의 가짜 골드가
    '코퍼스에 없는 골드'로 판정돼, 단독으로는 통과하는 테스트가 조합에서만 깨졌다.

        corpus_ids 비었을 때(단독) : ['bad_gold_chunk']
        corpus_ids 오염됐을 때     : ['corpus_gap', 'generation_failure']

    실행 순서에 따라 결과가 갈리므로 원인을 찾기 어렵고, 그 사이 진짜 회귀가 묻힌다.
    수동 스크립트는 `check_*.py` 로 둔다(이 저장소의 `run_corpus.py`·`verify_*.py` 와 같은 결).

판정 방식:
    import 하지 않고 AST 로만 본다 — 이 검사가 바로 그 import 를 문제 삼는데 스스로 import
    하면 같은 부작용을 일으킨다.
"""
from __future__ import annotations

import ast
import pathlib
import unittest

TESTS_DIR = pathlib.Path(__file__).resolve().parent


def _has_collectable_test(tree: ast.AST) -> bool:
    """pytest·unittest 어느 쪽이든 실제로 수집될 것이 하나라도 있나."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test"):
                return True
        elif isinstance(node, ast.ClassDef):
            # unittest.TestCase / TestCase / 프로젝트 공용 베이스 모두 이름으로 잡는다.
            for base in node.bases:
                name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
                if "TestCase" in name:
                    return True
    return False


# 최상위에서 허용하는 호출. 이 저장소의 test_*.py 가 실제로 쓰는 건 이거 하나뿐이라
# (62개 중 44개, 다른 최상위 호출은 0개) 예외를 넓게 열 이유가 없다.
_ALLOWED_TOPLEVEL_CALLS = ("sys.path.insert(",)


def _toplevel_side_effects(tree: ast.Module) -> list[str]:
    """import 만으로 실행되는 최상위 호출. `if __name__` 가드 안이나 함수 안은 안 센다."""
    out: list[str] = []
    for node in tree.body:
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
            continue
        src = ast.unparse(node.value)
        if not src.startswith(_ALLOWED_TOPLEVEL_CALLS):
            out.append(src.splitlines()[0][:50])
    return out


class TestFileNamingTest(unittest.TestCase):
    def test_every_test_file_actually_contains_tests(self):
        offenders: list[str] = []
        for path in sorted(TESTS_DIR.glob("test_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            if not _has_collectable_test(tree):
                offenders.append(path.name)

        self.assertEqual(
            offenders, [],
            "테스트가 없는 파일이 test_*.py 로 있습니다. pytest 가 import 만으로 최상위 코드를"
            " 실행해 시간을 태우고 전역 상태를 오염시킵니다. 수동 점검 스크립트라면"
            " check_*.py 로 이름을 바꾸세요: " + ", ".join(offenders),
        )

    def test_no_test_file_runs_anything_on_import(self):
        """'테스트가 있나'만 보면 스크립트에 테스트 하나만 붙어도 다시 새어 들어온다.
        수집을 유발하는 건 이름이지만 시간을 태우고 전역 상태를 더럽히는 건 최상위 실행이다."""
        offenders: list[str] = []
        for path in sorted(TESTS_DIR.glob("test_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            calls = _toplevel_side_effects(tree)
            if calls:
                offenders.append(f"{path.name}({', '.join(calls[:2])})")

        self.assertEqual(
            offenders, [],
            "import 만으로 실행되는 최상위 코드가 있습니다. `if __name__ == '__main__':` 안으로"
            " 옮기거나, 수동 점검 스크립트라면 check_*.py 로 이름을 바꾸세요: "
            + ", ".join(offenders),
        )

    def test_detector_recognizes_both_styles(self):
        """탐지기가 조용히 '전부 통과'로 굳는 회귀를 막는다."""
        self.assertTrue(_has_collectable_test(ast.parse("def test_x():\n    pass\n")))
        self.assertTrue(_has_collectable_test(
            ast.parse("import unittest\nclass T(unittest.TestCase):\n    pass\n")))
        self.assertTrue(_has_collectable_test(
            ast.parse("from unittest import TestCase\nclass T(TestCase):\n    pass\n")))

    def test_detector_rejects_a_bare_script(self):
        self.assertFalse(_has_collectable_test(
            ast.parse("print('실행됨')\nrun_everything()\n")))

    def test_side_effect_detector_catches_a_script_that_also_has_a_test(self):
        """이 조합이 첫 탐지기의 구멍이다 — 테스트가 있어도 최상위는 그대로 돈다."""
        tree = ast.parse("run_everything()\ndef test_x():\n    pass\n")
        self.assertTrue(_has_collectable_test(tree))          # 첫 검사는 통과시킨다
        self.assertEqual(_toplevel_side_effects(tree), ["run_everything()"])

    def test_side_effect_detector_allows_the_repo_conventions(self):
        self.assertEqual(_toplevel_side_effects(ast.parse(
            "import sys\nsys.path.insert(0, '..')\n")), [])
        self.assertEqual(_toplevel_side_effects(ast.parse(
            "def main():\n    run()\nif __name__ == '__main__':\n    main()\n")), [])

    def test_scan_actually_reads_files(self):
        """glob 이 빈 목록이면 본 테스트가 무의미하게 통과한다."""
        self.assertGreater(len(list(TESTS_DIR.glob("test_*.py"))), 20)


if __name__ == "__main__":
    unittest.main()
