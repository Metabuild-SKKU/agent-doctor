"""
tests/test_parallel.py
core/parallel.py::parallel_map 의 계약과 진행률 출력을 고정한다.

왜 필요한가:
    진행률을 붙이면서 구현이 pool.map → submit + as_completed 로 바뀌었다. pool.map 이
    공짜로 주던 성질(입력 순서 보존)이 이제 직접 인덱스를 들고 다녀야 성립하고,
    깨져도 **조용히** 깨진다 — 결과가 섞여도 예외가 안 나고, RAGAS
    context_precision 의 순위 가중 평균 같은 소비처에서 점수만 미묘하게 틀어진다.

    docstring 에 적힌 세 계약을 여기서 실행으로 못 박는다:
      1. 결과는 입력 순서 그대로
      2. max_workers <= 1 이면 executor 없이 순수 순차 (EVAL_LLM_CONCURRENCY=1 kill-switch)
      3. 워커 예외는 삼키지 않고 전파
"""
from __future__ import annotations

import os
import random
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import progress
from core.parallel import parallel_map


class _Boom(RuntimeError):
    pass


class ParallelMapContractTest(unittest.TestCase):
    """진행률과 무관하게 지켜야 하는 계약."""

    def test_결과는_입력_순서를_보존한다(self):
        """완료 순서를 입력 순서와 정반대로 강제해도 결과 순서는 입력 순서여야 한다."""
        items = list(range(12))

        def slow_in_reverse(x: int) -> int:
            # 뒤 항목일수록 빨리 끝난다 → as_completed 순서 = 입력 역순.
            time.sleep((len(items) - x) * 0.005)
            return x * 10

        self.assertEqual(parallel_map(slow_in_reverse, items, max_workers=8),
                         [x * 10 for x in items])

    def test_무작위_완료순서에서도_순서를_보존한다(self):
        rng = random.Random(1234)
        items = [f"item-{i}" for i in range(30)]

        def jittered(text: str) -> str:
            time.sleep(rng.random() * 0.01)
            return text.upper()

        self.assertEqual(parallel_map(jittered, items, max_workers=8),
                         [t.upper() for t in items])

    def test_동시성_1이면_executor_를_만들지_않는다(self):
        """kill-switch(EVAL_LLM_CONCURRENCY=1)는 '스레드 1개'가 아니라 '스레드 없음'이다."""
        items = list(range(5))
        with mock.patch("core.parallel.ThreadPoolExecutor") as pool:
            result = parallel_map(lambda x: x + 1, items, max_workers=1)
        pool.assert_not_called()
        self.assertEqual(result, [1, 2, 3, 4, 5])

    def test_동시성_1이면_호출_스레드에서_입력_순서대로_실행한다(self):
        seen: list[int] = []
        threads: set[int] = set()

        def record(x: int) -> int:
            seen.append(x)
            threads.add(threading.get_ident())
            return x

        parallel_map(record, [3, 1, 2], max_workers=1)
        self.assertEqual(seen, [3, 1, 2])
        self.assertEqual(threads, {threading.get_ident()})

    def test_항목이_하나면_executor_를_만들지_않는다(self):
        with mock.patch("core.parallel.ThreadPoolExecutor") as pool:
            self.assertEqual(parallel_map(lambda x: x * 2, [7], max_workers=8), [14])
        pool.assert_not_called()

    def test_빈_입력은_빈_리스트(self):
        self.assertEqual(parallel_map(lambda x: x, [], max_workers=8), [])

    def test_이터레이터도_받는다(self):
        self.assertEqual(parallel_map(lambda x: x + 1, iter([1, 2, 3]), max_workers=4),
                         [2, 3, 4])

    def test_워커_예외는_병렬_경로에서_전파된다(self):
        def maybe_boom(x: int) -> int:
            if x == 5:
                raise _Boom("워커 실패")
            time.sleep(0.01)
            return x

        with self.assertRaises(_Boom):
            parallel_map(maybe_boom, list(range(10)), max_workers=4)

    def test_워커_예외는_순차_경로에서도_전파된다(self):
        def boom(_x):
            raise _Boom("순차 실패")

        with self.assertRaises(_Boom):
            parallel_map(boom, [1, 2], max_workers=1)

    def test_예외를_결과로_삼키지_않는다(self):
        """예전 버그 방지 — as_completed 쪽에서 잡아 로그만 찍고 None 을 넣으면 안 된다."""
        def boom_on_last(x: int) -> int:
            if x == 9:
                raise _Boom("마지막만 실패")
            return x

        try:
            result = parallel_map(boom_on_last, list(range(10)), max_workers=4)
        except _Boom:
            return
        self.fail(f"예외가 전파되지 않고 결과로 흡수됐다: {result}")


class ParallelMapProgressTest(unittest.TestCase):
    """label 인자와 진행률 출력."""

    def setUp(self):
        # 주기를 0 이하로 두면 기본값으로 되돌아가므로, '항상 찍히는' 상태는
        # 아주 작은 양수로 만든다.
        self._env = {k: os.environ.get(k) for k in ("PROGRESS_LOG", "PROGRESS_INTERVAL_SEC")}

    def tearDown(self):
        for name, value in self._env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    @staticmethod
    def _slow(x):
        # 항목마다 최소 주기(아래 테스트의 0.001초)보다 오래 걸려야 진행줄이 나온다.
        time.sleep(0.005)
        return x

    def _run(self, **env) -> list[str]:
        os.environ.update(env)
        lines: list[str] = []
        with mock.patch("builtins.print", side_effect=lambda *a, **k: lines.append(" ".join(map(str, a)))):
            result = parallel_map(self._slow, list(range(6)), max_workers=1, label="[테스트] 단계")
        self.assertEqual(result, list(range(6)))
        return lines

    def test_라벨을_안_주면_아무것도_찍지_않는다(self):
        os.environ["PROGRESS_INTERVAL_SEC"] = "0.001"
        lines: list[str] = []
        with mock.patch("builtins.print", side_effect=lambda *a, **k: lines.append(" ".join(map(str, a)))):
            parallel_map(lambda x: x, list(range(6)), max_workers=2)
        self.assertEqual(lines, [], "라벨 없는 호출은 기존처럼 조용해야 한다")

    def test_주기가_지나면_진행률과_완료줄을_찍는다(self):
        lines = self._run(PROGRESS_INTERVAL_SEC="0.001", PROGRESS_LOG="1")
        self.assertTrue(lines, "주기가 지났는데도 아무것도 안 찍혔다")
        self.assertTrue(all(line.startswith("[테스트] 단계") for line in lines), lines)
        self.assertIn("6/6 (100%)", lines[-1])
        self.assertIn("완료", lines[-1])

    def test_주기가_안_지나면_완전히_침묵한다(self):
        """짧은 단계는 완료줄조차 안 찍는다 — 기존 출력에 잡음을 더하지 않기 위해서."""
        self.assertEqual(self._run(PROGRESS_INTERVAL_SEC="3600", PROGRESS_LOG="1"), [])

    def test_PROGRESS_LOG_0_이면_꺼진다(self):
        self.assertEqual(self._run(PROGRESS_INTERVAL_SEC="0.001", PROGRESS_LOG="0"), [])

    def test_off_와_false_도_끄는_값이다(self):
        for value in ("off", "false", "FALSE", "Off"):
            with self.subTest(value=value):
                self.assertEqual(self._run(PROGRESS_INTERVAL_SEC="0.001", PROGRESS_LOG=value), [])

    def test_병렬_경로에서도_진행률이_나오고_순서는_유지된다(self):
        os.environ.update(PROGRESS_LOG="1", PROGRESS_INTERVAL_SEC="0.001")
        lines: list[str] = []
        with mock.patch("builtins.print", side_effect=lambda *a, **k: lines.append(" ".join(map(str, a)))):
            result = parallel_map(lambda x: (time.sleep(0.005), x * 2)[1], list(range(20)),
                                  max_workers=4, label="[테스트] 병렬")
        self.assertEqual(result, [x * 2 for x in range(20)])
        self.assertIn("20/20 (100%)", lines[-1])

    def test_진행줄은_줄바꿈으로_끝난다(self):
        """`\\r` 덮어쓰기 바를 쓰면 _Tee 가 로그 파일에 프레임을 그대로 박는다."""
        lines = self._run(PROGRESS_INTERVAL_SEC="0.001", PROGRESS_LOG="1")
        for line in lines:
            self.assertNotIn("\r", line, f"진행줄에 캐리지 리턴이 있다: {line!r}")

    def test_출력_문구가_cp949_에서_깨지지_않는다(self):
        """Windows 콘솔(cp949)에서 '????' 로 보이면 진행률의 의미가 없다."""
        lines = self._run(PROGRESS_INTERVAL_SEC="0.001", PROGRESS_LOG="1")
        self.assertTrue(lines)
        for line in lines:
            # 라벨의 한글은 테스트가 준 것이고, 여기서 보는 건 progress.py 가 붙이는
            # 장식 문자다. cp949 불가 문자가 하나라도 있으면 실패.
            line.encode("cp949")


class ProgressFormatTest(unittest.TestCase):
    """core/progress.py 의 표시 규칙."""

    def test_소요시간_표기(self):
        self.assertEqual(progress._fmt_duration(0), "0s")
        self.assertEqual(progress._fmt_duration(45), "45s")
        self.assertEqual(progress._fmt_duration(59), "59s")
        self.assertEqual(progress._fmt_duration(60), "1m00s")
        self.assertEqual(progress._fmt_duration(90), "1m30s")
        self.assertEqual(progress._fmt_duration(552), "9m12s")

    def test_잘못된_주기는_기본값으로_흘린다(self):
        """진행률 설정 오타 때문에 파이프라인이 죽으면 안 된다."""
        for bad in ("", "abc", "0", "-5"):
            with mock.patch.dict(os.environ, {"PROGRESS_INTERVAL_SEC": bad}):
                self.assertEqual(progress._interval_sec(), progress._DEFAULT_INTERVAL_SEC)

    def test_유효한_주기는_그대로_읽는다(self):
        with mock.patch.dict(os.environ, {"PROGRESS_INTERVAL_SEC": "2.5"}):
            self.assertEqual(progress._interval_sec(), 2.5)

    def test_총량이_0이면_리포터를_만들지_않는다(self):
        with mock.patch.dict(os.environ, {"PROGRESS_LOG": "1"}):
            self.assertIsNone(progress.start("[테스트]", 0))

    def test_None_안전_헬퍼(self):
        progress.tick(None)      # 예외 없이 통과해야 한다
        progress.finish(None)


if __name__ == "__main__":
    unittest.main()
