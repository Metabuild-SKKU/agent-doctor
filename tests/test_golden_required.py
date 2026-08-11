"""
tests/test_golden_required.py
골든셋을 진단의 기본 입력으로 요구하는 경로 (멘토 피드백 2026-08-07 §1).

왜 필요한가: 골든셋 없이 돌리면 검색축 라벨 3종이 침묵하고 환각도 예비에 머무는데,
그 사실을 모른 채 "진단 받았다"고 오해하는 것이 가장 나쁘다. CLI 가 기본적으로
거부하고, --no-golden 으로 명시해야만 얕은 진단을 내준다(contexts 파일 게이트와
같은 구조).
"""
import json
import tempfile
import unittest
from pathlib import Path

from agents.eval.log_intake import load_external_log
from agents.eval.replay import _main, apply_golden_set

GOLD = "입사 1년 이상 직원은 연 15일의 연차 휴가가 부여되며, 3년 이상 근속 시 추가된다."


def _write(tmp: Path, name: str, rows: list[dict]) -> str:
    path = tmp / name
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                    encoding="utf-8")
    return str(path)


def _log_rows():
    return [{"question": "연차 휴가는 며칠인가요?", "answer": "15일입니다.",
             "contexts": ["인사 제도 안내 " + GOLD]}]


class ApplyGoldenSetTests(unittest.TestCase):
    """메모리 병합 — 진단 한 번에 중간 파일을 만들지 않는다."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _golden(self, entries):
        path = self.tmp / "golden.json"
        path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def test_fills_ground_truth_and_gold_contexts(self):
        logs, _ = load_external_log(_write(self.tmp, "log.jsonl", _log_rows()))
        stats = apply_golden_set(logs, self._golden([
            {"question": "연차 휴가는 며칠인가요?", "ground_truth": "연 15일",
             "gold_contexts": [GOLD]}]))
        self.assertEqual(stats["matched"], 1)
        self.assertEqual(logs[0].ground_truth, "연 15일")
        self.assertEqual(logs[0].gold_contexts, [GOLD])

    def test_question_normalization_matches(self):
        """공백·문장부호·대소문자 차이는 흡수한다 — 골든셋과 로그의 표기가
        완전히 같기를 기대하면 매칭이 대부분 실패한다."""
        logs, _ = load_external_log(_write(self.tmp, "log.jsonl", _log_rows()))
        stats = apply_golden_set(logs, self._golden([
            {"question": "  연차  휴가는 며칠인가요  ", "ground_truth": "연 15일"}]))
        self.assertEqual(stats["matched"], 1)

    def test_existing_value_is_not_overwritten(self):
        """로그 제공자가 직접 넣은 값을 신뢰한다 — 덮지 않고 충돌로 집계."""
        rows = _log_rows()
        rows[0]["ground_truth"] = "로그가 준 정답"
        logs, _ = load_external_log(_write(self.tmp, "log.jsonl", rows))
        stats = apply_golden_set(logs, self._golden([
            {"question": "연차 휴가는 며칠인가요?", "ground_truth": "골든셋 정답"}]))
        self.assertEqual(logs[0].ground_truth, "로그가 준 정답")
        self.assertEqual(stats["conflicts"], 1)

    def test_unmatched_entries_counted(self):
        """못 맞춘 항목을 조용히 버리면 '골든셋 줬으니 됐다'고 오해하게 된다."""
        logs, _ = load_external_log(_write(self.tmp, "log.jsonl", _log_rows()))
        stats = apply_golden_set(logs, self._golden([
            {"question": "로그에 없는 질문", "ground_truth": "x"}]))
        self.assertEqual(stats["matched"], 0)
        self.assertEqual(stats["qa_entries"], 1)


class GoldenRequiredCliTests(unittest.TestCase):
    """CLI 게이트 — 기본은 거부, --no-golden 으로만 옵트인."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.log = _write(self.tmp, "log.jsonl", _log_rows())

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_golden_is_rejected(self):
        self.assertEqual(_main([self.log]), 2)

    def test_no_golden_opt_in_proceeds(self):
        """명시적으로 포기하면 돈다(RAGAS 없이도 규칙 지표는 계산되므로 0)."""
        self.assertEqual(_main([self.log, "--no-golden"]), 0)

    def test_golden_path_proceeds(self):
        golden = self.tmp / "golden.json"
        golden.write_text(json.dumps(
            [{"question": "연차 휴가는 며칠인가요?", "ground_truth": "연 15일"}],
            ensure_ascii=False), encoding="utf-8")
        self.assertEqual(_main([self.log, f"--golden={golden}"]), 0)


if __name__ == "__main__":
    unittest.main()
