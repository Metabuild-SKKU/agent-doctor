"""
tests/test_golden_required.py
골든셋을 진단의 기본 입력으로 요구하는 경로 (멘토 피드백 2026-08-07 §1).

왜 필요한가: 골든셋 없이 돌리면 검색축 라벨 3종이 침묵하고 환각도 예비에 머무는데,
그 사실을 모른 채 "진단 받았다"고 오해하는 것이 가장 나쁘다. CLI 가 기본적으로
거부하고, --no-golden 으로 명시해야만 얕은 진단을 내준다(contexts 파일 게이트와
같은 구조).
"""
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agents.eval import replay
from agents.eval.log_intake import load_external_log
from agents.eval.replay import _main, apply_golden_set

# _main 이 CLI 규약대로 load_dotenv(override=True) 를 부르므로, 실사용 .env 에
# EVAL_ENABLE_LLM=1 이 있으면(정상적인 설정) @patch.dict(os.environ, ...) 만으로는
# 못 막는다 - override=True 가 패치를 되돌린다(재현됨). llm_eval_enabled 자체를
# 패치해야 어떤 .env 에서도 이 테스트 파일이 실제 LLM 을 부르지 않는다.
def _no_llm():
    return patch.object(replay, "llm_eval_enabled", return_value=False)

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
        self._llm_patch = _no_llm()
        self._llm_patch.start()
        self.addCleanup(self._llm_patch.stop)

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

    def test_inline_ground_truth_satisfies_gate(self):
        """문서·시뮬레이터가 안내하는 --golden 없는 명령이, 로그 자체에 이미
        ground_truth 가 있으면 거부돼선 안 된다(실행 재현된 버그)."""
        rows = _log_rows()
        rows[0]["ground_truth"] = "연 15일"
        log = _write(self.tmp, "inline_gt.jsonl", rows)
        self.assertEqual(_main([log]), 0)

    def test_inline_gold_contexts_satisfies_gate(self):
        rows = _log_rows()
        rows[0]["gold_contexts"] = [GOLD]
        log = _write(self.tmp, "inline_gc.jsonl", rows)
        self.assertEqual(_main([log]), 0)

    def test_no_inline_golden_still_rejected(self):
        """인라인 필드가 하나도 없으면 종전대로 거부해야 한다(과한 완화 방지)."""
        self.assertEqual(_main([self.log]), 2)


class GoldenParseErrorReportingTests(unittest.TestCase):
    """골든셋 파싱 오류는 개수만이 아니라 내용도 나와야 한다(openpyxl 미설치 등이
    '0건 매칭'으로 은폐되던 문제)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.log = _write(self.tmp, "log.jsonl", _log_rows())
        self._llm_patch = _no_llm()
        self._llm_patch.start()
        self.addCleanup(self._llm_patch.stop)

    def tearDown(self):
        self._tmp.cleanup()

    def test_broken_golden_file_prints_error_content(self):
        golden = self.tmp / "golden.jsonl"
        golden.write_text('{"question": "깨진 줄"', encoding="utf-8")  # 닫는 괄호 없음
        buf = io.StringIO()
        with redirect_stdout(buf):
            _main([self.log, f"--golden={golden}"])
        self.assertIn("골든셋 파싱 오류", buf.getvalue())


class GoldenSetSizeCapTests(unittest.TestCase):
    """골든셋은 사람이 채워야 하는 필드(ground_truth/gold_contexts)라 크기가 커질수록
    상대 팀 부담·리플레이 LLM 비용(매칭된 레코드마다 context precision/recall/
    correctness 추가 측정)이 함께 는다. 권장 범위를 안내하고, 과도하게 크면
    비용을 쓰기 전에 막는다."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.log = _write(self.tmp, "log.jsonl", _log_rows())
        self._llm_patch = _no_llm()
        self._llm_patch.start()
        self.addCleanup(self._llm_patch.stop)

    def tearDown(self):
        self._tmp.cleanup()

    def _golden_with(self, n: int) -> str:
        path = self.tmp / "golden.json"
        entries = [{"question": f"질문 {i}", "ground_truth": f"정답 {i}"} for i in range(n)]
        path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def test_recommendation_is_shown(self):
        golden = self._golden_with(1)
        buf = io.StringIO()
        with redirect_stdout(buf):
            _main([self.log, f"--golden={golden}"])
        self.assertIn("권장", buf.getvalue())

    def test_within_hard_cap_proceeds(self):
        golden = self._golden_with(150)
        self.assertEqual(_main([self.log, f"--golden={golden}"]), 0)

    def test_over_hard_cap_is_rejected(self):
        golden = self._golden_with(301)
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = _main([self.log, f"--golden={golden}"])
        self.assertEqual(code, 2)
        self.assertIn("300", buf.getvalue())


class LlmCallIsolationTests(unittest.TestCase):
    """이 테스트 파일이 실사용 .env(EVAL_ENABLE_LLM=1 + 유효 키)에서도 실제 LLM 을
    부르면 안 된다. _main 은 CLI 규약대로 load_dotenv(override=True) 를 부르므로,
    @patch.dict(os.environ, {"EVAL_ENABLE_LLM": "0"}) 로 걸어 둬도 .env 에 값이
    있으면 override 가 되돌려 버린다(재현: load_dotenv 가 패치된 "0" 을 다시 "1"
    로 덮어씀). llm_eval_enabled 자체를 패치하는 것만이 .env 내용과 무관하게 안전하다."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.log = _write(self.tmp, "log.jsonl", _log_rows())
        golden = self.tmp / "golden.json"
        golden.write_text(json.dumps(
            [{"question": "연차 휴가는 며칠인가요?", "ground_truth": "연 15일"}],
            ensure_ascii=False), encoding="utf-8")
        self.golden = str(golden)

    def tearDown(self):
        self._tmp.cleanup()

    def test_hostile_env_still_skips_judge(self):
        """os.environ 에 EVAL_ENABLE_LLM=1 이 이미 있어도(정상적인 .env 설정 시나리오)
        _judge 호출까지 가면 안 된다 - llm_eval_enabled 패치가 진짜 방어선인지 확인."""
        with patch.dict(os.environ, {"EVAL_ENABLE_LLM": "1"}), \
             _no_llm(), \
             patch.object(replay, "_judge") as mock_judge:
            code = _main([self.log, f"--golden={self.golden}"])
        mock_judge.assert_not_called()
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
