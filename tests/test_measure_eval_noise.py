"""
tests/test_measure_eval_noise.py
tools/measure_eval_noise.py 의 순수 계산부 검증.

[이 파일이 하는 일]
  재측정 편차 도구가 내리는 **판정**이 맞는지 본다. LLM·파이프라인은 타지 않는다 —
  실행부(Ingest/Index/Eval 호출)는 얇은 배관이고, 틀리기 쉬운 곳은 편차 계산과
  "마진 대비 어느 쪽인가" 판정이라 그쪽만 가짜 데이터로 고정한다.

[왜 스케일 비교가 핵심인가]
  composite 는 0~100, MIN_IMPROVEMENT_MARGIN 은 0~1 이다. 둘을 그대로 비교하면
  2.0 >= 0.02 가 되어 **항상** 경고가 뜬다. 이 도구의 존재 이유가 "노이즈가 마진을
  넘는가"를 가르는 것이므로, 스케일을 안 맞추면 도구 자체가 무의미해진다.
  → test_margin_is_compared_on_the_same_scale 가 이걸 못 박는다.
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from tools.measure_eval_noise import _label_counts, _print_summary, _spread


def _row(composite, labels=None, components=None):
    return {
        "composite": composite,
        "overall": None,
        "components": components or {},
        "labels": labels or {},
        "seconds": 1.0,
    }


def _summary(rows, margin=0.02) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_summary(rows, margin)
    return buf.getvalue()


class SpreadTest(unittest.TestCase):
    def test_spread_is_max_minus_min(self):
        self.assertEqual(_spread([75, 73, 74]), 2)

    def test_single_observation_has_no_spread(self):
        """1회 측정으로는 편차를 알 수 없다 — 0 이 아니라 None 이어야 한다.
        0 을 돌려주면 '노이즈 없음'으로 잘못 읽힌다."""
        self.assertIsNone(_spread([75]))
        self.assertIsNone(_spread([]))

    def test_none_values_are_skipped_not_counted_as_zero(self):
        """미측정(None)이 0 으로 취급되면 편차가 75 로 폭증해 거짓 경고를 낸다."""
        self.assertEqual(_spread([75, None, 73]), 2)

    def test_identical_measurements_give_zero(self):
        self.assertEqual(_spread([75, 75, 75]), 0)


class MarginVerdictTest(unittest.TestCase):
    def test_margin_is_compared_on_the_same_scale(self):
        """편차 1점 < 마진 2점(=0.02×100) → 통과 판정.

        스케일을 안 맞추고 1.0 >= 0.02 로 비교하면 경고가 떠서 이 단언이 깨진다.
        """
        out = _summary([_row(75), _row(74), _row(75)])
        self.assertIn("✓", out)
        self.assertNotIn("재보정 필요", out)

    def test_spread_at_margin_warns(self):
        """실측 사건 재현 — 같은 config 가 75/73 으로 나온 경우. 마진과 같으면 경고."""
        out = _summary([_row(75), _row(73), _row(74)])
        self.assertIn("재보정 필요", out)
        self.assertIn("2.0", out)

    def test_recommended_margin_exceeds_observed_spread(self):
        """권고 마진은 관측 폭보다 커야 한다 — 같거나 작으면 또 노이즈에 걸린다."""
        out = _summary([_row(80), _row(70), _row(75)])
        self.assertIn("권고", out)
        self.assertIn("15.0", out)   # 폭 10 × 1.5

    def test_single_run_does_not_claim_a_verdict(self):
        out = _summary([_row(75)])
        self.assertIn("부족", out)
        self.assertNotIn("재보정 필요", out)


class LabelStabilityTest(unittest.TestCase):
    def test_label_appearing_in_only_some_runs_is_flagged(self):
        """같은 config 인데 라벨이 나타났다 사라지면 처방 후보가 회차마다 달라진다."""
        out = _summary([
            _row(75, {"retrieval_low_rank": 7}),
            _row(75, {"retrieval_low_rank": 7, "generation_contradiction": 1}),
        ])
        self.assertIn("나타났다 사라짐", out)
        self.assertIn("라벨 1개가", out)

    def test_stable_labels_are_not_flagged(self):
        out = _summary([
            _row(75, {"bad_gold_chunk": 4}),
            _row(75, {"bad_gold_chunk": 4}),
        ])
        self.assertNotIn("나타났다 사라짐", out)
        self.assertNotIn("건수 흔들림", out)

    def test_count_drift_is_distinguished_from_appearing(self):
        """7→6 은 흔들림이지 '사라짐'이 아니다. 둘을 같은 문구로 묶으면
        '라벨이 통째로 없어졌다'는 더 심각한 사건이 묻힌다."""
        out = _summary([
            _row(75, {"retrieval_low_rank": 7}),
            _row(75, {"retrieval_low_rank": 6}),
        ])
        self.assertIn("건수 흔들림", out)
        self.assertNotIn("나타났다 사라짐", out)


class LabelCountsTest(unittest.TestCase):
    def test_counts_come_from_findings_not_summary(self):
        class _F:
            def __init__(self, label):
                self.label = label

        class _R:
            findings = [_F("a"), _F("a"), _F("b")]

        self.assertEqual(_label_counts(_R()), {"a": 2, "b": 1})

    def test_missing_findings_is_empty_not_error(self):
        class _R:
            findings = None

        self.assertEqual(_label_counts(_R()), {})


if __name__ == "__main__":
    unittest.main()
