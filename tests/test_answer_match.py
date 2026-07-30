"""
tests/test_answer_match.py
정답 매칭 지표(metrics_basic.answer_match)의 길이 경로 고정.

배경: KorQuAD char-F1 은 추출형 짧은 정답용 지표라, 긴 서술형 gold 를 상대로 근거·소제목·
부연을 갖춘 '맞은' 답변이 precision 감점만으로 0.3~0.4 로 깎였다. 그 결과 f1 < 0.5 게이트에서
실패로 잡히고 C그룹(context_noise_interference 등)으로 오진돼 optimize 가 엉뚱한 처방을 받았다.
여기서는 창(window) 경로가 그 저평가를 되살리면서도, 길게 쓴 무관한 답변은 통과시키지 않는
분리(separation)를 못 박는다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import Probe
from agents.eval.metrics_basic import answer_match, best_window_char_f1, char_f1
from agents.eval.scoring import reliability_score
from agents.eval.types import (
    ANSWER_PASS_THRESHOLD, ANSWER_SEMANTIC_WEIGHT, EvalRecord, F1_PASS_THRESHOLD,
    blend_answer_score,
)


# 실측 실패 사례(probe_multi_abstract_021)에서 가져온 gold·답변 쌍.
GOLD_LONG = (
    "영화는 서구 제국주의 시대의 산물로서 서구가 세계의 중심이라는 서구 중심주의를 내포하고 "
    "있습니다. 이는 에드워드 사이드가 정의한 오리엔탈리즘과 연결되는데, 서양인이 동양을 "
    "자신들과는 다른 '타자'로 규정하고 정형화된 이미지를 덧씌워 동양에 대한 권위적인 "
    "사고방식을 구축하는 인식론적 구별의 과정으로 나타납니다."
)
ANSWER_VERBOSE_CORRECT = (
    "서구 중심주의는 서구가 세계의 중심이며 그 외의 지역은 서구의 그림자에 불과하다는 관념으로, "
    "오리엔탈리즘과 밀접한 관련이 있습니다 [1]. 에드워드 사이드는 오리엔탈리즘을 서양인이 동양을 "
    "지배하고, 재구성하며, 억압하기 위한 서양의 방식이자 사고방식으로 정의합니다 [1, 2]. 이러한 "
    "관점에서 영화가 동양을 재현하는 방식은 다음과 같은 맥락적 연관성을 갖습니다. 첫째, 타자화와 "
    "이분법적 구별입니다. 서구는 동양과의 대조를 통해 스스로를 합리적이고 문명화된 존재로 "
    "규정하며, 동양을 신비롭고 영적이지만 미개하고 야만적인 '타자'로 재구성합니다 [2]. 영화는 "
    "이러한 존재론적·인식론적 구별을 바탕으로 동양을 정형화된 이미지로 묘사합니다 [1, 3]. 둘째, "
    "권위 부여와 재구성입니다. 오리엔탈리즘은 동양에 관한 견해에 권위를 부여하고 동양을 서술하는 "
    "종합적인 제도입니다 [2]. 결국 영화가 동양을 정형화하는 것은 동양을 서구의 지배 아래 두기 "
    "위한 오리엔탈리즘적 사고방식이 투영된 결과라고 할 수 있습니다 [2, 5]."
)
ANSWER_LONG_OFF_TOPIC = (
    "우리 회사의 재택근무 제도는 주 2일까지 신청할 수 있으며, 신청은 인사팀 포털의 근무형태 변경 "
    "메뉴에서 진행합니다. 승인권자는 소속 팀장이고, 승인 후에는 근태 시스템에 자동 반영됩니다. "
    "관련 규정은 취업규칙 제12조와 원격근무 운영지침에 정리되어 있습니다. 장비 지원은 노트북과 "
    "모니터가 기본이며, 통신비는 월 2만원 한도로 실비 정산됩니다. 보안 교육을 이수하지 않은 "
    "직원은 신청이 제한될 수 있습니다. 자세한 문의는 인사팀으로 연락하시면 안내받을 수 있습니다."
) * 2


class TestLongGoldWindowPath(unittest.TestCase):
    def test_verbose_correct_answer_passes(self):
        """정답을 담은 서술형 답변: char-F1 은 문턱 미달인데 창 경로로 통과한다."""
        self.assertLess(char_f1(ANSWER_VERBOSE_CORRECT, GOLD_LONG), F1_PASS_THRESHOLD)
        self.assertGreaterEqual(answer_match(ANSWER_VERBOSE_CORRECT, GOLD_LONG), F1_PASS_THRESHOLD)

    def test_long_off_topic_answer_still_fails(self):
        """길게 쓴 무관한 답변은 창 경로로도 문턱을 넘지 못한다(길이로 점수를 벌 수 없음)."""
        self.assertLess(answer_match(ANSWER_LONG_OFF_TOPIC, GOLD_LONG), F1_PASS_THRESHOLD)

    def test_window_is_at_least_char_f1_for_verbose(self):
        """창 경로는 답변이 정답보다 길 때 char-F1 을 밑돌지 않는다(구제만, 감점 없음)."""
        self.assertGreaterEqual(
            best_window_char_f1(ANSWER_VERBOSE_CORRECT, GOLD_LONG),
            char_f1(ANSWER_VERBOSE_CORRECT, GOLD_LONG),
        )

    def test_window_off_when_answer_not_verbose(self):
        """답변이 정답보다 길지 않으면(1.3배 미만) 기존 char-F1 그대로 — 경로가 안 켜진다."""
        answer = GOLD_LONG[:200]
        self.assertEqual(answer_match(answer, GOLD_LONG), char_f1(answer, GOLD_LONG))

    def test_short_answer_shorter_than_gold_falls_back(self):
        """창을 잡을 수 없는(답변 < 정답) 경우 best_window_char_f1 은 char_f1 로 폴백한다."""
        answer = "서구 중심주의와 오리엔탈리즘"
        self.assertEqual(
            best_window_char_f1(answer, GOLD_LONG), char_f1(answer, GOLD_LONG)
        )


class TestShortGoldPathUnchanged(unittest.TestCase):
    """짧은 정답(KorQuAD 추출형) 경로는 창 경로 도입 전과 동일해야 한다."""

    def test_containment_still_rescues_short_gold(self):
        self.assertEqual(answer_match("높이는 332cm입니다", "332cm"), 1.0)

    def test_near_miss_short_gold_still_fails(self):
        self.assertLess(answer_match("150명입니다", "145"), F1_PASS_THRESHOLD)

    def test_char_f1_untouched(self):
        """char_f1 자체는 KorQuAD 공식 계산 그대로(창 보정이 섞이지 않는다)."""
        self.assertEqual(char_f1("재택근무는 주 2일 가능", "재택근무는 주 2일 가능"), 1.0)
        self.assertEqual(char_f1("전혀 다른 문장", "재택근무 규정"), 0.0)


class TestBlendedAnswerScore(unittest.TestCase):
    """혼합 점수(types.blend_answer_score)와 그 소비처(신뢰도 축)의 계약."""

    def _record(self, *, f1, ragas=None, recall=1.0):
        probe = Probe(probe_id="p1", question="질문", source="taxonomy",
                      gold_chunk_ids=["g_a"], ground_truth="정답")
        rec = EvalRecord(probe=probe)
        rec.recall_at_k = recall
        rec.f1_score = f1
        rec.ragas = dict(ragas or {})
        rec.ragas_done = True
        return rec

    def test_lexical_only_when_semantic_unmeasured(self):
        self.assertEqual(blend_answer_score(0.42, None), 0.42)

    def test_weighted_mix(self):
        expected = (1 - ANSWER_SEMANTIC_WEIGHT) * 0.49 + ANSWER_SEMANTIC_WEIGHT * 0.73
        self.assertAlmostEqual(blend_answer_score(0.49, 0.73), expected)
        self.assertGreaterEqual(blend_answer_score(0.49, 0.73), ANSWER_PASS_THRESHOLD)

    def test_semantic_axis_takes_coverage_only_when_grounded(self):
        """의미축 = max(answer_correctness, 커버리지) — 커버리지는 근거 있을 때만."""
        grounded = self._record(f1=0.3, ragas={
            "faithfulness": 0.9, "answer_correctness": 0.45,
            "answer_correctness_tp": 5, "answer_correctness_fp": 6, "answer_correctness_fn": 0,
        })
        self.assertEqual(grounded.answer_semantic, 1.0)
        ungrounded = self._record(f1=0.3, ragas={
            "faithfulness": 0.3, "answer_correctness": 0.45,
            "answer_correctness_tp": 5, "answer_correctness_fp": 6, "answer_correctness_fn": 0,
        })
        self.assertEqual(ungrounded.answer_semantic, 0.45)

    def test_partial_coverage_is_not_partial_credit(self):
        """gold 절반만 담은 답변의 커버리지(0.5)는 의미축으로 인정하지 않는다.

        인정하면 부분 답변이 커버리지를 들고 통과해 generation_partial_answer 가 사라진다.
        누락 감점은 FN 을 분모에 넣는 answer_correctness 몫으로 남긴다."""
        partial = self._record(f1=0.3, ragas={
            "faithfulness": 0.9, "answer_correctness": 0.4,
            "answer_correctness_tp": 2, "answer_correctness_fp": 0, "answer_correctness_fn": 2,
        })
        self.assertEqual(partial.answer_semantic, 0.4)   # 커버리지 0.5 가 아니라 ac

    def test_degraded_correctness_is_not_promoted(self):
        """factual 분류 실패로 유사도 단독이 된 answer_correctness 는 의미축에서 뺀다.

        한국어는 같은 주제면 사실이 틀려도 코사인이 높게 나와, 승격에 쓰면 오답이 통과한다.
        (강등 전용이던 시절엔 안전했던 폴백 — 게이트가 승격도 하게 되며 위험해졌다.)"""
        rec = self._record(f1=0.3, ragas={
            "faithfulness": 0.9, "answer_correctness": 0.85,
            "answer_correctness_degraded": True,
        })
        self.assertIsNone(rec.answer_semantic)           # → lexical 단독 판정
        self.assertEqual(rec.answer_score, 0.3)

    def test_reliability_axis_matches_gate_score(self):
        """신뢰도 축이 게이트와 같은 값을 본다 — 통과한 probe 가 낮은 신뢰도로 남지 않는다."""
        rec = self._record(f1=0.49, recall=1.0, ragas={
            "faithfulness": 0.9, "answer_correctness": 0.73,
        })
        self.assertAlmostEqual(reliability_score([rec]), rec.answer_score)
        self.assertGreaterEqual(rec.answer_score, ANSWER_PASS_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
