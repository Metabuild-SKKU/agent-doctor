"""
tests/test_answer_match.py
정답 매칭 지표(metrics_basic.answer_match)의 길이 경로 고정.

배경: KorQuAD char-F1 은 추출형 짧은 정답용 지표라, 긴 서술형 gold 를 상대로 근거·소제목·
부연을 갖춘 '맞은' 답변이 precision 감점만으로 0.3~0.4 로 깎였다. 그 결과 f1 < 0.5 게이트에서
실패로 잡히고 C그룹(context_noise_interference 등)으로 오진돼 optimize 가 엉뚱한 처방을 받았다.
여기서는 창(window) 경로가 그 저평가를 되살리면서도, 길게 쓴 무관한 답변은 통과시키지 않는
분리(separation)를 못 박는다.
"""
import itertools
import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import Probe
from agents.eval import diagnose, metrics_common
from agents.eval.metrics_basic import answer_match, best_window_char_f1, char_f1
from agents.eval.scoring import reliability_score
from agents.eval.types import (
    ANSWER_CORRECTNESS_MIN, ANSWER_PASS_THRESHOLD, ANSWER_SEMANTIC_WEIGHT,
    EvalRecord, F1_EXACT_MATCH, F1_PASS_THRESHOLD, Mode, blend_answer_score,
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


class TestGateMonotonicity(unittest.TestCase):
    """새 게이트의 통과집합이 옛 게이트(lexical 단독 + ac 강등)의 상위집합인지 전수 확인.

    왜 고정하나: `_oracle_ok` 실패를 전제로 하는 라벨(bad_gold_answer, B그룹 생성 실패)의
    발동률이 '줄기만 하고 늘지 않는다'가 이 성질에서 바로 나온다. 리뷰에서 나온 '발동 이동'
    질문을 실행 없이 정적으로 답할 수 있는 근거이자, 앞으로 문턱을 만질 때 그 보장이 조용히
    깨지지 않게 하는 잠금장치다. (degrade 값이 승격에서 빠지면서 강등 권한까지 잃어
    실제로 이 성질이 한 번 깨졌었다 — _degraded_near_miss 로 복구.)
    """

    def _old_pass(self, lexical, ac):
        """이 PR 이전 게이트: lexical 문턱 + answer_correctness 강등."""
        if lexical < F1_PASS_THRESHOLD:
            return False
        return True if ac is None else ac >= ANSWER_CORRECTNESS_MIN

    def _record(self, lexical, ragas):
        probe = Probe(probe_id="p1", question="질문", source="taxonomy",
                      gold_chunk_ids=["g_a"], ground_truth="정답")
        rec = EvalRecord(probe=probe)
        rec.recall_at_k, rec.f1_score = 1.0, lexical
        rec.ragas, rec.ragas_done = dict(ragas), True
        return rec

    def test_new_gate_never_stricter_than_old(self):
        metrics_common.set_mode(Mode.DEEP)
        try:
            grid = [i / 10 for i in range(11)]
            counts_cases = [None, (0, 0, 4), (2, 0, 2), (4, 0, 0), (5, 6, 0)]
            checked = 0
            for lexical, ac, faith in itertools.product(grid, grid + [None], grid + [None]):
                for counts in counts_cases:
                    for degraded in (False, True):
                        if degraded and counts is not None:
                            continue            # degrade = factual 미측정 = 카운트 없음
                        ragas = {}
                        if ac is not None:
                            ragas["answer_correctness"] = ac
                            if degraded:
                                ragas["answer_correctness_degraded"] = True
                        if faith is not None:
                            ragas["faithfulness"] = faith
                        if counts is not None:
                            tp, fp, fn = counts
                            ragas.update({"answer_correctness_tp": tp,
                                          "answer_correctness_fp": fp,
                                          "answer_correctness_fn": fn})
                        checked += 1
                        if self._old_pass(lexical, ac):
                            self.assertTrue(
                                diagnose._f1_ok(self._record(lexical, ragas)),
                                f"옛 게이트는 통과인데 새 게이트가 실패: "
                                f"lexical={lexical}, ragas={ragas}",
                            )
            self.assertGreater(checked, 1000)   # 격자가 비어 공허하게 통과하지 않도록
        finally:
            metrics_common.set_mode(Mode.FAST)

    def test_degraded_correctness_still_demotes(self):
        """degrade 값은 승격엔 못 쓰지만 강등엔 쓴다 — 판정기가 죽어도 게이트가 안 헐거워진다."""
        metrics_common.set_mode(Mode.DEEP)
        try:
            rec = self._record(0.6, {"answer_correctness": 0.1,
                                     "answer_correctness_degraded": True})
            self.assertFalse(diagnose._f1_ok(rec))
        finally:
            metrics_common.set_mode(Mode.FAST)


class TestDegradeDoesNotFlipExactMatch(unittest.TestCase):
    """버그1(probe_qa_26360): degrade 된 심판이 어휘 완전일치(정답)를 오답으로 뒤집으면
    recall=1·oracle 통과와 겹쳐 context_noise_interference 로 오진되고, degrade 비결정성
    때문에 같은 답이 반복마다 통과/실패를 오간다. 완전일치는 강등 면제, 애매한 근접 오답은 강등 유지."""

    def _record(self, *, f1, ragas, recall=1.0, oracle_f1=1.0):
        probe = Probe(probe_id="p1", question="질문", source="taxonomy",
                      gold_chunk_ids=["g_a"], ground_truth="세종대왕")
        rec = EvalRecord(probe=probe, generated_answer="세종대왕", oracle_answer="세종대왕")
        rec.recall_at_k, rec.f1_score, rec.oracle_f1 = recall, f1, oracle_f1
        rec.ragas = dict(ragas)
        rec.ragas_done = True
        rec.oracle_ragas = {}
        rec.oracle_ragas_done = True
        return rec

    def test_exact_match_exempt_from_degrade_demotion(self):
        """어휘 완전일치(f1=1.0) + degrade 낮은 ac → 강등 면제 → 정답 판정 유지."""
        metrics_common.set_mode(Mode.DEEP)
        try:
            rec = self._record(f1=F1_EXACT_MATCH, ragas={
                "faithfulness": 1.0, "answer_correctness": 0.1,
                "answer_correctness_degraded": True})
            self.assertFalse(diagnose._degraded_near_miss(rec, oracle=False))
            self.assertTrue(diagnose._f1_ok(rec))
        finally:
            metrics_common.set_mode(Mode.FAST)

    def test_near_miss_below_exact_still_demotes(self):
        """완전일치 미만(f1=0.6)은 degrade 시 여전히 강등 — 안전망 유지(경계 케이스)."""
        metrics_common.set_mode(Mode.DEEP)
        try:
            rec = self._record(f1=0.6, ragas={
                "faithfulness": 1.0, "answer_correctness": 0.1,
                "answer_correctness_degraded": True})
            self.assertTrue(diagnose._degraded_near_miss(rec, oracle=False))
            self.assertFalse(diagnose._f1_ok(rec))
        finally:
            metrics_common.set_mode(Mode.FAST)

    def test_exact_match_ungrounded_still_demotes(self):
        """부정문 오답('X 가 아니다'): 완전일치라도 문맥과 충돌해 근거성이 낮으면 강등 유지 —
        면제 조건에 faithfulness 를 걸어 degrade 실행의 면제 구멍을 좁힌다."""
        metrics_common.set_mode(Mode.DEEP)
        try:
            rec = self._record(f1=F1_EXACT_MATCH, ragas={
                "faithfulness": 0.1, "answer_correctness": 0.1,
                "answer_correctness_degraded": True})
            self.assertTrue(diagnose._degraded_near_miss(rec, oracle=False))
            self.assertFalse(diagnose._f1_ok(rec))
        finally:
            metrics_common.set_mode(Mode.FAST)

    def test_exact_match_faith_unmeasured_still_demotes(self):
        """근거성 미측정(faithfulness None)이면 면제하지 않고 기존 강등으로 흐른다(보수적)."""
        metrics_common.set_mode(Mode.DEEP)
        try:
            rec = self._record(f1=F1_EXACT_MATCH, ragas={
                "answer_correctness": 0.1, "answer_correctness_degraded": True})
            self.assertIsNone(diagnose._faith(rec))
            self.assertTrue(diagnose._degraded_near_miss(rec, oracle=False))
            self.assertFalse(diagnose._f1_ok(rec))
        finally:
            metrics_common.set_mode(Mode.FAST)

    def test_no_false_context_noise_on_exact_match(self):
        """probe 전체 흐름: 완전일치 + degrade 여도 성공 처리되어 context_noise_interference
        (유령 실패)가 붙지 않는다 — diagnose 가 곧장 [] 로 종료."""
        metrics_common.set_mode(Mode.DEEP)
        try:
            rec = self._record(f1=F1_EXACT_MATCH, ragas={
                "faithfulness": 1.0, "answer_correctness": 0.1,
                "answer_correctness_degraded": True})
            with unittest.mock.patch.object(diagnose, "_compute_metrics"), \
                 unittest.mock.patch.object(diagnose, "_compute_ragas_real"), \
                 unittest.mock.patch.object(diagnose, "_compute_ragas_oracle"):
                findings = diagnose.diagnose(rec, mode=int(Mode.DEEP))
            self.assertEqual(findings, [])
        finally:
            metrics_common.set_mode(Mode.FAST)


class TestAnswerReasonObservability(unittest.TestCase):
    """관측성: reason 문자열이 판정을 뒤집은 실제 근거를 드러낸다 — degrade 로 의미축이
    빠져 오답 처리됐는데 로그엔 'f1=1.00'만 찍혀 'f1 완벽인데 실패'가 설명 안 되던 문제."""

    def _record(self, *, f1, ragas):
        probe = Probe(probe_id="p1", question="q", source="taxonomy",
                      gold_chunk_ids=["g_a"], ground_truth="정답")
        rec = EvalRecord(probe=probe)
        rec.recall_at_k, rec.f1_score = 1.0, f1
        rec.ragas, rec.ragas_done = dict(ragas), True
        return rec

    def test_degrade_surfaces_ac_value(self):
        # degrade 로 의미축 빠짐 → f1 뿐 아니라 판정을 뒤집은 ac_degraded 를 드러낸다.
        rec = self._record(f1=1.0, ragas={
            "faithfulness": 1.0, "answer_correctness": 0.1,
            "answer_correctness_degraded": True})
        reason = diagnose._answer_reason(rec)
        self.assertIn("의미측정실패", reason)
        self.assertIn("ac_degraded=0.100", reason)

    def test_low_mode_unmeasured_stays_f1_only(self):
        # degrade 가 아닌 단순 미측정(저모드)은 기존대로 f1 만 — 숨은 신호가 없다.
        rec = self._record(f1=0.42, ragas={})
        self.assertEqual(diagnose._answer_reason(rec), "f1=0.420")

    def test_measured_semantic_shows_blend(self):
        # 의미축이 측정된 정상 경로는 혼합 점수와 두 축을 모두 보인다(기존 형식).
        rec = self._record(f1=0.49, ragas={
            "faithfulness": 0.9, "answer_correctness": 0.73})
        reason = diagnose._answer_reason(rec)
        self.assertIn("answer=", reason)
        self.assertIn("f1 0.490", reason)
        self.assertIn("의미 0.730", reason)


if __name__ == "__main__":
    unittest.main()
