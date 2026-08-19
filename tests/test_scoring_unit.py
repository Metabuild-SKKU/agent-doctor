"""
tests/test_scoring_unit.py
채점 단위 선택(metrics_basic.scoring_unit)과 영어 단어 단위 F1 검증.

char-F1 은 **한국어 전용 보정**이다. DragonBall(영어) 실측에서 이걸 그대로 쓰다가 문제가
드러났다 — 알파벳 26자를 단어들이 공유해서, 한 글자 다른 고유명사 오답이 0.95 를 받는다.
통과선이 0.5 라 **틀린 답을 거를 힘이 사실상 없고**, 그만큼 generation_* 진단이 사라진다.

여기서 고정하는 건 두 가지다.
  · 어느 정답이 어느 단위로 채점되나 — 잘못 넘어가면 **한국어 점수가 통째로 바뀐다**
  · 단위를 바꿔서 실제로 오답이 걸러지나 — 안 걸러지면 바꾼 의미가 없다
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.eval.metrics_basic import (
    answer_f1, answer_match, best_window_char_f1, char_f1, char_recall,
    exact_match, scoring_unit, token_f1, word_f1,
)
from agents.eval.metrics_basic import (
    _best_window_f1, _chars, _recall_from_units, _words,
)
from agents.eval.types import F1_PASS_THRESHOLD


class ScoringUnitSelectionTest(unittest.TestCase):
    """단위는 **정답(reference)** 만 보고 정한다."""

    def test_korean_reference_uses_characters(self):
        self.assertEqual(scoring_unit("재택근무는 주 2일 가능"), "char")

    def test_english_reference_uses_words(self):
        self.assertEqual(scoring_unit("Acme Logistics, down 12 percent"), "word")

    def test_numeric_reference_stays_on_characters(self):
        """숫자형 정답을 단어로 넘기면 조사가 붙은 어절과 교집합이 0 이 된다.

        '332cm' 에는 라틴 문자(cm)가 있지만 절반이 안 된다. 단어로 넘어가면
        answer_match('높이는 332cm입니다', '332cm') 이 1.0 → 0.0 으로 무너진다.
        """
        for reference in ("332cm", "1450", "14:33", "1,450원"):
            self.assertEqual(scoring_unit(reference), "char", reference)

    def test_mixed_reference_follows_hangul(self):
        """한글이 하나라도 있으면 문자 단위다 — 영어 용어가 섞인 한국어 정답이 흔하다.

        라틴 문자가 **과반이어도** 그렇다. 여기 골라둔 두 정답은 라틴 비율이 0.76·0.89 라,
        한글 검사가 빠지면 단어 단위로 넘어가 조사 붙은 어절과 교집합이 무너진다
        (비율만 낮은 예 'BGE-M3 임베딩 모델'(0.40)로는 이 결함이 안 잡힌다 — 실제로
        뮤테이션이 그 케이스를 그냥 통과했다).
        """
        for reference in ("Windows Server 라이선스", "Acme 사의 CEO John Carter"):
            self.assertEqual(scoring_unit(reference), "char", reference)

    def test_hangul_reference_scores_the_same_as_before(self):
        """단위가 넘어갔는지를 점수로도 못 박는다 — 한글 정답은 조사가 붙어도 통과해야 한다."""
        self.assertGreaterEqual(
            answer_match("정답은 Windows Server 라이선스입니다", "Windows Server 라이선스"),
            F1_PASS_THRESHOLD,
        )

    def test_empty_reference_is_safe(self):
        self.assertEqual(scoring_unit(""), "char")
        self.assertEqual(scoring_unit(None), "char")

    def test_answer_is_never_consulted(self):
        """답변으로 단위를 정하면 모델이 언어를 바꾸는 것만으로 채점 단위가 흔들린다.

        같은 정답에 대해 답변 언어가 달라도 단위는 그대로여야 한다.
        """
        gold = "Guadeloupe"
        self.assertEqual(answer_f1("과들루프입니다", gold),
                         word_f1("과들루프입니다", gold))


class KoreanScoringIsUnchangedTest(unittest.TestCase):
    """기존 KorQuAD 실행의 점수가 바뀌면 안 된다 — 회귀 가드."""

    CASES = [
        ("재택근무는 주 2일 가능", "재택근무는 주 2일 가능"),
        ("높이는 332cm입니다", "332cm"),
        ("사망하지 않았다", "사망"),
        ("150명입니다", "145"),
        ("전혀 다른 문장", "재택근무 규정"),
    ]

    def test_answer_match_matches_char_path(self):
        for prediction, reference in self.CASES:
            with self.subTest(reference=reference):
                self.assertEqual(scoring_unit(reference), "char")
                self.assertEqual(answer_f1(prediction, reference),
                                 char_f1(prediction, reference))

    def test_known_fixed_points_hold(self):
        self.assertEqual(answer_match("높이는 332cm입니다", "332cm"), 1.0)
        self.assertEqual(answer_match("사망하지 않았다", "사망"), 1.0)
        self.assertLess(answer_match("150명입니다", "145"), F1_PASS_THRESHOLD)


class EnglishNearMissIsRejectedTest(unittest.TestCase):
    """이 클래스가 바꾼 이유 자체다 — 영어 오답이 char 로는 통과한다."""

    # (정답, 명백한 오답, char-F1 실측)
    NEAR_MISSES = [
        ("the Republic of Ireland", "the Republic of Iceland", 0.950),
        ("Sony Corporation", "Sanyo Corporation", 0.968),
        ("located in Barcelona", "located in Bangalore", 0.944),
        ("Dr. Alan Carter", "Dr. Clara Newton", 0.800),
    ]

    def test_char_f1_scores_wrong_answers_near_one(self):
        """문자 단위가 영어에서 왜 못 쓰는 지표인지 수치로 남긴다."""
        for gold, wrong, expected in self.NEAR_MISSES:
            with self.subTest(gold=gold):
                self.assertAlmostEqual(char_f1(wrong, gold), expected, places=2)
                self.assertGreater(char_f1(wrong, gold), F1_PASS_THRESHOLD)

    def test_word_f1_scores_them_lower(self):
        for gold, wrong, _ in self.NEAR_MISSES:
            with self.subTest(gold=gold):
                self.assertLess(word_f1(wrong, gold), char_f1(wrong, gold))

    def test_completely_wrong_short_answer_scores_zero(self):
        """짧은 정답에서 차이가 가장 크다 — char 는 0.44, word 는 0."""
        self.assertEqual(word_f1("April 2019", "March 2020"), 0.0)
        self.assertGreater(char_f1("April 2019", "March 2020"), 0.4)
        self.assertEqual(answer_match("April 2019", "March 2020"), 0.0)


class EnglishCorrectAnswerIsAcceptedTest(unittest.TestCase):
    """단위를 바꾼 대가로 맞은 답이 깎이면 안 된다(과소 쪽 실패)."""

    def test_short_gold_wrapped_in_a_sentence_passes(self):
        """정답을 통째로 담은 완결 문장 — char 로는 프레이밍 문자에 precision 이 깎인다."""
        score = answer_match("The factory is located in Guadeloupe.", "Guadeloupe")
        self.assertEqual(score, 1.0)
        self.assertLess(char_f1("The factory is located in Guadeloupe.", "Guadeloupe"),
                        score)

    def test_two_word_gold_still_takes_the_containment_path(self):
        self.assertEqual(answer_match("It was Acme Logistics.", "Acme Logistics"), 1.0)

    def test_articles_do_not_change_the_score(self):
        """SQuAD 공식 normalize_answer 의 remove_articles."""
        self.assertEqual(word_f1("Eiffel Tower", "the Eiffel Tower"), 1.0)
        self.assertTrue(exact_match("Eiffel Tower", "the Eiffel Tower"))

    def test_verbose_answer_to_a_long_gold_takes_the_window_path(self):
        """긴 서술형 정답에 근거를 덧붙인 답변 — 창 경로가 단어 단위에서도 살아야 한다.

        문턱이 문자 수(30)로만 있으면 영어 서술형 정답이 전부 창 경로를 못 타고,
        precision 분모가 답변 전체가 되어 맞은 답이 실패로 떨어진다.
        """
        gold = ("The board approved the merger in June after the regulator "
                "cleared the transaction without conditions")
        verbose = (
            "According to the filing, the board approved the merger in June after "
            "the regulator cleared the transaction without conditions. The company "
            "said integration would begin in the following quarter and that no "
            "further approvals were required from any other authority."
        )
        self.assertGreaterEqual(answer_match(verbose, gold), F1_PASS_THRESHOLD)
        self.assertLess(word_f1(verbose, gold), answer_match(verbose, gold))


class DispatchWiringTest(unittest.TestCase):
    """헬퍼만 맞아도 배선이 빠지면 아무 효과가 없다 — 진입점이 실제로 갈리는지 본다."""

    def test_answer_f1_dispatches_on_the_reference(self):
        self.assertEqual(answer_f1("April 2019", "March 2020"),
                         word_f1("April 2019", "March 2020"))
        self.assertEqual(answer_f1("150명", "145"), char_f1("150명", "145"))

    def test_token_f1_alias_follows_the_dispatcher(self):
        """구 호출부(tests/check_eval.py)가 쓰는 이름이 문자 단위에 묶여 있으면 안 된다."""
        self.assertIs(token_f1, answer_f1)

    def test_char_recall_follows_the_unit(self):
        """짧은 정답 containment 판정도 단위를 따라가야 한다.

        'Guadeloupe 를 담은 문장' 처럼 둘 다 1.0 이 나오는 예로는 이 배선이 안 잡힌다
        (문자로 고정해도 통과했다). 두 단위의 값이 **갈리는** 쌍으로 못 박는다.
        """
        # gold 'March 2020' 의 문자는 절반 가까이 우연히 겹치지만 단어는 하나도 안 겹친다.
        self.assertEqual(char_recall("April 2019", "March 2020"), 0.0)
        self.assertGreater(_recall_from_units(_chars("April 2019"), _chars("March 2020")), 0.4)
        # 한국어는 문자 단위 그대로.
        self.assertEqual(char_recall("높이는 332cm입니다", "332cm"), 1.0)

    def test_best_window_follows_the_unit(self):
        """창 경로의 미끄러뜨리는 단위도 정답 언어를 따라야 한다.

        answer_match 는 내부 헬퍼를 직접 부르므로, 공개 함수만 문자로 되돌려도
        answer_match 테스트는 전부 통과한다 — 여기서 따로 잡는다.
        """
        gold = ("The board approved the merger in June after the regulator "
                "cleared the transaction")
        # 두 사실이 틀린 답변(June→July, board→panel). 문자 창은 0.826 을 주고 단어 창은
        # 0.444 를 준다 — 값이 갈리는 쌍이어야 배선을 잡는다(정답을 그대로 담은 답변은
        # 두 단위가 다 1.0 근처라 문자로 고정해도 이 테스트가 통과한다).
        wrong = ("The regulator cleared the transaction in July, the filing said, "
                 "and the panel then approved the merger without further review.")
        self.assertAlmostEqual(
            best_window_char_f1(wrong, gold),
            _best_window_f1(_words(wrong), _words(gold)),
        )
        self.assertGreater(
            _best_window_f1(_chars(wrong), _chars(gold)),
            best_window_char_f1(wrong, gold) + 0.3,
        )

    def test_exact_match_uses_word_comparison_for_english(self):
        self.assertTrue(exact_match("Acme Logistics", "acme logistics"))
        self.assertFalse(exact_match("Acme Logistics", "Acme Holdings"))


if __name__ == "__main__":
    unittest.main()
