"""
tests/test_answer_language.py
답변 언어 지시(RAG_ANSWER_LANGUAGE)와 심판 응답의 코드펜스 처리 검증.

둘 다 **영어 코퍼스(DragonBall) 실측에서 나온 버그**다.

  · 언어 — 프롬프트에 "한국어로" 가 박혀 있어 영어 질문에 한국어로 답했다. char-F1 은
    문자 겹침이라 언어가 다르면 구조적으로 0 이 되고, _is_success 의 lexical 축이 항상
    실패라 **검색이 완벽해도 실패로 집계된다**(실측: recall=1.00·gold 2/2 검색인데 f1=0.00).
  · 코드펜스 — 스키마 강제가 없는 심판 호출(aspect_critic)이 ```json 으로 감싸 응답해
    파싱이 실패하고 그 지표가 조용히 결측됐다(실측 10건 중 3건 전부).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.eval.llm_provider import _strip_code_fence
from agents.eval.metrics_basic import is_abstention
from agents.rag.generator import _PROMPT_EN, _PROMPT_KO, _build_prompt

_CONTEXT = ["Acme opened a factory in March 2020."]

_Q_EN = "When did Acme open the factory?"
_Q_KO = "Acme 는 공장을 언제 열었나?"


def _system(config=None, question=_Q_EN):
    system, _ = _build_prompt(question, _CONTEXT, max_context_chars=2000, config=config)
    return system


def _user(config=None, question=_Q_EN):
    _, user = _build_prompt(question, _CONTEXT, max_context_chars=2000, config=config)
    return user


class AnswerLanguageTest(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("RAG_ANSWER_LANGUAGE", None)

    def tearDown(self):
        os.environ.pop("RAG_ANSWER_LANGUAGE", None)
        if self._saved is not None:
            os.environ["RAG_ANSWER_LANGUAGE"] = self._saved

    def test_default_is_korean(self):
        """기본값이 바뀌면 기존 KorQuAD 실행의 답변 언어가 통째로 달라진다."""
        self.assertIn("한국어로", _system())

    def test_config_can_force_english(self):
        self.assertIn("Answer in English", _system({"answer_language": "en"}))

    def test_match_follows_the_question(self):
        """match 는 질문의 문자 구성으로 고른다.

        예전엔 "질문과 같은 언어로" 라는 **지시문 한 조각**만 넣고 모델에 맡겼는데,
        나머지 프롬프트가 전부 한국어라 모델이 프롬프트 언어를 따라갔다(실측: 영어
        질문 10건이 전부 한국어 답변). 코드가 정한다.
        """
        self.assertIn("Answer in English",
                      _system({"answer_language": "match"}, question=_Q_EN))
        self.assertIn("한국어로", _system({"answer_language": "match"}, question=_Q_KO))

    def test_env_is_used_when_config_is_silent(self):
        os.environ["RAG_ANSWER_LANGUAGE"] = "match"
        self.assertIn("Answer in English", _system(question=_Q_EN))

    def test_config_wins_over_env(self):
        os.environ["RAG_ANSWER_LANGUAGE"] = "en"
        self.assertIn("한국어로", _system({"answer_language": "ko"}))

    def test_unknown_value_falls_back_to_korean(self):
        """오타가 프롬프트를 깨뜨리지 않게 — 모르는 값이면 기존 동작."""
        self.assertIn("한국어로", _system({"answer_language": "zzz"}))

    def test_whole_prompt_switches_not_just_one_clause(self):
        """조각만 바꾸면 모델이 프롬프트 언어를 따라간다 — 실측으로 확인된 실패 모드다.

        지시문에 한국어가 **한 글자도** 남으면 안 된다. 모든 플래그를 켜서 조립되는
        조각을 전부 태운다(하나만 번역이 빠져도 여기서 걸린다).
        """
        config = {
            "answer_language": "en", "grounding_strict": True,
            "abstention_strict": True, "completeness_mode": True,
            "restate_question": True, "require_citation": True,
        }
        for text in (_system(config), _user(config)):
            self.assertFalse(_has_hangul(text), text)

    def test_relaxed_abstention_branch_is_also_translated(self):
        """abstention 은 배타 분기라 한쪽만 번역하면 조용히 한국어가 새어 나온다."""
        relaxed = _system({"answer_language": "en", "abstention_relaxed": True})
        self.assertFalse(_has_hangul(relaxed), relaxed)

    def test_user_message_headers_switch_too(self):
        """지시문만 영어로 두고 [컨텍스트]/[질문] 머리말이 한국어면 다시 무너진다."""
        self.assertIn("[Question]", _user({"answer_language": "en"}))
        self.assertIn("[질문]", _user({"answer_language": "ko"}))

    def test_language_applies_to_both_grounding_modes(self):
        """grounding_strict 분기가 둘이라 한쪽만 고치면 조용히 어긋난다."""
        for strict in (True, False):
            system = _system({"answer_language": "en", "grounding_strict": strict})
            self.assertIn("in English", system)
            self.assertFalse(_has_hangul(system), system)

    def test_english_abstention_phrase_is_a_registered_marker(self):
        """지시한 기권 문구가 판별기에 없으면 **기권이 기권으로 안 읽힌다**.

        그러면 모델이 정직하게 기권했는데 파이프라인은 '답했다' 로 보고
        generation_abstention_failure / wrongful_abstention 계열이 통째로 어긋난다.
        한국어 문구도 같은 계약이라 함께 고정한다.
        """
        for prompts in (_PROMPT_KO, _PROMPT_EN):
            phrase = _extract_quoted(prompts["grounded"])
            self.assertTrue(is_abstention(phrase), phrase)


def _has_hangul(text: str) -> bool:
    return any("가" <= c <= "힣" or "ᄀ" <= c <= "ᇿ" or "㄰" <= c <= "㆏" for c in text)


def _extract_quoted(clause: str) -> str:
    """지시문에서 작은따옴표로 감싼 기권 예시 문구를 꺼낸다."""
    start = clause.index("'") + 1
    return clause[start:clause.index("'", start)]


class CodeFenceTest(unittest.TestCase):
    def test_real_failure_shape_is_parsed(self):
        """실측 로그에 남은 형태 그대로."""
        raw = ('```json\n{\n  "reason": "The response declines to answer.",\n'
               '  "verdict": 1\n}\n```')
        self.assertEqual(_strip_code_fence(raw).strip(),
                         '{\n  "reason": "The response declines to answer.",\n  "verdict": 1\n}')

    def test_fence_without_language_tag(self):
        self.assertEqual(_strip_code_fence('```\n{"verdict": 0}\n```'), '{"verdict": 0}')

    def test_plain_json_is_untouched(self):
        """펜스가 없으면 원문 그대로 — 정상 경로를 건드리면 안 된다."""
        self.assertEqual(_strip_code_fence('{"verdict": 1}'), '{"verdict": 1}')

    def test_truncated_response_still_yields_body(self):
        """상한 절단으로 닫는 펜스가 없어도 본문은 살린다."""
        self.assertEqual(_strip_code_fence('```json\n{"verdict": 1}'), '{"verdict": 1}')

    def test_empty_input_is_safe(self):
        self.assertEqual(_strip_code_fence(""), "")
        self.assertEqual(_strip_code_fence(None), "")


class CodeFenceIsWiredIntoChatJsonTest(unittest.TestCase):
    """`chat_json` 이 실제로 펜스를 벗기는지.

    헬퍼만 테스트하면 **배선이 빠져도 통과한다** — 실제로 뮤테이션(`json.loads(raw)` 로
    되돌리기)이 헬퍼 테스트를 그대로 통과했다. 심판 응답이 들어오는 경로로 검증한다.
    """

    def _chat_json(self, raw):
        from unittest.mock import patch

        from agents.eval import llm_provider
        with patch.object(llm_provider, "_run_with_retry", return_value=raw):
            return llm_provider.chat_json(object(), "prompt", label="t")

    def test_fenced_response_is_parsed(self):
        self.assertEqual(self._chat_json('```json\n{"verdict": 1}\n```'), {"verdict": 1})

    def test_plain_response_still_works(self):
        self.assertEqual(self._chat_json('{"verdict": 0}'), {"verdict": 0})

    def test_broken_json_still_falls_back_to_empty(self):
        """펜스를 벗겨도 JSON 이 아니면 기존대로 {} — 폴백 계약은 그대로다."""
        self.assertEqual(self._chat_json('```json\nnot json at all\n```'), {})


if __name__ == "__main__":
    unittest.main()
