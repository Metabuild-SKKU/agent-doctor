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
from agents.rag.generator import _build_prompt

_CONTEXT = ["Acme opened a factory in March 2020."]


def _system(config=None):
    system, _ = _build_prompt("When?", _CONTEXT, max_context_chars=2000, config=config)
    return system


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

    def test_config_can_ask_to_match_the_question(self):
        self.assertIn("질문과 같은 언어로", _system({"answer_language": "match"}))

    def test_config_can_force_english(self):
        self.assertIn("in English", _system({"answer_language": "en"}))

    def test_env_is_used_when_config_is_silent(self):
        os.environ["RAG_ANSWER_LANGUAGE"] = "match"
        self.assertIn("질문과 같은 언어로", _system())

    def test_config_wins_over_env(self):
        os.environ["RAG_ANSWER_LANGUAGE"] = "en"
        self.assertIn("한국어로", _system({"answer_language": "ko"}))

    def test_unknown_value_falls_back_to_korean(self):
        """오타가 프롬프트를 깨뜨리지 않게 — 모르는 값이면 기존 동작."""
        self.assertIn("한국어로", _system({"answer_language": "zzz"}))

    def test_language_applies_to_both_grounding_modes(self):
        """grounding_strict 분기가 둘이라 한쪽만 고치면 조용히 어긋난다."""
        loose = _system({"answer_language": "en", "grounding_strict": False})
        strict = _system({"answer_language": "en", "grounding_strict": True})
        self.assertIn("in English", loose)
        self.assertIn("in English", strict)
        self.assertNotIn("한국어로", loose)
        self.assertNotIn("한국어로", strict)

    def test_abstention_marker_stays_korean(self):
        """기권 문구는 언어와 함께 바꾸지 않는다.

        metrics_basic._ABSTENTION_MARKERS 가 이 문자열로 기권을 판별한다. 영어 마커도
        함께 등록돼 있어(cannot answer 등) 모델이 영어로 기권해도 잡히지만, 지시문의
        예시 문구까지 번역하면 한국어 실행의 판별이 흔들린다.
        """
        self.assertIn("제공된 정보로는 알 수 없습니다",
                      _system({"answer_language": "en"}))


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
