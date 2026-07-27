"""
tests/test_llm_provider.py
llm_provider.chat_json 단위 테스트.

chat_json 은 그동안 직접 테스트가 없었다(합성 경로 테스트는 chat_json 을 목킹해 우회).
여기서는 (1) Gemini 가 dict 를 [ {…} ] 로 감싸 반환한 경우의 언랩, (2) 빈 응답/파싱
실패/타입 불일치의 사유별 로그를 검증한다. 실제 API 대신 _run_with_retry 를 patch 해
raw 응답 문자열만 주입한다(transport·재시도는 core 쪽 관심사).
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.eval import llm_provider


def _chat_json_with_raw(raw: str):
    """_run_with_retry 를 patch 해 raw 를 주입하고 (반환값, stdout) 을 돌려준다."""
    buf = io.StringIO()
    with patch.object(llm_provider, "_run_with_retry", return_value=raw), \
            redirect_stdout(buf):
        result = llm_provider.chat_json("sys", "user")
    return result, buf.getvalue()


class ChatJsonUnwrapTest(unittest.TestCase):
    def test_dict_passthrough(self):
        result, _ = _chat_json_with_raw('{"question": "q", "ground_truth": "a"}')
        self.assertEqual(result, {"question": "q", "ground_truth": "a"})

    def test_single_element_list_is_unwrapped(self):
        # Gemini 가 dict 를 한 겹 감싸 반환한 경우.
        result, _ = _chat_json_with_raw('[{"question": "q", "ground_truth": "a"}]')
        self.assertEqual(result, {"question": "q", "ground_truth": "a"})

    def test_multi_element_list_is_not_unwrapped(self):
        # 길이 2+ 는 스키마 위반 — 억지로 풀지 않는다.
        result, log = _chat_json_with_raw('[{"a": 1}, {"b": 2}]')
        self.assertEqual(result, {})
        self.assertIn("타입 불일치", log)

    def test_list_of_non_dict_is_not_unwrapped(self):
        result, log = _chat_json_with_raw('[1]')
        self.assertEqual(result, {})
        self.assertIn("타입 불일치", log)


class ChatJsonFailureReasonTest(unittest.TestCase):
    def test_empty_response_logs_reason(self):
        result, log = _chat_json_with_raw("")
        self.assertEqual(result, {})
        self.assertIn("빈 응답", log)

    def test_whitespace_only_response_is_treated_as_empty(self):
        result, log = _chat_json_with_raw("   \n  ")
        self.assertEqual(result, {})
        self.assertIn("빈 응답", log)

    def test_parse_failure_logs_reason(self):
        result, log = _chat_json_with_raw("이건 JSON 이 아니다")
        self.assertEqual(result, {})
        self.assertIn("파싱 실패", log)


if __name__ == "__main__":
    unittest.main()
