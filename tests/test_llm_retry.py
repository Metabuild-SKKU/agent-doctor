import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.llm_retry import is_rate_limit


class FakeExc(Exception):
    def __init__(self, message="", status_code=None, code=None):
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code


class IsRateLimitTest(unittest.TestCase):
    def test_status_code_429_is_rate_limit(self):
        self.assertTrue(is_rate_limit(FakeExc("boom", status_code=429)))

    def test_code_429_is_rate_limit(self):
        self.assertTrue(is_rate_limit(FakeExc("boom", code=429)))

    def test_quota_message_is_rate_limit(self):
        self.assertTrue(is_rate_limit(Exception("Quota exceeded for requests")))

    def test_resource_exhausted_message_is_rate_limit(self):
        self.assertTrue(is_rate_limit(Exception("RESOURCE_EXHAUSTED: rate limited")))

    def test_rate_limit_message_is_rate_limit(self):
        self.assertTrue(is_rate_limit(Exception("Rate limit reached, too many requests")))

    def test_context_length_exceeded_is_not_rate_limit(self):
        self.assertFalse(
            is_rate_limit(Exception("This model's maximum context_length_exceeded"))
        )

    def test_token_count_exceeded_is_not_rate_limit(self):
        self.assertFalse(
            is_rate_limit(Exception("token count exceeded maximum allowed"))
        )

    def test_unrelated_error_is_not_rate_limit(self):
        self.assertFalse(is_rate_limit(Exception("invalid API key")))

    def test_insufficient_quota_is_not_retried(self):
        # OpenAI 크레딧 소진. 같은 429 지만 기다려도 풀리지 않는다.
        self.assertFalse(is_rate_limit(FakeExc(
            "Error code: 429 - {'error': {'message': 'You exceeded your current quota', "
            "'type': 'insufficient_quota'}}", status_code=429)))

    def test_insufficient_quota_code_is_not_retried(self):
        self.assertFalse(is_rate_limit(FakeExc("boom", code="insufficient_quota")))

    def test_gemini_resource_exhausted_quota_is_still_retried(self):
        # Gemini 무료 티어의 429 도 "quota" 문구를 쓴다 — 이쪽은 재시도가 맞다.
        self.assertTrue(is_rate_limit(Exception(
            "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded "
            "your current quota, please check your plan and billing details.', "
            "'status': 'RESOURCE_EXHAUSTED'}}")))


if __name__ == "__main__":
    unittest.main()
