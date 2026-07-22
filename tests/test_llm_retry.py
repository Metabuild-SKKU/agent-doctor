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


if __name__ == "__main__":
    unittest.main()
