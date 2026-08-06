import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.llm_retry import is_rate_limit, is_transient


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


class TransientClassificationTests(unittest.TestCase):
    """재시도 기준은 "다시 부르면 성공할 가능성이 있나" 하나다."""

    def test_5xx_is_transient(self):
        # 색인 임베딩이 이 경로를 탄다. 문서 하나가 청크 800개면 API 를 13번 부르는데,
        # 그중 한 번의 5xx 로 문서 전체가 색인에서 빠졌다.
        for status in (500, 502, 503, 504):
            with self.subTest(status=status):
                self.assertTrue(is_transient(FakeExc("boom", status_code=status)))

    def test_5xx_message_is_transient(self):
        self.assertTrue(is_transient(Exception("503 Service Unavailable")))
        self.assertTrue(is_transient(Exception("Bad gateway")))

    def test_timeout_and_connection_are_transient(self):
        self.assertTrue(is_transient(TimeoutError("read timed out")))
        self.assertTrue(is_transient(ConnectionResetError("connection reset by peer")))
        self.assertTrue(is_transient(Exception("Connection error.")))

    def test_rate_limit_still_transient(self):
        self.assertTrue(is_transient(FakeExc("boom", status_code=429)))

    def test_auth_and_model_errors_are_not_transient(self):
        # 다시 불러도 같은 결과다. 재시도는 실패를 수십 초 늦출 뿐이고,
        # 그동안 원인(키 오타 등)이 로그에 안 드러난다.
        self.assertFalse(is_transient(FakeExc("invalid api key", status_code=401)))
        self.assertFalse(is_transient(FakeExc("model not found", status_code=404)))
        self.assertFalse(is_transient(FakeExc("bad request", status_code=400)))

    def test_insufficient_quota_is_not_transient(self):
        # 크레딧 소진은 기다려도 안 풀린다(기존 규칙 유지).
        self.assertFalse(is_transient(FakeExc("boom", code="insufficient_quota")))

    def test_plain_error_is_not_transient(self):
        self.assertFalse(is_transient(ValueError("schema mismatch")))


class TransientFalsePositiveTests(unittest.TestCase):
    """숫자만으로 5xx 를 판정하면 영구 실패를 5회씩 재시도한다.

    "500" 같은 부분 문자열은 토큰 수·모델명·청크 번호에 그대로 걸린다. 그러면
    40초를 버리고도 실패하며, 그동안 진짜 원인(키 오타 등)이 로그에 안 뜬다.
    """

    def test_numbers_in_message_are_not_5xx(self):
        for msg in (
            "This model supports a maximum of 500 tokens",
            "input must be under 8500 characters",
            "invalid model: gpt-500-turbo",
            "chunk 5041 exceeded limit",
        ):
            with self.subTest(msg=msg):
                self.assertFalse(is_transient(Exception(msg)))

    def test_status_context_is_5xx(self):
        for msg in (
            "Error code: 503 - service unavailable",
            "HTTP 502 Bad Gateway",
            "status 500 returned",
        ):
            with self.subTest(msg=msg):
                self.assertTrue(is_transient(Exception(msg)))


if __name__ == "__main__":
    unittest.main()
