"""
tests/test_llm_clients.py
openai_chat 의 모델별 파라미터 분기 단위 테스트.

추론 모델(o-series/gpt-5)은 max_tokens/temperature 를 거부하고 내부 추론 토큰이
출력 상한을 함께 소진한다. 실제 API 대신 openai.OpenAI 를 목으로 대체해
create() 에 실린 인자와 경고 로그만 검사한다.
"""
import io
import os
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import llm_clients
from core.llm_clients import _is_reasoning_model, openai_chat


class FakeOpenAI:
    """마지막 create() 인자를 클래스 변수에 남기는 목 클라이언트."""

    last_kwargs: dict = {}
    finish_reason: str | None = "stop"

    def __init__(self, **_):
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        FakeOpenAI.last_kwargs = kwargs
        choice = types.SimpleNamespace(
            message=types.SimpleNamespace(content="응답"),
            finish_reason=FakeOpenAI.finish_reason,
        )
        return types.SimpleNamespace(choices=[choice], usage=None)


def call(model: str, **kwargs) -> tuple[str, str]:
    """openai_chat 1회 호출 → (반환 텍스트, stdout)."""
    buf = io.StringIO()
    with patch("openai.OpenAI", FakeOpenAI), redirect_stdout(buf):
        text = openai_chat("sys", "user", model, **kwargs)
    return text, buf.getvalue()


class IsReasoningModelTest(unittest.TestCase):
    def test_reasoning_prefixes(self):
        for model in ("o1", "o1-preview", "o3-mini", "o4-mini", "gpt-5", "gpt-5-mini"):
            self.assertTrue(_is_reasoning_model(model), model)

    def test_publisher_prefixed_name(self):
        # GitHub Models 는 "<publisher>/<model>" 형식.
        self.assertTrue(_is_reasoning_model("openai/o3-mini"))
        self.assertFalse(_is_reasoning_model("openai/gpt-4o"))

    def test_non_reasoning_models(self):
        for model in ("gpt-4o", "gpt-4o-mini", "gpt-4.1-mini", "text-embedding-3-small"):
            self.assertFalse(_is_reasoning_model(model), model)

    def test_gpt5_chat_is_not_reasoning(self):
        # 접두사만 같고 temperature 를 받는 chat 모델.
        self.assertFalse(_is_reasoning_model("gpt-5-chat-latest"))


class OpenAiChatParamsTest(unittest.TestCase):
    def setUp(self):
        FakeOpenAI.last_kwargs = {}
        FakeOpenAI.finish_reason = "stop"
        llm_clients._warned_ignored_temperature.clear()

    def test_standard_model_uses_max_tokens_and_temperature(self):
        call("gpt-4o", max_output_tokens=2048, temperature=0.0)
        kwargs = FakeOpenAI.last_kwargs
        self.assertEqual(kwargs["max_tokens"], 2048)
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertNotIn("max_completion_tokens", kwargs)

    def test_reasoning_model_swaps_parameters(self):
        call("o3-mini", max_output_tokens=2048)
        kwargs = FakeOpenAI.last_kwargs
        self.assertNotIn("max_tokens", kwargs)
        self.assertNotIn("temperature", kwargs)
        self.assertEqual(kwargs["max_completion_tokens"],
                         llm_clients._REASONING_MIN_OUTPUT_TOKENS)

    def test_reasoning_model_keeps_larger_requested_cap(self):
        larger = llm_clients._REASONING_MIN_OUTPUT_TOKENS + 1000
        call("gpt-5-mini", max_output_tokens=larger)
        self.assertEqual(FakeOpenAI.last_kwargs["max_completion_tokens"], larger)

    def test_json_mode_sets_response_format(self):
        call("gpt-4o", json_mode=True)
        self.assertEqual(FakeOpenAI.last_kwargs["response_format"],
                         {"type": "json_object"})

    def test_ignored_temperature_warns_once_per_model(self):
        _, first = call("o3-mini", temperature=0.0)
        _, second = call("o3-mini", temperature=0.0)
        self.assertIn("temperature", first)
        self.assertEqual(second, "")

    def test_standard_model_does_not_warn(self):
        _, out = call("gpt-4o", temperature=0.7)
        self.assertEqual(out, "")

    def test_truncated_response_is_logged(self):
        FakeOpenAI.finish_reason = "length"
        _, out = call("gpt-4o")
        self.assertIn("finish_reason=length", out)


if __name__ == "__main__":
    unittest.main()
