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
    """create() 호출을 기록하는 목 클라이언트.

    last_kwargs 는 마지막 호출, calls 는 전체 호출 이력(재시도 검증용).
    script 를 주면 호출 순서대로 (content, finish_reason) 을 소비한다."""

    last_kwargs: dict = {}
    calls: list = []
    finish_reason: str | None = "stop"
    content: str = "응답"
    script: list | None = None

    def __init__(self, **_):
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        FakeOpenAI.last_kwargs = kwargs
        FakeOpenAI.calls.append(kwargs)
        if FakeOpenAI.script:
            content, finish = FakeOpenAI.script.pop(0)
        else:
            content, finish = FakeOpenAI.content, FakeOpenAI.finish_reason
        choice = types.SimpleNamespace(
            message=types.SimpleNamespace(content=content),
            finish_reason=finish,
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


class NeedsLargeOutputTest(unittest.TestCase):
    """추론 토큰을 뱉지만 API 규약은 일반 모델과 같은 계열(_LARGE_OUTPUT_PREFIXES)."""

    def test_deepseek_needs_large_output(self):
        for model in ("deepseek-v4-flash-0731", "deepseek/deepseek-v4-flash-0731",
                      "deepseek/deepseek-chat"):
            self.assertTrue(llm_clients._needs_large_output(model), model)

    def test_deepseek_is_not_openai_style_reasoning_model(self):
        # temperature/max_tokens 규약은 일반 모델과 같다 — 여기에 걸리면 temperature 가
        # 버려져 Optimize 의 generation.temperature 스윕이 no-op 이 된다.
        self.assertFalse(_is_reasoning_model("deepseek/deepseek-v4-flash-0731"))

    def test_ordinary_models_do_not_need_large_output(self):
        for model in ("gpt-4o", "upstage/solar-pro-3", "google/gemini-3.1-flash-lite"):
            self.assertFalse(llm_clients._needs_large_output(model), model)


class OpenAiChatParamsTest(unittest.TestCase):
    def setUp(self):
        FakeOpenAI.last_kwargs = {}
        FakeOpenAI.calls = []
        FakeOpenAI.finish_reason = "stop"
        FakeOpenAI.content = "응답"
        FakeOpenAI.script = None
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

    def test_deepseek_gets_large_cap_but_keeps_temperature(self):
        # 회귀: 2048 이면 추론 토큰이 상한을 다 써 JSON 본문이 안 나온다.
        call("deepseek/deepseek-v4-flash-0731", max_output_tokens=2048, temperature=0.3)
        kwargs = FakeOpenAI.last_kwargs
        self.assertEqual(kwargs["max_tokens"], llm_clients._LARGE_OUTPUT_MIN_TOKENS)
        self.assertEqual(kwargs["temperature"], 0.3)
        self.assertNotIn("max_completion_tokens", kwargs)

    def test_large_output_cap_does_not_open_a_retry_path(self):
        # 이 계열의 상한을 재시도 목표치보다 낮추면, 상한을 넘는 호출이 "잘린 응답 값 +
        # 재시도 값" 을 둘 다 지불한다(8K+25K=33K > 25K). 상한이 막으려던 폭주에서
        # 오히려 비싸지므로 낮추지 않는다 — 이 관계가 깨지면 이중 과금이 되살아난다.
        self.assertGreaterEqual(llm_clients._LARGE_OUTPUT_MIN_TOKENS,
                                llm_clients._REASONING_MIN_OUTPUT_TOKENS)

    def test_large_output_model_keeps_larger_requested_cap(self):
        # 호출부가 이미 더 큰 값을 줬으면(fused RAGAS 등) 그쪽을 존중한다.
        larger = llm_clients._LARGE_OUTPUT_MIN_TOKENS + 1000
        call("deepseek/deepseek-chat", max_output_tokens=larger)
        self.assertEqual(FakeOpenAI.last_kwargs["max_tokens"], larger)


class TruncationRetryTest(unittest.TestCase):
    """접두사 목록이 못 잡은 모델의 최종 안전망 — 잘리면 상한을 올려 1회만 재시도."""

    def setUp(self):
        FakeOpenAI.last_kwargs = {}
        FakeOpenAI.calls = []
        FakeOpenAI.finish_reason = "stop"
        FakeOpenAI.content = "응답"
        FakeOpenAI.script = None
        llm_clients._warned_ignored_temperature.clear()

    def test_json_mode_truncation_retries_once_with_larger_cap(self):
        FakeOpenAI.script = [("", "length"), ('{"ok": 1}', "stop")]
        text, out = call("some-new-model", json_mode=True, max_output_tokens=2048)
        self.assertEqual(text, '{"ok": 1}')
        self.assertEqual(len(FakeOpenAI.calls), 2)
        self.assertEqual(FakeOpenAI.calls[0]["max_tokens"], 2048)
        self.assertGreater(FakeOpenAI.calls[1]["max_tokens"], 2048)
        self.assertIn("재시도", out)

    def test_empty_content_truncation_retries(self):
        FakeOpenAI.script = [("", "length"), ("본문", "stop")]
        text, _ = call("some-new-model", max_output_tokens=2048)
        self.assertEqual(text, "본문")
        self.assertEqual(len(FakeOpenAI.calls), 2)

    def test_usable_prose_truncation_does_not_retry(self):
        # 산문이 길어 잘린 건 부분 답변이라도 쓸모가 있다. 무조건 재시도하면 비용이 두 배.
        FakeOpenAI.script = [("긴 답변인데 잘림", "length")]
        text, out = call("some-new-model", max_output_tokens=2048)
        self.assertEqual(text, "긴 답변인데 잘림")
        self.assertEqual(len(FakeOpenAI.calls), 1)
        self.assertIn("finish_reason=length", out)

    def test_retry_happens_at_most_once(self):
        # 재시도 후에도 잘리면 포기한다 — 비용 상한이 예측 가능해야 한다.
        FakeOpenAI.script = [("", "length"), ("", "length")]
        text, out = call("some-new-model", json_mode=True, max_output_tokens=2048)
        self.assertEqual(text, "")
        self.assertEqual(len(FakeOpenAI.calls), 2)
        self.assertIn("finish_reason=length", out)

    def test_no_retry_when_cap_already_at_ceiling(self):
        # 이미 상한이 재시도 목표치 이상이면 같은 값으로 다시 부를 이유가 없다.
        FakeOpenAI.script = [("", "length")]
        call("o3-mini", json_mode=True,
             max_output_tokens=llm_clients._REASONING_MIN_OUTPUT_TOKENS * 4)
        self.assertEqual(len(FakeOpenAI.calls), 1)

    def test_reasoning_model_starts_at_target_so_never_retries(self):
        # 추론 모델은 첫 호출부터 25K 를 받으므로 올릴 여지가 없다 —
        # 잘렸다고 같은 상한으로 다시 부르면 비용만 두 배가 된다.
        FakeOpenAI.script = [("", "length")]
        call("o3-mini", json_mode=True, max_output_tokens=2048)
        self.assertEqual(len(FakeOpenAI.calls), 1)
        self.assertEqual(FakeOpenAI.calls[0]["max_completion_tokens"],
                         llm_clients._REASONING_MIN_OUTPUT_TOKENS)

    def test_large_output_model_never_pays_twice(self):
        # 목록에 잡힌 계열은 첫 호출부터 재시도 목표치로 시작하므로, 잘려도 재시도가
        # 열리지 않는다 — 이중 과금(잘린 응답 + 재시도) 이 발생하면 안 된다.
        FakeOpenAI.script = [("", "length")]
        call("deepseek/deepseek-v4-flash-0731", json_mode=True, max_output_tokens=2048)
        self.assertEqual(len(FakeOpenAI.calls), 1)
        self.assertEqual(FakeOpenAI.calls[0]["max_tokens"],
                         llm_clients._REASONING_MIN_OUTPUT_TOKENS)


if __name__ == "__main__":
    unittest.main()
