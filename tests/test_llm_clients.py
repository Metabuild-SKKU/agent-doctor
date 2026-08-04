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

    def test_openrouter_disables_reasoning_by_default(self):
        # 추론 토큰은 output 으로 과금된다. 기본은 끔.
        with patch.dict(os.environ, {"OPENROUTER_REASONING": ""}, clear=False):
            call("deepseek/deepseek-v4-flash-0731",
                 base_url=llm_clients.OPENROUTER_BASE_URL)
        extra = FakeOpenAI.last_kwargs["extra_body"]
        self.assertEqual(extra["reasoning"], {"enabled": False})
        self.assertEqual(extra["usage"], {"include": True})   # 비용 집계는 유지

    def test_openrouter_reasoning_can_be_reenabled(self):
        with patch.dict(os.environ, {"OPENROUTER_REASONING": "1"}, clear=False):
            call("deepseek/deepseek-v4-flash-0731",
                 base_url=llm_clients.OPENROUTER_BASE_URL)
        self.assertNotIn("reasoning", FakeOpenAI.last_kwargs["extra_body"])

    def test_non_openrouter_endpoints_are_untouched(self):
        # OpenRouter 전용 파라미터라 순정 OpenAI·GitHub Models 로는 보내지 않는다.
        call("gpt-4o")
        self.assertNotIn("extra_body", FakeOpenAI.last_kwargs)

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
class ReasoningOffSkipsLargeCapTest(unittest.TestCase):
    """추론을 끈 호출에는 큰 출력 예산 승급을 적용하지 않는다.

    승급의 근거가 "추론 토큰이 상한을 먹는다" 인데, 추론이 없으면 근거가 사라진다.
    남겨두면 호출부가 정한 상한(RAG 답변 4096 등)이 조용히 25K 로 덮인다."""

    def setUp(self):
        FakeOpenAI.last_kwargs = {}
        FakeOpenAI.calls = []
        FakeOpenAI.finish_reason = "stop"
        FakeOpenAI.content = "응답"
        FakeOpenAI.script = None

    def test_caller_cap_survives_when_reasoning_is_off(self):
        with patch.dict(os.environ, {"OPENROUTER_REASONING": "0"}, clear=False):
            call("deepseek/deepseek-v4-flash-0731", max_output_tokens=4096,
                 base_url=llm_clients.OPENROUTER_BASE_URL)
        self.assertEqual(FakeOpenAI.last_kwargs["max_tokens"], 4096)

    def test_cap_is_raised_again_when_reasoning_is_on(self):
        with patch.dict(os.environ, {"OPENROUTER_REASONING": "1"}, clear=False):
            call("deepseek/deepseek-v4-flash-0731", max_output_tokens=4096,
                 base_url=llm_clients.OPENROUTER_BASE_URL)
        self.assertEqual(FakeOpenAI.last_kwargs["max_tokens"],
                         llm_clients._LARGE_OUTPUT_MIN_TOKENS)

    def test_non_openrouter_endpoint_keeps_the_bump(self):
        # 추론 끄기는 OpenRouter 전용 파라미터라, 다른 엔드포인트에선 추론이 살아 있다.
        with patch.dict(os.environ, {"OPENROUTER_REASONING": "0"}, clear=False):
            call("deepseek/deepseek-v4-flash-0731", max_output_tokens=4096)
        self.assertEqual(FakeOpenAI.last_kwargs["max_tokens"],
                         llm_clients._LARGE_OUTPUT_MIN_TOKENS)

class JsonModeKeepsLargeCapTest(unittest.TestCase):
    """json_mode 는 추론을 꺼도 큰 출력 예산 승급을 유지한다.

    잘린 JSON 은 파싱 실패로 전량 손실이고 재시도가 반드시 걸려 "잘린 값 + 재시도 값"
    을 둘 다 지불한다. 실측: 이 예외 없이 fused RAGAS 를 돌리면 4096 에서 잘려
    결손 13건·재시도 4건(건당 4,096+25,000=29,096 토큰), 승급하면 잘림 0건."""

    def setUp(self):
        FakeOpenAI.last_kwargs = {}
        FakeOpenAI.calls = []
        FakeOpenAI.finish_reason = "stop"
        FakeOpenAI.content = '{"ok": 1}'
        FakeOpenAI.script = None

    def test_json_mode_keeps_bump_even_with_reasoning_off(self):
        with patch.dict(os.environ, {"OPENROUTER_REASONING": "0"}, clear=False):
            call("deepseek/deepseek-v4-flash-0731", json_mode=True,
                 max_output_tokens=4096, base_url=llm_clients.OPENROUTER_BASE_URL)
        self.assertEqual(FakeOpenAI.last_kwargs["max_tokens"],
                         llm_clients._LARGE_OUTPUT_MIN_TOKENS)

    def test_prose_still_honours_caller_cap(self):
        # e5e1734 의 의도 — RAG 답변은 잘려도 부분 답변이 쓸모 있어 좁은 상한이 이득.
        with patch.dict(os.environ, {"OPENROUTER_REASONING": "0"}, clear=False):
            call("deepseek/deepseek-v4-flash-0731", json_mode=False,
                 max_output_tokens=4096, base_url=llm_clients.OPENROUTER_BASE_URL)
        self.assertEqual(FakeOpenAI.last_kwargs["max_tokens"], 4096)

    def test_json_mode_never_pays_twice(self):
        # 승급값이 재시도 목표치 이상이라 retry_cap > cap 이 거짓 → 재시도가 안 열린다.
        FakeOpenAI.script = [("", "length")]
        with patch.dict(os.environ, {"OPENROUTER_REASONING": "0"}, clear=False):
            call("deepseek/deepseek-v4-flash-0731", json_mode=True,
                 max_output_tokens=4096, base_url=llm_clients.OPENROUTER_BASE_URL)
        self.assertEqual(len(FakeOpenAI.calls), 1)


class TestFileWiringTest(unittest.TestCase):
    """__main__ 블록 아래에 클래스를 덧붙이면 파일 직접 실행에서 그 테스트가 안 돈다.

    실제로 이 PR 의 회귀 핀 12개가 그렇게 묻혔다(직접 27 vs -m unittest 30).
    블록이 파일 맨 끝에 하나만 있는지 고정한다."""

    def test_main_block_is_last_in_test_files(self):
        import pathlib as _p

        root = _p.Path(__file__).parent
        for name in ("test_llm_clients.py", "test_provider_notices.py"):
            lines = (root / name).read_text(encoding="utf-8").splitlines()
            # 들여쓰기 없는 선언만 센다 — 이 테스트 본문에도 같은 문자열이 들어 있어
            # 원문 전체를 세면 자기 자신을 잡는다.
            guards = [i for i, ln in enumerate(lines) if ln.startswith("if __name__")]
            classes = [i for i, ln in enumerate(lines) if ln.startswith("class ")]
            with self.subTest(name=name):
                self.assertEqual(len(guards), 1, "__main__ 블록은 하나여야 한다")
                self.assertTrue(classes, "테스트 클래스가 없다")
                self.assertLess(max(classes), guards[0],
                                "__main__ 블록 뒤의 클래스는 직접 실행에서 안 돈다")


if __name__ == "__main__":
    unittest.main()
