"""
tests/test_provider_notices.py
OPENAI_API_KEY 존재만으로 바뀌는 경로의 안내 로그 테스트.

Index 그래프 추출과 RAG auto 는 키가 있으면 말없이 OpenAI 를 쓴다. 그 전환을
프로세스당 한 줄만 알리는지, 그리고 Eval provider 오타가 조용히 openai 로
떨어지지 않는지 확인한다.
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.eval import llm_provider
from agents.index import graph_index
from agents.rag import generator
from core.schema import Chunk


def _chunk(text: str = "본문") -> Chunk:
    return Chunk(chunk_id="c1", doc_id="d1", text=text, hash="h1")


class EvalProviderFallbackTest(unittest.TestCase):
    def setUp(self):
        llm_provider._warned_providers.clear()

    def _provider(self, value: str | None) -> tuple[str, str]:
        env = {} if value is None else {"EVAL_LLM_PROVIDER": value}
        buf = io.StringIO()
        with patch.dict(os.environ, env, clear=False), redirect_stdout(buf):
            if value is None:
                os.environ.pop("EVAL_LLM_PROVIDER", None)
            provider = llm_provider._provider()
        return provider, buf.getvalue()

    def test_unset_defaults_to_openai_silently(self):
        self.assertEqual(self._provider(None), ("openai", ""))

    def test_empty_value_is_silent(self):
        self.assertEqual(self._provider(""), ("openai", ""))

    def test_known_provider_is_silent(self):
        self.assertEqual(self._provider("gemini"), ("gemini", ""))

    def test_typo_falls_back_to_openai_with_warning(self):
        provider, log = self._provider("gemeni")
        self.assertEqual(provider, "openai")
        self.assertIn("알 수 없는", log)

    def test_typo_warns_once_per_value(self):
        self._provider("gemeni")
        _, second = self._provider("gemeni")
        self.assertEqual(second, "")

    def test_github_models_alias_is_accepted(self):
        # RAG(_llm_generate)가 받아주는 철자라 여기서도 같은 값을 받는다.
        self.assertEqual(self._provider("github_models"), ("github", ""))


class IndexGraphNoticeTest(unittest.TestCase):
    def setUp(self):
        graph_index._llm_extraction_notified = False

    def _extract(self, config: dict) -> str:
        buf = io.StringIO()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-dummy"}, clear=False), \
                patch.object(graph_index, "_llm_entities", return_value=([], [])), \
                redirect_stdout(buf):
            graph_index._extract(_chunk(), config)
        return buf.getvalue()

    def test_auto_mode_notifies_once(self):
        first = self._extract({})
        self.assertIn("OPENAI_API_KEY 감지", first)
        self.assertIn("graph_extraction=keyword", first)
        self.assertEqual(self._extract({}), "")

    def test_notice_names_the_configured_model(self):
        self.assertIn("gpt-4.1-nano", self._extract({"graph_llm_model": "gpt-4.1-nano"}))

    def test_explicit_llm_mode_is_silent(self):
        # 명시적으로 켠 경우는 놀랄 일이 아니라 알리지 않는다.
        self.assertEqual(self._extract({"graph_extraction": "llm"}), "")

    def test_keyword_mode_is_silent(self):
        self.assertEqual(self._extract({"graph_extraction": "keyword"}), "")

    def test_no_key_is_silent(self):
        buf = io.StringIO()
        with patch.dict(os.environ, {}, clear=False), redirect_stdout(buf):
            os.environ.pop("OPENAI_API_KEY", None)
            graph_index._extract(_chunk(), {})
        self.assertEqual(buf.getvalue(), "")


class RagAutoNoticeTest(unittest.TestCase):
    def setUp(self):
        generator._notified_auto_openai = False

    def _generate(self, openai_result: str | None) -> tuple[str | None, str]:
        env = {"OPENAI_API_KEY": "sk-dummy", "GEMINI_API_KEY": "g-dummy"}
        buf = io.StringIO()
        with patch.dict(os.environ, env, clear=False), \
                patch.object(generator, "_openai_generate", return_value=openai_result), \
                patch.object(generator, "_gemini_generate", return_value="gemini 답변"), \
                redirect_stdout(buf):
            os.environ.pop("RAG_LLM_PROVIDER", None)  # 미설정 = auto
            answer = generator._llm_generate("질문", ["컨텍스트"])
        return answer, buf.getvalue()

    def test_notifies_after_openai_answers(self):
        answer, log = self._generate("openai 답변")
        self.assertEqual(answer, "openai 답변")
        self.assertIn("auto → OpenAI", log)

    def test_notifies_once(self):
        self._generate("openai 답변")
        _, second = self._generate("openai 답변")
        self.assertEqual(second, "")

    def test_no_notice_when_openai_falls_back(self):
        # 키가 만료돼 Gemini 로 넘어간 실행에 "OpenAI 사용" 이라고 적으면 거짓말이 된다.
        answer, log = self._generate(None)
        self.assertEqual(answer, "gemini 답변")
        self.assertNotIn("auto → OpenAI", log)


if __name__ == "__main__":
    unittest.main()
