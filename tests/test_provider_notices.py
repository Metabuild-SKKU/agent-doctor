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

    def test_openrouter_is_known_and_silent(self):
        # OpenRouter transport 를 추가하면서 _KNOWN_PROVIDERS 갱신을 빠뜨리면
        # openrouter 가 "미지원 값"으로 openai 에 폴백해 유료 호출이 엉뚱한 곳으로
        # 청구된다. 파일 내 위치가 달라 git 이 충돌로 잡아주지 않으므로 핀으로 고정한다.
        self.assertEqual(self._provider("openrouter"), ("openrouter", ""))

    def test_openrouter_spelling_variants_are_accepted(self):
        for value in ("open_router", "open-router", "openrouter_ai", "openrouter.ai"):
            with self.subTest(value=value):
                self.assertEqual(self._provider(value), ("openrouter", ""))

    def test_openrouter_is_case_insensitive(self):
        self.assertEqual(self._provider("OpenRouter"), ("openrouter", ""))

    def test_every_transport_has_a_known_provider(self):
        # 위 핀들의 일반형 — 새 provider transport(_<name>_generate)를 추가하고
        # _KNOWN_PROVIDERS 갱신을 잊으면 그 transport 는 도달 불가 코드가 된다.
        transports = {
            name[1:-len("_generate")]
            for name in vars(llm_provider)
            if name.startswith("_") and name.endswith("_generate")
        }
        self.assertEqual(transports - llm_provider._KNOWN_PROVIDERS, set())


class ProviderSpellingSymmetryTest(unittest.TestCase):
    """EVAL_LLM_PROVIDER 와 RAG_LLM_PROVIDER 는 같은 철자를 받아야 한다.

    한쪽만 받아주면 심판(Eval)은 지정한 provider 로 돌고 답변 생성(RAG)만 extractive 로
    저하돼, 평가가 저하된 RAG 를 재는 상태가 된다. 두 표가 갈라지는 걸 여기서 막는다."""

    def setUp(self):
        generator._warned_providers.clear()

    def test_alias_tables_are_in_sync(self):
        self.assertEqual(llm_provider._PROVIDER_ALIASES, generator._PROVIDER_ALIASES)

    def test_rag_accepts_every_eval_alias(self):
        for spelling, canonical in llm_provider._PROVIDER_ALIASES.items():
            with self.subTest(spelling=spelling):
                self.assertEqual(generator._selected_provider(spelling), canonical)

    def test_rag_normalizes_case_and_whitespace(self):
        self.assertEqual(generator._selected_provider("  OpenRouter "), "openrouter")

    def test_rag_alias_reaches_openrouter_transport(self):
        # 정규화가 _has_provider()/_llm_generate() 양쪽에 모두 걸리는지 —
        # 한쪽만 걸리면 generation_mode 표기와 실제 호출이 어긋난다.
        env = {"OPENROUTER_API_KEY": "or-dummy", "RAG_LLM_PROVIDER": "open_router"}
        buf = io.StringIO()
        with patch.dict(os.environ, env, clear=False), \
                patch.object(generator, "_openrouter_generate", return_value="OR 답변"), \
                redirect_stdout(buf):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("GEMINI_API_KEY", None)
            self.assertTrue(generator._has_provider())
            self.assertEqual(generator._llm_generate("질문", ["컨텍스트"]), "OR 답변")
        self.assertEqual(buf.getvalue(), "")


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
            # auto 는 OPENAI_API_KEY 다음으로 OPENROUTER_API_KEY 도 본다 — 둘 다 없어야
            # keyword 로 떨어진다.
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("OPENROUTER_API_KEY", None)
            graph_index._extract(_chunk(), {})
        self.assertEqual(buf.getvalue(), "")

    def test_openrouter_auto_notice_names_openrouter_key(self):
        # OpenAI 키 없이 OpenRouter 키만 있는 실행에서 "OPENAI_API_KEY 감지" 라고
        # 찍으면 어느 키를 빼야 청크당 유료 호출이 멈추는지 알 수 없다.
        buf = io.StringIO()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-dummy"}, clear=False), \
                patch.object(graph_index, "_llm_entities", return_value=([], [])), \
                redirect_stdout(buf):
            os.environ.pop("OPENAI_API_KEY", None)
            graph_index._extract(_chunk(), {})
        log = buf.getvalue()
        self.assertIn("OPENROUTER_API_KEY 감지", log)
        self.assertNotIn("OPENAI_API_KEY 감지", log)


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


class EmbeddingUnavailableNoticeTest(unittest.TestCase):
    """임베딩을 못 쓰는 provider 조합에서 안내가 실행당 1회인지."""

    def setUp(self):
        llm_provider._embed_unavailable_notified = False

    def _embed(self, env):
        buf = io.StringIO()
        with patch.dict(os.environ, env, clear=False), redirect_stdout(buf):
            vecs = llm_provider.embed_texts(["가", "나"])
        return vecs, buf.getvalue()

    def test_openrouter_without_openai_key_returns_empty_and_notifies(self):
        env = {"EVAL_LLM_PROVIDER": "openrouter", "OPENAI_API_KEY": ""}
        vecs, log = self._embed(env)
        self.assertEqual(vecs, [])          # 예외 대신 빈 결과 → 호출부가 결측으로 흡수
        self.assertIn("임베딩", log)

    def test_notice_is_printed_once_per_run(self):
        env = {"EVAL_LLM_PROVIDER": "openrouter", "OPENAI_API_KEY": ""}
        self._embed(env)
        _, second = self._embed(env)
        self.assertEqual(second, "")        # probe 마다 반복되면 다른 로그를 덮는다

    def test_embeddings_available_reflects_active_provider(self):
        with patch.dict(os.environ, {"EVAL_LLM_PROVIDER": "gemini",
                                     "GEMINI_API_KEY": "k"}, clear=False):
            self.assertTrue(llm_provider.embeddings_available())
        with patch.dict(os.environ, {"EVAL_LLM_PROVIDER": "openrouter",
                                     "OPENAI_API_KEY": ""}, clear=False):
            self.assertFalse(llm_provider.embeddings_available())


class RagasTrackSurvivesMissingEmbeddingTest(unittest.TestCase):
    """임베딩이 없어도 RAGAS 트랙 전체가 날아가지 않는지(회귀).

    예전엔 _response_relevancy 의 임베딩 예외가 parallel_map → evaluate_real_track →
    _ragas_track 으로 전파돼 트랙이 통째로 {} 가 됐다. 심판 호출 비용은 이미 쓴 뒤라
    faithfulness·context_* 까지 잃고 진단 라벨이 안 붙었다."""

    def test_embedding_failure_yields_none_not_exception(self):
        from agents.eval import metrics_ragas

        with (
            patch.object(metrics_ragas, "_chat",
                         return_value={"question": "재생성 질문", "noncommittal": 0}),
            patch.object(metrics_ragas, "_embed",
                         side_effect=RuntimeError("Missing credentials")),
        ):
            score = metrics_ragas._response_relevancy(
                object(), "질문?", "답변입니다.")
        self.assertIsNone(score)   # 예외 전파가 아니라 결측(None)

    def test_sibling_metrics_still_computed(self):
        """같은 트랙의 다른 지표는 임베딩과 무관하게 계산돼야 한다."""
        from agents.eval import metrics_ragas

        # faithfulness 는 chat 2회(문장 분해 → NLI 판정)를 쓰고 임베딩은 안 쓴다.
        with (
            patch.object(metrics_ragas, "_chat", side_effect=[
                {"statements": ["주장 하나"]},          # 1) 분해
                {"statements": [{"verdict": 1}]},       # 2) NLI 판정
            ]),
            patch.object(metrics_ragas, "_embed",
                         side_effect=RuntimeError("Missing credentials")),
        ):
            faith = metrics_ragas._faithfulness(
                object(), "질문?", "답변입니다.", ["근거 문장."])
        self.assertEqual(faith, 1.0)


if __name__ == "__main__":
    unittest.main()
