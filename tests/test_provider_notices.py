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
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("OPENROUTER_API_KEY", None)
            graph_index._extract(_chunk(), {})
        self.assertEqual(buf.getvalue(), "")

    def test_openrouter_key_alone_does_not_trigger_auto(self):
        # auto 는 OPENROUTER_API_KEY 로 켜지지 않는다 — Eval/RAG 용으로 넣은 키가
        # 검색 품질과 무관한 이 단계의 청크당 유료 호출을 켜면 안 된다.
        buf = io.StringIO()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-dummy"}, clear=False), \
                patch.object(graph_index, "_llm_entities", return_value=([], [])), \
                redirect_stdout(buf):
            os.environ.pop("OPENAI_API_KEY", None)
            _e, _r, mode = graph_index._extract(_chunk(), {})
        self.assertEqual(mode, "keyword")
        self.assertEqual(buf.getvalue(), "")

    def test_explicit_openrouter_notice_names_openrouter_key(self):
        # 명시적으로 켠 경우엔 "OPENAI_API_KEY 감지" 라고 찍으면 어느 키를 빼야
        # 청크당 유료 호출이 멈추는지 알 수 없다.
        buf = io.StringIO()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-dummy",
                                     "INDEX_LLM_PROVIDER": "openrouter"}, clear=False), \
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


class GraphExtractionDefaultOffTest(unittest.TestCase):
    """지식그래프 단계가 기본으로 꺼져 있는지(회귀).

    켜져 있으면 graph_extraction="auto" 가 API 키만 있으면 청크마다 LLM 을 부른다 —
    Eval/RAG 용으로 넣은 키가 검색 품질과 무관한 단계까지 켜서 과금이 발생했다."""

    def test_graph_disabled_by_default(self):
        from core.state import AgentDoctorState

        self.assertFalse(AgentDoctorState().index_config["graph_enabled"])

    def test_llm_extraction_is_unreachable_while_disabled(self):
        # graph_enabled=False 면 build_graph_artifacts 자체를 안 부르므로
        # graph_extraction 값과 무관하게 LLM 호출 경로에 도달하지 않는다.
        from core.state import AgentDoctorState

        config = AgentDoctorState().index_config
        self.assertFalse(config.get("graph_enabled", True))


class GraphAutoExcludesOpenRouterTest(unittest.TestCase):
    """auto 는 OpenRouter 키로 켜지지 않는다(명시적 opt-in 필요).

    Eval/RAG 용으로 넣은 키 하나가 청크당 1회 호출을 켜는 것을 막는다."""

    _ONLY_OR = {"OPENAI_API_KEY": "", "OPENROUTER_API_KEY": "sk-or-test",
                "INDEX_LLM_PROVIDER": ""}

    def test_auto_does_not_use_openrouter(self):
        with patch.dict(os.environ, self._ONLY_OR, clear=False):
            self.assertIsNone(graph_index._graph_llm_target({}))

    def test_explicit_openrouter_still_works(self):
        env = dict(self._ONLY_OR, INDEX_LLM_PROVIDER="openrouter")
        with patch.dict(os.environ, env, clear=False):
            target = graph_index._graph_llm_target({})
        self.assertIsNotNone(target)
        self.assertEqual(target[1], "https://openrouter.ai/api/v1")

    def test_config_key_also_opts_in(self):
        with patch.dict(os.environ, self._ONLY_OR, clear=False):
            target = graph_index._graph_llm_target({"graph_llm_provider": "openrouter"})
        self.assertIsNotNone(target)

    def test_auto_still_uses_openai_key(self):
        env = {"OPENAI_API_KEY": "sk-test", "OPENROUTER_API_KEY": "",
               "INDEX_LLM_PROVIDER": ""}
        with patch.dict(os.environ, env, clear=False):
            target = graph_index._graph_llm_target({})
        self.assertIsNotNone(target)
        self.assertIsNone(target[1])       # base_url 없음 = 순정 OpenAI

    def test_cache_signature_tracks_actual_provider(self):
        # keyword 로 만든 캐시가 openrouter 로 켠 실행에 재사용되면 안 된다.
        from agents.index.agent import _graph_cache_signature

        config = {"graph_extraction": "auto"}
        with patch.dict(os.environ, self._ONLY_OR, clear=False):
            off = _graph_cache_signature(config)
        with patch.dict(os.environ, dict(self._ONLY_OR,
                                         INDEX_LLM_PROVIDER="openrouter"), clear=False):
            on = _graph_cache_signature(config)
        self.assertFalse(off["llm_available"])
        self.assertTrue(on["llm_available"])


class EmbeddingFallbackTest(unittest.TestCase):
    """임베딩 API 를 못 쓰는 조합의 폴백 사슬: API > 로컬 BGE-M3 > 결측.

    OpenRouter 는 임베딩 엔드포인트가 있으므로(baai/bge-m3 등 31개) 키가 있으면
    API 경로를 탄다. 여기서 폴백을 보려면 그 키까지 비워야 한다 — 예전 주석은
    "OpenRouter 에 임베딩 모델 0개" 였는데 사실이 아니었다."""

    def setUp(self):
        llm_provider._embed_notified.clear()    # 메시지별 1회 알림(set) — 테스트 간 격리
        # 임베딩축을 비워 심판축을 따르게 되돌린다. 이게 없으면 실행 머신의 .env 에
        # EVAL_EMBED_PROVIDER=openrouter 가 있을 때 아래 대역(_openai_embed 하나)을
        # 우회해 **실제 키로 openrouter.ai 에 요청이 나간다** - clear=False 라 env 가
        # 살아 있고, _openrouter_embed 는 패치 대상이 아니다.
        # conftest.py 에도 같은 핀이 있지만 그건 pytest 전용이고, 이 프로젝트의 표준
        # 실행은 python -m unittest 라 여기 파일 단위 방어가 실제로 일하는 쪽이다.
        patcher = patch.dict(os.environ, {"EVAL_EMBED_PROVIDER": ""}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _embed(self, env, local_ok):
        buf = io.StringIO()
        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(llm_provider, "_local_embeddings_available",
                         return_value=local_ok),
            patch("agents.index.qdrant_store.embed_batch",
                  return_value=[[0.1, 0.2], [0.3, 0.4]]),
            redirect_stdout(buf),
        ):
            vecs = llm_provider.embed_texts(["가", "나"])
        return vecs, buf.getvalue()

    # provider 는 openrouter 지만 임베딩에 쓸 키가 하나도 없는 상태.
    _OR = {"EVAL_LLM_PROVIDER": "openrouter",
           "EVAL_EMBED_PROVIDER": "",          # 개발 머신 env 의 override 가 새지 않게 고정
           "OPENAI_API_KEY": "",
           "OPENROUTER_API_KEY": ""}

    def test_falls_back_to_local_embeddings(self):
        vecs, log = self._embed(self._OR, local_ok=True)
        self.assertEqual(vecs, [[0.1, 0.2], [0.3, 0.4]])
        self.assertIn("로컬 임베딩", log)

    def test_missing_only_when_local_also_unavailable(self):
        # 로컬 모델이 해시 폴백 상태면 쓰지 않는다 — 무의미한 코사인이 정상 점수처럼
        # 리포트에 박히는 것이 결측보다 나쁘다.
        vecs, log = self._embed(self._OR, local_ok=False)
        self.assertEqual(vecs, [])
        self.assertIn("건너뜁니다", log)

    def test_notice_is_printed_once_per_run(self):
        self._embed(self._OR, local_ok=True)
        _, second = self._embed(self._OR, local_ok=True)
        self.assertEqual(second, "")        # probe 마다 반복되면 다른 로그를 덮는다

    def test_api_key_still_wins_over_local(self):
        # 기존 사용자 동작 보존 — 키가 있으면 예전처럼 API 임베딩을 쓴다.
        called = {}

        def _fake_openai_embed(texts, model):
            called["model"] = model
            return [[1.0]]

        with (
            patch.dict(os.environ, {"EVAL_LLM_PROVIDER": "openai",
                                    "OPENAI_API_KEY": "sk-test"}, clear=False),
            patch.object(llm_provider, "_openai_embed", _fake_openai_embed),
            patch.object(llm_provider, "_local_embeddings_available",
                         return_value=True),
        ):
            vecs = llm_provider.embed_texts(["가"])
        self.assertEqual(vecs, [[1.0]])
        self.assertEqual(called["model"], "text-embedding-3-small")

    def test_embeddings_available_covers_local_path(self):
        with (
            patch.dict(os.environ, self._OR, clear=False),
            patch.object(llm_provider, "_local_embeddings_available", return_value=True),
        ):
            self.assertTrue(llm_provider.embeddings_available())
        with (
            patch.dict(os.environ, self._OR, clear=False),
            patch.object(llm_provider, "_local_embeddings_available", return_value=False),
        ):
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
class GenerationMaxTokensTest(unittest.TestCase):
    """답변 생성 출력 상한 — 공용 기본값(2048)은 긴 컨텍스트에서 잘렸다.

    잘린 답변은 그대로 채점에 들어가 faithfulness·answer_correctness 를 부당하게
    떨어뜨리므로, 상한 부족은 비용 문제가 아니라 진단 신뢰도 문제다."""

    def test_default_is_larger_than_shared_default(self):
        from core.llm_clients import DEFAULT_MAX_OUTPUT_TOKENS

        self.assertGreater(generator.DEFAULT_GENERATION_MAX_TOKENS,
                           DEFAULT_MAX_OUTPUT_TOKENS)

    def test_env_overrides_default(self):
        with patch.dict(os.environ, {"RAG_MAX_OUTPUT_TOKENS": "9000"}, clear=False):
            self.assertEqual(generator._generation_max_tokens(None), 9000)

    def test_config_wins_over_env(self):
        with patch.dict(os.environ, {"RAG_MAX_OUTPUT_TOKENS": "9000"}, clear=False):
            self.assertEqual(
                generator._generation_max_tokens({"generation_max_tokens": 512}), 512)

    def test_invalid_values_fall_back_to_default(self):
        for bad in ("", "abc", "0", "-1", None):
            with patch.dict(os.environ, {"RAG_MAX_OUTPUT_TOKENS": str(bad)}, clear=False):
                self.assertEqual(generator._generation_max_tokens(None),
                                 generator.DEFAULT_GENERATION_MAX_TOKENS, bad)

    def test_cap_reaches_the_transport(self):
        captured = {}

        def fake_chat(*a, **kw):
            captured.update(kw)
            return "답변"

        env = {"OPENROUTER_API_KEY": "sk-or-test", "RAG_MAX_OUTPUT_TOKENS": "6000"}
        with patch.dict(os.environ, env, clear=False), \
                patch.object(generator, "openai_chat", fake_chat):
            generator._openrouter_generate("sys", "user")
        self.assertEqual(captured["max_output_tokens"], 6000)


class ProviderAliasThreeWaySymmetryTest(unittest.TestCase):
    """Eval·RAG·Index 세 축이 같은 철자표를 본다.

    Eval/RAG 만 대칭을 맞춰두고 Index 가 원문 비교만 하던 시절, 같은 값을 세 env 에
    넣으면 두 곳은 OpenRouter 로 가고 Index 만 keyword 로 갈렸다."""

    def setUp(self):
        graph_index._warned_graph_providers.clear()

    def test_all_three_share_one_table(self):
        from core.llm_clients import PROVIDER_ALIASES

        self.assertIs(llm_provider._PROVIDER_ALIASES, PROVIDER_ALIASES)
        self.assertIs(generator._PROVIDER_ALIASES, PROVIDER_ALIASES)

    def test_index_accepts_openrouter_aliases(self):
        env = {"OPENROUTER_API_KEY": "or-dummy", "OPENAI_API_KEY": ""}
        for spelling in ("openrouter", "OpenRouter", "  open_router ",
                         "open-router", "openrouter.ai"):
            with self.subTest(spelling=spelling):
                with patch.dict(os.environ, dict(env, INDEX_LLM_PROVIDER=spelling),
                                clear=False):
                    target = graph_index._graph_llm_target({})
                self.assertIsNotNone(target, spelling)
                self.assertEqual(target[1], "https://openrouter.ai/api/v1")

    def test_index_warns_once_on_unknown_value(self):
        env = {"OPENROUTER_API_KEY": "or-dummy", "OPENAI_API_KEY": "",
               "INDEX_LLM_PROVIDER": "openroutr"}
        buf = io.StringIO()
        with patch.dict(os.environ, env, clear=False), redirect_stdout(buf):
            first = graph_index._graph_llm_target({})
            log = buf.getvalue()
            graph_index._graph_llm_target({})
        self.assertIsNone(first)                       # keyword 로 폴백
        self.assertIn("알 수 없는 INDEX_LLM_PROVIDER", log)
        self.assertEqual(buf.getvalue(), log)          # 값당 1회만

    def test_index_typo_with_valid_key_is_not_silent(self):
        # 키가 멀쩡한데 오타 하나로 LLM 추출이 꺼지는 게 가장 알아채기 어렵다.
        env = {"OPENAI_API_KEY": "sk-dummy", "INDEX_LLM_PROVIDER": "opnai"}
        buf = io.StringIO()
        with patch.dict(os.environ, env, clear=False), redirect_stdout(buf):
            self.assertIsNone(graph_index._graph_llm_target({}))
        self.assertIn("keyword", buf.getvalue())

if __name__ == "__main__":
    unittest.main()
