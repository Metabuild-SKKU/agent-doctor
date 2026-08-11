# Eval 의 임베딩 경로 선택(API provider / 로컬 폴백 / 결측) 계약을 고정한다.
# 실제 API 도 실제 모델도 부르지 않는다.
from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.eval import llm_provider


def _env(**overrides):
    """관련 env 를 전부 비우고 필요한 것만 채운다 — 실행 환경의 .env 에 안 휘둘리게."""
    base = {
        "EVAL_LLM_PROVIDER": "",
        "OPENAI_API_KEY": "",
        "GEMINI_API_KEY": "",
        "OPENROUTER_API_KEY": "",
        "GITHUB_TOKEN": "",
    }
    base.update(overrides)
    return patch.dict("os.environ", base)


class _NoticeReset(unittest.TestCase):
    def setUp(self):
        # 안내는 실행당 1회라 전역 플래그를 쓴다. 테스트마다 되돌린다.
        llm_provider._embed_notified = False
        self.addCleanup(setattr, llm_provider, "_embed_notified", False)


class ApiAvailabilityTests(unittest.TestCase):
    def test_openrouter_uses_its_own_key(self):
        with _env(EVAL_LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="k"):
            self.assertTrue(llm_provider._api_embeddings_available())

    def test_openrouter_without_key_is_unavailable(self):
        with _env(EVAL_LLM_PROVIDER="openrouter"):
            self.assertFalse(llm_provider._api_embeddings_available())

    def test_gemini_uses_gemini_key(self):
        with _env(EVAL_LLM_PROVIDER="gemini", GEMINI_API_KEY="k"):
            self.assertTrue(llm_provider._api_embeddings_available())

    def test_github_falls_back_to_openai_key(self):
        # GitHub Models 는 임베딩 엔드포인트가 없어 OPENAI_API_KEY 를 본다.
        with _env(EVAL_LLM_PROVIDER="github", GITHUB_TOKEN="t"):
            self.assertFalse(llm_provider._api_embeddings_available())
        with _env(EVAL_LLM_PROVIDER="github", GITHUB_TOKEN="t", OPENAI_API_KEY="k"):
            self.assertTrue(llm_provider._api_embeddings_available())


class LocalAvailabilityTests(unittest.TestCase):
    def test_asks_index_about_the_local_path_specifically(self):
        """Index 의 기본 provider 는 openrouter 라 그냥 물으면 항상 '가용' 이 나온다.

        API 경로에는 해시 fallback 이라는 상태가 없어서다. provider='local' 을
        못 박지 않으면 로컬 모델이 못 뜨는 환경에서도 True 가 된다."""
        with patch("agents.index.qdrant_store.embedding_is_fallback") as is_fallback:
            is_fallback.return_value = True     # 로컬은 해시 fallback 상태
            self.assertFalse(llm_provider._local_embeddings_available())
        self.assertEqual(is_fallback.call_args.kwargs, {"provider": "local"})


class RoutingTests(_NoticeReset):
    def test_openrouter_provider_routes_to_openrouter_embed(self):
        with _env(EVAL_LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="k"), \
             patch.object(llm_provider, "openai_embed") as embed:
            embed.return_value = [[1.0]]
            llm_provider.embed_texts(["a"])

        texts, model = embed.call_args.args
        self.assertEqual(texts, ["a"])
        self.assertEqual(model, "baai/bge-m3")
        self.assertEqual(embed.call_args.kwargs["base_url"],
                         llm_provider.OPENROUTER_BASE_URL)
        self.assertEqual(embed.call_args.kwargs["api_key"], "k")

    def test_openrouter_model_override(self):
        with _env(EVAL_LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="k",
                  EVAL_EMBED_MODEL_OPENROUTER="other/embed"), \
             patch.object(llm_provider, "openai_embed") as embed:
            embed.return_value = [[1.0]]
            llm_provider.embed_texts(["a"])
        self.assertEqual(embed.call_args.args[1], "other/embed")

    def test_local_fallback_pins_local_provider(self):
        """'비용 0, 외부 호출 없음' 이라 찍어놓고 API 를 부르면 안 된다.

        Index 의 embed_batch 기본 provider 가 openrouter 라, 못 박지 않으면
        폴백 경로가 그대로 과금 호출이 된다."""
        with _env(EVAL_LLM_PROVIDER="openrouter"), \
             patch("agents.index.qdrant_store.embedding_is_fallback",
                   return_value=False), \
             patch("agents.index.qdrant_store.embed_batch") as embed_batch, \
             patch("builtins.print"):
            embed_batch.return_value = [[1.0]]
            llm_provider.embed_texts(["a"])

        self.assertEqual(embed_batch.call_args.kwargs.get("provider"), "local")

    def test_missing_everything_returns_empty(self):
        # 결측이면 response_relevancy 뿐 아니라 bad_gold_answer 라벨과
        # probe 자동 재생성까지 멈춘다 — 조용히 넘어가지 않고 안내를 남긴다.
        with _env(EVAL_LLM_PROVIDER="openrouter"), \
             patch("agents.index.qdrant_store.embedding_is_fallback",
                   return_value=True), \
             patch("builtins.print") as printed:
            self.assertEqual(llm_provider.embed_texts(["a"]), [])
        self.assertTrue(printed.called)

    def test_empty_input_makes_no_call(self):
        with patch.object(llm_provider, "openai_embed") as embed:
            self.assertEqual(llm_provider.embed_texts([]), [])
        embed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
