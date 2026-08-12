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
        # 이 축을 결정하는 변수라 반드시 비운다 - 안 비우면 기능을 실제로 켠 머신에서만
        # 5건이 깨진다(그중에 "키만으로는 안 바뀐다" 핀이 있어 설계 주장이 거짓이 된다).
        "EVAL_EMBED_PROVIDER": "",
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

    def test_openrouter_key_does_not_hijack_other_providers(self):
        """임베딩 경로는 기본적으로 심판 provider 를 따라간다.

        OPENROUTER_API_KEY 가 있다는 이유로 anthropic/github 실행을 OpenRouter 임베딩으로
        보내면, 심판 설정을 하나도 안 바꾼 사람의 실행이 OpenRouter 가용성에 새로 묶인다
        (예전엔 로컬로 오프라인 계산하던 조합이다). 전환은 EVAL_EMBED_PROVIDER 를 적은
        사람만 받는다 — 여기서는 '키만으로는 안 바뀐다' 를 핀으로 잡는다."""
        for provider in ("anthropic", "github"):
            with self.subTest(provider=provider), \
                 _env(EVAL_LLM_PROVIDER=provider, OPENROUTER_API_KEY="k"):
                self.assertFalse(llm_provider._api_embeddings_available())


class EmbedProviderOptInTests(unittest.TestCase):
    def test_defaults_to_judge_provider(self):
        with _env(EVAL_LLM_PROVIDER="anthropic"):
            self.assertEqual(llm_provider._embed_provider(), "anthropic")

    def test_explicit_opt_in_splits_the_axis(self):
        """심판은 anthropic 그대로 두고 임베딩만 OpenRouter 로 — 이 PR 의 목적."""
        with _env(EVAL_LLM_PROVIDER="anthropic", OPENROUTER_API_KEY="k",
                  EVAL_EMBED_PROVIDER="openrouter"):
            self.assertEqual(llm_provider._embed_provider(), "openrouter")
            self.assertEqual(llm_provider._provider(), "anthropic")
            self.assertTrue(llm_provider._api_embeddings_available())

    def test_alias_is_normalized(self):
        """EVAL_LLM_PROVIDER 와 같은 철자표를 쓴다(PROVIDER_ALIASES)."""
        with _env(EVAL_LLM_PROVIDER="anthropic", OPENROUTER_API_KEY="k",
                  EVAL_EMBED_PROVIDER="Open-Router"):
            self.assertEqual(llm_provider._embed_provider(), "openrouter")

    def _forget_warning(self, raw: str) -> None:
        """경고 dedup 전역을 되돌린다. addCleanup 이라 단언이 실패해도 실행된다 —
        assert 뒤에 두면 실패한 실행이 전역을 오염시킨 채 남는다."""
        self.addCleanup(llm_provider._warned_providers.discard, f"embed:{raw}")

    def test_provider_without_embeddings_is_rejected(self):
        """anthropic·github 를 명시하면 아래 분기가 openai 로 떨어져, 'anthropic 으로
        임베딩한다'고 적어둔 실행이 실제로는 OpenAI 에 과금 호출을 한다.
        심판축의 폴백 사슬과 달리 여기서는 사용자가 직접 고른 값이라 조용히 바꾸면 안 된다."""
        for bad in ("anthropic", "github"):
            self._forget_warning(bad)
            with self.subTest(provider=bad), \
                 _env(EVAL_LLM_PROVIDER="openai", OPENAI_API_KEY="k",
                      EVAL_EMBED_PROVIDER=bad), \
                 patch("builtins.print") as printed:
                self.assertEqual(llm_provider._embed_provider(), "openai")
                self.assertTrue(printed.called)
                self.assertIn("임베딩 엔드포인트가 없습니다", printed.call_args.args[0])

    def test_unknown_value_falls_back_to_judge_with_warning(self):
        """조용히 openai 로 떨어뜨리지 않는다 — 임베딩 모델이 바뀌면 코사인 분포가
        달라져 실행 간 response_relevancy 비교가 깨진다."""
        self._forget_warning("opebrouter")
        with _env(EVAL_LLM_PROVIDER="anthropic", EVAL_EMBED_PROVIDER="opebrouter"), \
             patch("builtins.print") as printed:
            self.assertEqual(llm_provider._embed_provider(), "anthropic")
        self.assertTrue(printed.called)
        self.assertIn("지원하지 않는 값", printed.call_args.args[0])

    def test_local_is_a_first_class_value(self):
        """형제 축(INDEX_EMBED_PROVIDER)의 대표 값이다. 미지원으로 두면 '로컬로 돌리려던'
        오타가 심판축(과금 경로)으로 떨어진다 - 실측: local -> openrouter bge-m3 호출."""
        self._forget_warning("local")
        with _env(EVAL_LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="k",
                  EVAL_EMBED_PROVIDER="local"):
            self.assertEqual(llm_provider._embed_provider(), "local")
            # 키가 있어도 API 를 부르지 않는다 - 사용자가 로컬을 못 박았다.
            self.assertFalse(llm_provider._api_embeddings_available())

    def test_local_actually_reaches_the_local_path(self):
        with _env(EVAL_LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="k",
                  EVAL_EMBED_PROVIDER="local"), \
             patch.object(llm_provider, "openai_embed") as api_embed, \
             patch("agents.index.qdrant_store.embedding_is_fallback", return_value=False), \
             patch("agents.index.qdrant_store.embed_batch") as embed_batch, \
             patch("builtins.print"):
            embed_batch.return_value = [[1.0]]
            llm_provider._embed_notified = False
            self.assertEqual(llm_provider.embed_texts(["a"]), [[1.0]])

        api_embed.assert_not_called()
        self.assertEqual(embed_batch.call_args.kwargs.get("provider"), "local")

    def test_notice_names_the_embed_axis_not_the_judge(self):
        """축을 나눠 놓고 로그가 심판축 이름만 말하면 엉뚱한 변수를 들여다보게 된다."""
        with _env(EVAL_LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="k",
                  EVAL_EMBED_PROVIDER="local"), \
             patch("agents.index.qdrant_store.embedding_is_fallback", return_value=False), \
             patch("agents.index.qdrant_store.embed_batch", return_value=[[1.0]]), \
             patch("builtins.print") as printed:
            llm_provider._embed_notified = False
            llm_provider.embed_texts(["a"])

        message = printed.call_args.args[0]
        self.assertIn("EVAL_EMBED_PROVIDER", message)
        self.assertNotIn("EVAL_LLM_PROVIDER", message)

    def test_explicit_route_reaches_the_transport(self):
        with _env(EVAL_LLM_PROVIDER="anthropic", OPENROUTER_API_KEY="k",
                  EVAL_EMBED_PROVIDER="openrouter"), \
             patch.object(llm_provider, "openai_embed") as embed:
            embed.return_value = [[1.0]]
            llm_provider.embed_texts(["a"])

        self.assertEqual(embed.call_args.args[1], "baai/bge-m3")
        self.assertEqual(embed.call_args.kwargs["base_url"],
                         llm_provider.OPENROUTER_BASE_URL)


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

    def test_api_failure_falls_back_to_local(self):
        """임베딩 API 장애가 진단 기능까지 끌고 내려가면 안 된다.

        결측이 되면 response_relevancy 뿐 아니라 bad_gold_answer 라벨과 probe 자동
        재생성까지 멈춘다(embed_texts 독스트링). 로컬 모델이 뜨는 환경이면 그쪽으로 계속한다."""
        with _env(EVAL_LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="k"), \
             patch.object(llm_provider, "openai_embed",
                          side_effect=RuntimeError("503")), \
             patch("agents.index.qdrant_store.embedding_is_fallback",
                   return_value=False), \
             patch("agents.index.qdrant_store.embed_batch") as embed_batch, \
             patch("builtins.print"):
            embed_batch.return_value = [[1.0]]
            self.assertEqual(llm_provider.embed_texts(["a"]), [[1.0]])

        self.assertEqual(embed_batch.call_args.kwargs.get("provider"), "local")

    def test_api_failure_propagates_when_local_unusable(self):
        """로컬도 못 쓰면 조용히 넘기지 않는다 — 해시 벡터로 메우면 무의미한 코사인이
        정상 점수처럼 리포트에 박힌다(_local_embeddings_available 참고)."""
        with _env(EVAL_LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="k"), \
             patch.object(llm_provider, "openai_embed",
                          side_effect=RuntimeError("503")), \
             patch("agents.index.qdrant_store.embedding_is_fallback",
                   return_value=True), \
             patch("builtins.print"):
            with self.assertRaises(RuntimeError):
                llm_provider.embed_texts(["a"])

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
