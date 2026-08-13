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
        llm_provider._embed_notified.clear()    # 메시지별 1회 알림(set) — 테스트 간 격리
        self.addCleanup(llm_provider._embed_notified.clear)


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


class EmbedProviderOptInTests(_NoticeReset):
    """EVAL_EMBED_PROVIDER - 채점 임베딩축을 심판축에서 분리하는 스위치.

    경로 판정은 embedding_route() 하나가 소유한다(llm_provider 의 같은 이름 독스트링).
    여기서 고정하는 계약은 셋이다: 미지정=심판 따름 / 명시=강제값 / 받을 수 없는
    값=설정 무시 + 귀착지를 소리내어 말함."""

    def _local_ok(self):
        return patch("agents.index.qdrant_store.embedding_is_fallback", return_value=False)

    def test_defaults_to_judge_provider(self):
        with _env(EVAL_LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="k"):
            self.assertEqual(llm_provider.embedding_route(), "openrouter")

    def test_explicit_opt_in_splits_the_axis(self):
        """심판은 anthropic 그대로 두고 임베딩만 OpenRouter 로 - 이 PR 의 목적."""
        with _env(EVAL_LLM_PROVIDER="anthropic", OPENROUTER_API_KEY="k",
                  EVAL_EMBED_PROVIDER="openrouter"):
            self.assertEqual(llm_provider.embedding_route(), "openrouter")
            self.assertEqual(llm_provider._provider(), "anthropic")

    def test_alias_is_normalized(self):
        """EVAL_LLM_PROVIDER 와 같은 철자표를 쓴다(PROVIDER_ALIASES)."""
        with _env(EVAL_LLM_PROVIDER="anthropic", OPENROUTER_API_KEY="k",
                  EVAL_EMBED_PROVIDER="Open-Router"):
            self.assertEqual(llm_provider.embedding_route(), "openrouter")

    def test_explicit_route_is_a_hard_value(self):
        """명시했는데 그 경로를 못 쓰면 다른 provider 로 새지 않고 결측이다.

        이 스위치가 생긴 사고(2026-08-11)의 핵심이 "로컬로 조용히 새서 GPU 경합" 이라,
        opt-in 한 사용자의 의도(그 경로만)를 지키는 것이 폴백보다 중요하다. 로컬이
        멀쩡히 떠 있어도 그쪽으로 가지 않는다는 게 이 테스트의 요점이다."""
        with _env(EVAL_LLM_PROVIDER="anthropic", EVAL_EMBED_PROVIDER="openrouter"), \
             self._local_ok(), \
             patch("agents.index.qdrant_store.embed_batch") as embed_batch, \
             patch("builtins.print") as printed:
            self.assertEqual(llm_provider.embedding_route(), "none")
            self.assertEqual(llm_provider.embed_texts(["a"]), [])

        embed_batch.assert_not_called()
        message = printed.call_args.args[0]
        # 사유는 키를 지목해야 한다 - openrouter 는 임베딩 엔드포인트가 **있다**.
        self.assertIn("EVAL_EMBED_PROVIDER=openrouter", message)
        self.assertIn("임베딩 키가 없습니다", message)
        self.assertNotIn("엔드포인트가 없", message)

    def test_provider_without_embeddings_is_not_accepted(self):
        """anthropic·github 는 심판으로는 되지만 임베딩 엔드포인트가 없다.

        받는 값이 아니므로 강제값이 아니라 "설정을 못 읽었다" 로 처리한다 - 그대로
        받아주면 아래 분기가 openai 로 떨어져 'anthropic 으로 임베딩한다'고 적어둔
        실행이 실제로는 OpenAI 에 과금 호출을 한다."""
        for bad in ("anthropic", "github"):
            with self.subTest(provider=bad):
                llm_provider._embed_notified.clear()
                with _env(EVAL_LLM_PROVIDER="openai", OPENAI_API_KEY="k",
                          EVAL_EMBED_PROVIDER=bad), \
                     patch("builtins.print") as printed:
                    self.assertEqual(llm_provider.embedding_route(), "openai")
                self.assertTrue(printed.called)
                self.assertIn("임베딩 엔드포인트가 없는 provider 라",
                              printed.call_args.args[0])

    def test_unknown_value_is_ignored_with_a_warning(self):
        """오타까지 결측으로 만들면 철자 하나가 진단을 통째로 끈다 - 강제값 정책은
        '받는 값을 적었는데 못 쓸 때' 의 이야기다. 대신 조용히 넘기지는 않는다."""
        with _env(EVAL_LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="k",
                  EVAL_EMBED_PROVIDER="opebrouter"), \
             patch("builtins.print") as printed:
            self.assertEqual(llm_provider.embedding_route(), "openrouter")
        self.assertTrue(printed.called)
        self.assertIn("지원하지 않는 값", printed.call_args.args[0])

    def test_warning_names_where_the_run_actually_went(self):
        """유효값 목록만 알려주면 "이번 실행은 어디로 갔나" 가 안 남는다.

        받을 수 없는 값은 폴백 사슬로 되돌아가므로 그 끝이 API provider 면 오타 한 번이
        과금 경로가 된다. 형제 축(qdrant_store)은 방향을 local 로 틀어 그걸 막는데,
        이 축은 방향 대신 '조용히' 를 없앤다. 정식 값이 된 local 과 달리 local 의 철자
        오타는 여전히 여기로 온다."""
        with _env(EVAL_LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="k",
                  EVAL_EMBED_PROVIDER="locl"), \
             patch("builtins.print") as printed:
            self.assertEqual(llm_provider.embedding_route(), "openrouter")

        message = printed.call_args.args[0]
        self.assertIn("'openrouter' 로 임베딩합니다", message)
        self.assertIn("과금", message)

    def test_warning_names_the_effective_route_not_the_judge(self):
        """심판이 anthropic·github 이면 실제 임베딩은 그 이름으로 나가지 않는다 -
        엔드포인트가 없어 openai 로 흡수된다. 귀착지를 심판 이름으로 찍으면 경고가
        거짓말을 한다(리뷰 지적: 'anthropic 로 임베딩합니다' 인데 실제는 OpenAI)."""
        with _env(EVAL_LLM_PROVIDER="anthropic", OPENAI_API_KEY="k",
                  EVAL_EMBED_PROVIDER="locl"), \
             patch("builtins.print") as printed:
            self.assertEqual(llm_provider.embedding_route(), "openai")

        message = printed.call_args.args[0]
        self.assertIn("'openai' 로 임베딩합니다", message)
        self.assertNotIn("'anthropic'", message)

    def test_local_is_a_first_class_value(self):
        """형제 축(INDEX_EMBED_PROVIDER)의 대표 값이다. 미지원으로 두면 '로컬로 돌리려던'
        설정이 심판축(과금 경로)으로 떨어진다 - 실측: local -> openrouter bge-m3 호출."""
        with _env(EVAL_LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="k",
                  EVAL_EMBED_PROVIDER="local"), self._local_ok():
            # 키가 있어도 API 를 부르지 않는다 - 사용자가 로컬을 못 박았다.
            self.assertEqual(llm_provider.embedding_route(), "local")

    def test_local_actually_reaches_the_local_path(self):
        with _env(EVAL_LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="k",
                  EVAL_EMBED_PROVIDER="local"), \
             patch.object(llm_provider, "openai_embed") as api_embed, \
             self._local_ok(), \
             patch("agents.index.qdrant_store.embed_batch") as embed_batch, \
             patch("builtins.print"):
            embed_batch.return_value = [[1.0]]
            self.assertEqual(llm_provider.embed_texts(["a"]), [[1.0]])

        api_embed.assert_not_called()
        self.assertEqual(embed_batch.call_args.kwargs.get("provider"), "local")

    def test_notice_names_the_embed_axis_not_the_judge(self):
        """축을 나눠 놓고 로그가 심판축 이름만 말하면 엉뚱한 변수를 들여다보게 된다."""
        with _env(EVAL_LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="k",
                  EVAL_EMBED_PROVIDER="local"), \
             self._local_ok(), \
             patch("agents.index.qdrant_store.embed_batch", return_value=[[1.0]]), \
             patch("builtins.print") as printed:
            llm_provider.embed_texts(["a"])

        message = printed.call_args.args[0]
        self.assertIn("EVAL_EMBED_PROVIDER", message)
        self.assertNotIn("EVAL_LLM_PROVIDER", message)

    def test_local_fallback_notice_blames_the_key_when_there_is_an_endpoint(self):
        """심판축을 따라가다 로컬로 내려온 사유는 둘이다 - 엔드포인트가 없거나, 키가
        없거나. 뭉쳐서 전자로 적으면 openai·gemini·openrouter 심판 실행에 거짓 사유가
        찍히고, 읽는 사람이 키가 아니라 provider 선택을 의심한다."""
        with _env(EVAL_LLM_PROVIDER="openai"), \
             self._local_ok(), \
             patch("agents.index.qdrant_store.embed_batch", return_value=[[1.0]]), \
             patch("builtins.print") as printed:
            llm_provider.embed_texts(["a"])

        message = printed.call_args.args[0]
        self.assertIn("EVAL_LLM_PROVIDER=openai 의 임베딩 키가 없어", message)
        self.assertNotIn("엔드포인트가 없", message)

    def test_local_fallback_notice_still_blames_the_endpoint_when_that_is_true(self):
        """심판이 anthropic 이면 '엔드포인트 없음' 이 참이다."""
        with _env(EVAL_LLM_PROVIDER="anthropic"), \
             self._local_ok(), \
             patch("agents.index.qdrant_store.embed_batch", return_value=[[1.0]]), \
             patch("builtins.print") as printed:
            llm_provider.embed_texts(["a"])

        self.assertIn("EVAL_LLM_PROVIDER=anthropic 는 임베딩 엔드포인트가 없",
                      printed.call_args.args[0])

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

    def test_missing_notice_does_not_blame_keys_when_local_is_pinned(self):
        """local 을 못 박은 실행에 "키도 없다" 는 성립하지 않는다 — 이 경로는 키를 아예
        안 본다(_api_embeddings_available). 고칠 것이 키가 아니라 로컬 모델이다."""
        with _env(EVAL_LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="k",
                  EVAL_EMBED_PROVIDER="local"), \
             patch("agents.index.qdrant_store.embedding_is_fallback",
                   return_value=True), \
             patch("builtins.print") as printed:
            self.assertEqual(llm_provider.embed_texts(["a"]), [])

        message = printed.call_args.args[0]
        self.assertIn("로컬 임베딩 모델을 쓸 수 없습니다", message)
        self.assertNotIn("OPENAI_API_KEY", message)

    def test_empty_input_makes_no_call(self):
        with patch.object(llm_provider, "openai_embed") as embed:
            self.assertEqual(llm_provider.embed_texts([]), [])
        embed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
