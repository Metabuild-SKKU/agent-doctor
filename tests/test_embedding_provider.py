# 임베딩 provider(openrouter/local) 선택 계층의 계약을 고정한다.
# 실제 API 는 부르지 않는다 — core.llm_clients.openai_embed 를 대역으로 바꾼다.
from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.index import qdrant_store as store


class ResolveProviderTests(unittest.TestCase):
    def test_default_is_openrouter(self):
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("INDEX_EMBED_PROVIDER", None)
            self.assertEqual(store.resolve_embedding_provider(), "openrouter")

    def test_alias_is_normalized(self):
        # core.llm_clients.PROVIDER_ALIASES 를 그대로 쓴다 — 같은 값을
        # EVAL_/RAG_/INDEX_ 어디에 넣어도 같게 해석돼야 한다.
        self.assertEqual(store.resolve_embedding_provider("open-router"), "openrouter")

    def test_unknown_value_does_not_become_openrouter(self):
        # 오타가 "조용히 API 과금" 으로 끝나면 안 된다. 로컬로 처리된다.
        self.assertEqual(store.resolve_embedding_provider("openrouterr"), "openrouterr")


class _ApiRouted(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(
            "os.environ",
            {
                "INDEX_EMBED_PROVIDER": "openrouter",
                "OPENROUTER_API_KEY": "k",
                # 질의 축은 색인과 분리돼 기본이 local 이다. 이 클래스는 색인 경로를
                # 다루므로 질의도 API 로 못 박아 실제 모델 로드가 새지 않게 한다.
                "INDEX_QUERY_EMBED_PROVIDER": "openrouter",
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        store._models.clear()
        # 안내 플래그는 경로별 집합이다(전역 bool 하나였을 때는 먼저 찍힌 경로가
        # 나머지를 삼켰다). 비우지 않으면 앞 테스트의 안내가 남아 0건으로 보인다.
        store._routes_notified.clear()
        self.addCleanup(store._models.clear)
        self.addCleanup(store._routes_notified.clear)


class OpenRouterRoutingTests(_ApiRouted):
    def test_embed_batch_uses_api_and_skips_local_model(self):
        with patch.object(store, "openai_embed") as embed:
            embed.side_effect = lambda texts, model, **kw: [[float(len(t))] for t in texts]
            vectors = store.embed_batch(["aa", "bbb"], model_name="BAAI/bge-m3")

        self.assertEqual(vectors, [[2.0], [3.0]])
        self.assertEqual(store._models, {})   # 로컬 모델은 안 올라온다
        _texts, model = embed.call_args.args
        # 로컬은 "BAAI/bge-m3", OpenRouter 는 소문자 철자를 쓴다.
        self.assertEqual(model, "baai/bge-m3")

    def test_model_name_override(self):
        with patch.dict("os.environ", {"INDEX_EMBED_MODEL_OPENROUTER": "other/model"}), \
             patch.object(store, "openai_embed") as embed:
            embed.side_effect = lambda texts, model, **kw: [[1.0] for _ in texts]
            store.embed_batch(["a"], model_name="BAAI/bge-m3")
        self.assertEqual(embed.call_args.args[1], "other/model")

    def test_query_embed_follows_query_axis(self):
        # 질의 축을 openrouter 로 명시하면 질의도 API 를 탄다(setUp 에서 지정).
        with patch.object(store, "openai_embed") as embed:
            embed.side_effect = lambda texts, model, **kw: [[7.0] for _ in texts]
            self.assertEqual(store.embed("q", model_name="BAAI/bge-m3"), [7.0])
        embed.assert_called_once()

    def test_missing_key_raises_loudly(self):
        # 조용히 로컬로 새면 "API 로 돌리는 중" 이라 믿은 색인이 2 chunks/sec 가 된다.
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}):
            with self.assertRaises(RuntimeError) as ctx:
                store.embed_batch(["a"], model_name="BAAI/bge-m3")
        self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))

    def test_is_fallback_is_false_for_api(self):
        """API 경로에는 해시 fallback 이라는 상태가 없다(성공 아니면 예외).

        로컬 경로를 막아야 이 단언이 의미를 갖는다. 안 막으면 API 분기를 통째로
        지워도 통과한다 — _get_embedding_model 로 흘러 실제 BGE-M3(2.2GB)가 뜨고,
        모델이 있으니 is None 이 False 라 같은 값이 나오기 때문이다(실측 44초).
        통과 이유가 "API 경로엔 fallback 이 없어서" 가 아니라 "로컬 모델이 마침
        있어서" 가 되어, 로컬 모델이 없는 CI 에서만 우연히 잡히는 테스트가 된다.

        이 분기가 중요한 이유는 소비처다 — agents/index/agent.py 가 이 값으로
        재임베딩 여부를 정한다. 잘못 True 가 되면 API 로 잘 색인된 청크를 매번
        다시 임베딩한다(과금 반복)."""
        with patch.object(store, "_get_embedding_model", return_value=None):
            self.assertFalse(store.embedding_is_fallback("BAAI/bge-m3"))


    def test_query_axis_inherits_index_axis_when_unset(self):
        """색인 에러가 시킨 대로 고쳤는데 질의에서 또 막히면 안 된다.

        색인 실패 메시지는 INDEX_EMBED_PROVIDER=local 을 안내한다. 질의 축이 그걸
        상속하지 않으면, 그대로 따른 사용자가 곧바로 질의 preflight 에서 두 번째
        에러를 만난다. 오프라인·무키 환경이 정확히 이 경로를 밟는다."""
        with patch.dict("os.environ", {"INDEX_EMBED_PROVIDER": "local",
                                       "INDEX_QUERY_EMBED_PROVIDER": ""}):
            self.assertEqual(store.resolve_embedding_provider(), "local")
            self.assertEqual(store.resolve_query_embedding_provider(), "local")
            # 키가 없어도 질의가 막히지 않아야 한다.
            self.assertIsNone(store.query_embedding_config_error())

    def test_explicit_query_axis_still_wins_over_index(self):
        """상속은 기본값 폴백일 뿐 - 명시하면 두 축이 갈라진다(설계 유지)."""
        with patch.dict("os.environ", {"INDEX_EMBED_PROVIDER": "local",
                                       "INDEX_QUERY_EMBED_PROVIDER": "openrouter"}):
            self.assertEqual(store.resolve_embedding_provider(), "local")
            self.assertEqual(store.resolve_query_embedding_provider(), "openrouter")

    def test_env_default_matches_cli_default(self):
        """CLI 와 env 의 기본 규칙이 같아야 한다.

        core/embedding_cli.py 는 --query-embed 미지정 시 --embed 를 따른다
        (query_target = args.query_embed or target). env 만 다르게 두면 같은 설정을
        어디에 쓰느냐로 결과가 갈린다."""
        import argparse

        from core.embedding_cli import add_embedding_args, apply_embedding_args

        parser = argparse.ArgumentParser()
        add_embedding_args(parser)
        with patch.dict("os.environ", {"INDEX_EMBED_PROVIDER": "",
                                       "INDEX_QUERY_EMBED_PROVIDER": ""}):
            apply_embedding_args(parser.parse_args(["--embed", "cpu"]))
            cli_query = store.resolve_query_embedding_provider()
        with patch.dict("os.environ", {"INDEX_EMBED_PROVIDER": "local",
                                       "INDEX_QUERY_EMBED_PROVIDER": ""}):
            env_query = store.resolve_query_embedding_provider()
        self.assertEqual(cli_query, env_query)

    def test_empty_input_makes_no_call(self):
        with patch.object(store, "openai_embed") as embed:
            self.assertEqual(store.embed_batch([], model_name="BAAI/bge-m3"), [])
        embed.assert_not_called()


class RouteNoticeTests(_ApiRouted):
    """provider 는 env 로 정해져 실행 기록만 봐선 어디서 계산됐는지 알 수 없다.
    비용과 속도가 100배 넘게 갈리는 축이라 실행당 한 번은 남겨야 한다."""

    def test_api_route_is_announced_once(self):
        with patch.object(store, "openai_embed") as embed, \
             patch("builtins.print") as printed:
            embed.side_effect = lambda texts, model, **kw: [[1.0] for _ in texts]
            store.embed_batch(["a"], model_name="BAAI/bge-m3")
            store.embed_batch(["b"], model_name="BAAI/bge-m3")

        notices = [c.args[0] for c in printed.call_args_list if "임베딩" in str(c.args[0])]
        self.assertEqual(len(notices), 1)          # 청크마다 찍으면 로그를 덮는다
        self.assertIn("OpenRouter", notices[0])
        self.assertIn("baai/bge-m3", notices[0])


class OrderAndRetryTests(_ApiRouted):
    def test_order_preserved_across_concurrent_batches(self):
        """호출부가 청크와 zip 으로 짝짓는다 — 순서가 섞이면 벡터가 엉뚱한 청크에 붙는다."""
        import time

        def _slow(texts, model, **kw):
            # 먼저 보낸 배치가 늦게 끝나도록 뒤집어 놓는다.
            time.sleep(0.05 if texts[0].startswith("0") else 0.0)
            return [[float(t)] for t in texts]

        texts = [str(i) for i in range(8)]
        with patch.dict("os.environ", {"INDEX_EMBED_API_BATCH": "2",
                                       "INDEX_EMBED_CONCURRENCY": "4"}), \
             patch.object(store, "openai_embed", side_effect=_slow):
            vectors = store.embed_batch(texts, model_name="BAAI/bge-m3")

        self.assertEqual(vectors, [[float(i)] for i in range(8)])

    def test_429_is_retried_not_dropped(self):
        """재시도 없이 삼키면 그 청크의 벡터가 조용히 빠진 컬렉션이 만들어진다.

        실측(AGENTS.md "임베딩 provider" 절)에서 429 는 동시 1 에서도 요청의 19% 였고,
        재시도가 없던 벤치는 1,000청크 중 170개 넘게 잃었다."""
        import openai

        calls = {"n": 0}

        def _flaky(texts, model, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise openai.RateLimitError(
                    "429", response=_FakeResponse(), body=None
                )
            return [[1.0] for _ in texts]

        with patch.dict("os.environ", {"EVAL_LLM_RETRY_WAIT": "0",
                                       "EVAL_LLM_MAX_RETRIES": "2"}), \
             patch.object(store, "openai_embed", side_effect=_flaky), \
             patch("builtins.print"):
            vectors = store.embed_batch(["a"], model_name="BAAI/bge-m3")

        self.assertEqual(vectors, [[1.0]])
        self.assertEqual(calls["n"], 2)

    def test_exhausted_retry_propagates(self):
        # 재시도를 소진하면 예외로 올려야 한다. 해시 fallback 으로 떨어지면
        # 어느 벡터가 쓰레기인지 구분할 수 없어 전체 재색인이 필요해진다.
        import openai

        def _always_429(texts, model, **kw):
            raise openai.RateLimitError("429", response=_FakeResponse(), body=None)

        with patch.dict("os.environ", {"EVAL_LLM_RETRY_WAIT": "0",
                                       "EVAL_LLM_MAX_RETRIES": "1"}), \
             patch.object(store, "openai_embed", side_effect=_always_429), \
             patch("builtins.print"):
            with self.assertRaises(openai.RateLimitError):
                store.embed_batch(["a"], model_name="BAAI/bge-m3")


class _FakeResponse:
    """openai.RateLimitError 생성에 필요한 최소 응답 객체."""

    status_code = 429
    headers: dict[str, str] = {}
    request = None


class QueryAxisTests(unittest.TestCase):
    """질의 임베딩은 색인과 별개 축이다(INDEX_QUERY_EMBED_PROVIDER)."""

    def setUp(self):
        store._models.clear()
        store._routes_notified.clear()
        self.addCleanup(store._models.clear)
        self.addCleanup(store._routes_notified.clear)

    def test_default_is_openrouter(self):
        # 두 축을 모두 비워야 코드 기본값이 드러난다 — 질의 축이 미지정이면 색인 축을
        # 상속하므로, 색인만 설정돼 있으면 그 값이 나온다(conftest 가 local 로 고정).
        with patch.dict("os.environ", {"INDEX_QUERY_EMBED_PROVIDER": "",
                                       "INDEX_EMBED_PROVIDER": ""}):
            self.assertEqual(store.resolve_query_embedding_provider(), "openrouter")

    def test_can_be_pinned_local_independently_of_index(self):
        """축을 나눈 값어치는 여기다 — 색인은 API 로 두고 질의만 로컬로 내릴 수 있다.

        429 재시도가 단건 질의 지연으로 그대로 나타나면 이 축만 내리면 된다.
        코사인 0.99997 이라 섞어도 순위가 흔들리지 않는다."""
        with patch.dict("os.environ", {"INDEX_EMBED_PROVIDER": "openrouter",
                                       "INDEX_QUERY_EMBED_PROVIDER": "local"}):
            self.assertEqual(store.resolve_embedding_provider(), "openrouter")
            self.assertEqual(store.resolve_query_embedding_provider(), "local")

    def test_local_pin_does_not_call_api(self):
        with patch.dict("os.environ", {"INDEX_EMBED_PROVIDER": "openrouter",
                                       "OPENROUTER_API_KEY": "k",
                                       "INDEX_QUERY_EMBED_PROVIDER": "local"}),              patch.object(store, "openai_embed") as embed,              patch.object(store, "_load_embedding_model",
                          return_value=(None, "cpu")),              patch("builtins.print"):
            vector = store.embed("q", model_name="BAAI/bge-m3")

        embed.assert_not_called()
        self.assertEqual(len(vector), store.VECTOR_DIM)   # 해시 fallback

    def test_preflight_flags_missing_key(self):
        """설정 오류는 keyword 폴백으로 흡수하면 안 된다.

        키를 안 넣은 실행이 영구히 keyword 검색으로 도는데 증상은 "검색 품질이 좀
        나쁘다" 로만 나타나 원인을 찾을 수 없다."""
        with patch.dict("os.environ", {"INDEX_QUERY_EMBED_PROVIDER": "openrouter",
                                       "OPENROUTER_API_KEY": ""}):
            reason = store.query_embedding_config_error()
        self.assertIsNotNone(reason)
        self.assertIn("OPENROUTER_API_KEY", reason)

    def test_preflight_silent_when_configured(self):
        with patch.dict("os.environ", {"INDEX_QUERY_EMBED_PROVIDER": "openrouter",
                                       "OPENROUTER_API_KEY": "k"}):
            self.assertIsNone(store.query_embedding_config_error())

    def test_preflight_silent_for_local(self):
        # 로컬 경로는 키가 필요 없다.
        with patch.dict("os.environ", {"INDEX_QUERY_EMBED_PROVIDER": "local",
                                       "OPENROUTER_API_KEY": ""}):
            self.assertIsNone(store.query_embedding_config_error())


class ResponseIntegrityTests(_ApiRouted):
    def test_short_response_raises(self):
        """호출부가 zip(청크, 벡터) 로 짝짓는다 — 짧게 오면 뒤쪽 청크가 조용히 사라진다.

        벤치에서 429 를 삼켰을 때 1,000청크 중 170개가 없어진 것과 같은 실패 모양이고,
        그때처럼 '성공한 색인' 으로 보이는 게 가장 나쁘다."""
        with patch.object(store, "openai_embed") as embed:
            embed.side_effect = lambda texts, model, **kw: [[1.0]]   # 항상 1건만
            with self.assertRaises(RuntimeError) as ctx:
                store.embed_batch(["a", "b", "c"], model_name="BAAI/bge-m3")
        self.assertIn("개수", str(ctx.exception))

    def test_long_response_also_raises(self):
        with patch.object(store, "openai_embed") as embed:
            embed.side_effect = lambda texts, model, **kw: [[1.0]] * (len(texts) + 1)
            with self.assertRaises(RuntimeError):
                store.embed_batch(["a"], model_name="BAAI/bge-m3")

    def test_zero_api_batch_does_not_crash(self):
        # 설정 오타가 ValueError: range() arg 3 must not be zero 로 끝나면 안 된다.
        with patch.dict("os.environ", {"INDEX_EMBED_API_BATCH": "0"}),              patch.object(store, "openai_embed") as embed:
            embed.side_effect = lambda texts, model, **kw: [[1.0] for _ in texts]
            self.assertEqual(len(store.embed_batch(["a", "b"],
                                                   model_name="BAAI/bge-m3")), 2)


class RouteNoticePerRouteTests(unittest.TestCase):
    def setUp(self):
        store._routes_notified.clear()
        self.addCleanup(store._routes_notified.clear)

    def test_fallback_notice_is_not_swallowed_by_earlier_route(self):
        """플래그가 하나면 먼저 찍힌 경로가 나머지를 삼킨다.

        API 로 시작한 실행이 도중에 해시 fallback 으로 열화되는 게 정확히 그 경우고,
        그때 안내가 사라지는 건 '어디서 계산됐는지 기록한다' 는 목적과 반대다."""
        store._notify_route_once("openrouter", "[Index] A")
        with patch("builtins.print") as printed:
            store._notify_route_once("hash_fallback", "[Index] B")
            store._notify_route_once("hash_fallback", "[Index] B")

        self.assertEqual([c.args[0] for c in printed.call_args_list], ["[Index] B"])


if __name__ == "__main__":
    unittest.main()
