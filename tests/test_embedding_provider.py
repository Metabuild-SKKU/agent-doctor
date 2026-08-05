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

    def test_env_supplies_default(self):
        with patch.dict("os.environ", {"INDEX_EMBED_PROVIDER": "local"}):
            self.assertEqual(store.resolve_embedding_provider(), "local")

    def test_argument_beats_env(self):
        with patch.dict("os.environ", {"INDEX_EMBED_PROVIDER": "openrouter"}):
            self.assertEqual(store.resolve_embedding_provider("local"), "local")

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
            {"INDEX_EMBED_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "k"},
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        store._models.clear()
        self.addCleanup(store._models.clear)


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

    def test_query_embed_uses_same_provider(self):
        with patch.object(store, "openai_embed") as embed:
            embed.side_effect = lambda texts, model, **kw: [[7.0] for _ in texts]
            self.assertEqual(store.embed("q", model_name="BAAI/bge-m3"), [7.0])

    def test_missing_key_raises_loudly(self):
        # 조용히 로컬로 새면 "API 로 돌리는 중" 이라 믿은 색인이 2 chunks/sec 가 된다.
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}):
            with self.assertRaises(RuntimeError) as ctx:
                store.embed_batch(["a"], model_name="BAAI/bge-m3")
        self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))

    def test_is_fallback_is_false_for_api(self):
        # API 경로에는 해시 fallback 이라는 상태가 없다(성공 아니면 예외).
        self.assertFalse(store.embedding_is_fallback("BAAI/bge-m3"))

    def test_empty_input_makes_no_call(self):
        with patch.object(store, "openai_embed") as embed:
            self.assertEqual(store.embed_batch([], model_name="BAAI/bge-m3"), [])
        embed.assert_not_called()


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

        실측(tools/bench_embedding.py)에서 429 는 동시 1 에서도 요청의 19% 였고,
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


if __name__ == "__main__":
    unittest.main()
