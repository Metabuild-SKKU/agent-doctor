from __future__ import annotations

import sys
import types
import unittest
from importlib.util import find_spec
from unittest.mock import patch

from agents.rag.retriever import RetrievalSettings


class _FakeFastAPI:
    def __init__(self, *args, **kwargs):
        pass

    def add_middleware(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        def decorator(fn):
            return fn

        return decorator

    def post(self, *args, **kwargs):
        def decorator(fn):
            return fn

        return decorator


if find_spec("fastapi") is None:
    fastapi_module = types.ModuleType("fastapi")
    fastapi_module.FastAPI = _FakeFastAPI
    fastapi_module.HTTPException = Exception
    cors_module = types.ModuleType("fastapi.middleware.cors")
    cors_module.CORSMiddleware = object
    sys.modules.setdefault("uvicorn", types.ModuleType("uvicorn"))
    sys.modules.setdefault("fastapi", fastapi_module)
    sys.modules.setdefault("fastapi.middleware", types.ModuleType("fastapi.middleware"))
    sys.modules.setdefault("fastapi.middleware.cors", cors_module)

from agents.serve import api  # noqa: E402


class ServeApiTests(unittest.TestCase):
    def _fake_retriever(self):
        return type(
            "FakeRetriever",
            (),
            {
                "client": object(),
                "settings": RetrievalSettings(
                    embedding_model="test-model",
                    embedding_dimension=2,
                    top_k=3,
                    use_hybrid=True,
                    use_reranker=True,
                    qdrant_url="https://qdrant.example",
                    qdrant_api_key="secret-token",
                ),
            },
        )()

    def test_health_does_not_expose_qdrant_secret_or_url(self):
        original_retriever = api._retriever
        original_chunks = api._chunks_raw
        try:
            api._chunks_raw = [{"chunk_id": "c1"}]
            api._retriever = self._fake_retriever()

            response = api.health()

            self.assertEqual(response["status"], "ok")
            settings = response["index_settings"]
            self.assertEqual(settings["embedding_model"], "test-model")
            self.assertNotIn("qdrant_api_key", settings)
            self.assertNotIn("qdrant_url", settings)
            self.assertNotIn("secret-token", repr(response))
        finally:
            api._retriever = original_retriever
            api._chunks_raw = original_chunks

    def test_answer_passes_context_compression_config_to_generator(self):
        original_retriever = api._retriever
        original_chunks = api._chunks_raw
        try:
            api._chunks_raw = [{"chunk_id": "c1"}]
            api._retriever = self._fake_retriever()
            with (
                patch.dict("os.environ", {"RAG_CONTEXT_COMPRESSION": "1"}),
                patch("agents.serve.api.answer_question", return_value={"answer": "ok"}) as answer_question,
            ):
                response = api.answer("재택근무 가능 일수는?")

            self.assertEqual(response, {"answer": "ok"})
            config = answer_question.call_args.kwargs["config"]
            self.assertEqual(config["context_compression"], "1")
            self.assertEqual(config["context.compression.enabled"], "1")
            self.assertTrue(config["use_reranker"])
        finally:
            api._retriever = original_retriever
            api._chunks_raw = original_chunks

    def test_answer_uses_context_compression_config_from_chunk_metadata(self):
        original_retriever = api._retriever
        original_chunks = api._chunks_raw
        try:
            api._chunks_raw = [
                {
                    "chunk_id": "c1",
                    "metadata": {
                        "context_compression": True,
                        "context_compression_max_contexts": 2,
                        "context_compression_min_contexts": 1,
                    },
                }
            ]
            api._retriever = self._fake_retriever()
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("agents.serve.api.answer_question", return_value={"answer": "ok"}) as answer_question,
            ):
                response = api.answer("?ы깮洹쇰Т 媛???쇱닔??")

            self.assertEqual(response, {"answer": "ok"})
            config = answer_question.call_args.kwargs["config"]
            self.assertTrue(config["context_compression"])
            self.assertTrue(config["context.compression.enabled"])
            self.assertEqual(config["context_compression_max_contexts"], 2)
            self.assertEqual(config["context_compression_min_contexts"], 1)
        finally:
            api._retriever = original_retriever
            api._chunks_raw = original_chunks


if __name__ == "__main__":
    unittest.main()
