from __future__ import annotations

import sys
import types
import unittest
from importlib.util import find_spec

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
    def test_health_does_not_expose_qdrant_secret_or_url(self):
        original_retriever = api._retriever
        original_chunks = api._chunks_raw
        try:
            api._chunks_raw = [{"chunk_id": "c1"}]
            api._retriever = type(
                "FakeRetriever",
                (),
                {
                    "client": object(),
                    "settings": RetrievalSettings(
                        embedding_model="test-model",
                        embedding_dimension=2,
                        top_k=3,
                        qdrant_url="https://qdrant.example",
                        qdrant_api_key="secret-token",
                    ),
                },
            )()

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


if __name__ == "__main__":
    unittest.main()
