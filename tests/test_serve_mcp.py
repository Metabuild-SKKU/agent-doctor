from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.serve import mcp_server


class _Response:
    def __init__(self, payload: dict, *, status_ok: bool = True):
        self.payload = payload
        self.status_ok = status_ok

    def raise_for_status(self) -> None:
        if not self.status_ok:
            raise RuntimeError("HTTP error")

    def json(self) -> dict:
        return self.payload


class ServeMcpTests(unittest.TestCase):
    def test_health_check_formats_public_status(self):
        payload = {
            "status": "ok",
            "chunks": 7,
            "qdrant": True,
            "fingerprint": "abc123",
            "index_settings": {
                "top_k": 5,
                "use_reranker": True,
                "embedding_model": "bge-m3",
                "qdrant_api_key": "secret",
            },
        }

        with (
            patch.object(mcp_server, "_ensure_api_running", return_value=True),
            patch.object(mcp_server.requests, "get", return_value=_Response(payload)) as get,
        ):
            result = mcp_server.health_check()

        self.assertIn("상태: ok", result)
        self.assertIn("청크 수: 7", result)
        self.assertIn("top_k=5", result)
        self.assertIn("use_reranker=True", result)
        self.assertNotIn("secret", result)
        get.assert_called_once_with(f"{mcp_server.API_URL}/health", params={}, timeout=5)

    def test_search_docs_delegates_query_and_top_k_to_serve_api(self):
        payload = {
            "search_mode": "keyword",
            "fallback_used": True,
            "results": [
                {
                    "doc_id": "policy",
                    "chunk_id": "policy#1",
                    "text": "재택근무는 주 2회까지 가능하다.",
                    "score": 0.81,
                    "metadata": {"title": "근무 규정"},
                }
            ],
        }

        with (
            patch.object(mcp_server, "_ensure_api_running", return_value=True),
            patch.object(mcp_server.requests, "get", return_value=_Response(payload)) as get,
        ):
            result = mcp_server.search_docs("재택근무 가능 일수", top_k=2)

        self.assertIn("검색 결과 1개", result)
        self.assertIn("mode=keyword", result)
        self.assertIn("keyword fallback", result)
        self.assertIn("근무 규정", result)
        self.assertIn("score=0.810", result)
        get.assert_called_once_with(
            f"{mcp_server.API_URL}/search",
            params={"query": "재택근무 가능 일수", "top_k": 2},
            timeout=15,
        )

    def test_ask_docs_formats_answer_with_citations(self):
        payload = {
            "answer": "재택근무는 주 2회까지 가능합니다.",
            "generation_mode": "extractive",
            "citations": [
                {
                    "title": "근무 규정",
                    "doc_id": "policy",
                    "chunk_id": "policy#1",
                    "score": 0.91,
                }
            ],
        }

        with (
            patch.object(mcp_server, "_ensure_api_running", return_value=True),
            patch.object(mcp_server.requests, "get", return_value=_Response(payload)) as get,
        ):
            result = mcp_server.ask_docs("재택근무는 몇 회 가능해?", top_k=4)

        self.assertIn("재택근무는 주 2회", result)
        self.assertIn("생성 방식: extractive", result)
        self.assertIn("근거:", result)
        self.assertIn("근무 규정 / policy#1", result)
        get.assert_called_once_with(
            f"{mcp_server.API_URL}/answer",
            params={"query": "재택근무는 몇 회 가능해?", "top_k": 4},
            timeout=60,
        )

    def test_list_documents_formats_indexed_documents(self):
        payload = {
            "total": 2,
            "documents": [
                {"doc_id": "policy", "title": "근무 규정"},
                {"doc_id": "benefit", "title": "복지 규정"},
            ],
        }

        with (
            patch.object(mcp_server, "_ensure_api_running", return_value=True),
            patch.object(mcp_server.requests, "get", return_value=_Response(payload)) as get,
        ):
            result = mcp_server.list_documents()

        self.assertIn("총 2개 문서", result)
        self.assertIn("근무 규정", result)
        self.assertIn("복지 규정", result)
        get.assert_called_once_with(f"{mcp_server.API_URL}/documents", params={}, timeout=10)

    def test_search_docs_rejects_invalid_top_k_before_http_call(self):
        with patch.object(mcp_server.requests, "get") as get:
            result = mcp_server.search_docs("재택근무", top_k=0)

        self.assertIn("top_k는 1 이상", result)
        get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
