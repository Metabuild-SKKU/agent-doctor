from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agents.index.corpus_visualization import (
    build_corpus_visualization_artifacts,
    build_corpus_visualization_data,
)
from core.schema import Chunk


def _chunk(
    chunk_id: str,
    doc_id: str,
    text: str,
    embedding: list[float] | None = None,
    token_count: int | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        section="Guide",
        token_count=token_count,
        hash=chunk_id,
        embedding=embedding,
        metadata={"title": f"{doc_id} title", "source": f"https://example.com/{doc_id}"},
    )


class CorpusVisualizationTests(unittest.TestCase):
    def test_builds_summary_and_projection_data(self):
        chunks = [
            _chunk("c1", "doc-a", "Qdrant vector search guide", [1.0, 0.0, 0.2], 5),
            _chunk("c2", "doc-a", "RAG chunk overlap tuning", [0.9, 0.1, 0.1], 7),
            _chunk("c3", "doc-b", "Notion ingest and metadata", [0.0, 1.0, 0.2], 4),
        ]

        data = build_corpus_visualization_data(
            chunks,
            {"embedding_model": "test-embedding"},
            max_points=20,
        )

        self.assertEqual(data["summary"]["documents"], 2)
        self.assertEqual(data["summary"]["chunks"], 3)
        self.assertEqual(data["summary"]["total_tokens"], 16)
        self.assertEqual(data["summary"]["embedding_dimension"], 3)
        self.assertEqual(len(data["projection"]["points"]), 3)
        self.assertEqual(data["documents"][0]["doc_id"], "doc-a")

    def test_writes_report_artifacts(self):
        chunks = [
            _chunk("c1", "doc-a", "Qdrant vector search guide", [1.0, 0.0], 5),
            _chunk("c2", "doc-b", "RAG evaluation probe result", [0.0, 1.0], 6),
        ]

        with tempfile.TemporaryDirectory() as directory:
            artifacts = build_corpus_visualization_artifacts(
                chunks,
                {
                    "embedding_model": "test-embedding",
                    "corpus_visualization_output_dir": directory,
                },
            )

            html_path = Path(artifacts["html"])
            json_path = Path(artifacts["json"])
            self.assertTrue(html_path.exists())
            self.assertTrue(json_path.exists())
            self.assertEqual(artifacts["summary"]["documents"], 2)
            self.assertEqual(artifacts["point_count"], 2)
            self.assertIn("검색 자료 현황", html_path.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["summary"]["chunks"], 2)

    def test_no_embeddings_still_generates_summary(self):
        chunks = [_chunk("c1", "doc-a", "plain text only", None, None)]

        data = build_corpus_visualization_data(chunks, {}, max_points=20)

        self.assertEqual(data["summary"]["embedding_coverage"], 0)
        self.assertEqual(data["projection"]["method"], "none")
        self.assertEqual(data["projection"]["points"], [])


if __name__ == "__main__":
    unittest.main()
