from __future__ import annotations

import unittest

from agents.index.corpus_visualization import build_corpus_visualization_data
from core.schema import Chunk
from tools.korquad_corpus_visualization import ChunkRow, _project_chunks
from tools.render_corpus_dashboard import _dashboard_payload


class CorpusDashboardToolTests(unittest.TestCase):
    def test_renderer_rejects_index_summary_schema_with_clear_error(self):
        index_data = build_corpus_visualization_data(
            [
                Chunk(
                    chunk_id="c1",
                    doc_id="doc-a",
                    text="plain index chunk",
                    token_count=3,
                    embedding=[1.0, 0.0],
                    metadata={"title": "doc-a title"},
                )
            ],
            {"embedding_model": "test-embedding"},
        )

        with self.assertRaisesRegex(ValueError, "KorQuAD dashboard"):
            _dashboard_payload(index_data)

    def test_korquad_projection_handles_one_sample(self):
        projection = _project_chunks(
            [
                ChunkRow(
                    chunk_id="c1",
                    doc_id="doc-a",
                    title="Doc A",
                    url="",
                    text="single chunk text",
                    token_count=3,
                    char_count=17,
                    qa_count=1,
                    index=0,
                )
            ],
            vector_dim=8,
        )

        self.assertEqual(projection["point_count"], 1)
        self.assertEqual(len(projection["explained_variance"]), 2)
        self.assertEqual(projection["points"][0]["x"], 0.0)
        self.assertEqual(projection["points"][0]["y"], 0.0)


if __name__ == "__main__":
    unittest.main()
