from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agents.ingest.agent import _ingest_json_corpus


class JsonCorpusIngestTests(unittest.TestCase):
    def test_item_metadata_and_document_type_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "id": "math-doc",
                            "text": "1. x=2cos^3(t) 문제와 풀이",
                            "source": "math.pdf",
                            "document_type": "math",
                            "metadata": {
                                "title": "수학 교재",
                                "chapter": "미적분",
                            },
                        },
                        {
                            "id": "general-doc",
                            "text": "API rate limit 정책 설명",
                            "source": "api.md",
                            "metadata": {
                                "document_type": "general",
                                "title": "API 정책",
                            },
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            docs = _ingest_json_corpus(str(path))

        by_id = {doc.doc_id: doc for doc in docs}
        self.assertEqual(by_id["math-doc"].metadata["document_type"], "math")
        self.assertEqual(by_id["math-doc"].metadata["retrieval_profile"], "math_formula")
        self.assertEqual(by_id["math-doc"].metadata["title"], "수학 교재")
        self.assertEqual(by_id["math-doc"].metadata["chapter"], "미적분")
        self.assertEqual(by_id["math-doc"].metadata["source_file"], "math.pdf")

        self.assertEqual(by_id["general-doc"].metadata["document_type"], "general")
        self.assertEqual(by_id["general-doc"].metadata["title"], "API 정책")
        self.assertNotIn("retrieval_profile", by_id["general-doc"].metadata)

    def test_metadata_must_be_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "id": "bad",
                            "text": "본문",
                            "metadata": ["not", "object"],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "metadata"):
                _ingest_json_corpus(str(path))


if __name__ == "__main__":
    unittest.main()
