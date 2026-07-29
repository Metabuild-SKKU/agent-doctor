from __future__ import annotations

import unittest

from agents.ingest.document_type import (
    annotate_document_metadata,
    detect_document_type,
    has_math_signal,
)


class DocumentTypeTests(unittest.TestCase):
    def test_math_document_metadata_is_added_during_ingest_preprocess(self):
        metadata = annotate_document_metadata(
            "SET 17\n172번 문제\nx=2cos^3(t), y=3sin^3(t)\nt=π/4 일 때 접선의 기울기를 구한다.",
            {"filename": "math.pdf"},
        )

        self.assertEqual(metadata["document_type"], "math")
        self.assertEqual(metadata["retrieval_profile"], "math_formula")
        self.assertEqual(metadata["filename"], "math.pdf")

    def test_general_limit_and_user_id_are_not_math_signals(self):
        text = "API rate limit 정책은 분당 요청 수를 제한한다. user_id별 quota를 기록한다."

        self.assertFalse(has_math_signal(text))
        self.assertEqual(detect_document_type(text), "general")

    def test_explicit_document_type_is_preserved(self):
        metadata = annotate_document_metadata(
            "일반 본문",
            {"document_type": "finance_table", "retrieval_profile": "custom"},
        )

        self.assertEqual(metadata["document_type"], "finance_table")
        self.assertEqual(metadata["retrieval_profile"], "custom")


if __name__ == "__main__":
    unittest.main()
