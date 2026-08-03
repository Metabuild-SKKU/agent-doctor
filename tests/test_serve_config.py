"""tests/test_serve_config.py
Serve 영속화(리뷰 blocker#2): 파이프라인이 고른 검색·생성 설정이 사이드카를 통해
별도 프로세스인 API 로 전달돼 retriever/generator 에 실제 반영되는지 검증.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.serve import serve_config as sc


class ServeConfigRoundTripTest(unittest.TestCase):
    def test_extract_keeps_only_serving_keys(self):
        cfg = sc.extract_serve_config({
            "top_k": 3, "use_mmr": True, "abstention_strict": True,
            "chunk_size": 512, "chunk_strategy": "fixed",  # 재색인 키 → 제외
        })
        self.assertEqual(cfg, {"top_k": 3, "use_mmr": True, "abstention_strict": True})

    def test_write_then_read_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            chunks = Path(d) / "chunks.json"
            chunks.write_text("[]", encoding="utf-8")
            written = sc.write_serve_config(chunks, {"top_k": 5, "use_reranker": True, "chunk_size": 999})
            self.assertEqual(sc.read_serve_config(chunks), written)
            self.assertNotIn("chunk_size", written)

    def test_read_missing_sidecar_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(sc.read_serve_config(Path(d) / "nope.json"), {})

    def test_fingerprint_changes_when_config_only_changes(self):
        # 재색인 없는 generation-only 변경도 지문이 달라져야 API 가 reload 한다.
        chunks = [{"chunk_id": "a", "hash": "h1"}]
        fp_off = sc.serving_fingerprint(chunks, {"abstention_strict": False})
        fp_on = sc.serving_fingerprint(chunks, {"abstention_strict": True})
        self.assertNotEqual(fp_off, fp_on)

    def test_generation_subset_excludes_retrieval_keys(self):
        gen = sc.generation_subset({"top_k": 3, "temperature": 0.1, "abstention_strict": True})
        self.assertEqual(gen, {"temperature": 0.1, "abstention_strict": True})

    def test_generation_subset_includes_context_compression_keys(self):
        gen = sc.generation_subset({
            "temperature": 0.1,
            "context_compression": True,
            "context_compression_max_contexts": 2,
            "context_compression_min_contexts": 1,
            "context_compression_max_sentences": 3,
        })
        self.assertEqual(
            gen,
            {
                "temperature": 0.1,
                "context_compression": True,
                "context_compression_max_contexts": 2,
                "context_compression_min_contexts": 1,
                "context_compression_max_sentences": 3,
            },
        )

    def test_fingerprint_changes_when_context_compression_changes(self):
        chunks = [{"chunk_id": "a", "hash": "h1"}]
        fp_off = sc.serving_fingerprint(chunks, {"context_compression": False})
        fp_on = sc.serving_fingerprint(chunks, {"context_compression": True})
        self.assertNotEqual(fp_off, fp_on)


class ApiInitAppliesServeConfigTest(unittest.TestCase):
    """init_qdrant 가 사이드카 설정을 retriever·generation 에 실제 주입하는지 end-to-end."""

    def test_init_qdrant_applies_retrieval_and_generation_config(self):
        from agents.serve import api

        with tempfile.TemporaryDirectory() as d:
            chunks_file = Path(d) / "chunks.json"
            chunks = [
                {"chunk_id": "a", "doc_id": "x", "text": "재택근무 규정",
                 "hash": "h1", "embedding": [1.0, 0.0]},
                {"chunk_id": "b", "doc_id": "x", "text": "연차 규정",
                 "hash": "h2", "embedding": [0.0, 1.0]},
            ]
            chunks_file.write_text(json.dumps(chunks), encoding="utf-8")
            index_config = {
                "top_k": 1, "use_mmr": True, "mmr_lambda": 0.3,
                "embedding_model": "test-model", "embedding_dimension": 2,
                "abstention_strict": True, "temperature": 0.15,
                "context_compression": True,
                "context_compression_max_contexts": 2,
                "chunk_size": 999,  # 재색인 키 → 사이드카에서 제외돼야
            }
            served = sc.write_serve_config(chunks_file, index_config)

            api.init_qdrant(str(chunks_file))

            # 검색 설정이 retriever 에 반영됨
            self.assertTrue(api._retriever.settings.use_mmr)
            self.assertEqual(api._retriever.settings.top_k, 1)
            self.assertAlmostEqual(api._retriever.settings.mmr_lambda, 0.3)
            # 생성 설정이 generator 로 주입됨(재색인 키는 제외)
            self.assertTrue(api._generation_config.get("abstention_strict"))
            self.assertAlmostEqual(api._generation_config.get("temperature"), 0.15)
            self.assertTrue(api._generation_config.get("context_compression"))
            self.assertEqual(api._generation_config.get("context_compression_max_contexts"), 2)
            self.assertNotIn("chunk_size", api._generation_config)
            # 지문은 코퍼스+설정 결합 → Serve agent 계산과 일치
            self.assertEqual(api._fingerprint, sc.serving_fingerprint(chunks, served))


if __name__ == "__main__":
    unittest.main()
