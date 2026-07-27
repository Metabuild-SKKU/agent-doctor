"""
tests/test_serve_api_reload.py
Serve API 의 코퍼스 지문 + /reload 통합 검증 (실서버 없이 FastAPI TestClient).

리뷰 회귀(P1-②): 이미 실행 중인 API 는 시작 시점에만 chunks.json 을 읽으므로,
새 파이프라인이 다른 코퍼스를 써도 /search·/answer 가 낡은 코퍼스를 계속 서빙할 수
있었다. /health 가 코퍼스 지문을 노출하고 /reload 가 최신 파일을 다시 읽어
_chunks_raw·retriever 를 교체하는지, 그래서 두 번째 코퍼스가 실제 API 에 반영되는지 본다.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from agents.serve import api
from agents.serve.api import app, corpus_fingerprint


def _write_chunks(path: Path, chunks: list[dict]) -> None:
    path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")


_CORPUS_A = [
    {"chunk_id": "a1", "doc_id": "docA", "text": "첫 번째 코퍼스", "hash": "ha1",
     "metadata": {"title": "문서 A"}},
]
_CORPUS_B = [
    {"chunk_id": "b1", "doc_id": "docB", "text": "두 번째 코퍼스", "hash": "hb1",
     "metadata": {"title": "문서 B"}},
]


class ApiReloadTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.chunks_file = Path(self._tmp.name) / "chunks.json"

    def test_reload_reflects_second_corpus(self):
        # 1) 코퍼스 A 로드
        _write_chunks(self.chunks_file, _CORPUS_A)
        api.init_qdrant(str(self.chunks_file))
        client = TestClient(app)

        health_a = client.get("/health").json()
        self.assertEqual(health_a["fingerprint"], corpus_fingerprint(_CORPUS_A))
        docs_a = client.get("/documents").json()
        self.assertEqual([d["doc_id"] for d in docs_a["documents"]], ["docA"])

        # 2) 같은 경로에 코퍼스 B 를 쓴다 (새 파이프라인이 chunks.json 을 덮어쓴 상황)
        _write_chunks(self.chunks_file, _CORPUS_B)

        # 3) reload 전에는 여전히 A 를 서빙한다 (시작 시점에만 파일을 읽으므로)
        self.assertEqual(
            client.get("/health").json()["fingerprint"], corpus_fingerprint(_CORPUS_A)
        )

        # 4) /reload 후 지문·문서가 B 로 바뀐다
        reloaded = client.post("/reload").json()
        self.assertEqual(reloaded["fingerprint"], corpus_fingerprint(_CORPUS_B))
        docs_b = client.get("/documents").json()
        self.assertEqual([d["doc_id"] for d in docs_b["documents"]], ["docB"])

    def test_fingerprint_changes_with_corpus(self):
        self.assertNotEqual(
            corpus_fingerprint(_CORPUS_A), corpus_fingerprint(_CORPUS_B)
        )
        # 순서가 달라도 같은 청크 집합이면 지문은 동일하다(정렬 후 해시).
        self.assertEqual(
            corpus_fingerprint(_CORPUS_A + _CORPUS_B),
            corpus_fingerprint(_CORPUS_B + _CORPUS_A),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
