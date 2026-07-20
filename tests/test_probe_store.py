"""
tests/test_probe_store.py
Probe 캐시 버전 키 검증 — 4단계(버전 키 분리).

corpus_version 은 Probe(골든 테스트셋) 캐시 무효화 키다. 핵심 계약:
  - 원문 문서가 있으면 문서에 의존하고 청킹 설정에는 의존하지 않는다.
  - 원문 문서가 없는 legacy 호출은 청크 id + 텍스트를 사용한다.

probe_store 는 qdrant 의존이 없어 agent.py 없이 단독 테스트가 가능하다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import Chunk, Document, Probe
from agents.eval.probe_store import corpus_version, save_probes, load_probes


def _chunks(*pairs):
    """(chunk_id, text) 쌍들로 Chunk 리스트를 만든다."""
    return [Chunk(chunk_id=cid, doc_id="d1", text=text) for cid, text in pairs]


class CorpusVersionTest(unittest.TestCase):
    def test_same_corpus_same_version(self):
        a = _chunks(("d1_chunk_000", "가나다"), ("d1_chunk_001", "라마바"))
        b = _chunks(("d1_chunk_000", "가나다"), ("d1_chunk_001", "라마바"))
        self.assertEqual(corpus_version(a), corpus_version(b))

    def test_order_independent(self):
        # 청크 순서가 달라도(정렬 후 해싱) 같은 코퍼스면 같은 버전.
        a = _chunks(("d1_chunk_000", "가나다"), ("d1_chunk_001", "라마바"))
        b = _chunks(("d1_chunk_001", "라마바"), ("d1_chunk_000", "가나다"))
        self.assertEqual(corpus_version(a), corpus_version(b))

    def test_text_change_changes_version(self):
        # chunk_id 목록은 같지만 텍스트(경계 이동)가 바뀌면 버전이 달라져야 한다.
        # 이게 위치 기반 chunk_id 의 충돌을 막는 핵심 안전장치다.
        a = _chunks(("d1_chunk_000", "가나다라"), ("d1_chunk_001", "마바사"))
        b = _chunks(("d1_chunk_000", "가나"), ("d1_chunk_001", "다라마바사"))
        self.assertNotEqual(corpus_version(a), corpus_version(b))

    def test_added_chunk_changes_version(self):
        a = _chunks(("d1_chunk_000", "가나다"))
        b = _chunks(("d1_chunk_000", "가나다"), ("d1_chunk_001", "라마바"))
        self.assertNotEqual(corpus_version(a), corpus_version(b))

    def test_delimiter_prevents_boundary_collision(self):
        # 구분자(\x00)가 없으면 ("ab","c")와 ("a","bc")가 같은 해시가 될 수 있다.
        a = _chunks(("d1_chunk_000", "ab"), ("d1_chunk_001", "c"))
        b = _chunks(("d1_chunk_000", "a"), ("d1_chunk_001", "bc"))
        self.assertNotEqual(corpus_version(a), corpus_version(b))

    def test_same_documents_keep_version_across_rechunking(self):
        document = Document("d1", "memory", "txt", "가나다라마바사")
        before = _chunks(("d1_chunk_000", "가나다라"), ("d1_chunk_001", "마바사"))
        after = _chunks(("d1_chunk_000", "가나"), ("d1_chunk_001", "다라마바사"))

        self.assertEqual(
            corpus_version(before, [document]),
            corpus_version(after, [document]),
        )

    def test_document_content_change_invalidates_version(self):
        chunks = _chunks(("d1_chunk_000", "가나다"))
        before = Document("d1", "memory", "txt", "가나다")
        after = Document("d1", "memory", "txt", "가나다 변경")

        self.assertNotEqual(
            corpus_version(chunks, [before]),
            corpus_version(chunks, [after]),
        )


class SaveLoadTest(unittest.TestCase):
    """버전이 일치할 때만 저장된 probe 를 재사용한다(top_k 변경은 코퍼스 버전을
    안 바꾸므로 재사용, 코퍼스 변경은 버전이 바뀌어 재생성 유도)."""

    def _path(self):
        base = os.environ.get("TEMP") or "."
        return os.path.join(base, "test_probe_store_probes.json")

    def setUp(self):
        self.path = self._path()
        self._cleanup()

    def tearDown(self):
        self._cleanup()

    def _cleanup(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_round_trip_same_version(self):
        chunks = _chunks(("d1_chunk_000", "가나다"))
        version = corpus_version(chunks)
        probes = [Probe(probe_id="p1", question="질문", source="taxonomy")]

        save_probes(probes, version, path=self.path)
        loaded = load_probes(version, path=self.path)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded[0].probe_id, "p1")

    def test_version_mismatch_returns_none(self):
        probes = [Probe(probe_id="p1", question="질문", source="taxonomy")]
        save_probes(probes, "v-old", path=self.path)

        # 코퍼스가 바뀌어 버전이 달라지면 재사용하지 않는다(→ 재생성).
        self.assertIsNone(load_probes("v-new", path=self.path))


if __name__ == "__main__":
    unittest.main()
