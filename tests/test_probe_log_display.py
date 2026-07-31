"""
tests/test_probe_log_display.py
검색/골드 청크 로그 표시의 문서 간 동명 청크 충돌 방지 검증.

문제(관측성): _short_cid 는 'doc_A_chunk_005' → 'chunk_005' 로 문서 접두를 버린다.
검색이 골드와 '다른 문서'의 chunk_005 를 가져오면 로그에 둘 다 'chunk_005' 로 찍혀,
recall=0 인데도 "골드를 검색했다"는 착시를 만든다. 문서가 섞일 때만 '@문서태그' 를
붙여 이 충돌을 드러내되, 단일 문서(대다수) 케이스는 표시를 바꾸지 않는다.
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import Probe
from agents.eval.types import EvalRecord
from agents.eval.agent import (
    _short_cid, _cid_doc, _doc_tag, _fmt_cids, _colliding_short_cids, _log_probe,
)


class DocTagTest(unittest.TestCase):
    def test_short_cid_drops_doc_prefix(self):
        self.assertEqual(_short_cid("doc_ace9d8c1ce5d_chunk_005"), "chunk_005")

    def test_cid_doc_keeps_full_doc_id(self):
        # 동일성 판정은 전체 doc id 로 — 절단하면 접두 겹치는 문서가 같은 것으로 접힌다.
        self.assertEqual(_cid_doc("doc_ace9d8c1ce5d_chunk_005"), "doc_ace9d8c1ce5d")

    def test_doc_tag_distinguishes_documents(self):
        a = _doc_tag("doc_ace9d8c1ce5d_chunk_005")
        b = _doc_tag("doc_8536fd7d1ce4_chunk_005")
        self.assertNotEqual(a, b)                 # 동명 청크라도 문서가 다르면 태그가 다르다
        self.assertEqual(a, _doc_tag("doc_ace9d8c1ce5d_chunk_011"))  # 같은 문서면 같은 태그

    def test_fmt_no_collision_unchanged(self):
        cids = ["doc_A_chunk_000", "doc_A_chunk_001"]
        self.assertEqual(_fmt_cids(cids, colliding=set()), "chunk_000, chunk_001")

    def test_fmt_tags_only_colliding_items(self):
        # 충돌 집합에 든 short_cid 에만 태그가 붙고, 나머지는 그대로다(태그 남발 방지).
        out = _fmt_cids(
            ["doc_ace9d8c1ce5d_chunk_005", "doc_ace9d8c1ce5d_chunk_001"],
            colliding={"chunk_005"},
        )
        self.assertEqual(out, "chunk_005@ace9d8, chunk_001")


class CollidingShortCidsTest(unittest.TestCase):
    def test_cross_document_homonym_detected(self):
        colliding = _colliding_short_cids(
            ["doc_pearlXXXXXX_chunk_005"], ["doc_ace9d8c1ce5d_chunk_005"],
        )
        self.assertEqual(colliding, {"chunk_005"})

    def test_same_document_not_colliding(self):
        colliding = _colliding_short_cids(
            ["doc_A_chunk_005", "doc_A_chunk_001"], ["doc_A_chunk_005"],
        )
        self.assertEqual(colliding, set())

    def test_multi_doc_without_homonym_not_colliding(self):
        # Low 1 회귀: 여러 문서에 걸쳐도 같은 short_cid 충돌이 없으면 태그를 켜지 않는다.
        colliding = _colliding_short_cids(
            ["doc_A_chunk_000", "doc_B_chunk_001"], ["doc_C_chunk_002"],
        )
        self.assertEqual(colliding, set())

    def test_identity_uses_full_doc_id_not_truncation(self):
        # Low 2 회귀: 접두 6자가 겹쳐도(전체 id 는 다름) 서로 다른 문서로 판정해야 한다.
        colliding = _colliding_short_cids(
            ["doc_abcdef111111_chunk_005"], ["doc_abcdef222222_chunk_005"],
        )
        self.assertEqual(colliding, {"chunk_005"})     # 전체 id 로 보면 충돌


class LogProbeCollisionTest(unittest.TestCase):
    """_log_probe 통합: 검색과 골드가 다른 문서면 동명 청크가 태그로 구분되어야 한다."""

    def _rec(self, retrieved_ids, gold_ids):
        probe = Probe(probe_id="p1", question="q", source="taxonomy",
                      ground_truth="정답", gold_chunk_ids=gold_ids)
        rec = EvalRecord(probe=probe)
        rec.retrieved_chunk_ids = retrieved_ids
        rec.recall_at_k, rec.f1_score, rec.oracle_f1 = 0.0, 1.0, 1.0
        rec.generated_answer = "답"
        return rec

    def _capture(self, rec):
        buf = io.StringIO()
        with redirect_stdout(buf):
            _log_probe(16, 30, rec)
        return buf.getvalue()

    def _cid_line(self, rec):
        return next(line for line in self._capture(rec).splitlines() if "검색 [" in line)

    def test_cross_document_homonym_is_disambiguated(self):
        # 골드는 니미츠 문서 chunk_005, 검색된 chunk_005 는 '다른 문서' — 충돌하는 chunk_005 만
        # 태그로 구분되고, 충돌 없는 chunk_001/004 는 태그가 붙지 않는다.
        rec = self._rec(
            retrieved_ids=["doc_pearlXXXXXX_chunk_005", "doc_pearlXXXXXX_chunk_001"],
            gold_ids=["doc_ace9d8c1ce5d_chunk_004", "doc_ace9d8c1ce5d_chunk_005"],
        )
        line = self._cid_line(rec)
        self.assertIn("chunk_005@pearlX", line)             # 검색 쪽 chunk_005
        self.assertIn("chunk_005@ace9d8", line)             # 골드 쪽 chunk_005 — 다른 문서
        self.assertIn("검색 [chunk_005@pearlX, chunk_001]", line)  # 충돌 없는 001 은 태그 없음
        self.assertNotIn("chunk_001@", line)
        self.assertNotIn("chunk_004@", line)

    def test_multi_document_without_homonym_has_no_tags(self):
        # Low 1 회귀: 검색·골드가 여러 문서에 걸쳐도 동명 충돌이 없으면 태그를 붙이지 않는다.
        rec = self._rec(
            retrieved_ids=["doc_AAAAAAAAAAAA_chunk_003", "doc_BBBBBBBBBBBB_chunk_007"],
            gold_ids=["doc_CCCCCCCCCCCC_chunk_009"],
        )
        cid_line = self._cid_line(rec)
        self.assertNotIn("@", cid_line)
        self.assertIn("검색 [chunk_003, chunk_007] / 골드 [chunk_009]", cid_line)

    def test_single_document_display_unchanged(self):
        # 검색과 골드가 같은 문서면 태그 없이 기존 표시를 유지한다(하위호환).
        rec = self._rec(
            retrieved_ids=["doc_ace9d8c1ce5d_chunk_005", "doc_ace9d8c1ce5d_chunk_001"],
            gold_ids=["doc_ace9d8c1ce5d_chunk_004", "doc_ace9d8c1ce5d_chunk_005"],
        )
        cid_line = self._cid_line(rec)
        self.assertNotIn("@", cid_line)           # 검색/골드 표시에 태그가 붙지 않음
        self.assertIn("검색 [chunk_005, chunk_001] / 골드 [chunk_004, chunk_005]", cid_line)


if __name__ == "__main__":
    unittest.main()
