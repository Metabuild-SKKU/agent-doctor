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
from agents.eval.agent import _short_cid, _doc_tag, _fmt_cids, _log_probe


class DocTagTest(unittest.TestCase):
    def test_short_cid_drops_doc_prefix(self):
        self.assertEqual(_short_cid("doc_ace9d8c1ce5d_chunk_005"), "chunk_005")

    def test_doc_tag_distinguishes_documents(self):
        a = _doc_tag("doc_ace9d8c1ce5d_chunk_005")
        b = _doc_tag("doc_8536fd7d1ce4_chunk_005")
        self.assertNotEqual(a, b)                 # 동명 청크라도 문서가 다르면 태그가 다르다
        self.assertEqual(a, _doc_tag("doc_ace9d8c1ce5d_chunk_011"))  # 같은 문서면 같은 태그

    def test_fmt_single_doc_unchanged(self):
        cids = ["doc_A_chunk_000", "doc_A_chunk_001"]
        self.assertEqual(_fmt_cids(cids, multi_doc=False), "chunk_000, chunk_001")

    def test_fmt_multi_doc_appends_tag(self):
        out = _fmt_cids(["doc_ace9d8c1ce5d_chunk_005"], multi_doc=True)
        self.assertEqual(out, "chunk_005@ace9d8")


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

    def test_cross_document_homonym_is_disambiguated(self):
        # 골드는 니미츠 문서 chunk_005, 검색된 chunk_005 는 '다른 문서' — 로그가
        # 둘을 구분해 recall=0 이 착시가 아님을 드러내야 한다.
        rec = self._rec(
            retrieved_ids=["doc_pearlXXXXXX_chunk_005", "doc_pearlXXXXXX_chunk_001"],
            gold_ids=["doc_ace9d8c1ce5d_chunk_004", "doc_ace9d8c1ce5d_chunk_005"],
        )
        out = self._capture(rec)
        self.assertIn("chunk_005@", out)                    # 태그가 붙어 구분됨
        # 검색 chunk_005 와 골드 chunk_005 의 태그가 서로 달라야(다른 문서) 한다.
        self.assertIn("chunk_005@pearlX", out)
        self.assertIn("chunk_005@ace9d8", out)

    def test_single_document_display_unchanged(self):
        # 검색과 골드가 같은 문서면 태그 없이 기존 표시를 유지한다(하위호환).
        rec = self._rec(
            retrieved_ids=["doc_ace9d8c1ce5d_chunk_005", "doc_ace9d8c1ce5d_chunk_001"],
            gold_ids=["doc_ace9d8c1ce5d_chunk_004", "doc_ace9d8c1ce5d_chunk_005"],
        )
        out = self._capture(rec)
        cid_line = next(line for line in out.splitlines() if "검색 [" in line)
        self.assertNotIn("@", cid_line)           # 검색/골드 표시에 태그가 붙지 않음
        self.assertIn("검색 [chunk_005, chunk_001] / 골드 [chunk_004, chunk_005]", cid_line)


if __name__ == "__main__":
    unittest.main()
