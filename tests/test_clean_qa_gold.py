"""
tests/test_clean_qa_gold.py
골든 QA 정제 — 골드가 '정답이 실제로 있는 좁은 구간'을 가리키는지 고정한다.

배경: qa_pairs.jsonl 의 골드는 정답 위치가 아니라 **정답이 든 청크 통째**(중앙값 497자,
정답은 7자)를 가리킨다. span_recall_at_k 는 그 구간을 빈틈없이 덮어야 1점을 주는 이진
판정이라, corpus 청크 경계와 Index 재청킹 경계가 다르면 정답을 맞힌 실행도 recall=0 이
된다. 실측(corpus_20260804_103059) 30문항 중 4건(13%)이 그랬다.

더 나쁜 건 골드가 **엉뚱한 곳**을 가리키는 경우다 — 정답 텍스트를 문서에서 다시 찾는
방식이라 표 문서에서 같은 값이 여러 행에 나오면 앞의 것에 꽂힌다.

    "파스칼레 소틸레의 스파이크 높이는?" → "332cm" 가 문서에 8곳
    골드 1188~2028(다른 선수) vs 실제 5268("소틸레" 는 문서에 1회)

여기서 고정하는 계약은 둘이다.
  - 로더는 명시 gold_spans 를 우선하되, 없으면 기존 청크 id 환산으로 흐른다(하위호환)
  - 빌더는 정답이 문서에 **정확히 한 번** 나오는 QA 만 남기고 골드를 그 위치로 좁힌다
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import pathlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.eval.datasets.korquad import _gold_spans_of, _stitch
from tools.build_clean_qa import build, covering_chunk_ids


class GoldSpanResolutionTest(unittest.TestCase):
    """로더 — 명시 좌표 우선, 없으면 청크 환산."""

    SPAN_OF = {("d1", "d1_0"): (0, 500), ("d1", "d1_1"): (450, 950)}

    def test_explicit_spans_win_over_chunk_ids(self):
        qa = {"gold_spans": [{"doc_id": "d1", "start": 300, "end": 306}],
              "positive_chunk_ids": ["d1_0", "d1_1"]}
        self.assertEqual(_gold_spans_of(qa, "d1", self.SPAN_OF),
                         [{"doc_id": "d1", "start": 300, "end": 306}])

    def test_falls_back_to_chunk_ids(self):
        """기존 qa_pairs.jsonl 이 그대로 돌아야 한다 — 정제본은 선택이지 강제가 아니다."""
        qa = {"positive_chunk_ids": ["d1_0", "d1_1"]}
        self.assertEqual(_gold_spans_of(qa, "d1", self.SPAN_OF),
                         [{"doc_id": "d1", "start": 0, "end": 500},
                          {"doc_id": "d1", "start": 450, "end": 950}])

    def test_malformed_spans_fall_back_instead_of_yielding_empty(self):
        """좌표가 깨졌는데 빈 골드로 흘리면 recall 이 조용히 미측정(None)이 된다."""
        qa = {"gold_spans": [{"start": "300", "end": None}],
              "positive_chunk_ids": ["d1_0"]}
        self.assertEqual(_gold_spans_of(qa, "d1", self.SPAN_OF),
                         [{"doc_id": "d1", "start": 0, "end": 500}])

    def test_reversed_or_negative_span_is_rejected(self):
        qa = {"gold_spans": [{"doc_id": "d1", "start": 400, "end": 100}],
              "positive_chunk_ids": ["d1_0"]}
        self.assertEqual(_gold_spans_of(qa, "d1", self.SPAN_OF),
                         [{"doc_id": "d1", "start": 0, "end": 500}])

    def test_unknown_chunk_id_is_dropped(self):
        qa = {"positive_chunk_ids": ["d1_0", "d1_없음"]}
        self.assertEqual(len(_gold_spans_of(qa, "d1", self.SPAN_OF)), 1)


class CleanQaBuilderTest(unittest.TestCase):
    """빌더 — 모호한 정답을 걸러내고 골드를 정답 위치로 좁힌다."""

    # "332cm" 가 두 번 나오는 표 문서(실제 실패 사례의 축소판)와, 한 번만 나오는 문서.
    DOC_TABLE = "가" * 100 + "선수A 332cm " + "나" * 300 + "소틸레 332cm " + "다" * 100
    DOC_UNIQUE = "라" * 200 + "정답은 MKS 벵진 이다" + "마" * 200

    def _write(self, tmp, corpus_rows, qa_rows):
        cp = pathlib.Path(tmp) / "corpus.jsonl"
        qp = pathlib.Path(tmp) / "qa.jsonl"
        op = pathlib.Path(tmp) / "out.jsonl"
        cp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in corpus_rows),
                      encoding="utf-8")
        qp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in qa_rows),
                      encoding="utf-8")
        return str(cp), str(qp), str(op)

    def _corpus(self, doc_id, text, size=200):
        return [{"doc_id": doc_id, "chunk_id": f"{doc_id}_{i // size}", "title": "t",
                 "text": text[i:i + size], "char_start": i, "char_end": min(i + size, len(text))}
                for i in range(0, len(text), size)]

    def _run(self, corpus_rows, qa_rows, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            cp, qp, op = self._write(tmp, corpus_rows, qa_rows)
            stats = build(cp, qp, op, kw.get("max_answer", 50), kw.get("min_answer", 2))
            rows = [json.loads(l) for l in open(op, encoding="utf-8") if l.strip()]
        return rows, stats

    def test_ambiguous_answer_is_dropped(self):
        """정답이 여러 곳이면 어느 쪽이 정답인지 모르므로 골드를 만들 수 없다."""
        rows, stats = self._run(
            self._corpus("d1", self.DOC_TABLE),
            [{"qa_id": "1", "question": "소틸레의 높이는?", "answer_text": "332cm",
              "doc_id": "d1", "positive_chunk_ids": ["d1_0"]}])
        self.assertEqual(rows, [])
        self.assertEqual(stats["정답이 여러 곳(모호)"], 1)

    def test_unique_answer_gets_a_narrow_gold(self):
        rows, _ = self._run(
            self._corpus("d2", self.DOC_UNIQUE),
            [{"qa_id": "2", "question": "어느 팀?", "answer_text": "MKS 벵진",
              "doc_id": "d2", "positive_chunk_ids": ["d2_0", "d2_1", "d2_2"]}])
        self.assertEqual(len(rows), 1)
        span = rows[0]["gold_spans"][0]
        # 골드 폭이 정답 길이와 같다 — 청크 통째(200자)가 아니라.
        self.assertEqual(span["end"] - span["start"], len("MKS 벵진"))
        self.assertEqual(self.DOC_UNIQUE[span["start"]:span["end"]], "MKS 벵진")

    def test_positive_chunk_ids_are_rebuilt_not_inherited(self):
        """원본의 넓은(때로 엉뚱한) 청크 목록을 그대로 물려주면 하위호환 경로가 계속 틀린다."""
        rows, _ = self._run(
            self._corpus("d2", self.DOC_UNIQUE),
            [{"qa_id": "2", "question": "어느 팀?", "answer_text": "MKS 벵진",
              "doc_id": "d2", "positive_chunk_ids": ["d2_0", "d2_1", "d2_2"]}])
        self.assertLess(len(rows[0]["positive_chunk_ids"]), 3)
        self.assertNotIn("d2_0", rows[0]["positive_chunk_ids"])   # 정답은 뒤쪽에 있다

    def test_long_answer_is_dropped(self):
        """1회 등장 필터는 긴 서술형 정답 쪽으로 치우친다 — char-F1 이 요약을 오답으로 깎는다."""
        long_answer = "정답" * 40                       # 80자
        text = "바" * 50 + long_answer + "사" * 50
        rows, stats = self._run(
            self._corpus("d3", text),
            [{"qa_id": "3", "question": "왜?", "answer_text": long_answer,
              "doc_id": "d3", "positive_chunk_ids": ["d3_0"]}])
        self.assertEqual(rows, [])
        self.assertEqual(stats["정답이 50자 초과"], 1)

    def test_too_short_answer_is_dropped(self):
        """1~2자 정답은 우연 일치라 '1회 등장' 판정 자체를 못 믿는다."""
        rows, stats = self._run(
            self._corpus("d4", "아" * 100 + "2" + "자" * 100),
            [{"qa_id": "4", "question": "몇 회?", "answer_text": "2",
              "doc_id": "d4", "positive_chunk_ids": ["d4_0"]}])
        self.assertEqual(rows, [])
        self.assertEqual(stats["정답이 2자 미만"], 1)

    def test_answer_absent_from_document_is_dropped(self):
        rows, stats = self._run(
            self._corpus("d5", "차" * 300),
            [{"qa_id": "5", "question": "?", "answer_text": "없는정답",
              "doc_id": "d5", "positive_chunk_ids": ["d5_0"]}])
        self.assertEqual(rows, [])
        self.assertEqual(stats["정답이 문서에 없음"], 1)

    def test_output_is_loadable_by_the_loader_contract(self):
        """빌더가 쓴 형식을 로더가 그대로 읽어야 한다 — 두 파일이 어긋나면 조용히 폴백한다."""
        rows, _ = self._run(
            self._corpus("d2", self.DOC_UNIQUE),
            [{"qa_id": "2", "question": "어느 팀?", "answer_text": "MKS 벵진",
              "doc_id": "d2", "positive_chunk_ids": ["d2_0"]}])
        resolved = _gold_spans_of(rows[0], "d2", {})     # span_of 가 비어도 명시 좌표로 해결
        self.assertEqual(resolved, rows[0]["gold_spans"])


class CoveringChunkIdsTest(unittest.TestCase):
    SPANS = [("c0", 0, 100), ("c1", 100, 200), ("c2", 200, 300)]

    def test_picks_only_overlapping_chunks(self):
        self.assertEqual(covering_chunk_ids(self.SPANS, 150, 160), ["c1"])

    def test_span_across_a_boundary_needs_both(self):
        self.assertEqual(covering_chunk_ids(self.SPANS, 95, 105), ["c0", "c1"])

    def test_touching_edge_does_not_count(self):
        """[100,105) 는 c0 의 끝(100)에 닿기만 한다 — 겹침이 아니다."""
        self.assertEqual(covering_chunk_ids(self.SPANS, 100, 105), ["c1"])


class CoordinateSystemMatchesPipelineTest(unittest.TestCase):
    """빌더가 낸 좌표를 **파이프라인이 읽었을 때** 정답을 가리키는지.

    빌더와 파이프라인이 원문을 서로 다르게 복원하면 골드가 통째로 밀린다 — 이 도구가
    고치려던 결함(골드가 엉뚱한 곳을 가리킴)을 그대로 다시 만드는 셈이다.

    처음 구현은 `"".join(청크 본문)` 으로 이어붙였고 그게 틀렸다(리뷰 지적). 파이프라인의
    `_stitch` 는 각 청크를 **char_start 위치에 놓는데**, 이어붙이기는 좌표를 무시한다.
    둘은 청크가 딱 맞닿을 때만 같고 **겹치거나 틈이 있으면 갈라진다.**

    기존 테스트가 이걸 못 잡은 이유는 헬퍼(`_corpus`)가 맞닿는 청크만 만들었기 때문이다.
    실제 코퍼스는 1,000개 문서가 **전부** 겹쳐 있었고, 정제본 549건 중 539건(98%)의
    좌표가 어긋나 있었다. 그래서 여기서는 겹치는 코퍼스로만 검증한다.
    """

    DOC = "가" * 300 + "정답은 MKS 벵진 이다" + "나" * 300

    def _overlapping_corpus(self, doc_id, text, size=200, overlap=50):
        """실제 코퍼스처럼 **겹치는** 청크. step < size 라 구간이 서로 물린다."""
        step = size - overlap
        rows, i, idx = [], 0, 0
        while i < len(text):
            end = min(i + size, len(text))
            rows.append({"doc_id": doc_id, "chunk_id": f"{doc_id}_{idx}", "title": "t",
                         "text": text[i:end], "char_start": i, "char_end": end})
            if end >= len(text):
                break
            i += step
            idx += 1
        return rows

    def _build(self, corpus_rows, qa_rows):
        with tempfile.TemporaryDirectory() as tmp:
            cp = pathlib.Path(tmp) / "corpus.jsonl"
            qp = pathlib.Path(tmp) / "qa.jsonl"
            op = pathlib.Path(tmp) / "out.jsonl"
            cp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in corpus_rows),
                          encoding="utf-8")
            qp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in qa_rows),
                          encoding="utf-8")
            build(str(cp), str(qp), str(op), 50, 2)
            return [json.loads(l) for l in open(op, encoding="utf-8") if l.strip()]

    def test_gold_span_points_at_the_answer_in_pipeline_coordinates(self):
        """빌더 좌표를 파이프라인이 복원한 문서에 그대로 대면 정답이 나와야 한다.

        이 단언이 이 파일의 핵심이다 — 나머지 테스트가 전부 통과해도 이게 깨지면
        정제본은 쓸 수 없다. 이어붙이기 방식으로 되돌리면 여기서 잡힌다.
        """
        rows = self._build(
            self._overlapping_corpus("d1", self.DOC),
            [{"qa_id": "1", "question": "정답은?", "answer_text": "MKS 벵진",
              "doc_id": "d1", "positive_chunk_ids": ["d1_0"]}])
        self.assertEqual(len(rows), 1)

        span = rows[0]["gold_spans"][0]
        restored = _stitch([(r["char_start"], r["char_end"], r["text"])
                            for r in self._overlapping_corpus("d1", self.DOC)])
        self.assertEqual(restored[span["start"]:span["end"]], "MKS 벵진")

    def test_overlapping_corpus_actually_diverges_between_the_two_restorations(self):
        """이 fixture 가 두 복원 방식을 실제로 가르는지 — 안 그러면 위 테스트가 무의미하다.

        겹침이 없으면 join 과 _stitch 가 같아져 회귀를 못 잡는다. fixture 가 조건을
        만족하는지 여기서 못박는다.
        """
        rows = self._overlapping_corpus("d1", self.DOC)
        joined = "".join(r["text"] for r in rows)
        stitched = _stitch([(r["char_start"], r["char_end"], r["text"]) for r in rows])
        self.assertNotEqual(joined, stitched)
        self.assertEqual(stitched, self.DOC)      # _stitch 는 원문을 그대로 복원한다

    def test_covering_chunk_ids_are_in_the_same_coordinates(self):
        """하위호환용 positive_chunk_ids 도 같은 좌표계여야 한다 — 겹치면 여러 개가 나온다."""
        rows = self._build(
            self._overlapping_corpus("d1", self.DOC),
            [{"qa_id": "1", "question": "정답은?", "answer_text": "MKS 벵진",
              "doc_id": "d1", "positive_chunk_ids": ["d1_0"]}])
        span = rows[0]["gold_spans"][0]
        for cid in rows[0]["positive_chunk_ids"]:
            chunk = next(r for r in self._overlapping_corpus("d1", self.DOC)
                         if r["chunk_id"] == cid)
            self.assertLess(chunk["char_start"], span["end"])
            self.assertGreater(chunk["char_end"], span["start"])


class GoldSpanBoolRejectionTest(unittest.TestCase):
    """리뷰 지적(soongwo0o) — bool 은 int 의 서브클래스라 isinstance(x, int) 만으로는
    안 걸러진다. True 가 로더를 통과하면 명시 좌표로 채택돼 폴백(청크 id 환산)을 안 타고,
    그 좌표로는 span_recall_at_k 가 None 을 내 recall_at_k=-1(미측정)이 된다 —
    'malformed 는 폴백으로 흘려 조용한 미측정을 막는다'는 이 파일의 계약과 정반대다."""

    SPAN_OF = {("d1", "d1_0"): (0, 500)}

    def test_bool_start_is_rejected_like_other_malformed_values(self):
        qa = {"gold_spans": [{"doc_id": "d1", "start": True, "end": 5}],
              "positive_chunk_ids": ["d1_0"]}
        self.assertEqual(_gold_spans_of(qa, "d1", self.SPAN_OF),
                         [{"doc_id": "d1", "start": 0, "end": 500}])

    def test_bool_end_is_rejected_like_other_malformed_values(self):
        qa = {"gold_spans": [{"doc_id": "d1", "start": 0, "end": False}],
              "positive_chunk_ids": ["d1_0"]}
        self.assertEqual(_gold_spans_of(qa, "d1", self.SPAN_OF),
                         [{"doc_id": "d1", "start": 0, "end": 500}])


class BuilderCliRobustnessTest(unittest.TestCase):
    """리뷰 지적(soongwo0o) — main() 이 잘못된 입력에서 트레이스백 대신 오류 메시지로 죽는다."""

    def test_empty_qa_file_does_not_raise_zero_division(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = pathlib.Path(tmp) / "corpus.jsonl"
            qp = pathlib.Path(tmp) / "qa.jsonl"
            op = pathlib.Path(tmp) / "out.jsonl"
            cp.write_text("", encoding="utf-8")
            qp.write_text("", encoding="utf-8")   # 유효한 줄이 0건
            import subprocess
            import sys as _sys
            result = subprocess.run(
                [_sys.executable, str(pathlib.Path(__file__).resolve().parent.parent
                                      / "tools" / "build_clean_qa.py"),
                 "--corpus", str(cp), "--qa", str(qp), "--out", str(op)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertNotIn("Traceback", result.stderr)

    def test_missing_out_directory_is_created(self):
        rows, _ = self._run_with_nested_out()
        self.assertTrue(rows)

    def _run_with_nested_out(self):
        doc = "라" * 200 + "정답은 MKS 벵진 이다" + "마" * 200
        corpus_rows = [{"doc_id": "d1", "chunk_id": "d1_0", "title": "t",
                        "text": doc, "char_start": 0, "char_end": len(doc)}]
        qa_rows = [{"qa_id": "1", "question": "?", "answer_text": "MKS 벵진",
                   "doc_id": "d1", "positive_chunk_ids": ["d1_0"]}]
        with tempfile.TemporaryDirectory() as tmp:
            cp = pathlib.Path(tmp) / "corpus.jsonl"
            qp = pathlib.Path(tmp) / "qa.jsonl"
            op = pathlib.Path(tmp) / "nested" / "dir" / "out.jsonl"   # 없는 디렉터리
            cp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in corpus_rows),
                          encoding="utf-8")
            qp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in qa_rows),
                          encoding="utf-8")
            build(str(cp), str(qp), str(op), 50, 2)
            rows = [json.loads(l) for l in open(op, encoding="utf-8") if l.strip()]
        return rows, None


if __name__ == "__main__":
    unittest.main()
