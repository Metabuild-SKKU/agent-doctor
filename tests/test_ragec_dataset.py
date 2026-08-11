"""
tests/test_ragec_dataset.py
RAGEC + DragonBall → 파이프라인 입력 변환(tools/build_ragec_dataset.py) 검증.

여기서 고정하는 건 **무응답 질문 처리**다. 구현 중에 같은 자리에서 버그가 둘 나왔다.

  ① 근거가 없다고 버렸다 → E9 Abstention Failure 23건 중 17건이 날아갔다.
     답할 수 없는 질문은 근거가 없는 게 정상이라, 버리면 그 라벨을 못 재게 된다.
  ② 골드를 비웠는데 positive_chunk_ids 를 남겼다 → 로더 폴백(`korquad._gold_spans_of`)이
     **문서 통째**를 골드로 만들어, 올바른 기권이 검색 실패로 집계된다.

둘 다 "무응답 probe 에 골드가 생기면 안 된다" 는 한 성질의 다른 얼굴이다.
"""
import json
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.build_ragec_dataset import build, locate


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _fixture(tmp: pathlib.Path):
    """문서 2개 · 질의 3개(답 있음 / 무응답 / 무응답인데 근거 있음)."""
    docs = [
        {"language": "en", "doc_id": 1, "company_name": "Acme",
         "content": "Acme was founded in 1999. It opened a factory in March 2020."},
        {"language": "en", "doc_id": 2, "company_name": "Globex",
         "content": "Globex reported record revenue in 2021."},
        {"language": "zh", "doc_id": 9, "company_name": "無關", "content": "중국어 문서"},
    ]
    queries = [
        {"language": "en", "query": {"query_id": 10, "query_type": "Factual Question",
                                     "content": "When did Acme open a factory?"},
         "ground_truth": {"doc_ids": [1], "content": "March 2020.",
                          "references": ["It opened a factory in March 2020."]}},
        {"language": "en", "query": {"query_id": 11, "query_type": "Irrelevant Unsolvable Question",
                                     "content": "What is Acme's CEO salary?"},
         "ground_truth": {"doc_ids": [1], "content": "Unable to answer", "references": []}},
        {"language": "en", "query": {"query_id": 12, "query_type": "Irrelevant Unsolvable Question",
                                     "content": "Compare Acme and Globex ethics timing."},
         "ground_truth": {"doc_ids": [1, 2], "content": "Unable to answer",
                          "references": ["Globex reported record revenue in 2021."]}},
        # 정답 본문에 "unable to answer" 가 **스쳐 지나가는** 정상 답변. 무응답 판정을
        # 포함으로 하면 이게 무응답으로 뒤집힌다(실측 qa_id=2399).
        {"language": "en", "query": {"query_id": 13, "query_type": "Summary Question",
                                     "content": "Summarize Acme's 2020 disclosure stance."},
         "ground_truth": {"doc_ids": [1],
                          "content": "Acme stated it was unable to answer certain investor "
                                     "questions, and opened a factory in March 2020.",
                          "references": ["It opened a factory in March 2020."]}},
    ]
    annotations = [
        "query_id,question,answer,query_type,rag_answer,error_stage,error_category",
        '10,"When did Acme open a factory?","March 2020.",Factual Question,"...",Retrieval,E4 Missed Retrieval',
        '11,"What is Acme\'s CEO salary?","Unable to answer",Irrelevant Unsolvable Question,"...",Generation,E9 Abstention Failure',
        '12,"Compare Acme and Globex ethics timing.","Unable to answer",Irrelevant Unsolvable Question,"...",Generation,E9 Abstention Failure',
        '13,"Summarize Acme\'s 2020 disclosure stance.","...",Summary Question,"...",Chunking,E1 Overchunking',
    ]
    _write(tmp / "docs.jsonl", docs)
    _write(tmp / "queries.jsonl", queries)
    (tmp / "ragec.csv").write_text("\n".join(annotations) + "\n", encoding="utf-8")
    return (str(tmp / "ragec.csv"), str(tmp / "docs.jsonl"), str(tmp / "queries.jsonl"))


class RagecDatasetBuildTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = pathlib.Path(self._tmp.name)
        self.inputs = _fixture(tmp)
        self.out = {k: str(tmp / f"{k}.jsonl") for k in ("corpus", "qa", "key")}
        self.stats = build(*self.inputs, self.out["corpus"], self.out["qa"], self.out["key"])
        self.qa = [json.loads(l) for l in open(self.out["qa"], encoding="utf-8") if l.strip()]

    def tearDown(self):
        self._tmp.cleanup()

    def _row(self, qa_id):
        return next(r for r in self.qa if r["qa_id"] == qa_id)

    def test_unanswerable_question_is_kept_not_dropped(self):
        """근거가 없다고 버리면 E9 를 못 잰다 — 무응답 질문도 probe 로 남아야 한다."""
        self.assertEqual(len(self.qa), 4)
        self.assertFalse(self._row("11")["answer_exists"])

    def test_phrase_appearing_inside_a_real_answer_is_not_abstention(self):
        """무응답 판정은 정답 칸이 그 문구 **하나뿐**일 때다.

        포함으로 판정하면 본문에 "unable to answer" 가 스쳐 지나가는 정상 답변이
        무응답으로 뒤집히고, 골드까지 버려져 그 probe 가 통째로 오염된다(실측 qa_id=2399).
        """
        row = self._row("13")
        self.assertTrue(row["answer_exists"])
        self.assertEqual(len(row["gold_spans"]), 1)
        self.assertTrue(row["positive_chunk_ids"])

    def test_unanswerable_probe_carries_no_gold_at_all(self):
        """골드가 조금이라도 남으면 recall 이 계산돼 올바른 기권이 검색 실패로 집계된다.

        gold_spans 뿐 아니라 **positive_chunk_ids 도** 비어야 한다 — 로더가 gold_spans 가
        비면 그쪽으로 폴백해 문서 통째를 골드로 만든다.
        """
        row = self._row("11")
        self.assertEqual(row["gold_spans"], [])
        self.assertEqual(row["positive_chunk_ids"], [])

    def test_unanswerable_with_references_drops_the_gold(self):
        """데이터 모순(답할 수 없는데 근거가 달림)은 무응답 판정을 우선한다."""
        row = self._row("12")
        self.assertFalse(row["answer_exists"])
        self.assertEqual(row["gold_spans"], [])
        self.assertEqual(row["positive_chunk_ids"], [])
        self.assertEqual(self.stats.get("무응답인데 근거 있음(버림)"), 1)

    def test_answerable_question_gets_coordinates(self):
        row = self._row("10")
        self.assertTrue(row["answer_exists"])
        self.assertEqual(len(row["gold_spans"]), 1)
        span = row["gold_spans"][0]
        content = next(
            json.loads(l)["text"] for l in open(self.out["corpus"], encoding="utf-8")
            if json.loads(l)["doc_id"] == span["doc_id"]
        )
        self.assertEqual(content[span["start"]:span["end"]],
                         "It opened a factory in March 2020.")

    def test_answer_key_is_separate_from_pipeline_input(self):
        """정답 라벨이 qa 파일에 새면 채점이 아니라 커닝이 된다."""
        for row in self.qa:
            self.assertNotIn("ragec_category", row)
            self.assertNotIn("error_stage", row)
        key = [json.loads(l) for l in open(self.out["key"], encoding="utf-8") if l.strip()]
        self.assertEqual({k["qa_id"] for k in key}, {"10", "11", "12", "13"})
        self.assertEqual(self._key(key, "10")["ragec_category"], "E4 Missed Retrieval")

    @staticmethod
    def _key(key, qa_id):
        return next(k for k in key if k["qa_id"] == qa_id)

    def test_other_languages_are_excluded(self):
        """언어를 섞으면 좌표계가 무의미해진다 — 영어만 싣는다."""
        docs = [json.loads(l) for l in open(self.out["corpus"], encoding="utf-8") if l.strip()]
        self.assertEqual({d["doc_id"] for d in docs}, {"ragec_1", "ragec_2"})


class LocateTest(unittest.TestCase):
    """근거 문장 → 좌표. 좌표는 **원문 기준**이어야 한다(정규화한 문자열 위치가 아니라)."""

    def test_exact_match(self):
        content = "Alpha. Beta gamma delta. Epsilon."
        self.assertEqual(locate("Beta gamma delta.", content), (7, 24))

    def test_whitespace_difference_is_absorbed_but_offsets_stay_original(self):
        content = "Alpha.  Beta   gamma\ndelta. Epsilon."
        found = locate("Beta gamma delta.", content)
        self.assertIsNotNone(found)
        self.assertEqual(content[found[0]:found[1]], "Beta   gamma\ndelta.")

    def test_missing_reference_returns_none(self):
        self.assertIsNone(locate("Nowhere to be found.", "Alpha. Beta."))

    def test_short_prefix_does_not_match_by_accident(self):
        """40자 미만 접두는 우연히 맞을 수 있어 인정하지 않는다."""
        self.assertIsNone(locate("Alpha zzz", "Alpha. Beta."))


if __name__ == "__main__":
    unittest.main()
