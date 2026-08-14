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

from agents.eval.datasets.korquad import _as_qtype
from tools.build_ragec_dataset import _QTYPE_BY_QUERY_TYPE, _qtype_of, build, locate


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

    def test_csv_with_bom_still_builds(self):
        """팀원이 엑셀로 한 번 열어 저장하면 BOM 이 붙는다 — DictReader 의 첫 컬럼명이
        '﻿query_id' 가 되어 ann["query_id"] 가 KeyError 로 죽는다."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            ragec, docs, queries = _fixture(tmp)
            raw = pathlib.Path(ragec).read_text(encoding="utf-8")
            pathlib.Path(ragec).write_text("﻿" + raw, encoding="utf-8")
            out = {k: str(tmp / f"out_{k}.jsonl") for k in ("corpus", "qa", "key")}
            stats = build(ragec, docs, queries, out["corpus"], out["qa"], out["key"])
        self.assertEqual(stats["QA"], 4)

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

    def test_prefix_fallback_never_runs_past_the_document(self):
        """접두 폴백은 원문과 참조가 다를 때만 오므로 start+len(ref) 가 문서를 넘을 수 있다.

        넘으면 좌표 자체가 무효고, 다음 문장을 삼키면 gold span 이 부풀어 span_recall
        판정이 어긋난다.
        """
        content = "The board approved the merger in June after review."
        ref = "The board approved the merger in June after review, and the CFO resigned later."
        found = locate(ref, content)
        self.assertIsNotNone(found)
        self.assertLessEqual(found[1], len(content))

    def test_prefix_fallback_stops_at_the_sentence_end(self):
        """뒤 문장을 통째로 삼키면 그 문장이 골드가 아닌데 골드로 채점된다."""
        content = ("The board approved the merger in June after a long review. "
                   "The CFO resigned in December for unrelated reasons entirely.")
        ref = "The board approved the merger in June after a long review, per the filing."
        start, end = locate(ref, content)
        self.assertEqual(content[start:end],
                         "The board approved the merger in June after a long review.")


class QtypeMappingTest(unittest.TestCase):
    """질문 유형을 안 실으면 진단이 '근거가 몇 개 필요한 질문인지' 를 모른 채 판정한다.

    retrieval_incomplete_enumeration 은 qtype=aggregation 일 때만 확정이라, 없으면 예비로
    강등돼 확정 라벨(retrieval_low_rank)에게 슬롯을 뺏긴다. 실측: 18건 자체 라벨링에서
    사람이 '나열형 슬롯 부족' 이라 본 5건(gold span 8~23개인데 top_k=5)을 전부 놓쳤다.
    처방이 정반대다 — "검색 개수를 늘려라" ↔ "리랭커를 켜라".
    """

    def test_multi_evidence_types_become_aggregation(self):
        for query_type in ("Multi-document Information Integration Question",
                           "Summarization Question", "Summary Question"):
            self.assertEqual(_qtype_of(query_type), "aggregation", query_type)

    def test_comparison_types_become_comparison(self):
        """Time Sequence 는 이름과 달리 'Compare A and B. Which…' 형태의 비교 질문이다."""
        for query_type in ("Multi-document Comparison Question",
                           "Multi-document Time Sequence Question"):
            self.assertEqual(_qtype_of(query_type), "comparison", query_type)

    def test_multi_hop_is_not_mapped_to_bridge(self):
        """bridge 는 1단계 답을 알아야 2단계 근거를 찾는 경우다.

        DragonBall 의 multi-hop 은 "How did X in April lead to Y by August?" 처럼 양쪽
        사건이 질문에 다 적혀 있어 해당하지 않는다. bridge 로 넣으면 semantic/lexical
        mismatch 가 양보해(diagnose 의 qtype=='bridge' 게이트) 50건의 검색 진단이
        통째로 침묵한다.
        """
        self.assertEqual(_qtype_of("Multi-hop Reasoning Question"), "aggregation")

    def test_single_hop_types_have_no_qtype(self):
        for query_type in ("Factual Question", "Irrelevant Unsolvable Question"):
            self.assertIsNone(_qtype_of(query_type), query_type)

    def test_unknown_type_is_none_not_a_guess(self):
        """모르는 유형을 임의로 매핑하면 진단 경로가 조용히 바뀐다."""
        self.assertIsNone(_qtype_of("Brand New Question Type"))
        self.assertIsNone(_qtype_of(""))

    def test_build_actually_writes_qtype(self):
        """매핑 함수만 맞아도 **배선이 빠지면 아무 효과가 없다.**

        실제로 뮤테이션(어댑터에서 qtype 줄 삭제)이 매핑 테스트를 그대로 통과했다.
        같은 유형의 사고가 이 프로젝트에서 두 번째다(observations 배선 누락).
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            out = {k: str(tmp / f"{k}.jsonl") for k in ("corpus", "qa", "key")}
            build(*_fixture(tmp), out["corpus"], out["qa"], out["key"])
            rows = {r["qa_id"]: r for r in
                    (json.loads(l) for l in open(out["qa"], encoding="utf-8") if l.strip())}
        self.assertEqual(rows["13"]["qtype"], "aggregation")   # Summary Question
        self.assertIsNone(rows["10"]["qtype"])                 # Factual Question

    def test_every_query_type_in_the_answer_key_is_mapped(self):
        """정답지에 있는 유형이 표에 없으면 그 건들이 조용히 qtype 없이 나간다."""
        path = pathlib.Path("data/ragec_answer_key.jsonl")
        if not path.exists():
            self.skipTest("data/ragec_answer_key.jsonl 없음")
        types = {
            json.loads(line)["query_type"].strip()
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        }
        self.assertEqual(types - set(_QTYPE_BY_QUERY_TYPE), set())


class LoaderQtypeTest(unittest.TestCase):
    """어댑터가 실어도 로더가 안 읽으면 그대로 끊긴다 — 양쪽을 함께 고정한다."""

    def test_known_values_pass_through(self):
        for value in ("aggregation", "comparison", "bridge"):
            self.assertEqual(_as_qtype(value), value)

    def test_case_and_padding_are_absorbed(self):
        self.assertEqual(_as_qtype("  Aggregation "), "aggregation")

    def test_unknown_or_missing_is_none(self):
        """오타 하나가 진단 경로를 바꾸는 것보다 '미표시' 로 떨어지는 쪽이 안전하다."""
        for value in ("aggregatoin", "", None, 3, True):
            self.assertIsNone(_as_qtype(value), repr(value))

    def test_loader_puts_qtype_on_the_probe(self):
        """어댑터가 실어도 로더가 안 읽으면 그대로 끊긴다 — 실제 Probe 까지 가서 확인한다."""
        from agents.eval.datasets.korquad import load_taxonomy_probes
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            _write(tmp / "corpus.jsonl", [{
                "doc_id": "d1", "chunk_id": "d1_0", "title": "T",
                "text": "Acme opened a factory in March 2020.", "char_start": 0, "char_end": 36,
            }])
            _write(tmp / "qa.jsonl", [
                {"qa_id": "1", "question": "When?", "answer_text": "March 2020",
                 "doc_id": "d1", "qtype": "aggregation",
                 "gold_spans": [{"doc_id": "d1", "start": 0, "end": 36}],
                 "positive_chunk_ids": ["d1_0"], "answer_exists": True},
                {"qa_id": "2", "question": "When?", "answer_text": "March 2020",
                 "doc_id": "d1", "gold_spans": [{"doc_id": "d1", "start": 0, "end": 36}],
                 "positive_chunk_ids": ["d1_0"], "answer_exists": True},
            ])
            probes = load_taxonomy_probes(str(tmp / "qa.jsonl"), str(tmp / "corpus.jsonl"))
        by_id = {p.metadata["qa_id"]: p for p in probes}
        self.assertEqual(by_id["1"].qtype, "aggregation")
        self.assertIsNone(by_id["2"].qtype)     # 미표시 데이터셋(KorQuAD)은 그대로 None


if __name__ == "__main__":
    unittest.main()
