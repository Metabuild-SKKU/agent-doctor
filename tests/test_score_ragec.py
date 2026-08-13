"""
tests/test_score_ragec.py
RAGEC 대조 채점기(tools/score_ragec.py) 검증.

여기서 고정하는 건 **무엇을 맞은 것으로 세고 무엇을 세지 않는가** 다. 채점 규칙이 조금만
느슨해도 정확도가 거짓으로 올라가고, 그 수치로 8-a(격자) 를 할지 말지를 정하게 된다.

  · 포함 — 그들 라벨이 우리 findings 안에 있으면 맞음. 우리가 더 말한 건 오답이 아니다
  · 미진단(우리가 아무 라벨도 안 냄)은 **오답**으로 센다. 빼면 "말 안 하면 안 틀린다" 가 된다
  · 대응 라벨이 없는 카테고리는 **제외**한다. 맞출 수단이 없는 걸 오답으로 세면 거짓이다
  · bad_gold_* 는 제외한다 — 우리 오탐인지 정답지 오류인지 갈리지 않는다
  · 검색 단계 주장인데 recall=1.0 이면 제외한다 — 그들 시스템의 실패 지점이라 우리에겐
    성립하지 않는다. 판정 근거가 **우리 라벨이 아니라 recall** 인 게 핵심이다(아래 클래스)

제외 규칙은 정확도를 **올리는 방향**으로만 작동하므로, 하나 늘릴 때마다 두 가지를 함께
고정한다: (1) 빠지면 안 되는 것이 안 빠지는가, (2) 몇 건이 빠졌는지 화면에 남는가.
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.score_ragec import (
    RAGEC_TO_OURS, _width, findings_from_report, format_detail, format_report,
    score, stage_of,
)


def _key(qa_id, category, stage):
    return {"qa_id": qa_id, "ragec_category": category, "ragec_stage": stage,
            "query_type": "Factual Question"}


def _found(qa_id, *labels, failed=True):
    """우리 진단 한 줄. 기본은 '실패했고 이 라벨을 냈다' — 채점 대상이 되는 상태다."""
    return {"qa_id": qa_id, "labels": list(labels), "failed": failed}


class ContainmentScoringTest(unittest.TestCase):
    def test_exact_match_counts(self):
        result = score([_found("1", "retrieval_missing_gold")],
                       [_key("1", "E4 Missed Retrieval", "Retrieval")])
        self.assertEqual((result["total_hit"], result["total"]), (1, 1))

    def test_extra_labels_do_not_break_the_match(self):
        """우리가 더 말한 건 오답이 아니다 — RAGEC 은 최초 단계 하나만 적는다."""
        result = score(
            [_found("1", "retrieval_missing_gold", "generation_hallucination")],
            [_key("1", "E4 Missed Retrieval", "Retrieval")])
        self.assertEqual((result["total_hit"], result["total"]), (1, 1))

    def test_wrong_label_is_counted_wrong(self):
        result = score([_found("1", "generation_hallucination")],
                       [_key("1", "E4 Missed Retrieval", "Retrieval")])
        self.assertEqual((result["total_hit"], result["total"]), (0, 1))

    def test_any_of_the_mapped_labels_counts(self):
        """1:N 대응은 그중 하나만 맞아도 통과다(우리가 원인별로 쪼갠 것이라 처방이 다르다)."""
        for label in RAGEC_TO_OURS["E7 Low Recall"]:
            result = score([_found("1", label)], [_key("1", "E7 Low Recall", "Reranking")])
            self.assertEqual(result["total_hit"], 1, label)

    def test_failed_but_no_label_is_counted_as_wrong(self):
        """실패했는데 원인을 못 짚은 건 오답이다.

        빼면 '아무 말도 안 하면 안 틀린다' 가 되어 정확도가 거짓으로 오른다.
        """
        result = score([{"qa_id": "1", "labels": [], "failed": True}],
                       [_key("1", "E4 Missed Retrieval", "Retrieval")])
        self.assertEqual((result["total_hit"], result["total"]), (0, 1))
        self.assertEqual(result["no_diagnosis"], 1)


class OurPipelineMayNotFailTest(unittest.TestCase):
    """RAGEC 377건은 *그들* 시스템이 실패한 질문이다.

    검색기·생성 모델이 다른 우리는 같은 질문에서 성공할 수 있다. 그걸 오답으로 세면
    정확도가 진단 품질이 아니라 **"얼마나 그들과 비슷하게 실패하나"** 를 재게 된다.
    """

    def test_probe_we_passed_is_excluded_not_failed(self):
        result = score([{"qa_id": "1", "labels": [], "failed": False}],
                       [_key("1", "E4 Missed Retrieval", "Retrieval")])
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["we_passed"], 1)
        self.assertEqual(result["no_diagnosis"], 0)

    def test_probe_missing_from_dump_is_excluded(self):
        """덤프에 없으면 안 돌린 것이다 — 성공과 구분한다."""
        result = score([], [_key("1", "E4 Missed Retrieval", "Retrieval")])
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["not_run"], 1)

    def test_dump_without_status_is_flagged(self):
        """failed 필드가 없으면 '성공' 과 '못 짚음' 이 구분되지 않는다 — 리포트가 밝힌다."""
        legacy = score([{"qa_id": "1", "labels": ["retrieval_missing_gold"]}],
                       [_key("1", "E4 Missed Retrieval", "Retrieval")])
        self.assertFalse(legacy["has_status"])
        self.assertEqual(legacy["total_hit"], 1)     # 라벨이 있으면 실패로 추론
        typed = score([{"qa_id": "1", "labels": ["retrieval_missing_gold"], "failed": True}],
                      [_key("1", "E4 Missed Retrieval", "Retrieval")])
        self.assertTrue(typed["has_status"])


class ExclusionTest(unittest.TestCase):
    def test_category_without_our_label_is_excluded_not_failed(self):
        """대응 라벨이 빈 카테고리는 제외한다 — 맞출 수단이 없는 걸 오답으로 세면 거짓이다.

        지금은 빈 항목이 하나도 없다(E15 가 main #132 의 라벨 신설로 채워진 마지막이었다).
        그래도 이 분기는 살아 있어야 한다 — 새 카테고리가 들어오면 라벨이 생기기 전까지
        빈 집합으로 두는 게 정상 경로다. 그래서 가짜 항목으로 분기를 직접 태운다.
        """
        with mock.patch.dict(RAGEC_TO_OURS, {"E99 Not Yet Labeled": set()}):
            result = score([_found("1", "generation_misinterpretation")],
                           [_key("1", "E99 Not Yet Labeled", "Generation")])
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["unmappable"], 1)

    def test_e15_now_has_a_label(self):
        """main #132 가 generation_chronological_error 를 신설해 대응이 생겼다.

        **대응이 있다는 것과 검증된다는 건 다르다** — RAGEC 정답지에 E15 는 0건이라
        이 대조로는 E15 진단의 유효성을 확인할 수 없다(docs/ragec_label_mapping.md).
        """
        self.assertEqual(RAGEC_TO_OURS["E15 Chronological Inconsistency"],
                         {"generation_chronological_error"})
        result = score([_found("1", "generation_chronological_error")],
                       [_key("1", "E15 Chronological Inconsistency", "Generation")])
        self.assertEqual((result["total_hit"], result["total"]), (1, 1))

    def test_gold_error_claim_is_excluded(self):
        """우리 오탐인지 DragonBall 정답지 오류인지 갈리지 않아 정확도에 섞지 않는다."""
        result = score([_found("1", "bad_gold_chunk")],
                       [_key("1", "E4 Missed Retrieval", "Retrieval")])
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["gold_error"], 1)

    def test_unknown_category_is_excluded(self):
        """정답지가 바뀌어 새 카테고리가 오면 조용히 오답으로 세지 않는다."""
        result = score([_found("1", "retrieval_missing_gold")],
                       [_key("1", "E99 Something New", "Retrieval")])
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["unmappable"], 1)


class RetrievalSucceededExclusionTest(unittest.TestCase):
    """RAGEC 라벨은 **그들** 시스템의 실패 원인이다.

    검색기가 다른 우리는 같은 질문의 다른 지점에서 넘어질 수 있고, 그때 라벨이 다른 건
    오진이 아니라 시스템이 다른 것이다. 실측(qa_id 2205): RAGEC 은 E4(검색 실패)인데
    우리는 recall=1.00 으로 gold 를 다 찾아 답변에 정답을 그대로 담고도 날짜 하나 때문에
    기권했다 — generation_wrongful_abstention 이 맞는 진단인데 오답으로 집계됐다.

    **판정 근거가 recall 이어야 하는 이유가 여기 다 들어 있다.** 우리 라벨로 판정하면
    "진단이 다르니까 봐준다" 는 순환이 되어 틀릴 수가 없는 채점이 된다. recall 은 진단보다
    먼저·진단과 무관하게 계산되므로 우리 진단을 반박할 수도 있다.
    """

    def _row(self, recall, *labels):
        row = _found("1", *labels)
        if recall is not None:
            row["recall_at_k"] = recall
            row["recall_basis"] = "span"
        return row

    def test_full_recall_excludes_a_retrieval_claim(self):
        result = score([self._row(1.0, "generation_wrongful_abstention")],
                       [_key("1", "E4 Missed Retrieval", "Retrieval")])
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["retrieval_ok"], 1)

    def test_partial_recall_is_still_scored(self):
        """부분 검색은 반박이 아니다 — 실측에서 부분은 0.16·0.58 로 1.0 과 뚜렷이 갈린다."""
        for recall in (0.0, 0.16, 0.58, 0.99):
            result = score([self._row(recall, "generation_wrongful_abstention")],
                           [_key("1", "E4 Missed Retrieval", "Retrieval")])
            self.assertEqual((result["total"], result["retrieval_ok"]), (1, 0), recall)

    def test_generation_category_is_never_excluded(self):
        """생성 단계 라벨은 검색이 성공했다고 해서 성립하지 않게 되지 않는다.

        오히려 recall=1.0 은 생성 실패의 **전제**다(근거는 왔는데 답을 틀렸다).
        여기를 안 막으면 B그룹 채점이 통째로 사라진다.
        """
        result = score([self._row(1.0, "retrieval_low_rank")],
                       [_key("1", "E10 Fabricated Content", "Generation")])
        self.assertEqual((result["total"], result["retrieval_ok"]), (1, 0))

    def test_low_precision_is_not_refuted_by_recall(self):
        """E8 은 '쓰레기가 섞였다' 라 gold 를 다 가져와도 성립한다.

        단계(Reranking)로 뭉치면 E7 과 함께 빠져나간다 — 그래서 카테고리 단위로 판정한다.
        """
        result = score([self._row(1.0, "retrieval_low_rank")],
                       [_key("1", "E8 Low Precision", "Reranking")])
        self.assertEqual((result["total"], result["retrieval_ok"]), (1, 0))

    def test_chunking_is_not_refuted_by_recall(self):
        """gold 조각을 전부 검색해 recall 이 1.0 이어도 경계가 잘못 잘린 건 그대로다."""
        result = score([self._row(1.0, "retrieval_low_rank")],
                       [_key("1", "E3 Context Mismatch", "Chunking")])
        self.assertEqual((result["total"], result["retrieval_ok"]), (1, 0))

    def test_low_recall_category_is_refuted(self):
        """E7 은 정의상 recall 주장이라 반박된다."""
        result = score([self._row(1.0, "generation_hallucination")],
                       [_key("1", "E7 Low Recall", "Reranking")])
        self.assertEqual(result["retrieval_ok"], 1)

    def test_exclusion_beats_no_diagnosis(self):
        """순서 검증 — 이 분기가 no_diagnosis 뒤로 가면 영영 안 걸린다.

        '검색은 됐는데 우리가 아무 라벨도 못 낸' probe 가 먼저 오답으로 세어지기 때문이다.
        """
        result = score([{"qa_id": "1", "labels": [], "failed": True, "recall_at_k": 1.0}],
                       [_key("1", "E4 Missed Retrieval", "Retrieval")])
        self.assertEqual((result["retrieval_ok"], result["no_diagnosis"]), (1, 0))

    def test_we_passed_beats_exclusion(self):
        """우리가 성공한 probe 는 그대로 we_passed 다 — 두 통에 겹쳐 세면 안 된다."""
        result = score([{"qa_id": "1", "labels": [], "failed": False, "recall_at_k": 1.0}],
                       [_key("1", "E4 Missed Retrieval", "Retrieval")])
        self.assertEqual((result["we_passed"], result["retrieval_ok"]), (1, 0))

    def test_gold_error_beats_exclusion(self):
        result = score([self._row(1.0, "bad_gold_chunk")],
                       [_key("1", "E4 Missed Retrieval", "Retrieval")])
        self.assertEqual((result["gold_error"], result["retrieval_ok"]), (1, 0))

    def test_missing_recall_never_excludes(self):
        """구버전 덤프(recall 없음)는 조용히 빠지면 안 된다 — 전부 채점 대상이다."""
        result = score([self._row(None, "generation_wrongful_abstention")],
                       [_key("1", "E4 Missed Retrieval", "Retrieval")])
        self.assertEqual((result["total"], result["retrieval_ok"]), (1, 0))
        self.assertFalse(result["has_recall"])

    def test_report_states_when_recall_is_unavailable(self):
        """'이번엔 0건' 과 '애초에 못 잰다' 가 구분돼야 한다."""
        without = format_report(score([self._row(None, "generation_hallucination")],
                                      [_key("1", "E4 Missed Retrieval", "Retrieval")]))
        self.assertIn("덤프에 recall 이 없어", without)
        with_recall = format_report(score([self._row(0.5, "generation_hallucination")],
                                          [_key("1", "E4 Missed Retrieval", "Retrieval")]))
        self.assertNotIn("덤프에 recall 이 없어", with_recall)

    def test_exclusion_count_is_always_printed(self):
        """빼는 것보다 숨기는 게 나쁘다 — 몇 건을 뺐는지 화면에 남아야 한다."""
        out = format_report(score([self._row(1.0, "generation_wrongful_abstention")],
                                  [_key("1", "E4 Missed Retrieval", "Retrieval")]))
        self.assertIn("우리 검색은 성공", out)
        self.assertIn("1건", out)

    def test_detail_marks_the_excluded_probe(self):
        out = format_detail([self._row(1.0, "generation_wrongful_abstention")],
                            [_key("1", "E4 Missed Retrieval", "Retrieval")])
        self.assertIn("[검색OK]", out)
        self.assertIn("recall@k(span)=1.00", out)


class StageScoringTest(unittest.TestCase):
    def test_stage_matches_even_when_label_is_wrong(self):
        """단계 채점은 라벨보다 거칠어 정책 차이에 강하다 — 그래서 병기한다.

        E3(청킹 단계)를 우리가 retrieval 라벨로 잡은 경우: 라벨은 틀렸는데 단계는
        둘 다 검색 계열이 아니라 갈린다 — 여기서는 라벨·단계 모두 틀린 예다.
        """
        result = score([_found("1", "generation_hallucination")],
                       [_key("1", "E4 Missed Retrieval", "Retrieval")])
        self.assertEqual(result["total_hit"], 0)          # 라벨 틀림
        self.assertEqual((result["stage_hit"], result["stage_total"]), (0, 1))  # 단계도 틀림

    def test_stage_can_be_right_while_label_is_wrong(self):
        """E3(Chunking)를 우리가 다른 청킹 라벨로 잡으면 라벨은 틀리고 단계는 맞는다."""
        result = score([_found("1", "chunking_overchunking")],
                       [_key("1", "E3 Context Mismatch", "Chunking")])
        self.assertEqual(result["total_hit"], 0)
        self.assertEqual((result["stage_hit"], result["stage_total"]), (1, 1))

    def test_e4_accepts_the_specific_cause_labels(self):
        """E4 는 '검색 결과에 정답이 없다' 는 롤업이다 — 원인을 더 짚었으면 맞은 것이다.

        처음엔 retrieval_missing_gold 하나로 좁혔다가 실측 10건에서 E4 3건을 전부
        놓쳤다(우리는 retrieval_low_rank 를 냈다).
        """
        for label in ("retrieval_missing_gold", "retrieval_low_rank",
                      "retrieval_rerank_candidate_miss", "retrieval_semantic_mismatch"):
            result = score([_found("1", label)],
                           [_key("1", "E4 Missed Retrieval", "Retrieval")])
            self.assertEqual(result["total_hit"], 1, label)

    def test_reranker_labels_are_reranking_not_retrieval(self):
        """우리 A그룹 안에 청킹·리랭킹이 섞여 있어, 그룹으로 재면 A 정확도가 거저 오른다.

        라벨 이름에서 RAGEC 단계를 직접 유도해 같은 해상도로 맞춘다.
        """
        self.assertEqual(stage_of("retrieval_rerank_candidate_miss"), "Reranking")
        self.assertEqual(stage_of("reranker_low_precision"), "Reranking")
        self.assertEqual(stage_of("retrieval_missing_gold"), "Retrieval")
        self.assertEqual(stage_of("chunking_overchunking"), "Chunking")

    def test_labels_without_a_ragec_counterpart_have_no_stage(self):
        """C그룹·D그룹은 RAGEC 4단계에 자리가 없다."""
        self.assertIsNone(stage_of("too_long_context"))
        self.assertIsNone(stage_of("bad_gold_answer"))


class FindingsFromReportTest(unittest.TestCase):
    class _Finding:
        def __init__(self, label, probes):
            self.label = label
            self.affected_probes = probes

    class _Report:
        def __init__(self, findings):
            self.findings = findings

    def test_groups_labels_by_probe_and_strips_prefix(self):
        report = self._Report([
            self._Finding("retrieval_missing_gold", ["probe_qa_2135", "probe_qa_2159"]),
            self._Finding("generation_hallucination", ["probe_qa_2135"]),
        ])
        rows = {r["qa_id"]: set(r["labels"]) for r in findings_from_report(report)}
        self.assertEqual(rows["2135"], {"retrieval_missing_gold", "generation_hallucination"})
        self.assertEqual(rows["2159"], {"retrieval_missing_gold"})

    def test_findings_without_label_are_skipped(self):
        report = self._Report([self._Finding(None, ["probe_qa_1"])])
        self.assertEqual(findings_from_report(report), [])

    def test_probe_ids_as_strings_still_work(self):
        """구 호출부는 probe_id 문자열 목록을 넘긴다 — 계약을 깨지 않는다."""
        report = self._Report([self._Finding("retrieval_low_rank", ["probe_qa_1"])])
        rows = findings_from_report(report, ["probe_qa_1", "probe_qa_2"])
        self.assertEqual([r["qa_id"] for r in rows], ["1", "2"])
        self.assertEqual([r["failed"] for r in rows], [True, False])


class _Probe:
    def __init__(self, probe_id, question, ground_truth):
        self.probe_id = probe_id
        self.question = question
        self.ground_truth = ground_truth


class DumpCarriesSourceTextTest(unittest.TestCase):
    """덤프에 질문·답변·정답 원문을 싣는다.

    라벨만 보면 **진단이 틀린 건지 데이터가 틀린 건지** 갈리지 않는다. 실측에서 영어 질문에
    한국어 답변이 붙은 걸 답변 원문을 보고서야 찾았다 — 라벨(generation_hallucination)만
    봤으면 진단 오류로 결론냈을 사안이다.
    """

    def _report(self):
        finding = FindingsFromReportTest._Finding("retrieval_low_rank", ["probe_qa_1"])
        report = FindingsFromReportTest._Report([finding])
        report.failed_questions = [{
            "probe_id": "probe_qa_1", "question": "Where?",
            "expected_answer": "Guadeloupe", "actual_answer": "Bangalore",
        }]
        return report

    def test_failed_probe_carries_question_answer_and_gold(self):
        rows = findings_from_report(
            self._report(), [_Probe("probe_qa_1", "Where?", "Guadeloupe")])
        self.assertEqual(rows[0]["question"], "Where?")
        self.assertEqual(rows[0]["answer"], "Bangalore")
        self.assertEqual(rows[0]["gold_answer"], "Guadeloupe")

    def test_passed_probe_carries_question_and_gold_without_answer(self):
        """성공 probe 는 report 에 답변이 안 남는다(failed_questions 에만 있다).

        그래도 질문·정답은 실어야 '우리가 성공했다' 는 판정을 사람이 검증할 수 있다.
        """
        rows = findings_from_report(
            self._report(), [_Probe("probe_qa_9", "Which plant?", "Osaka")])
        self.assertFalse(rows[0]["failed"])
        self.assertEqual(rows[0]["question"], "Which plant?")
        self.assertEqual(rows[0]["gold_answer"], "Osaka")
        self.assertNotIn("answer", rows[0])


class DetailReportTest(unittest.TestCase):
    KEY = [_key("1", "E4 Missed Retrieval", "Retrieval")]

    def _detail(self, row):
        return format_detail([row], self.KEY)

    def test_question_answer_and_gold_all_appear(self):
        out = self._detail({
            "qa_id": "1", "labels": ["retrieval_low_rank"], "failed": True,
            "question": "Where is the plant?", "answer": "In Bangalore.",
            "gold_answer": "Guadeloupe",
        })
        for expected in ("Where is the plant?", "In Bangalore.", "Guadeloupe",
                         "retrieval_low_rank", "E4 Missed Retrieval"):
            self.assertIn(expected, out, expected)

    def test_verdict_matches_the_scorer(self):
        """대조표의 판정이 채점기와 어긋나면 표를 근거로 고칠 수 없다.

        gold 오류 주장은 score() 가 **제일 먼저** 걸러 정확도에서 뺀다. 표가 그걸 'X'(틀림)로
        표시하면 사람이 없는 오진을 쫓게 된다.
        """
        gold_claim = {"qa_id": "1", "labels": ["bad_gold_chunk"], "failed": True}
        self.assertIn("[gold]", self._detail(gold_claim))
        self.assertEqual(score([gold_claim], self.KEY)["total"], 0)

    def test_unmappable_category_is_marked_excluded_not_wrong(self):
        row = {"qa_id": "1", "labels": ["generation_misinterpretation"], "failed": True}
        key = [_key("1", "E99 Something New", "Generation")]
        self.assertIn("[-]", format_detail([row], key))

    def test_passed_probe_is_marked_and_says_why_the_answer_is_missing(self):
        out = self._detail({"qa_id": "1", "labels": [], "failed": False,
                            "question": "Where?", "gold_answer": "Guadeloupe"})
        self.assertIn("[성공]", out)
        self.assertIn("실패 probe 만 보존", out)

    def test_summary_table_still_follows_the_blocks(self):
        """블록만 남기고 표를 없애면 30건 넘는 실행에서 전체를 못 본다."""
        out = self._detail({"qa_id": "1", "labels": ["retrieval_low_rank"],
                            "failed": True, "question": "Q", "gold_answer": "G"})
        self.assertIn("요약 표", out)
        self.assertLess(out.index("qa_id 1"), out.index("요약 표"))

    def test_missing_text_fields_do_not_break_output(self):
        """구버전 덤프(라벨만 있는 파일)로도 돌아야 한다."""
        out = self._detail({"qa_id": "1", "labels": ["retrieval_low_rank"], "failed": True})
        self.assertIn("(없음)", out)
        self.assertIn("retrieval_low_rank", out)

    def test_probe_absent_from_the_dump_is_skipped(self):
        self.assertIn("공통으로 있는 probe 가 없습니다", format_detail([], self.KEY))


class TableAlignmentTest(unittest.TestCase):
    """한글은 터미널에서 두 칸을 먹는다 — 글자 수로 채우면 그 줄만 밀린다.

    헤더에만 한글이 있어(카테고리/맞음/전체/정확도) 헤더와 본문이 어긋났다. 표가
    밀리면 결과를 눈으로 훑을 수 없으니 표시 폭으로 고정한다.
    """

    KEY = [_key("1", "E4 Missed Retrieval", "Retrieval"),
           _key("2", "E13 Misinterpretation", "Generation")]
    ROWS = [_found("1", "retrieval_missing_gold"), _found("2", "generation_hallucination")]

    def test_category_table_columns_line_up(self):
        out = format_report(score(self.ROWS, self.KEY))
        rows = [line for line in out.splitlines()
                if "카테고리" in line or line.startswith("  E")]
        self.assertGreaterEqual(len(rows), 3)          # 헤더 + 카테고리 2줄
        self.assertEqual(len({_width(line) for line in rows}), 1,
                         "\n".join(f"{_width(l):>4}  {l}" for l in rows))

    def test_summary_table_columns_line_up(self):
        """'성공'(2글자=4칸)과 'O'(1칸)가 같은 열에 온다 — 판정 열이 특히 잘 밀린다."""
        rows = [{**r, "failed": i == 0} for i, r in enumerate(self.ROWS)]
        out = format_detail(rows, self.KEY).splitlines()
        # 각 줄에서 'RAGEC 정답' 열이 시작하는 표시 폭을 잰다(헤더 + 데이터 2줄).
        # 제목 줄에도 'RAGEC 정답' 이 들어 있어 헤더는 '  qa_id' 로 집는다.
        anchors = [("  qa_id", "RAGEC 정답"),
                   ("  1 ", "E4 Missed Retrieval"),
                   ("  2 ", "E13 Misinterpretation")]
        starts = set()
        for prefix, column in anchors:
            line = next(l for l in out if l.startswith(prefix))
            starts.add(_width(line[:line.index(column)]))
        self.assertEqual(len(starts), 1, f"{starts}\n" + "\n".join(out[-6:]))

    def test_width_counts_hangul_as_two_columns(self):
        self.assertEqual(_width("카테고리"), 8)
        self.assertEqual(_width("qa_id"), 5)


class MappingIntegrityTest(unittest.TestCase):
    """대조표(docs/ragec_label_mapping.md)가 정본이고 이 표는 그 구현이다."""

    def test_every_mapped_label_is_a_real_label(self):
        import pathlib
        import re
        md = pathlib.Path("tests/diagnose_grid/LABELS.md").read_text(encoding="utf-8")
        real = set(re.findall(r"^\| `([a-z_]+)` \|", md, re.M))
        self.assertTrue(real, "LABELS.md 에서 라벨을 못 읽었습니다")
        used = {label for labels in RAGEC_TO_OURS.values() for label in labels}
        self.assertEqual(used - real, set())

    def test_answer_key_categories_are_all_mapped(self):
        """정답지에 있는 카테고리가 표에 없으면 그 건들이 조용히 제외된다."""
        import json
        import pathlib
        path = pathlib.Path("data/ragec_answer_key.jsonl")
        if not path.exists():
            self.skipTest("data/ragec_answer_key.jsonl 없음 (tools/build_ragec_dataset.py 로 생성)")
        categories = {
            json.loads(line)["ragec_category"].strip()
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        }
        self.assertEqual(categories - set(RAGEC_TO_OURS), set())


if __name__ == "__main__":
    unittest.main()
