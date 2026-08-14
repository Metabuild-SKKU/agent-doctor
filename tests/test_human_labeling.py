"""
tests/test_human_labeling.py
라벨 시트 생성(make_label_sheet)과 대조 채점(score_human_labels) 검증.

이 경로가 존재하는 이유는 **RAGEC 라벨을 그대로 쓰면 출처가 어긋나서**다. 그 라벨은 사람이
*다른* 시스템의 관측을 보고 붙인 것이라, 검색기가 다른 우리 관측에는 성립하지 않을 수 있다
(실측 qa_id 2205: RAGEC 은 E4 '검색 실패' 인데 우리는 recall=1.00 으로 gold 를 다 찾았다).

그래서 여기서 고정하는 건 두 가지다.
  · 시트에 **우리 진단이 새지 않는가** — 새면 라벨러가 '판단' 이 아니라 '동의' 를 하게 되어
    검증이 통째로 무의미해진다. 이게 이 파일에서 제일 중요한 계약이다
  · 채점이 **관대해지지 않는가** — 제외 통은 정확도를 올리는 방향으로만 작동한다
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.make_label_sheet import (
    NO_LABEL, PRIMARY_FIELD, UNSURE, build_sheet, stratified_sample, to_entry,
)
from tools.score_human_labels import group_of, read_sheet, score


def _row(qa_id, *labels, failed=True, recall=0.0, **obs):
    base = {"f1": 0.1, "gold_chunk_hit": 0, "gold_chunk_total": 1,
            "gold_chunk_ids": ["d_chunk_1"], "retrieved_chunk_ids": ["d_chunk_9"],
            "search_mode": "dense", "reranker_status": "disabled"}
    base.update(obs)
    return {"qa_id": qa_id, "labels": list(labels), "failed": failed,
            "question": f"Q{qa_id}", "gold_answer": f"G{qa_id}", "answer": f"A{qa_id}",
            "recall_at_k": recall, "recall_basis": "span", "observations": base}


class SheetHidesOurDiagnosisTest(unittest.TestCase):
    """시트에 우리 라벨이 새면 라벨러가 '동의하는지' 를 답하게 된다 — 검증이 무의미해진다.

    이 파일에서 제일 중요한 계약이다.
    """

    ROW = _row("1", "retrieval_low_rank", "generation_hallucination")

    def test_entry_does_not_contain_our_labels(self):
        blob = json.dumps(to_entry(self.ROW), ensure_ascii=False)
        self.assertNotIn("retrieval_low_rank", blob)
        self.assertNotIn("generation_hallucination", blob)

    def test_sheet_does_not_contain_our_labels(self):
        blob = json.dumps(build_sheet([self.ROW]), ensure_ascii=False)
        self.assertNotIn("retrieval_low_rank", blob)
        self.assertNotIn("generation_hallucination", blob)

    def test_entry_keeps_the_observations_needed_to_judge(self):
        """지표는 진단이 아니라 관측이다 — 가리면 검색 계열 라벨을 구분할 수 없다."""
        entry = to_entry(_row("1", "retrieval_low_rank", recall=0.58,
                              gold_chunk_hit=1, gold_chunk_total=2))
        self.assertEqual(entry["검색_recall"], 0.58)
        self.assertEqual(entry["정답청크_검색됨"], "1/2")
        self.assertEqual(entry["질문"], "Q1")
        self.assertEqual(entry["검색방식"], "dense")

    def test_entry_shows_gold_chunk_ranks(self):
        """'몇 위였나' 가 없으면 low_rank 와 missing_gold 를 사람이 못 가른다."""
        hit = to_entry(_row("1", "x", gold_chunk_ids=["g1"],
                            retrieved_chunk_ids=["a", "b", "g1"]))
        self.assertEqual(hit["정답청크_순위"], "g1=3위")
        miss = to_entry(_row("1", "x", gold_chunk_ids=["g1"], retrieved_chunk_ids=["a"]))
        self.assertEqual(miss["정답청크_순위"], "g1=검색안됨")

    def test_fill_field_is_present_and_empty(self):
        self.assertEqual(to_entry(self.ROW)[PRIMARY_FIELD], "")

    def test_sheet_offers_the_two_escape_hatches(self):
        """둘이 없으면 라벨러가 억지로 하나를 고르고, 택소노미 구멍이 오답으로 뭉개진다."""
        guide = json.dumps(build_sheet([self.ROW])["_안내"], ensure_ascii=False)
        self.assertIn(NO_LABEL, guide)
        self.assertIn(UNSURE, guide)


class EndToEndFromRealReportTest(unittest.TestCase):
    """build_report → findings_from_report → 라벨 시트까지 **실제 경로**로 한 번 태운다.

    손으로 만든 덤프로만 테스트하면 중간 배선이 빠져도 통과한다 — 실제로 그랬다.
    report 는 observations 를 제대로 실었는데 findings_from_report 가 그걸 덤프로 옮기지
    않아, 30건 실측에서 라벨 시트가 **지표 없이** 나갔다. 시트에 recall·순위가 없으면
    사람이 검색 계열 라벨을 구분할 수 없어 라벨링 자체가 불가능해진다.
    """

    def _sheet_entry(self):
        from agents.eval.report import build_report
        from agents.eval.types import EvalRecord
        from core.schema import Finding, Probe
        from tools.score_ragec import findings_from_report

        probe = Probe(probe_id="probe_qa_7", question="질문?", source="taxonomy",
                      ground_truth="정답", gold_chunk_ids=["d_chunk_1"])
        record = EvalRecord(
            probe=probe,
            generated_answer="틀린 답",
            retrieved_chunk_ids=["d_chunk_9", "d_chunk_1"],
            retrieval_details={"search_mode": "dense", "reranker_status": "disabled"},
            findings=[Finding(finding_id="f1", type="retrieval_failure",
                              severity="warning", description="근거 문구",
                              label="retrieval_low_rank", affected_probes=["probe_qa_7"])],
        )
        report = build_report([record], iteration=1, mode=1)
        rows = findings_from_report(report, [probe])
        return rows[0], to_entry(rows[0])

    def test_observations_reach_the_sheet(self):
        row, entry = self._sheet_entry()
        self.assertTrue(row.get("observations"), "덤프에 observations 가 실리지 않았습니다")
        self.assertEqual(entry["정답청크_검색됨"], "1/1")
        self.assertEqual(entry["정답청크_순위"], "d_chunk_1=2위")
        self.assertEqual(entry["검색방식"], "dense")

    def test_sheet_from_real_report_still_hides_our_diagnosis(self):
        _row, entry = self._sheet_entry()
        blob = json.dumps(entry, ensure_ascii=False)
        self.assertNotIn("retrieval_low_rank", blob)
        self.assertNotIn("근거 문구", blob)


class StratifiedSamplingTest(unittest.TestCase):
    """무작위로 뽑으면 실측처럼 한 라벨로 쏠려(10건 중 5건이 low_rank) 희귀 라벨을 영영 못 잰다."""

    ROWS = ([_row(str(i), "retrieval_low_rank") for i in range(20)]
            + [_row("90", "generation_hallucination"), _row("91", "too_long_context")])

    def test_rare_labels_survive_the_sample(self):
        picked = stratified_sample(self.ROWS, limit=6)
        labels = {r["labels"][0] for r in picked}
        self.assertEqual(len(picked), 6)
        self.assertIn("generation_hallucination", labels)
        self.assertIn("too_long_context", labels)

    def test_successful_probes_are_not_sampled(self):
        """성공 probe 는 진단할 게 없어 라벨링 대상이 아니다."""
        rows = self.ROWS + [_row("99", failed=False)]
        self.assertNotIn("99", {r["qa_id"] for r in stratified_sample(rows, limit=0)})

    def test_sampling_is_reproducible(self):
        a = [r["qa_id"] for r in stratified_sample(self.ROWS, 6, seed=7)]
        b = [r["qa_id"] for r in stratified_sample(self.ROWS, 6, seed=7)]
        self.assertEqual(a, b)


class ScoringTest(unittest.TestCase):
    def _score(self, gold, *predicted, failed=True):
        return score([{"qa_id": "1", PRIMARY_FIELD: gold}],
                     [_row("1", *predicted, failed=failed)])

    def test_containment_hit(self):
        r = self._score("retrieval_low_rank", "retrieval_low_rank", "generation_hallucination")
        self.assertEqual((r["hit"], r["total"]), (1, 1))

    def test_top1_is_counted_separately(self):
        """포함만 보면 라벨을 남발할수록 점수가 오른다 — 남발 여부를 함께 봐야 해석된다."""
        r = self._score("retrieval_low_rank", "generation_hallucination", "retrieval_low_rank")
        self.assertEqual((r["hit"], r["top1"]), (1, 0))

    def test_wrong_label_counts_wrong(self):
        r = self._score("retrieval_low_rank", "generation_hallucination")
        self.assertEqual((r["hit"], r["total"]), (0, 1))

    def test_no_diagnosis_counts_wrong(self):
        """빼면 '아무 말 안 하면 안 틀린다' 가 된다."""
        r = self._score("retrieval_low_rank")
        self.assertEqual((r["hit"], r["total"], r["no_diagnosis"]), (0, 1, 1))

    def test_no_label_and_unsure_are_excluded_not_wrong(self):
        """맞출 수단이 없는 걸 오답으로 세면 수치가 거짓이 된다."""
        for answer in (NO_LABEL, UNSURE):
            r = self._score(answer, "retrieval_low_rank")
            self.assertEqual(r["total"], 0, answer)
            self.assertEqual(r["excluded"].get(answer), 1, answer)

    def test_probe_we_passed_is_excluded(self):
        r = self._score("retrieval_low_rank", failed=False)
        self.assertEqual(r["total"], 0)
        self.assertEqual(r["excluded"].get("우리는 성공"), 1)

    def test_blank_label_is_excluded(self):
        r = self._score("")
        self.assertEqual(r["total"], 0)
        self.assertEqual(r["excluded"].get("미기입"), 1)

    def test_stage_can_be_right_while_label_is_wrong(self):
        r = self._score("retrieval_missing_gold", "retrieval_low_rank")
        self.assertEqual(r["hit"], 0)
        self.assertEqual((r["stage_hit"], r["stage_total"]), (1, 1))

    def test_stage_denominator_matches_the_label_axis(self):
        """미진단이나 단계 개념 없는 라벨만 낸 경우를 단계 분모에서 빼면 위로 편향된다 —
        논문 57.8% 옆에 병기하는 값이라 어긋나면 안 된다."""
        no_label = self._score("retrieval_low_rank")               # 우리가 라벨 0개
        self.assertEqual((no_label["total"], no_label["stage_total"]), (1, 1))
        self.assertEqual(no_label["stage_hit"], 0)
        c_only = self._score("retrieval_low_rank", "too_long_context")   # C그룹만 냄
        self.assertEqual((c_only["stage_total"], c_only["stage_hit"]), (1, 0))

    def test_group_axis_is_coarser_than_stage(self):
        """A그룹 안에 검색·청킹·리랭킹이 다 있어 그룹은 단계보다 관대하다."""
        r = self._score("chunking_overchunking", "retrieval_low_rank")
        self.assertEqual((r["group_hit"], r["group_total"]), (1, 1))
        self.assertEqual(r["stage_hit"], 0)      # Chunking ≠ Retrieval

    def test_confusion_pairs_are_recorded(self):
        r = self._score("retrieval_missing_gold", "generation_hallucination")
        self.assertEqual(r["confusion"][("retrieval_missing_gold",
                                         "generation_hallucination")], 1)


class GroupMappingTest(unittest.TestCase):
    def test_every_group_resolves(self):
        cases = {"retrieval_low_rank": "A", "chunking_overchunking": "A",
                 "reranker_low_precision": "A", "generation_hallucination": "B",
                 "too_long_context": "C", "context_noise_interference": "C",
                 "bad_gold_answer": "D", "corpus_gap": "D"}
        for label, expected in cases.items():
            self.assertEqual(group_of(label), expected, label)


class ResultSavingTest(unittest.TestCase):
    """사람 시간이 몇 시간 들어간 산출물이라 콘솔에만 두면 안 된다.

    창을 닫으면 사라지고, 나중에 "그때 몇 % 였지" 를 확인하려면 라벨링을 다시 해야 한다.
    """

    def _save(self):
        import pathlib
        import tempfile
        from tools.score_human_labels import format_report, save_result
        result = score([{"qa_id": "1", PRIMARY_FIELD: "retrieval_missing_gold"}],
                       [_row("1", "retrieval_low_rank")])
        tmp = tempfile.mkdtemp()
        txt, js = save_result(result, pathlib.Path(tmp), format_report(result))
        return (pathlib.Path(txt).read_text(encoding="utf-8"),
                json.loads(pathlib.Path(js).read_text(encoding="utf-8")))

    def test_text_report_is_written(self):
        text, _ = self._save()
        self.assertIn("사람 라벨 대조", text)
        self.assertIn("retrieval_missing_gold", text)

    def test_json_carries_the_confusion_pairs(self):
        """혼동 쌍은 튜플 키라 그냥 담으면 직렬화가 죽는다 — 어디가 어긋났는지가
        제일 쓸모 있는 정보라 리스트로 펴서 싣는다."""
        _text, data = self._save()
        self.assertEqual(data["confusion"],
                         [{"사람": "retrieval_missing_gold",
                           "우리": "retrieval_low_rank", "건수": 1}])

    def test_json_keeps_the_headline_numbers(self):
        """실행 간 비교(개선 전후)를 하려면 파싱 가능한 형태여야 한다."""
        _text, data = self._save()
        for key in ("total", "hit", "top1", "stage_hit", "per_label", "측정시각"):
            self.assertIn(key, data, key)


class SheetRoundTripTest(unittest.TestCase):
    """사람이 손으로 채우는 파일이라 읽기가 관대해야 한다."""

    def _read(self, text):
        import pathlib
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "label_sheet.json"
            path.write_text(text, encoding="utf-8")
            return read_sheet(str(path))

    def test_generated_sheet_round_trips(self):
        sheet = build_sheet([_row("1", "retrieval_low_rank")])
        sheet["항목"][0][PRIMARY_FIELD] = "retrieval_missing_gold"
        rows = self._read(json.dumps(sheet, ensure_ascii=False, indent=2))
        self.assertEqual(rows[0]["qa_id"], "1")
        self.assertEqual(rows[0][PRIMARY_FIELD], "retrieval_missing_gold")

    def test_bare_array_is_accepted(self):
        """안내문을 지우고 항목 배열만 남겨 저장해도 읽힌다."""
        rows = self._read(json.dumps([{"qa_id": "1", PRIMARY_FIELD: "x"}],
                                     ensure_ascii=False))
        self.assertEqual(rows[0]["qa_id"], "1")

    def test_bom_is_absorbed(self):
        """편집기에 따라 BOM 이 붙는다 — utf-8 로 읽으면 첫 글자가 깨져 파싱이 죽는다."""
        rows = self._read("﻿" + json.dumps([{"qa_id": "1", PRIMARY_FIELD: "x"}]))
        self.assertEqual(rows[0]["qa_id"], "1")

    def test_broken_json_says_where_to_fix(self):
        """손으로 채우다 쉼표가 남는 일이 흔하다 — 줄 번호를 알려줘야 고칠 수 있다."""
        with self.assertRaises(SystemExit) as ctx:
            self._read('{"항목": [{"qa_id": "1",}]}')
        self.assertIn("JSON 형식이 깨졌습니다", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
