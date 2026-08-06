# -*- coding: utf-8 -*-
"""리플레이 모드(agents/eval/replay.py) 단위 테스트.

외부 API 없이 돈다 - LLM 경로는 evaluate_real_track을 모킹하고, 비LLM 경로는
EVAL_ENABLE_LLM=0 을 강제해 폴백(빈 RAGAS)으로 검증한다.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.eval import replay
from agents.eval.log_intake import parse_record
from agents.eval.metrics_basic import char_f1
from core.schema import DiagnosticReport

NO_LLM = {"EVAL_ENABLE_LLM": "0"}


def _log(question="공제 한도는?", answer="700만원입니다", contexts=None, **extra):
    obj = {"question": question, "answer": answer}
    if contexts is not None:
        obj["contexts"] = contexts
    obj.update(extra)
    return parse_record(obj)


class TestBuildReplayRecords(unittest.TestCase):
    def test_triad_record_fields(self):
        logs = [_log(contexts=[
            {"text": "청크 원문", "chunk_id": "doc3_c12"},
            "아이디 없는 원문",
        ])]
        rec = replay.build_replay_records(logs)[0]
        self.assertEqual(rec.probe.source, "user_log")
        self.assertEqual(rec.probe.question, "공제 한도는?")
        self.assertEqual(rec.generated_answer, "700만원입니다")
        self.assertEqual(rec.retrieved_context, ["청크 원문", "아이디 없는 원문"])
        # chunk_id 없는 컨텍스트는 합성 id (진단 재료 아님, 자리 채움)
        self.assertEqual(rec.retrieved_chunk_ids[0], "doc3_c12")
        self.assertTrue(rec.retrieved_chunk_ids[1].startswith("ext_0000_ctx_"))

    def test_recall_is_unmeasured_sentinel(self):
        # 기본값 0.0 을 두면 report 가 "진짜 0점"으로 집계한다 - 반드시 -1
        rec = replay.build_replay_records([_log(contexts=["ctx"])])[0]
        self.assertEqual(rec.recall_at_k, -1.0)

    def test_ground_truth_passthrough_and_rule_f1(self):
        logs = [_log(answer="700만원입니다", contexts=["ctx"],
                     ground_truth="700만원입니다")]
        rec = replay.build_replay_records(logs)[0]
        self.assertEqual(rec.probe.ground_truth, "700만원입니다")
        self.assertAlmostEqual(
            rec.f1_score, char_f1("700만원입니다", "700만원입니다"))
        self.assertGreater(rec.f1_score, 0.9)

    def test_no_ground_truth(self):
        rec = replay.build_replay_records([_log(contexts=["ctx"])])[0]
        self.assertIsNone(rec.probe.ground_truth)
        self.assertEqual(rec.f1_score, 0.0)

    def test_gold_contexts_carried_via_probe_metadata(self):
        # 공유 스키마(Probe/EvalRecord)를 바꾸지 않고 metadata 로 전달하는 규약
        rec = replay.build_replay_records(
            [_log(contexts=["ctx"], gold_contexts=["근거 문단"])])[0]
        self.assertEqual(rec.probe.metadata["gold_contexts"], ["근거 문단"])


class TestRunReplayWithoutLLM(unittest.TestCase):
    @patch.dict(os.environ, NO_LLM)
    def test_report_builds_without_llm(self):
        records = replay.build_replay_records(
            [_log(contexts=["ctx"]), _log(question="실업률은?", answer="3%다")])
        report = replay.run_replay(records)
        self.assertIsInstance(report, DiagnosticReport)
        # RAGAS 미측정 - 지표가 끼어들면 안 된다
        self.assertNotIn("faithfulness", report.ragas_scores)
        # recall 센티널(-1)이 평균에 섞이면 안 된다
        self.assertNotIn("mean_recall_at_k", report.ragas_scores)

    @patch.dict(os.environ, NO_LLM)
    def test_gt_records_report_rule_f1(self):
        records = replay.build_replay_records(
            [_log(contexts=["ctx"], ground_truth="700만원입니다")])
        report = replay.run_replay(records)
        self.assertIn("mean_f1", report.ragas_scores)
        self.assertGreater(report.ragas_scores["mean_f1"], 0.9)
        self.assertIsNotNone(report.overall_score)


class TestRunReplayMockedRagas(unittest.TestCase):
    def _run(self, ragas_result):
        records = replay.build_replay_records([_log(contexts=["ctx"])])
        with patch.object(replay, "llm_eval_enabled", return_value=True), \
             patch.object(replay, "_judge", return_value=object()), \
             patch.object(replay, "evaluate_real_track",
                          return_value=ragas_result):
            report = replay.run_replay(records)
        return records, report

    def test_ragas_fills_scores_and_retrieval_axis(self):
        records, report = self._run(
            {"faithfulness": 0.4, "response_relevancy": 0.9})
        self.assertEqual(records[0].ragas["faithfulness"], 0.4)
        # faithfulness -> 검색축 seam (reliability 의 recall(-1) 오염 방지)
        self.assertEqual(records[0].retrieval_axis, 0.4)
        # overall = 존재 지표만 재정규화: (0.30*0.4 + 0.20*0.9) / 0.50
        self.assertAlmostEqual(report.overall_score, 0.6, places=6)
        self.assertAlmostEqual(report.ragas_scores["faithfulness"], 0.4)

    def test_ragas_failure_falls_back(self):
        records = replay.build_replay_records([_log(contexts=["ctx"])])
        with patch.object(replay, "llm_eval_enabled", return_value=True), \
             patch.object(replay, "_judge", return_value=object()), \
             patch.object(replay, "evaluate_real_track",
                          side_effect=RuntimeError("boom")):
            report = replay.run_replay(records)
        self.assertEqual(records[0].ragas, {})
        self.assertTrue(records[0].ragas_done)
        self.assertIsInstance(report, DiagnosticReport)


class TestDiagnoseExternalLog(unittest.TestCase):
    @patch.dict(os.environ, NO_LLM)
    def test_end_to_end_from_jsonl(self):
        lines = [
            json.dumps({"question": "q1", "answer": "a1",
                        "contexts": ["c1"]}, ensure_ascii=False),
            json.dumps({"question": "q2", "answer": "a2",
                        "contexts": ["c2"]}, ensure_ascii=False),
            "깨진 줄 {",
        ]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "log.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            report, cap, errors = replay.diagnose_external_log(path)
        self.assertIsNotNone(report)
        self.assertEqual(cap["tier"], "triad")
        self.assertEqual(len(errors), 1)

    @patch.dict(os.environ, NO_LLM)
    def test_empty_log_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "log.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
            report, cap, errors = replay.diagnose_external_log(path)
        self.assertIsNone(report)
        self.assertEqual(cap["tier"], "none")

    @patch.dict(os.environ, NO_LLM)
    def test_qa_only_file_is_refused_by_default(self):
        # 파일 단위 게이트(docs §3) - contexts 없는 로그에 "진단"을 내주지 않는다
        lines = [json.dumps({"question": f"q{i}", "answer": "a"}) for i in range(3)]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "log.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            report, cap, _ = replay.diagnose_external_log(path)
            self.assertIsNone(report)
            self.assertEqual(cap["tier"], "qa_only")
            # 옵트인하면 동문서답 검사용으로는 돌 수 있다
            report, _, _ = replay.diagnose_external_log(path, allow_qa_only=True)
            self.assertIsNotNone(report)

    @patch.dict(os.environ, NO_LLM)
    def test_limit_caps_records(self):
        lines = [json.dumps({"question": f"q{i}", "answer": "a",
                             "contexts": ["c"]}) for i in range(5)]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "log.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            report, cap, _ = replay.diagnose_external_log(path, limit=2)
        self.assertEqual(cap["records"], 5)          # 적재 판정은 전체 기준
        self.assertEqual(
            report.ragas_scores["outcome_distribution"]["ok"], 2)  # 진단은 2건


if __name__ == "__main__":
    unittest.main()
