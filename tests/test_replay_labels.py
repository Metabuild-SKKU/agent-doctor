# -*- coding: utf-8 -*-
"""ext_ 리플레이 라벨(agents/eval/replay_labels.py) 단위 테스트.

전부 오프라인 - ragas 점수는 레코드에 직접 채워 넣는다(LLM judge 는 이 모듈의
관심사가 아님: 문턱 규칙과 확정/예비 판정, 겹침 신호가 대상이다).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.eval import replay, replay_labels
from agents.eval.log_intake import parse_record
from agents.eval.replay_labels import (
    EXT_RECOMMENDATIONS, diagnose_replay_record, gold_context_recall,
    recommendation_ids,
)

GOLD = "연금저축 세액공제 한도는 연 700만원이다."


def _record(contexts=None, gold_contexts=None, ragas=None, ground_truth=None):
    obj = {"question": "공제 한도는?", "answer": "700만원입니다"}
    if contexts is not None:
        obj["contexts"] = contexts
    if gold_contexts is not None:
        obj["gold_contexts"] = gold_contexts
    if ground_truth is not None:
        obj["ground_truth"] = ground_truth
    rec = replay.build_replay_records([parse_record(obj)])[0]
    rec.ragas = ragas or {}
    return rec


class TestGoldContextRecall(unittest.TestCase):
    def test_exact_containment_scores_high(self):
        rec = _record(contexts=[f"머리말. {GOLD} 꼬리말."], gold_contexts=[GOLD])
        self.assertGreater(gold_context_recall(rec), 0.95)

    def test_whitespace_differences_tolerated(self):
        messy = GOLD.replace(" ", "\n  ")
        rec = _record(contexts=[messy], gold_contexts=[GOLD])
        self.assertGreater(gold_context_recall(rec), 0.9)

    def test_unrelated_context_scores_low(self):
        rec = _record(contexts=["고용보험 가입 절차는 고용센터에 신고한다." * 3],
                      gold_contexts=[GOLD])
        self.assertLess(gold_context_recall(rec), 0.6)

    def test_none_without_gold_or_contexts(self):
        self.assertIsNone(gold_context_recall(_record(contexts=["ctx"])))
        self.assertIsNone(gold_context_recall(_record(gold_contexts=[GOLD])))


class TestExtLabels(unittest.TestCase):
    def _labels(self, rec):
        return {f.label: f for f in diagnose_replay_record(rec)}

    def test_no_metrics_no_findings(self):
        self.assertEqual(diagnose_replay_record(_record(contexts=["ctx"])), [])

    def test_off_topic_fires_on_low_relevancy(self):
        labels = self._labels(_record(contexts=["ctx"],
                                      ragas={"response_relevancy": 0.2}))
        self.assertIn("ext_answer_off_topic", labels)
        self.assertTrue(labels["ext_answer_off_topic"].confirmed)

    def test_hallucination_confirmed_when_gold_was_retrieved(self):
        rec = _record(contexts=[GOLD], gold_contexts=[GOLD],
                      ragas={"faithfulness": 0.1, "response_relevancy": 0.9})
        labels = self._labels(rec)
        self.assertIn("ext_generation_hallucination", labels)
        self.assertTrue(labels["ext_generation_hallucination"].confirmed)

    def test_hallucination_preliminary_when_gold_missing_from_retrieval(self):
        rec = _record(contexts=["무관한 문서 내용" * 5], gold_contexts=[GOLD],
                      ragas={"faithfulness": 0.1, "response_relevancy": 0.9})
        labels = self._labels(rec)
        self.assertFalse(labels["ext_generation_hallucination"].confirmed)

    def test_hallucination_preliminary_without_gold_contexts(self):
        rec = _record(contexts=["ctx"],
                      ragas={"faithfulness": 0.1, "response_relevancy": 0.9})
        labels = self._labels(rec)
        self.assertFalse(labels["ext_generation_hallucination"].confirmed)

    def test_hallucination_yields_to_off_topic_when_both_low(self):
        labels = self._labels(_record(contexts=["ctx"],
                                      ragas={"faithfulness": 0.1,
                                             "response_relevancy": 0.1}))
        self.assertIn("ext_answer_off_topic", labels)
        self.assertNotIn("ext_generation_hallucination", labels)

    def test_context_overflow(self):
        rec = _record(contexts=["가" * 7000],
                      ragas={"faithfulness": 0.1, "response_relevancy": 0.9})
        self.assertIn("ext_context_overflow", self._labels(rec))

    def test_no_overflow_under_limit(self):
        rec = _record(contexts=["가" * 100],
                      ragas={"faithfulness": 0.1, "response_relevancy": 0.9})
        self.assertNotIn("ext_context_overflow", self._labels(rec))

    def test_grounded_but_wrong_needs_high_faith_low_correctness(self):
        rec = _record(contexts=["ctx"], ground_truth="정답",
                      ragas={"faithfulness": 0.9, "answer_correctness": 0.1})
        self.assertIn("ext_grounded_but_wrong", self._labels(rec))

    def test_finding_shape(self):
        rec = _record(contexts=["ctx"], ragas={"response_relevancy": 0.2})
        f = diagnose_replay_record(rec)[0]
        self.assertEqual(f.finding_id, f"{rec.probe.probe_id}:{f.label}")
        self.assertEqual(f.affected_probes, [rec.probe.probe_id])
        self.assertEqual(f.metadata["group"], "EXT")


class TestRecommendations(unittest.TestCase):
    def test_every_ext_label_maps_to_real_rules_entry(self):
        for label, entry in EXT_RECOMMENDATIONS.items():
            self.assertTrue(entry.get("prescriptions"),
                            f"{label} 의 rules 참조에 prescriptions 없음")
            self.assertTrue(recommendation_ids(label))

    def test_reference_not_copy(self):
        # 재참조 원칙 - rules 원본 dict 와 같은 객체여야 문구 원본이 하나로 유지된다
        from agents.optimize.rules import LABEL_TO_PRESCRIPTIONS
        self.assertIs(EXT_RECOMMENDATIONS["ext_generation_hallucination"],
                      LABEL_TO_PRESCRIPTIONS["generation_hallucination"])


if __name__ == "__main__":
    unittest.main()
