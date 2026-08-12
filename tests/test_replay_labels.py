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

    def test_short_gold_is_excluded_from_measurement(self):
        # "700만원" 같은 정답 조각은 무관한 문맥에 통째로 우연 포함될 수 있어
        # 근거로 안 친다 - 전부 짧으면 None(판정 재료 없음) → 환각은 예비 유지
        rec = _record(contexts=["대출 한도는 통상 연 700만원 수준이다."],
                      gold_contexts=["700만원"])
        self.assertIsNone(gold_context_recall(rec))

    def test_scattered_char_matches_do_not_count(self):
        # 낱글자·짧은 조각 우연 일치 누적 차단(_MIN_MATCH_BLOCK)
        from agents.eval.replay_labels import _coverage, _squash
        self.assertEqual(
            _coverage(_squash("700만원"), _squash("3,700개의 만장일치 원안을 처리했다")),
            0.0)


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

    def test_grounded_but_wrong_requires_relevancy_gate(self):
        """동문서답(rel 낮음)이면 answer_correctness 가 낮은 건 그 결과다 - 근거
        자체(코퍼스)가 틀렸다는 뜻이 아니다. corpus_gap 수동 처방을 잘못 확정하면 안 된다."""
        rec = _record(contexts=["ctx"], ground_truth="정답",
                      ragas={"response_relevancy": 0.2, "faithfulness": 0.9,
                             "answer_correctness": 0.1})
        self.assertNotIn("ext_grounded_but_wrong", self._labels(rec))

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


class AbstentionIsNotAFailureTests(unittest.TestCase):
    """기권한 답변은 소견을 내지 않는다.

    "자료에서 확인되지 않습니다"는 실패가 아니라 우리가 권고하는 처방
    (strengthen_abstention / require_citation)의 목표다. 그런데 기권문은 질문을
    되풀이하지 않아 relevancy 가 0 에 가깝게 나오고, 그대로 두면 동문서답으로
    오진된다 — 실측에서 처방 적용본의 기권 5건이 전부 ext_answer_off_topic 으로
    잡혀 처방 효과가 오히려 나빠진 것처럼 보였다.
    """

    def _rec(self, answer, rel=0.0, faith=0.0):
        rec = _record(contexts=["사내 규정 안내 문서 본문"],
                      ragas={"response_relevancy": rel, "faithfulness": faith})
        rec.generated_answer = answer
        return rec

    def test_abstained_answer_yields_no_finding(self):
        rec = self._rec("자료에서 확인되지 않습니다. 참고 자료에는 해당 내용이 없습니다.")
        self.assertEqual(diagnose_replay_record(rec), [])

    def test_non_abstained_low_relevancy_still_flagged(self):
        """기권이 아니면 종전대로 동문서답을 잡아야 한다(가드가 과하게 먹지 않는지)."""
        rec = self._rec("연차 휴가는 연 15일입니다.", rel=0.1, faith=0.9)
        labels = [f.label for f in diagnose_replay_record(rec)]
        self.assertIn("ext_answer_off_topic", labels)

    def test_abstention_marker_covers_common_phrasings(self):
        """'확인할 수 없'만 있어 '확인되지 않습니다'류를 통째로 놓쳤다 —
        인용 강제 프롬프트를 쓰는 RAG 가 가장 흔히 내는 형태다."""
        from agents.eval.metrics_basic import is_abstention
        for text in ("자료에서 확인되지 않습니다.",
                     "제공된 자료에는 포함되어 있지 않습니다.",
                     "관련 내용을 찾을 수 없습니다.",
                     "명시되어 있지 않습니다.",
                     "답변드리기 어렵습니다.",
                     "관련 정보가 부족합니다.",
                     "해당 내용은 문서에서 다루지 않습니다."):
            # 이 확장 마커는 replay 전용(extended=True) - 메인 파이프라인 기본값은 좁은
            # 세트를 쓴다(근거 있는 정상 부정 답변을 기권으로 오진하지 않도록).
            self.assertTrue(is_abstention(text, extended=True), text)

    def test_normal_answers_are_not_abstention(self):
        """과검출 방지 — 정상 답변을 기권으로 보면 진단이 통째로 침묵한다."""
        from agents.eval.metrics_basic import is_abstention
        for text in ("연차 휴가는 연 15일이 부여됩니다.",
                     "통신비는 월 5만원이 지원됩니다."):
            self.assertFalse(is_abstention(text), text)


class RetrievalAxisLabelTests(unittest.TestCase):
    """검색축 라벨 3종 — 골든셋이 있으면 '검색이 근거를 가져왔나'가 실측된다.

    이 신호가 없으면 검색 결함이 답변축 지표로만 드러나 생성 문제로 오진된다
    (실측: offtopic 로그가 동문서답·근거부족으로만 잡혔다)."""

    def _rec(self, answer="700만원입니다", *, gold=None, ctx=None, prec=None,
             rel=0.9, faith=0.9):
        ragas = {"response_relevancy": rel, "faithfulness": faith}
        if prec is not None:
            ragas["context_precision"] = prec
        rec = _record(contexts=ctx or ["무관한 사내 규정 본문입니다."],
                      gold_contexts=gold, ragas=ragas)
        rec.generated_answer = answer
        return rec

    def test_irrelevant_retrieval_is_flagged(self):
        rec = self._rec(prec=0.2)
        labels = [f.label for f in diagnose_replay_record(rec)]
        self.assertIn("ext_retrieval_irrelevant", labels)

    def test_good_retrieval_is_not_flagged(self):
        rec = self._rec(prec=1.0)
        labels = [f.label for f in diagnose_replay_record(rec)]
        self.assertNotIn("ext_retrieval_irrelevant", labels)

    def test_wrongful_abstention_when_evidence_existed(self):
        """근거가 있었는데 기권했다 — 답할 수 있었으므로 사용자 손실이다."""
        rec = self._rec("자료에서 확인되지 않습니다.", gold=[GOLD],
                        ctx=[f"머리말 {GOLD} 꼬리말"])
        labels = [f.label for f in diagnose_replay_record(rec)]
        self.assertIn("ext_wrongful_abstention", labels)

    def test_starved_abstention_when_evidence_missing(self):
        """근거가 없어서 기권했다 — 생성 탓이 아니라 검색/코퍼스 탓이다."""
        rec = self._rec("자료에서 확인되지 않습니다.", prec=0.1)
        labels = [f.label for f in diagnose_replay_record(rec)]
        self.assertIn("ext_retrieval_starved_abstention", labels)

    def test_abstention_without_axis_stays_silent(self):
        """검색축 미측정이면 기권을 판정하지 않는다 — 놓치는 것이 오진보다 낫다."""
        rec = self._rec("자료에서 확인되지 않습니다.")
        self.assertEqual(diagnose_replay_record(rec), [])

    def test_conflicting_signals_take_the_worse(self):
        """겹침은 gold 하나만 걸려도 1.00 이 된다 — top_k 를 넉넉히 잡아 무관한
        청크가 잔뜩 섞여도 정상으로 읽힌다(실측 offtopic: 겹침 1.00 / precision 0.20).
        검색 품질을 묻는 자리에서는 나쁜 쪽 신호를 믿어야 한다."""
        rec = self._rec(gold=[GOLD], ctx=[f"머리말 {GOLD} 꼬리말"], prec=0.2)
        labels = [f.label for f in diagnose_replay_record(rec)]
        self.assertIn("ext_retrieval_irrelevant", labels)

    def test_off_topic_is_preliminary_when_retrieval_failed(self):
        """검색이 무관한 걸 가져와 그 여파로 relevancy 도 낮아진 경우, off_topic 을
        독립 확정으로 보면 안 된다 - 검색을 고치면 사라질 소견이라 확정이 과하다."""
        rec = self._rec(rel=0.2, prec=0.2)
        findings = {f.label: f.confirmed for f in diagnose_replay_record(rec)}
        self.assertIn("ext_answer_off_topic", findings)
        self.assertFalse(findings["ext_answer_off_topic"])

    def test_off_topic_stays_confirmed_when_retrieval_is_fine(self):
        """검색이 멀쩡한데 답이 동문서답이면 여전히 확정이어야 한다(과한 강등 방지)."""
        rec = self._rec(rel=0.2, prec=1.0)
        findings = {f.label: f.confirmed for f in diagnose_replay_record(rec)}
        self.assertTrue(findings["ext_answer_off_topic"])

    def test_wrongful_not_starved_when_overlap_says_evidence_present(self):
        """겹침 1.0 인데 precision 만 낮으면(top_k 를 넉넉히 잡아 무관한 청크가 섞인
        경우), 기권 분기는 precision 이 아니라 겹침을 믿어야 한다 - '검색이 근거를
        못 가져왔다'가 사실과 다르다. irrelevant 라벨(위 테스트)과 같은 입력이지만
        묻는 질문이 다르다(검색 품질 vs 근거 도달 여부)."""
        rec = self._rec("자료에서 확인되지 않습니다.", gold=[GOLD],
                        ctx=[f"머리말 {GOLD} 꼬리말"], prec=0.2)
        labels = [f.label for f in diagnose_replay_record(rec)]
        self.assertIn("ext_wrongful_abstention", labels)
        self.assertNotIn("ext_retrieval_starved_abstention", labels)


if __name__ == "__main__":
    unittest.main()
