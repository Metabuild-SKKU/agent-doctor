"""
tests/test_bad_gold_chunk.py
골드 '청크' 라벨 오류(bad_gold_chunk) 탐지·억제·점수 제외 검증.

문제(1번): 정답 텍스트는 맞는데 근거로 지정된 골드 청크가 엉뚱한 곳을 가리키면(표 문서에서
같은 값이 여러 행에 나와 오매칭되는 등), 검색은 실제 근거를 찾아 정답을 냈는데도 recall=0 이
되어 retrieval_low_rank·generation_misinterpretation 로 오진되고 헛처방을 부른다.

판별 열쇠: '실제 답이 맞았나(_f1_ok)'. 답은 맞고(f1↑) 골드론 못 맞추면(oracle 실패) 원인은
검색·생성이 아니라 골드 청크 라벨이다. 이때 경쟁 슬롯의 거짓 원인을 억제하고 점수에서 제외한다.
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import Probe, Finding
from agents.eval import metrics_common, diagnose, report
from agents.eval.types import EvalRecord, Mode


def _rec(*, f1, oracle_f1, faith, oracle_answer="오라클 답", recall=0.0):
    """bad_gold_chunk 가 읽는 필드만 채운 record(지표 계산 없이 판정만 검증)."""
    probe = Probe(probe_id="p1", question="질문", source="taxonomy",
                  ground_truth="332cm", gold_chunk_ids=["g_a"])
    rec = EvalRecord(probe=probe, generated_answer="332cm", oracle_answer=oracle_answer)
    rec.recall_at_k = recall
    rec.f1_score = f1
    rec.oracle_f1 = oracle_f1
    rec.ragas = {"faithfulness": faith}
    rec.ragas_done = True
    rec.oracle_ragas = {}
    rec.oracle_ragas_done = True
    return rec


class BadGoldChunkLabelTest(unittest.TestCase):
    # faithfulness(real)는 DEEP+ 에서만 측정된다 — bad_gold_chunk 도 실제 진단이 도는
    # deep/full 모드에서만 발동한다. 그 모드에서 판정을 고정한다.
    def setUp(self):
        metrics_common.set_mode(Mode.DEEP)

    def tearDown(self):
        metrics_common.set_mode(Mode.FAST)

    def test_fires_when_answer_correct_but_gold_insufficient(self):
        # 실제 답 정답(f1↑) + 골드론 실패(oracle↓) + 답이 검색 근거에 붙음(faith↑) → 골드 청크 오라벨
        finding = diagnose.bad_gold_chunk(_rec(f1=1.0, oracle_f1=0.0, faith=1.0))
        self.assertIsNotNone(finding)
        self.assertEqual(finding.label, "bad_gold_chunk")
        self.assertTrue(finding.confirmed)

    def test_silent_when_oracle_ok(self):
        # 골드로 답이 나오면 골드는 정상 → 침묵
        self.assertIsNone(diagnose.bad_gold_chunk(_rec(f1=1.0, oracle_f1=1.0, faith=1.0)))

    def test_silent_when_real_answer_wrong(self):
        # 실제 답도 틀리면 골드 문제로 단정 못 함(진짜 실패 영역) → 침묵
        self.assertIsNone(diagnose.bad_gold_chunk(_rec(f1=0.0, oracle_f1=0.0, faith=1.0)))

    def test_silent_when_answer_ungrounded(self):
        # faith 낮음 = 답이 검색 근거에 안 붙음(parametric 등) → 골드 단정 불가 → 침묵
        self.assertIsNone(diagnose.bad_gold_chunk(_rec(f1=1.0, oracle_f1=0.0, faith=0.0)))

    def test_silent_when_no_gold_context(self):
        # 골드 컨텍스트가 없으면(코퍼스 결손 등) 청크 오라벨로 단정 불가 → 침묵
        rec = _rec(f1=1.0, oracle_f1=0.0, faith=1.0, oracle_answer=None)
        self.assertIsNone(diagnose.bad_gold_chunk(rec))


class DiagnoseShortCircuitTest(unittest.TestCase):
    """diagnose 단락: 골드 청크 오라벨이면 거짓 원인(retrieval_low_rank·generation_*)을 막고
    bad_gold_chunk 하나만 남긴다."""

    def setUp(self):
        metrics_common.set_mode(Mode.DEEP)
        # tier2 자원(chunks 등)은 전역이라 앞선 모듈이 남긴 코퍼스가 그대로 보인다.
        # 그러면 이 가짜 record 의 gold 가 '코퍼스에 없음' 으로 읽혀 corpus_gap 이 먼저
        # 발동하고 단락 검증이 무너진다 — 판정만 보게 자원을 비우고 시작한다.
        metrics_common.set_context()

    def tearDown(self):
        metrics_common.set_mode(Mode.FAST)
        metrics_common.set_context()

    def test_mislabel_suppresses_false_causes(self):
        # recall=0 이라 정상 경로면 retrieval_low_rank 가 붙어야 하지만, 골드 오라벨이 감지되면
        # 단락되어 그 거짓 원인이 안 붙는다. 지표·RAGAS 재계산은 주입값을 보존하려 no-op 처리.
        rec = _rec(f1=1.0, oracle_f1=0.0, faith=1.0, recall=0.0)
        with patch.object(diagnose, "_compute_metrics"), \
             patch.object(diagnose, "_compute_ragas_real"), \
             patch.object(diagnose, "_compute_ragas_oracle"):
            findings = diagnose.diagnose(rec, mode=int(Mode.DEEP))
        self.assertEqual([f.label for f in findings], ["bad_gold_chunk"])

    def test_corpus_gap_preempts_bad_gold_chunk(self):
        # 리뷰 지적(Medium): 골드가 코퍼스에 없으면(corpus_gap) 단락하지 않고 양보한다 —
        # 없는 청크는 '재지정' 불가라 additive corpus_gap 이 그 사실을 보고해야 한다.
        from core.schema import Finding
        sentinel = Finding(finding_id="cg", type="gap", severity="warning", description="d",
                           label="corpus_gap", confirmed=True, affected_probes=["p1"])
        rec = _rec(f1=1.0, oracle_f1=0.0, faith=1.0, recall=0.0)
        with patch.object(diagnose, "_compute_metrics"), \
             patch.object(diagnose, "_compute_ragas_real"), \
             patch.object(diagnose, "_compute_ragas_oracle"), \
             patch.object(diagnose, "_corpus_gap_premise", return_value=True), \
             patch.object(diagnose, "_retrieval_fixable", return_value=False), \
             patch.object(diagnose, "corpus_gap", return_value=sentinel), \
             patch.object(diagnose, "corpus_gap_partial_hop", return_value=None), \
             patch.object(diagnose, "_generation_failed", return_value=False), \
             patch.object(diagnose, "_context_failed", return_value=False), \
             patch.object(diagnose, "generation_parametric_overreliance", return_value=None), \
             patch.object(diagnose, "generation_abstention_failure", return_value=None):
            labels = [f.label for f in diagnose.diagnose(rec, mode=int(Mode.DEEP))]
        self.assertIn("corpus_gap", labels)            # additive corpus_gap 보존
        self.assertNotIn("bad_gold_chunk", labels)     # 단락 안 탐


class GoldLabelingErrorScoringTest(unittest.TestCase):
    """점수 제외는 넓히되(정답 텍스트+청크 둘 다), 재생성 대상은 답 오류만 유지한다."""

    def _labeled(self, label, confirmed=True):
        probe = Probe(probe_id="p", question="q", source="taxonomy", ground_truth="gt")
        rec = EvalRecord(probe=probe)
        rec.f1_score, rec.recall_at_k = 0.0, 1.0
        rec.findings = [Finding(finding_id="p", type="gap", severity="warning",
                                description="d", label=label, confirmed=confirmed,
                                affected_probes=["p"])]
        return rec

    def _good(self, pid):
        probe = Probe(probe_id=pid, question="q", source="taxonomy", ground_truth="gt")
        rec = EvalRecord(probe=probe)
        rec.f1_score, rec.recall_at_k = 0.9, 1.0
        return rec

    def test_chunk_error_counts_as_gold_error_not_regeneration(self):
        rec = self._labeled("bad_gold_chunk")
        self.assertTrue(report.is_gold_labeling_error(rec))    # 점수 제외 대상
        self.assertFalse(report.is_bad_gold_probe(rec))        # 답 재생성 대상 아님(답은 정상)

    def test_answer_error_still_both(self):
        rec = self._labeled("bad_gold_answer")
        self.assertTrue(report.is_gold_labeling_error(rec))
        self.assertTrue(report.is_bad_gold_probe(rec))         # 답 오류는 재생성 대상 유지

    def test_bad_gold_chunk_excluded_from_composite(self):
        good = [self._good(f"g{i}") for i in range(3)]
        bad = [self._labeled("bad_gold_chunk")]
        only_good = report.build_report(good, 0, mode=1).composite_score["total"]
        with_bad = report.build_report(good + bad, 0, mode=1).composite_score["total"]
        # 거짓 실패(골드 청크 오라벨)는 점수에서 빠지므로 두 종합점수가 같아야 한다.
        self.assertEqual(only_good, with_bad)

    def test_unconfirmed_chunk_error_not_excluded(self):
        rec = self._labeled("bad_gold_chunk", confirmed=False)
        self.assertFalse(report.is_gold_labeling_error(rec))   # 예비면 제외 대상 아님


if __name__ == "__main__":
    unittest.main()
