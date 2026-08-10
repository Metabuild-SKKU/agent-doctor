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

    def test_ungrounded_answer_demotes_instead_of_erasing(self):
        """faith 가 **측정됐는데 미달**이면 라벨을 지우지 않고 예비로 남긴다(2026-08-07).

        앞의 두 조건(f1 통과 · 오라클 실패)은 글자 비교라 재실행해도 같고, 그 둘만으로
        "이 골드로는 이 답이 안 나온다"가 이미 선다. faith 는 **왜** 그런지(골드 오라벨 vs
        파라미터 기억)만 가르므로 라벨의 존재가 아니라 확정/예비를 정해야 한다.

        예전엔 여기서 None 이라 라벨이 통째로 사라졌고, 실측에서 그 자리를
        retrieval_rerank_candidate_miss 가 차지해 엉뚱한 처방까지 만들었다
        (output/logs/corpus_20260804_103059.txt, probe_qa_4195 반복 3).
        """
        finding = diagnose.bad_gold_chunk(_rec(f1=1.0, oracle_f1=0.0, faith=0.0))
        self.assertIsNotNone(finding)
        self.assertEqual(finding.label, "bad_gold_chunk")
        self.assertFalse(finding.confirmed)

    def test_grounded_answer_stays_confirmed(self):
        """faith 가 문턱 이상이면 예전과 같이 확정이다 — 정상 경로는 안 바뀐다."""
        finding = diagnose.bad_gold_chunk(_rec(f1=1.0, oracle_f1=0.0, faith=1.0))
        self.assertIsNotNone(finding)
        self.assertTrue(finding.confirmed)

    def test_silent_when_faithfulness_is_unmeasured(self):
        """미측정(DEEP 미만)은 예비로도 세우지 않는다 — 측정 없이 주장하면 안 된다.

        여기서 예비를 세우면 RAGAS 를 안 켠 실행 **전체**가 검색 진단을 잃는다
        (예비가 검색 슬롯을 닫으므로).
        """
        self.assertIsNone(diagnose.bad_gold_chunk(_rec(f1=1.0, oracle_f1=0.0, faith=None)))

    def test_silent_when_no_gold_context(self):
        # 골드 컨텍스트가 없으면(코퍼스 결손 등) 청크 오라벨로 단정 불가 → 침묵
        rec = _rec(f1=1.0, oracle_f1=0.0, faith=1.0, oracle_answer=None)
        self.assertIsNone(diagnose.bad_gold_chunk(rec))


class DiagnoseShortCircuitTest(unittest.TestCase):
    """diagnose 단락: 골드 청크 오라벨이면 거짓 원인(retrieval_low_rank·generation_*)을 막고
    bad_gold_chunk 하나만 남긴다."""

    def setUp(self):
        metrics_common.set_mode(Mode.DEEP)

    def tearDown(self):
        metrics_common.set_mode(Mode.FAST)

    def test_mislabel_suppresses_false_causes(self):
        # recall=0 이라 정상 경로면 retrieval_low_rank 가 붙어야 하지만, 골드 오라벨이 감지되면
        # 단락되어 그 거짓 원인이 안 붙는다. 지표·RAGAS 재계산은 주입값을 보존하려 no-op 처리.
        rec = _rec(f1=1.0, oracle_f1=0.0, faith=1.0, recall=0.0)
        with patch.object(diagnose, "_compute_metrics"), \
             patch.object(diagnose, "_compute_ragas_real"), \
             patch.object(diagnose, "_compute_ragas_oracle"):
            findings = diagnose.diagnose(rec, mode=int(Mode.DEEP))
        self.assertEqual([f.label for f in findings], ["bad_gold_chunk"])

    def test_preliminary_still_blocks_the_false_retrieval_cause(self):
        """실측 사고의 회귀 테스트 — faith 가 튀어도 엉뚱한 검색 라벨이 붙으면 안 된다.

        output/logs/corpus_20260804_103059.txt 의 probe_qa_4195 재현이다. 답도 검색 결과도
        5회 내내 같았는데 반복 3 에서만 faithfulness 가 1.000→0.000 으로 튀었고, 그 한 번에
        bad_gold_chunk 가 사라지면서 retrieval_rerank_candidate_miss 가 자리를 차지해
        처방(rerank_candidates 20→22)까지 만들어 최종 config 에 남았다.

        검색 라벨을 막는 근거는 faith 와 무관하다 — 오라클이 실패했다는 건 그 골드에 답이
        없다는 뜻이고, 답 없는 청크를 "왜 안 가져왔나" 고 탓하는 건 이미 거짓이다.
        """
        rec = _rec(f1=1.0, oracle_f1=0.0, faith=0.0, recall=0.0)   # faith 측정됐고 미달
        with patch.object(diagnose, "_compute_metrics"), \
             patch.object(diagnose, "_compute_ragas_real"), \
             patch.object(diagnose, "_compute_ragas_oracle"):
            labels = [f.label for f in diagnose.diagnose(rec, mode=int(Mode.DEEP))]

        self.assertIn("bad_gold_chunk", labels)                     # 라벨이 사라지지 않는다
        self.assertFalse([l for l in labels if l.startswith("retrieval_")],
                         f"검색 라벨이 붙으면 안 된다: {labels}")

    def test_preliminary_label_is_marked_unconfirmed_in_the_result(self):
        """결과에 실리되 예비로 표시돼야 한다 — 확정으로 실리면 처방 가중치를 다 가져간다."""
        rec = _rec(f1=1.0, oracle_f1=0.0, faith=0.0, recall=0.0)
        with patch.object(diagnose, "_compute_metrics"), \
             patch.object(diagnose, "_compute_ragas_real"), \
             patch.object(diagnose, "_compute_ragas_oracle"):
            findings = diagnose.diagnose(rec, mode=int(Mode.DEEP))
        gold = [f for f in findings if f.label == "bad_gold_chunk"]
        self.assertEqual(len(gold), 1)
        self.assertFalse(gold[0].confirmed)

    def test_preliminary_does_not_swallow_generation_evidence(self):
        """예비는 검색만 막는다 — 생성 슬롯은 살려 둔다.

        오라클 트랙이 스스로 근거 없이 답한 경우는 골드가 아니라 **생성기**에 대한 증거라,
        미확인 상태에서 그것까지 지우면 측정된 신호를 추측으로 덮는 셈이다.
        """
        sentinel = Finding(finding_id="gh", type="generation_failure", severity="warning",
                           description="d", label="generation_hallucination",
                           confirmed=True, affected_probes=["p1"])
        rec = _rec(f1=1.0, oracle_f1=0.0, faith=0.0, recall=0.0)
        with patch.object(diagnose, "_compute_metrics"), \
             patch.object(diagnose, "_compute_ragas_real"), \
             patch.object(diagnose, "_compute_ragas_oracle"), \
             patch.object(diagnose, "_generation_failed", return_value=True), \
             patch.object(diagnose, "_pick", return_value=sentinel):
            labels = [f.label for f in diagnose.diagnose(rec, mode=int(Mode.DEEP))]
        self.assertIn("bad_gold_chunk", labels)
        self.assertIn("generation_hallucination", labels)

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

    def test_unconfirmed_chunk_error_is_also_excluded(self):
        """예비도 점수에서 뺀다(2026-08-07). 확정만 빼면 **제외 여부가 심판 노이즈를 탄다.**

        실측에서 probe_qa_4195 는 5회 중 4회 확정 bad_gold_chunk 라 제외됐고,
        faithfulness 가 1.000→0.000 으로 튄 1회만 다른 라벨이 붙어 실패로 집계됐다.
        같은 probe 가 회차마다 빠졌다 들어왔다 하면 그것만으로 종합점수가 흔들린다.

        제외의 근거는 결정론적인 쪽에 있다 — _f1_ok(답이 맞았다)와 not _oracle_ok
        (골드만으론 못 맞힌다)는 재실행해도 같은 값이다. faithfulness 는 왜인지만 가르고,
        그건 라벨의 확정/예비로 표시된다.
        """
        rec = self._labeled("bad_gold_chunk", confirmed=False)
        self.assertTrue(report.is_gold_labeling_error(rec))

    def test_confirmed_chunk_error_still_excluded(self):
        rec = self._labeled("bad_gold_chunk", confirmed=True)
        self.assertTrue(report.is_gold_labeling_error(rec))

    def test_unrelated_label_is_not_excluded(self):
        """제외는 골드 오류 라벨에만 걸린다 — 조건을 넓히면서 다른 실패까지 새면 안 된다."""
        rec = self._labeled("retrieval_low_rank", confirmed=True)
        self.assertFalse(report.is_gold_labeling_error(rec))

    # ── 점수 말고 '실패 집계'에서도 빠지는가 ────────────────────────
    # 제외가 build_report 의 점수 계산에만 있어서, 같은 probe 가 점수에선 빠지고 실패
    # 목록·콘솔 마크에는 실패로 남았다. 실측(corpus_20260804): 답 만점(answer=1.00)인
    # probe 4개가 매 반복 ❌ 로 찍히고 '실패한 검증 질문'을 채웠다.

    def test_gold_error_is_not_listed_as_a_failed_question(self):
        real_failure = self._labeled("retrieval_low_rank")
        real_failure.probe.probe_id = "real"
        gold_error = self._labeled("bad_gold_chunk")

        listed = report.build_report([real_failure, gold_error], 0, mode=1).failed_questions

        self.assertEqual([q["probe_id"] for q in listed], ["real"])

    def test_gold_error_count_survives_the_exclusion(self):
        """조용히 빼면 '30문항 중 몇 개가 애초에 채점 불가였나'를 알 수 없다."""
        rep = report.build_report(
            [self._good("g1"), self._labeled("bad_gold_chunk"),
             self._labeled("bad_gold_answer")], 0, mode=1)

        self.assertEqual(rep.ragas_scores["gold_labeling_errors"], 2)
        # 진단 자체는 남는다 — 검수하라는 신호가 사라지면 정답셋이 영원히 안 고쳐진다.
        self.assertEqual(len(rep.findings), 2)

    def test_no_gold_error_leaves_the_key_absent(self):
        rep = report.build_report([self._good("g1")], 0, mode=1)
        self.assertNotIn("gold_labeling_errors", rep.ragas_scores)


class GoldErrorProbeMarkTest(unittest.TestCase):
    """콘솔 마크 — 골드 오류 probe 를 ❌ 로 찍으면 '맞은 답을 틀렸다고 한다'가 된다."""

    def _rec_with(self, label):
        probe = Probe(probe_id="p", question="q", source="taxonomy", ground_truth="gt")
        rec = EvalRecord(probe=probe, generated_answer="답")
        rec.f1_score, rec.recall_at_k = 1.0, 0.0
        rec.findings = [Finding(finding_id="p", type="gap", severity="warning",
                                description="d", label=label, confirmed=True,
                                affected_probes=["p"])]
        return rec

    def test_gold_error_gets_the_review_mark_not_the_failure_mark(self):
        from agents.eval import agent as eval_agent

        self.assertEqual(eval_agent._mark(None), "🔍")
        self.assertNotEqual(eval_agent._mark(None), eval_agent._mark(False))

    def test_real_failure_still_marked_failed(self):
        from agents.eval import agent as eval_agent

        self.assertEqual(eval_agent._mark(False), "❌")
        self.assertEqual(eval_agent._mark(True), "✅")

    def test_mark_falls_back_to_ascii_on_cp949(self):
        """cp949 콘솔에서 세 상태가 '?' 로 뭉개지면 안 된다(기존 계약을 3상태로 확장)."""
        from agents.eval import agent as eval_agent

        class _Cp949Stdout:
            encoding = "cp949"

        with patch.object(eval_agent, "sys") as fake_sys:
            fake_sys.stdout = _Cp949Stdout()
            marks = {eval_agent._mark(v) for v in (True, False, None)}

        self.assertEqual(marks, {"[OK]", "[FAIL]", "[검수]"})

    def test_probe_line_uses_the_review_mark(self):
        import io
        import contextlib
        from agents.eval import agent as eval_agent

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            eval_agent._log_probe(1, 30, self._rec_with("bad_gold_chunk"))
        header = buf.getvalue().splitlines()[0]

        self.assertIn("골드 검수", header)
        self.assertNotIn("❌", header)

    def test_probe_line_still_fails_a_real_failure(self):
        import io
        import contextlib
        from agents.eval import agent as eval_agent

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            eval_agent._log_probe(1, 30, self._rec_with("retrieval_low_rank"))
        header = buf.getvalue().splitlines()[0]

        self.assertIn("❌", header)
        self.assertNotIn("골드 검수", header)

    def _summary_line(self, records):
        import io
        import contextlib
        from agents.eval import agent as eval_agent

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            eval_agent._log_diagnosis_summary(records)
        return buf.getvalue()

    def test_step4_summary_splits_gold_review_out_of_failures(self):
        """probe 줄은 🔍 인데 마감 요약만 실패에 얹으면 같은 STEP 안에서 수가 어긋난다."""
        ok = EvalRecord(probe=Probe(probe_id="ok", question="q", source="taxonomy"))
        line = self._summary_line([ok, self._rec_with("bad_gold_chunk"),
                                   self._rec_with("retrieval_low_rank")])

        self.assertIn("성공 1 / 실패 1 / 골드 검수 1", line)

    def test_step4_summary_omits_the_slot_when_there_is_no_gold_error(self):
        ok = EvalRecord(probe=Probe(probe_id="ok", question="q", source="taxonomy"))
        line = self._summary_line([ok, self._rec_with("retrieval_low_rank")])

        self.assertIn("성공 1 / 실패 1", line)
        self.assertNotIn("골드 검수", line)


if __name__ == "__main__":
    unittest.main()
