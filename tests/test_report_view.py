"""
tests/test_report_view.py
agents/serve/report_view.build_report_view 가 리포트 헤드라인 '종합 점수'를
설계 종합점수(composite_score)로 노출하는지 검증한다.

핵심 계약: 웹이 보여주는 '종합 점수'는 overall_score(품질 단일축)가 아니라
composite_score(품질×신뢰도) 여야 한다. composite 가 없으면 overall×100 로 폴백.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schema import DiagnosticReport, Finding, Probe
from core.state import AgentDoctorState
from agents.eval.report import build_report
from agents.eval.types import EvalRecord
from agents.optimize.schemas import OptimizationHistoryItem
from agents.serve.report_view import build_report_view


def make_report(overall=0.90, composite_total=12, pass_threshold=False):
    """overall(0~1)과 composite(0~100)를 일부러 크게 벌려, 어느 쪽이 노출되는지 구분."""
    composite = None
    if composite_total is not None:
        composite = {
            "total": composite_total,
            "components": [
                {"key": "quality", "label": "품질", "score": 89},
                {"key": "reliability", "label": "신뢰도", "score": 7},
            ],
        }
    return DiagnosticReport(
        report_id="r",
        findings=[Finding(finding_id="1", type="retrieval_failure", severity="warning",
                          description="d", label="too_long_context", affected_probes=["p1"])],
        overall_score=overall,
        ragas_scores={"context_recall": 0.7, "faithfulness": 0.7},
        composite_score=composite,
        pass_threshold=pass_threshold,
    )


def make_state(report):
    return AgentDoctorState(report=report, source_url="uploaded.pdf")


class ReportViewCompositeTest(unittest.TestCase):
    def test_headline_score_uses_composite_not_overall(self):
        # overall 0.90(→90점)과 composite 12 를 크게 벌려둠. 헤드라인은 composite(12)여야.
        view = build_report_view(make_state(make_report(overall=0.90, composite_total=12)))
        self.assertEqual(view["score"]["after"], 12)
        self.assertNotEqual(view["score"]["after"], 90.0)  # overall×100 이면 버그

    def test_falls_back_to_overall_when_composite_missing(self):
        # 구버전 리포트(composite 없음) → overall×100 로 폴백.
        view = build_report_view(make_state(make_report(overall=0.90, composite_total=None)))
        self.assertEqual(view["score"]["after"], 90.0)

    def test_no_optimization_before_equals_after(self):
        # 최적화 이력이 없으면 before==after, delta 0.
        view = build_report_view(make_state(make_report(composite_total=88)))
        self.assertEqual(view["score"]["before"], view["score"]["after"])
        self.assertEqual(view["score"]["delta"], 0.0)


class TreatmentCourseViewTest(unittest.TestCase):
    @staticmethod
    def _history_item(index, status, before, after, prescription, **metadata):
        score_metadata = {"pending": False}
        if before is not None:
            score_metadata["before_composite"] = before
        if after is not None:
            score_metadata["after_composite"] = after
        score_metadata.update(metadata)
        return OptimizationHistoryItem(
            trial_id=f"trial-{index}",
            request_id=f"request-{index}",
            iteration=index,
            failure_labels=["test_failure"],
            optimizer="rules",
            status=status,
            selected_prescription_id=prescription,
            metadata=score_metadata,
        )

    def test_rollback_adds_failed_score_and_restored_score_as_separate_points(self):
        state = make_state(make_report(composite_total=84))
        state.optimization_history = [
            self._history_item(1, "applied", 62, 70, "increase_top_k"),
            self._history_item(2, "failed", 70, 66, "decrease_chunk_size"),
            self._history_item(3, "applied", 70, 84, "enable_reranker"),
        ]

        view = build_report_view(state)
        course = view["course"]

        self.assertEqual(
            [(point["kind"], point["score"]) for point in course],
            [
                ("baseline", 62),
                ("kept", 70),
                ("failed", 66),
                ("rollback", 70),
                ("kept", 84),
            ],
        )
        self.assertEqual(course[1]["label"], "top_k 확대")
        self.assertEqual(course[2]["label"], "청크 축소 실패")
        self.assertEqual(course[3]["label"], "원상 복구")
        self.assertEqual(course[4]["label"], "리랭커 활성화")
        self.assertEqual(view["score"]["rolled"], 1)
        self.assertEqual(view["score"]["errors"], 0)

    def test_fallback_label_uses_prescription_order_not_chart_point_order(self):
        state = make_state(make_report(composite_total=84))
        state.optimization_history = [
            self._history_item(1, "applied", 62, 70, "increase_top_k"),
            self._history_item(2, "failed", 70, 66, "decrease_chunk_size"),
            self._history_item(3, "applied", 70, 84, None),
        ]

        course = build_report_view(state)["course"]

        # 두 번째 처방이 실패·복구 두 점을 만들더라도 다음 항목은 세 번째 처방이다.
        self.assertEqual(course[-1]["label"], "처방 3")

    def test_study_error_is_not_plotted_as_measured_rollback(self):
        state = make_state(make_report(composite_total=84))
        state.optimization_history = [
            self._history_item(1, "applied", 62, 78, "increase_top_k"),
            self._history_item(
                2,
                "failed",
                None,
                None,
                "switch_chunking_strategy",
                study_error="adapter 연결 실패",
            ),
            self._history_item(3, "applied", 78, 84, "enable_reranker"),
        ]

        view = build_report_view(state)

        self.assertEqual(
            [(point["kind"], point["score"]) for point in view["course"]],
            [("baseline", 62), ("kept", 78), ("kept", 84)],
        )
        self.assertEqual(view["score"]["rolled"], 0)
        self.assertEqual(view["score"]["errors"], 1)
        self.assertEqual(view["transparency"]["rx_rolled"], 0)
        self.assertEqual(view["transparency"]["rx_errors"], 1)

        error_rx = view["rxs"][1]
        self.assertEqual(error_rx["state"], "error")
        self.assertIsNone(error_rx["score"])
        self.assertEqual(error_rx["verdict"], ["error", "실험 오류 · 설정 원복"])
        self.assertEqual(error_rx["drill"], {
            "label": "오류 원인",
            "rows": [],
            # 선택 근거는 rows 와 모양이 달라 notes 로 따로 싣는다. 이 이력에는
            # action 스냅샷이 없어 비어 있다.
            "notes": [],
            "caption": "adapter 연결 실패",
        })


class FailedQuestionViewTest(unittest.TestCase):
    def test_report_keeps_question_expected_and_actual_answer(self):
        probe = Probe(
            probe_id="p1",
            question="무료 체험 기간은 며칠인가요?",
            source="llm_generated",
            ground_truth="14일",
        )
        finding = Finding(
            finding_id="f1",
            type="generation_failure",
            severity="warning",
            description="답변이 기대 정답과 다릅니다.",
            label="generation_wrong_answer",
            affected_probes=["p1"],
            prescription="생성 프롬프트 조정",
        )
        record = EvalRecord(
            probe=probe,
            generated_answer="30일입니다.",
            findings=[finding],
        )

        report = build_report([record], iteration=1, mode=1)
        view = build_report_view(AgentDoctorState(report=report, probes=[probe]))

        self.assertEqual(report.failed_questions, [{
            "probe_id": "p1",
            "question": "무료 체험 기간은 며칠인가요?",
            "expected_answer": "14일",
            "actual_answer": "30일입니다.",
        }])
        self.assertEqual(len(view["qas"]), 1)
        self.assertEqual(view["qas"][0]["q"], probe.question)
        self.assertEqual(view["qas"][0]["gold"], probe.ground_truth)
        self.assertEqual(view["qas"][0]["actual"], "30일입니다.")
        self.assertEqual(view["qas"][0]["diagnosis"], finding.description)

    def test_failed_question_without_answer_has_explicit_empty_value(self):
        report = make_report()
        report.failed_questions = [{
            "probe_id": "p1",
            "question": "답을 찾을 수 있나요?",
            "expected_answer": "예",
            "actual_answer": "",
        }]
        view = build_report_view(make_state(report))

        self.assertIn("actual", view["qas"][0])
        self.assertEqual(view["qas"][0]["actual"], "")

    def test_multiple_findings_merge_reasons_and_include_preliminary(self):
        confirmed = Finding(
            finding_id="p1:retrieval_missing_gold",
            type="retrieval_failure",
            severity="critical",
            description="[A그룹] retrieval_missing_gold",
            label="retrieval_missing_gold",
            affected_probes=["p1"],
            prescription="검색 범위 확대",
            metadata={
                "group": "A",
                "reason": "gold_in_corpus=True, recall@k=0.00",
            },
        )
        preliminary = Finding(
            finding_id="p1:too_long_context",
            type="context_failure",
            severity="warning",
            description="[예비] [C그룹] too_long_context",
            label="too_long_context",
            confirmed=False,
            affected_probes=["p1"],
            prescription="검색 범위 확대",
            metadata={
                "group": "C",
                "reason": "context_tokens=8192 > limit=4096",
            },
        )
        report = DiagnosticReport(
            report_id="r",
            findings=[confirmed, preliminary],
            failed_questions=[{
                "probe_id": "p1",
                "question": "환불 조건은 무엇인가요?",
                "expected_answer": "구매 후 7일 이내",
                "actual_answer": "환불할 수 없습니다.",
            }],
        )

        qas = build_report_view(make_state(report))["qas"]

        self.assertEqual(len(qas), 1)
        self.assertEqual(
            qas[0]["label"],
            "A · retrieval_missing_gold / C · too_long_context",
        )
        self.assertEqual(
            qas[0]["diagnosis"],
            "gold_in_corpus=True, recall@k=0.00 / context_tokens=8192 > limit=4096",
        )
        self.assertEqual(qas[0]["fix"], "검색 범위 확대")

    def test_diagnosis_falls_back_to_description_when_reason_is_empty(self):
        finding = Finding(
            finding_id="p1:legacy",
            type="generation_failure",
            severity="warning",
            description="구버전 진단 설명",
            label="generation_wrong_answer",
            affected_probes=["p1"],
            metadata={"group": "B", "reason": ""},
        )
        report = DiagnosticReport(
            report_id="r",
            findings=[finding],
            failed_questions=[{
                "probe_id": "p1",
                "question": "질문",
                "expected_answer": "정답",
                "actual_answer": "오답",
            }],
        )

        qa = build_report_view(make_state(report))["qas"][0]

        self.assertEqual(qa["diagnosis"], "구버전 진단 설명")

    def test_all_failed_questions_are_exposed_without_legacy_card_limit(self):
        findings = [
            Finding(
                finding_id=f"p{i}:failure",
                type="generation_failure",
                severity="warning",
                description=f"질문 {i} 실패",
                label="generation_wrong_answer",
                affected_probes=[f"p{i}"],
                metadata={"group": "B", "reason": f"f1={i / 10:.1f}"},
            )
            for i in range(7)
        ]
        report = DiagnosticReport(
            report_id="r",
            findings=findings,
            failed_questions=[
                {
                    "probe_id": f"p{i}",
                    "question": f"질문 {i}",
                    "expected_answer": f"정답 {i}",
                    "actual_answer": f"오답 {i}",
                }
                for i in range(7)
            ],
        )

        qas = build_report_view(make_state(report))["qas"]

        self.assertEqual(len(qas), 7)
        self.assertEqual(qas[-1]["q"], "질문 6")


class ReportViewRecommendationsTest(unittest.TestCase):
    """남은 권고: D그룹 매뉴얼 처방 + 예비(의심) finding 을 웹 뷰로 노출하는지 고정."""

    def _state(self, findings, probes):
        st = AgentDoctorState(report=DiagnosticReport(report_id="r", findings=findings))
        st.probes = probes
        return st

    def test_corpus_gap_surfaces_manual_steps_and_where(self):
        probe = Probe(probe_id="u12", question="재택근무 상한은?", source="user_log",
                      gold_chunk_ids=["c1"], gold_doc_id="policy_2024")
        finding = Finding(finding_id="u12:corpus_gap", type="gap", severity="critical",
                          description="[D그룹] corpus_gap", label="corpus_gap",
                          confirmed=True, affected_probes=["u12"], metadata={"group": "D"})
        recs = build_report_view(self._state([finding], [probe]))["recommendations"]
        self.assertEqual(len(recs), 1)
        rec = recs[0]
        self.assertEqual(rec["kind"], "manual")
        self.assertEqual(rec["badge"], ["data", "D · 데이터"])
        self.assertTrue(rec["steps"])                       # 매뉴얼 스텝 실림
        self.assertEqual(rec["items"][0]["q"], "재택근무 상한은?")
        self.assertIn("policy_2024", rec["items"][0]["where"])  # 어디가 문제인지

    def test_missing_gold_ids_shows_precise_count(self):
        # Eval이 missing_gold_ids를 실으면 'gold N개 중 M개 누락'으로 정밀 표기.
        probe = Probe(probe_id="u12", question="q?", source="user_log",
                      gold_chunk_ids=["c1", "c2", "c3"], gold_doc_id="policy_2024")
        finding = Finding(finding_id="u12:corpus_gap", type="gap", severity="critical",
                          description="[D그룹] corpus_gap", label="corpus_gap",
                          confirmed=True, affected_probes=["u12"],
                          metadata={"group": "D", "missing_gold_ids": ["c2", "c3"]})
        recs = build_report_view(self._state([finding], [probe]))["recommendations"]
        self.assertIn("3개 중 2개 누락", recs[0]["items"][0]["where"])

    def test_preliminary_finding_becomes_prelim_recommendation(self):
        finding = Finding(finding_id="p1:too_long_context", type="context_failure",
                          severity="warning", description="[예비] [C그룹] too_long_context",
                          label="too_long_context", confirmed=False, affected_probes=["p1"],
                          metadata={"group": "C"})
        recs = build_report_view(self._state([finding], []))["recommendations"]
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["kind"], "prelim")
        self.assertEqual(recs[0]["badge"], ["prelim", "의심"])

    def test_bad_gold_answer_branches_by_probe_source(self):
        def action(source):
            probe = Probe(probe_id="q1", question="설립연도?", source=source, ground_truth="1998")
            f = Finding(finding_id="q1:bad_gold_answer", type="gap", severity="warning",
                        description="[D그룹] bad_gold_answer", label="bad_gold_answer",
                        confirmed=True, affected_probes=["q1"], metadata={"group": "D"})
            recs = build_report_view(self._state([f], [probe]))["recommendations"]
            return recs[0]["items"][0]["where"]

        # 사용자 제공 정답은 자동 재생성 대상이 아니라 검수 요청
        self.assertIn("검수", action("user_log"))
        self.assertNotIn("검수", action("taxonomy"))  # 우리가 만든 probe는 재생성 대상
        self.assertIn("재생성", action("taxonomy"))

    def test_confirmed_actionable_excluded_from_recommendations(self):
        # 확정 자동처방 대상(retrieval_low_rank)은 dxs/rxs 몫 — 남은 권고에서 제외.
        finding = Finding(finding_id="p1:retrieval_low_rank", type="retrieval_failure",
                          severity="warning", description="[A그룹] retrieval_low_rank",
                          label="retrieval_low_rank", confirmed=True, affected_probes=["p1"],
                          metadata={"group": "A"})
        recs = build_report_view(self._state([finding], []))["recommendations"]
        self.assertEqual(recs, [])


class ActionCenteredRxCardTest(unittest.TestCase):
    """처방 카드는 "무엇을 바꿨나"(action)와 "무엇이 지지했나"(라벨 전체)를 말한다."""

    @staticmethod
    def _item(**overrides):
        values = {
            "trial_id": "t1",
            "request_id": "r1",
            "iteration": 1,
            "failure_labels": ["retrieval_missing_gold"],
            "optimizer": "rules",
            "status": "applied",
            "selected_prescription_id": "increase_top_k",
            "before_config": {"top_k": 5},
            "after_config": {"top_k": 10},
            "reason": "gold가 검색 결과에 없음",
            "action_key": "retriever.top_k:increase",
            "supporting_labels": [
                "retrieval_missing_gold",
                "retrieval_incomplete_enumeration",
            ],
            "supporting_probes": ["p1", "p2", "p3"],
        }
        values.update(overrides)
        item = OptimizationHistoryItem(**values)
        item.metadata.update(
            {
                "before_score": 0.6,
                "after_score": 0.8,
                "resolved_labels": ["retrieval_missing_gold"],
                "remaining_labels": ["retrieval_incomplete_enumeration"],
            }
        )
        return item

    def _rxs(self, item):
        state = make_state(make_report())
        state.optimization_history = [item]
        return build_report_view(state)["rxs"]

    def test_card_shows_action_and_every_supporting_label(self):
        card = self._rxs(self._item())[0]

        self.assertEqual(card["action"], "retriever.top_k:increase")
        # 대표 라벨 하나로 좁히면 "여러 문제가 같은 변경을 원했다"는 선택 근거가 사라진다.
        self.assertEqual(
            card["target"],
            "retrieval_missing_gold, retrieval_incomplete_enumeration",
        )

    def test_card_separates_supported_from_resolved(self):
        """지지받은 라벨을 그대로 성과로 읽으면 리포트가 실제보다 좋아 보인다."""
        card = self._rxs(self._item())[0]

        self.assertEqual(card["resolved"], ["retrieval_missing_gold"])
        self.assertEqual(card["remaining"], ["retrieval_incomplete_enumeration"])
        notes = dict(card["drill"]["notes"])
        self.assertEqual(notes["해결됨"], "retrieval_missing_gold")
        self.assertEqual(notes["남음"], "retrieval_incomplete_enumeration")
        self.assertEqual(notes["영향 질문"], "3건")

    def test_selection_notes_do_not_leak_into_sweep_rows(self):
        """drill.rows 는 sweep 막대 전용이다 — 모양이 섞이면 프론트 렌더가 깨진다."""
        card = self._rxs(self._item())[0]

        self.assertEqual(card["drill"]["rows"], [])

    def test_rolled_back_card_claims_nothing_resolved(self):
        """웹 카드가 CLI 리포트와 같은 판정을 보여야 한다.

        `_build_rxs` 는 `OptimizationReport` 가 아니라 raw 이력 metadata 를 직접
        읽으므로 reporter 의 `verdict.keep` 가드가 걸리지 않는 경로다. 실제로 이
        경로만 "롤백"이라고 표시하면서 동시에 "해결됨"을 출력했다.
        """
        item = self._item(status="failed")
        item.rollback_reason = "종합점수 상승폭 부족"
        # 저장 시점 가드를 우회한 구버전 이력이 섞여 들어와도 표시가 새면 안 된다.
        item.metadata["resolved_labels"] = ["retrieval_missing_gold"]

        card = self._rxs(item)[0]

        self.assertEqual(card["verdict"], ["roll", "롤백"])
        self.assertEqual(card["resolved"], [])
        self.assertEqual(
            card["remaining"],
            ["retrieval_missing_gold", "retrieval_incomplete_enumeration"],
        )
        self.assertNotIn("해결됨", dict(card["drill"]["notes"]))

    def test_course_point_uses_a_human_action_name(self):
        state = make_state(make_report())
        state.optimization_history = [self._item()]

        labels = [p["label"] for p in build_report_view(state)["course"]]

        self.assertIn("top_k 확대", labels)

    def test_legacy_history_without_action_still_renders(self):
        """이전 실행이 저장한 이력에는 action 필드가 없다 — 그래도 읽혀야 한다."""
        legacy = OptimizationHistoryItem(
            trial_id="old",
            request_id="old",
            iteration=1,
            failure_labels=["retrieval_missing_gold"],
            optimizer="rules",
            status="applied",
            selected_prescription_id="increase_top_k",
            before_config={"top_k": 5},
            after_config={"top_k": 10},
        )
        legacy.metadata.update({"before_score": 0.6, "after_score": 0.8})
        state = make_state(make_report())
        state.optimization_history = [legacy]

        view = build_report_view(state)

        self.assertEqual(view["rxs"][0]["action"], "")
        self.assertEqual(view["rxs"][0]["target"], "retrieval_missing_gold")
        self.assertEqual(view["rxs"][0]["drill"]["notes"], [])
        # 처방 id 표를 통한 구버전 이름이 그대로 나온다.
        self.assertIn("top_k 확대", [p["label"] for p in view["course"]])


if __name__ == "__main__":
    unittest.main()
