"""
tests/test_ext_report_view.py
외부 RAG 진단서 뷰(report_view.build_ext_report_view) 회귀 테스트.

이 계층이 지켜야 할 것:
  - 템플릿(web/prototype/report.html)이 읽는 키를 빠뜨리지 않는다
    (빠지면 화면이 조용히 빈 섹션을 그린다 - 에러가 안 난다)
  - 외부 모드에 없는 것(치료경과·처방이력)을 있는 척하지 않는다
  - 권고 카드가 ext_ 라벨에서도 비지 않는다
    (기존 _build_recommendations 는 get_rule(ext_...)=None 이라 조용히 사라진다)
"""
import unittest
from datetime import datetime

from core.schema import DiagnosticReport, Finding

from agents.serve.report_view import build_ext_report_view

# 템플릿이 data.<키> 로 읽는 최상위 키. 하나라도 빠지면 그 섹션이 빈 채 그려진다.
TEMPLATE_KEYS = {
    "meta", "score", "priority", "metrics",
    "course", "rxs", "dxs", "qas", "recommendations",
}


def _report(findings=None, scores=None, composite=None):
    return DiagnosticReport(
        report_id="t1",
        created_at=datetime(2026, 8, 10, 12, 0, 0),
        iteration=1,
        overall_score=0.5,
        findings=findings or [],
        ragas_scores=scores or {},
        composite_score=composite or {"total": 16.0},
    )


def _finding(label, confirmed=True, severity="warning"):
    return Finding(
        finding_id=f"f_{label}", type="generation", severity=severity,
        description=f"{label} 사유", label=label, confirmed=confirmed)


class ExtReportViewTests(unittest.TestCase):

    def test_template_keys_present(self):
        view = build_ext_report_view(_report(), {})
        self.assertTrue(TEMPLATE_KEYS <= set(view), TEMPLATE_KEYS - set(view))

    def test_course_and_rxs_are_empty(self):
        """외부 모드에는 치료경과·처방이력이 원리적으로 없다 — 남의 인덱스에는
        처방을 적용할 수 없으므로 '아직 안 한 것'이 아니라 '할 수 없는 것'이다."""
        view = build_ext_report_view(
            _report([_finding("ext_answer_off_topic")]), {"records": 3})
        self.assertEqual(view["course"], [])
        self.assertEqual(view["rxs"], [])
        self.assertEqual(view["score"]["kept"], 0)
        self.assertEqual(view["score"]["rolled"], 0)

    def test_before_equals_after(self):
        """비교 대상(처방 적용 후 두 번째 로그)이 없으므로 개선폭을 지어내지 않는다."""
        view = build_ext_report_view(_report(), {})
        self.assertEqual(view["score"]["before"], view["score"]["after"])
        self.assertEqual(view["score"]["delta"], 0.0)

    def test_recommendations_not_empty_for_ext_labels(self):
        """ext_ 라벨에서도 권고가 나와야 한다 — 기존 _build_recommendations 는
        get_rule(ext_...)=None 이라 카드가 조용히 사라진다(그래서 전용 경로가 있다)."""
        view = build_ext_report_view(
            _report([_finding("ext_answer_off_topic", severity="critical")]), {})
        self.assertEqual(len(view["recommendations"]), 1)
        rec = view["recommendations"][0]
        self.assertTrue(rec["title"].strip())
        self.assertTrue(rec["steps"])
        self.assertTrue(rec["steps"][0]["action"].strip())

    def test_recommendation_card_shape_matches_template(self):
        """템플릿 renderRecommendations 가 읽는 필드."""
        view = build_ext_report_view(_report([_finding("ext_grounded_but_wrong")]), {})
        rec = view["recommendations"][0]
        for key in ("kind", "badge", "title", "desc", "items", "steps", "cta"):
            self.assertIn(key, rec)
        self.assertEqual(len(rec["badge"]), 2)      # [클래스, 라벨]
        for step in rec["steps"]:
            self.assertIn("action", step)
            self.assertIn("detail", step)

    def test_config_flows_into_step_action(self):
        """로그 config 가 있으면 권고 문구에 현재값이 실린다."""
        view = build_ext_report_view(
            _report([_finding("ext_context_overflow")]),
            {"config": {"top_k": 5}})
        actions = " ".join(s["action"] for s in view["recommendations"][0]["steps"])
        self.assertIn("top_k=5", actions)

    def test_tentative_counted_in_cta(self):
        view = build_ext_report_view(
            _report([_finding("ext_answer_off_topic", confirmed=False)]), {})
        self.assertIn("예비", view["recommendations"][0]["cta"])

    def test_metrics_from_ragas_scores(self):
        view = build_ext_report_view(
            _report(scores={"faithfulness": 1.0, "response_relevancy": 0.5}), {})
        names = {m["en"] for m in view["metrics"]}
        self.assertIn("faithfulness", names)
        self.assertIn("response_relevancy", names)

    def test_capability_is_surfaced(self):
        """어디까지 잰 진단인지 화면이 밝힐 수 있어야 한다."""
        view = build_ext_report_view(_report(), {
            "records": 6, "tier": "triad",
            "with_ground_truth": 6, "with_gold_contexts": 6, "notes": ["n"]})
        ext = view["transparency"]["external"]
        # tier 코드가 아니라 사람 말로 나와야 한다("triad" 는 우리 적재기 내부 이름).
        self.assertEqual(ext["tier"], "컨텍스트까지")
        self.assertEqual(ext["with_ground_truth"], 6)
        self.assertEqual(view["meta"]["question_count"], 6)

    def test_qa_only_tier_is_labelled(self):
        """제한적 진단임이 제목에 드러나야 한다."""
        view = build_ext_report_view(_report(), {"tier": "qa_only"})
        self.assertIn("제한적", view["meta"]["depth"])

    def test_empty_report_does_not_crash(self):
        view = build_ext_report_view(_report(), {})
        self.assertEqual(view["recommendations"], [])
        self.assertEqual(view["dxs"], [])

    def test_json_serializable(self):
        """HTML 에 심으려면 직렬화돼야 한다."""
        import json
        view = build_ext_report_view(
            _report([_finding("ext_answer_off_topic")]), {"records": 1})
        self.assertTrue(json.dumps(view, ensure_ascii=False))


class ReportHtmlInjectionTests(unittest.TestCase):
    """뷰를 템플릿에 심는 공용 유틸 — run_corpus 와 run_replay_report 가 공유한다."""

    def test_injection_removes_fetch_branch(self):
        from tools.report_html import REPORT_TEMPLATE, inject_view
        if not REPORT_TEMPLATE.exists():
            self.skipTest("report.html 템플릿 없음")
        html = inject_view({"score": {"after": 1}},
                           REPORT_TEMPLATE.read_text(encoding="utf-8"))
        # fetch 실패 배너를 띄우는 더미 분기가 남으면 안 된다
        self.assertNotIn("renderReport({}, false)", html)
        self.assertIn("__AGENT_DOCTOR_REPORT__", html)

    def test_data_block_precedes_render(self):
        """데이터가 렌더 스크립트보다 뒤에 오면 undefined 라 빈 리포트가 그려진다."""
        from tools.report_html import REPORT_TEMPLATE, inject_view
        if not REPORT_TEMPLATE.exists():
            self.skipTest("report.html 템플릿 없음")
        html = inject_view({"score": {"after": 1}},
                           REPORT_TEMPLATE.read_text(encoding="utf-8"))
        self.assertLess(html.index("window.__AGENT_DOCTOR_REPORT__ ="),
                        html.index("renderReport(window.__AGENT_DOCTOR_REPORT__"))

    def test_script_close_is_escaped(self):
        """</script> 가 payload 에 들어가면 HTML 파싱이 깨진다."""
        from tools.report_html import REPORT_TEMPLATE, inject_view
        if not REPORT_TEMPLATE.exists():
            self.skipTest("report.html 템플릿 없음")
        html = inject_view({"meta": {"corpus": "</script><b>x"}},
                           REPORT_TEMPLATE.read_text(encoding="utf-8"))
        self.assertNotIn("</script><b>x", html)
        self.assertIn("<\\/script>", html)


class ExtModeSectionTests(unittest.TestCase):
    """외부 모드에서 '없음'과 '할 수 없음'을 화면이 구분할 수 있어야 한다."""

    def test_mode_marks_impossible_sections(self):
        view = build_ext_report_view(_report(), {})
        mode = view["mode"]
        self.assertEqual(mode["kind"], "external")
        # Rx 목록은 치료경과 섹션 안에 있으므로 course 를 숨기면 함께 사라진다.
        self.assertIn("course", mode["hidden_sections"])
        self.assertTrue(mode["notice"].strip())
        self.assertEqual(view["rxs"], [])

    def test_qas_filled_from_failed_questions(self):
        """로그에도 질문·답변·정답이 있으므로 실패 질문은 채운다 —
        '어떤 질문에서 무너졌나'가 상대에게 넘길 진단서의 핵심이다."""
        rep = _report([_finding("ext_answer_off_topic")])
        rep.failed_questions = [{
            "probe_id": "ext_0000", "question": "연차는 며칠인가요?",
            "expected_answer": "연 15일", "actual_answer": "병가는 60일입니다",
        }]
        view = build_ext_report_view(rep, {})
        self.assertEqual(len(view["qas"]), 1)
        qa = view["qas"][0]
        self.assertEqual(qa["q"], "연차는 며칠인가요?")
        self.assertEqual(qa["gold"], "연 15일")
        self.assertIn("병가", qa["actual"])

    def test_template_has_hooks_for_mode(self):
        """템플릿이 hidden_sections 를 실제로 숨길 수 있어야 한다 —
        뷰만 고치고 마크업이 없으면 빈 섹션이 그대로 남는다."""
        from tools.report_html import REPORT_TEMPLATE
        if not REPORT_TEMPLATE.exists():
            self.skipTest("report.html 템플릿 없음")
        html = REPORT_TEMPLATE.read_text(encoding="utf-8")
        self.assertIn('data-section="course"', html)
        self.assertIn("function renderMode", html)
        self.assertIn('id="modeNotice"', html)


class ExtSectionLayoutTests(unittest.TestCase):
    """섹션 구성 — 내부 모드 전제가 담긴 제목·순서를 외부 뜻으로 바꾼다."""

    def test_titles_replace_internal_assumptions(self):
        """'처방 전후'는 비교 대상이 없고, '남은 권고'는 자동 처방 뒤 잔여물이란
        뜻이라 외부 모드에서 성립하지 않는다."""
        view = build_ext_report_view(_report(), {})
        titles = {t["section"]: t["title"] for t in view["mode"]["section_titles"]}
        self.assertEqual(titles["metrics"], "품질 지표")
        self.assertEqual(titles["recommendations"], "처방 추천")
        self.assertEqual(titles["transparency"], "진단 범위")

    def test_recommendations_move_above_dxs(self):
        """권고는 이 진단서의 결론이다 — 꼬리가 아니라 지표 다음 자리로 올린다."""
        moves = build_ext_report_view(_report(), {})["mode"]["move_before"]
        self.assertIn({"section": "recommendations", "before": "dxs"}, moves)

    def test_notice_warns_score_is_not_comparable(self):
        """검색축이 recall 이 아니라 faithfulness 대역이라 내부 점수와 비교 불가."""
        notice = build_ext_report_view(_report(), {})["mode"]["notice"]
        self.assertIn("faithfulness", notice)
        self.assertIn("비교", notice)

    def test_hero_gets_evidence_counts(self):
        """외부 hero 는 처방 수 대신 확정/예비를 보여준다."""
        view = build_ext_report_view(_report([
            _finding("ext_answer_off_topic"),
            _finding("ext_generation_hallucination", confirmed=False)]), {"tier": "triad"})
        self.assertEqual(view["score"]["confirmed"], 1)
        self.assertEqual(view["score"]["tentative"], 1)
        self.assertEqual(view["meta"]["tier"], "컨텍스트까지")

    def test_template_has_section_hooks(self):
        """뷰가 가리키는 섹션 이름표가 마크업에 실제로 있어야 한다 —
        없으면 제목 교체·이동이 조용히 no-op 이 된다."""
        from tools.report_html import REPORT_TEMPLATE
        if not REPORT_TEMPLATE.exists():
            self.skipTest("report.html 템플릿 없음")
        html = REPORT_TEMPLATE.read_text(encoding="utf-8")
        view = build_ext_report_view(_report(), {})
        names = set(view["mode"]["hidden_sections"])
        names |= {t["section"] for t in view["mode"]["section_titles"]}
        for m in view["mode"]["move_before"]:
            names |= {m["section"], m["before"]}
        for name in names:
            self.assertIn(f'data-section="{name}"', html, f"{name} 이름표 없음")


class TransparencyRenderTests(unittest.TestCase):
    """진단 내역 — 하드코딩 목업이 실제 진단서에 나가던 것을 데이터 렌더로 바꿨다."""

    def test_mockup_numbers_are_gone_from_markup(self):
        from tools.report_html import REPORT_TEMPLATE
        if not REPORT_TEMPLATE.exists():
            self.skipTest("report.html 템플릿 없음")
        html = REPORT_TEMPLATE.read_text(encoding="utf-8")
        # 마크업에 박힌 값은 어느 코퍼스든 똑같이 표시됐다.
        self.assertNotIn('<div class="tn">2분 14초</div>', html)
        self.assertNotIn('<div class="tn">218</div>', html)
        self.assertIn('id="transList"', html)
        self.assertIn("function renderTransparency", html)

    def test_external_transparency_payload(self):
        view = build_ext_report_view(_report(), {
            "records": 6, "tier": "triad",
            "with_ground_truth": 6, "with_gold_contexts": 5, "notes": []})
        ext = view["transparency"]["external"]
        self.assertEqual((ext["tier"], ext["with_ground_truth"], ext["with_gold_contexts"]),
                         ("컨텍스트까지", 6, 5))


class ExtLabelExposureTests(unittest.TestCase):
    """라벨 코드는 화면에 노출되지 않는다 — 진단서는 우리 코드를 모르는 상대가 읽는다."""

    def _view(self):
        f1 = _finding("ext_answer_off_topic", severity="critical")
        f1.metadata = {"reason": "answer relevancy 0.43 - 답이 질문을 비껴감"}
        f1.affected_probes = ["p1"]
        f2 = _finding("ext_grounded_but_wrong")
        f2.metadata = {"reason": "correctness 0.12인데 faithfulness 1.00"}
        f2.affected_probes = ["p2"]
        rep = _report([f1, f2], scores={"faithfulness": 1.0})
        rep.failed_questions = [
            {"probe_id": "p1", "question": "q1", "expected_answer": "a", "actual_answer": "b"},
            {"probe_id": "p2", "question": "q2", "expected_answer": "a", "actual_answer": "b"},
        ]
        return build_ext_report_view(rep, {})

    def test_no_raw_label_codes_anywhere(self):
        import json
        blob = json.dumps(self._view(), ensure_ascii=False)
        # mode/code 매핑 자체가 아니라 사용자에게 보이는 문구에 코드가 없어야 한다.
        for section in ("priority", "dxs", "qas"):
            text = json.dumps(json.loads(blob)[section], ensure_ascii=False)
            self.assertNotIn("ext_", text, f"{section} 에 라벨 코드 노출")

    def test_priority_uses_human_titles(self):
        titles = [p["title"] for p in self._view()["priority"]]
        self.assertTrue(any("동문서답" in t for t in titles), titles)
        self.assertTrue(all("질문" in t for t in titles), titles)   # "— 질문 N건"

    def test_dxs_groups_by_label(self):
        """probe 마다 카드를 만들면 같은 문장이 반복된다 — 라벨 단위로 묶는다."""
        f = [_finding("ext_grounded_but_wrong") for _ in range(6)]
        for i, x in enumerate(f):
            x.affected_probes = [f"p{i}"]
        dxs = build_ext_report_view(_report(f), {})["dxs"]
        self.assertEqual(len(dxs), 1)
        self.assertIn("6건", dxs[0]["foot"])

    def test_dxs_code_is_short_korean(self):
        codes = [d["code"] for d in self._view()["dxs"]]
        self.assertIn("동문서답", codes)
        self.assertIn("근거 부족", codes)

    def test_rx_points_to_recommendation_section(self):
        """'미처방'이라고 적으면 고칠 방법이 없는 것처럼 읽힌다."""
        for d in self._view()["dxs"]:
            self.assertNotEqual(d["rx"], "미처방")

    def test_qas_badge_uses_short_name(self):
        labels = " ".join(q["label"] for q in self._view()["qas"])
        self.assertNotIn("ext_", labels)


class ExtMetricsSingleTests(unittest.TestCase):
    """품질 지표 — 외부 모드에는 '처방 전' 값이 없다."""

    def test_metrics_are_single_valued(self):
        view = build_ext_report_view(
            _report(scores={"faithfulness": 1.0, "response_relevancy": 0.5}), {})
        self.assertTrue(view["metrics"])
        for m in view["metrics"]:
            self.assertTrue(m["single"])
            # before 를 남기면 화면이 전후 막대 2개를 그리고 '변화 없음'으로 읽힌다.
            self.assertNotIn("before", m)

    def test_template_handles_single_metrics(self):
        """템플릿이 single 을 모르면 m.before.toFixed 에서 죽는다."""
        from tools.report_html import REPORT_TEMPLATE
        if not REPORT_TEMPLATE.exists():
            self.skipTest("report.html 템플릿 없음")
        self.assertIn("m.single", REPORT_TEMPLATE.read_text(encoding="utf-8"))


class ExtVerdictBadgeTests(unittest.TestCase):
    """통과/미달 배지 — 외부 진단은 내부 문턱으로 판정하지 않는다."""

    def test_pass_threshold_is_none(self):
        """상단 안내가 '내부 점수와 비교 불가'라고 밝히면서 내부 문턱(composite >= 90)으로
        배지를 그리면 스스로 모순이다. 판정을 아예 싣지 않는다."""
        view = build_ext_report_view(_report(), {})
        self.assertIsNone(view["score"]["pass_threshold"])

    def test_gate_detail_is_still_carried(self):
        """판정만 빼고 근거는 남긴다 — 진단 범위 섹션이 composite 구성을 보여준다."""
        view = build_ext_report_view(_report(), {})
        self.assertIn("gate", view["score"])
        self.assertIn("composite_total", view["score"]["gate"])

    def test_template_hides_badge_without_verdict(self):
        """템플릿이 None 을 모르면 !!undefined 로 '미달' 배지를 그린다."""
        from tools.report_html import REPORT_TEMPLATE
        if not REPORT_TEMPLATE.exists():
            self.skipTest("report.html 템플릿 없음")
        html = REPORT_TEMPLATE.read_text(encoding="utf-8")
        self.assertIn("score.pass_threshold == null", html)


if __name__ == "__main__":
    unittest.main()
