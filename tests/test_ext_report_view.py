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
        self.assertEqual(ext["tier"], "triad")
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
        from tests.report_html import REPORT_TEMPLATE, inject_view
        if not REPORT_TEMPLATE.exists():
            self.skipTest("report.html 템플릿 없음")
        html = inject_view({"score": {"after": 1}},
                           REPORT_TEMPLATE.read_text(encoding="utf-8"))
        # fetch 실패 배너를 띄우는 더미 분기가 남으면 안 된다
        self.assertNotIn("renderReport({}, false)", html)
        self.assertIn("__AGENT_DOCTOR_REPORT__", html)

    def test_data_block_precedes_render(self):
        """데이터가 렌더 스크립트보다 뒤에 오면 undefined 라 빈 리포트가 그려진다."""
        from tests.report_html import REPORT_TEMPLATE, inject_view
        if not REPORT_TEMPLATE.exists():
            self.skipTest("report.html 템플릿 없음")
        html = inject_view({"score": {"after": 1}},
                           REPORT_TEMPLATE.read_text(encoding="utf-8"))
        self.assertLess(html.index("window.__AGENT_DOCTOR_REPORT__ ="),
                        html.index("renderReport(window.__AGENT_DOCTOR_REPORT__"))

    def test_script_close_is_escaped(self):
        """</script> 가 payload 에 들어가면 HTML 파싱이 깨진다."""
        from tests.report_html import REPORT_TEMPLATE, inject_view
        if not REPORT_TEMPLATE.exists():
            self.skipTest("report.html 템플릿 없음")
        html = inject_view({"meta": {"corpus": "</script><b>x"}},
                           REPORT_TEMPLATE.read_text(encoding="utf-8"))
        self.assertNotIn("</script><b>x", html)
        self.assertIn("<\\/script>", html)


if __name__ == "__main__":
    unittest.main()
