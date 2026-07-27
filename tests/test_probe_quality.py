"""
tests/test_probe_quality.py
probe_gen 의 Probe 품질 게이트 단위 테스트.

케이스는 전부 실제 실행 로그(web_run_20260723_180513)에서 뽑았다. 그 실행에서 Finding 의
44%가 bad_gold_answer 였는데, 원인이 RAG 품질이 아니라 "질문 = 정답" 인 쓰레기 Probe 였다.
LLM 응답이 토큰 상한 없이 잘리며 JSON 파싱에 실패 → 휴리스틱 폴백이 원문 조각을 그대로
질문·정답으로 쓴 결과다.
"""
import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.eval.probe_gen import _probe_surplus_count, probe_quality_issue


class ProbeQualityGateTest(unittest.TestCase):
    def test_accepts_normal_probe(self):
        # 정상 Probe 는 통과해야 한다 — 게이트가 과하게 잡으면 Probe 가 고갈된다.
        self.assertIsNone(probe_quality_issue(
            "2026년 1분기 동안 회사가 연구개발비 중 무형자산으로 처리한 금액이 얼마인지 확인해.",
            "2026년 1분기에 무형자산으로 자본화된 연구개발 지출액은 990억 원입니다.",
        ))

    def test_rejects_table_row_as_ground_truth(self):
        # probe_single_specific_002: 표 행을 그대로 질문과 정답으로 만든 것.
        issue = probe_quality_issue(
            "법인세비용차감전순이익 50,466 23,886 26,580 111.3%에 대해 설명해줘.",
            "법인세비용차감전순이익 50,466 23,886 26,580 111.3%",
        )
        self.assertIsNotNone(issue)

    def test_rejects_page_footer_topic(self):
        # probe_multi_specific_016: 페이지 푸터가 엔티티로 쓰인 경우.
        issue = probe_quality_issue(
            "전자공시시스템 dart.fss.or.kr Page 522 그리고 (2) 최대주주의 관계를 설명해줘.",
            "전자공시시스템 dart.fss.or.kr Page 522 (2) 최대주주(법인 또는 단체)의 기본정보",
        )
        self.assertIsNotNone(issue)

    def test_rejects_self_referential_question(self):
        # 질문의 알맹이가 정답에 그대로 들어 있으면 답을 묻는 게 아니라 되풀이하는 것.
        # 길이 미달로 먼저 걸리지 않도록 충분히 긴 주제를 쓴다(사유를 정확히 검증).
        issue = probe_quality_issue(
            "기업회계기준서 제1109호 금융상품의 리스부채 상환 회계처리에 대해 설명해줘.",
            "기업회계기준서 제1109호 금융상품의 리스부채 상환 회계처리는 재무활동으로 분류됩니다.",
        )
        self.assertEqual(issue, "질문이 정답을 그대로 포함(자기참조)")

    def test_rejects_multihop_self_reference_by_fragment(self):
        # 멀티홉 휴리스틱 질문("A 그리고 B의 관계를 설명해줘.")은 topic 전체("A 그리고 B")로는
        # 정답에 통째로 안 들어가 예전 검사를 빠져나갔다. 어미를 떼고 " 그리고 " 로 나눈 각
        # 조각이 정답에 그대로 있으면 자기참조로 잡아야 한다.
        issue = probe_quality_issue(
            "첫 번째 정책의 신청 기한은 매월 마지막 영업일입니다. "
            "그리고 두 번째 정책의 승인 결과는 다음 달 첫 영업일에 통지됩니다.의 관계를 설명해줘.",
            "첫 번째 정책의 신청 기한은 매월 마지막 영업일입니다.\n"
            "두 번째 정책의 승인 결과는 다음 달 첫 영업일에 통지됩니다.",
        )
        self.assertEqual(issue, "질문이 정답을 그대로 포함(자기참조)")

    def test_accepts_multihop_with_relation_prose(self):
        # 관계를 실제로 서술한 정상 멀티홉 정답은 통과해야 한다 — 조각 검사가 과하게 잡으면
        # LLM 이 만든 멀쩡한 멀티홉 Probe 까지 고갈된다.
        self.assertIsNone(probe_quality_issue(
            "재고자산 평가손실 그리고 매출원가율 상승은 어떤 관계인지 설명해줘.",
            "재고자산 평가손실이 매출원가에 반영되면서 매출원가율이 3%p 상승했습니다.",
        ))

    def test_rejects_too_short_question(self):
        self.assertIsNotNone(probe_quality_issue("금액은?", "1,234억 원입니다."))

    def test_numeric_prose_answer_survives(self):
        # 숫자가 섞인 정상 서술형 정답까지 표 행으로 오인하면 안 된다.
        self.assertIsNone(probe_quality_issue(
            "미국 주식시장의 현재 P/E 수준이 다른 국가와 비교해 어느 정도인지 알려주세요.",
            "미국은 22.4배로 EU 15.4배, 일본 17.9배보다 높은 가장 높은 수준입니다.",
        ))

    def test_single_number_factual_answer_survives(self):
        # 금액·인원처럼 숫자 자체가 정답인 짧은 사실형 정답은 살아남아야 한다.
        # 예전 문자 비율 게이트는 "185명"(0.75)·"1,234억 원"(0.75)을 표 행으로 오탐했다.
        # 표 행의 특징은 문자 비율이 아니라 "숫자 열이 여러 개 나열"된다는 것이다.
        for answer in ("185명", "1,234억 원", "전체의 60%", "3.5% 인상되었습니다."):
            with self.subTest(answer=answer):
                self.assertIsNone(probe_quality_issue("직원 수는 몇 명인가요?", answer))


class ProbeSurplusCountTest(unittest.TestCase):
    def test_surplus_scales_with_target(self):
        self.assertEqual(_probe_surplus_count(20), 5)

    def test_surplus_is_at_least_one_for_small_target(self):
        # 반올림하면 0이 되는 작은 목표에서도 최소 1개는 여유를 둔다.
        self.assertEqual(_probe_surplus_count(1), 1)

    def test_no_surplus_for_zero_target(self):
        self.assertEqual(_probe_surplus_count(0), 0)


if __name__ == "__main__":
    unittest.main()
