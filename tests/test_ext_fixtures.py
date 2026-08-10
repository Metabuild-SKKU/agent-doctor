"""
tests/test_ext_fixtures.py
결함 주입 로그(fixtures)가 이름값을 하는지 고정한다.

왜 필요한가: hallucinate 라는 이름을 달고 실제로는 동문서답만 담긴 로그가 한 번
커밋됐다(검색만 어긋내면 모델이 무관하지만 유효한 문서를 충실히 요약해버린다).
진단기가 아니라 **대조군이 틀린** 경우라 진단 테스트로는 잡히지 않는다.

여기서는 LLM 을 부르지 않는다 — 로그의 구조적 사실(답변이 컨텍스트에 있는가,
기권했는가)만 본다. 지표 기반 판정은 replay 통합 실행이 담당한다.
"""
import json
import unittest
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "external_rag"
ABSTAIN_MARKERS = ("확인되지 않습니다", "찾을 수 없습니다", "포함되어 있지 않습니다")


def _load(name):
    path = FIXTURES / f"ext_{name}.jsonl"
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _abstained(answer: str) -> bool:
    return any(m in answer for m in ABSTAIN_MARKERS)


class FixtureShapeTests(unittest.TestCase):

    def test_all_fixtures_follow_schema_v1(self):
        for name in ("none", "starve", "offtopic", "hallucinate", "hallucinate_fixed"):
            rows = _load(name)
            if rows is None:
                continue
            for r in rows:
                self.assertTrue(r["question"].strip(), name)
                self.assertTrue(r["answer"].strip(), name)
                self.assertTrue(r["contexts"], f"{name}: contexts 없음 → tier 가 qa_only 로 떨어진다")
                for c in r["contexts"]:
                    self.assertTrue(c["text"].strip())

    def test_starve_retrieves_less(self):
        """굶주림은 top_k 로 재현한다 — 설정이 로그에 남아야 진단이 현재값을 인용한다."""
        rows = _load("starve")
        if rows is None:
            self.skipTest("fixture 없음")
        self.assertEqual(rows[0]["config"]["top_k"], 1)
        self.assertLess(rows[0]["config"]["chunk_size"], 512)

    def test_normal_log_answers_are_grounded(self):
        """정상 로그는 대조군이다 — 답이 컨텍스트에 근거해야 한다.
        정답(ground_truth)이 검색된 컨텍스트 안에 실제로 있는지로 확인한다."""
        rows = _load("none")
        if rows is None:
            self.skipTest("fixture 없음")
        grounded = sum(
            1 for r in rows
            if r.get("ground_truth")
            and any(r["ground_truth"] in c["text"] for c in r["contexts"])
        )
        self.assertGreaterEqual(grounded, len(rows) // 2,
                                "정상 로그인데 정답 근거가 검색되지 않았다")


class HallucinationFixtureTests(unittest.TestCase):
    """hallucinate 는 '근거 없는 답변'이어야 한다 — 동문서답이 아니라."""

    def test_questions_have_no_answer_in_corpus(self):
        """환각을 유도하려면 근거 자체가 없어야 한다. gold 질문셋을 쓰면
        모델이 컨텍스트를 요약해버려 faithfulness 가 1.00 으로 나온다."""
        rows = _load("hallucinate")
        if rows is None:
            self.skipTest("fixture 없음")
        # 코퍼스에 답이 없는 질문이라 정답지를 실을 수 없다.
        self.assertTrue(all(not r.get("ground_truth") for r in rows))

    def test_hallucinate_mostly_fabricates(self):
        """대부분 기권하지 않고 답해야 환각 대조군이 된다."""
        rows = _load("hallucinate")
        if rows is None:
            self.skipTest("fixture 없음")
        answered = [r for r in rows if not _abstained(r["answer"])]
        self.assertGreaterEqual(len(answered), len(rows) // 2,
                                "전부 기권했다면 환각 대조군이 아니다")

    def test_fixed_variant_abstains_instead(self):
        """처방(인용 강제 + 기권 강화) 적용본은 반대로 대부분 기권해야 한다.
        이게 '처방이 듣는다'의 구조적 증거다."""
        rows = _load("hallucinate_fixed")
        if rows is None:
            self.skipTest("fixture 없음")
        abstained = [r for r in rows if _abstained(r["answer"])]
        self.assertGreater(len(abstained), len(rows) // 2,
                           "처방 적용본인데 여전히 지어내고 있다")

    def test_ab_pair_shares_questions(self):
        """A/B 는 같은 질문·같은 검색 설정이어야 점수 차이를 처방 탓으로 귀속할 수 있다."""
        before, after = _load("hallucinate"), _load("hallucinate_fixed")
        if before is None or after is None:
            self.skipTest("fixture 없음")
        self.assertEqual([r["question"] for r in before], [r["question"] for r in after])
        self.assertEqual(before[0]["config"]["top_k"], after[0]["config"]["top_k"])
        self.assertEqual(before[0]["config"]["chunk_size"], after[0]["config"]["chunk_size"])


if __name__ == "__main__":
    unittest.main()
