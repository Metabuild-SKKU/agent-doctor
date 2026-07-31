"""
tests/test_topic_cluster_annotate.py
agent._annotate_topic_cluster 통합 테스트 — 실패 gold 임베딩 → finding.metadata 배선.

리뷰 후속 2건 회귀 고정:
- Low : 실패 gold 도 baseline 과 같은 순서(_valid 필터 → stride 절단)로 표본을 잘라야,
        영벡터가 표본 슬롯을 먼저 잡아먹고 뒤에 걸러져 실측 신호가 unmeasured 로
        떨어지는 걸 막는다. 그 순서 보장은 classify_detail 안에 있으므로(호출부가
        private 유효성 규칙을 몰라도 되게), 여기서는 배선이 그 계약을 타는지만 본다.
- Medium: 버킷뿐 아니라 판정 근거 수치(ratio·baseline·표본 수)를 metadata 에 남겨야,
        소비 유예(관측용) 동안 임계값 캘리브레이션 근거가 finding 에 쌓인다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.eval.agent import _annotate_topic_cluster
from agents.eval.types import EvalRecord
from core.schema import Chunk, Finding, Probe


def _chunk(cid: str, vec) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d1", text="x", embedding=vec)


def _sem_finding(cid: str) -> Finding:
    return Finding(
        finding_id=f"f_{cid}",
        type="retrieval_failure",
        severity="warning",
        description="semantic mismatch",
        label="retrieval_semantic_mismatch",
    )


def _record(probe_id: str, gold_ids: list[str], findings: list[Finding]) -> EvalRecord:
    # missed_gold = gold_chunk_ids - retrieved_chunk_ids. retrieved 를 비워 전 gold 를 '놓침'으로.
    probe = Probe(probe_id=probe_id, question="q", source="taxonomy", gold_chunk_ids=gold_ids)
    return EvalRecord(probe=probe, retrieved_chunk_ids=[], findings=findings)


class AnnotateTopicClusterTest(unittest.TestCase):
    def test_no_sem_findings_is_noop(self):
        rec = _record("p1", ["c1"], [Finding(
            finding_id="f", type="gap", severity="info", description="d", label="corpus_missing_gold")])
        _annotate_topic_cluster([rec], [_chunk("c1", [1.0, 0.0])])
        self.assertNotIn("topic_cluster", rec.findings[0].metadata)

    def test_metadata_carries_bucket_and_detail(self):
        # 실패 gold 가 코퍼스 대비 뭉침 → concentrated + 근거 수치 기록(Medium).
        failed_ids = ["g0", "g1", "g2"]
        chunks = [
            _chunk("g0", [1.0, 0.0, 0.0]),
            _chunk("g1", [0.98, 0.1, 0.0]),
            _chunk("g2", [0.97, 0.15, 0.05]),
            # 코퍼스는 축마다 흩어짐(baseline 낮음)
            _chunk("c3", [0.0, 1.0, 0.0]),
            _chunk("c4", [0.0, 0.0, 1.0]),
            _chunk("c5", [1.0, 1.0, 0.0]),
        ]
        findings = [_sem_finding(g) for g in failed_ids]
        rec = _record("p1", failed_ids, findings)
        _annotate_topic_cluster([rec], chunks)

        for f in findings:
            self.assertEqual(f.metadata["topic_cluster"], "concentrated")
            detail = f.metadata["topic_cluster_detail"]
            self.assertIsNotNone(detail["ratio"])
            self.assertIsNotNone(detail["baseline"])
            self.assertEqual(detail["failed_sample_size"], 3)
            # 임계값도 함께 실려 캘리브레이션 때 그 회차의 판정 기준을 재현할 수 있다.
            self.assertIn("concentrated_ratio", detail)
            self.assertIn("spread_ratio", detail)

    def test_zero_vectors_do_not_starve_valid_sample(self):
        """Low 회귀 — 영벡터가 표본 슬롯을 먼저 먹어 실측 신호가 unmeasured 로 떨어지지 않는다.

        유효 gold 3개 + 영벡터 gold 297개. 잘못된 순서(stride 먼저)면 표본 100개 중
        유효분이 1개로 줄어 쌍이 없어 unmeasured. 올바른 순서(_valid 먼저)면 유효 3개가
        모두 살아 실측 신호가 나온다.
        """
        valid_ids = ["g0", "g1", "g2"]
        zero_ids = [f"z{i}" for i in range(297)]
        failed_ids = valid_ids + zero_ids
        chunks = [
            _chunk("g0", [1.0, 0.0]),
            _chunk("g1", [0.99, 0.1]),
            _chunk("g2", [0.98, 0.15]),
        ]
        chunks += [_chunk(zid, [0.0, 0.0]) for zid in zero_ids]
        # 코퍼스 baseline 용으로 흩어진 청크 추가(유효 gold 만으론 baseline 이 부실).
        chunks += [_chunk("c_a", [0.0, 1.0]), _chunk("c_b", [1.0, 1.0]), _chunk("c_c", [-1.0, 1.0])]

        findings = [_sem_finding(g) for g in valid_ids]
        rec = _record("p1", failed_ids, findings)
        _annotate_topic_cluster([rec], chunks)

        signal = findings[0].metadata["topic_cluster"]
        self.assertNotEqual(signal, "unmeasured",
                            "영벡터가 표본을 굶겨 실측 신호가 unmeasured 로 떨어졌다(순서 회귀)")
        # 유효 3개가 표본에 모두 살아남았는지 근거 수치로도 확인.
        self.assertEqual(findings[0].metadata["topic_cluster_detail"]["failed_sample_size"], 3)


if __name__ == "__main__":
    unittest.main()
