"""agents/eval/knowledge_graph.py — top-k 이웃 그래프 구축 테스트.

멀티홉 후보 페어링(원인1: 무관한 청크가 억지로 엮이던 문제)의 회귀 방지.
임베딩을 손으로 준 2차원 벡터로 결정적으로 검증한다 — BGE-M3 다운로드/API 불필요.
"""
import math
import unittest

from core.schema import Chunk
from agents.eval import knowledge_graph as kg
from agents.eval.types import KG_EMBEDDING_SIM_MIN


def _unit(angle_deg: float) -> list[float]:
    """단위원 위 한 점. 두 벡터의 코사인 = 두 각도 차의 cos 라 유사도를 각도로 제어한다."""
    r = math.radians(angle_deg)
    return [math.cos(r), math.sin(r)]


def _chunk(cid: str, angle: float, text: str | None = None) -> Chunk:
    # 청크마다 겹치지 않는 고유 단어들을 준다 — 키워드 Jaccard 를 0 으로 눌러 엣지 판정을
    # 코사인으로 고정한다(실제 한국어 코퍼스에서 조사·어미 때문에 Jaccard 가 거의 0 이라
    # 코사인이 판정을 지배하는 상황을 재현). 같은 본문을 쓰면 Jaccard=1 이 돼 코사인
    # 하한 가드가 무의미해지므로 반드시 서로 다른 단어를 써야 한다.
    body = text or f"고유단어{cid} 별개주제{cid} 서로다른내용{cid} 최소길이확보문장{cid}"
    return Chunk(chunk_id=cid, doc_id="d", text=body, embedding=_unit(angle))


class TopKGraphTest(unittest.TestCase):
    def test_edges_limited_to_top_k_neighbors(self):
        # A(0°) 주변에 가까운 순서로 B,C,D 를 두고, 모두 하한(cos>=.5, 즉 <=60°) 안에 둔다.
        # top_k=1 이면 A 는 가장 가까운 B 하고만 연결돼야 한다.
        chunks = [
            _chunk("A", 0),
            _chunk("B", 10),   # cos(A,B)=cos10°≈0.985
            _chunk("C", 25),   # cos(A,C)=cos25°≈0.906
            _chunk("D", 40),   # cos(A,D)=cos40°≈0.766
        ]
        graph = kg.build_graph(chunks, top_k=1)
        a_neighbors = {nid for nid, _ in graph.edges["A"]}
        # A 의 top-1 은 B. C·D 는 A 쪽 상위 1 에 못 들지만, 각자의 top-1 이 A 를 가리키면
        # 무방향 엣지로 남을 수 있다 — 여기선 B 가 모두의 최근접이라 A 는 B 하고만 연결.
        self.assertIn("B", a_neighbors)
        self.assertNotIn("D", a_neighbors)

    def test_top_k_larger_admits_more_pairs(self):
        chunks = [_chunk("A", 0), _chunk("B", 10), _chunk("C", 25), _chunk("D", 40)]
        p1 = kg.connected_pairs(kg.build_graph(chunks, top_k=1), n=2)
        p3 = kg.connected_pairs(kg.build_graph(chunks, top_k=3), n=2)
        self.assertLessEqual(len(p1), len(p3))

    def test_floor_guard_drops_unrelated_pair(self):
        # 90° 벌어진 두 청크는 cos=0 < KG_EMBEDDING_SIM_MIN. top-k 가 이웃을 채우려 해도
        # 바닥 가드가 무관 쌍을 끊어야 한다 — 후보가 하나도 없어야 정상.
        assert KG_EMBEDDING_SIM_MIN > 0.0
        chunks = [_chunk("A", 0), _chunk("B", 90)]
        graph = kg.build_graph(chunks, top_k=5)
        self.assertEqual(graph.edges["A"], [])
        self.assertEqual(kg.connected_pairs(graph, n=2), [])

    def test_edges_sorted_by_weight_desc(self):
        # A 에 서로 다른 거리의 이웃을 두고 top_k 를 넉넉히 주면, 인접 리스트가
        # 가중치 내림차순이어야 한다(레버 C 의 소비 순서가 여기에 의존).
        chunks = [_chunk("A", 0), _chunk("B", 10), _chunk("C", 30), _chunk("D", 55)]
        graph = kg.build_graph(chunks, top_k=3)
        weights = [w for _, w in graph.edges["A"]]
        self.assertEqual(weights, sorted(weights, reverse=True))


if __name__ == "__main__":
    unittest.main()
