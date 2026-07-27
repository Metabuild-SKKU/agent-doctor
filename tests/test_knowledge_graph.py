"""Eval 질문 생성용 지식그래프의 블록 top-k 회귀 테스트."""
from __future__ import annotations

import math
import unittest

from agents.eval import knowledge_graph as kg
from agents.eval.knowledge_graph import build_graph
from agents.eval.types import KG_EMBEDDING_SIM_MIN
from core.schema import Chunk


def _chunk(chunk_id: str, text: str, embedding=None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        text=text,
        embedding=embedding,
    )


class KnowledgeGraphTests(unittest.TestCase):
    def test_embedding_top_k_connects_nearest_chunks(self):
        graph = build_graph(
            [
                _chunk("c1", "원격 근무 정책", [1.0, 0.0]),
                _chunk("c2", "재택 근무 지침", [0.99, 0.01]),
                _chunk("c3", "연차 사용 안내", [0.0, 1.0]),
            ],
            {
                "eval_graph_device": "cpu",
                "eval_graph_top_k": 1,
                "eval_graph_batch_size": 2,
            },
        )

        self.assertIn("c2", {chunk_id for chunk_id, _score in graph.edges["c1"]})
        self.assertNotIn("c3", {chunk_id for chunk_id, _score in graph.edges["c1"]})

    def test_keyword_inverted_index_keeps_keyword_only_edge(self):
        graph = build_graph(
            [
                _chunk("c1", "alpha beta gamma delta"),
                _chunk("c2", "alpha beta gamma epsilon"),
                _chunk("c3", "vacation annual leave policy"),
            ],
            {
                "eval_graph_device": "cpu",
                "eval_graph_top_k": 1,
                "eval_graph_batch_size": 2,
            },
        )

        self.assertIn("c2", {chunk_id for chunk_id, _score in graph.edges["c1"]})

    def test_invalid_performance_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "eval_graph_top_k"):
            build_graph([], {"eval_graph_top_k": 0})
def _unit(angle_deg: float) -> list[float]:
    """단위원 위 한 점. 두 벡터의 코사인 = 두 각도 차의 cos 라 유사도를 각도로 제어한다."""
    r = math.radians(angle_deg)
    return [math.cos(r), math.sin(r)]


def _angle_chunk(cid: str, angle: float, text: str | None = None) -> Chunk:
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
            _angle_chunk("A", 0),
            _angle_chunk("B", 10),   # cos(A,B)=cos10°≈0.985
            _angle_chunk("C", 25),   # cos(A,C)=cos25°≈0.906
            _angle_chunk("D", 40),   # cos(A,D)=cos40°≈0.766
        ]
        graph = kg.build_graph(chunks, top_k=1)
        a_neighbors = {nid for nid, _ in graph.edges["A"]}
        # A 의 top-1 은 B. C·D 는 A 쪽 상위 1 에 못 들지만, 각자의 top-1 이 A 를 가리키면
        # 무방향 엣지로 남을 수 있다 — 여기선 B 가 모두의 최근접이라 A 는 B 하고만 연결.
        self.assertIn("B", a_neighbors)
        self.assertNotIn("D", a_neighbors)

    def test_top_k_larger_admits_more_pairs(self):
        chunks = [_angle_chunk("A", 0), _angle_chunk("B", 10), _angle_chunk("C", 25), _angle_chunk("D", 40)]
        p1 = kg.connected_pairs(kg.build_graph(chunks, top_k=1), n=2)
        p3 = kg.connected_pairs(kg.build_graph(chunks, top_k=3), n=2)
        self.assertLessEqual(len(p1), len(p3))

    def test_floor_guard_drops_unrelated_pair(self):
        # 90° 벌어진 두 청크는 cos=0 < KG_EMBEDDING_SIM_MIN. top-k 가 이웃을 채우려 해도
        # 바닥 가드가 무관 쌍을 끊어야 한다 — 후보가 하나도 없어야 정상.
        assert KG_EMBEDDING_SIM_MIN > 0.0
        chunks = [_angle_chunk("A", 0), _angle_chunk("B", 90)]
        graph = kg.build_graph(chunks, top_k=5)
        self.assertEqual(graph.edges["A"], [])
        self.assertEqual(kg.connected_pairs(graph, n=2), [])

    def test_edges_sorted_by_weight_desc(self):
        # A 에 서로 다른 거리의 이웃을 두고 top_k 를 넉넉히 주면, 인접 리스트가
        # 가중치 내림차순이어야 한다(레버 C 의 소비 순서가 여기에 의존).
        chunks = [_angle_chunk("A", 0), _angle_chunk("B", 10), _angle_chunk("C", 30), _angle_chunk("D", 55)]
        graph = kg.build_graph(chunks, top_k=3)
        weights = [w for _, w in graph.edges["A"]]
        self.assertEqual(weights, sorted(weights, reverse=True))


if __name__ == "__main__":
    unittest.main()
