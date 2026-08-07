"""agents/eval/knowledge_graph.py — top-k 이웃 그래프 구축 테스트.

멀티홉 후보 페어링(원인1: 무관한 청크가 억지로 엮이던 문제)의 회귀 방지.
임베딩을 손으로 준 2차원 벡터로 결정적으로 검증한다 — BGE-M3 다운로드/API 불필요.
"""
import math
import os
import random
import sys
import unittest
from unittest.mock import patch

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


class VectorizedEdgeParityTests(unittest.TestCase):
    """벡터화 경로가 루프 경로와 같은 그래프를 만드는지 고정한다.

    build_graph 는 모든 청크 쌍을 도는 O(n^2) 라, 순수 파이썬 루프로는 6,584청크에
    약 16분이 걸린다(진행률 로그가 없어 '멈춤' 과 구분되지 않는다). numpy 행렬곱으로
    바꾸되 **결과가 같아야** 한다 — 다르면 최적화가 아니라 동작 변경이고, 나중에
    검색 점수가 흔들릴 때 원인을 여기서 찾을 수 없게 된다.
    """

    @staticmethod
    def _chunks(n, seed, *, dim=32, drop_emb=0.0, no_kw=0.0):
        rng = random.Random(seed)
        vocab = [f"키워드{i}" for i in range(40)]
        out = []
        for i in range(n):
            words = "" if rng.random() < no_kw else " ".join(
                rng.sample(vocab, rng.randint(1, 6)))
            emb = None if rng.random() < drop_emb else [
                rng.random() for _ in range(dim)]
            out.append(Chunk(chunk_id=f"c{i}", doc_id="d",
                             text=f"본문 {i} {words}", embedding=emb))
        return out

    @staticmethod
    def _loop_graph(chunks, top_k=kg.KG_TOP_K_NEIGHBORS):
        """벡터화를 꺼서 기존 루프 경로를 강제한다."""
        with patch.object(kg, "_candidate_edges_vectorized",
                          return_value=None):
            return kg.build_graph(chunks, top_k=top_k)

    def _assert_same(self, chunks, label, top_k=kg.KG_TOP_K_NEIGHBORS):
        fast = kg.build_graph(chunks, top_k=top_k)
        slow = self._loop_graph(chunks, top_k=top_k)
        fast_edges = {(cid, nid): w for cid, lst in fast.edges.items() for nid, w in lst}
        slow_edges = {(cid, nid): w for cid, lst in slow.edges.items() for nid, w in lst}
        self.assertEqual(set(fast_edges), set(slow_edges), f"{label}: 엣지 집합이 다르다")
        for key, weight in fast_edges.items():
            self.assertAlmostEqual(weight, slow_edges[key], places=12,
                                   msg=f"{label}: {key} 가중치가 다르다")

    def test_matches_loop_on_normal_corpus(self):
        self._assert_same(self._chunks(60, seed=7), "일반")

    def test_matches_loop_when_some_embeddings_missing(self):
        # 임베딩이 없는 청크는 코사인 0 이어야 한다(cosine() 규약).
        self._assert_same(self._chunks(60, seed=8, drop_emb=0.3), "임베딩 일부 없음")

    def test_matches_loop_when_all_embeddings_missing(self):
        # mock 데이터처럼 임베딩이 아예 없으면 키워드만으로 엣지가 결정된다.
        self._assert_same(self._chunks(60, seed=9, drop_emb=1.0), "임베딩 전부 없음")

    def test_matches_loop_when_some_keywords_missing(self):
        self._assert_same(self._chunks(60, seed=10, no_kw=0.3), "키워드 일부 없음")

    def test_matches_loop_across_row_block_boundary(self):
        """행 블록 단위로 도므로 블록 경계에서 엣지가 새거나 빠지면 안 된다."""
        with patch.dict(os.environ, {"EVAL_KG_BLOCK_ROWS": "16"}):
            self._assert_same(self._chunks(40, seed=11), "블록 경계")

    def test_falls_back_to_loop_without_numpy(self):
        """numpy 가 없는 환경에서도 그래프가 만들어져야 한다(느릴 뿐)."""
        chunks = self._chunks(30, seed=12)
        with patch.dict(sys.modules, {"numpy": None}):
            graph = kg.build_graph(chunks)
        self.assertEqual(set(graph.nodes), {c.chunk_id for c in chunks})
        self.assertEqual(graph.edges, self._loop_graph(chunks).edges)

    def test_self_is_not_a_neighbor(self):
        graph = kg.build_graph(self._chunks(30, seed=13))
        for cid, neigh in graph.edges.items():
            self.assertNotIn(cid, [nid for nid, _ in neigh], f"{cid} 가 자기 이웃이다")

    def test_matches_loop_when_top_k_takes_everything(self):
        """top_k<=0 은 절단 없음이다 — 벡터화가 limit 를 n 으로 열어야 같아진다."""
        for top_k in (0, -1, 999):
            with self.subTest(top_k=top_k):
                self._assert_same(self._chunks(40, seed=14), f"top_k={top_k}", top_k=top_k)

    def test_empty_embedding_list_is_cosine_zero_not_a_dim(self):
        """빈 리스트 임베딩은 '차원 0' 이 아니라 '코사인 0' 이다(cosine() 규약).

        차원 판정에 빈 리스트가 끼면 전체 임베딩 행렬이 사라져 코사인 신호가 통째로
        0 이 된다 — 엣지가 자카드만으로 결정돼 그래프가 조용히 달라진다.
        """
        chunks = self._chunks(40, seed=15)
        for i in range(0, 40, 4):
            chunks[i].embedding = []
        self._assert_same(chunks, "빈 리스트 임베딩")

    def test_mixed_embedding_dims_fall_back_to_loop(self):
        """차원이 섞이면 행렬로 담을 수 없다 — 패딩하지 말고 루프로 넘겨야 한다.

        짧은 쪽을 0 으로 패딩한 코사인은 루프 경로의 cosine()(zip 이라 긴 쪽을 자른다)과
        값이 다르고, 그 차이가 임계값을 넘나들어 엣지가 통째로 갈린다(실측 60청크에서
        절반). 어느 셈이 옳은지는 이 최적화가 정할 문제가 아니라 값이 갈리면 안 된다.
        """
        chunks = self._chunks(40, seed=16, dim=8)
        chunks[3].embedding = chunks[3].embedding[:5]
        ids = [c.chunk_id for c in chunks]
        nodes, embs = {}, {}
        for c in chunks:
            enrich = kg._heuristic_enrich(c.text)
            nodes[c.chunk_id] = kg.KGNode(
                chunk_id=c.chunk_id, doc_id=c.doc_id, text=c.text,
                summary=enrich["summary"], entities=enrich["keywords"],
                keywords=enrich["keywords"])
            embs[c.chunk_id] = c.embedding
        self.assertIsNone(
            kg._candidate_edges_vectorized(ids, nodes, embs, kg.KG_TOP_K_NEIGHBORS),
            "차원이 섞였는데 벡터화 경로가 결과를 냈다")
        self._assert_same(chunks, "차원 불일치")


if __name__ == "__main__":
    unittest.main()
