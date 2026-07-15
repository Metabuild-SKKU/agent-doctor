"""
agents/eval/knowledge_graph.py
STEP1: 청크 간 관계를 나타내는 지식그래프(KG) 구축

RAGAS TestsetGenerator 방식의 멀티홉 질문 생성은 "관련 있는 청크 쌍/트리플"을
먼저 찾아야 한다. 이 모듈은 그 후보를 찾는 그래프(노드=청크, 엣지=관련도)만
만든다 — 실제 질문 합성(LLM 호출)은 probe_gen.py 쪽 다음 단계에서 이 그래프를
입력으로 받아 처리한다.

[구현 포인트] (다음 단계로 남겨둠)
    - 지금은 노드 enrich가 전부 휴리스틱(빈도 기반 키워드)이다. 실제 시나리오로
      샘플링된 소수 노드에 한해서만 LLM 기반 요약/entity 추출(_llm_enrich)을
      지연 적용하는 lazy-enrichment가 다음 단계 몫 — 코퍼스 전체를 LLM으로
      enrich하면 testset_size와 무관하게 비용이 청크 수에 비례해버린다.
    - 한국어는 대문자 신호가 없어 keyword==entity로 취급한다(휴리스틱 한계,
      해결하지 않고 남겨둠).
    - _tokenize가 공백 분리라 조사/어미가 안 떨어져("재택근무는" vs "재택근무")
      같은 단어를 써도 키워드 Jaccard가 0이 되기 쉽다(metrics.py::_tokenize와
      동일한 한계) — 이 때문에 사실상 엣지 판정이 임베딩 코사인 쪽에 치우친다.
    - mock 5청크로 실측한 결과, 무관한 주제 간에도 코사인 유사도가 0.5 근처까지
      올라오는 경우가 있어(예: '재택근무' vs '온보딩' 청크 cos=0.562) 이
      임계값에서 노이즈 엣지가 섞일 수 있음을 확인함 — 형태소 분석기로 키워드
      신호를 살리거나, 절대 임계값 대신 top-k 최근접 이웃 방식으로 바꾸면
      개선 여지가 있다. 지금 단계에서는 해결하지 않고 관찰만 기록한다.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from core.schema import Chunk
from agents.eval.types import KG_EMBEDDING_SIM_MIN, KG_ENTITY_OVERLAP_MIN

_TOP_K_KEYWORDS = 8
_PUNCT = re.compile(r"[^\w가-힣]+")
_SENT_SPLIT = re.compile(r"(?<=[.!?。？！\n])\s+")


@dataclass
class KGNode:
    chunk_id: str
    doc_id: str
    text: str
    summary: str = ""
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass
class KGraph:
    nodes: dict[str, KGNode] = field(default_factory=dict)
    edges: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    # edges[chunk_id] = [(neighbor_chunk_id, weight), ...], weight 내림차순 정렬


# ── 그래프 구축 ───────────────────────────────────────────────────

def build_graph(chunks: list[Chunk]) -> KGraph:
    """
    청크 리스트로 KG를 만든다. 전량 휴리스틱(무료) — LLM 호출 없음.
    엣지는 키워드 Jaccard 또는 임베딩 코사인 유사도 중 하나라도 임계값을
    넘으면 연결한다(둘 중 하나만 강해도 관련 있다고 보는 편이, 둘 다 강해야만
    연결하는 것보다 멀티홉 후보를 지나치게 좁히지 않는다).
    """
    nodes: dict[str, KGNode] = {}
    embeddings: dict[str, list[float] | None] = {}
    for c in chunks:
        enrich = _heuristic_enrich(c.text)
        nodes[c.chunk_id] = KGNode(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            text=c.text,
            summary=enrich["summary"],
            entities=enrich["keywords"],
            keywords=enrich["keywords"],
        )
        embeddings[c.chunk_id] = c.embedding

    edges: dict[str, list[tuple[str, float]]] = {cid: [] for cid in nodes}
    ids = list(nodes.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a_id, b_id = ids[i], ids[j]
            weight = _edge_weight(nodes[a_id], nodes[b_id], embeddings[a_id], embeddings[b_id])
            if weight is None:
                continue
            edges[a_id].append((b_id, weight))
            edges[b_id].append((a_id, weight))

    for cid in edges:
        edges[cid].sort(key=lambda pair: pair[1], reverse=True)
    return KGraph(nodes=nodes, edges=edges)


def neighbors(graph: KGraph, chunk_id: str, min_weight: float = 0.0) -> list[str]:
    """chunk_id와 연결된 이웃 청크 id들(가중치 내림차순, min_weight 미만은 제외)."""
    return [nid for nid, w in graph.edges.get(chunk_id, []) if w >= min_weight]


def connected_pairs(graph: KGraph, n: int = 2) -> list[list[str]]:
    """
    멀티홉 시나리오 샘플링용 후보. n=2는 엣지로 직접 연결된 쌍,
    n=3은 셋이 서로 다 연결된(삼각형) 조합만 후보로 삼는다(임의의 3개 조합이
    아니라 실제로 서로 관련 있는 조합만 — 무관한 청크를 엮은 멀티홉 질문
    방지).
    """
    pair_set: set[tuple[str, str]] = set()
    for a_id, neigh in graph.edges.items():
        for b_id, _ in neigh:
            pair_set.add(tuple(sorted((a_id, b_id))))

    if n == 2:
        return [list(pair) for pair in sorted(pair_set)]

    if n == 3:
        ids = sorted(graph.nodes.keys())
        triples = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if (ids[i], ids[j]) not in pair_set:
                    continue
                for k in range(j + 1, len(ids)):
                    if (ids[i], ids[k]) in pair_set and (ids[j], ids[k]) in pair_set:
                        triples.append([ids[i], ids[j], ids[k]])
        return triples

    raise ValueError(f"connected_pairs: n={n} 은 지원하지 않음(2 또는 3만 가능)")


# ── 노드 enrich (휴리스틱 전용, LLM 경로는 다음 단계) ──────────────

def _heuristic_enrich(chunk_text: str) -> dict:
    """LLM 미사용: 첫 문장을 요약으로, 빈도 상위 토큰을 키워드/entity로 사용."""
    text = (chunk_text or "").strip()
    if not text:
        return {"summary": "", "keywords": []}
    summary = _SENT_SPLIT.split(text, maxsplit=1)[0].strip()
    tokens = [t for t in _tokenize(text) if len(t) >= 2]
    keywords = [w for w, _ in Counter(tokens).most_common(_TOP_K_KEYWORDS)]
    return {"summary": summary, "keywords": keywords}


def _tokenize(text: str) -> list[str]:
    """agents/eval/metrics.py::_tokenize 와 동일한 방식(소문자화+구두점 제거+공백 분리)."""
    if not text:
        return []
    text = _PUNCT.sub(" ", text.lower())
    return [t for t in text.split() if t]


# ── 엣지 가중치 ───────────────────────────────────────────────────

def _edge_weight(
    a: KGNode, b: KGNode, emb_a: list[float] | None, emb_b: list[float] | None
) -> float | None:
    """
    a, b 를 연결할지와 그 가중치를 함께 결정한다.
    entity_jaccard, embedding_cosine 둘 다 각자의 임계값(KG_ENTITY_OVERLAP_MIN,
    KG_EMBEDDING_SIM_MIN) 미만이면 연결하지 않음(None). 연결되면 가중치는
    0.6*jaccard + 0.4*cosine (임베딩이 더 믿을 만한 신호지만, 임베딩이 없는
    경우—예: mock 데이터—에도 키워드만으로 동작하도록 키워드 쪽에 더 큰 비중).
    """
    jaccard = _keyword_jaccard(a.keywords, b.keywords)
    cosine = _cosine(emb_a, emb_b)
    if jaccard < KG_ENTITY_OVERLAP_MIN and cosine < KG_EMBEDDING_SIM_MIN:
        return None
    return 0.6 * jaccard + 0.4 * cosine


def _keyword_jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _cosine(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
