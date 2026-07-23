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
import time
import heapq
from collections import Counter
from dataclasses import dataclass, field

from core.schema import Chunk
from agents.index.qdrant_store import resolve_embedding_device
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

def build_graph(chunks: list[Chunk], config: dict | None = None) -> KGraph:
    """
    청크 리스트로 KG를 만든다. 전량 휴리스틱(무료) — LLM 호출 없음.
    임베딩 후보는 GPU/CPU 블록 행렬곱으로 청크별 top-k만 고르고, 키워드 후보는
    역색인으로 좁힌다. 두 후보 중 하나라도 임계값을 넘으면 연결한다.
    """
    config = config or {}
    top_k = _positive_int(config.get("eval_graph_top_k", 12), "eval_graph_top_k")
    batch_size = _positive_int(
        config.get("eval_graph_batch_size", 512),
        "eval_graph_batch_size",
    )
    requested_device = str(config.get("eval_graph_device", "auto") or "auto")
    started = time.perf_counter()

    nodes: dict[str, KGNode] = {}
    embeddings: list[list[float] | None] = []
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
        embeddings.append(c.embedding)

    ids = list(nodes.keys())
    semantic_pairs, actual_device = _embedding_candidate_pairs(
        embeddings,
        requested_device,
        top_k,
        batch_size,
    )
    keyword_pairs = _keyword_candidate_pairs(
        [nodes[chunk_id] for chunk_id in ids],
        top_k,
    )

    edges: dict[str, list[tuple[str, float]]] = {cid: [] for cid in ids}
    for left_index, right_index in set(semantic_pairs) | keyword_pairs:
        left_id, right_id = ids[left_index], ids[right_index]
        cosine = semantic_pairs.get((left_index, right_index))
        if cosine is None:
            cosine = _cosine(embeddings[left_index], embeddings[right_index])
        weight = _edge_weight(
            nodes[left_id],
            nodes[right_id],
            embeddings[left_index],
            embeddings[right_index],
            cosine=cosine,
        )
        if weight is None:
            continue
        edges[left_id].append((right_id, weight))
        edges[right_id].append((left_id, weight))

    for cid in edges:
        edges[cid].sort(key=lambda pair: pair[1], reverse=True)
    edge_count = sum(len(neighbors) for neighbors in edges.values()) // 2
    print(
        f"[Eval] STEP1 KG: 노드 {len(ids)}개, 엣지 {edge_count}개, "
        f"device={actual_device}, top_k={top_k}, block={batch_size}, "
        f"{time.perf_counter() - started:.3f}초"
    )
    return KGraph(nodes=nodes, edges=edges)


def _positive_int(value: object, name: str) -> int:
    """그래프 성능 설정을 양의 정수로 검증한다."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name}은 1 이상의 정수여야 합니다.")
    return value


def _valid_embedding_rows(
    embeddings: list[list[float] | None],
) -> tuple[list[int], list[list[float]]]:
    """같은 차원의 유효 임베딩만 행렬 연산 입력으로 모은다."""
    dimension = next((len(vector) for vector in embeddings if vector), 0)
    if dimension <= 0:
        return [], []
    indices: list[int] = []
    vectors: list[list[float]] = []
    for index, vector in enumerate(embeddings):
        if vector and len(vector) == dimension:
            indices.append(index)
            vectors.append(vector)
    return indices, vectors


def _embedding_candidate_pairs(
    embeddings: list[list[float] | None],
    device: str,
    top_k: int,
    batch_size: int,
) -> tuple[dict[tuple[int, int], float], str]:
    """블록 행렬곱으로 각 청크의 top-k 임베딩 이웃만 반환한다."""
    indices, vectors = _valid_embedding_rows(embeddings)
    if len(vectors) < 2:
        return {}, "none"

    actual_device = resolve_embedding_device(device)
    try:
        return (
            _torch_embedding_candidates(
                indices,
                vectors,
                actual_device,
                top_k,
                batch_size,
            ),
            actual_device,
        )
    except Exception as exc:
        if actual_device.startswith("cuda"):
            print(f"[Eval] STEP1 KG GPU 실패, CPU 행렬곱으로 전환: {exc}")
            try:
                return (
                    _torch_embedding_candidates(
                        indices,
                        vectors,
                        "cpu",
                        top_k,
                        batch_size,
                    ),
                    "cpu",
                )
            except Exception as cpu_exc:
                print(f"[Eval] STEP1 KG PyTorch 실패, Python 폴백 사용: {cpu_exc}")
        else:
            print(f"[Eval] STEP1 KG PyTorch 실패, Python 폴백 사용: {exc}")
        return _python_embedding_candidates(indices, vectors, top_k), "cpu-python"


def _torch_embedding_candidates(
    indices: list[int],
    vectors: list[list[float]],
    device: str,
    top_k: int,
    batch_size: int,
) -> dict[tuple[int, int], float]:
    """PyTorch에서 정규화·블록 행렬곱·top-k를 한 번에 수행한다."""
    import torch
    import torch.nn.functional as functional

    matrix = torch.as_tensor(vectors, dtype=torch.float32, device=device)
    matrix = functional.normalize(matrix, p=2, dim=1)
    neighbor_count = min(top_k, len(indices) - 1)
    pairs: dict[tuple[int, int], float] = {}

    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        similarities = matrix[start:end] @ matrix.T
        local_rows = torch.arange(end - start, device=device)
        global_rows = torch.arange(start, end, device=device)
        similarities[local_rows, global_rows] = float("-inf")
        values, neighbors = torch.topk(similarities, k=neighbor_count, dim=1)
        values_cpu = values.detach().cpu().tolist()
        neighbors_cpu = neighbors.detach().cpu().tolist()

        for local_index, (scores, neighbor_rows) in enumerate(
            zip(values_cpu, neighbors_cpu)
        ):
            left = indices[start + local_index]
            for score, neighbor_row in zip(scores, neighbor_rows):
                if score < KG_EMBEDDING_SIM_MIN:
                    break
                right = indices[neighbor_row]
                pair = (left, right) if left < right else (right, left)
                pairs[pair] = max(pairs.get(pair, -1.0), float(score))

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return pairs


def _python_embedding_candidates(
    indices: list[int],
    vectors: list[list[float]],
    top_k: int,
) -> dict[tuple[int, int], float]:
    """PyTorch가 없는 최소 환경에서 기존 Python 코사인 방식으로 폴백한다."""
    pairs: dict[tuple[int, int], float] = {}
    for left_row, left_vector in enumerate(vectors):
        candidates = (
            (_cosine(left_vector, right_vector), right_row)
            for right_row, right_vector in enumerate(vectors)
            if right_row != left_row
        )
        for score, right_row in heapq.nlargest(top_k, candidates):
            if score < KG_EMBEDDING_SIM_MIN:
                continue
            left, right = indices[left_row], indices[right_row]
            pair = (left, right) if left < right else (right, left)
            pairs[pair] = max(pairs.get(pair, -1.0), float(score))
    return pairs


def _keyword_candidate_pairs(
    nodes: list[KGNode],
    top_k: int,
) -> set[tuple[int, int]]:
    """키워드 역색인에서 청크별 유력 후보만 골라 전수 비교를 피한다."""
    inverted: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        for keyword in set(node.keywords):
            inverted.setdefault(keyword, []).append(index)

    # 대규모 코퍼스에서 거의 모든 청크에 등장하는 단어는 연결 신호가 아니라
    # 불용어에 가까우므로 후보 확장을 막는다. 작은 테스트 코퍼스는 전량 유지한다.
    max_posting = len(nodes) if len(nodes) <= 500 else max(50, int(len(nodes) * 0.05))
    pairs: set[tuple[int, int]] = set()
    for left, node in enumerate(nodes):
        overlap_counts: Counter[int] = Counter()
        for keyword in set(node.keywords):
            posting = inverted.get(keyword, [])
            if len(posting) > max_posting:
                continue
            for right in posting:
                if right != left:
                    overlap_counts[right] += 1
        for right, _count in overlap_counts.most_common(top_k):
            if _keyword_jaccard(node.keywords, nodes[right].keywords) < KG_ENTITY_OVERLAP_MIN:
                continue
            pair = (left, right) if left < right else (right, left)
            pairs.add(pair)
    return pairs


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
    a: KGNode,
    b: KGNode,
    emb_a: list[float] | None,
    emb_b: list[float] | None,
    *,
    cosine: float | None = None,
) -> float | None:
    """
    a, b 를 연결할지와 그 가중치를 함께 결정한다.
    entity_jaccard, embedding_cosine 둘 다 각자의 임계값(KG_ENTITY_OVERLAP_MIN,
    KG_EMBEDDING_SIM_MIN) 미만이면 연결하지 않음(None). 연결되면 가중치는
    0.6*jaccard + 0.4*cosine (임베딩이 더 믿을 만한 신호지만, 임베딩이 없는
    경우—예: mock 데이터—에도 키워드만으로 동작하도록 키워드 쪽에 더 큰 비중).
    """
    jaccard = _keyword_jaccard(a.keywords, b.keywords)
    cosine = _cosine(emb_a, emb_b) if cosine is None else cosine
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
