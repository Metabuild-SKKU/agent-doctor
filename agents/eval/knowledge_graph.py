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
import os
import re
from collections import Counter
from dataclasses import dataclass, field

from core.schema import Chunk
from agents.eval.types import (
    KG_EMBEDDING_SIM_MIN,
    KG_ENTITY_OVERLAP_MIN,
    KG_TOP_K_NEIGHBORS,
)

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


def _env_int(name: str, default: int) -> int:
    """환경변수 int 파싱 - 비정수/오타면 기본값(core, index 와 같은 규약)."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def build_graph(chunks: list[Chunk], top_k: int = KG_TOP_K_NEIGHBORS) -> KGraph:
    """
    청크 리스트로 KG를 만든다. 전량 휴리스틱(무료) — LLM 호출 없음.

    관련도 판정은 top-k 최근접 이웃 방식이다: 각 청크를 가중치 상위 k개 이웃하고만
    연결한다(상호 연결이면 어느 쪽에서 뽑혔든 엣지로 남긴다). 예전엔 모든 쌍에 대해
    절대 임계값 OR(jaccard≥.2 OR cos≥.5)을 걸었는데, 실측(HR 41청크·BGE-M3)에서
    무관 쌍끼리도 cos 중앙값이 0.46 이라 같은 도메인 문서가 통째로 임계값을 넘어
    후보의 78%가 cos<0.6 노이즈였다(억지 멀티홉의 근원). top-k 는 각 청크 기준
    "상대적으로 가장 가까운" 이웃만 남겨, 코퍼스 전체가 높/낮게 깔려도 자동 보정된다.
    절대 임계값(_edge_weight 의 KG_*_MIN)은 top-k 가 억지로 끌어온 먼 이웃을 끊는
    바닥 가드로만 남는다.
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

    ids = list(nodes.keys())
    # 1) 바닥 가드를 통과하는 모든 쌍의 가중치를 계산해 각 노드의 후보 이웃을 모은다.
    #    쌍 수가 O(n^2) 라 청크가 늘면 순수 파이썬 루프로는 감당이 안 된다 — 실측으로
    #    6,584청크(2,167만 쌍)에 약 16분이고, 그동안 진행률 로그가 없어 "멈춤" 과
    #    구분되지 않는다. numpy 가 있으면 행렬곱 두 번으로 같은 값을 1초에 낸다.
    candidates = _candidate_edges_vectorized(ids, nodes, embeddings, top_k)
    if candidates is None:
        candidates = {cid: [] for cid in nodes}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a_id, b_id = ids[i], ids[j]
                weight = _edge_weight(nodes[a_id], nodes[b_id], embeddings[a_id], embeddings[b_id])
                if weight is None:
                    continue
                candidates[a_id].append((b_id, weight))
                candidates[b_id].append((a_id, weight))

    # 2) 각 노드에서 상위 k개만 엣지로 승격한다(top-k). a→b 또는 b→a 어느 방향이든
    #    상위에 들면 엣지를 남긴다(상호 최근접이 아니어도 연결 — 후보가 지나치게
    #    좁아지는 것을 막는다). 무방향 엣지라 양쪽 인접 리스트에 같은 가중치로 넣는다.
    kept: dict[tuple[str, str], float] = {}
    for cid, cand in candidates.items():
        cand.sort(key=lambda pair: pair[1], reverse=True)
        for nid, w in cand[:top_k] if top_k > 0 else cand:
            kept[tuple(sorted((cid, nid)))] = w

    edges: dict[str, list[tuple[str, float]]] = {cid: [] for cid in nodes}
    for (a_id, b_id), w in kept.items():
        edges[a_id].append((b_id, w))
        edges[b_id].append((a_id, w))
    for cid in edges:
        edges[cid].sort(key=lambda pair: pair[1], reverse=True)
    return KGraph(nodes=nodes, edges=edges)


def _candidate_edges_vectorized(
    ids: list[str],
    nodes: dict[str, KGNode],
    embeddings: dict[str, list[float] | None],
    top_k: int,
) -> dict[str, list[tuple[str, float]]] | None:
    """_edge_weight 를 모든 쌍에 대해 계산한다. numpy 가 없거나 노드가 2개 미만이면 None.

    같은 값을 다른 방법으로 낸다:
      코사인  = 행 정규화 후 E @ E.T          (영벡터·임베딩 없음은 0 행 → 코사인 0)
      자카드  = 역색인으로 교집합을 세고,
                합집합 = |a| + |b| - 교집합    (한쪽이 비면 0 — cosine() 규약과 같음)
      가중치  = 0.6*jaccard + 0.4*cosine, 단 둘 다 임계값 미만이면 엣지 없음

    **행 블록 단위로 돈다.** 전체 n x n 행렬을 한 번에 만들면 6,584청크에서 실측 peak
    2.76GB 였다(sim/jac/weight/argsort 출력이 각각 n^2). 블록으로 끊으면 n^2 짜리
    중간산물이 O(블록 x n) 으로 떨어진다 — 시간은 여전히 O(n^2) 지만 그쪽은 행렬곱이라
    감당할 수 있다.

    블록에 안 걸리는 상주 메모리는 두 가지뿐이고 둘 다 O(n)이다: 임베딩 행렬
    (n x 차원, 6,584청크면 54MB)과 키워드 역색인(실측 0.4MB). 키워드를 이진행렬로
    세웠다면 여기가 n x |어휘| 밀집이라 969MB 로 블록의 노력을 무의미하게 만들었다 —
    역색인을 쓰는 이유가 그거다(아래 주석).

    호출부가 노드당 상위 top_k 만 쓰므로 그 절단도 블록 안에서 끝낸다. 통과한 쌍을
    전부 파이썬으로 옮기면 다시 O(n^2) 가 되는데, 임베딩이 서로 비슷한 코퍼스에서는
    대부분의 쌍이 바닥 가드를 통과해 수천만 건이 된다.

    float64 로 계산한다. float32 면 메모리가 절반이지만 가중치가 ~1e-7 어긋나,
    임계값(0.2/0.5) 바로 위아래 쌍에서 엣지 포함 여부가 뒤집힐 수 있다. 그러면 이
    최적화가 '같은 결과를 빠르게' 가 아니라 '결과가 조금 다름' 이 되어, 나중에 회귀를
    의심하게 만든다.

    [파리티의 유일한 한계] 가중치가 ULP 수준(~1e-16)으로 같은 후보 여럿이 top_k 경계에
    정확히 걸치면, 그중 누가 뽑히는지는 루프 경로와 갈릴 수 있다. 행렬곱과 파이썬 루프는
    덧셈 순서가 달라 같은 쌍의 마지막 자리가 다르게 떨어지기 때문이고(루프 경로끼리도
    쌍마다 0.64 / 0.6400000000000001 로 갈린다), 어느 쪽으로도 고칠 수 없다. 실제로
    걸리려면 임베딩이 정확히 같은 청크(반복되는 머리말·상용구 등)가 있어야 하고, 그때
    탈락하는 후보는 가중치가 같은 동률이라 어느 쪽을 남겨도 등가다. 임베딩이 아예 없어
    자카드만으로 판정하는 코퍼스에서는 이 문제가 없다 — 이진행렬 곱은 float64 에서
    정확해서 동률이 동률로 유지된다(테스트로 고정).
    """
    n = len(ids)
    if n < 2:
        return None
    try:
        import numpy as np
    except ImportError:
        return None

    dim = next((len(embeddings[cid]) for cid in ids if embeddings.get(cid)), 0)
    if dim:
        # 차원이 섞인 코퍼스는 행렬로 담을 수 없다. 짧은 쪽을 0 으로 패딩하면 코사인이
        # 달라지는데(루프 경로의 cosine() 은 zip 이라 긴 쪽을 잘라 계산한다) 그 차이가
        # 임계값을 넘나들어 엣지가 통째로 바뀐다 — 실측으로 60청크에서 엣지 절반이
        # 갈렸다. 어느 쪽 셈이 옳은지는 이 커밋이 정할 문제가 아니므로(여기서 정하면
        # 최적화가 아니라 동작 변경이다) 루프 경로로 넘긴다. 정상 코퍼스는 한 모델로
        # 임베딩하므로 이 분기를 타지 않는다.
        if any(len(v) != dim for v in embeddings.values() if v):
            return None
        emb = np.zeros((n, dim), dtype=np.float64)
        for row, cid in enumerate(ids):
            vec = embeddings.get(cid)
            if vec:
                emb[row] = vec
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        # 영벡터·임베딩 없음은 0 행으로 남긴다 → 그 행의 코사인이 전부 0.
        np.divide(emb, norms, out=emb, where=norms > 0)
    else:
        emb = None

    # 키워드 교집합은 역색인(키워드 → 그 키워드를 가진 행 번호)으로 센다. 이진행렬 B 를
    # 세워 B @ B.T 로 내도 값은 같지만, 그 행렬이 n x |어휘| 밀집이라 블록 스킴 밖에서
    # 통째로 상주한다 — 어휘는 청크 수에 비례해 늘어서(실측 3,880청크 11,284어휘 /
    # 6,584청크 18,391어휘) 메모리가 사실상 n^2 로 자란다. 실측 6,584청크에서 밀집은
    # 969MB·17.7s 인데 역색인은 0.4MB·1.5s 이고 교집합 행렬은 비트 단위로 같다.
    # 밀집이 이렇게 손해인 이유: 청크당 키워드가 _TOP_K_KEYWORDS(8)개로 묶여 있어
    # 실제 밀도가 0.04% 라, 행렬의 99.96% 가 0 이다.
    keyword_rows = [sorted(set(nodes[cid].keywords)) for cid in ids]
    rows_by_keyword: dict[str, list[int]] = {}
    for row, kws in enumerate(keyword_rows):
        for kw in kws:
            rows_by_keyword.setdefault(kw, []).append(row)
    # 누적할 때 fancy index 로 한 번에 더하려고 배열로 굳힌다.
    postings = {kw: np.array(rws) for kw, rws in rows_by_keyword.items()}
    kw_sizes = np.array([len(kws) for kws in keyword_rows], dtype=np.float64)

    limit = n if top_k <= 0 else min(top_k, n)
    # 기본 512(실측: 6,584청크 x 1024차원, 실제 코퍼스 어휘 18,391, 3회 중앙값).
    # 블록별 시간/이 루프가 더 쓰는 메모리:
    #   128=10.3s/126MB   256=10.1s/198MB   512=9.7s/329MB   1024=11.0s/546MB   2048=11.5s/1025MB
    # 시간은 128~512 구간이 6% 안이라 사실상 평평하다 — 키워드 교집합이 밀집 행렬곱이
    # 아니게 되면서(위 역색인) 블록이 좌우하는 건 코사인 행렬곱과 argsort 뿐이라 그렇다.
    # 그래서 이 값은 '시간을 사는 손잡이' 가 아니라 거의 순수한 메모리 손잡이다. 중앙값이
    # 가장 낮은 512 를 기본으로 두되, 메모리가 빠듯하면 128 까지 낮춰도 시간 손해가
    # 6% 안이다(-203MB). 2048 이상은 시간·메모리가 같이 나빠져 올릴 이유가 없다.
    block = max(1, _env_int("EVAL_KG_BLOCK_ROWS", 512))
    candidates: dict[str, list[tuple[str, float]]] = {cid: [] for cid in ids}

    for begin in range(0, n, block):
        stop = min(begin + block, n)
        rows = stop - begin

        sim = emb[begin:stop] @ emb.T if emb is not None else np.zeros((rows, n))
        if postings:
            inter = np.zeros((rows, n))
            for local, kws in enumerate(keyword_rows[begin:stop]):
                for kw in kws:
                    inter[local, postings[kw]] += 1.0
            union = kw_sizes[begin:stop, None] + kw_sizes[None, :] - inter
            jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        else:
            jac = np.zeros((rows, n))

        weight = 0.6 * jac + 0.4 * sim
        keep = (jac >= KG_ENTITY_OVERLAP_MIN) | (sim >= KG_EMBEDDING_SIM_MIN)
        keep[np.arange(rows), np.arange(begin, stop)] = False   # 자기 자신 제외
        weight[~keep] = -np.inf

        # 내림차순 안정 정렬 — 가중치가 같으면 열 인덱스가 작은 쪽이 먼저다(결정적).
        order = np.argsort(-weight, axis=1, kind="stable")[:, :limit]
        for local, cid in enumerate(ids[begin:stop]):
            row_w = weight[local]
            for col in order[local].tolist():
                w = row_w[col]
                if w == -np.inf:
                    break            # 정렬돼 있으므로 뒤는 전부 탈락
                candidates[cid].append((ids[col], float(w)))

    return candidates


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
    similarity = cosine(emb_a, emb_b)
    if jaccard < KG_ENTITY_OVERLAP_MIN and similarity < KG_EMBEDDING_SIM_MIN:
        return None
    return 0.6 * jaccard + 0.4 * similarity


def _keyword_jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def cosine(a: list[float] | None, b: list[float] | None) -> float:
    """두 임베딩의 코사인 유사도. 한쪽이 비었거나 영벡터면 0.0.

    모듈 밖에서도 쓰인다(topic_cluster) — 같은 임베딩 소스에 같은 셈을 적용해야
    신호끼리 비교가 성립하기 때문에 공개 헬퍼로 둔다.

    TODO(eval-정리) metrics_ragas._cosine / index.graph_index._cosine 에 같은 구현이
    따로 있다. 셋을 core 쪽 한 곳으로 합치는 건 이 파일 밖의 별도 과제.
    """
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
