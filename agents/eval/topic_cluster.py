"""
agents/eval/topic_cluster.py
STEP4 후처리: retrieval_semantic_mismatch 실패의 토픽 분포 신호(topic_cluster).

목적: dense·BM25 둘 다 놓친 의미 불일치(retrieval_semantic_mismatch)가 코퍼스 전반에
흩어져 나오는지, 특정 주제에 몰려 나오는지를 판정해 Optimize 처방을 가른다
(rules.py 의 retrieval_semantic_mismatch.applies_when.topic_cluster):
    "concentrated" → 특정 도메인만 약함     → 도메인특화 임베딩 교체
    "spread"       → 전 주제에 흩어져 실패   → 임베딩 모델 자체 교체
    "none"         → 신호 약함/판정 불가     → planner 순차 fallback

판정은 개별 probe 로는 불가능하다 — "실패가 뭉쳤나 흩어졌나"는 여러 실패 probe 의 gold 를
함께 봐야 나온다. 그래서 diagnose() 밖(agent.py STEP4 직후, 전 record 가 준비된 뒤)에서
_annotate_topic_cluster 가 이 모듈을 호출한다.

비용: 코사인만 쓴다(LLM 0). baseline 은 표본을 잘라 O(n^2) 폭발을 막는다. 임베딩은
state.chunks 에 이미 실려 있어(Chunk.embedding) 별도 조회가 없다 — knowledge_graph 와
같은 소스·같은 _cosine 을 쓴다.

절대 코사인 임계값은 쓰지 않는다: 코퍼스마다 임베딩이 깔린 수준이 달라(같은 도메인 문서는
무관 쌍도 cos 0.4~0.5) 절대값으로는 "뭉침"을 못 가른다. 그래서 실패 gold 응집도를 코퍼스
전체 평균 응집도(baseline)로 나눈 비율로 상대 판정한다(KG_TOP_K 와 같은 상대화 철학).
"""
from __future__ import annotations

from typing import Optional, Sequence

from agents.eval.knowledge_graph import _cosine
from agents.eval.types import (
    TOPIC_CLUSTER_BASELINE_SAMPLE,
    TOPIC_CLUSTER_CONCENTRATED_RATIO,
    TOPIC_CLUSTER_SPREAD_RATIO,
)

Vector = Sequence[float]

# 값 도메인 — rules.py 의 applies_when.topic_cluster 리스트와 문자열이 일치해야 한다.
CONCENTRATED = "concentrated"
SPREAD = "spread"
NONE = "none"


def _valid(vec: Optional[Vector]) -> bool:
    """실측 임베딩인가 — None·빈 벡터(임베딩 미부착/fallback 흔적)를 거른다."""
    return bool(vec) and any(vec)


def _mean_pairwise_cosine(vectors: list[Vector]) -> Optional[float]:
    """벡터 집합의 평균 쌍별 코사인(응집도). 쌍이 없으면 None.

    O(m^2) 지만 호출부가 표본을 잘라 m 을 작게 유지한다(실패 gold 는 보통 수십 개,
    baseline 은 TOPIC_CLUSTER_BASELINE_SAMPLE 로 상한).
    """
    usable = [v for v in vectors if _valid(v)]
    if len(usable) < 2:
        return None
    total = 0.0
    count = 0
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            total += _cosine(list(usable[i]), list(usable[j]))
            count += 1
    return total / count if count else None


def _baseline_cohesion(corpus_vectors: list[Vector]) -> Optional[float]:
    """코퍼스 전체의 평균 쌍별 코사인. 표본을 잘라 O(n^2) 폭발을 막는다(결정적: 앞에서 자름)."""
    usable = [v for v in corpus_vectors if _valid(v)]
    if len(usable) < 2:
        return None
    # 무작위 대신 앞에서 자른다 — 결정적이어야 회차 간 신호가 흔들리지 않는다
    # (probe_gen 전반의 결정성 원칙과 동일). chunk 순서는 문서 순서라 편향이 크지 않다.
    sample = usable[:TOPIC_CLUSTER_BASELINE_SAMPLE]
    return _mean_pairwise_cosine(sample)


def classify(
    failed_gold_vectors: list[Vector],
    corpus_vectors: list[Vector],
) -> str:
    """실패 gold 응집도를 코퍼스 baseline 대비 비율로 판정한다.

    반환: "concentrated" | "spread" | "none".
    - 실패 gold 가 2개 미만이거나 baseline 을 못 재면 "none"(판정 불가).
    - ratio = 실패 gold 응집도 / baseline.
        ratio >= CONCENTRATED_RATIO → "concentrated"
        ratio <= SPREAD_RATIO       → "spread"
        그 사이                     → "none"
    """
    failed_cohesion = _mean_pairwise_cosine(failed_gold_vectors)
    if failed_cohesion is None:
        return NONE
    baseline = _baseline_cohesion(corpus_vectors)
    if not baseline:            # None 또는 0 — 나눌 수 없음
        return NONE
    ratio = failed_cohesion / baseline
    if ratio >= TOPIC_CLUSTER_CONCENTRATED_RATIO:
        return CONCENTRATED
    if ratio <= TOPIC_CLUSTER_SPREAD_RATIO:
        return SPREAD
    return NONE
