"""
agents/eval/topic_cluster.py
STEP4 후처리: retrieval_semantic_mismatch 실패의 토픽 분포 신호(topic_cluster).

목적: dense·BM25 둘 다 놓친 의미 불일치(retrieval_semantic_mismatch)가 코퍼스 전반에
흩어져 나오는지, 특정 주제에 몰려 나오는지를 판정해 Optimize 처방을 가른다
(rules.py 의 retrieval_semantic_mismatch.applies_when.topic_cluster):
    "concentrated" → 특정 도메인만 약함     → 도메인특화 임베딩 교체
    "spread"       → 전 주제에 흩어져 실패   → 임베딩 모델 자체 교체
    "none"         → 재봤으나 응집 안 보임   → 청킹 조정(임베딩 문제 아님)
    "unmeasured"   → 아예 재지 못함          → planner 순차 fallback

"none" 과 "unmeasured" 는 다르다 — 전자는 "쟀고, 임베딩 문제가 아니라는 결론"이라 rules.py
에서 청킹 처방을 고르는 근거가 되고, 후자는 "근거 자체가 없음"이라 신호를 무시하고 기존
순차 fallback 으로 돌아가야 한다. 둘을 같은 값으로 뭉개면 못 잰 회차가 청킹 처방을 확정
선택해버린다(rules.py 의 applies_when 은 "unmeasured" 를 어느 허용 리스트에도 담지 않고,
planner 가 미측정 신호를 fallback 으로 처리한다).

판정은 개별 probe 로는 불가능하다 — "실패가 뭉쳤나 흩어졌나"는 여러 실패 probe 의 gold 를
함께 봐야 나온다. 그래서 diagnose() 밖(agent.py STEP4 직후, 전 record 가 준비된 뒤)에서
_annotate_topic_cluster 가 이 모듈을 호출한다.

비용: 코사인만 쓴다(LLM 0). baseline·실패 gold 양쪽 다 표본을 잘라 O(n^2) 폭발을 막는다.
임베딩은 state.chunks 에 이미 실려 있어(Chunk.embedding) 별도 조회가 없다 —
knowledge_graph 와 같은 소스·같은 cosine 을 쓴다.

절대 코사인 임계값은 쓰지 않는다: 코퍼스마다 임베딩이 깔린 수준이 달라(같은 도메인 문서는
무관 쌍도 cos 0.4~0.5) 절대값으로는 "뭉침"을 못 가른다. 그래서 실패 gold 응집도를 코퍼스
전체 평균 응집도(baseline)로 나눈 비율로 상대 판정한다(KG_TOP_K 와 같은 상대화 철학).
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

from agents.eval.knowledge_graph import cosine
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
# 어느 applies_when 허용 리스트에도 없는 값 — planner 가 '신호 없음'으로 보아 fallback 한다.
UNMEASURED = "unmeasured"


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
            total += cosine(list(usable[i]), list(usable[j]))
            count += 1
    return total / count if count else None


def stride_sample(vectors: list[Vector], limit: int = TOPIC_CLUSTER_BASELINE_SAMPLE) -> list[Vector]:
    """전체에 고르게 걸친 결정적 표본 — O(n^2) 폭발을 막되 순서 편향은 만들지 않는다.

    앞에서 limit 개를 자르면 안 된다: chunk 순서는 곧 문서 순서라, head-100 은 보통 문서
    1~2개 = 단일 주제다. 그 좁은 표본의 응집도가 baseline 이 되면 baseline 이 크게 부풀려져
    (실측 시뮬: 전체 0.082 vs head-100 0.538, 약 7배) 모든 ratio 가 눌린다 — 실패가 실제로
    뭉쳐 있어도 "코퍼스 대비 안 뭉침"으로 읽혀 concentrated 를 놓친다.

    전 구간에 고르게 걸친 인덱스를 직접 계산해 뽑으므로 그 편향이 사라지고, 무작위가 아니라
    결정적이라 회차 간 신호도 흔들리지 않는다(probe_gen 전반의 결정성 원칙과 동일).

    정수 stride(`vectors[::n//limit]`)를 쓰면 안 된다: n < 2*limit 구간에서 몫이 1 로
    뭉개져 앞에서 limit 개를 자른 것과 똑같아진다(n=150·limit=100 이면 앞 100개 = 코퍼스의
    66% 만 훑는다). 그 구간이 바로 head 편향이 되살아나는 지점이라, 실측으로 concentrated
    코퍼스가 spread 로 뒤집히는 걸 확인했다.
    """
    n = len(vectors)
    if limit <= 0 or n <= limit:
        return list(vectors)
    # i*n//limit — 마지막 표본이 항상 끝 구간에서 나오도록 전 구간을 균등 분할한다.
    return [vectors[i * n // limit] for i in range(limit)]


class TopicClusterResult(NamedTuple):
    """classify 의 판정 + 그 근거 수치. 버킷만으로는 캘리브레이션을 못 하므로 함께 남긴다.

    소비 유예(관측용) 동안에도 이 수치가 finding.metadata 에 쌓여야, 임계값
    (CONCENTRATED/SPREAD_RATIO)을 실측 분산에 맞춰 재보정할 근거가 된다.
    unmeasured 면 ratio·baseline·failed_cohesion 은 None 이고, 표본 수만 남는다.
    """
    bucket: str
    ratio: Optional[float]
    failed_cohesion: Optional[float]
    baseline: Optional[float]
    failed_sample_size: int   # baseline·failed 모두 _valid 통과 + 표본 절단 후 수(응집도 계산 실입력 수).
    corpus_sample_size: int


def classify_detail(
    failed_gold_vectors: list[Vector],
    corpus_vectors: list[Vector],
) -> TopicClusterResult:
    """classify 와 같은 판정을 하되, 근거 수치(ratio·baseline·표본 수)를 함께 돌려준다.

    버킷 규칙은 classify 문서 참고. 표본 수는 _valid 를 통과하고 표본 절단까지 거친
    (=응집도 계산에 실제 들어간) 벡터 수라, unmeasured 원인이 "2개 미만"인지
    "baseline 불가"인지 구분된다. 전량 개수가 아니어야 하는 이유: 추정량 분산은 실입력
    표본 수에 달려 있어, 캘리브레이션 근거로는 절단 후 수가 맞다.

    양쪽 입력 모두 여기서 _valid 필터 → stride_sample 순으로 정규화한다. 순서가 뒤집히면
    영벡터/미부착 벡터가 표본 슬롯을 먼저 잡아먹고 뒤에 걸러져, 실측 유효분이 반토막 나
    실측 신호가 있는데도 unmeasured 로 떨어진다(유효 3 + 영벡터 297 → 표본 유효 1개).
    호출부는 이 규칙(_valid 의 정의·절단 상한)을 몰라도 되게 raw 벡터를 그대로 넘긴다.
    """
    failed_usable = stride_sample([v for v in failed_gold_vectors if _valid(v)])
    corpus_usable = stride_sample([v for v in corpus_vectors if _valid(v)])
    failed_n = len(failed_usable)
    corpus_n = len(corpus_usable)

    failed_cohesion = _mean_pairwise_cosine(failed_usable)
    if failed_cohesion is None:
        return TopicClusterResult(UNMEASURED, None, None, None, failed_n, corpus_n)
    baseline = _mean_pairwise_cosine(corpus_usable)
    # 음수 baseline(코퍼스가 서로 등질) 도 막는다 — 나누면 ratio 부호가 뒤집혀
    # 실패가 아무리 뭉쳐 있어도 무조건 spread 로 떨어진다.
    if baseline is None or baseline <= 0:
        return TopicClusterResult(UNMEASURED, None, failed_cohesion, baseline, failed_n, corpus_n)
    ratio = failed_cohesion / baseline
    if ratio >= TOPIC_CLUSTER_CONCENTRATED_RATIO:
        bucket = CONCENTRATED
    elif ratio <= TOPIC_CLUSTER_SPREAD_RATIO:
        bucket = SPREAD
    else:
        bucket = NONE
    return TopicClusterResult(bucket, ratio, failed_cohesion, baseline, failed_n, corpus_n)


def classify(
    failed_gold_vectors: list[Vector],
    corpus_vectors: list[Vector],
) -> str:
    """실패 gold 응집도를 코퍼스 baseline 대비 비율로 판정한다.

    반환: "concentrated" | "spread" | "none" | "unmeasured".
    - 실패 gold 가 2개 미만이거나 baseline 을 못 재면 "unmeasured"(판정 불가 → fallback).
    - ratio = 실패 gold 응집도 / baseline.
        ratio >= CONCENTRATED_RATIO → "concentrated"
        ratio <= SPREAD_RATIO       → "spread"
        그 사이                     → "none"

    근거 수치(ratio·표본 수 등)가 필요하면 classify_detail 을 쓴다 — 이 함수는 그
    버킷 문자열만 노출하는 얇은 래퍼다(기존 호출부·계약 유지).
    """
    return classify_detail(failed_gold_vectors, corpus_vectors).bucket
