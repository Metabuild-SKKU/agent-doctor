"""
core/embedding_cli.py
임베딩·리랭크 실행 위치를 명령줄에서 고르는 공용 인자.

왜 필요한가:
    임베딩 provider·device 는 env 로 정해지는데, 파이프라인 엔트리포인트들은
    `load_dotenv(override=True)` 를 쓴다. 셸 값보다 `.env` 가 이기는 규약이라
    `INDEX_EMBED_PROVIDER=local python graph.py` 가 조용히 무시된다 — 셋을
    번갈아 비교하려는 용도에서 정확히 걸림돌이다.

    그래서 플래그는 `load_dotenv` 뒤에 os.environ 에 써넣는다. 우선순위가
    **플래그 > .env > 셸** 이 되어, 비교 실행에서 의도한 값이 항상 이긴다.

사용:
    parser = argparse.ArgumentParser()
    add_embedding_args(parser)
    args = parser.parse_args()
    apply_embedding_args(args)     # load_dotenv 뒤에 부를 것

    python graph.py --embed openrouter
    python graph.py --embed gpu
    python graph.py --embed cpu
    python graph.py --embed openrouter --query-embed cpu   # 질의만 로컬로
    python graph.py --embed openrouter --rerank gpu        # 리랭크는 로컬 GPU 로

주의:
    소스 선택(SOURCE_TYPE/SOURCE_URL)은 기존대로 env 규약을 따른다. 여기서
    플래그로 받는 건 임베딩·리랭크 축뿐이다.

    --embed 와 --rerank 는 **기본값이 반대다**(임베딩 openrouter / 리랭크 local).
    과금 단위가 달라서다 — 임베딩은 색인 1회, 리랭크는 질의마다다. 근거는
    agents/index/qdrant_store.py 의 resolve_reranker_provider 주석.
"""
from __future__ import annotations

import os

# 사용자가 고르는 3지선다 → 코드가 쓰는 2축(provider, device).
# device 는 provider=local 일 때만 의미가 있어 openrouter 는 None 이다.
_TARGETS: dict[str, tuple[str, str | None]] = {
    "openrouter": ("openrouter", None),
    "gpu": ("local", "cuda"),
    "cpu": ("local", "cpu"),
}
EMBED_CHOICES = tuple(_TARGETS)


def add_embedding_args(parser) -> None:
    """--embed / --query-embed / --rerank 를 파서에 단다."""
    parser.add_argument(
        "--embed",
        choices=EMBED_CHOICES,
        help="임베딩을 어디서 계산할지 (기본: env INDEX_EMBED_PROVIDER, 미설정 시 openrouter). "
             "색인과 질의에 함께 적용된다. gpu 는 CUDA 가 없으면 경고 후 cpu 로 내려간다.",
    )
    parser.add_argument(
        "--query-embed",
        choices=EMBED_CHOICES,
        help="질의(검색) 임베딩만 다르게 할 때. 미지정이면 --embed 를 따른다. "
             "색인은 API 로 두고 질의만 로컬로 내리는 조합이 대표적이다 "
             "(429 재시도가 단건 질의 지연으로 그대로 나타날 때).",
    )
    parser.add_argument(
        "--rerank",
        choices=EMBED_CHOICES,
        help="리랭크(cross-encoder)를 어디서 계산할지 (기본: env INDEX_RERANKER_PROVIDER, "
             "미설정 시 local). openrouter 는 질의 1건당 검색 1건으로 과금되고 모델도 "
             "바뀐다(로컬 bge-reranker-v2-m3 → cohere/rerank-v3.5) — 두 실행의 점수는 "
             "직접 비교하지 말 것. --embed 를 따라가지 않는다(기본값이 서로 다르다).",
    )


def apply_embedding_args(args) -> dict[str, str]:
    """플래그를 os.environ 에 반영하고, 실제로 설정한 값을 돌려준다(로그용).

    **load_dotenv 뒤에 불러야 한다.** 먼저 부르면 .env 가 덮어써 플래그가 무시된다.
    미지정 인자는 건드리지 않는다 — 그래야 .env·셸 설정이 그대로 살아 있다.
    """
    applied: dict[str, str] = {}

    target = getattr(args, "embed", None)
    query_target = getattr(args, "query_embed", None) or target

    if target:
        provider, device = _TARGETS[target]
        applied["INDEX_EMBED_PROVIDER"] = provider
        if device:
            applied["INDEX_EMBED_DEVICE"] = device

    if query_target:
        applied["INDEX_QUERY_EMBED_PROVIDER"] = _TARGETS[query_target][0]
        # 질의만 로컬로 내린 경우에도 장치가 필요하다. --embed 가 이미 device 를
        # 정했으면 그 값을 유지하고(같은 로컬 장치를 쓴다), 아니면 여기서 정한다.
        device = _TARGETS[query_target][1]
        if device and "INDEX_EMBED_DEVICE" not in applied:
            applied["INDEX_EMBED_DEVICE"] = device

    rerank_target = getattr(args, "rerank", None)
    if rerank_target:
        provider, device = _TARGETS[rerank_target]
        applied["INDEX_RERANKER_PROVIDER"] = provider
        # 리랭커 장치는 임베딩 축(INDEX_EMBED_DEVICE)을 물려받는 게 기본이라, 여기서는
        # 전용 키에 써야 한다. INDEX_EMBED_DEVICE 에 쓰면 `--embed cpu --rerank gpu` 가
        # 임베딩까지 GPU 로 끌고 간다 — VRAM 이 한쪽만 감당할 때 정확히 쓰고 싶은 조합이다.
        if device:
            applied["INDEX_RERANKER_DEVICE"] = device

    os.environ.update(applied)
    return applied


def describe_embedding_route() -> str:
    """지금 설정으로 임베딩·리랭크가 어디서 계산되는지 한 줄 요약(실행 로그 머리에 찍기용).

    provider 가 env 로 정해져 실행 기록만 봐서는 어디서 계산됐는지 알 수 없다 —
    비용과 속도가 100배 넘게 갈리는 축이라 실행마다 남겨두는 편이 낫다.
    리랭크도 같이 찍는다: 경로에 따라 모델이 바뀌어 점수가 달라지는 데다, API 경로는
    질의마다 과금돼 "왜 이 실행만 비싼가" 의 답이 대개 여기 있다."""
    from agents.index.qdrant_store import (
        openrouter_reranker_model,
        resolve_embedding_device,
        resolve_embedding_provider,
        resolve_query_embedding_provider,
        resolve_reranker_device,
        resolve_reranker_provider,
    )

    def _label(provider: str) -> str:
        if provider != "local":
            return provider
        return f"local:{resolve_embedding_device()}"

    index_route = _label(resolve_embedding_provider())
    query_route = _label(resolve_query_embedding_provider())
    if index_route == query_route:
        embed_line = f"임베딩: {index_route} (색인·질의)"
    else:
        embed_line = f"임베딩: 색인={index_route}, 질의={query_route}"

    rerank_provider = resolve_reranker_provider()
    if rerank_provider == "local":
        rerank_route = f"local:{resolve_reranker_device()}"
    else:
        rerank_route = f"{rerank_provider}:{openrouter_reranker_model()}"
    return f"{embed_line} / 리랭크: {rerank_route}"
