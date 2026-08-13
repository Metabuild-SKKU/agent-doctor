"""테스트 스위트 공통 설정.

**여기서 정하는 값은 테스트 전용이다. 프로덕션 기본값이 아니다.**
색인·질의 임베딩의 실제 기본값은 둘 다 `openrouter` 이고
(agents/index/qdrant_store.py 의 resolve_embedding_provider /
resolve_query_embedding_provider), 이 파일은 그걸 테스트 동안만 `local` 로 덮는다.
이 파일만 보고 "기본값이 local" 이라고 읽지 말 것.

임베딩 provider 를 local 로 못 박는다. 색인·질의 기본값이 openrouter 라, 이게 없으면
테스트가 실행 머신의 `.env` 에 따라 갈린다:

  - `OPENROUTER_API_KEY` 가 있는 개발 머신: 임베딩 테스트가 **실제 API 를 호출해
    과금**된다. 스위트 전체를 한 번 돌릴 때마다 조용히 돈이 나간다.
  - 키가 없는 CI: retriever 의 preflight 가 예외로 끊어 검색 관련 테스트가 무더기로
    깨진다(그 preflight 자체는 의도된 동작이다).

둘 다 "테스트가 환경에 의존한다" 는 같은 문제의 양면이라 여기서 한 번에 못 박는다.
API 경로를 검증하는 테스트는 `openai_embed` 를 대역으로 바꿔 확인하므로 이 고정에
영향받지 않는다(tests/test_embedding_provider.py 는 setUp 에서 자기 값을 덮어쓴다).

리랭크 provider 도 같은 이유로 local 로 고정한다. `.env` 에
INDEX_RERANKER_PROVIDER=openrouter 를 둔 머신에서 스위트를 돌리면 리랭커 테스트들이
API 경로로 새고, probe_reranker_capability 의 smoke inference 가 openrouter.ai 로
실제 POST 를 날린다 — 리랭크는 질의마다 과금되는 축이라 임베딩보다 더 조용히 샌다.
API 경로 테스트(tests/test_reranker_route.py)는 patch.dict 로 자기 값을 덮어쓴다.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _pin_embedding_provider_to_local():
    """테스트 동안만 local 로 고정한다(프로덕션 기본값은 openrouter)."""
    keys = {
        "INDEX_EMBED_PROVIDER": "local",
        "INDEX_QUERY_EMBED_PROVIDER": "local",
        "INDEX_RERANKER_PROVIDER": "local",
        # Eval 채점 임베딩축. 여기만 "" 인 이유는 이 축의 기본값이 local 이 아니라
        # "심판 provider 를 따름" 이기 때문이다 - 빈 값이 그 기본을 뜻한다
        # (llm_provider._embed_provider_override 가 빈 값을 "미지정"으로 읽는다).
        # 이게 없으면 실행 머신의 .env 에 EVAL_EMBED_PROVIDER=openrouter 가 있을 때
        # 심판축 대역만 걸어둔 테스트가 임베딩축으로 새어 **실제 API 를 호출**한다
        # (tests/test_provider_notices.py 의 clear=False 경로가 그렇다).
        # 셸 export 만의 이야기가 아니다 - graph.py 가 load_dotenv(override=True) 라
        # 그걸 import 하는 테스트가 먼저 수집되면 .env 가 os.environ 에 들어온다.
        "EVAL_EMBED_PROVIDER": "",
    }
    previous = {name: os.environ.get(name) for name in keys}
    os.environ.update(keys)
    yield
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
