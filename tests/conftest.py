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
    }
    previous = {name: os.environ.get(name) for name in keys}
    os.environ.update(keys)
    yield
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
