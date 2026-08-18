"""테스트가 임베딩 provider 를 스스로 local 로 못박는 헬퍼.

**왜 테스트마다 박아야 하나.** 색인·질의 임베딩의 기본값은 둘 다 `openrouter` 다
(qdrant_store 의 resolve_embedding_provider / resolve_query_embedding_provider).
고정하지 않으면 검색 경로를 타는 테스트의 결과가 실행 머신의 `.env` 에 달린다 —
`OPENROUTER_API_KEY` 가 없으면 Retriever.search_with_details 의 설정 preflight 가
RuntimeError 로 끊고, 있으면 통과한다. 테스트가 코드가 아니라 환경을 재는 셈이다.

**왜 conftest.py 로 부족한가.** conftest 는 pytest 전용이다. README/CLAUDE.md 가
안내하는 `python -m unittest discover` 는 conftest 를 아예 읽지 않는다. 실제로 두
러너의 결과가 갈렸다(pytest 초록 / unittest 빨강).

**왜 tests/__init__.py 의 import 시점이 아니라 setUp 인가.** graph.py 는 import
시점에 `load_dotenv(override=True)` 를 부른다. graph 를 import 하는 테스트 모듈이
하나라도 있으면(현재 6개) 패키지 import 때 박아둔 값이 그 뒤 `.env` 값으로 덮인다 —
실측으로 local → openrouter 로 되돌아간다. 모든 import 가 끝난 뒤에 도는 setUp 에서
박아야 확실하다.
"""
from __future__ import annotations

from unittest.mock import patch


LOCAL_EMBEDDING_ENV = {
    "INDEX_EMBED_PROVIDER": "local",
    "INDEX_QUERY_EMBED_PROVIDER": "local",
}


def pin_local_embedding(test_case) -> None:
    """setUp 에서 부른다 — 두 축을 local 로 덮고 정리는 addCleanup 에 맡긴다.

    질의축을 빼먹으면 안 된다. embed() 는 색인축이 아니라 질의축을 읽는다
    (qdrant_store.embed 독스트링)."""
    patcher = patch.dict("os.environ", LOCAL_EMBEDDING_ENV)
    patcher.start()
    test_case.addCleanup(patcher.stop)
