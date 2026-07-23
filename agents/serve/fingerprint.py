"""코퍼스 지문 — Serve agent 와 API 서버가 공유하는 순수 함수.

api.py 안에 두면 Serve agent 가 지문 하나 계산하려고 uvicorn/fastapi/qdrant_client 까지
import 하게 되어, 그 패키지들이 없는 환경(테스트 등)에서 Serve 가 통째로 실패한다.
지문 알고리즘이 양쪽에서 드리프트하지 않도록 여기 한 곳에만 두고 둘 다 import 한다.
"""
from __future__ import annotations

import hashlib


def corpus_fingerprint(chunks: list[dict]) -> str:
    """로드된 코퍼스의 지문(정렬된 chunk_id:hash sha1 앞 12자).

    Serve(agent.py)가 chunks.json 에 쓴 코퍼스와 이미 실행 중인 API 가 들고 있는
    코퍼스가 같은지 대조하는 데 쓴다 — 지문이 다르면 이전 파이프라인의 API 가 낡은
    코퍼스를 서빙 중이라는 뜻이다. hash 가 없는(레거시) 청크는 text 로 폴백한다."""
    parts = []
    for chunk in chunks:
        cid = chunk.get("chunk_id", "")
        digest = chunk.get("hash") or hashlib.sha1(
            (chunk.get("text") or "").encode("utf-8")
        ).hexdigest()
        parts.append(f"{cid}:{digest}")
    joined = "|".join(sorted(parts))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
