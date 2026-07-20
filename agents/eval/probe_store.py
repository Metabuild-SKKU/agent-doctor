"""
agents/eval/probe_store.py
STEP1: Probe(골든 테스트셋) 영속화

목적: generate_probes() 는 LLM 호출 비용이 드는데, 문서가 안 바뀌었으면 매 반복
(Optimize 루프)마다 다시 생성할 필요가 없다 — 같은 코퍼스 버전이면 이전에 만든
Probe 를 그대로 재사용(골든 테스트셋)한다.

버전 키: corpus_version — 코퍼스(청크 id + 텍스트) 해시. Probe 생성은 청크 텍스트만
쓰고 index_config(top_k 등)에는 의존하지 않으므로, 버전 키에서 index_config 를
제외한다 — top_k 만 바꾼 Optimize 재실행에서 probe(와 그에 딸린 LLM 생성 비용)를
불필요하게 재생성하지 않기 위해서다. 단 chunk_id 는 위치 기반(doc_id_chunk_NNN)이라
재청킹 후 우연히 id 목록이 같아질 수 있어(경계만 이동), 텍스트까지 해싱해 내용 변화가
반드시 버전을 바꾸게 한다 → stale probe 재사용 위험을 막는다.
(진단 신호 캐시는 top_k 에 의존하므로 여전히 agent.py::_pipeline_version 을 쓴다.)

파일 하나(JSON)에 {version, probes:[...]} 형태로 저장한다. probes 는 Probe 의
필드를 그대로 dict 화한 것 — dataclass 이므로 asdict/생성자 재조립만으로 충분하다.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from core.schema import Chunk, Probe

DEFAULT_STORE_PATH = "eval_probes.json"


def corpus_version(chunks: list[Chunk]) -> str:
    """Probe 캐시 무효화 키 — 코퍼스(청크 id + 텍스트)에만 의존.

    index_config(top_k 등)는 probe 내용에 영향이 없어 키에서 제외한다(top_k 만 바뀐
    재실행에서 probe 를 재생성하지 않음). chunk_id 는 위치 기반이라 재청킹 후 목록이
    우연히 같아질 수 있어, 텍스트까지 해싱해 경계 이동(내용 변화)이 반드시 버전을
    바꾸게 한다. 청크 순서에 무관하도록 chunk_id 로 정렬한 뒤 스트리밍 해싱한다."""
    h = hashlib.sha1()
    for c in sorted(chunks, key=lambda c: c.chunk_id):
        h.update(c.chunk_id.encode("utf-8"))
        h.update(b"\x00")
        h.update((c.text or "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


def save_probes(probes: list[Probe], version: str, path: str = DEFAULT_STORE_PATH) -> None:
    """probes 를 버전과 함께 JSON으로 저장. 쓰기 실패(권한 등)는 조용히 무시(캐시일 뿐 필수 아님)."""
    try:
        data = {"version": version, "probes": [asdict(p) for p in probes]}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"[Eval] STEP1: probe 저장 실패({e}) → 다음 실행에서 재생성됨")


def load_probes(version: str, path: str = DEFAULT_STORE_PATH) -> list[Probe] | None:
    """
    version 이 일치하면 저장된 Probe 리스트를 반환, 아니면(버전 불일치·파일 없음·손상) None
    (호출부가 generate_probes 로 재생성하도록).
    """
    if not os.path.exists(path):
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("version") != version:
        return None
    try:
        return [Probe(**p) for p in data.get("probes", [])]
    except TypeError:
        return None  # 스키마가 바뀐 옛 캐시 — 재생성으로 폴백
