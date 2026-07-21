"""
agents/eval/probe_store.py
STEP1: Probe(골든 테스트셋) 영속화

목적: generate_probes() 는 LLM 호출 비용이 드는데, 문서가 안 바뀌었으면 매 반복
(Optimize 루프)마다 다시 생성할 필요가 없다 — 같은 코퍼스 버전이면 이전에 만든
Probe 를 그대로 재사용(골든 테스트셋)한다.

버전 키: corpus_version — 원문 문서(doc_id + content) 해시. Probe의 질문과
gold_spans는 원문에 의존하므로 chunk_size/overlap이 바뀌어도 같은 Probe를 재사용하고,
현재 청킹에 맞춰 gold_chunk_ids만 다시 계산한다. 원문 문서가 없는 legacy 호출만
청크 id + 텍스트 해시를 사용한다. index_config(top_k 등)는 버전 키에서 제외한다.
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

from core.schema import Chunk, Document, Probe

DEFAULT_STORE_PATH = "eval_probes.json"
# gold_spans 생성 계약이 바뀌면 이전 Probe 캐시를 재사용하지 않는다.
PROBE_SCHEMA_VERSION = "gold-spans-v2"


def corpus_version(
    chunks: list[Chunk],
    documents: list[Document] | None = None,
) -> str:
    """Probe 캐시 무효화 키 — 원문 문서가 있으면 청킹 결과와 분리한다.

    Probe 질문과 원문 gold_spans는 원문 문서에 의존하므로 documents가 있으면
    doc_id+content를 해싱한다. 이 경우 chunk_size/overlap 변경은 버전을 바꾸지 않아
    같은 Probe를 재사용하고 현재 청크에 gold_chunk_ids만 다시 맞출 수 있다.
    documents가 없는 legacy 호출은 기존처럼 청크 id+텍스트를 해싱한다."""
    h = hashlib.sha1()
    h.update(PROBE_SCHEMA_VERSION.encode("utf-8"))
    h.update(b"\x00")
    if documents:
        h.update(b"documents\x00")
        for document in sorted(documents, key=lambda item: item.doc_id):
            h.update(document.doc_id.encode("utf-8"))
            h.update(b"\x00")
            h.update((document.content or "").encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:12]

    h.update(b"chunks\x00")
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


def load_probes(
    version: str, path: str = DEFAULT_STORE_PATH, *, ignore_version: bool = False
) -> list[Probe] | None:
    """
    저장된 Probe 리스트를 반환, 없거나(파일 없음·손상·스키마 불일치) 버전이 어긋나면 None
    (호출부가 generate_probes 로 재생성하도록).

    ignore_version=True 면 버전 검사를 건너뛰고 저장된 Probe 를 그대로 재사용한다
    (EVAL_PROBE_SOURCE=made — 코퍼스가 바뀌어도 고정 테스트셋으로 진단하고 싶을 때).
    """
    if not os.path.exists(path):
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not ignore_version and data.get("version") != version:
        return None
    try:
        return [Probe(**p) for p in data.get("probes", [])]
    except TypeError:
        return None  # 스키마가 바뀐 옛 캐시 — 재생성으로 폴백
