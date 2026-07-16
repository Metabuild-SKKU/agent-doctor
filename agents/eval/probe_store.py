"""
agents/eval/probe_store.py
STEP1: Probe(골든 테스트셋) 영속화

목적: generate_probes() 는 LLM 호출 비용이 드는데, 문서가 안 바뀌었으면 매 반복
(Optimize 루프)마다 다시 생성할 필요가 없다 — 같은 코퍼스 버전이면 이전에 만든
Probe 를 그대로 재사용(골든 테스트셋)한다.

버전 키: agent.py::_pipeline_version 과 동일하게 "index_config + 코퍼스(청크 id 목록)"
해시를 쓴다. Probe 생성은 index_config 에 의존하지 않지만(청크 텍스트만 씀), 같은
해시를 재사용하면 별도 무효화 로직을 두 벌 관리하지 않아도 된다 — index_config 만
바뀐 경우 불필요하게 재생성되는 손해보다, 캐시 무효화 로직이 어긋나 stale probe 를
쓰는 위험을 피하는 쪽을 택했다.

파일 하나(JSON)에 {version, probes:[...]} 형태로 저장한다. probes 는 Probe 의
필드를 그대로 dict 화한 것 — dataclass 이므로 asdict/생성자 재조립만으로 충분하다.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from core.schema import Probe

DEFAULT_STORE_PATH = "eval_probes.json"


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
