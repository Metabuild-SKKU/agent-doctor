"""
core/run_registry.py
웹 프로토타입이 트리거하는 파이프라인 실행을 추적하는 프로세스 내 메모리 저장소.

agents/serve/web_api.py 의 백그라운드 스레드(실행 주체)와 HTTP 폴링 핸들러(조회 주체)가
같은 run_id 키로 이 dict를 주고받는다. DB/큐 없이 단일 프로세스 로컬 프로토타입 용도.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RunEvent:
    ts: float
    stage: str
    tag: str
    text: str
    kind: str = ""  # "" | "ok" | "find" — index.html 의 addLine() 세 번째 인자와 대응


@dataclass
class RunState:
    run_id: str
    status: str = "queued"  # queued | running | done | error
    stage: str = ""         # AgentDoctorState.current_agent 미러
    iteration: int = 0
    max_iterations: int = 3
    percent: int = 0
    events: list[RunEvent] = field(default_factory=list)
    error: Optional[str] = None
    final_state: Optional[Any] = None  # 완료 시 AgentDoctorState
    depth: str = "standard"
    upload_path: str = ""
    created_at: float = 0.0


_LOCK = threading.Lock()
_RUNS: dict[str, RunState] = {}


def create(run_id: str, depth: str, upload_path: str, created_at: float) -> RunState:
    with _LOCK:
        run = RunState(run_id=run_id, depth=depth, upload_path=upload_path, created_at=created_at)
        _RUNS[run_id] = run
        return run


def get(run_id: str) -> Optional[RunState]:
    with _LOCK:
        return _RUNS.get(run_id)


def update(run_id: str, **fields: Any) -> None:
    with _LOCK:
        run = _RUNS.get(run_id)
        if run is None:
            return
        for key, value in fields.items():
            setattr(run, key, value)


def add_event(run_id: str, stage: str, tag: str, text: str, kind: str = "", ts: float = 0.0) -> None:
    with _LOCK:
        run = _RUNS.get(run_id)
        if run is None:
            return
        run.events.append(RunEvent(ts=ts, stage=stage, tag=tag, text=text, kind=kind))


def events_since(run_id: str, cursor: int) -> tuple[list[RunEvent], int]:
    with _LOCK:
        run = _RUNS.get(run_id)
        if run is None:
            return [], cursor
        events = run.events[cursor:]
        return list(events), len(run.events)
