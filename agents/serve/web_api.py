"""
agents/serve/web_api.py
web/prototype 정적 프론트엔드가 호출하는 "파이프라인 제어" API.

agents/serve/api.py(단일 chunks.json 기반 검색 API, 8766 포트)와는 별개의 서버다.
여기는 PDF 업로드를 받아 LangGraph 파이프라인(Ingest→Index→Eval→Optimize→Serve)을
백그라운드 스레드로 돌리고, run_id 로 진행 상황과 완료된 리포트를 조회하게 해준다.

Run:
  python agents/serve/web_api.py --port 8767
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core import run_registry
from core.state import AgentDoctorState
from graph import build_graph
from agents.serve.report_view import build_report_view

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"

# 그래프 실제 노드 → index.html UI 5단계 매핑. eval 완료는 probe+diagnose 둘 다 만족시킨다.
_STAGE_ORDER = ["ingest", "index", "eval", "optimize", "serve"]
_STAGE_WEIGHT = {"ingest": 10, "index": 20, "eval": 30, "optimize": 30, "serve": 10}

app = FastAPI(title="Agent Doctor Web API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _save_upload(run_id: str, upload: UploadFile) -> Path:
    run_dir = UPLOAD_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dest = run_dir / (upload.filename or "document.pdf")
    with dest.open("wb") as f:
        f.write(upload.file.read())
    return dest


def _percent_for(stage: str, done: bool) -> int:
    idx = _STAGE_ORDER.index(stage) if stage in _STAGE_ORDER else 0
    completed_weight = sum(_STAGE_WEIGHT[s] for s in _STAGE_ORDER[:idx])
    if done:
        completed_weight += _STAGE_WEIGHT[stage]
    pct = int(completed_weight / sum(_STAGE_WEIGHT.values()) * 100)
    if stage != "serve" or not done:
        pct = min(pct, 95)
    return pct


def _summarize_stage_event(stage: str, snapshot: AgentDoctorState) -> tuple[str, str, str]:
    """(tag, text, kind) 반환 — index.html 티커 라인 포맷과 동일."""
    if stage == "ingest":
        return ("수집", f"문서 {len(snapshot.documents)}건 읽어들임", "ok")
    if stage == "index":
        return ("색인", f"청크 {len(snapshot.chunks)}개 생성", "ok")
    if stage == "eval":
        if snapshot.report:
            summary = snapshot.report.findings_summary or {}
            confirmed = summary.get("confirmed", len(snapshot.report.findings))
            return ("진단", f"테스트 질문 {len(snapshot.probes)}개 · 확정 문제 {confirmed}건 발견", "find" if confirmed else "ok")
        return ("진단", f"테스트 질문 {len(snapshot.probes)}개로 검사", "")
    if stage == "optimize":
        history = snapshot.optimization_history or []
        if history:
            last = history[-1]
            before = last.metadata.get("before_score")
            after = last.metadata.get("after_score")
            if before is not None and after is not None:
                verdict = "유지" if (last.status == "applied" and not last.metadata.get("pending")) else "롤백"
                return ("처방", f"{last.selected_prescription_id or ''} · 종합 {before:.0f}→{after:.0f} {verdict}", "ok" if verdict == "유지" else "find")
        return ("처방", "설정 조정 시도", "")
    if stage == "serve":
        return ("완료", "리포트 준비 완료", "ok")
    return (stage, "진행 중", "")


def _run_pipeline_background(run_id: str, file_path: Path, depth: str) -> None:
    from core.run_logger import setup_run_logging
    setup_run_logging(prefix="web_run")  # Windows 콘솔 인코딩 문제로 print 가 예외를 던지지 않도록 보호

    os.environ["EVAL_MODE"] = "standard"

    run_registry.update(run_id, status="running")
    graph = build_graph()
    initial_state = AgentDoctorState(
        source_url=str(file_path),
        source_type="file",
        status="running",
    )

    try:
        last_state: AgentDoctorState | None = None
        seen_stage_done: set[str] = set()

        for snapshot in graph.stream(initial_state, stream_mode="values"):
            state = AgentDoctorState(**snapshot) if isinstance(snapshot, dict) else snapshot
            last_state = state

            stage = state.current_agent
            if not stage:
                continue

            marker = f"{stage}:{state.iteration}"
            if marker in seen_stage_done:
                continue
            seen_stage_done.add(marker)

            tag, text, kind = _summarize_stage_event(stage, state)
            run_registry.add_event(run_id, stage=stage, tag=tag, text=text, kind=kind, ts=time.time())
            run_registry.update(
                run_id,
                stage=stage,
                iteration=state.iteration,
                max_iterations=state.max_iterations,
                percent=_percent_for(stage, done=True),
            )

        if last_state is None or last_state.status == "error":
            error_msg = last_state.error if last_state else "파이프라인이 결과를 반환하지 않았습니다."
            run_registry.update(run_id, status="error", error=error_msg)
            return

        run_registry.update(run_id, status="done", percent=100, final_state=last_state)
    except Exception as exc:  # noqa: BLE001 — 백그라운드 스레드 최상단이라 반드시 잡아야 함
        run_registry.update(run_id, status="error", error=str(exc))


@app.post("/runs")
async def create_run(file: UploadFile = File(...), depth: str = Form("standard")) -> dict:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 지원합니다.")

    run_id = uuid.uuid4().hex
    dest = _save_upload(run_id, file)
    run_registry.create(run_id, depth=depth, upload_path=str(dest), created_at=time.time())

    import threading
    thread = threading.Thread(target=_run_pipeline_background, args=(run_id, dest, depth), daemon=True)
    thread.start()

    return {"run_id": run_id}


@app.get("/runs/{run_id}/status")
def run_status(run_id: str, since: int = 0) -> dict:
    run = run_registry.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="알 수 없는 run_id")

    events, cursor = run_registry.events_since(run_id, since)
    return {
        "status": run.status,
        "stage": run.stage,
        "iteration": run.iteration,
        "max_iterations": run.max_iterations,
        "percent": run.percent,
        "error": run.error,
        "cursor": cursor,
        "events": [
            {"tag": e.tag, "text": e.text, "kind": e.kind}
            for e in events
        ],
    }


@app.get("/runs/{run_id}/report")
def run_report(run_id: str) -> dict:
    run = run_registry.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="알 수 없는 run_id")
    if run.status == "error":
        raise HTTPException(status_code=500, detail=run.error or "파이프라인 실행 실패")
    if run.status != "done" or run.final_state is None:
        raise HTTPException(status_code=409, detail="아직 완료되지 않았습니다.")

    return build_report_view(run.final_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
