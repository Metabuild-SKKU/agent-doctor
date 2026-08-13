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
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.optimize.score_display import display_scores_from_metadata
from core import run_registry
from core.state import AgentDoctorState
from graph import build_graph
from agents.serve.report_view import build_ext_report_view, build_report_view
from agents.eval.replay import diagnose_external_log

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
    return _save_upload_as(run_id, upload, suffix=".pdf")


def _save_upload_as(run_id: str, upload: UploadFile, suffix: str) -> Path:
    """업로드를 run별 폴더에 저장 — 원본 파일명 대신 uuid+고정 확장자를 써서
    경로 탈출·이름 충돌을 막는다(리플레이 로그/골든셋도 같은 규칙을 쓴다)."""
    run_dir = UPLOAD_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dest = (run_dir / f"{uuid.uuid4().hex}{suffix}").resolve()
    if run_dir.resolve() not in dest.parents:
        raise HTTPException(status_code=400, detail="잘못된 업로드 경로입니다.")
    with dest.open("wb") as f:
        f.write(upload.file.read())
    return dest


# 골든셋은 qa_merge.load_qa_set 이 확장자로 파서를 고른다 — 여기서 허용 확장자를
# 한 번 더 거르는 이유는 잘못된 파일이 백그라운드 스레드까지 가서야 죽는 것보다
# 업로드 시점에 바로 400 으로 알리는 게 사용자 경험이 낫기 때문이다.
_GOLDEN_SUFFIXES = (".json", ".jsonl", ".csv", ".xlsx", ".xlsm")

# 로그는 JSONL 내용만 받지만 확장자는 .json 도 허용한다 - 상대가 .json 이름으로 JSONL
# 을 주는 실무 케이스가 흔하고(qa_merge 가 골든셋에 대해 이미 지원하는 그 케이스),
# 내용 검증은 프론트의 validateLogFile 이 줄 단위 파싱으로 이미 한다. 여기서 .jsonl
# 만 받으면 프론트가 통과시킨 파일이 진행 화면 전환 뒤에 400 으로 떨어진다.
_LOG_SUFFIXES = (".jsonl", ".json")


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
            # 표시 점수 변환 규약은 score_display 가 단독으로 갖는다. 여기서 직접
            # 폴백하면 한쪽만 composite 인 이력에서 축이 섞인다(prescreener 경로).
            scores = display_scores_from_metadata(last.metadata)
            verdict = "유지" if (last.status == "applied" and not last.metadata.get("pending")) else "롤백"
            # 무엇을 바꿨는지는 action 이 말한다. 구버전 이력에는 없으므로
            # 처방 id 로 폴백한다(이전 실행의 저장 상태도 계속 읽혀야 한다).
            subject = (
                getattr(last, "action_key", None)
                or last.selected_prescription_id
                or ""
            )
            tone = "ok" if verdict == "유지" else "find"
            if scores.available:
                return (
                    "처방",
                    f"{subject} · 종합 {scores.before:.0f}→{scores.after:.0f} {verdict}",
                    tone,
                )
            # 점수를 못 만드는 경우에도 무엇을 했고 어떻게 판정됐는지는 말해준다.
            # 숫자만 뺀다 — 없는 종합점수를 지어내지 않는다.
            if subject:
                return ("처방", f"{subject} · {verdict}", tone)
        return ("처방", "설정 조정 시도", "")
    if stage == "serve":
        return ("완료", "리포트 준비 완료", "ok")
    return (stage, "진행 중", "")


# index.html 이 노출하는 depth 선택지(fast/standard/full) → EVAL_MODE 매핑.
# "full"은 UI 상 가장 깊은 진단 — EVAL_MODE 쪽에서 DEEP(=LLM/RAGAS 전량)으로 접힌다.
_DEPTH_TO_EVAL_MODE = {"fast": "fast", "standard": "standard", "full": "full"}

# Eval 에이전트가 EVAL_MODE/EVAL_ENABLE_LLM 을 프로세스 전역 환경변수로 읽기 때문에(agents/eval/types.py),
# 백그라운드 스레드 여러 개가 동시에 그래프를 돌리면 서로 값을 덮어쓸 수 있다.
# run 단위로 상태를 스레드에 넘기려면 Eval 내부까지 리팩터링해야 하므로, 대신 이 락으로
# "환경변수 설정 → 그래프 실행" 구간 전체를 직렬화해 실행 중 값이 섞이지 않게 한다.
_PIPELINE_LOCK = threading.Lock()


def _run_pipeline_background(run_id: str, file_path: Path, depth: str) -> None:
    from core.console import force_utf8_stdio
    force_utf8_stdio()   # 콘솔 인코딩 보정(로깅과 독립 — Tee 설치 여부와 무관하게 보호)

    from core.run_logger import setup_run_logging
    setup_run_logging(prefix="web_run")

    run_registry.update(run_id, status="running")

    try:
        with _PIPELINE_LOCK:
            eval_mode = _DEPTH_TO_EVAL_MODE.get(depth, "standard")
            os.environ["EVAL_MODE"] = eval_mode
            os.environ["EVAL_ENABLE_LLM"] = "1" if eval_mode in ("deep", "full") else "0"

            graph = build_graph()
            initial_state = AgentDoctorState(
                source_url=str(file_path),
                source_type="file",
                status="running",
            )

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


def _run_replay_background(
    run_id: str, log_path: Path, golden_path: Path | None,
    preloaded: tuple | None = None,
) -> None:
    """로그 리플레이 모드 — Ingest/Index/Optimize 없이 Eval STEP3~5 만 돈다
    (agents/eval/replay.py 가 이미 하는 일을 웹 실행 단위로 감쌀 뿐).

    preloaded 는 create_run 이 게이트 검사로 이미 파싱한 (logs, errors, qa) 다.
    없으면 여기서 다시 읽는다(테스트·직접 호출 경로).

    파이프라인 모드와 같은 run_registry/이벤트 계약을 쓰므로 프론트엔드
    폴링·진행률 UI 는 그대로 재사용된다 — stage="eval" 로 찍어 STAGE_UI_MAP
    의 probe/diagnose 박스가 활성화되게 한다(리플레이 전용 박스는 없음)."""
    from core.console import force_utf8_stdio
    force_utf8_stdio()

    from core.run_logger import setup_run_logging
    setup_run_logging(prefix="web_replay")

    run_registry.update(run_id, status="running", stage="eval", percent=10)

    try:
        with _PIPELINE_LOCK:
            # 리플레이는 depth 를 받지 않는다 — 항상 LLM 심층으로 돈다(tools/run_replay_report.py
            # 와 같은 처리). 리플레이엔 색인·검색·답변생성이 없어 실제 작업이 RAGAS 채점 하나뿐이라
            # LLM 을 꺼도 아끼는 시간이 거의 없는 반면(실측 6건: 0.03초 vs 13.8초), 끄면 생성축
            # 라벨 4종이 통째로 죽고 검색축도 gold 겹침이 낮을 때만 남아 "점수는 낮은데 소견 0건"
            # 인 진단서가 나간다. 프론트가 무엇을 보내든 여기서 고정해 그 경로를 막는다.
            os.environ["EVAL_MODE"] = "deep"
            os.environ["EVAL_ENABLE_LLM"] = "1"

            run_registry.add_event(
                run_id, stage="eval", tag="적재", text="로그 파일을 읽는 중", ts=time.time(),
            )
            # create_run 이 게이트 검사로 이미 파싱해 둔 것을 그대로 넘긴다.
            # diagnose_external_log 의 logs=/qa= 파라미터가 정확히 이 이중 파싱을
            # 없애려고 있는 것인데(CLI 프리플라이트가 같은 이유로 쓴다), 웹 경로만
            # 안 쓰고 있었다.
            report, cap, errors = diagnose_external_log(
                str(log_path), golden_path=str(golden_path) if golden_path else None,
                logs=preloaded[0] if preloaded else None,
                errors=preloaded[1] if preloaded else None,
                qa=preloaded[2] if preloaded else None,
            )

        if report is None:
            tier = cap.get("tier")
            if tier == "qa_only":
                msg = ("검색 컨텍스트(contexts)가 없거나 부족해 진단이 성립하지 않습니다. "
                       "답변 생성에 쓰인 검색 결과 원문을 로그에 포함해 주세요.")
            else:
                msg = "유효한 로그 레코드가 없어 진단할 수 없습니다."
            run_registry.update(run_id, status="error", error=msg)
            return

        summary = report.findings_summary or {}
        confirmed = summary.get("confirmed", len(report.findings))
        run_registry.add_event(
            run_id, stage="eval", tag="진단",
            text=f"로그 {cap.get('records', 0)}건 · 확정 문제 {confirmed}건 발견",
            kind="find" if confirmed else "ok", ts=time.time(),
        )
        run_registry.update(
            run_id, status="done", percent=100, ext_report=report, ext_cap=cap,
        )
    except Exception as exc:  # noqa: BLE001 — 백그라운드 스레드 최상단이라 반드시 잡아야 함
        run_registry.update(run_id, status="error", error=str(exc))


@app.post("/runs")
async def create_run(
    file: UploadFile | None = File(None),
    logfile: UploadFile | None = File(None),
    goldenfile: UploadFile | None = File(None),
    depth: str = Form("standard"),
    mode: str = Form("pipeline"),
) -> dict:
    run_id = uuid.uuid4().hex

    if mode == "replay":
        try:
            return _start_replay_run(run_id, logfile, goldenfile)
        except HTTPException:
            # 거부한 업로드는 남기지 않는다. 400 응답에는 run_id 가 없어서 올린 사람이
            # 찾아 지울 수 없는데, 이 화면 안내는 "끝나면 직접 지우세요"라고 말한다.
            # 남는 파일이 실행 로그(실제 질문·정답·답변 원문)라 그냥 두면 안 된다 -
            # 진단서가 스스로 민감정보 경고를 붙이는 그 내용이다.
            _discard_uploads(run_id)
            raise

    if file is None or not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 지원합니다.")

    dest = _save_upload(run_id, file)
    run_registry.create(run_id, depth=depth, upload_path=str(dest), created_at=time.time())

    thread = threading.Thread(target=_run_pipeline_background, args=(run_id, dest, depth), daemon=True)
    thread.start()

    return {"run_id": run_id}


def _discard_uploads(run_id: str) -> None:
    """run 폴더째 지운다. 정리 실패가 400 을 500 으로 바꾸면 안 되므로 조용히 넘어간다."""
    shutil.rmtree(UPLOAD_DIR / run_id, ignore_errors=True)


def _start_replay_run(
    run_id: str, logfile: UploadFile | None, goldenfile: UploadFile | None,
) -> dict:
    """리플레이 업로드를 검사하고 백그라운드 진단을 띄운다. 거부는 전부 HTTPException.

    create_run 에서 떼어낸 이유는 거부 경로가 여섯 개이고, 그 전부가 이미 디스크에
    쓴 업로드를 남기기 때문이다 - 호출부에서 한 번만 정리하려면 경계가 하나여야 한다."""
    if logfile is None or Path(logfile.filename or "").suffix.lower() not in _LOG_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="로그는 JSONL 형식만 지원합니다 (확장자 .jsonl 또는 .json).")
    # 확장자와 무관하게 .jsonl 로 저장한다 - load_external_log 는 줄 단위로 읽는다.
    log_dest = _save_upload_as(run_id, logfile, suffix=".jsonl")

    # 크기 정책은 CLI 와 같은 함수를 쓴다(agents/eval/replay.py). 예전에는 그 검사가
    # CLI 의 _main 안에만 있어서 웹으로 올리면 상한 없이 통과했다 - 진단이 끝날 때까지
    # _PIPELINE_LOCK 이 잡혀 있으므로 큰 로그 하나가 서버를 통째로 점유했다.
    # 백그라운드 스레드를 띄우기 **전에** 400 으로 돌려준다.
    from agents.eval.replay import (
        golden_size_error, log_bytes_error, log_size_error,
    )
    from agents.eval.log_intake import load_external_log

    # 줄 수를 세려면 파싱이 필요하지만, 파싱 자체가 비싼 크기가 있다. 바이트로
    # 먼저 자른 뒤 줄 수를 본다 - 상한의 몇 배짜리 파일에 파서를 태우지 않는다.
    oversize = log_bytes_error(log_dest.stat().st_size)
    if oversize:
        raise HTTPException(status_code=400, detail=oversize)

    logs, _log_errors = load_external_log(str(log_dest))
    oversize = log_size_error(len(logs))
    if oversize:
        raise HTTPException(status_code=400, detail=oversize)

    # 로그 줄에 정답이 인라인으로 들어 있으면 골든셋 파일이 없어도, 붙지 않아도
    # 진단이 성립한다. 게이트가 보는 재료를 점수층(scoring._is_evaluable)과 같은
    # 것으로 맞춘다 - gold_contexts 는 검색축까지만 재고 답변축(answer_score)은
    # ground_truth 없이 못 재므로, 신뢰도 축이 통째로 빠져 총점이 안 나온다.
    # 그런 로그를 통과시키면 게이트가 막겠다고 적어둔 그 결과("점수 없는 진단서")를
    # 전 레코드 RAGAS 를 돌리고 _PIPELINE_LOCK 을 놓은 뒤에야 알게 된다.
    has_inline_gt = any(r.ground_truth for r in logs)

    golden_dest: Path | None = None
    qa: tuple | None = None
    if goldenfile is not None and goldenfile.filename:
        golden_suffix = Path(goldenfile.filename).suffix.lower()
        if golden_suffix not in _GOLDEN_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"골든셋은 {', '.join(_GOLDEN_SUFFIXES)} 형식만 지원합니다.",
            )
        golden_dest = _save_upload_as(run_id, goldenfile, suffix=golden_suffix)
        from agents.eval.qa_merge import load_qa_set, normalize_question
        qa_map, qa_errors = load_qa_set(str(golden_dest))
        oversize = golden_size_error(len(qa_map))
        if oversize:
            raise HTTPException(status_code=400, detail=oversize)
        # 매칭 0건이면 지금 끊는다. 병합은 질문 텍스트 매칭이라 표기가 다르면 한 건도
        # 안 붙는데, 그대로 두면 레코드 전량 RAGAS 를 돌린 뒤에야 "정답 0건" 리포트가
        # 나온다 - 비싸고, 사용자는 골든셋을 줬으니 대조된 줄 안다. 같은 정규화를
        # 쓰므로 여기서 세는 값이 실제 병합 결과와 같다.
        # 단, 로그에 정답이 인라인이면 막지 않는다 - 그 로그는 골든셋이 한 건도 안
        # 붙어도 정답 대조가 되고, 파일을 안 준 경우(아래 elif)보다 재료가 많은데
        # 거부하면 재료를 더 줄수록 거부되는 게이트가 된다. 매칭률은 진단서가 밝힌다.
        log_questions = {normalize_question(r.question) for r in logs}
        matched_keys = log_questions & set(qa_map)
        if not has_inline_gt:
            if not matched_keys:
                raise HTTPException(
                    status_code=400,
                    detail=f"골든셋 {len(qa_map)}건이 로그의 질문과 한 건도 매칭되지 "
                           "않았습니다. 골든셋 질문과 로그 질문의 표기를 확인해 주세요 "
                           "(공백·문장부호·대소문자는 자동 정규화됩니다).",
                )
            # 매칭됐다고 정답이 채워지는 건 아니다 - gold_contexts 만 있는 골든셋이
            # 그 경우다. 위 has_inline_gt 게이트와 같은 재료(ground_truth)를 봐야
            # 파일로 준 정답 없는 골든셋만 통과하는 구멍이 안 생긴다. 사후에는
            # report_view._reliability_unavailable_how 가 정확히 이 사유를 말하는데,
            # 사전에 막으라고 세운 게이트가 통과시키면 그 진단서를 전 레코드 RAGAS 를
            # 돌린 뒤에야 받게 된다.
            # 순서도 그쪽과 같다 - 매칭 0건이면 정답도 0건이라, 매칭을 먼저 보지 않으면
            # "골든셋에 정답이 없다"는 엉뚱한 사유가 나간다(고치는 방법이 다르다).
            if not any(qa_map[key].get("ground_truth") for key in matched_keys):
                raise HTTPException(
                    status_code=400,
                    detail=f"골든셋의 매칭된 {len(matched_keys)}건에 정답"
                           "(ground_truth)이 없습니다. 정답이 없으면 답변이 맞았는지 "
                           "대조할 수 없어 종합점수를 낼 수 없고, 원인도 7종 중 3종만 "
                           "나옵니다 (gold_contexts 만으로는 검색축까지만 잽니다). "
                           "골든셋에 정답 열을 채우거나 로그 줄에 ground_truth 를 "
                           "넣어 주세요.",
                )
        qa = (qa_map, qa_errors)
    elif not has_inline_gt:
        # 이 화면에는 골든셋 면제가 없다. 정답지가 없으면 신뢰도 축을 못 재고 종합점수
        # 자체가 안 나오는데(report_view 가 총점을 감춘다), 원인 7종 중 3종만 담긴
        # "점수 없는 진단서"를 받아가는 건 오해만 만든다. 정답지를 아직 못 만든 경우는
        # CLI 의 --no-golden 이 개발용 통로로 남아 있다.
        # 로그에 정답이 인라인으로 들어 있으면 골든셋이 없는 게 아니다(CLI 와 같은 판정).
        raise HTTPException(
            status_code=400,
            detail="골든셋(질문·정답)이 필요합니다. 정답이 없으면 답변이 맞았는지 "
                   "대조할 수 없어 종합점수를 낼 수 없고, 원인도 7종 중 3종만 "
                   "나옵니다. 로그 줄에 ground_truth 를 넣어 주셔도 됩니다 "
                   "(gold_contexts 만으로는 검색축까지만 재므로 통과하지 않습니다).",
        )

    # 리플레이는 깊이 선택이 없다(항상 LLM 심층) — depth 폼 값이 와도 무시한다.
    run_registry.create(
        run_id, depth="full", upload_path=str(log_dest), created_at=time.time(), mode="replay",
    )
    thread = threading.Thread(
        target=_run_replay_background,
        args=(run_id, log_dest, golden_dest, (logs, _log_errors, qa)), daemon=True,
    )
    thread.start()
    return {"run_id": run_id}

    if file is None or not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 지원합니다.")

    dest = _save_upload(run_id, file)
    run_registry.create(run_id, depth=depth, upload_path=str(dest), created_at=time.time())

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
    if run.mode == "replay":
        if run.status != "done" or run.ext_report is None:
            raise HTTPException(status_code=409, detail="아직 완료되지 않았습니다.")
        return build_ext_report_view(run.ext_report, run.ext_cap)

    if run.status != "done" or run.final_state is None:
        raise HTTPException(status_code=409, detail="아직 완료되지 않았습니다.")

    eval_mode = _DEPTH_TO_EVAL_MODE.get(run.depth, "standard")
    return build_report_view(run.final_state, depth=eval_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    # access_log=False: 브라우저가 1.5초마다 폴링하는 /runs/{id}/status 요청이
    # 매번 "INFO ... 200 OK" 한 줄로 찍혀 파이프라인 로그를 덮는 것을 막는다.
    # 서버 시작/에러 등 다른 INFO 는 그대로 유지.
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=False)
