"""
agents/eval/agent.py
Eval Agent — RAG 파이프라인 품질 진단

읽기: state.chunks, state.user_questions, state.index_config, state.iteration
쓰기: state.probes, state.report, state.iteration, state.status, state.error, state.current_agent

설계 문서(Evaluate Module)의 STEP 1~5 를 순서대로 실행한다:
    STEP1  Probe 생성            → probe_gen.generate_probes
    STEP2  각 Probe로 검색·생성   → retrieval.retrieve / generate_answer
    STEP3-1 규칙 지표            → diagnose 내부 _compute_metrics (recall_at_k / token_f1)
    STEP3-2 LLM(RAGAS) 진단      → diagnose 내부 signals RAGAS 신호 (옵션, 기본 꺼짐)
    STEP4  원인 판정(Finding)     → diagnose.diagnose
    STEP5  DiagnosticReport 생성  → report.build_report

그 뒤 graph.route_after_eval() 이 report.pass_threshold 로 Serve/Optimize 를 정한다.
반복 카운터(state.iteration)는 이 에이전트가 증가시킨다(측정 시점 = 반복 경계).

계약(AGENTS.md): run() 은 반드시 state 를 반환한다. 오류는 예외를 던지지 말고
state.status="error" / state.error 에 기록하고 state 를 반환한다.
"""
from __future__ import annotations

import hashlib
import json

from core.schema import Probe
from core.state import AgentDoctorState

from agents.eval.types import (
    EvalRecord, DEFAULT_TOP_K, resolve_mode, llm_eval_enabled,
    resolve_probe_source, PROBE_SOURCE_MADE,
)
from agents.eval.probe_gen import generate_probes, uses_user_log
from agents.eval.probe_store import save_probes, load_probes
from agents.index.qdrant_store import keyword_search
from agents.rag.generator import generate_answer
from agents.rag.retriever import Retriever, get_retriever
from agents.eval.metrics_ragas import evaluate_real_track, evaluate_oracle_track, _judge as _ragas_judge
from agents.eval.diagnose import diagnose, set_context as set_diag_context
from agents.eval.report import build_report


def _retrieve_with_rag(retriever: Retriever, chunks, question: str, top_k: int) -> list[dict]:
    return retriever.search(question, top_k=top_k)


def _ragas_track(record: EvalRecord, track: str) -> dict:
    """diagnose 가 lazy 로 부르는 RAGAS 트랙 계산기(set_context 로 주입).
    비활성(EVAL_ENABLE_LLM)·키없음·실패 → {} 폴백. (DEEP 게이트는 diagnose 신호가 담당.)"""
    if not llm_eval_enabled():
        return {}
    judge = _ragas_judge()
    if judge is None:
        return {}
    try:
        if track == "oracle":
            return evaluate_oracle_track(record, judge) if record.oracle_answer is not None else {}
        return evaluate_real_track(record, judge)
    except Exception as e:
        print(f"[Eval] RAGAS({track}) 실패({e}) → 폴백")
        return {}


def run(state: AgentDoctorState) -> AgentDoctorState:
    """Eval Agent 진입점."""
    state.current_agent = "eval"
    print(f"[Eval] 청크 {len(state.chunks)}개, 반복 {state.iteration + 1}/{state.max_iterations}")

    if not state.chunks:
        state.status = "error"
        state.error = "청크가 없습니다. Index Agent 완료 여부를 확인하세요."
        print(f"[Eval] 오류: {state.error}")
        return state

    # 반복 카운터 증가 (route_after_eval 의 종료 조건)
    state.iteration += 1

    # 진단 모드(비용 tier 상한): EVAL_MODE 환경변수. STEP3-2/STEP4/리포트가 이 값으로 게이팅된다.
    mode = resolve_mode()
    print(f"[Eval] 진단 모드 = {mode} (1=fast·2=standard·3=deep·4=full)")

    # 진단 신호 캐시: 파이프라인 버전(index_config+코퍼스)이 바뀌면 무효화 → stale 재사용 방지.
    version = _pipeline_version(state)
    if state.diagnosis_cache_version != version:
        state.diagnosis_cache = {}
        state.diagnosis_cache_version = version

    try:
        # ── STEP1: Probe 생성 ──────────────────────────────────
        # user_log 소스는 매번 그대로 변환하는 저비용 경로라 캐시하지 않는다.
        # 판정은 generate_probes 와 같은 술어(uses_user_log)로 한다 — state.user_questions
        # 유무만 보면 EVAL_PROBE_SOURCE=auto 일 때 실제로는 LLM 생성으로 가는데도
        # 캐시를 건너뛰어, 문서가 그대로여도 매 실행 골든 테스트셋을 다시 만든다.
        # LLM 생성(llm_generated) 경로만 영속화 대상 — 코퍼스 버전이 그대로면 이전에
        # 만든 골든 테스트셋을 재사용해 매 Optimize 반복마다 LLM 재호출을 피한다.
        # made: 코퍼스 버전과 무관하게 이미 만들어 둔 eval_probes.json 을 그대로 재사용
        # (파일 없음/비었으면 자동 생성으로 폴백해 저장). user_questions 보다 우선한다.
        if resolve_probe_source() == PROBE_SOURCE_MADE:
            probes = load_probes(version, ignore_version=True)
            if probes:
                print(f"[Eval] STEP1: made 소스 — 저장된 Probe {len(probes)}개 재사용")
            else:
                print("[Eval] STEP1: made 소스지만 저장된 Probe 없음 → 자동 생성 후 저장")
                probes = generate_probes(state)
                save_probes(probes, version)
        elif uses_user_log(state):
            probes = generate_probes(state)
        else:
            probes = load_probes(version)
            if probes is None:
                probes = generate_probes(state)
                save_probes(probes, version)
            else:
                print(f"[Eval] STEP1: 저장된 Probe {len(probes)}개 재사용(버전 일치)")
        if not probes:
            print("[Eval] 경고: Probe 0개 생성 → 평가 불가, 통과 처리")
            state.probes = []
            state.report = build_report([], state.iteration, mode)
            state.status = "evaluated"
            return state

        # 검색 인덱스 준비: 공통 RAG retriever가 Qdrant/keyword fallback을 오케스트레이션한다.
        # Eval은 검색 구현을 직접 들고 있지 않고, RAG 모듈의 동일한 검색 규칙을 재사용한다.
        # get_retriever(=캐시판): Index가 방금 적재한 같은 청크 집합이면 그 결과를 그대로
        # 재사용한다 — 예전엔 여기서 컬렉션 준비와 upsert를 통째로 한 번 더 했다.
        retriever = get_retriever(state.chunks, state.index_config)
        chunk_text = {c.chunk_id: c.text for c in state.chunks}
        top_k = int(state.index_config.get("top_k", DEFAULT_TOP_K))

        # tier2/tier4 판별 훅(재검색·코퍼스·재생성)이 쓸 자원 주입
        set_diag_context(client=retriever, chunks=state.chunks,
                         retrieve_fn=_retrieve_with_rag, keyword_fn=keyword_search,
                         generate_fn=generate_answer, ragas_fn=_ragas_track)

        # ── STEP2~4: probe 별 평가 ────────────────────────────
        #   각 probe 의 신호 캐시(state.diagnosis_cache[probe_id])를 record 에 뷰로 주입 →
        #   진단 중 계산한 비싼 신호가 state 에 누적되어 재진단 시 재사용된다.
        print(f"\n[Eval] STEP2-4: probe별 평가 ({len(probes)}개)\n")
        records = []
        for i, p in enumerate(probes, 1):
            rec = _evaluate_probe(p, retriever, state.chunks, chunk_text, top_k, mode,
                                  state.diagnosis_cache.setdefault(p.probe_id, {}))
            _log_probe(i, len(probes), rec)
            records.append(rec)

        # ── STEP5: 리포트 ─────────────────────────────────────
        state.probes = probes
        state.report = build_report(records, state.iteration, mode)
        state.status = "evaluated"

    except Exception as e:  # 계약: 예외를 밖으로 던지지 않는다
        state.status = "error"
        state.error = f"평가 실패: {e}"
        print(f"[Eval] 오류: {e}")

    return state


# ── probe 1개 평가 (STEP2 → STEP3 → STEP4) ───────────────────────

def _clip(text: str, n: int = 50) -> str:
    """로그용 한 줄 축약(줄바꿈 제거 + n자 컷)."""
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[:n] + "…"


def _fmt_metric(v, applicable: bool = True) -> str:
    """지표 포맷: 미측정/미해당(-1·None·비적용)이면 '-'."""
    if not applicable or v is None or (isinstance(v, (int, float)) and v < 0):
        return "-"
    return f"{v:.2f}" if isinstance(v, (int, float)) else str(v)


def _short_cid(cid: str) -> str:
    """로그용 청크 id 축약: '<doc-uuid>_chunk_016' → 'chunk_016', 그 외는 원본."""
    i = cid.rfind("_chunk_")
    return cid[i + 1:] if i != -1 else cid


def _log_probe(idx: int, total: int, rec: EvalRecord) -> None:
    """probe 1개 평가 결과를 블록 형태로 출력(STEP2~4 진행 가시성용).
    질문(Q)·정답(A)·생성 답변(R)·검색/gold·지표·판정 라벨을 한 블록으로 남기고 빈 줄로 구분한다."""
    p = rec.probe
    meta = "·".join(filter(None, [p.source, p.qtype or "single"]))
    recall = _fmt_metric(rec.recall_at_k)
    f1 = _fmt_metric(rec.f1_score, bool(p.ground_truth))
    oracle = _fmt_metric(rec.oracle_f1, rec.oracle_answer is not None)
    retrieved = ", ".join(_short_cid(c) for c in rec.retrieved_chunk_ids)
    gold = ", ".join(_short_cid(c) for c in p.gold_chunk_ids)

    print(f"[{idx}/{total}] {p.probe_id}  ({meta})")
    print(f"Q: {_clip(p.question, 80)}")
    print(f"A: {_clip(p.ground_truth, 80) if p.ground_truth else '-'}")
    print(f"R: {_clip(rec.generated_answer, 80) if rec.generated_answer else '-'}")
    print(f"검색: [{retrieved}]")
    print(f"골드: [{gold}]")
    print(f"메트릭 결과: recall@k={recall}  f1={f1}  oracle_f1={oracle}")
    print(f"-Findings({len(rec.findings)})-")
    if rec.findings:
        for i, f in enumerate(rec.findings, 1):
            mark = "" if f.confirmed else "(예비)"
            print(f"[{i}] {f.label}{mark}: {f.metadata.get('reason') or '-'}")
    else:
        print("없음(정상)")
    print()  # 블록 구분 빈 줄


def _pipeline_version(state: AgentDoctorState) -> str:
    """진단 신호 캐시 무효화 키. index_config(Optimize가 바꿈)+코퍼스가 바뀌면 값이 달라진다.
    (재실행/코퍼스 의존 신호는 이 버전 내에서만 재사용 안전.)

    청크 id 와 함께 본문 hash 도 넣는다 — doc_id 는 출처로 고정돼 있어서(Ingest 의
    _stable_doc_id) 같은 파일을 고쳐도 id 는 그대로다. hash 를 빼면 본문이 바뀌었는데도
    버전이 같아져, 옛 gold_chunk_ids 를 가리키는 stale probe 를 재사용하게 된다."""
    key = json.dumps(state.index_config, sort_keys=True, default=str)
    key += "|chunks=" + ",".join(sorted(f"{c.chunk_id}:{c.hash}" for c in state.chunks))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _evaluate_probe(
    probe: Probe,
    retriever: Retriever,
    chunks: list,
    chunk_text: dict[str, str],
    top_k: int,
    mode: int,
    sig_cache: dict,
) -> EvalRecord:
    """한 Probe 에 대해 검색·생성·지표·판정을 수행하고 EvalRecord 반환.
    sig_cache 는 state.diagnosis_cache[probe_id] 뷰 — 진단 신호 memoize 가 여기(=state)에 누적된다."""
    rec = EvalRecord(probe=probe, signals=sig_cache)

    # STEP2: 공통 RAG retriever로 검색
    hits = retriever.search(probe.question, top_k=top_k)
    rec.retrieved = hits
    rec.retrieved_context = [h.get("text", "") for h in hits]
    rec.retrieved_chunk_ids = [h.get("chunk_id", "") for h in hits]

    # STEP2: 답변 생성 (실제 트랙)
    rec.generated_answer = generate_answer(probe.question, rec.retrieved_context)

    # STEP2: Oracle 답변 (gold context 가 있을 때만)
    gold_ctx = [chunk_text[cid] for cid in probe.gold_chunk_ids if cid in chunk_text]
    if gold_ctx:
        rec.oracle_context = gold_ctx
        rec.oracle_answer = generate_answer(probe.question, gold_ctx)

    # STEP3-1(지표)·STEP3-2(RAGAS)·STEP4(진단)는 전부 diagnose 안에서 계산·판정한다.
    # 구조를 계산 -> diagnose에서 diagnose -> 모든 계산으로 변경함.
    #   agent 는 STEP2(파이프라인 실행)까지만 — record 는 raw I/O(검색·생성 결과)만 담는다.
    rec.findings = diagnose(rec, mode)
    return rec
