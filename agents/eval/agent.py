"""
agents/eval/agent.py
Eval Agent — RAG 파이프라인 품질 진단

읽기: state.chunks, state.user_questions, state.index_config, state.iteration
쓰기: state.probes, state.report, state.iteration, state.status, state.error, state.current_agent

설계 문서(Evaluate Module)의 STEP 1~5 를 순서대로 실행한다:
    STEP1  Probe 생성            → probe_gen.generate_probes
    STEP2  각 Probe로 검색·생성   → retrieval.retrieve / generate_answer
    STEP3-1 규칙 지표·브랜치      → metrics.recall_at_k / token_f1 / decide_branch
    STEP3-2 LLM(RAGAS) 진단      → ragas_eval.evaluate   (옵션, 기본 꺼짐)
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

from agents.eval.types import EvalRecord, DEFAULT_TOP_K, resolve_mode, llm_eval_enabled
from agents.eval.probe_gen import generate_probes
# ⚠️ 임시: Index Agent가 검색 리트리버를 제공하기 전까지만 retrieval_temp 사용.
#     Index 검색이 준비되면 retrieval_temp 를 삭제하고 여기 import 를 교체할 것.
from agents.eval.retrieval_temp import build_eval_index, retrieve, generate_answer, _keyword_search
from agents.eval.ragas_eval import evaluate_real_track, evaluate_oracle_track, _judge as _ragas_judge
from agents.eval.diagnose import diagnose, set_context as set_diag_context


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
from agents.eval.report import build_report


def run(state: AgentDoctorState) -> AgentDoctorState:
    """Eval Agent 진입점."""
    state.current_agent = "eval"
    print(f"[Eval] 시작 - 청크 {len(state.chunks)}개, 반복 {state.iteration + 1}/{state.max_iterations}")

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
        probes = generate_probes(state)
        if not probes:
            print("[Eval] 경고: Probe 0개 생성 → 평가 불가, 통과 처리")
            state.probes = []
            state.report = build_report([], state.iteration, mode)
            state.status = "evaluated"
            return state

        # 검색 인덱스 준비(임시): Index 리트리버 미개발 → retrieval_temp 로 state.chunks 재적재
        client = build_eval_index(state.chunks)
        # client = build_eval_index(state.chunks)
        chunk_text = {c.chunk_id: c.text for c in state.chunks}
        top_k = int(state.index_config.get("top_k", DEFAULT_TOP_K))

        # tier2/tier4 판별 훅(재검색·코퍼스·재생성)이 쓸 자원 주입
        set_diag_context(client=client, chunks=state.chunks,
                         retrieve_fn=retrieve, keyword_fn=_keyword_search,
                         generate_fn=generate_answer, ragas_fn=_ragas_track)

        # ── STEP2~4: probe 별 평가 ────────────────────────────
        #   각 probe 의 신호 캐시(state.diagnosis_cache[probe_id])를 record 에 뷰로 주입 →
        #   진단 중 계산한 비싼 신호가 state 에 누적되어 재진단 시 재사용된다.
        records = [
            _evaluate_probe(p, client, state.chunks, chunk_text, top_k, mode,
                            state.diagnosis_cache.setdefault(p.probe_id, {}))
            for p in probes
        ]

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

def _pipeline_version(state: AgentDoctorState) -> str:
    """진단 신호 캐시 무효화 키. index_config(Optimize가 바꿈)+코퍼스가 바뀌면 값이 달라진다.
    (재실행/코퍼스 의존 신호는 이 버전 내에서만 재사용 안전.)"""
    key = json.dumps(state.index_config, sort_keys=True, default=str)
    key += "|chunks=" + ",".join(sorted(c.chunk_id for c in state.chunks))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _evaluate_probe(
    probe: Probe,
    client,
    chunks: list,
    chunk_text: dict[str, str],
    top_k: int,
    mode: int,
    sig_cache: dict,
) -> EvalRecord:
    """한 Probe 에 대해 검색·생성·지표·판정을 수행하고 EvalRecord 반환.
    sig_cache 는 state.diagnosis_cache[probe_id] 뷰 — 진단 신호 memoize 가 여기(=state)에 누적된다."""
    rec = EvalRecord(probe=probe, signals=sig_cache)

    # STEP2: 검색(임시): Index 리트리버 미개발 → retrieval_temp.retrieve 사용
    hits = retrieve(client, chunks, probe.question, top_k)
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
    #   agent 는 STEP2(파이프라인 실행)까지만 — record 는 raw I/O(검색·생성 결과)만 담는다.
    rec.findings = diagnose(rec, mode)
    return rec
