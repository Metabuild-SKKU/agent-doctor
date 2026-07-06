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

from core.schema import Probe
from core.state import AgentDoctorState

from agents.eval.types import Branch, EvalRecord, DEFAULT_TOP_K
from agents.eval.probe_gen import generate_probes
# ⚠️ 임시: Index Agent가 검색 리트리버를 제공하기 전까지만 retrieval_temp 사용.
#     Index 검색이 준비되면 retrieval_temp 를 삭제하고 여기 import 를 교체할 것.
from agents.eval.retrieval_temp import build_eval_index, retrieve, generate_answer
from agents.eval.metrics import recall_at_k, token_f1, is_abstention, decide_branch
from agents.eval.ragas_eval import evaluate as run_llm_metrics
from agents.eval.diagnose import diagnose
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

    try:
        # ── STEP1: Probe 생성 ──────────────────────────────────
        probes = generate_probes(state)
        if not probes:
            print("[Eval] 경고: Probe 0개 생성 → 평가 불가, 통과 처리")
            state.probes = []
            state.report = build_report([], state.iteration)
            state.status = "evaluated"
            return state

        # 검색 인덱스 준비(임시): Index 리트리버 미개발 → retrieval_temp 로 state.chunks 재적재
        client = build_eval_index(state.chunks)
        # client = build_eval_index(state.chunks)
        chunk_text = {c.chunk_id: c.text for c in state.chunks}
        top_k = int(state.index_config.get("top_k", DEFAULT_TOP_K))

        # ── STEP2~4: probe 별 평가 ────────────────────────────
        records = [
            _evaluate_probe(p, client, state.chunks, chunk_text, top_k)
            for p in probes
        ]

        # ── STEP5: 리포트 ─────────────────────────────────────
        state.probes = probes
        state.report = build_report(records, state.iteration)
        state.status = "evaluated"

    except Exception as e:  # 계약: 예외를 밖으로 던지지 않는다
        state.status = "error"
        state.error = f"평가 실패: {e}"
        print(f"[Eval] 오류: {e}")

    return state


# ── probe 1개 평가 (STEP2 → STEP3 → STEP4) ───────────────────────

def _evaluate_probe(
    probe: Probe,
    client,
    chunks: list,
    chunk_text: dict[str, str],
    top_k: int,
) -> EvalRecord:
    """한 Probe 에 대해 검색·생성·지표·판정을 수행하고 EvalRecord 반환."""
    rec = EvalRecord(probe=probe)

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

    # STEP3-1: 규칙 지표 + 브랜치
    rec.recall_at_k = recall_at_k(probe.gold_chunk_ids, rec.retrieved_chunk_ids)

    if probe.ground_truth:  # 정답이 있어야 규칙 판정 의미 있음
        ref = probe.ground_truth
        rec.f1_score = token_f1(rec.generated_answer, ref)
        rec.oracle_f1 = token_f1(rec.oracle_answer or "", ref) if rec.oracle_answer else 0.0
        answer_exists = True if probe.answer_exists is None else probe.answer_exists
        rec.branch = decide_branch(
            rec.recall_at_k, rec.f1_score, rec.oracle_f1,
            answer_exists=answer_exists,
            abstained=is_abstention(rec.generated_answer),
        )
    else:
        # 정답 미보유(user_log 등) → 규칙 판정 불가. RAGAS(무정답 지표)에만 의존.
        rec.branch = Branch.SUCCESS

    # STEP3-2: LLM(RAGAS) 진단 — 활성화 시에만 record 를 채움
    run_llm_metrics(rec)

    # STEP4: 원인 판정
    rec.findings = diagnose(rec)
    return rec
