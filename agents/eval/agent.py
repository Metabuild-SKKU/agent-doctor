"""
agents/eval/agent.py
Eval Agent — RAG 파이프라인 품질 진단

읽기: state.documents, state.chunks, state.user_questions, state.index_config, state.iteration
쓰기: state.probes, state.report, state.status, state.error, state.current_agent

설계 문서(Evaluate Module)의 STEP 1~5 를 순서대로 실행한다:
    STEP1  Probe 생성            → probe_gen.generate_probes
    STEP2  각 Probe로 검색·생성   → retrieval.retrieve / generate_answer
    STEP3-1 규칙 지표            → diagnose 내부 _compute_metrics (recall_at_k / char_f1)
    STEP3-2 LLM(RAGAS) 진단      → diagnose 내부 _compute_ragas_real (DEEP 이상 전 probe)
                                   + _compute_ragas_oracle (실패로 판정된 probe 만)
    STEP4  원인 판정(Finding)     → diagnose.diagnose
    STEP5  DiagnosticReport 생성  → report.build_report

그 뒤 graph.route_after_eval() 이 report.pass_threshold 로 Serve/Optimize 를 정한다.
반복 카운터(state.iteration)는 Optimize가 새 라벨 study를 시작할 때만 증가시킨다.
Eval은 같은 라벨의 후보별 측정에서 카운터를 바꾸지 않는다.

계약(AGENTS.md): run() 은 반드시 state 를 반환한다. 오류는 예외를 던지지 말고
state.status="error" / state.error 에 기록하고 state 를 반환한다.
"""
from __future__ import annotations

import hashlib
import json

from core.schema import Probe
from core.llm_usage import print_summary
from core.parallel import parallel_map
from core.state import AgentDoctorState

from agents.eval.types import (
    EvalRecord, DEFAULT_TOP_K, Mode, resolve_mode, llm_eval_enabled,
    resolve_llm_concurrency, resolve_probe_source,
    PROBE_SOURCE_MADE, PROBE_SOURCE_TAXONOMY,
)
from agents.eval.probe_gen import generate_probes, uses_user_log, _resync_gold_chunk_ids
from agents.eval.probe_store import save_probes, load_probes, corpus_version
from agents.index.qdrant_store import keyword_search
from agents.rag.generator import generate_answer
from agents.rag.retriever import Retriever, get_retriever
from agents.eval.metrics_ragas import (
    evaluate_real_track, evaluate_oracle_track, _judge as _ragas_judge,
)
from agents.eval.metrics_common import set_context as set_diag_context
from agents.eval.metrics_basic import _compute_metrics
from agents.eval.diagnose import diagnose, _is_success
from agents.eval.report import build_report


def _retrieve_with_rag(retriever: Retriever, chunks, question: str, top_k: int) -> list[dict]:
    return retriever.search(question, top_k=top_k)


def _ragas_track(record: EvalRecord, track: str) -> dict:
    """diagnose 가 트랙별 1회 부르는 RAGAS 계산기(set_context 로 주입).
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

    # 진단 모드(비용 tier 상한): EVAL_MODE 환경변수. STEP3-2/STEP4/리포트가 이 값으로 게이팅된다.
    mode = resolve_mode()
    print(f"[Eval] 진단 모드 = {mode} (1=fast·2=standard·3=deep·4=full)")

    # 진단 신호 캐시: 파이프라인 버전(index_config+코퍼스)이 바뀌면 무효화 → stale 재사용 방지.
    # 진단 신호(예: gold_in_wider_candidates)는 top_k 로 검색한 결과에 의존하므로,
    # 이 캐시는 index_config 를 포함하는 _pipeline_version 을 그대로 쓴다.
    version = _pipeline_version(state)
    if state.diagnosis_cache_version != version:
        state.diagnosis_cache = {}
        state.diagnosis_cache_version = version

    # Probe 캐시는 원문 문서에 의존한다. top_k뿐 아니라 chunk_size가 바뀌어도 같은
    # 질문/gold_spans를 유지하고, 불러온 뒤 현재 청크 기준 gold_chunk_ids만 재동기화한다.
    probe_version = corpus_version(state.chunks, state.documents)

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
        probe_source = resolve_probe_source()
        if probe_source == PROBE_SOURCE_MADE:
            probes = load_probes(probe_version, ignore_version=True)
            if probes:
                probes = _resync_gold_chunk_ids(
                    probes,
                    state.chunks,
                    state.documents,
                )
                print(f"[Eval] STEP1: made 소스 — 저장된 Probe {len(probes)}개 재사용")
            else:
                print("[Eval] STEP1: made 소스지만 저장된 Probe 없음 → 자동 생성 후 저장")
                probes = generate_probes(state)
                save_probes(probes, probe_version)
        elif probe_source == PROBE_SOURCE_TAXONOMY:
            # taxonomy 는 probe_version 키(corpus_version=청크+문서)에 없는 입력(QA 파일·
            # KORQUAD_MAX_DOCS/QA_LIMIT)에 좌우되므로 캐시를 타면 auto/다른 QA 의 Probe 를 재사용해 오염된다.
            # 파일 로드+resync 는 LLM 없이 저비용이라 매번 새로 만든다(캐시 우회).
            probes = generate_probes(state)
        elif uses_user_log(state):
            probes = generate_probes(state)
        else:
            probes = load_probes(probe_version)
            if probes is None:
                probes = generate_probes(state)
                save_probes(probes, probe_version)
            else:
                probes = _resync_gold_chunk_ids(
                    probes,
                    state.chunks,
                    state.documents,
                )
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

        # tier2/tier3 판별 훅(재검색·코퍼스·RAGAS)이 쓸 자원 주입
        set_diag_context(client=retriever, chunks=state.chunks,
                         retrieve_fn=_retrieve_with_rag, keyword_fn=keyword_search,
                         ragas_fn=_ragas_track)

        # ── STEP2~4: probe 별 평가 ────────────────────────────
        #   각 probe 의 신호 캐시(state.diagnosis_cache[probe_id])를 record 에 뷰로 주입 →
        #   진단 중 계산한 비싼 신호가 state 에 누적되어 재진단 시 재사용된다.
        #   LLM 호출(답변 생성)만 병렬화하고 검색·진단은 순차 유지 — Qdrant/임베딩/
        #   signals 전역이 병렬 구간에 들어가지 않게 하는 설계(계획 B안).
        print(f"\n[Eval] STEP2-4: probe별 평가 ({len(probes)}개)\n")

        # Phase A(순차): 검색 + record 준비, 답변 생성 태스크 수집
        records = []
        gen_tasks: list[tuple[EvalRecord, str, list[str]]] = []
        for p in probes:
            rec = _prepare_record(p, retriever, chunk_text, top_k,
                                  state.diagnosis_cache.setdefault(p.probe_id, {}))
            records.append(rec)
            gen_tasks.append((rec, "real", rec.retrieved_context))
            if rec.oracle_context:  # gold context 가 있을 때만 Oracle 트랙 (기존 동일)
                gen_tasks.append((rec, "oracle", rec.oracle_context))

        # Phase B(병렬): LLM 답변 생성만 동시 실행 (EVAL_LLM_CONCURRENCY, 1이면 순차)
        concurrency = resolve_llm_concurrency()
        if concurrency > 1 and len(gen_tasks) > 1:
            print(f"[Eval] STEP2: 답변 생성 {len(gen_tasks)}건 병렬 실행 (동시성 {concurrency})")
        answers = parallel_map(lambda t: generate_answer(t[0].probe.question, t[2]),
                               gen_tasks, concurrency)
        for (rec, track, _ctx), answer in zip(gen_tasks, answers):
            if track == "real":
                rec.generated_answer = answer
            else:
                rec.oracle_answer = answer

        # Phase B2(병렬): RAGAS 선계산 — 트랙별로 필요한 probe 만 동시 실행하고 *_done
        # 플래그를 세워, Phase C 의 _compute_ragas_real/_oracle 이 캐시 히트만 하게 한다.
        # _ragas_track 은 진단 전역과 무관한 모듈 함수라 병렬 구간에 안전.
        # 게이트는 _compute_ragas_* 와 동일(mode >= DEEP); LLM 비활성·키없음은
        # _ragas_track 이 {} 폴백이라 기존과 같은 동작으로 수렴한다.
        # 동시성 1이면 태스크 순서가 Phase C 호출 순서(probe 순)와 일치.
        if mode >= Mode.DEEP:
            # B2-1: 실제 트랙은 전 probe 에 필요하다 — 성공/실패 판정(_f1_ok 의 강등)과
            # 리포트 RAGAS 평균이 모두 실제 트랙을 쓴다.
            if concurrency > 1 and len(records) > 1:
                print(f"[Eval] STEP3-2: RAGAS 실제 트랙 {len(records)}건 병렬 실행 (동시성 {concurrency})")
            real_scores = parallel_map(lambda r: _ragas_track(r, "real") or {}, records, concurrency)
            for rec, score in zip(records, real_scores):
                rec.ragas, rec.ragas_done = score, True

            # B2-2: 오라클 트랙은 '실패로 판정된 probe' 에만 필요하다 — 소비처가 B그룹 라벨과
            # _oracle_ok 뿐이고, 리포트 평균은 실제 트랙만 쓴다. 성공 probe 는 diagnose 가
            # 성공 게이트에서 바로 끝나므로 오라클 LLM 비용을 지불할 이유가 없다.
            # 판정에 쓸 규칙 지표를 먼저 채운다(_compute_metrics 는 순수·멱등이라 Phase C 에서
            # diagnose 가 다시 불러도 같은 값이 나온다).
            for rec in records:
                _compute_metrics(rec)
            failed = [rec for rec in records if _is_success(rec) is False]
            if failed:
                if concurrency > 1 and len(failed) > 1:
                    print(f"[Eval] STEP3-2: RAGAS 오라클 트랙 {len(failed)}건 병렬 실행 "
                          f"(실패 probe 만, 전체 {len(records)}건 중)")
                oracle_scores = parallel_map(lambda r: _ragas_track(r, "oracle") or {}, failed, concurrency)
                for rec, score in zip(failed, oracle_scores):
                    rec.oracle_ragas, rec.oracle_ragas_done = score, True

        # Phase C(순차): 지표·진단·로그 — diagnose 는 signals 전역·진단 캐시·tier2
        # 재검색을 쓰므로 병렬 구간 밖에서 실행한다.
        for i, rec in enumerate(records, 1):
            rec.findings = diagnose(rec, mode)
            _log_probe(i, len(records), rec)

        # ── STEP5: 리포트 ─────────────────────────────────────
        state.probes = probes
        state.report = build_report(records, state.iteration, mode)
        state.status = "evaluated"

    except Exception as e:  # 계약: 예외를 밖으로 던지지 않는다
        state.status = "error"
        state.error = f"평가 실패: {e}"
        print(f"[Eval] 오류: {e}")

    print_summary(tag="Eval")
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


def _prepare_record(
    probe: Probe,
    retriever: Retriever,
    chunk_text: dict[str, str],
    top_k: int,
    sig_cache: dict,
) -> EvalRecord:
    """[STEP2 Phase A·순차] 검색을 수행하고 답변 생성 직전까지의 EvalRecord 를 준비한다.
    답변 생성(LLM)은 run() 의 Phase B 에서 병렬로, 지표·진단(STEP3~4, diagnose)은
    Phase C 에서 순차로 이어진다 — record 는 raw I/O(검색·생성 결과)만 담는다.
    sig_cache 는 state.diagnosis_cache[probe_id] 뷰 — 진단 신호 memoize 가 여기(=state)에 누적된다."""
    rec = EvalRecord(probe=probe, signals=sig_cache)

    # STEP2: 공통 RAG retriever로 검색
    hits = retriever.search(probe.question, top_k=top_k)
    rec.retrieved = hits
    rec.retrieved_context = [h.get("text", "") for h in hits]
    rec.retrieved_chunk_ids = [h.get("chunk_id", "") for h in hits]

    # Oracle 트랙 컨텍스트 (gold context 가 있을 때만 — 답변은 Phase B 에서 생성)
    gold_ctx = [chunk_text[cid] for cid in probe.gold_chunk_ids if cid in chunk_text]
    if gold_ctx:
        rec.oracle_context = gold_ctx
    return rec
