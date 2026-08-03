"""
agents/eval/agent.py
Eval Agent — RAG 파이프라인 품질 진단

읽기: state.documents, state.chunks, state.user_questions, state.index_config, state.iteration,
      state.optimization_history, state.active_index_key, state.eval_cache
쓰기: state.probes, state.report, state.diagnosis_cache, state.diagnosis_cache_version,
      state.eval_cache, state.active_eval_key, state.eval_cache_hit,
      state.status, state.error, state.current_agent

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
import os
import sys
from copy import deepcopy
from pathlib import Path

from core.schema import Probe, EvalSnapshot
from core.llm_usage import print_summary, snapshot_usage, step
from core.parallel import parallel_map
from core.state import AgentDoctorState

from agents.eval.types import (
    EvalRecord, DEFAULT_TOP_K, Mode, resolve_mode, llm_eval_enabled,
    resolve_llm_concurrency, resolve_probe_source,
    PROBE_SOURCE_MADE, PROBE_SOURCE_TAXONOMY,
)
from agents.eval.probe_gen import (
    generate_probes, uses_user_log, _resync_gold_chunk_ids, regenerate_probes,
)
from agents.eval.probe_store import (
    DEFAULT_STORE_PATH,
    save_probes,
    load_probes,
    corpus_version,
)
from agents.index.qdrant_store import keyword_search
from agents.rag.generator import generate_answer
from agents.rag.retriever import Retriever, get_retriever
from agents.eval.metrics_ragas import (
    evaluate_real_track, evaluate_oracle_track, evaluate_abstention,
    evaluate_reasoning_mode, _judge as _ragas_judge,
)
from agents.eval.metrics_common import (
    DEFAULT_MAX_RERANK_CANDIDATES,
    DEFAULT_RERANK_CANDIDATES,
    missed_gold_ids,
    set_context as set_diag_context,
    set_mode,
)
from agents.eval import topic_cluster
from agents.eval.metrics_basic import _compute_metrics
from agents.eval.diagnose import diagnose, _is_success
from agents.eval.report import build_report, is_bad_gold_probe


_EVAL_CACHE_ENV_KEYS = (
    "EVAL_PROBE_SOURCE",
    "EVAL_ENABLE_LLM",
    "EVAL_LLM_PROVIDER",
    "EVAL_GEN_MODEL",
    "EVAL_GEN_MODEL_GEMINI",
    "EVAL_GEN_MODEL_GITHUB",
    "EVAL_JUDGE_MODEL",
    "EVAL_JUDGE_MODEL_GEMINI",
    "EVAL_JUDGE_MODEL_GITHUB",
    "EVAL_RELEVANCY_STRICTNESS",
    "EVAL_EMBED_MODEL",
    "EVAL_EMBED_MODEL_GEMINI",
    "EVAL_TESTSET_SIZE",
    "EVAL_TAXONOMY_QA",
    "KORQUAD_MAX_DOCS",
    "KORQUAD_QA_LIMIT",
    "RAG_LLM_PROVIDER",
    "RAG_LLM_MODEL",
    "RAG_OPENAI_MODEL",
    "RAG_GEMINI_MODEL",
    "RAG_GITHUB_MODEL",
)
_EVAL_CACHE_SECRET_KEYS = (
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GITHUB_MODELS_TOKEN",
    "GITHUB_TOKEN",
)


def _eval_cache_limit(state: AgentDoctorState) -> int:
    if not state.index_config.get("rollback_cache_enabled", True):
        return 0
    try:
        requested = int(
            state.index_config.get("rollback_cache_max_versions", 2)
        )
    except (TypeError, ValueError):
        requested = 2
    return max(1, min(2, requested))


def _eval_cache_key(
    state: AgentDoctorState,
    mode: Mode,
    pipeline_version: str,
    probe_version: str,
) -> str:
    """검색·생성·진단 결과를 바꾸는 입력을 묶어 완전 진단 캐시 키를 만든다."""
    meaningful_config = {
        key: value
        for key, value in state.index_config.items()
        if key not in {
            "rollback_cache_enabled",
            "rollback_cache_max_versions",
            "recreate_collection_on_dimension_mismatch",
        }
    }
    probe_source = resolve_probe_source()
    payload = {
        "schema_version": 1,
        "pipeline_version": pipeline_version,
        "probe_version": probe_version,
        "active_index_key": state.active_index_key,
        "index_config": meaningful_config,
        "user_questions": state.user_questions,
        "mode": int(mode),
        "probe_source": probe_source,
        "probe_file": _probe_file_identity(state, probe_source),
        "environment": {
            key: os.getenv(key)
            for key in _EVAL_CACHE_ENV_KEYS
        },
        "provider_available": {
            key: bool(os.getenv(key))
            for key in _EVAL_CACHE_SECRET_KEYS
        },
    }
    raw = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _probe_file_identity(
    state: AgentDoctorState,
    probe_source: str,
) -> dict:
    """실제 STEP1 경로가 읽는 외부 Probe 파일만 캐시 키에 넣는다."""
    if probe_source == PROBE_SOURCE_TAXONOMY:
        return _file_identity(
            os.getenv("EVAL_TAXONOMY_QA", "data/qa_pairs.jsonl")
        )
    if probe_source == PROBE_SOURCE_MADE or not uses_user_log(state):
        return _file_identity(DEFAULT_STORE_PATH)
    return {}


def _file_identity(path: str) -> dict:
    """외부 Probe 파일이 바뀌면 진단 캐시도 무효화한다."""
    target = Path(path)
    try:
        stat = target.stat()
        digest = hashlib.sha256()
        with target.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return {
            "path": str(target.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest.hexdigest(),
        }
    except OSError:
        return {"path": str(target), "missing": True}


def _find_eval_snapshot(
    state: AgentDoctorState,
    cache_key: str,
) -> EvalSnapshot | None:
    """캐시 hit를 LRU 최신 위치로 옮긴다."""
    if _eval_cache_limit(state) == 0:
        state.eval_cache = []
        return None
    for index, snapshot in enumerate(state.eval_cache):
        if snapshot.cache_key != cache_key:
            continue
        state.eval_cache.append(state.eval_cache.pop(index))
        return state.eval_cache[-1]
    return None


def _store_eval_snapshot(
    state: AgentDoctorState,
    cache_key: str,
) -> None:
    """성공한 Eval 결과만 저장하고 현재/직전 두 버전만 남긴다."""
    limit = _eval_cache_limit(state)
    state.active_eval_key = cache_key
    if _reranker_evaluation_incomplete(state.report):
        # 같은 config로 즉시 재시도할 때 실패 리포트가 cache hit로 복원되면
        # CrossEncoder는 영원히 다시 실행되지 않는다. 진단 신호도 실패 검색 결과에
        # 의존할 수 있으므로 함께 비워 다음 Eval이 처음부터 다시 측정하게 한다.
        state.eval_cache = [
            snapshot
            for snapshot in state.eval_cache
            if snapshot.cache_key != cache_key
        ]
        state.diagnosis_cache = {}
        state.diagnosis_cache_version = ""
        print("[Eval] reranker 실행 불완전 → 진단 캐시를 저장하지 않고 재시도 허용")
        return
    if limit == 0 or state.report is None:
        if limit == 0:
            state.eval_cache = []
        return
    state.eval_cache = [
        snapshot
        for snapshot in state.eval_cache
        if snapshot.cache_key != cache_key
    ]
    state.eval_cache.append(
        EvalSnapshot(
            cache_key=cache_key,
            index_key=state.active_index_key,
            probes=deepcopy(state.probes),
            report=deepcopy(state.report),
            diagnosis_cache=deepcopy(state.diagnosis_cache),
            diagnosis_cache_version=state.diagnosis_cache_version,
        )
    )
    pinned_key = _pending_baseline_eval_key(state)
    pinned = next(
        (
            snapshot
            for snapshot in state.eval_cache
            if snapshot.cache_key == pinned_key
        ),
        None,
    )
    current = state.eval_cache[-1]
    if (
        limit == 2
        and pinned is not None
        and pinned.cache_key != current.cache_key
    ):
        # top_k sweep처럼 후보가 여러 개여도 study baseline과 현재 후보만 남긴다.
        state.eval_cache = [pinned, current]
    else:
        state.eval_cache = state.eval_cache[-limit:]


def _reranker_evaluation_incomplete(report) -> bool:
    """reranker가 켜졌지만 대상 검색 중 하나라도 실제 재정렬되지 않았는지 확인한다."""
    if report is None:
        return False
    runtime = (report.runtime_summary or {}).get("reranker")
    if not isinstance(runtime, dict) or not runtime.get("enabled"):
        return False
    try:
        attempted = int(runtime.get("attempted", 0))
        applied = int(runtime.get("applied", 0))
    except (TypeError, ValueError):
        return True
    return attempted == 0 or applied < attempted


def _pending_baseline_eval_key(state: AgentDoctorState) -> str:
    """판정 대기 study가 되돌아갈 baseline Eval 키를 찾는다."""
    for item in reversed(state.optimization_history):
        metadata = getattr(item, "metadata", {}) or {}
        if not metadata.get("pending"):
            continue
        key = metadata.get("before_eval_key")
        if key:
            return str(key)
    return ""


def _retrieve_with_rag(retriever: Retriever, chunks, question: str, top_k: int) -> list[dict]:
    """순위 측정용 wide 재검색 — **리랭크는 끈다**.

    리랭크를 태우면 wide_n(=100) 개를 재정렬한 순서가 나오는데 프로덕션 검색은
    rerank_candidates(=20) 개만 재정렬한다. 두 순서가 달라 '후보창 밖'과 '리랭커가 강등'을
    가르는 기준값이 오염되므로, 여기서는 융합 단계까지의 순위만 잰다
    (리랭크 이후 순위는 record.retrieval_details 로 프로덕션 검색에서 그대로 온다).
    부수 효과로 probe 마다 cross-encoder 100쌍을 태우던 비용도 사라진다.
    """
    return retriever.search(question, top_k=top_k, apply_rerank=False)


def _dense_retrieve_with_rag(
    retriever: Retriever, chunks, question: str, top_k: int
) -> list[dict]:
    """dense 단일 채널 wide 재검색 — 융합 손실(한 채널은 상위인데 융합이 밀어냄) 판정용.

    하이브리드가 꺼져 있으면 융합 자체가 없어 대조할 게 없으므로 빈 결과를 준다.
    """
    view = retriever.dense_only_view()
    if view is None:
        return []
    return view.search(question, top_k=top_k, apply_rerank=False)


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
        if track == "abstention":
            return evaluate_abstention(record, judge)
        if track == "reasoning_mode":
            return evaluate_reasoning_mode(record, judge) if record.oracle_answer is not None else {}
        return evaluate_real_track(record, judge)
    except Exception as e:
        print(f"[Eval] RAGAS({track}) 실패({e}) → 폴백")
        return {}


_MODE_NAMES = {Mode.FAST: "fast", Mode.STANDARD: "standard", Mode.DEEP: "deep"}


def _maybe_regenerate_bad_gold(
    state: AgentDoctorState,
    records: list[EvalRecord],
    probes: list[Probe],
    probe_version: str,
) -> bool:
    """bad_gold_answer 로 확정된 우리 probe 를 재생성해 probe 파일에 반영한다(Option A).

    반환: 실제로 재생성·저장이 일어났으면 True. 그 경우 호출부(STEP5)는 이번 실행의 eval
    snapshot 을 저장하지 않아야 한다 — 재생성 probe 는 같은 probe_version(=cache key)을
    유지하므로, 옛 probe/report 로 만든 snapshot 을 저장하면 다음 Eval 이 새 probe 파일을
    읽기 전에 그 snapshot 을 cache hit 해 옛 성적표를 복원한다(리뷰 blocker: 루프 미완결).
    저장 실패 시엔 재생성이 반영되지 않은 것이므로 무효화·성공 로그·skip 을 하지 않는다."""
    bad_probes = [r.probe for r in records if is_bad_gold_probe(r)]
    if not bad_probes:
        return False
    replaced = regenerate_probes(bad_probes, state)
    if not replaced:
        # 재생성 불가(전부 user_log·멀티홉·LLM 실패 등) → 리포트의 검수 요청으로만 남는다.
        return False
    updated = [replaced.get(p.probe_id, p) for p in probes]
    # save_probes 는 OSError 를 내부에서 삼키고 성공 여부만 bool 로 돌려준다(예외가 밖으로
    # 안 나옴). 저장이 실패하면 파일엔 옛 probe 가 남으므로, 캐시 무효화·성공 로그·snapshot
    # skip 을 하면 안 된다(안 그러면 다음 실행이 옛 probe 를 다시 읽어 재생성을 무한 반복,
    # 1회 가드도 파일에 반영 안 됨). 이번 실행은 기존 probe/캐시를 그대로 유지한다.
    if not save_probes(updated, probe_version):
        print("[Eval] bad_gold probe 재생성 저장 실패 — 기존 probe 유지(다음 실행 재시도)")
        return False
    _invalidate_regenerated_caches(state, set(replaced))
    print(f"[Eval] bad_gold probe {len(replaced)}개 재생성 → 다음 실행에서 재평가 "
          f"(재생성 불가 {len(bad_probes) - len(replaced)}개는 사용자 검수 요청)")
    return True


def _invalidate_regenerated_caches(state: AgentDoctorState, regenerated_ids: set[str]) -> None:
    """재생성 probe 의 stale 캐시를 무효화한다.

    재생성 probe 는 같은 probe_id·probe_version 을 유지하므로(corpus_version 기반), 그 id 를
    담은 캐시가 남으면 다음 평가가 옛 진단 신호/리포트를 그대로 재사용해 재생성이 조용히
    무효화될 수 있다(리뷰 blocker#3). 해당 probe 의 진단 신호와, 그 probe 를 담은 eval
    스냅샷을 제거해 다음 평가가 새 probe 로 처음부터 진단·채점하게 강제한다."""
    if not regenerated_ids:
        return
    for pid in regenerated_ids:
        state.diagnosis_cache.pop(pid, None)
    state.eval_cache = [
        snapshot
        for snapshot in state.eval_cache
        if not any(p.probe_id in regenerated_ids for p in snapshot.probes)
    ]


def run(state: AgentDoctorState) -> AgentDoctorState:
    """Eval Agent 진입점."""
    state.current_agent = "eval"

    if not state.chunks:
        state.status = "error"
        state.error = "청크가 없습니다. Index Agent 완료 여부를 확인하세요."
        print(f"[Eval] 오류: {state.error}")
        return state

    # 진단 모드(비용 tier 상한): EVAL_MODE 환경변수. STEP3-2/STEP4/리포트가 이 값으로 게이팅된다.
    mode = resolve_mode()
    # 측정 self-gate 의 전역 tier 도 여기서 맞춘다 — 예전엔 diagnose() 진입점만 set_mode 를 불러,
    # STEP3 의 실패 판정이 모듈 기본값(FAST)으로 돌았다. 그러면 _grounded_ok/_abstained 의 tier3
    # 축이 통째로 빠져(_faith 가 None) 근거성만으로 실패하는 probe 가 성공으로 분류되고, 그
    # probe 는 오라클 답변 생성 대상에서 빠진다(STEP4 diagnose 는 DEEP 으로 실패 판정 → 불일치).
    # 매 run 마다 다시 세팅하므로 이전 iteration 의 전역이 새 실행에 새는 것도 함께 막는다.
    set_mode(mode)
    # state.iteration 은 raw 값을 그대로 찍는다(graph.py Orchestrator 로그와 표시 일치).
    # 예전엔 +1 을 더해 "다음 Optimize 방문에서 증가할 값"을 미리 보여줬는데, 같은 라벨이
    # 이어지는 방문(starts_new_label=False)에서는 실제로 증가하지 않아 Eval 로그만 매번
    # 부풀려진 값을 반복 표시하는 불일치가 있었다.
    print(f"[Eval] 청크 {len(state.chunks)}개, 반복 {state.iteration}/{state.max_iterations}"
          f" · 진단 모드 {_MODE_NAMES.get(mode, mode)}({mode})")
    state.eval_cache_hit = False

    # 진단 신호 캐시: 파이프라인 버전(index_config+코퍼스)이 바뀌면 무효화 → stale 재사용 방지.
    # 진단 신호(예: gold_in_wider_candidates)는 top_k 로 검색한 결과에 의존하므로,
    # 이 캐시는 index_config 를 포함하는 _pipeline_version 을 그대로 쓴다.
    version = _pipeline_version(state)
    if state.diagnosis_cache_version != version:
        state.diagnosis_cache = {}
        state.diagnosis_cache_version = version

    probe_version = corpus_version(state.chunks, state.documents)
    eval_cache_key = _eval_cache_key(
        state,
        mode,
        version,
        probe_version,
    )
    snapshot = _find_eval_snapshot(state, eval_cache_key)
    if snapshot is not None and snapshot.report is not None:
        state.probes = deepcopy(snapshot.probes)
        state.report = deepcopy(snapshot.report)
        state.diagnosis_cache = deepcopy(snapshot.diagnosis_cache)
        state.diagnosis_cache_version = snapshot.diagnosis_cache_version
        state.active_eval_key = eval_cache_key
        state.eval_cache_hit = True
        state.status = "evaluated"
        print(f"[Eval] 롤백 진단 캐시 복원: {eval_cache_key[:12]}")
        return state

    # Probe 캐시는 원문 문서에 의존한다. top_k뿐 아니라 chunk_size가 바뀌어도 같은
    # 질문/gold_spans를 유지하고, 불러온 뒤 현재 청크 기준 gold_chunk_ids만 재동기화한다.
    run_usage = snapshot_usage()
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
        with step("Eval", 1, "Probe 생성"):
            if probe_source == PROBE_SOURCE_MADE:
                probes = load_probes(probe_version, ignore_version=True)
                if probes:
                    probes = _resync_gold_chunk_ids(
                        probes,
                        state.chunks,
                        state.documents,
                    )
                    print(f"  made 소스 — 저장된 Probe {len(probes)}개 재사용")
                else:
                    print("  made 소스지만 저장된 Probe 없음 → 자동 생성 후 저장")
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
                    print(f"  저장된 Probe {len(probes)}개 재사용(버전 일치)")
        if not probes:
            print("[Eval] 경고: Probe 0개 생성 → 평가 불가, 통과 처리")
            state.probes = []
            state.report = build_report([], state.iteration, mode)
            state.status = "evaluated"
            eval_cache_key = _eval_cache_key(
                state,
                mode,
                version,
                probe_version,
            )
            _store_eval_snapshot(state, eval_cache_key)
            return state

        # 검색 인덱스 준비: 공통 RAG retriever가 Qdrant/keyword fallback을 오케스트레이션한다.
        # Eval은 검색 구현을 직접 들고 있지 않고, RAG 모듈의 동일한 검색 규칙을 재사용한다.
        # get_retriever(=캐시판): Index가 방금 적재한 같은 청크 집합이면 그 결과를 그대로
        # 재사용한다 — 예전엔 여기서 컬렉션 준비와 upsert를 통째로 한 번 더 했다.
        retriever = get_retriever(state.chunks, state.index_config)
        chunk_text = {c.chunk_id: c.text for c in state.chunks}
        top_k = int(state.index_config.get("top_k", DEFAULT_TOP_K))

        # tier2/tier3 판별 훅(재검색·코퍼스·RAGAS)이 쓸 자원 주입.
        # rerank_candidates 는 순위 라벨의 '도달 가능 창'(metrics_common.reachable_window)이다 —
        # 그보다 뒤 순위의 gold 는 리랭커 처방이 원리적으로 닿지 못한다.
        set_diag_context(client=retriever, chunks=state.chunks,
                         retrieve_fn=_retrieve_with_rag, keyword_fn=keyword_search,
                         dense_fn=_dense_retrieve_with_rag, ragas_fn=_ragas_track,
                         rerank_candidates=int(
                             state.index_config.get("rerank_candidates")
                             or DEFAULT_RERANK_CANDIDATES
                         ),
                         max_rerank_candidates=int(
                             (state.index_config.get("rerank_candidate_policy") or {})
                             .get("max_candidates")
                             or DEFAULT_MAX_RERANK_CANDIDATES
                         ))

        # ── STEP2: 검색 + 답변 생성 ───────────────────────────
        #   각 probe 의 신호 캐시(state.diagnosis_cache[probe_id])를 record 에 뷰로 주입 →
        #   진단 중 계산한 비싼 신호가 state 에 누적되어 재진단 시 재사용된다.
        #   LLM 호출(답변 생성)만 병렬화하고 검색·진단은 순차 유지 — Qdrant/임베딩/
        #   signals 전역이 병렬 구간에 들어가지 않게 하는 설계(계획 B안).
        concurrency = resolve_llm_concurrency()
        # index_config 를 전달해 Optimize(B그룹)의 프롬프트·온도 처방(temperature,
        # grounding_strict 등)이 실제 답변 생성에 반영되게 한다. generator 는 아는
        # 키만 읽고 나머지(chunk_size 등)는 무시한다. 없으면 기본값이 현 동작 유지.
        gen_config = state.index_config or {}
        with step("Eval", 2, "검색 + 답변 생성"):
            # Phase A(순차): 검색 + record 준비
            records = []
            for p in probes:
                rec = _prepare_record(p, retriever, chunk_text, top_k,
                                      state.diagnosis_cache.setdefault(p.probe_id, {}))
                records.append(rec)

            # Phase B(병렬): 실제 트랙 답변 생성만 동시 실행 (EVAL_LLM_CONCURRENCY, 1이면 순차).
            # 오라클 트랙은 여기서 안 만든다 — 실패로 판정된 probe 에만 필요해서 STEP3 로 미룬다.
            parallel_note = (f" 병렬 (동시성 {concurrency})"
                             if concurrency > 1 and len(records) > 1 else " 순차")
            print(f"  probe {len(probes)}개 · 실제 답변 {len(records)}건{parallel_note}")
            answers = parallel_map(
                lambda r: generate_answer(r.probe.question, r.retrieved_context, config=gen_config),
                records, concurrency)
            for rec, answer in zip(records, answers):
                rec.generated_answer = answer

        # ── STEP3: 지표 · 오라클 트랙 · RAGAS 진단 ─────────────
        # Phase B2(병렬): 실패 판정을 먼저 세우고, 그 판정으로 오라클 트랙(답변 생성 + RAGAS)을
        # 실패 probe 로 한정한다 — 오라클 소비처가 실패 경로뿐이라 성공 probe 몫은 순수 낭비다.
        # RAGAS 는 *_done 플래그를 세워 Phase C 의 _compute_ragas_real/_oracle 이 캐시 히트만
        # 하게 한다. _ragas_track 은 진단 전역과 무관한 모듈 함수라 병렬 구간에 안전.
        # 게이트는 _compute_ragas_* 와 동일(mode >= DEEP); LLM 비활성·키없음은
        # _ragas_track 이 {} 폴백이라 기존과 같은 동작으로 수렴한다.
        # 동시성 1이면 태스크 순서가 Phase C 호출 순서(probe 순)와 일치.
        with step("Eval", 3, "지표 · RAGAS 진단"):
            # B2-1: 실제 트랙 RAGAS 는 전 probe 에 필요하다 — 성공/실패 판정(_f1_ok 의 의미축)과
            # 리포트 RAGAS 평균이 모두 실제 트랙을 쓴다.
            if mode < Mode.DEEP:
                print(f"  모드 {_MODE_NAMES.get(mode, mode)} — RAGAS 생략 (deep 이상에서 실행)")
            else:
                real_scores = parallel_map(lambda r: _ragas_track(r, "real") or {}, records, concurrency)
                for rec, score in zip(records, real_scores):
                    rec.ragas, rec.ragas_done = score, True

            # B2-2: 실패 판정 — _is_success 는 오라클 필드를 안 읽으므로(recall·실제 트랙 정답
            # 판정·근거성·기권) 오라클보다 먼저 세울 수 있다. 이 판정 하나로 아래 오라클 답변
            # 생성과 오라클 RAGAS 를 함께 자른다. RAGAS 를 안 돌리는 모드에서도 답변 생성은
            # 잘라야 하므로 게이트가 mode 밖에 있다.
            # (_compute_metrics 는 순수·멱등이라 Phase C 에서 diagnose 가 다시 불러도 같은 값.)
            for rec in records:
                _compute_metrics(rec)
            failed = [rec for rec in records if _is_success(rec) is False]

            # B2-3: 오라클 답변 생성 — 실패 probe 중 gold context 가 있는 것만. 소비처가
            # B/C그룹 전제(_oracle_ok)뿐이고, 성공 probe 는 diagnose 가 성공 게이트에서 바로
            # 끝나 오라클 답을 아예 읽지 않는다. 미측정 성공분은 report 가 _oracle_ok 의
            # 추론 분기로 통과 처리한다.
            oracle_targets = [rec for rec in failed if rec.oracle_context]
            print(f"  실패 판정 {len(failed)}/{len(records)}건"
                  f" · 오라클 답변 {len(oracle_targets)}건 (실패 probe 만)")
            if oracle_targets:
                oracle_answers = parallel_map(
                    lambda r: generate_answer(r.probe.question, r.oracle_context, config=gen_config),
                    oracle_targets, concurrency)
                for rec, answer in zip(oracle_targets, oracle_answers):
                    rec.oracle_answer = answer
                    _compute_metrics(rec)       # 방금 생긴 오라클 답으로 oracle_f1 채우기

            # B2-4: 오라클 트랙 RAGAS — 같은 실패 집합. gold context 없는 probe 는
            # _ragas_track 이 {} 를 돌려주므로 기존과 같다.
            if mode >= Mode.DEEP and failed:
                print(f"  RAGAS 실제 {len(records)}건 / 오라클 {len(failed)}건")
                oracle_scores = parallel_map(lambda r: _ragas_track(r, "oracle") or {}, failed, concurrency)
                for rec, score in zip(failed, oracle_scores):
                    rec.oracle_ragas, rec.oracle_ragas_done = score, True

        # ── STEP4: 원인 판정 ──────────────────────────────────
        # Phase C(순차): 지표·진단·로그 — diagnose 는 signals 전역·진단 캐시·tier2
        # 재검색을 쓰므로 병렬 구간 밖에서 실행한다.
        with step("Eval", 4, "원인 판정"):
            print()
            for i, rec in enumerate(records, 1):
                rec.findings = diagnose(rec, mode)
                _log_probe(i, len(records), rec)
            _log_diagnosis_summary(records)
            _annotate_topic_cluster(records, state.chunks)

        # ── STEP4.5: bad_gold probe 재생성 (Option A — 다음 실행에서 재평가) ──
        # 정답셋 오류로 확정된 '우리(llm_generated)' probe 를 같은 근거 청크에서 재합성해
        # probe 파일에 반영한다. 이번 방문의 report/records/snapshot 은 건드리지 않아(현
        # 진단·점수 일관) 다음 파이프라인 실행이 재생성된 probe 를 로드해 재평가한다.
        # (user_log·멀티홉은 regenerate_probes 가 제외 → 리포트의 사용자 검수 요청으로 남는다.)
        regenerated = _maybe_regenerate_bad_gold(state, records, probes, probe_version)

        # ── STEP5: 리포트 ─────────────────────────────────────
        with step("Eval", 5, "리포트"):
            state.probes = probes
            state.report = build_report(records, state.iteration, mode)
            state.status = "evaluated"
            eval_cache_key = _eval_cache_key(
                state,
                mode,
                version,
                probe_version,
            )
            # 재생성이 일어났으면 이번 report/probes 는 곧 옛 시험지 기준이라, 같은 cache key
            # 로 snapshot 을 저장하면 다음 Eval 이 새 probe 대신 이 옛 성적표를 cache hit 한다.
            # 저장을 건너뛰어 다음 Eval 이 cache miss → 새 probe 로 재평가하게 한다.
            if regenerated:
                print("[Eval] bad_gold 재생성 실행 → 이번 평가 snapshot 저장 생략"
                      "(다음 실행이 새 probe 로 재평가)")
            else:
                _store_eval_snapshot(state, eval_cache_key)

    except Exception as e:  # 계약: 예외를 밖으로 던지지 않는다
        state.status = "error"
        state.error = f"평가 실패: {e}"
        print(f"[Eval] 오류: {e}")
    finally:
        print_summary(tag="Eval", stage="전체", since=run_usage)

    return state


def _annotate_topic_cluster(records: list[EvalRecord], chunks: list) -> None:
    """STEP4 후처리 — retrieval_semantic_mismatch 실패의 토픽 분포 신호를 finding 에 기록.

    개별 probe 로는 못 내는 cross-probe 신호라 diagnose() 밖(전 record 준비 후)에서 계산한다.
    실패한 semantic_mismatch probe 들의 '놓친 gold' 임베딩이 서로 뭉쳤나 흩어졌나를
    코퍼스 baseline 대비 비율로 판정해(agents/eval/topic_cluster.py), 그 값을 해당 라벨의
    모든 finding metadata['topic_cluster'] 에 실어 Optimize(planner)가 처방을 가르게 한다.

    'none' 도 명시적으로 단다 — rules.py 의 semantic_mismatch 처방은 none 을 "청크 희석
    (Case1) → 청킹 조정" 신호로 쓴다(shrink_chunk_size / switch_chunking 의
    applies_when={"topic_cluster":["none"]}). 여기서 none 을 안 달면 planner 가 '미측정
    =순차 fallback'으로 보아 임베딩 교체 처방까지 통과시켜, none 이 청킹만 선택하려던
    rules.py 계약이 깨진다.

    반대로 '아예 못 잰' 경우(임베딩 미부착/fallback, 실패 gold 2개 미만, baseline 측정
    불가)는 none 이 아니라 'unmeasured' 로 나간다 — 근거 없이 청킹 처방을 확정 선택하면
    안 되기 때문이다. unmeasured 는 어느 applies_when 허용 리스트에도 없어 planner 가
    순차 fallback 으로 되돌린다(agents/eval/topic_cluster.py 의 값 도메인 주석 참고).
    """
    sem_findings = [
        f
        for r in records
        for f in r.findings
        if f.label == "retrieval_semantic_mismatch"
    ]
    if not sem_findings:
        return

    embed_by_id = {c.chunk_id: c.embedding for c in chunks}

    # 실패 gold = semantic_mismatch probe 들이 '놓친' gold 청크(검색된 건 실패 근거가 아님).
    failed_ids: set[str] = set()
    for r in records:
        if any(f.label == "retrieval_semantic_mismatch" for f in r.findings):
            failed_ids |= missed_gold_ids(r)

    # sorted: set 순회 순서는 회차마다 달라질 수 있어, 표본을 자를 때 신호가 흔들린다.
    failed_vecs = [embed_by_id[cid] for cid in sorted(failed_ids) if embed_by_id.get(cid)]
    corpus_vecs = [c.embedding for c in chunks if c.embedding]

    # 유효성 필터·표본 절단은 classify_detail 안에서 양쪽 입력에 같은 순서로 걸린다
    # (_valid → stride). 여기서 미리 자르면 그 순서가 뒤집혀 영벡터가 표본 슬롯을 먼저
    # 먹는 문제가 되살아나고, 호출부가 topic_cluster 의 private 유효성 규칙에 묶인다.
    result = topic_cluster.classify_detail(failed_vecs, corpus_vecs)
    # 버킷뿐 아니라 판정 근거 수치도 metadata 에 남긴다 — 소비 유예(관측용) 동안
    # 이번 회차가 어떤 경계로 갈렸는지 finding 에서 재현·관측되게 한다.
    # 경계는 동적(1.0 ± k·C/sqrt(N)) 이라 이번 판정에 실제로 쓰인 값을 남긴다 —
    # 고정 상수를 남기면 어떤 경계로 갈렸는지 근거가 사라진다.
    spread_ratio, concentrated_ratio = topic_cluster.dynamic_bounds(result.failed_sample_size)
    for f in sem_findings:
        f.metadata["topic_cluster"] = result.bucket
        f.metadata["topic_cluster_detail"] = {
            "ratio": result.ratio,
            "failed_cohesion": result.failed_cohesion,
            "baseline": result.baseline,
            "failed_sample_size": result.failed_sample_size,
            "corpus_sample_size": result.corpus_sample_size,
            "concentrated_ratio": concentrated_ratio,
            "spread_ratio": spread_ratio,
        }
    ratio_str = f"{result.ratio:.3f}" if result.ratio is not None else "n/a"
    print(
        f"  topic_cluster={result.bucket} (ratio={ratio_str}, "
        f"failed_gold={result.failed_sample_size}, semantic_mismatch {len(sem_findings)}건)"
    )


def _log_diagnosis_summary(records: list[EvalRecord]) -> None:
    """STEP4 마감 요약 — 성공/실패 probe 수와 Finding 확정·예비 내역."""
    findings = [f for r in records for f in r.findings]
    failed = sum(1 for r in records if r.findings)
    confirmed = sum(1 for f in findings if f.confirmed)
    # '↳' 는 step() 의 마감줄 전용 — 여기서 같이 쓰면 STEP4 끝에 화살표 두 줄이 붙는다.
    line = f"  판정: 성공 {len(records) - failed} / 실패 {failed}"
    if findings:
        line += (f" · Finding {len(findings)}건 "
                 f"(확정 {confirmed} · 예비 {len(findings) - confirmed})")
    print(line)


# ── probe 1개 평가 (STEP2 → STEP3 → STEP4) ───────────────────────

def _full_log_text(text: str | None) -> str:
    """QAR 로그용 전체 텍스트. 줄바꿈만 접고 길이는 자르지 않는다."""
    return " ".join((text or "").split()) or "-"


def _fmt_metric(v, applicable: bool = True) -> str:
    """지표 포맷: 미측정/미해당(-1·None·비적용)이면 '-'."""
    if not applicable or v is None or (isinstance(v, (int, float)) and v < 0):
        return "-"
    return f"{v:.2f}" if isinstance(v, (int, float)) else str(v)


def _short_cid(cid: str) -> str:
    """로그용 청크 id 축약: '<doc-uuid>_chunk_016' → 'chunk_016', 그 외는 원본."""
    i = cid.rfind("_chunk_")
    return cid[i + 1:] if i != -1 else cid


def _cid_doc(cid: str) -> str:
    """청크 id 의 문서 부분(전체) — 문서 동일성 판정용: 'doc_ace9d8c1ce5d_chunk_016' → 'doc_ace9d8c1ce5d'.

    동일성은 반드시 전체 doc id 로 판정한다 — 절단값(_doc_tag)으로 판정하면 접두가 겹치는
    두 문서가 같은 것으로 접혀 걷어내려던 착시가 그대로 남는다."""
    i = cid.rfind("_chunk_")
    return cid[:i] if i != -1 else cid


def _doc_tag(cid: str) -> str:
    """로그 표시용 짧은 문서 태그(문서 해시 앞 6자): 'doc_ace9d8c1ce5d_chunk_016' → 'ace9d8'.

    _short_cid 가 문서 접두를 버려서 서로 다른 문서의 'chunk_005' 가 똑같이 보이는 착시를
    구분하려고 붙인다. 절단이라 표시 전용이고, 문서 동일성 판정에는 쓰지 않는다(_cid_doc)."""
    doc = _cid_doc(cid)
    if doc.startswith("doc_"):
        doc = doc[4:]
    return doc[:6]


def _fmt_cids(cids: list[str], colliding: set[str]) -> str:
    """청크 id 리스트를 로그용으로 축약. 같은 short_cid 가 문서 간 충돌하는 항목에만 문서
    태그(@)를 붙여 그 착시만 걷어낸다 — 충돌 없는 항목은 기존 표시를 유지한다(태그 남발 방지)."""
    parts = []
    for c in cids:
        short = _short_cid(c)
        parts.append(f"{short}@{_doc_tag(c)}" if short in colliding else short)
    return ", ".join(parts)


def _colliding_short_cids(*cid_groups: list[str]) -> set[str]:
    """검색∪골드에서 같은 short_cid 가 서로 다른 문서(_cid_doc)에 걸친 것들 — 실제 착시 대상.

    다문서 코퍼스에선 top-k 가 여러 문서에 걸치는 게 정상이라 '문서 ≥2'만으로 태그를 켜면
    거의 모든 줄에 붙는다. 착시는 동명 청크가 문서 간 충돌할 때만 생기므로 그 조건으로 좁힌다."""
    docs_by_short: dict[str, set[str]] = {}
    for group in cid_groups:
        for c in group:
            if c:
                docs_by_short.setdefault(_short_cid(c), set()).add(_cid_doc(c))
    return {short for short, docs in docs_by_short.items() if len(docs) > 1}


def _mark(ok: bool) -> str:
    """성공/실패 마크. 콘솔이 이모지를 못 그리면(Windows cp949 등) ASCII 로 폴백한다 —
    run_logger 의 Tee 가 '?' 로 치환하면 성공/실패 구분이 사라지기 때문.

    getattr 로 encoding 을 읽는 이유: run_logger._Tee 로 교체된 stdout 에는 encoding
    속성이 아예 없다. 속성 접근을 그대로 두면 AttributeError 가 run() 의 except 로
    올라가, 정상 진행된 평가가 통째로 error 로 뒤집힌다(실제로 그랬다)."""
    glyph = "✅" if ok else "❌"
    try:
        glyph.encode(getattr(sys.stdout, "encoding", None) or "utf-8")
    except (UnicodeEncodeError, LookupError):
        return "[OK]" if ok else "[FAIL]"
    return glyph


def _log_probe(idx: int, total: int, rec: EvalRecord) -> None:
    """probe 1개 평가 결과를 블록 형태로 출력(STEP4 진행 가시성용).
    질문(Q)·정답(A)·생성 답변(R)·검색/gold·지표·판정 라벨을 한 블록으로 남기고 빈 줄로 구분한다."""
    p = rec.probe
    meta = "·".join(filter(None, [p.source, p.qtype or "single"]))
    recall = _fmt_metric(rec.recall_at_k)
    f1 = _fmt_metric(rec.f1_score, bool(p.ground_truth))
    oracle = _fmt_metric(rec.oracle_f1, rec.oracle_answer is not None)
    # 같은 short_cid 가 서로 다른 문서에 걸친 항목에만 문서 태그를 붙인다 — 골드
    # doc_A_chunk_005 와 검색 doc_B_chunk_005 가 축약 표시로 겹쳐 'recall=0 인데 chunk_005 가
    # 검색에 있다'는 착시를 만드는 걸, 그 충돌 항목만 골라 걷어낸다(태그 남발 없이).
    colliding = _colliding_short_cids(rec.retrieved_chunk_ids, p.gold_chunk_ids)
    retrieved = _fmt_cids(rec.retrieved_chunk_ids, colliding)
    gold = _fmt_cids(p.gold_chunk_ids, colliding)
    # 판정은 finding 유무로 — diagnose 가 원인을 하나도 못 붙였으면 정상 처리된 probe 다.
    status = _mark(not rec.findings) + (f" {len(rec.findings)}건" if rec.findings else "")

    print(f"  [{idx}/{total}] {p.probe_id}  ({meta})  {status}")
    print(f"    Q: {_full_log_text(p.question)}")
    print(f"    A: {_full_log_text(p.ground_truth)}")
    print(f"    R: {_full_log_text(rec.generated_answer)}")
    print(f"    검색 [{retrieved}] / 골드 [{gold}]")
    if rec.retrieval_details:
        print(
            "    검색 실행: "
            f"mode={rec.retrieval_details.get('search_mode', '-')}, "
            "search_fallback="
            f"{str(bool(rec.retrieval_details.get('search_fallback_used'))).lower()}, "
            f"reranker={rec.retrieval_details.get('reranker_status', 'disabled')}"
        )
    # recall 은 gold_spans 가 있으면 span 커버리지(빈틈없이 덮어야 1점)라, 이름만 'recall@k' 로
    # 찍으면 '골드 청크가 검색 목록에 있는데 recall=0' 이 모순처럼 보인다. 기준과 청크 적중수를
    # 함께 남겨 그 착시를 없앤다.
    recall_note = f"recall@k({rec.recall_basis})={recall}"
    if p.gold_chunk_ids:
        hit = len(set(p.gold_chunk_ids) & set(rec.retrieved_chunk_ids))
        recall_note += f"  gold청크 {hit}/{len(p.gold_chunk_ids)} 검색"
    metric_line = f"    {recall_note}  f1={f1}  oracle_f1={oracle}"
    # 판정 기준은 f1 단독이 아니라 혼합 점수(answer_score = lexical·의미 가중합)다 —
    # 그 값과 의미축을 함께 남기지 않으면 'f1 이 낮은데 왜 통과(또는 통과 못)했나'를 로그만
    # 보고 알 수 없다. 의미축은 DEEP 에서만 측정되므로 있을 때만 붙인다.
    if p.ground_truth and rec.answer_semantic is not None:
        metric_line += (f"  answer={_fmt_metric(rec.answer_score)}"
                        f"(의미 {_fmt_metric(rec.answer_semantic)}"
                        f", 커버리지 {_fmt_metric(rec.gold_coverage)})")
    elif p.ground_truth and rec.ragas.get("answer_correctness_degraded"):
        # 의미축이 죽어 lexical 단독으로 판정된 probe. 이 줄이 없으면 'f1=1.00 인데 실패'가
        # 설명 없이 남는다 — 판정을 뒤집은 게 어휘 점수가 아니라 판정기 실패라는 걸 못 본다.
        metric_line += "  answer=f1 단독(의미축 degrade)"
    print(metric_line)
    # gold 용어 별칭(자산총계↔총자산 등)이 실제로 점수를 올렸을 때만 — gold 품질 검수 신호다.
    if p.ground_truth and rec.best_gold_answer_f1 > rec.raw_f1_score:
        print(
            f"    gold_variant: raw_f1={_fmt_metric(rec.raw_f1_score)} "
            f"best_f1={_fmt_metric(rec.best_gold_answer_f1)} variants={rec.gold_answer_variant_count}"
        )
    for f in rec.findings:
        mark = "" if f.confirmed else "(예비)"
        print(f"    ! {f.label}{mark}: {f.metadata.get('reason') or '-'}")
    print()  # 블록 구분 빈 줄


def _pipeline_version(state: AgentDoctorState) -> str:
    """진단 신호 캐시 무효화 키. index_config(Optimize가 바꿈)+코퍼스가 바뀌면 값이 달라진다.
    (재실행/코퍼스 의존 신호는 이 버전 내에서만 재사용 안전.)

    청크 id 와 함께 본문 hash 도 넣는다 — doc_id 는 출처로 고정돼 있어서(Ingest 의
    _stable_doc_id) 같은 파일을 고쳐도 id 는 그대로다. hash 를 빼면 본문이 바뀌었는데도
    버전이 같아져, 옛 gold_chunk_ids 를 가리키는 stale probe 를 재사용하게 된다."""
    meaningful_config = {
        name: value
        for name, value in state.index_config.items()
        if name not in {
            "rollback_cache_enabled",
            "rollback_cache_max_versions",
            "recreate_collection_on_dimension_mismatch",
        }
    }
    key = json.dumps(meaningful_config, sort_keys=True, default=str)
    key += f"|index={state.active_index_key}"
    key += "|chunks=" + ",".join(
        sorted(
            f"{chunk.chunk_id}:{chunk.hash or hashlib.sha1(chunk.text.encode('utf-8')).hexdigest()}"
            for chunk in state.chunks
        )
    )
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
    detailed_search = getattr(retriever, "search_with_details", None)
    retrieval = (
        detailed_search(probe.question, top_k=top_k)
        if callable(detailed_search)
        else None
    )
    if isinstance(retrieval, dict) and isinstance(retrieval.get("results"), list):
        hits = list(retrieval["results"])
        rec.retrieval_details = {
            key: value
            for key, value in retrieval.items()
            if key != "results"
        }
    else:
        # 테스트·외부 주입 retriever가 기존 search() 계약만 구현해도 동작을 유지한다.
        hits = retriever.search(probe.question, top_k=top_k)
    rec.retrieved = hits
    rec.retrieved_context = [h.get("text", "") for h in hits]
    rec.retrieved_chunk_ids = [h.get("chunk_id", "") for h in hits]

    # Oracle 트랙 컨텍스트 (gold context 가 있을 때만 — 답변은 Phase B 에서 생성)
    gold_ctx = [chunk_text[cid] for cid in probe.gold_chunk_ids if cid in chunk_text]
    if gold_ctx:
        rec.oracle_context = gold_ctx
    return rec
