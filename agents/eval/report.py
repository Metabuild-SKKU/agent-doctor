"""
agents/eval/report.py
STEP5: 진단 리포트 생성

설계 문서 'STEP 5: 진단 리포트' 구현.
모든 Probe 판정 결과(EvalRecord)를 집계해 DiagnosticReport 를 만든다.
    - overall_score : RAGAS 가중 평균(있으면) / 없으면 규칙 지표 폴백
    - pass_threshold: overall_score >= PASS_SCORE_THRESHOLD (Eval 의 점수 판정, 관측용)
                      주의: serve/optimize 운영 게이트는 agents/optimize/gate.py 가 소유한다
                      (점수 + 검색 바닥선 등 정책). 이 필드는 그 정책의 점수 부분만 담는다.
    - ragas_scores  : RAGAS 평균 + 규칙 지표 평균 + 결과 분포(관측용)
    - findings      : 전 record 의 Finding 합침(확정 우선 정렬)
    - findings_summary: 확정/예비·라벨 집계(+진단 모드). Optimize 가 확정건 우선 처리하도록 요약

  주의: overall_score/pass_threshold 는 '지표' 기반이라 예비 Finding 이 pass 를 뒤집지 않는다.
        예비는 '더 깊은 모드에서 확정할 수 있는 의심 원인'으로만 싣는다(정보 제공).

graph.route_after_eval() 이 report.pass_threshold 로 Serve/Optimize 분기를 결정한다.
"""
from __future__ import annotations

import uuid
from collections import Counter

from core.schema import DiagnosticReport
from agents.eval.types import (
    EvalRecord, RAGAS_WEIGHTS, PASS_SCORE_THRESHOLD, F1_PASS_THRESHOLD,
    resolve_mode,
)
from agents.eval.scoring import compute_composite, format_composite
from agents.eval.diagnose import _oracle_ok   # oracle 통과 판정은 진단과 같은 함수를 쓴다

_RAGAS_KEYS = ("faithfulness", "context_precision", "context_recall", "response_relevancy")


def is_bad_gold_probe(record: EvalRecord) -> bool:
    """이 probe 가 정답셋 '정답 텍스트' 오류(bad_gold_answer)로 확정 판정됐나 — 점수 제외 +
    재생성 대상. Eval agent(답 재생성)와 build_report(점수 제외)가 공유한다.

    청크 라벨 오류(bad_gold_chunk)는 여기 넣지 않는다 — 그건 정답 텍스트가 맞으므로 답을
    재생성하면 멀쩡한 정답을 망친다. 점수 제외는 is_gold_labeling_error 가 함께 처리한다."""
    return any(f.label == "bad_gold_answer" and f.confirmed for f in record.findings)


# 점수 집계에서 제외할 '거짓 실패' 라벨 — 정답 텍스트 오류(bad_gold_answer)와 근거 청크
# 포인터 오류(bad_gold_chunk) 둘 다. 어느 쪽이든 RAG 파이프라인이 아니라 평가셋(골드) 결함이라
# 점수에 넣으면 고칠 수 없는 페널티가 되고 Optimize 가 헛처방을 쫓는다.
_GOLD_ERROR_LABELS = ("bad_gold_answer", "bad_gold_chunk")

# 그중 **예비(confirmed=False)여도** 제외하는 라벨. 지금은 bad_gold_chunk 하나다.
#
# 2026-08-07 변경이 confirmed 요구를 _GOLD_ERROR_LABELS 전체에서 뺐는데, 실제로 검토한 건
# bad_gold_chunk 뿐이었다(리뷰 지적). bad_gold_answer 는 현재 생산자 둘 다 confirmed=True 라
# 무해하지만, 나중에 예비 bad_gold_answer 를 만드는 코드가 붙으면 **아무 논의 없이** 그 probe
# 들도 채점에서 빠진다. 검토한 범위만 넓혀 둔다.
_PRELIMINARY_EXCLUDED_LABELS = ("bad_gold_chunk",)


def is_gold_labeling_error(record: EvalRecord) -> bool:
    """골드 라벨 오류(정답 텍스트 또는 근거 청크)로 판정된 probe 인가 — 점수 제외용.
    재생성 대상(is_bad_gold_probe)보다 넓다: 청크 오라벨은 점수만 빼고 답은 안 건드린다.

    **예비(confirmed=False)도 제외한다(2026-08-07).** 확정만 제외하면 제외 여부가
    심판 LLM 의 흔들림을 그대로 탄다 — 같은 probe 가 회차마다 빠졌다 들어왔다 하면서
    종합점수를 흔든다. 실측(output/logs/corpus_20260804_103059.txt)에서 probe_qa_4195 가
    5회 중 4회는 확정 bad_gold_chunk 라 제외됐고, faithfulness 가 1.000→0.000 으로 튄
    1회만 다른 라벨이 붙어 실패로 집계됐다.

    제외의 근거는 결정론적인 쪽에 있다 — `_f1_ok`(답이 맞았다) 와 `not _oracle_ok`
    (골드만으로는 못 맞힌다)는 글자 비교라 재실행해도 같다. 그 둘로 "이 골드로는 이 답이
    안 나온다" 가 이미 성립하므로, 파이프라인이 아니라 평가셋 결함이라는 판단은 서 있다.
    faithfulness 는 **왜** 그런지(골드 오라벨 vs 파라미터 기억)만 가르고, 그건 라벨의
    확정/예비로 표시된다.

    예비 제외는 **검토한 라벨에만** 적용한다(_PRELIMINARY_EXCLUDED_LABELS) — bad_gold_answer
    는 예비가 생기면 그때 따로 판단할 일이지, 이 변경에 묻어가면 안 된다.

    ⚠️ 확정만 보던 시절보다 제외 범위가 넓어지므로 점수가 올라간다 — 이 변경 이전
    실행과 직접 비교하지 말 것(agents/eval/README.md 재베이스라인 절)."""
    return any(
        f.label in _GOLD_ERROR_LABELS
        and (f.confirmed or f.label in _PRELIMINARY_EXCLUDED_LABELS)
        for f in record.findings
    )


# 하위호환 별칭(내부 호출부).
_is_bad_gold_probe = is_bad_gold_probe


def build_report(records: list[EvalRecord], iteration: int, mode: int | None = None) -> DiagnosticReport:
    """records 를 집계해 DiagnosticReport 반환.

    mode(진단 모드)는 findings_summary 에 기록해 '이 findings 가 어느 깊이에서 나왔는지'를 남긴다.
    (예비 findings 는 더 깊은 모드에서 확정 가능하다는 맥락.) 미지정 시 EVAL_MODE/기본값.
    """
    if mode is None:
        mode = resolve_mode()

    # 골드 라벨 오류(bad_gold_answer=정답 텍스트, bad_gold_chunk=근거 청크)로 판정된 probe 는
    # '거짓 실패'다 — RAG 결함이 아니라 평가셋이 틀린 것이라, 점수에 넣으면 파이프라인이 고칠 수
    # 없는 문제로 페널티를 받고 Optimize 가 존재하지 않는 결함을 쫓게 된다.
    # (전부 제외되면 scorable 이 비어 overall=None → 기존 '판정 보류' 통과 경로.)
    scorable = [r for r in records if not is_gold_labeling_error(r)]

    # **채점에서 뺀 probe 는 처방도 만들지 않는다**(2026-08-10, 리뷰 지적).
    #
    # 예전에는 findings 를 records 전체로 만들었다. 그러면 골드가 의심스러워 채점에서 뺀
    # probe 의 확정 라벨(generation_hallucination 등)이 planner 로 넘어가 처방을 만드는데,
    # 정작 그 probe 는 composite 에 없어 **KEEP/ROLLBACK 게이트가 효과를 볼 수 없다.**
    # 처방은 하는데 측정은 못 하는 상태였다. 실측 재현:
    #
    #     recall=1.0, real faith=0.0, oracle faith=0.0
    #       예비 bad_gold_chunk · 확정 generation_hallucination
    #                           · 확정 generation_parametric_overreliance
    #       → is_gold_labeling_error=True 라 채점 제외, 그런데 처방은 나감
    #
    # 그래서 점수와 처방을 **같은 집합**에서 낸다. 제외 판정은 결정론적 신호만 쓰므로
    # (오라클 있음 ∧ 오라클 실패 ∧ f1 통과) 회차마다 흔들리지 않는다.
    #
    # 골드 오류 라벨 자체는 남긴다 — 그게 이 probe 를 사람 검수로 보내는 통로이고
    # (report_view 의 "골드 청크 재지정 필요"), D그룹이라 자동 처방 대상도 아니다.
    # 같이 섰던 파이프라인 라벨은 findings_summary 에 집계로 남아 관측은 가능하다.
    gold_probe_ids = {r.probe.probe_id for r in records if is_gold_labeling_error(r)}
    findings = [
        f for r in records for f in r.findings
        if r.probe.probe_id not in gold_probe_ids or f.label in _GOLD_ERROR_LABELS
    ]
    # Optimize 소비 편의: 확정 우선. 동률은 원래 순서 유지(stable sort — probe별 D→A→C→B).
    findings.sort(key=lambda f: not f.confirmed)

    ragas_means = _ragas_means(scorable)
    rule_means = _rule_means(scorable)
    overall = _overall_score(ragas_means, rule_means)
    oracle_acc = _oracle_accuracy(scorable)

    scores = {**rule_means}
    scores.update(ragas_means)                          # RAGAS 평균(있으면)
    reranker_runtime = _reranker_runtime(records)
    # 브랜치 제거 → findings 유무로 결과 분포(진단됨/정상)
    n_diag = sum(1 for r in records if r.findings)
    scores["outcome_distribution"] = {"diagnosed": n_diag, "ok": len(records) - n_diag}

    # 정답 판정 degrade 건수 — answer_correctness 의 factual(TP/FP/FN 분류) 성분을 못 재서
    # 의미유사도 단독으로 계산된 probe 수. 유사도는 부정문·근접 오답에서도 높게 나오므로,
    # 이 값이 크면 '이번 실행의 정답 판정이 근접 오답을 못 걸렀을 수 있다'는 뜻이다.
    degraded = _degraded_correctness_count(records)
    if degraded:
        scores["answer_correctness_degraded"] = degraded

    # lexical 미달인데 의미축으로 통과한 probe 수 — 채점 방식 변경의 영향 크기이자
    # gold 품질 검수 후보(_semantic_rescued_count 주석 참고).
    rescued = _semantic_rescued_count(records)
    if rescued:
        scores["semantic_rescued"] = rescued

    # fused RAGAS 가 개별 호출로 되돌아간 지표 수 — 0 이 정상(트랙당 chat 1회 전제).
    repaired = _fused_repair_count(records)
    if repaired:
        scores["fused_repaired"] = repaired

    # 골드 라벨 오류로 채점에서 빠진 probe 수. 제외를 조용히 하면 '30문항 중 몇 개가 애초에
    # 채점 불가였나'를 알 수 없어 점수 해석과 정답셋 검수 우선순위가 둘 다 흐려진다.
    gold_errors = sum(1 for r in records if is_gold_labeling_error(r))
    if gold_errors:
        scores["gold_labeling_errors"] = gold_errors

    # 임베딩 출처(api|local|none) — response_relevancy·answer_correctness 유사도 성분이
    # 어떤 벡터 공간에서 나왔는지. 모델이 다르면 코사인 분포가 달라 실행 간 점수를
    # 그대로 비교할 수 없으므로, 나중에 성적표만 보고 구분할 수 있게 남긴다.
    scores["embedding_source"] = _embedding_source()

    # 평가 신호(GT 규칙지표/RAGAS)가 전혀 없으면 진단 불가 →
    # eval 한계로 파이프라인을 막지 않도록 통과 처리(overall_score=None).
    # [설계 결정] 이건 "판정 보류"이지 "품질 확인"이 아니다 — ground_truth 없는 probe만 있거나
    # (예: user_log 소스) RAGAS 미실행(EVAL_MODE<deep)이면 애초에 점수를 낼 근거 자체가 없다.
    # 근거 없이 fail 로 강제하면 Optimize 가 존재하지도 않는 문제를 잡으러 무한 루프를 돌 수
    # 있으므로, "판단 불가 → 막지 않음"을 택했다(Serve 이후에도 report.overall_score is None
    # 으로 이 상태는 추적 가능하다).
    if overall is None:
        overall_val, pass_thr = None, True
        print("[Eval] 경고: 평가 신호 없음(GT·RAGAS 부재) → 통과 처리")
    else:
        overall_val, pass_thr = round(overall, 4), overall >= PASS_SCORE_THRESHOLD

    report = DiagnosticReport(
        report_id=f"report_{uuid.uuid4().hex[:8]}",
        findings=findings,
        failed_questions=_failed_questions(records),
        findings_summary=_findings_summary(records, mode),
        ragas_scores=scores,
        runtime_summary={"reranker": reranker_runtime,
                         "search": _search_runtime(records)},
        oracle_accuracy=oracle_acc,
        overall_score=overall_val,
        composite_score=compute_composite(scorable).as_dict(),  # 종합점수(0~100) — bad_gold 제외
        pass_threshold=pass_thr,
        iteration=iteration,
    )

    _print_summary(records, report)
    return report


# ── 집계 헬퍼 ─────────────────────────────────────────────────────

def _failed_questions(records: list[EvalRecord]) -> list[dict]:
    """실패한 probe의 질문·기대 정답·실제 답변을 리포트에 보존한다.

    EvalRecord는 Eval 실행이 끝나면 사라지므로 UI가 재구성하지 않고 실제 생성 답변을
    표시할 수 있게 필요한 원문만 직렬화 가능한 dict로 남긴다.

    골드 라벨 오류(is_gold_labeling_error)는 뺀다 — 파이프라인이 실제 근거를 찾아 정답을
    낸 probe 라 '실패한 검증 질문'이 아니고, 점수에서도 이미 빠져 있다(scorable). 남기면
    사람이 볼 실패 목록이 고칠 수 없는 항목으로 채워져 진짜 실패가 묻힌다. 이 probe 들은
    사라지지 않는다 — findings_summary 의 bad_gold_* 집계와 Serve 의 수동 권고(골드 재지정)가
    질문 목록과 조치 방법을 따로 싣는다.
    """
    return [
        {
            "probe_id": record.probe.probe_id,
            "question": record.probe.question,
            "expected_answer": record.probe.ground_truth or "",
            "actual_answer": record.generated_answer or "",
        }
        for record in records
        if record.findings and not is_gold_labeling_error(record)
    ]


def _findings_summary(records: list[EvalRecord], mode: int) -> dict:
    """Finding 들을 확정/예비·라벨로 집계. Optimize 가 확정건 우선 처리하도록 요약 제공.

    라벨 분포는 **테스트셋(probe)당 1로 정규화**해 가중 집계한다: 한 probe 에서 finding 이
    N개 나오면 각 finding 은 1/N 로 계산된다(예: 3개 → 각 0.333). 특정 probe 가 여러 원인을
    동시에 내도, 원인별 분포·우선순위가 그 probe 하나에 의해 과대 계상되지 않게 하기 위함.

      mode              : 이 진단이 실행된 모드(1~4). 예비가 왜 예비인지의 맥락.
      total/confirmed/preliminary : finding **원시 개수**(정수).
      weighted_total    : probe당 정규화 합 = finding 이 1개 이상 나온 probe 수(라벨 가중합의 총계).
      confirmed_labels / preliminary_labels : 라벨별 **가중 개수**(probe당 1/N, 소수 3자리).
    """
    all_findings = [f for r in records for f in r.findings]
    confirmed = [f for f in all_findings if f.confirmed]
    preliminary = [f for f in all_findings if not f.confirmed]

    # probe당 정규화 가중치 w=1/N 로 라벨 분포를 가중 집계 (N = 그 probe 의 finding 수)
    conf_w: dict[str, float] = {}
    prelim_w: dict[str, float] = {}
    for r in records:
        n = len(r.findings)
        if not n:
            continue
        w = 1.0 / n
        for f in r.findings:
            bucket = conf_w if f.confirmed else prelim_w
            bucket[f.label] = bucket.get(f.label, 0.0) + w

    return {
        "mode": mode,
        "total": len(all_findings),
        "confirmed": len(confirmed),
        "preliminary": len(preliminary),
        "weighted_total": round(sum(conf_w.values()) + sum(prelim_w.values()), 3),
        "confirmed_labels": {k: round(v, 3) for k, v in conf_w.items()},
        "preliminary_labels": {k: round(v, 3) for k, v in prelim_w.items()},
    }


def _embedding_source() -> str:
    """임베딩 출처 — "api" | "local" | "none".

    같은 지표라도 벡터 공간이 다르면 코사인 분포가 달라 실행 간 비교가 성립하지 않는다.
    성적표만 보고 구분할 수 있게 리포트에 남긴다(agents/eval/llm_provider.embed_texts
    의 폴백 사슬과 같은 판정)."""
    try:
        from agents.eval import llm_provider

        if llm_provider._api_embeddings_available():
            return "api"
        return "local" if llm_provider._local_embeddings_available() else "none"
    except Exception:
        return "none"


def _semantic_rescued_count(records: list[EvalRecord]) -> int:
    """lexical(f1) 은 문턱 미달인데 의미축 덕에 통과한 probe 수 — gold 품질 검수 후보.

    두 가지 용도로 남긴다.
      1) 이번 실행에서 혼합 점수가 판정을 몇 건 뒤집었는지 = 채점 방식 변경의 영향 크기.
         (composite 재베이스라인·문턱 재보정 때 '점수가 왜 올랐나'의 답이 이 수치다.)
      2) 그 probe 들은 대개 gold 가 질문이 묻지 않은 요소까지 담아 lexical 이 구조적으로
         낮게 나온 경우다 — bad_gold_answer 가 '정답셋 이상'으로 잡던 유형인데, 이제
         통과하므로 라벨로는 안 남는다(성공 probe 는 findings 가 비어야 한다는 보장 때문).
         라벨 대신 이 카운터가 그 신호를 들고 있다.
    """
    return sum(
        1 for r in records
        if r.probe.ground_truth
        and r.answer_semantic is not None
        and r.f1_score < F1_PASS_THRESHOLD <= r.answer_score
    )


def _degraded_correctness_count(records: list[EvalRecord]) -> int:
    """answer_correctness 가 factual 성분 없이(=의미유사도 단독) 계산된 probe 수.
    두 트랙 중 하나라도 degrade 됐으면 1건으로 센다."""
    return sum(
        1 for r in records
        if r.ragas.get("answer_correctness_degraded")
        or r.oracle_ragas.get("answer_correctness_degraded")
    )


def _fused_repair_count(records: list[EvalRecord]) -> int:
    """fused RAGAS 가 결손으로 개별 호출을 되살린 지표 수(두 트랙 합).

    fused 는 '트랙당 chat 1회'가 전제인데, 응답 절단·파싱 실패로 보수가 돌면 그 전제가
    조용히 깨진다. 로그로만 남으면 회귀를 못 보므로 리포트 지표로 올린다 — 값이 크면
    EVAL_RAGAS_FUSED_MAX_TOKENS 를 올리거나 프롬프트 블록 수를 줄일 신호다."""
    return sum(
        int(r.ragas.get("fused_repaired") or 0) + int(r.oracle_ragas.get("fused_repaired") or 0)
        for r in records
    )


def _reranker_runtime(records: list[EvalRecord]) -> dict:
    """이번 Eval에서 reranker가 실제 점수를 만든 횟수를 운영 신호로 집계한다."""
    enabled_probes = sum(
        1 for record in records
        if record.retrieval_details.get("reranker_enabled")
    )
    attempted = sum(
        1
        for record in records
        if record.retrieval_details.get("reranker_attempted")
    )
    applied = sum(
        1
        for record in records
        if record.retrieval_details.get("reranker_status") == "applied"
        or record.retrieval_details.get("reranked")
    )
    status_counts = Counter(
        str(record.retrieval_details.get("reranker_status", "disabled"))
        for record in records
    )
    # 비용도 함께 집계한다 — 검색 시간의 대부분이 리랭크(쌍 수 × 쌍당 텍스트 길이)에서 나오는데
    # 실행 여부만 남기면 chunk_size 처방이 쌍당 비용을 몇 배로 올려도 아무 신호가 안 남는다.
    seconds = sum(_num(r.retrieval_details.get("rerank_seconds")) for r in records)
    pairs = int(sum(_num(r.retrieval_details.get("rerank_pairs")) for r in records))
    return {
        "enabled": enabled_probes > 0,
        "enabled_probes": enabled_probes,
        "attempted": attempted,
        "applied": applied,
        "failed": max(0, attempted - applied),
        "status_counts": dict(status_counts),
        "seconds": round(seconds, 2),
        "pairs": pairs,
        "ms_per_pair": round(seconds * 1000 / pairs, 1) if pairs else None,
        # 상한이 걸린 채 돌았나 — 폴백으로 빠진 실행과 구분해야 다음 처방 판정이
        # capped/uncapped 를 같은 조건으로 묶지 않는다. None = 상한 없이 돎.
        "max_lengths": sorted(
            {
                r.retrieval_details.get("rerank_max_length")
                for r in records
                if r.retrieval_details.get("rerank_pairs")
            },
            key=lambda v: (v is None, v),
        ),
        # 어디서 계산했나("local:cuda" / "openrouter:<model>"). max_lengths 와 같은 이유로
        # 남긴다 — 경로가 바뀌면 리랭커 **모델 자체**가 바뀌므로(로컬 bge ↔ Cohere), 두
        # 실행의 점수 차이를 처방 효과로 읽으면 안 된다. 실제로 리랭크가 돈 건만 센다.
        "routes": sorted(
            {
                str(r.retrieval_details.get("rerank_route"))
                for r in records
                if r.retrieval_details.get("rerank_pairs")
                and r.retrieval_details.get("rerank_route")
            }
        ),
    }


def _search_runtime(records: list[EvalRecord]) -> dict:
    """검색(STEP2 Phase A) 총시간과 그중 리랭크 몫. 느린 구간의 귀속을 한 번에 가른다.

    리랭크 시간만 있으면 분자만 아는 셈이라 '검색이 느렸나, 생성이 느렸나'를 콘솔 로그와
    눈으로 대조해야 한다. 검색 총시간을 함께 남기면 STEP2 소요에서 빼는 것만으로 생성 몫도
    나온다(검색 = Σsearch_seconds / 생성 = STEP2 소요 - 검색).
    """
    seconds = sum(_num(r.retrieval_details.get("search_seconds")) for r in records)
    rerank = sum(_num(r.retrieval_details.get("rerank_seconds")) for r in records)
    return {
        "seconds": round(seconds, 2),
        # search_seconds 를 실은 레코드만 센다 — 옛 계약(키 없음) 레코드까지 세면
        # 분자에 기여하지 않는 건이 분모에 들어가 건당 시간이 희석된다.
        "searches": sum(
            1 for r in records if r.retrieval_details.get("search_seconds") is not None
        ),
        "rerank_share": round(rerank / seconds, 3) if seconds > 0 else None,
    }


def _num(value) -> float:
    """retrieval_details 에서 읽은 수치를 float 로. 없거나 형식 불량이면 0.0
    (옛 계약의 retriever·테스트 주입 결과에는 키가 아예 없다)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _ragas_means(records: list[EvalRecord]) -> dict:
    """실제 트랙 RAGAS 지표별 평균 (측정된 것만)."""
    means = {}
    for key in _RAGAS_KEYS:
        vals = [r.ragas[key] for r in records
                if r.ragas.get(key) is not None]
        if vals:
            means[key] = sum(vals) / len(vals)
    return means


def _rule_means(records: list[EvalRecord]) -> dict:
    """
    규칙 지표 평균 (관측·폴백 점수용).
      - recall : gold_chunk_ids 있는 record 만 (-1 제외)
      - f1/oracle : ground_truth 있는 record 만 (정답 없으면 무의미)
      - exact_match : KorQuAD 공식 EM 평균 — 관측용으로만 남긴다(overall_score 는 recall·f1 만 씀).
    """
    recalls = [max(0.0, r.recall_at_k) for r in records if r.recall_at_k >= 0]
    gt = [r for r in records if r.probe.ground_truth]
    f1s = [r.f1_score for r in gt]
    # 판정에 쓰이는 값은 f1 단독이 아니라 혼합 점수(answer_score)다 — 의미축이 측정된 실행에서는
    # 그 평균도 함께 남긴다. f1 만 보면 '판정 기준'과 '표시 숫자'가 어긋난다.
    answer_scores = [r.answer_score for r in gt if r.answer_semantic is not None]
    # 오라클은 실패 probe 에만 생성되므로(agent.py STEP3) 실측된 record 만 평균 낸다 —
    # 미측정분을 0 으로 섞으면 성공이 많을수록 평균이 내려가는 거꾸로 된 지표가 된다.
    oracles = [r.oracle_f1 for r in gt if r.oracle_answer is not None]
    ems = [1.0 if r.exact_match else 0.0 for r in gt]
    out = {}
    if recalls:
        out["mean_recall_at_k"] = sum(recalls) / len(recalls)
    # 근거 밀도 — recall 의 짝(관측용, 게이트 미적용). recall 은 많이 퍼올수록 오르는데
    # 그 비용을 재는 축이 없어 top_k 부풀리기·거대 청크가 개선처럼 보였다.
    # span 좌표로만 계산되므로 chunk 폴백 record 는 None 이라 평균에서 빠진다.
    precisions = [r.span_precision for r in records if r.span_precision is not None]
    if precisions:
        out["mean_span_precision"] = sum(precisions) / len(precisions)
    if f1s:
        out["mean_f1"] = sum(f1s) / len(f1s)
    if ems:
        out["mean_exact_match"] = sum(ems) / len(ems)   # KorQuAD EM (F1 과 나란히, 관측)
    if answer_scores:
        out["mean_answer_score"] = sum(answer_scores) / len(answer_scores)
    if oracles:
        out["mean_oracle_f1"] = sum(oracles) / len(oracles)
        out["oracle_measured"] = len(oracles)   # mean_oracle_f1·oracle_accuracy 의 실측 표본 수
    return out


def _overall_score(ragas_means: dict, rule_means: dict) -> float | None:
    """
    RAGAS 4지표가 하나라도 있으면 설계 §7 가중 평균(있는 것만 재정규화).
    없으면 규칙 지표 폴백: recall 과 F1 의 평균.
    평가 신호가 전혀 없으면 None(진단 불가).
    """
    present = {k: v for k, v in ragas_means.items() if k in RAGAS_WEIGHTS}
    if present:
        wsum = sum(RAGAS_WEIGHTS[k] for k in present)
        return sum(present[k] * RAGAS_WEIGHTS[k] for k in present) / wsum

    # 폴백: 규칙 지표
    recall = rule_means.get("mean_recall_at_k")
    f1 = rule_means.get("mean_f1")
    parts = [v for v in (recall, f1) if v is not None]
    return sum(parts) / len(parts) if parts else None


def _oracle_accuracy(records: list[EvalRecord]) -> float | None:
    """Oracle 트랙 통과율 (ground_truth 보유 record 중 oracle 통과 비율).

    판정은 diagnose._oracle_ok 을 그대로 쓴다 — lexical(oracle_f1)뿐 아니라 DEEP 에서
    측정된 RAGAS answer_correctness 강등까지 반영해야, '진단은 oracle 실패인데 리포트엔
    성공으로 집계'되는 어긋남이 생기지 않는다.

    오라클 답변은 실패 probe 에만 생성하므로(agent.py STEP3), 미측정 성공 probe 는 통과로
    추론한다 — 성공의 전제가 recall=1 이라 gold 는 이미 실제 context 안에 있었고 그 답이
    근거까지 맞았다. 분모는 그대로 gt 전체라 이전 실행과 비교 가능하고, 실측 표본 수는
    scores 의 oracle_measured 로 남는다.

    성공 판정을 _is_success 로 다시 묻지 않고 findings 로 읽는 이유: _is_success 는
    _faith/_abstention_judged 를 타서 LLM 을 부를 수 있는데, 리포트는 LLM 을 안 부른다.
    'findings 가 비었으면 성공' 은 diagnose 의 성공 게이트가 세우는 보장이다."""
    gt = [r for r in records if r.probe.ground_truth]
    if not gt:
        return None
    passed = sum(1 for r in gt
                 if _oracle_ok(r) or (r.oracle_answer is None and not r.findings))
    return passed / len(gt)


# ── 로그 요약 ─────────────────────────────────────────────────────

def _fmt_gate_value(value: object) -> str:
    """gate 상세값을 로그 한 줄에 넣기 좋게 포맷한다."""
    if value is None:
        return "-"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def _print_summary(records: list[EvalRecord], report: DiagnosticReport) -> None:
    # 들여쓰기 2칸: agents/eval/agent.py 의 STEP5 스텝 블록 본문으로 출력된다
    # (스텝 헤더가 이미 'Eval STEP5' 를 말하므로 줄마다 접두사를 다시 붙이지 않는다).
    from agents.optimize import gate

    fs = report.findings_summary
    gate_summary = gate.explain_report(report)
    gate_pass = bool(gate_summary["pass"])
    gate_mark = "✓" if gate_pass else "✗"
    print(f"  종합점수 {format_composite(report.composite_score)}"
          f" · overall {report.overall_score} · gate_pass {gate_mark}")
    if gate_summary["reason"] != "passed":
        detail = []
        if gate_summary["composite_total"] is not None:
            detail.append(
                "composite="
                f"{_fmt_gate_value(gate_summary['composite_total'])}/"
                f"{_fmt_gate_value(gate_summary['composite_threshold'])}"
            )
        if gate_summary["mean_recall_at_k"] is not None:
            detail.append(
                "recall="
                f"{_fmt_gate_value(gate_summary['mean_recall_at_k'])}/"
                f"{_fmt_gate_value(gate_summary['recall_floor'])}"
            )
        suffix = f" ({', '.join(detail)})" if detail else ""
        print(f"  · gate reason: {gate_summary['reason']}{suffix}")
    if report.pass_threshold != gate_pass:
        eval_mark = "✓" if report.pass_threshold else "✗"
        print(f"  · eval_pass {eval_mark} — overall 기준 점수 판정이며 최종 gate 와 별도")
    rs = report.ragas_scores
    if "mean_f1" in rs:   # KorQuAD식 관측 지표: F1 과 EM 을 나란히
        line = f"  정답매칭(관측) F1={rs['mean_f1']:.3f}  EM={rs.get('mean_exact_match', 0.0):.3f}"
        if "mean_answer_score" in rs:   # 실제 판정 기준(혼합 점수) 평균
            line += f"  판정점수={rs['mean_answer_score']:.3f}"
        print(line)
    if rs.get("gold_labeling_errors"):
        print(f"  · 골드 검수 {rs['gold_labeling_errors']}건 — 답은 맞았는데 골드 라벨이 틀려 "
              f"채점에서 제외(실패 아님). 정답셋 수정 대상")
    if rs.get("semantic_rescued"):
        print(f"  · lexical 미달인데 의미축으로 통과 {rs['semantic_rescued']}건 — "
              f"gold 가 질문이 안 물은 요소까지 담았는지 검수 후보(정답셋 품질)")
    if rs.get("answer_correctness_degraded"):
        print(f"  ⚠ 정답 판정 degrade {rs['answer_correctness_degraded']}건 — "
              f"판정기(TP/FP/FN 분류) 실패로 의미유사도 단독 계산. 근접 오답을 못 걸렀을 수 있음")
    if rs.get("fused_repaired"):
        print(f"  ⚠ RAGAS fused 보수 {rs['fused_repaired']}건 — 통합 응답 결손으로 개별 호출 재실행. "
              f"EVAL_RAGAS_FUSED_MAX_TOKENS 를 올릴 신호")
    runtime = report.runtime_summary or {}
    search = runtime.get("search") or {}
    rerank = runtime.get("reranker") or {}
    # 검색 비용 — chunk_size 처방이 검색 시간에 준 영향과 그중 리랭크 몫을 한 줄로.
    if search.get("seconds"):
        line = f"  검색 {search['seconds']}초 (probe {search['searches']})"
        if rerank.get("pairs"):
            line += (f" · 리랭크 {rerank['seconds']}초 {rerank['pairs']}쌍 "
                     f"(쌍당 {rerank['ms_per_pair']}ms, 검색의 {search['rerank_share']})")
        # 실패한 리랭크의 벽시계는 쌍당 비용에서 빼지만(순도), 그 사실까지 지우면 안 된다 —
        # 큰 청크로 추론이 오래 돌다 실패하면 그 시간은 검색 총시간에만 남아, 리랭크가 원인인데
        # '검색이 느렸다'로 읽힌다.
        if rerank.get("failed"):
            line += f" · 리랭크 실패 {rerank['failed']}건(시간 미집계)"
        print(line)
    if report.findings:
        # 타입·라벨 분포 모두 probe당 1로 정규화(가중): 한 probe 의 N개 finding → 각 1/N
        # (타입=처방 그룹 4종, 라벨=세분화 진단명. 타입만 보면 gap 처럼 뭉뚱그려져 원인이 안 보인다.)
        by_type: dict[str, float] = {}
        for r in records:
            k = len(r.findings)
            if not k:
                continue
            for f in r.findings:
                by_type[f.type] = round(by_type.get(f.type, 0.0) + 1.0 / k, 3)
        # 라벨분포는 findings_summary 가 이미 같은 가중치로 집계한 값을 병합 재사용
        by_label: dict[str, float] = {}
        for src in (fs.get("confirmed_labels", {}), fs.get("preliminary_labels", {})):
            for label, w in src.items():
                by_label[label] = round(by_label.get(label, 0.0) + w, 3)
        # Finding 개수(확정/예비)는 STEP4 마감줄이 이미 보고했다 — 여기선 분포만.
        print(f"  가중 타입분포 {by_type}")
        print(f"  가중 라벨분포 {dict(sorted(by_label.items(), key=lambda kv: -kv[1]))}")
        if fs.get("preliminary"):
            print(f"  예비 {fs['preliminary']}개는 더 깊은 모드(EVAL_MODE=deep/full)에서 확정 가능")
