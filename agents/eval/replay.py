"""
agents/eval/replay.py
로그 리플레이 모드 - 외부 RAG 실행 로그를 Eval 진단기(STEP3~5)에 직접 잇는 연결부

다른 팀이 만든 RAG의 실행 로그(triad: 질문·검색 컨텍스트·답변)를 EvalRecord로
변환해, 우리 STEP2(자체 검색·생성)를 건너뛰고 STEP3~5(지표·점수·리포트)를
그대로 재사용한다(docs/external_rag_log_intake.md §6-1). 산출물 수준은 (a)
점수 리포트 - faithfulness(환각)·answer relevancy(동문서답) + overall/composite.
원인 라벨(ext_ 세트, 문서 §5)은 후속 PR.

정직성 규약:
- gold 청크가 없으므로 recall_at_k는 -1(미측정 센티널)로 명시한다. EvalRecord
  기본값 0.0을 그대로 두면 report._rule_means가 "진짜 0점"으로 집계한다.
- 로그에 ground_truth(정답 텍스트)가 있으면 규칙 char F1과 RAGAS의
  correctness/context_precision/recall까지 실측된다(없으면 faithfulness/relevancy만).
- faithfulness가 실측되면 retrieval_axis(검색축 신뢰도 seam)로 넘긴다 -
  reliability_score가 recall 센티널(-1→0 클램프)로 오염되는 것을 막는다.
  GT가 있는데 LLM이 없는 실행은 이 seam이 비어 composite가 보수적으로 나온다.

CLI: python -m agents.eval.replay <log.jsonl> [--limit N]
     RAGAS 실측에는 EVAL_ENABLE_LLM=1 과 LLM 키가 필요하다(없으면 규칙 지표만).
"""

from __future__ import annotations

import sys

from core.schema import DiagnosticReport, Probe

from agents.eval.log_intake import (
    TIER_NONE, ExternalLogRecord, assess_capability, load_external_log,
)
from agents.eval.metrics_basic import char_f1
from agents.eval.metrics_ragas import _judge, evaluate_real_track
from agents.eval.report import build_report
from agents.eval.types import EvalRecord, llm_eval_enabled
from agents.eval.scoring import format_composite


# ── 로그 → EvalRecord 변환 ───────────────────────────────────────

def build_replay_records(logs: list[ExternalLogRecord]) -> list[EvalRecord]:
    """외부 로그 레코드를 Eval이 소비하는 EvalRecord로 변환한다.

    probe.source는 "user_log" - 로그의 질문은 실사용 질문이라 probe 외부성
    원칙(청크에서 뽑지 않는다)을 자동으로 만족한다. oracle 트랙 재료
    (oracle_answer/context)는 gold 전제라 채우지 않는다(빈 값 = 스킵)."""
    records: list[EvalRecord] = []
    for i, log in enumerate(logs):
        gt = str(log.raw.get("ground_truth") or "").strip() or None
        probe = Probe(
            probe_id=f"ext_{i:04d}",
            question=log.question,
            source="user_log",
            ground_truth=gt,
        )
        rec = EvalRecord(
            probe=probe,
            retrieved_context=[c["text"] for c in log.contexts],
            retrieved_chunk_ids=[
                c["chunk_id"] or f"ext_{i:04d}_ctx_{j}"
                for j, c in enumerate(log.contexts)
            ],
            generated_answer=log.answer,
            recall_at_k=-1.0,
        )
        if gt:
            rec.f1_score = char_f1(log.answer, gt)
        records.append(rec)
    return records


# ── 리플레이 실행 ────────────────────────────────────────────────

def run_replay(records: list[EvalRecord], *, iteration: int = 1) -> DiagnosticReport:
    """RAGAS(가능하면)를 채우고 기존 build_report로 리포트를 만든다.

    STEP4(원인 진단)는 부르지 않는다 - 현행 라벨은 gold 전제라 전부 침묵하고
    (docs §4), findings 없는 record는 report 규약상 "정상"으로 집계되기 때문에
    빈 호출만 된다. LLM 비활성/실패는 기존 폴백 규약대로 {}로 넘어간다."""
    judge = _judge() if llm_eval_enabled() else None
    for rec in records:
        if judge is not None and not rec.ragas_done:
            try:
                rec.ragas = evaluate_real_track(rec, judge)
            except Exception as exc:
                print(f"[Replay] RAGAS 실패({exc}) -> 폴백")
                rec.ragas = {}
            rec.ragas_done = True
        faith = rec.ragas.get("faithfulness")
        if faith is not None:
            rec.retrieval_axis = float(faith)
    return build_report(records, iteration=iteration)


def diagnose_external_log(path: str, *, limit: int | None = None
                          ) -> tuple[DiagnosticReport | None, dict, list[str]]:
    """파일 경로 → (리포트, 적재 판정, 적재 오류). 리포트는 유효 레코드가 없으면 None."""
    logs, errors = load_external_log(path)
    cap = assess_capability(logs)
    if cap["tier"] == TIER_NONE:
        return None, cap, errors
    if limit is not None:
        logs = logs[:limit]
    report = run_replay(build_replay_records(logs))
    return report, cap, errors


# ── CLI ──────────────────────────────────────────────────────────

def _fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "미측정"


def _main(argv: list[str]) -> int:
    # graph.py와 같은 규약: CLI로 직접 부를 때만 .env를 읽는다(라이브러리 사용 시엔
    # 호출자 환경을 존중). 미설치면 조용히 넘어간다 - 키 없이도 규칙 지표는 돈다.
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass

    args = [a for a in argv if not a.startswith("--")]
    limit = None
    for a in argv:
        if a.startswith("--limit="):
            limit = int(a.split("=", 1)[1])
    if len(args) != 1:
        print("사용법: python -m agents.eval.replay <log.jsonl> [--limit=N]")
        return 2

    report, cap, errors = diagnose_external_log(args[0], limit=limit)
    print(f"적재: 정상 {cap['records']}건 / 오류 {len(errors)}건 / 진단 수준 {cap['tier']}")
    if report is None:
        print("유효 레코드가 없어 진단 불가")
        return 1

    scores = report.ragas_scores or {}
    print("- faithfulness(환각 없음)    :", _fmt(scores.get("faithfulness")))
    print("- answer relevancy(동문서답 없음):", _fmt(scores.get("response_relevancy")))
    print("- context precision(검색 정밀):", _fmt(scores.get("context_precision")), "(정답 텍스트 필요)")
    print("- context recall(검색 재현)  :", _fmt(scores.get("context_recall")), "(정답 텍스트 필요)")
    print("- 규칙 char F1               :", _fmt(scores.get("mean_f1")), "(정답 텍스트 필요)")
    print(f"overall_score: {_fmt(report.overall_score)}")
    print(f"종합점수: {format_composite(report.composite_score)}")
    if not llm_eval_enabled():
        print("(참고: EVAL_ENABLE_LLM=1 이 아니어서 RAGAS 지표는 미측정)")
        if scores.get("mean_f1") is not None:
            print("(주의: 정답 텍스트는 있는데 faithfulness 가 미측정이라 신뢰도의 "
                  "검색축이 0으로 보수 집계된다 - 종합점수는 참고만)")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
