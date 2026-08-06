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
    TIER_NONE, TIER_QA_ONLY, ExternalLogRecord, assess_capability, load_external_log,
)
from agents.eval.metrics_basic import char_f1
from agents.eval.metrics_ragas import _judge, evaluate_real_track
from agents.eval.replay_labels import apply_ext_labels, recommendation_ids
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
        gt = log.ground_truth
        probe = Probe(
            probe_id=f"ext_{i:04d}",
            question=log.question,
            source="user_log",
            ground_truth=gt,
            # gold 근거 문단 텍스트는 공유 스키마를 바꾸지 않고 metadata 로 전달 -
            # replay_labels.gold_context_recall 이 여기서 읽는다.
            metadata={"gold_contexts": log.gold_contexts} if log.gold_contexts else {},
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
    """RAGAS(가능하면) → ext_ 소견(replay_labels) → 기존 build_report.

    STEP4(내부 원인 진단)는 부르지 않는다 - 현행 30라벨은 gold 전제라 전부
    침묵한다(docs §4). 대신 리플레이 전용 ext_ 라벨(docs §5)이 record.findings
    를 채우고, build_report 는 라벨 이름을 해석하지 않으므로 무변경 합류.
    LLM 비활성/실패는 기존 폴백 규약대로 {}로 넘어간다(→ ext_ 라벨도 침묵)."""
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
    apply_ext_labels(records)
    return build_report(records, iteration=iteration)


def diagnose_external_log(path: str, *, limit: int | None = None,
                          allow_qa_only: bool = False,
                          ) -> tuple[DiagnosticReport | None, dict, list[str]]:
    """파일 경로 → (리포트, 적재 판정, 적재 오류). 리포트가 None 인 경우:
    유효 레코드 없음(tier none), 또는 컨텍스트 부족(tier qa_only)인데
    allow_qa_only=False - 파일 단위 게이트(docs §3). 줄 단위 결손은 관용하되
    파일 전체가 contexts 없는 로그에 "진단"을 내주지 않는다."""
    logs, errors = load_external_log(path)
    cap = assess_capability(logs)
    if cap["tier"] == TIER_NONE:
        return None, cap, errors
    if cap["tier"] == TIER_QA_ONLY and not allow_qa_only:
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
    from core.console import force_utf8_stdio
    force_utf8_stdio()
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
    allow_qa_only = "--allow-qa-only" in argv
    if len(args) != 1:
        print("사용법: python -m agents.eval.replay <log.jsonl> [--limit=N] [--allow-qa-only]")
        return 2

    report, cap, errors = diagnose_external_log(
        args[0], limit=limit, allow_qa_only=allow_qa_only)
    print(f"적재: 정상 {cap['records']}건 / 오류 {len(errors)}건 / 진단 수준 {cap['tier']}")
    if report is None:
        if cap["tier"] == TIER_QA_ONLY:
            print("검색 컨텍스트(contexts)가 없거나 부족해 진단이 성립하지 않습니다.")
            print("  - 답변 생성에 쓰인 검색 결과 원문을 로그에 추가해 주세요"
                  " (환각 검출·검색/생성 원인 분리가 가능해집니다)")
            print("  - 동문서답 검사만이라도 하려면: --allow-qa-only")
        else:
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

    if report.findings:
        # 같은 라벨의 probe 별 소견을 확정/예비로 갈라 요약. 권고는 rules 처방 재참조.
        by_label: dict = {}
        for f in report.findings:
            groups = by_label.setdefault(f.label, {"확정": [], "예비": []})
            groups["확정" if f.confirmed else "예비"].append(f)
        print("소견:")
        for label, groups in by_label.items():
            for grade in ("확정", "예비"):
                items = groups[grade]
                if not items:
                    continue
                print(f"  [{items[0].severity}] {label} ({grade} {len(items)}건)")
                print(f"      {items[0].metadata.get('reason', '')}")
            recs = recommendation_ids(label)
            if recs:
                print(f"      권고: {', '.join(recs)}")
    else:
        print("소견: 없음 (지표 미측정이거나 문턱 이상)")
    if not llm_eval_enabled():
        print("(참고: EVAL_ENABLE_LLM=1 이 아니어서 RAGAS 지표는 미측정)")
        if scores.get("mean_f1") is not None:
            print("(주의: 정답 텍스트는 있는데 faithfulness 가 미측정이라 신뢰도의 "
                  "검색축이 0으로 보수 집계된다 - 종합점수는 참고만)")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
