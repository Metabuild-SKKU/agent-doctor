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
- 검색축 신뢰도 seam(retrieval_axis)은 gold_contexts 겹침(gold_context_recall,
  결정적·LLM 불필요)을 우선으로, 없으면 faithfulness를 채운다. 둘 다 없으면
  None으로 남기고 reliability_score(scoring.py)가 그 레코드를 평균에서
  제외한다(recall 센티널을 0점으로 클램프해 신뢰도를 오염시키던 문제 수정).

CLI: python -m agents.eval.replay <log.jsonl> --golden=<golden.xlsx> [--limit N]
     골든셋(시험지)은 진단의 기본 입력이다 - 없으면 검색축 라벨 3종이 침묵하고
     환각도 예비에 머문다(7종 중 3종만). 그래서 CLI 는 기본적으로 거부하고,
     정답지가 아예 없는 상황만 --no-golden 으로 옵트인시킨다.
     RAGAS(LLM 심층)는 기본 켜짐이다 - 리플레이는 색인·검색·답변생성이 없어 RAGAS 가
     유일한 작업이라 꺼도 시간을 못 아끼면서(실측 6건: 0.03초 vs 13.8초) 생성축 라벨
     4종을 통째로 잃는다. 적재만 빠르게 확인할 때는 --no-llm 으로 끈다.
     (LLM 키가 없으면 켜져 있어도 규칙 지표로 폴백한다.)
"""

from __future__ import annotations

import os
import sys

from core.schema import DiagnosticReport, Probe

from agents.eval.log_intake import (
    TIER_NONE, TIER_QA_ONLY, ExternalLogRecord, assess_capability, load_external_log,
)
from agents.eval.metrics_basic import answer_match
from agents.eval.metrics_ragas import _judge, evaluate_real_track
from agents.eval.replay_labels import apply_ext_labels, gold_context_recall
from agents.eval.report import build_report
from agents.eval.types import EvalRecord, llm_eval_enabled, resolve_llm_concurrency
from core.parallel import parallel_map
from agents.eval.scoring import format_composite
# 권고 카드는 Optimize 소관 — 라벨→처방 지식은 rules.py 를 아는 쪽이 갖는다
# (reporter.py 가 내부 모드에서 하는 "결정 → 사람이 읽는 번역"의 외부 모드 짝).
from agents.optimize.ext_advisor import build_ext_recommendations


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
            # 내부 모드(_compute_metrics)와 같은 지표: raw char_f1 이 아니라 answer_match
            # (짧은 정답 containment·서술형 창 F1 보정). 같은 필드(f1_score)·같은 문턱(0.5)을
            # 두 모드가 다른 지표로 채우면 외부 RAG 가 체계적으로 저평가된다.
            score = answer_match(log.answer, gt)
            rec.f1_score = score
            rec.raw_f1_score = score
        records.append(rec)
    return records


# ── 리플레이 실행 ────────────────────────────────────────────────

def run_replay(records: list[EvalRecord], *, iteration: int = 1) -> DiagnosticReport:
    """RAGAS(가능하면) → ext_ 소견(replay_labels) → 기존 build_report.

    STEP4(내부 원인 진단)는 부르지 않는다 - 현행 30라벨은 gold 전제라 전부
    침묵한다(docs §4). 대신 리플레이 전용 ext_ 라벨(docs §5)이 record.findings
    를 채우고, build_report 는 라벨 이름을 해석하지 않으므로 무변경 합류.
    LLM 비활성/실패는 기존 폴백 규약대로 {}로 넘어간다(→ ext_ 라벨도 침묵).

    RAGAS 채점은 레코드 단위 병렬(EVAL_LLM_CONCURRENCY) - 일반 모드(agent.py
    STEP3)와 같은 팬아웃. 리플레이는 검색이 없어(로그가 이미 컨텍스트를 갖고 옴)
    signals 전역·진단 캐시 같은, 일반 모드가 병렬 구간을 좁힌 이유가 없다."""
    judge = _judge() if llm_eval_enabled() else None
    pending = [rec for rec in records if not rec.ragas_done] if judge is not None else []
    if pending:
        def _score(rec: EvalRecord) -> dict:
            try:
                return evaluate_real_track(rec, judge)
            except Exception as exc:
                print(f"[Replay] RAGAS 실패({exc}) -> 폴백")
                return {}
        concurrency = resolve_llm_concurrency()
        print(f"  probe {len(pending)}개 RAGAS 채점"
              f"{f' 병렬 (동시성 {concurrency})' if concurrency > 1 and len(pending) > 1 else ' 순차'}")
        scores = parallel_map(_score, pending, concurrency)
        for rec, score in zip(pending, scores):
            rec.ragas, rec.ragas_done = score, True

    for rec in records:
        # 검색축 신뢰도: gold 겹침(결정적, LLM 불필요)을 우선한다 - "질문의 근거를
        # 가져왔나"에 faithfulness("가져온 걸 충실히 썼나")보다 직접 답한다.
        # GT 있고 LLM 없는 실행에서도 이 경로로 실측된다.
        overlap = gold_context_recall(rec)
        faith = rec.ragas.get("faithfulness")
        if overlap is not None:
            rec.retrieval_axis = overlap
        elif faith is not None:
            rec.retrieval_axis = float(faith)
    apply_ext_labels(records)
    return build_report(records, iteration=iteration)


def apply_golden_set(logs: list, qa_path: str, *, qa: tuple[dict, list[str]] | None = None) -> dict:
    """골든셋(시험지)을 로그(답안지)에 질문 매칭으로 병합한다. 반환: 집계 dict.

    qa_merge 의 파일 경로 버전(merge_qa_into_log)과 같은 규칙을 메모리에서 적용한다 -
    진단 한 번에 중간 파일(.merged.jsonl)을 만들지 않기 위해서다. 로더·정규화·키
    관용 규칙은 qa_merge 가 단독으로 갖는다(여기서 재구현하지 않는다).

    채우기만 한다: 로그에 이미 값이 있으면 덮지 않고 충돌로 집계한다 - 로그 제공자가
    직접 넣은 값을 신뢰한다(qa_merge 와 동일 규칙 - apply_qa_entry 를 공유한다).

    qa 를 미리 넘기면(호출자가 이미 load_qa_set 을 불렀으면) 다시 읽지 않는다 -
    CLI 프리플라이트 안내 출력(_main)이 같은 골든셋 파일을 두 번 파싱하지 않도록."""
    from agents.eval.qa_merge import apply_qa_entry, load_qa_set, normalize_question

    qa_map, qa_errors = qa if qa is not None else load_qa_set(qa_path)
    stats = {"qa_entries": len(qa_map), "matched": 0, "filled_ground_truth": 0,
             "filled_gold_contexts": 0, "conflicts": 0, "qa_errors": qa_errors}
    for rec in logs:
        entry = qa_map.get(normalize_question(rec.question))
        if not entry:
            continue
        stats["matched"] += 1
        apply_qa_entry(
            entry,
            get_ground_truth=lambda r=rec: r.ground_truth,
            set_ground_truth=lambda v, r=rec: setattr(r, "ground_truth", v),
            get_gold_contexts=lambda r=rec: r.gold_contexts,
            set_gold_contexts=lambda v, r=rec: setattr(r, "gold_contexts", v),
            stats=stats,
        )
    return stats


def diagnose_external_log(path: str, *, limit: int | None = None,
                          allow_qa_only: bool = False,
                          golden_path: str | None = None,
                          logs: list[ExternalLogRecord] | None = None,
                          errors: list[str] | None = None,
                          qa: tuple[dict, list[str]] | None = None,
                          ) -> tuple[DiagnosticReport | None, dict, list[str]]:
    """파일 경로 → (리포트, 적재 판정, 적재 오류). 리포트가 None 인 경우:
    유효 레코드 없음(tier none), 또는 컨텍스트 부족(tier qa_only)인데
    allow_qa_only=False - 파일 단위 게이트(docs §3). 줄 단위 결손은 관용하되
    파일 전체가 contexts 없는 로그에 "진단"을 내주지 않는다.

    golden_path 가 주어지면 적재 직후·판정 전에 병합한다 - 순서가 중요하다.
    assess_capability 가 정답/근거 보유량을 세어 tier·notes 를 만들므로, 병합을
    나중에 하면 "정답 0건"으로 판정된 채 리포트가 나간다.

    logs/errors 를 미리 넘기면(호출자가 이미 load_external_log 를 불렀으면) 파일을
    다시 읽지 않는다 - CLI 프리플라이트 골든셋 게이트 체크(_main)와
    tools/run_replay_report.py 의 STEP1 통계 출력이 같은 로그를 두 번 파싱하지
    않도록. qa 는 apply_golden_set 에 그대로 전달한다(같은 취지)."""
    if logs is None:
        logs, errors = load_external_log(path)
    else:
        errors = list(errors or [])
    if golden_path:
        merge_stats = apply_golden_set(logs, golden_path, qa=qa)
        errors = errors + [f"골든셋 파싱 오류: {e}" for e in merge_stats["qa_errors"]]
    else:
        merge_stats = None
    cap = assess_capability(logs)
    cap["golden"] = merge_stats
    if cap["tier"] == TIER_NONE:
        return None, cap, errors
    if cap["tier"] == TIER_QA_ONLY and not allow_qa_only:
        return None, cap, errors
    if limit is not None:
        logs = logs[:limit]
    # tier 판정은 파일 전체(위)에서 하고, 표시용 "질문 수"는 실제로 진단한 건수를 쓴다 -
    # cap["records"] 는 적재 건수라 --limit 이 걸리면 "10건만 진단했는데 500건"으로
    # 읽힌다. 게이트 의미(파일 단위)를 바꾸지 않으려고 필드를 나눈다.
    cap["diagnosed"] = len(logs)
    report = run_replay(build_replay_records(logs))
    return report, cap, errors


# ── CLI ──────────────────────────────────────────────────────────

def _fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "미측정"


def _print_golden_summary(cap: dict) -> None:
    """골든셋 병합 결과. 매칭이 0건이면 조용히 넘어가면 안 된다 —
    질문 표기가 달라 못 붙었는데 '골든셋 줬으니 됐다'고 오해하게 된다."""
    g = cap.get("golden")
    if g is None:
        print("골든셋: 없음 (--no-golden) — 검색축 라벨 3종 침묵, 환각은 예비 고정")
        return
    print(f"골든셋: {g['qa_entries']}건 중 {g['matched']}건 매칭 "
          f"· 정답 {g['filled_ground_truth']}건 · 근거문단 {g['filled_gold_contexts']}건"
          + (f" · 충돌 {g['conflicts']}건" if g["conflicts"] else ""))
    if g["qa_entries"] and not g["matched"]:
        print("  ! 한 건도 매칭되지 않았습니다 — 골든셋 질문과 로그 질문의 표기를 확인하세요"
              " (공백·문장부호·대소문자는 자동 정규화됩니다).")
    elif g["matched"] < g["qa_entries"]:
        print(f"  · 골든셋 {g['qa_entries'] - g['matched']}건은 로그에서 해당 질문을 찾지 못했습니다.")


# 골든셋은 사람이 ground_truth/gold_contexts 를 직접 채워야 하는 자료라(docs §3),
# 커질수록 상대 팀 부담과 리플레이 LLM 비용(매칭된 레코드마다 context precision/
# recall/correctness 를 추가로 재는 비용)이 함께 는다. 리플레이는 STEP1 probe 합성·
# STEP2 답변 생성·오라클 트랙·내부 라벨 LLM 호출이 전부 빠져 일반 모드보다 훨씬
# 싸지만(레코드당 RAGAS 실제 트랙 chat 1 + embed 1 뿐), 그래도 무제한은 아니다.
GOLDEN_RECOMMENDED_MAX = 150
GOLDEN_HARD_CAP = 300


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
    golden = None
    for a in argv:
        if a.startswith("--limit="):
            limit = int(a.split("=", 1)[1])
        elif a.startswith("--golden="):
            golden = a.split("=", 1)[1]
    allow_qa_only = "--allow-qa-only" in argv
    if len(args) != 1:
        print("사용법: python -m agents.eval.replay <log.jsonl> --golden=<golden.xlsx>"
              " [--limit=N] [--allow-qa-only] [--no-golden] [--no-llm]")
        return 2

    # 리플레이는 RAGAS 채점이 유일한 작업이라(색인·검색·답변생성이 없다) LLM 을 꺼도
    # 아끼는 시간이 거의 없는 반면(실측 6건: 0.03초 vs 13.8초), 끄면 생성축 라벨 4종이
    # 통째로 죽고 검색축도 gold 겹침이 낮을 때만 남는다. 그래서 기본을 켬으로 둔다
    # (웹 UI·tools/run_replay_report.py 와 같은 판단 — 진입점마다 결과가 달라지지 않게).
    # 다만 CLI 는 개발자 도구라 "적재만 빨리 확인" 같은 정당한 용도가 있어 옵트아웃을 남긴다.
    if "--no-llm" not in argv:
        os.environ["EVAL_ENABLE_LLM"] = "1"
        if os.getenv("EVAL_MODE", "").strip().lower() not in ("deep", "full", "3", "4"):
            os.environ["EVAL_MODE"] = "deep"

    # 골든셋은 진단의 기본 입력이다(멘토 피드백 2026-08-07 §1). 빼먹으면 검색축
    # 라벨 3종이 침묵하고 환각도 예비에 머무는데, 그 사실을 모른 채 얕은 진단을
    # 받아가는 것을 막는다. 정답지가 아예 없는 상황은 --no-golden 으로 옵트인한다
    # (contexts 파일 게이트와 같은 구조 - 기본은 거부, 명시하면 허용).
    # 단, 로그 자체에 이미 ground_truth/gold_contexts 가 인라인으로 있으면(문서·
    # 시뮬레이터가 안내하는 --golden 없는 명령이 이 경우다) 골든셋이 없는 게
    # 아니므로 거부하지 않는다.
    # 아래서 미리 읽은 로그/골든셋은 diagnose_external_log 에 그대로 넘겨 같은 파일을
    # 두 번 파싱하지 않는다(logs=/qa= 파라미터 - 대용량 로그·골든셋에서 I/O 가 겹친다).
    preloaded_logs = preloaded_errors = preloaded_qa = None

    if not golden and "--no-golden" not in argv:
        preloaded_logs, preloaded_errors = load_external_log(args[0])
        inline_cap = assess_capability(preloaded_logs)
        if not inline_cap["with_ground_truth"] and not inline_cap["with_gold_contexts"]:
            print("골든셋(시험지)이 없습니다 — 진단의 기본 입력입니다.")
            print("  --golden=<golden.xlsx|csv|jsonl|json> 로 질문·정답 세트를 주세요.")
            print("  · 필수 열: question, ground_truth   · 권장: gold_contexts(정답 근거 원문)")
            print("  · 이미 쓰는 테스트셋이 있으면 그대로 주셔도 됩니다(열 이름 관용 인식)")
            print("  골든셋 없이 돌리려면: --no-golden")
            print("    → 검색축 라벨 3종이 침묵하고 환각은 '예비'에 머뭅니다(7종 중 3종만).")
            return 2

    if golden:
        from agents.eval.qa_merge import load_qa_set
        qa_map, qa_errors = load_qa_set(golden)
        preloaded_qa = (qa_map, qa_errors)
        n = len(qa_map)
        print(f"골든셋 권장 크기: 50~{GOLDEN_RECOMMENDED_MAX}건 (현재 {n}건)")
        if n > GOLDEN_HARD_CAP:
            print(f"골든셋이 {GOLDEN_HARD_CAP}건을 넘습니다 — 매칭된 레코드마다 검색축 지표"
                  "(context precision/recall/correctness)를 추가로 측정해 비용이 함께 늡니다.")
            print(f"  {GOLDEN_HARD_CAP}건 이하로 나눠서 실행해 주세요"
                  " (--limit 은 로그 표본만 줄일 뿐 골든셋 크기는 줄이지 못합니다).")
            return 2

    report, cap, errors = diagnose_external_log(
        args[0], limit=limit, allow_qa_only=allow_qa_only, golden_path=golden,
        logs=preloaded_logs, errors=preloaded_errors, qa=preloaded_qa)
    # "적재" 는 파일에서 읽은 건수라 cap['records'] 가 맞다. --limit 으로 그중 일부만
    # 진단했으면 그 사실을 함께 밝힌다(리포트의 질문 수는 diagnosed 를 쓴다).
    diagnosed = cap.get("diagnosed")
    limited = diagnosed is not None and diagnosed != cap["records"]
    print(f"적재: 정상 {cap['records']}건 / 오류 {len(errors)}건 / 진단 수준 {cap['tier']}"
          + (f" · --limit 으로 {diagnosed}건만 진단" if limited else ""))
    for err in errors[:10]:
        print(f"  ! {err}")
    if len(errors) > 10:
        print(f"  ! ... 외 {len(errors) - 10}건")
    _print_golden_summary(cap)
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
        # 같은 라벨의 probe 별 소견을 확정/예비로 갈라 요약.
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

        # 권고 카드는 Optimize 소관(ext_advisor) — 라벨→처방 지식은 rules.py 를
        # 아는 쪽이 갖는다. 여기서는 만들어진 재료를 출력만 한다.
        cards = build_ext_recommendations(report.findings, cap.get("config"))
        if cards:
            print("권고:")
            for c in cards:
                grade = f"확정 {c['confirmed']}건" if c["confirmed"] else ""
                if c["tentative"]:
                    grade = f"{grade} · 예비 {c['tentative']}건" if grade else f"예비 {c['tentative']}건"
                print(f"  [{c['severity']}] {c['summary']} ({grade})")
                for s in c["steps"]:
                    cur = f" — {s['current']}" if s["current"] else ""
                    tags = "".join(t for t in (
                        " [재색인 필요]" if s["needs_reindex"] else "",
                        " [수동 조치]" if s["manual"] else "") if t)
                    print(f"    · {s['action']}{cur}{tags}")
                    if s["detail"]:
                        print(f"        {s['detail']}")
    else:
        print("소견: 없음 (지표 미측정이거나 문턱 이상)")
    if not llm_eval_enabled():
        print("(참고: --no-llm 으로 RAGAS 지표가 미측정입니다 - 생성축 라벨 4종"
              "(환각·동문서답·컨텍스트 과다·근거 부족)이 침묵하고, 검색축도 gold 근거"
              " 겹침이 낮을 때만 잡힙니다. 온전한 진단은 --no-llm 없이 실행하세요.)")
        if scores.get("mean_f1") is not None and not cap.get("with_gold_contexts"):
            print("(주의: 정답 텍스트는 있는데 근거문단(gold_contexts)이 없어 검색축이 "
                  "미측정이다 - 해당 probe는 신뢰도 평균에서 제외된다. 근거문단을 "
                  "골든셋에 추가하면 LLM 없이도 검색축이 실측된다)")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
