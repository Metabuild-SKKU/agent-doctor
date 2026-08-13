"""
tools/run_replay_report.py
외부 RAG 실행 로그 하나로 리플레이 진단을 돌리고, 결과를 진단서 HTML 로 떨군다.

다른 팀의 RAG를 우리가 손대지 않고 진단하는 정식 진입점이다(README §"다른 팀의
RAG를 로그로 진단하기"). tests/run_corpus.py 의 외부 모드 짝이다 — 그쪽은
코퍼스로 우리 파이프라인(Ingest→Index→Eval→Optimize)을 돌리지만, 여기서는
남의 RAG 가 남긴 로그를 리플레이해 진단만 한다 — 남의 인덱스라 Optimize 를
돌 수 없기 때문이다(그래서 graph.py 의 LangGraph 루프에는 물리지 않는다).

서버를 띄우지 않는다. report.html 이 fetch 로 받아가던 JSON 을
report_view.build_ext_report_view() 로 직접 만들어 HTML 안에 심으므로,
브라우저에서 결과 파일을 그냥 열면 된다.

Run:
    python tools/run_replay_report.py                       # 기본 fixture(결함 로그)
    python tools/run_replay_report.py <log.jsonl>
    python tools/run_replay_report.py <log.jsonl> --no-open
    python tools/run_replay_report.py <log.jsonl> --allow-qa-only

RAGAS 지표(faithfulness/relevancy)를 재려면 EVAL_ENABLE_LLM=1 + EVAL_MODE>=deep
이 필요하다. 미설정이면 이 스크립트가 켜준다 — 지표가 없으면 소견이 0건이라
진단서에 볼 것이 없기 때문. LLM 을 실제로 호출하므로 비용이 든다.
"""
from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_LOG = REPO_ROOT / "tests" / "fixtures" / "external_rag" / "ext_hallucinate.jsonl"
OUT_DIR = REPO_ROOT / "tests" / "fixtures" / "external_rag" / "out"


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n] + "…"


def _log_questions(report) -> None:
    """질문별 Q/기대답/실제답/진단을 남긴다.

    파이프라인 로그(Eval STEP4)가 probe 마다 Q/A/R 을 찍는 것과 같은 자리다.
    점수만 남기면 나중에 "왜 저 점수였나"를 되짚을 수 없고, 대조군 로그가
    이름값을 하는지도 눈으로 확인할 수 없다."""
    rows = getattr(report, "failed_questions", []) or []
    if not rows:
        print("\n─── 질문별 결과 ─── 실패한 질문 없음")
        return
    by_probe: dict[str, list] = {}
    for f in report.findings or []:
        for pid in f.affected_probes:
            by_probe.setdefault(pid, []).append(f)

    print(f"\n─── 질문별 결과 (실패 {len(rows)}건) ───────────────────────")
    for i, row in enumerate(rows, 1):
        pid = row.get("probe_id", "")
        found = by_probe.get(pid, [])
        marks = " · ".join(
            f"{f.label}{'' if f.confirmed else '(예비)'}" for f in found) or "라벨 없음"
        print(f"\n  [{i}/{len(rows)}] {pid}  ❌ {marks}")
        print(f"    Q: {_clip(row.get('question'), 90)}")
        if row.get("expected_answer"):
            print(f"    기대: {_clip(row.get('expected_answer'), 70)}")
        print(f"    실제: {_clip(row.get('actual_answer'), 110)}")
        for f in found:
            print(f"    → {_clip(f.metadata.get('reason') or f.description, 100)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="외부 RAG 로그 → 진단서 HTML")
    parser.add_argument("log", nargs="?", default=str(DEFAULT_LOG),
                        help=f"외부 로그 JSONL (기본: {DEFAULT_LOG.name})")
    parser.add_argument("--no-open", action="store_true", help="브라우저를 띄우지 않는다")
    parser.add_argument("--allow-qa-only", action="store_true",
                        help="컨텍스트 없는 로그도 진단(동문서답 검사만)")
    parser.add_argument("--limit", type=int, default=None, help="레코드 수 제한")
    # fixture 로그는 골든셋이 이미 병합돼 있어(ground_truth/gold_contexts 인라인)
    # 기본값이 없다. 골든셋이 분리된 로그를 볼 때 이 인자를 쓴다.
    parser.add_argument("--golden", default=None,
                        help="골든셋 파일(xlsx/csv/jsonl/json) — 질문 매칭으로 병합")
    args = parser.parse_args(argv)

    from core.console import force_utf8_stdio
    force_utf8_stdio()
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass

    # 지표가 없으면 소견이 0건이라 진단서에 볼 것이 없다. .env 에 EVAL_ENABLE_LLM=0
    # 이 있어도 켠다 - setdefault 였을 때는 같은 로그가 CLI(agents.eval.replay)·웹과
    # 다른 결과를 냈다. "진입점마다 결과가 달라지지 않게" 가 이 셋의 공통 규약이다.
    # 얕게 보고 싶으면 진단서가 아니라 CLI 의 --no-llm 을 쓰면 된다.
    os.environ["EVAL_ENABLE_LLM"] = "1"
    if os.getenv("EVAL_MODE", "").strip().lower() not in ("deep", "full", "3", "4"):
        os.environ["EVAL_MODE"] = "deep"

    # 로깅은 파이프라인 import 보다 먼저 설치한다(run_corpus.py 와 같은 규약) —
    # 모델 로딩 경고처럼 import 시점에 나오는 출력까지 output/logs 에 담기게.
    from core.run_logger import setup_run_logging
    run_log = setup_run_logging(prefix="replay")

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"로그 파일이 없습니다: {log_path}")
        print(f"  뽑아둔 로그: {DEFAULT_LOG.parent}")
        print("  새로 뽑기:   python -m tools.make_external_rag --defect=hallucinate")
        return 2

    from core.llm_usage import step
    from agents.eval.log_intake import load_external_log
    from agents.eval.replay import diagnose_external_log
    from agents.serve.report_view import build_ext_report_view

    from tools.report_html import write_report_files

    # 파이프라인 로그와 같은 STEP 구획을 쓴다(core.llm_usage.step) — 구간별
    # LLM 호출수·토큰·비용·소요가 자동으로 붙어, 나중에 "왜 저 점수였나"를
    # 되짚을 때 재료가 남는다.
    with step("Replay", 1, f"로그 적재 — {log_path.name}"):
        raw, load_errors = load_external_log(str(log_path))
        print(f"  {len(raw)}건 적재 · 오류 {len(load_errors)}건")
        for err in load_errors[:5]:
            print(f"    ! {err}")
        if raw:
            n_ctx = sum(1 for r in raw if r.contexts)
            n_gt = sum(1 for r in raw if r.ground_truth)
            n_gold = sum(1 for r in raw if r.gold_contexts)
            print(f"  컨텍스트 {n_ctx}건 · 정답 텍스트 {n_gt}건 · 정답 근거 {n_gold}건")
            if raw[0].config:
                print(f"  상대 설정: {raw[0].config}")

    with step("Replay", 2, "지표 실측 + 원인 판정"):
        report, cap, errors = diagnose_external_log(
            str(log_path), limit=args.limit, allow_qa_only=args.allow_qa_only,
            golden_path=args.golden, logs=raw, errors=load_errors)
        g = cap.get("golden")
        if g:
            print(f"  골든셋 병합: {g['qa_entries']}건 중 {g['matched']}건 매칭 "
                  f"· 정답 {g['filled_ground_truth']}건 · 근거문단 {g['filled_gold_contexts']}건")
            if g["qa_entries"] and not g["matched"]:
                print("  ! 한 건도 매칭되지 않았습니다 — 질문 표기를 확인하세요")
        print(f"  진단 수준 {cap['tier']}")
        for note in cap.get("notes", []):
            print(f"    · {note}")
    if report is None:
        print("진단 불가 — 컨텍스트가 없거나(--allow-qa-only) 유효 레코드가 없습니다.")
        return 1

    _log_questions(report)

    with step("Replay", 3, "진단서 조립"):
        view = build_ext_report_view(report, cap)
        html_path, _ = write_report_files(view, OUT_DIR)
        print(f"  섹션 {len([s for s in view['mode']['hidden_sections']])}개 숨김 "
              f"· 권고 카드 {len(view['recommendations'])}장")

    score = view["score"]
    print("\n" + "=" * 56)
    print(f"  결과 — {log_path.name}")
    print("=" * 56)
    # "레코드/소견 6 / 4" 는 비율처럼 읽히는데 서로 무관한 두 값이다. 줄을 나누고
    # 단위를 붙인다. 소견 수와 카드 수가 다른 것도(라벨 단위로 묶으므로) 함께 밝힌다.
    n_find = score["findings_count"]
    n_card = len(view["recommendations"])
    # records 는 적재 건수라 --limit 이 걸리면 "진단한" 건수와 어긋난다.
    n_diag = cap.get("diagnosed", cap["records"])
    print(f"진단한 질문   : {n_diag}건"
          + (f" (적재 {cap['records']}건 중 --limit 적용)" if n_diag != cap["records"] else "")
          + f"  (진단 수준: {view['meta']['tier']})")
    print(f"발견한 문제   : {n_find}건"
          + (f" → 원인 {n_card}종으로 묶임" if n_card and n_card != n_find else ""))
    print(f"종합 점수     : {score['after']}점 / 100  "
          f"(기준선 {'통과' if score['gate'].get('pass') else '미달'})")
    # 어떤 지표가 점수를 끌어내렸는지 로그만 보고 알 수 있어야 한다.
    if view["metrics"]:
        print("품질 지표     :")
        for m in view["metrics"]:
            print(f"  · {m['name']:<8} {m['after']:.3f}  ({m['en']})")
    if n_card:
        print("처방 추천     :")
        for rec in view["recommendations"]:
            print(f"  · [{rec['badge'][1]}] {rec['title']} ({rec['cta']})")
    print(f"\n진단서: {html_path}")
    if run_log:
        print(f"실행 로그: {run_log}")

    if not args.no_open:
        webbrowser.open(html_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
