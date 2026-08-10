"""
tests/run_replay_report.py
외부 RAG 실행 로그 하나로 리플레이 진단을 돌리고, 결과를 진단서 HTML 로 떨군다.

run_corpus.py 의 외부 모드 짝이다. 그쪽은 코퍼스로 우리 파이프라인
(Ingest→Index→Eval→Optimize)을 돌리지만, 여기서는 남의 RAG 가 남긴 로그를
리플레이해 진단만 한다 — 남의 인덱스라 Optimize 를 돌 수 없기 때문이다.

서버를 띄우지 않는다. report.html 이 fetch 로 받아가던 JSON 을
report_view.build_ext_report_view() 로 직접 만들어 HTML 안에 심으므로,
브라우저에서 결과 파일을 그냥 열면 된다.

Run:
    python tests/run_replay_report.py                       # 기본 fixture(결함 로그)
    python tests/run_replay_report.py <log.jsonl>
    python tests/run_replay_report.py <log.jsonl> --no-open
    python tests/run_replay_report.py <log.jsonl> --allow-qa-only

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="외부 RAG 로그 → 진단서 HTML")
    parser.add_argument("log", nargs="?", default=str(DEFAULT_LOG),
                        help=f"외부 로그 JSONL (기본: {DEFAULT_LOG.name})")
    parser.add_argument("--no-open", action="store_true", help="브라우저를 띄우지 않는다")
    parser.add_argument("--allow-qa-only", action="store_true",
                        help="컨텍스트 없는 로그도 진단(동문서답 검사만)")
    parser.add_argument("--limit", type=int, default=None, help="레코드 수 제한")
    args = parser.parse_args(argv)

    from core.console import force_utf8_stdio
    force_utf8_stdio()
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass

    # 지표가 없으면 소견이 0건이라 진단서에 볼 것이 없다. 사용자가 명시적으로
    # 꺼둔 경우는 존중하고, 미설정일 때만 켠다.
    os.environ.setdefault("EVAL_ENABLE_LLM", "1")
    if os.getenv("EVAL_MODE", "").strip().lower() not in ("deep", "full", "3", "4"):
        os.environ["EVAL_MODE"] = "deep"

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"로그 파일이 없습니다: {log_path}")
        print(f"  뽑아둔 로그: {DEFAULT_LOG.parent}")
        print("  새로 뽑기:   python -m tools.make_external_rag --defect=hallucinate")
        return 2

    from agents.eval.replay import diagnose_external_log
    from agents.serve.report_view import build_ext_report_view

    from tests.report_html import write_report_files

    print(f"[1/3] 로그 적재·진단 — {log_path.name}")
    report, cap, errors = diagnose_external_log(
        str(log_path), limit=args.limit, allow_qa_only=args.allow_qa_only)
    print(f"      정상 {cap['records']}건 / 오류 {len(errors)}건 / 수준 {cap['tier']}")
    if report is None:
        print("진단 불가 — 컨텍스트가 없거나(--allow-qa-only) 유효 레코드가 없습니다.")
        return 1

    print("[2/3] 진단서 뷰 조립")
    view = build_ext_report_view(report, cap)

    print("[3/3] HTML 저장")
    html_path, _ = write_report_files(view, OUT_DIR)

    score = view["score"]
    print("\n" + "=" * 56)
    print(f"  결과 — {log_path.name}")
    print("=" * 56)
    print(f"레코드/소견   : {cap['records']} / {score['findings_count']}")
    print(f"종합 점수     : {score['after']}  (gate_pass={score['gate'].get('pass')})")
    print(f"권고 카드     : {len(view['recommendations'])}건")
    for rec in view["recommendations"]:
        print(f"  · [{rec['badge'][1]}] {rec['title']} ({rec['cta']})")
    print(f"\n진단서: {html_path}")

    if not args.no_open:
        webbrowser.open(html_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
