"""
tools/run_ragec_validation.py
RAGEC 정답지로 **진단 유효성**을 잰다 — 실행부터 채점까지 한 번에 (로드맵 8-c·8-d).

    python tools/run_ragec_validation.py --embed openrouter

## 왜 전용 스크립트인가

`run_local_pipeline.py` 는 범용 스모크다. 이 검증은 환경변수 셋(SOURCE_TYPE·SOURCE_URL·
EVAL_TAXONOMY_QA)을 정확히 맞춰야 하는데, 하나라도 어긋나면 **KorQuAD 설정으로 돌아
API 비용만 나가고 채점은 못 한다.** 여기서 고정해 그 사고를 없앤다.

## Optimize 를 돌리지 않는다

재려는 것은 **baseline 진단이 사람 라벨과 맞는가** 다. Optimize 를 태우면 config 가 바뀌고
재평가가 붙어 "어느 시점의 진단을 채점하는지" 가 흐려진다. Ingest → Index → Eval 까지만
간다.

## 우리가 성공한 질문은 오답이 아니다

RAGEC 377건은 *그들* RAG 시스템이 실패한 질문이다. 검색기·생성 모델이 다른 우리는 같은
질문에서 성공할 수 있고, 그건 진단 오류가 아니다. 그래서 덤프에 **성공/실패를 함께** 싣고
채점기가 따로 센다(`score_ragec.score` 의 we_passed).

## 내는 것

    output/ragec/findings.jsonl   probe 별 {qa_id, labels, failed}
    콘솔                          정확도 표

진단서 HTML 은 여기서 만들지 않는다 — `core/report_html` 은 다른 브랜치(#121)에 있고,
안 머지된 모듈에 의존하면 이 스크립트가 브랜치에 따라 다르게 동작한다. 필요하면 findings
파일과 `--key` 만으로 `tools/score_ragec.py` 를 다시 돌릴 수 있다.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CORPUS = "data/ragec_corpus.jsonl"
QA = "data/ragec_qa.jsonl"
KEY = "data/ragec_answer_key.jsonl"
OUT_DIR = REPO_ROOT / "output" / "ragec"


def _load_env() -> None:
    """.env 로드. import 시점이 아니라 여기서만 부른다(import 부작용 방지)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass


def main() -> int:
    import argparse

    from core.console import force_utf8_stdio
    force_utf8_stdio()

    from core.embedding_cli import (
        add_embedding_args, apply_embedding_args, describe_embedding_route,
    )

    parser = argparse.ArgumentParser(
        description="RAGEC 정답지로 진단 유효성 측정 (실행 + 채점)")
    parser.add_argument("--limit", type=int, default=0,
                        help="probe 상한(0=전체 377). 비용을 아껴 배선만 확인할 때 쓴다")
    parser.add_argument("--label-sample", type=int, default=60,
                        help="사람이 채울 라벨 시트의 표본 수(0=전체). "
                             "1건당 2~3분 소요를 감안할 것")
    add_embedding_args(parser)
    args = parser.parse_args()

    _load_env()
    applied = apply_embedding_args(args)   # .env 뒤에 적용해야 플래그가 이긴다

    missing = [p for p in (CORPUS, QA, KEY) if not (REPO_ROOT / p).exists()]
    if missing:
        print(f"[오류] 입력이 없습니다: {', '.join(missing)}", file=sys.stderr)
        print("       tools/build_ragec_dataset.py 로 먼저 만드세요.", file=sys.stderr)
        return 1

    # 이 검증의 설정을 여기서 확정한다. .env 가 무엇이든 덮어쓴다 — 절반만 맞은 설정으로
    # 도는 것이 가장 위험하다(비용은 나가고 결과는 못 쓴다).
    os.environ["SOURCE_TYPE"] = "korquad"          # 스키마가 같아 로더를 재사용한다
    os.environ["SOURCE_URL"] = CORPUS
    os.environ["EVAL_PROBE_SOURCE"] = "taxonomy"
    os.environ["EVAL_TAXONOMY_QA"] = QA
    os.environ["KORQUAD_MAX_DOCS"] = "0"           # 108개 전부
    os.environ["KORQUAD_QA_LIMIT"] = str(max(0, args.limit))
    os.environ.setdefault("EVAL_MODE", "deep")     # 라벨을 확정하려면 RAGAS 가 필요하다
    os.environ.setdefault("EVAL_ENABLE_LLM", "1")
    # DragonBall 은 영어 코퍼스다. 생성 프롬프트 기본값(한국어)으로 두면 질문은 영어인데
    # 답변만 한국어로 나와 char-F1 이 **구조적으로 0** 이 되고, _is_success 의 lexical 축이
    # 항상 실패라 **검색이 완벽해도 실패로 집계된다**(실측: recall=1.00·gold 2/2 검색인데
    # f1=0.00 → generation_hallucination 오진). 그러면 채점이 진단 품질이 아니라 언어
    # 불일치를 재게 된다.
    os.environ.setdefault("RAG_ANSWER_LANGUAGE", "match")

    from core.run_logger import setup_run_logging
    # 검증 로그는 output/logs 가 아니라 검증 폴더 아래에 모은다 — 파이프라인 스모크 로그와
    # 섞이면 "어느 게 유효성 검증이었지" 를 매번 뒤져야 한다. 산출물(findings·시트·채점)이
    # 전부 output/ragec 아래 있으니 로그도 같은 자리에 둔다.
    log_path = setup_run_logging(log_dir=str(OUT_DIR / "logs"), prefix="run")

    from agents.eval.agent import run as eval_run
    from agents.index.agent import run as index_run
    from agents.ingest.agent import run as ingest_run
    from core.state import AgentDoctorState
    from tools.make_label_sheet import (
        PRIMARY_FIELD, stratified_sample, summarize, write_sheet,
    )
    from tools.score_ragec import (
        _read_jsonl, findings_from_report, format_detail, format_report, score,
    )

    print(f"[RAGEC] {describe_embedding_route()}")
    if applied:
        print(f"[RAGEC] 임베딩 설정 적용: {applied}")
    print(f"[RAGEC] 코퍼스 {CORPUS} · QA {QA}"
          f"{f' (상한 {args.limit})' if args.limit else ' (전체)'}")

    state = AgentDoctorState()
    state.source_type = "korquad"
    state.source_url = CORPUS

    for name, step in (("Ingest", ingest_run), ("Index", index_run), ("Eval", eval_run)):
        print("\n" + "=" * 56 + f"\n  {name}\n" + "=" * 56)
        state = step(state)
        if state.error:
            print(f"[중단] {name} 오류: {state.error}", file=sys.stderr)
            return 1

    if state.report is None:
        print("[오류] Eval 이 리포트를 내지 않았습니다.", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 성공한 probe 도 실어야 채점기가 '우리는 성공' 과 '원인을 못 짚음' 을 가른다.
    # probe 객체를 그대로 넘겨 질문·정답 원문까지 싣는다(probe_id 만 넘기면 대조 출력에서
    # 라벨만 보이고, 진단이 틀린 건지 데이터가 틀린 건지 갈리지 않는다).
    rows = findings_from_report(state.report, state.probes)
    findings_path = OUT_DIR / "findings.jsonl"
    with open(findings_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nfindings → {findings_path}  ({len(rows)}건)")

    # 사람이 채울 라벨 시트. RAGEC 라벨은 *그들* 시스템의 관측을 보고 붙인 것이라 우리
    # 관측에는 성립하지 않을 수 있다(실측 qa_id 2205: 그쪽은 '검색 실패' 인데 우리는
    # recall=1.00). 그래서 **우리 관측을 보고 사람이 다시** 붙인 라벨이 따로 필요하다.
    # 진단은 이 시트에 들어가지 않는다 — 보고 나면 '맞는지' 가 아니라 '동의하는지' 를
    # 판단하게 되어 검증이 성립하지 않는다.
    sheet_path = OUT_DIR / "label_sheet.json"
    sample = stratified_sample(rows, args.label_sample)
    if sample:
        write_sheet(sample, sheet_path)
        print(f"라벨 시트 → {sheet_path}  ({len(sample)}건)")
        print(summarize(sample))
    print(f"\n  ↳ '{PRIMARY_FIELD}' 칸을 채운 뒤: python tools/score_human_labels.py")

    key_rows = _read_jsonl(str(REPO_ROOT / KEY))
    print(format_detail(rows, key_rows))
    print()
    print(format_report(score(rows, key_rows)))
    if log_path:
        print(f"\n[log] {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
