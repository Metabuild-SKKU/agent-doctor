"""
tools/bench_question_detail.py
문항별 상세 로그 — 파이프라인 로그(Eval STEP4)와 같은 밀도로, 전 문항을 replay 경로에서.

왜 필요한가: replay 산출물은 실패 문항만 상세히 남긴다(run_replay_report._log_questions).
`bench_question_table.py` 가 전 문항을 찍긴 하지만 통과/실패와 답변까지고, 지표는 없다.
두 시스템을 문항 단위로 파고들려면 "이 문항에서 무엇을 검색했고 점수가 어땠나"가 있어야 한다.

방법: replay 를 다시 돌리되 **EvalRecord 를 붙잡아 둔다.** run_replay 는 레코드를 제자리에서
채우므로(ragas·findings·retrieval_axis), 호출자가 리스트를 쥐고 있으면 문항별 지표가 남는다.
CLI(agents.eval.replay)는 리포트만 돌려주고 레코드를 버려서 이 값을 볼 수 없다.

파이프라인 로그와 **다를 수밖에 없는 것**:
  · `골드 [chunk_xxx]` / `recall@k(span)` — 내부 모드는 gold 를 문자 좌표로 들고 있다가
    현재 청크에 resync 해서 "정답 청크"를 특정한다. 남의 로그에는 그 좌표가 없고 청크
    네임스페이스도 다르므로 replay 는 recall_at_k 를 -1(미측정)로 둔다(replay.py 정직성 규약).
    대신 gold_contexts 텍스트가 검색 컨텍스트에 들었는지로 검색축을 판정한다.
  · `oracle_f1` — gold 청크를 넣고 다시 생성해 보는 트랙이라 gold 전제가 필요하다.

주의: RAGAS 를 다시 호출하므로 시스템당 비용이 든다(실측 50문항 ≈ $1).

사용법:
    python -m tools.bench_question_detail \
        "AutoRAG+리랭커=bench/out/report_autorag_rr" \
        "우리+리랭커=bench/out/report_ours_rr"
"""
from __future__ import annotations

import glob
import json
import os
import sys

DEFAULT_GOLDEN = "bench/golden_test_50.json"


def _fmt(value, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"


def _short(chunk_id: str) -> str:
    """청크 ID 를 읽을 수 있게 줄인다.

    우리 로그는 "<문서UUID>_chunk_253" 이고 AutoRAG 로그는 "chunk_00253" 이다. 문서가
    하나뿐인 벤치마크에서 UUID 접두사는 전부 같은 값이라 정보가 없으면서 줄을 넘기고,
    무엇보다 두 시스템의 로그를 나란히 놓고 대조할 수 없게 만든다. 뒤쪽 chunk_NNN 만 남겨
    양쪽 표기를 맞춘다(번호 자체는 서로 다른 청킹이라 같은 번호가 같은 글이 아니다 —
    비교는 '골드텍스트 포함 여부'로 하지 번호로 하지 않는다)."""
    tail = str(chunk_id).rsplit("_chunk_", 1)
    return f"chunk_{tail[1]}" if len(tail) == 2 else str(chunk_id)


def detail_for(label: str, folder: str, golden_path: str) -> str:
    from agents.eval.log_intake import load_external_log
    from agents.eval.replay import apply_golden_set, build_replay_records, run_replay

    logs_glob = glob.glob(os.path.join(folder, "*.jsonl"))
    if len(logs_glob) != 1:
        sys.exit(f"[detail] {folder} 에 triad 로그가 {len(logs_glob)}개입니다 — 1개여야 합니다.")
    log_path = logs_glob[0]

    logs, errors = load_external_log(log_path)
    apply_golden_set(logs, golden_path)
    records = build_replay_records(logs)
    report = run_replay(records)          # records 를 제자리에서 채운다

    out = [f"─── 문항별 상세 (전 {len(records)}건) ─ {label}",
           f"    로그 {os.path.basename(log_path)} · 골든셋 {os.path.basename(golden_path)}",
           f"    종합 {(report.composite_score or {}).get('total')} · overall {_fmt(report.overall_score, 3)}",
           ""]
    passed = 0
    for i, rec in enumerate(records, 1):
        marks = " · ".join(
            f"{f.label}{'' if f.confirmed else '(예비)'}" for f in rec.findings)
        ok = not rec.findings
        passed += ok
        out.append(f"  [{i}/{len(records)}] {rec.probe.probe_id}  " + ("✅" if ok else f"❌ {marks}"))
        out.append(f"    Q: {rec.probe.question}")
        out.append(f"    A: {rec.probe.ground_truth}")
        out.append(f"    R: {rec.generated_answer}")
        out.append(f"    검색 [{', '.join(_short(c) for c in rec.retrieved_chunk_ids)}]")
        gold = (rec.probe.metadata or {}).get("gold_contexts") or []
        # 남의 인덱스라 '정답 청크 ID'를 특정할 수 없다. 대신 gold 텍스트를 담은 청크를
        # 표시해 같은 자리를 메운다(내부 로그의 `골드 [chunk_xxx]` 대응물).
        hit = [cid for cid, ctx in zip(rec.retrieved_chunk_ids, rec.retrieved_context)
               if any(g and g in ctx for g in gold)]
        out.append(f"    골드텍스트 포함 청크 [{', '.join(_short(c) for c in hit) if hit else '없음'}]")
        r = rec.ragas or {}
        out.append(
            f"    검색축={_fmt(rec.retrieval_axis)}  f1={_fmt(rec.f1_score)}  "
            f"faith={_fmt(r.get('faithfulness'))}  relevancy={_fmt(r.get('response_relevancy'))}  "
            f"ctx_prec={_fmt(r.get('context_precision'))}  ctx_recall={_fmt(r.get('context_recall'))}  "
            f"correctness={_fmt(r.get('answer_correctness'))}")
        for f in rec.findings:
            out.append(f"    ! {f.label}: {f.metadata.get('reason', '')}")
        out.append("")
    out.append(f"─── 합계: 통과 {passed} / 실패 {len(records) - passed} / 전체 {len(records)}")
    if errors:
        out.append(f"    적재 오류 {len(errors)}건")
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    from core.console import force_utf8_stdio
    force_utf8_stdio()
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass
    # replay 는 RAGAS 가 유일한 작업이라 끄면 생성축 지표가 통째로 빈다(replay.py 주석).
    os.environ["EVAL_ENABLE_LLM"] = "1"
    if os.getenv("EVAL_MODE", "").strip().lower() not in ("deep", "full", "3", "4"):
        os.environ["EVAL_MODE"] = "deep"

    golden = DEFAULT_GOLDEN
    pairs = []
    for arg in argv:
        if arg.startswith("--golden="):
            golden = arg.split("=", 1)[1]
        elif "=" in arg:
            label, folder = arg.split("=", 1)
            pairs.append((label, folder))
    if not pairs:
        print(__doc__)
        return 2

    for label, folder in pairs:
        print(f"\n=== {label} · {folder} ===")
        text = detail_for(label, folder, golden)
        path = os.path.join(folder, "questions_detail.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(text.strip().splitlines()[-1])
        print(f"  → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
