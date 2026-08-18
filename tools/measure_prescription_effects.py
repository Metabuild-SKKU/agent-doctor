"""
tools/measure_prescription_effects.py
같은 baseline 에서 config 축을 하나씩만 바꿔 재고, 처방별 효과 크기를 낸다.

[왜 이 스크립트가 있나]
  Optimize 루프는 **진단이 가리키는 곳으로 수렴하는 탐색**이지 실험 도구가 아니다.
  실측(pipeline_20260817_125803 · pipeline_20260818_140611)이 드러낸 한계는 셋이다.
    ① 진단이 한 축을 가리키면 예산을 8회 줘도 그 축만 판다 — 리랭커 3형제가
       17+8+8건으로 검색 슬롯을 채우면 top_k·chunking 라벨은 차례가 오지 않는다.
    ② 처방을 draft 로 막아도 그 probe 의 표는 다른 라벨로 넘어가지 않고 **소멸**한다.
       진단(_pick)이 이미 라벨을 배정했기 때문이다. 08-17 실행이 그 실험이었고
       라벨 목록이 08-18 과 사실상 같았다.
    ③ **baseline 이 회차마다 바뀌어 처방끼리 비교가 안 된다.** 같은 restate_question 이
       08-17 에서 +5, 08-18 에서 +2 였다.
  이 스크립트는 baseline 을 고정하고 축 하나씩만 바꿔, 서로 비교 가능한 효과 크기를 낸다.
  이 값은 우선순위 공식이 의도적으로 비워 둔 `impact` 다(CONTEXT.md §4:
  "영향도(impact)는 공식에서 제외. 처방 전 예측 불가 → 검증에서 결과로 실측").

[설계 - measure_eval_sigma.py 에서 가져온 것]
  ① 실행마다 **별도 프로세스**. state.eval_cache 가 같은 config 를 cache hit 시켜
     이전 리포트를 복원하므로, 한 프로세스에서 반복하면 "흔들림 0" 이라는 가짜 답이
     나온다. hit 이 나면 그 실행은 실패로 처리한다.
  ② 인덱스도 실행마다 다시 만든다(실제 Optimize 루프도 방문마다 재색인한다).
  ③ 점수 축은 production 과 같은 표시 종합점수(0~100 정수).

[이 스크립트만의 것]
  · **probe 를 고정한다.** taxonomy 로더가 앞에서 N개를 자르므로 KORQUAD_QA_LIMIT 이
    같으면 매 실행이 같은 질문을 본다. A/B 비교에는 층화 추출보다 이쪽이 맞다 —
    비교 대상끼리 같은 문제를 풀어야 차이를 config 탓으로 돌릴 수 있다.
  · Optimize 를 **돌리지 않는다.** Ingest→Index→Eval 까지만.
  · 결과를 실행 직후 즉시 파일로 쓴다. 중간에 끊겨도 거기까지는 남는다.

[판정 기준 - 실측 σ 기준]
  output/sigma/20260814_155646 기준 σ_Δ(독립 두 측정의 차이) = 0.0107 @ probe 100.
  σ 는 대략 1/√N 로 줄어드므로 probe 를 줄이면 판별력이 떨어진다.
      probe 100 → 구분 가능한 최소 효과 약 3점 (MIN_IMPROVEMENT_MARGIN 과 같은 근거)
      probe  25 → 약 4~5점
  그래서 적은 probe 는 **선별용**이다. "효과 없음"을 걸러내는 데 쓰고, 살아남은 축만
  probe 를 늘려 확인한다.

[비용]
  실측(pipeline_20260818_140611) 기준 probe 1건당 약 $0.0058, 그중 **93% 가 RAGAS 채점**
  (STEP3, Claude). 답변 생성(DeepSeek)은 1.3%, 원인 판정은 6% 다.
  즉 비용은 (probe 수 × 실행 수)에 거의 정비례한다.

[사용법]
    # 무엇을 얼마에 돌릴지만 보여주고 끝낸다(기본 동작 - API 호출 없음)
    python tools/measure_prescription_effects.py \
        --axis use_reranker=true --axis top_k=10 --repeat 2 --set KORQUAD_QA_LIMIT=25

    # 배선 확인(LLM 호출 없음, 답변은 스텁)
    python tools/measure_prescription_effects.py --axis top_k=10 --dry-run --run \
        --set KORQUAD_MAX_DOCS=2 --set KORQUAD_QA_LIMIT=6

    # 본 측정
    python tools/measure_prescription_effects.py --run --repeat 2 \
        --set KORQUAD_QA_LIMIT=25 \
        --axis use_reranker=true \
        --axis top_k=10 \
        --axis rerank_candidates=40 \
        --axis use_hybrid=true

    # 이미 돌린 결과만 다시 집계(무료)
    python tools/measure_prescription_effects.py --aggregate-only output/effects/20260818_210000

축 이름은 **index_config 키**를 그대로 쓴다(canonical path 가 아니다).
  검색 : top_k, use_hybrid, use_reranker, rerank_candidates, use_mmr, mmr_lambda
  청킹 : chunk_size, chunk_overlap, chunk_strategy
  생성 : abstention_strict, abstention_relaxed, restate_question, require_citation,
         completeness_mode, temperature
값은 true/false/정수/실수/문자열을 자동 판별한다. 요청한 축이 실제로 반영됐는지는
결과 파일의 index_config 로 검증할 수 있다 — 오타나 미매핑이면 거기서 드러난다.

코퍼스·QA 는 평소 실행과 같은 env 로 고른다(SOURCE_TYPE/SOURCE_URL/EVAL_TAXONOMY_QA).
효과는 코퍼스에 딸린 값이므로 결과 파일에 그 조건을 함께 남긴다.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.console import force_utf8_stdio  # noqa: E402
from core.embedding_cli import add_embedding_args, apply_embedding_args  # noqa: E402

DEFAULT_OUT_ROOT = Path("output") / "effects"

# 실측 단가(pipeline_20260818_140611): probe 1건당 총액. 93% 가 RAGAS 채점이다.
COST_PER_PROBE_USD = 0.0058

# 실측 σ_Δ @ probe 100 (output/sigma/20260814_155646) 를 표시 스케일(0~100)로 환산한 값.
SIGMA_DELTA_AT_100_DISPLAY = 1.07


def _parse_value(raw: str):
    """true/false/int/float/문자열 자동 판별. index_config 값 타입을 맞춘다."""
    low = raw.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _parse_axes(items):
    axes = []
    for item in items:
        if "=" not in item:
            raise SystemExit("--axis 는 KEY=VALUE 형식이어야 한다: " + repr(item))
        key, _, raw = item.partition("=")
        axes.append((key.strip(), _parse_value(raw)))
    return axes


def _apply_env_overrides(items) -> None:
    for item in items:
        key, _, value = item.partition("=")
        if key.strip():
            os.environ[key.strip()] = value


def _state_from_env():
    """평소 실행과 같은 env 규약으로 state 를 만든다.

    measure_eval_sigma._state_from_env 와 같은 규약이다. run_local_pipeline 은 import 하면
    파이프라인이 통째로 돌아버려 재사용할 수 없어 여기서도 복제한다 — 규약이 바뀌면
    세 곳(run_local_pipeline / measure_eval_sigma / 여기)을 같이 고쳐야 한다.
    """
    from core.state import AgentDoctorState

    state = AgentDoctorState()
    source_type = os.getenv("SOURCE_TYPE", "file").strip().lower()
    state.source_type = source_type
    if source_type == "korquad":
        state.source_url = os.getenv("SOURCE_URL", "data/corpus.jsonl")
        os.environ.setdefault("EVAL_PROBE_SOURCE", "taxonomy")
    elif source_type == "file":
        state.source_url = os.getenv("SOURCE_URL", "sample_docs/hr_policy.md")
        state.user_questions = [
            "재택근무 며칠까지 가능해?",
            "연차는 며칠이야?",
            "성과급은 언제 나와?",
        ]
    else:
        state.source_url = os.getenv("SOURCE_URL", "")
    return state


def _stub_generation() -> None:
    """--dry-run 용. 답변 생성 LLM 만 스텁으로 바꾼다(measure_eval_sigma 와 같은 방식).

    EVAL_ENABLE_LLM=0 은 RAGAS 만 막고 STEP2 답변 생성은 그대로 API 를 태운다.
    배선 확인에 돈을 쓰지 않으려면 여기까지 끊어야 한다.
    """
    import agents.eval.agent as eval_agent

    def _fake(question, contexts, **kwargs):
        head = contexts[0][:200] if contexts else ""
        return "[dry-run] " + str(question) + " :: " + head

    eval_agent.generate_answer = _fake


def _worker(args) -> int:
    """측정 1회. config 축을 적용해 Ingest→Index→Eval 을 돌리고 결과를 덤프한다."""
    force_utf8_stdio()
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass
    _apply_env_overrides(args.overrides)
    apply_embedding_args(args)

    if args.dry_run:
        os.environ["EVAL_ENABLE_LLM"] = "0"
        os.environ["EVAL_MODE"] = "fast"
        _stub_generation()

    from agents.ingest.agent import run as ingest_run
    from agents.index.agent import run as index_run
    from agents.eval.agent import run as eval_run

    overrides = json.loads(args.axis_json)
    state = _state_from_env()
    state.index_config.update(overrides)

    for name, fn in (("Ingest", ingest_run), ("Index", index_run), ("Eval", eval_run)):
        state = fn(state)
        if state.error:
            raise RuntimeError(name + " 실패: " + str(state.error))
    if state.eval_cache_hit:
        # 프로세스를 나눈 이유가 이것이다. hit 이면 재측정이 아니라 캐시 복원이다.
        raise RuntimeError("eval_cache_hit=True — 재측정이 아니라 캐시 복원이다")

    composite = (state.report.composite_score or {}) if state.report else {}
    labels = {}
    if state.report is not None:
        summary = state.report.findings_summary or {}
        labels = dict(summary.get("confirmed_labels") or {})

    out = {
        "cell": args.cell,
        "repeat": args.repeat_index,
        "overrides": overrides,
        "composite_total": composite.get("total"),
        "components": composite.get("components"),
        "probe_count": len(state.probes or []),
        # 요청한 축이 실제로 config 에 박혔는지 검증용. 미매핑이면 여기서 드러난다.
        "index_config": dict(state.index_config),
        "labels": labels,
        "eval_mode": os.getenv("EVAL_MODE", ""),
    }
    path = Path(args.outdir) / (args.cell + "__r" + ("%02d" % args.repeat_index) + ".json")
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("[효과] " + args.cell + " 회차 " + str(args.repeat_index + 1)
          + ": probe " + str(out["probe_count"]) + "개, 종합점수 "
          + str(out["composite_total"]) + " → " + path.name, flush=True)
    return 0


def _cell_name(overrides) -> str:
    if not overrides:
        return "baseline"
    return "__".join(str(k) + "=" + str(v) for k, v in sorted(overrides.items()))


def _plan(axes, repeat):
    """측정 계획. baseline 을 맨 앞에 둔다 — 비교 기준이라 먼저 재는 편이 읽기 쉽다."""
    cells = [({}, "baseline")]
    for key, value in axes:
        ov = {key: value}
        cells.append((ov, _cell_name(ov)))
    return [(ov, name, r) for ov, name in cells for r in range(repeat)]


def _aggregate(outdir: Path) -> dict:
    """baseline 대비 효과를 낸다. 각 셀은 반복 평균."""
    runs = []
    for path in sorted(outdir.glob("*__r*.json")):
        try:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            print("  경고: " + path.name + " 읽기 실패(" + str(exc) + ") — 건너뜀")

    by_cell = {}
    for run in runs:
        if run.get("composite_total") is None:
            continue
        by_cell.setdefault(run["cell"], []).append(run)

    if "baseline" not in by_cell:
        return {"error": "baseline 측정이 없다 — 효과를 낼 기준이 없다", "cells": {}}

    def _mean(vals):
        return sum(vals) / len(vals) if vals else None

    base_mean = _mean([r["composite_total"] for r in by_cell["baseline"]])
    probe_n = by_cell["baseline"][0].get("probe_count") or 0
    # σ 는 1/√N 로 줄어든다. 실측(probe 100)에서 환산한다.
    sigma_delta = (SIGMA_DELTA_AT_100_DISPLAY * (100.0 / probe_n) ** 0.5) if probe_n else None
    threshold = round(2 * sigma_delta, 1) if sigma_delta else None

    cells = {}
    for name, group in by_cell.items():
        scores = [r["composite_total"] for r in group]
        mean = _mean(scores)
        cells[name] = {
            "overrides": group[0].get("overrides", {}),
            "runs": len(scores),
            "scores": scores,
            "mean": round(mean, 2),
            "spread": (max(scores) - min(scores)) if len(scores) > 1 else None,
            "effect": None if name == "baseline" else round(mean - base_mean, 2),
        }
    return {
        "baseline_mean": round(base_mean, 2),
        "probe_count": probe_n,
        "sigma_delta_display": round(sigma_delta, 2) if sigma_delta else None,
        "detect_threshold_display": threshold,
        "conditions": {
            "source_url": os.getenv("SOURCE_URL", ""),
            "taxonomy_qa": os.getenv("EVAL_TAXONOMY_QA", ""),
            "eval_mode": os.getenv("EVAL_MODE", ""),
        },
        "cells": cells,
    }


def _print_report(summary: dict) -> None:
    if summary.get("error"):
        print("\n집계 실패: " + summary["error"])
        return
    thr = summary.get("detect_threshold_display")
    print("\n" + "=" * 68)
    print("baseline 종합점수 " + str(summary["baseline_mean"])
          + "  ·  probe " + str(summary["probe_count"]) + "개")
    if thr:
        print("판별 문턱 ±" + str(thr) + "점 (2·σ_Δ, probe 수로 환산)"
              " — 이보다 작은 차이는 노이즈와 구분 불가")
    print("=" * 68)
    rows = [(n, c) for n, c in summary["cells"].items() if n != "baseline"]
    rows.sort(key=lambda x: -(x[1]["effect"] or 0))
    print("%-34s%7s%8s%7s  판정" % ("축", "평균", "효과", "편차"))
    for name, cell in rows:
        eff = cell["effect"]
        spread = cell["spread"] if cell["spread"] is not None else 0
        if thr is None:
            verdict = "-"
        elif eff >= thr:
            verdict = "개선"
        elif eff <= -thr:
            verdict = "악화"
        else:
            verdict = "구분 불가"
        print("%-34s%7s%+8s%7s  %s" % (name[:33], cell["mean"], eff, spread, verdict))


def main() -> int:
    force_utf8_stdio()
    p = argparse.ArgumentParser(
        description="baseline 고정 + 축 하나씩 변경으로 처방별 효과 크기를 잰다",
    )
    p.add_argument("--axis", dest="axes", action="append", default=[], metavar="KEY=VALUE",
                   help="측정할 index_config 축(반복 가능). 예: --axis top_k=10")
    p.add_argument("--repeat", type=int, default=2,
                   help="셀당 반복 횟수(기본 2). 노이즈 폭을 보려면 2 이상이어야 한다")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="env 덮어쓰기(반복 가능). .env 로드 뒤에 적용된다. "
                        "예: --set KORQUAD_QA_LIMIT=25")
    p.add_argument("--run", action="store_true",
                   help="실제로 측정한다. 없으면 계획과 예상 비용만 출력하고 끝낸다")
    p.add_argument("--dry-run", action="store_true",
                   help="LLM 호출 없이 배선만 확인(답변 스텁, RAGAS off)")
    p.add_argument("--outdir", default="",
                   help="결과 디렉터리(기본: output/effects/<타임스탬프>)")
    p.add_argument("--aggregate-only", default="",
                   help="이미 돌린 결과 디렉터리를 다시 집계만 한다(무료)")
    p.add_argument("--skip-existing", action="store_true",
                   help="already-measured cells in --outdir are skipped; "
                        "use to split axes across runs or resume an interrupted run")
    add_embedding_args(p)
    # 내부 전용 - 부모가 자기 자신을 서브프로세스로 부를 때 쓴다.
    p.add_argument("--_cell", dest="cell", default="", help=argparse.SUPPRESS)
    p.add_argument("--_repeat-index", dest="repeat_index", type=int, default=-1,
                   help=argparse.SUPPRESS)
    p.add_argument("--_axis-json", dest="axis_json", default="{}", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.repeat_index >= 0:
        return _worker(args)

    if args.aggregate_only:
        outdir = Path(args.aggregate_only)
        if not outdir.is_dir():
            print("디렉터리가 없다: " + str(outdir))
            return 1
        summary = _aggregate(outdir)
        (outdir / "effects_report.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        _print_report(summary)
        return 0

    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass
    _apply_env_overrides(args.overrides)

    axes = _parse_axes(args.axes)
    if not axes:
        print("측정할 축이 없다. --axis KEY=VALUE 를 하나 이상 지정할 것.")
        return 1

    plan = _plan(axes, args.repeat)
    # outdir must be resolved before the estimate - --skip-existing removes
    # already-measured cells from the cost estimate too.
    outdir = (Path(args.outdir) if args.outdir
              else DEFAULT_OUT_ROOT / time.strftime("%Y%m%d_%H%M%S"))
    skipped = []
    if args.skip_existing and outdir.is_dir():
        remaining = []
        for ov, name, rep in plan:
            if (outdir / (name + "__r" + ("%02d" % rep) + ".json")).exists():
                skipped.append((name, rep))
            else:
                remaining.append((ov, name, rep))
        plan = remaining
    probe_limit = int(os.getenv("KORQUAD_QA_LIMIT") or 0)
    est = len(plan) * probe_limit * COST_PER_PROBE_USD if probe_limit else None

    print("측정 계획: 셀 " + str(len(axes) + 1) + "개(baseline 포함) × 반복 "
          + str(args.repeat) + " = 실행 " + str(len(plan)) + "회")
    print("  코퍼스: " + (os.getenv("SOURCE_URL") or "(미지정)"))
    print("  질문 수: " + (str(probe_limit) if probe_limit
                        else "(KORQUAD_QA_LIMIT 미설정 — 전체)"))
    print("  EVAL_MODE=" + os.getenv("EVAL_MODE", "")
          + " EVAL_ENABLE_LLM=" + os.getenv("EVAL_ENABLE_LLM", ""))
    for key, value in axes:
        print("  축: " + key + " = " + repr(value))
    if est is not None:
        print("  예상 비용: 약 $%.2f (probe 1건당 $%s, 93%%가 RAGAS 채점)"
              % (est, COST_PER_PROBE_USD))
    else:
        print("  예상 비용: KORQUAD_QA_LIMIT 이 없어 산정 불가 — 전체 질문셋은 비싸다")
    if args.dry_run:
        print("  (--dry-run: LLM 호출 없음)")
    if not args.run:
        print("\n계획만 출력했다. 실제로 돌리려면 --run 을 붙일 것.")
        return 0

    if not plan:
        print("")
    outdir.mkdir(parents=True, exist_ok=True)
    print("")

    failed = 0
    for idx, (ov, name, rep) in enumerate(plan, start=1):
        cmd = [sys.executable, os.path.abspath(__file__),
               "--_cell", name, "--_repeat-index", str(rep),
               "--_axis-json", json.dumps(ov, ensure_ascii=False),
               "--outdir", str(outdir)]
        for item in args.overrides:
            cmd += ["--set", item]
        if args.dry_run:
            cmd.append("--dry-run")
        print("[" + str(idx) + "/" + str(len(plan)) + "] " + name
              + " 회차 " + str(rep + 1) + " 실행 중...")
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            failed += 1
            print("  실패(exit " + str(proc.returncode) + ") — 이 셀은 집계에서 빠진다")

    summary = _aggregate(outdir)
    (outdir / "effects_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_report(summary)
    if failed:
        print("\n경고: " + str(failed) + "회 실패. 남은 결과로만 집계했다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
