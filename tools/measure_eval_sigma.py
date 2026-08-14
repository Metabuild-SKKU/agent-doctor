"""
tools/measure_eval_sigma.py
같은 config 를 반복 측정해 Eval 종합점수의 노이즈 σ 를 재고, 개선 마진을 제안한다. (#102)

[왜 이 스크립트가 있나]
  MIN_IMPROVEMENT_MARGIN(0.02) 은 측정에서 나온 값이 아니다(history.py 주석이 스스로
  그렇게 적어 두고 σ 측정을 요구한다). 마진의 목적은 "노이즈로 우연히 오른 점수를 개선으로
  확정하지 않는다" 인데, 노이즈가 마진보다 크면 judge 의 유지/롤백이 동전 던지기와
  구분되지 않는다. 이 스크립트는 그 σ 를 재서 마진을 숫자로 바꾼다.

[설계 - 왜 이런 모양인가]
  ① 반복은 **별도 프로세스**로 돌린다. state.eval_cache 는 같은 config 를 cache hit 시켜
     이전 리포트를 그대로 복원한다. 한 프로세스 안에서 반복하면 "흔들림 0" 이라는 가짜
     답이 나온다. 프로세스를 나누면 in-memory 캐시가 매번 비어 진짜 재측정이 된다.
  ② 인덱스도 회차마다 다시 만든다. 실제 Optimize 루프도 방문마다 재색인하므로, 색인
     단계에서 생기는 흔들림(있다면)은 마진이 막아야 할 노이즈에 포함된다.
  ③ probe 는 **층화 부분표집**한다. 종합점수는 probe 평균이라 필요한 것은 probe 1개당
     흔들림이다. 전체 100개×5회 대신 25개×5회면 관측치가 125개라 추정은 오히려 촘촘하고
     비용은 1/4 이다. 다만 통과/실패 비율이 치우친 표본은 σ 를 편향시키므로(통과 probe 는
     천장에 붙어 안 흔들린다) 기존 실행 로그의 라벨 분포를 그대로 따라간다.
  ④ 목표 N(운영 probe 수)으로의 환산은 **몬테카를로**로 한다. composite 은 조화평균이라
     비선형이고, 손으로 분산을 전파하면 틀린다. 측정한 관측치를 다시 뽑아 "가상의 N-probe
     실행"을 만들고 그 실행들이 얼마나 흩어지는지를 본다. 계산만이라 API 를 더 쓰지 않는다.
  ⑤ 마진은 σ 가 아니라 **σ_Δ** 로 정한다. 마진은 서로 다른 두 config 를 각각 한 번씩 잰
     차이에 걸리므로 기준은 독립 두 측정의 차이 분포다(독립이면 σ_Δ = √2·σ).
  ⑥ sweep 경로용으로 **best-of-T** 마진도 같이 낸다. internal sweep 은 후보 T개 중 최고를
     고른 뒤 baseline 과 비교하는데, 최고값은 그 자체로 위쪽으로 치우친다(승자의 저주).
     쌍대비교용 마진을 그대로 쓰면 후보가 많을수록 헛통과가 늘어난다.
  ⑦ 점수 축은 judge 와 같은 **정규화 composite(0~1)**. 표시용 0~100 과 헷갈리면 100배
     어긋난다. production 은 round(x*100) 으로 정수 양자화하므로(scoring.compute_composite)
     양자화를 반영한 값도 함께 낸다 - 실제 판정이 보는 눈금이 그쪽이다.

[측정하지 않는 것 - 이 실험의 범위 밖]
  · judge 모델을 바꿨을 때의 σ. 모델마다 다시 재야 한다.
  · 점수대가 크게 다른 config 의 σ. 분산은 등분산이 아니다(천장·바닥 근처에서 작아진다).
    운영점 근처에서 재고, 그 조건을 sigma_report.json 의 conditions 에 남긴다.
  · composite 이 아닌 축(예: chunk prescreener 의 span 포함률)에 걸리는 마진.

[사용법]
    # 배선 확인. LLM 호출 없음(답변 생성은 스텁, RAGAS 는 off)
    python tools/measure_eval_sigma.py --dry-run --probes 5 --repeat 2 \
        --set KORQUAD_MAX_DOCS=2 --set KORQUAD_QA_LIMIT=12

    # 본 측정 (API 비용 - probe 1건당 약 $0.006, 25×5 = 125건이면 대략 $0.75)
    python tools/measure_eval_sigma.py --probes 25 --repeat 5 \
        --labels-from-log output/logs/pipeline_20260814_133343.log

    # 이미 돌린 결과만 다시 집계 (무료. 목표 N 이나 k 만 바꿔 볼 때)
    python tools/measure_eval_sigma.py --aggregate-only output/sigma/20260814_153000

코퍼스·QA 는 평소 실행과 **같은 env** 로 고른다(SOURCE_TYPE/SOURCE_URL/EVAL_TAXONOMY_QA
등 - run_local_pipeline.py 와 같은 규약). σ 는 코퍼스·QA셋 크기·judge 모델에 딸린 값이라
결과 리포트(sigma_report.json)에 그 조건을 함께 적는다. 조건이 바뀌면 다시 재야 한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from random import Random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.console import force_utf8_stdio  # noqa: E402
from core.embedding_cli import add_embedding_args, apply_embedding_args  # noqa: E402

DEFAULT_OUT_ROOT = Path("output") / "sigma"
# 부분표집한 probe 파일은 EVAL_PROBE_SOURCE=made 로 읽힌다(probe_store 형식).
SUBSET_NAME = "subset_probes.json"
POOL_NAME = "pool_probes.json"
# sweep 후보 수. 실측 로그에서 관측된 범위(후보 1~3개)를 덮는다.
BEST_OF_T = (2, 3, 4)


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="같은 config 반복 측정으로 종합점수 σ 와 개선 마진을 구한다 (#102)",
    )
    p.add_argument("--probes", type=int, default=25,
                   help="부분표집할 probe 수 M (기본 25)")
    p.add_argument("--repeat", type=int, default=5,
                   help="반복 측정 횟수 R (기본 5). 각 회차는 별도 프로세스")
    p.add_argument("--target-n", type=int, default=0,
                   help="환산할 운영 probe 수 N (기본 0=pool 크기)")
    p.add_argument("--k", type=float, default=2.0,
                   help="마진 = k·σ_Δ 의 k (기본 2)")
    p.add_argument("--pool", default="",
                   help="probe pool 파일(probe_store 형식). 없으면 ingest+index 후 새로 만든다")
    p.add_argument("--labels-from-log", default="",
                   help="기존 파이프라인 로그에서 probe 별 라벨을 읽어 층화 기준으로 쓴다. "
                        "1차 표집의 통과/실패 비율 편향을 공짜로 없앤다")
    p.add_argument("--outdir", default="",
                   help="결과 디렉터리 (기본 output/sigma/<timestamp>)")
    p.add_argument("--aggregate-only", default="",
                   help="이 디렉터리의 기존 회차 결과만 재집계(실행·비용 없음)")
    p.add_argument("--seed", type=int, default=102, help="표집·몬테카를로 시드")
    p.add_argument("--mc-sets", type=int, default=120,
                   help="몬테카를로: 가상 probe 세트 개수 (기본 120)")
    p.add_argument("--mc-pairs", type=int, default=120,
                   help="몬테카를로: 세트당 실행 쌍 개수 (기본 120)")
    p.add_argument("--dry-run", action="store_true",
                   help="LLM 호출 없이 배선만 확인(답변 생성 스텁 + RAGAS off)")
    p.add_argument("--yes", action="store_true",
                   help="비용 확인 프롬프트를 건너뛴다(비대화형 실행에 필요)")
    p.add_argument("--cost-per-probe", type=float, default=0.006,
                   help="probe 1건당 예상 비용 USD (기본 0.006 - 100 probe 실행 $0.58 실측 기준)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="env 덮어쓰기(반복 가능). .env 를 load_dotenv(override=True) 로 읽은 "
                        "**뒤에** 적용되므로 셸 변수와 달리 확실히 먹는다. "
                        "예: --set KORQUAD_QA_LIMIT=20")
    add_embedding_args(p)
    # 내부 전용 - 부모가 자기 자신을 서브프로세스로 부를 때 쓴다.
    p.add_argument("--_build-pool", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_worker", type=int, default=-1, help=argparse.SUPPRESS)
    p.add_argument("--_subset", default="", help=argparse.SUPPRESS)
    return p


# ══════════════════════════════════════════════════════════════════
#  자식 프로세스 - 파이프라인 1회 실행
# ══════════════════════════════════════════════════════════════════

def _state_from_env():
    """평소 실행과 같은 env 규약으로 state 를 만든다(run_local_pipeline.py 와 같은 규약).

    여기서 소스 선택 규약을 복제하는 이유: run_local_pipeline 은 import 하면 파이프라인이
    통째로 돌아버리는 스크립트라 재사용할 수 없다. 규약이 바뀌면 두 곳을 같이 고쳐야 한다."""
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
    """--dry-run 전용: 답변 생성 LLM 만 스텁으로 바꾼다.

    EVAL_ENABLE_LLM=0 은 RAGAS 만 막고 STEP2 답변 생성은 그대로 API 를 태운다.
    배선 확인에 돈을 쓰지 않으려면 여기까지 끊어야 한다."""
    import agents.eval.agent as eval_agent

    def _fake(question: str, contexts: list[str], **kwargs) -> str:
        head = (contexts[0][:200] if contexts else "")
        return f"[dry-run] {question} :: {head}"

    eval_agent.generate_answer = _fake


def _run_pipeline(dry_run: bool):
    """ingest → index → eval 1회. (records, state) 반환. optimize 는 부르지 않는다."""
    from agents.ingest.agent import run as ingest_run
    from agents.index.agent import run as index_run
    from agents.eval.agent import run as eval_run
    import agents.eval.report as report_mod

    if dry_run:
        os.environ["EVAL_ENABLE_LLM"] = "0"
        os.environ["EVAL_MODE"] = "fast"
        _stub_generation()

    # probe 별 원자료를 꺼내는 유일한 지점. DiagnosticReport 에는 probe 별 수치가 남지
    # 않으므로(집계만 남는다) 점수 재료가 살아 있는 compute_composite 호출을 가로챈다.
    # 인자로 오는 records 는 bad_gold 를 제외한 '점수에 실제로 들어간' 집합이라,
    # 여기서 잡으면 production 종합점수와 같은 재료를 그대로 본다.
    captured: list = []
    _orig = report_mod.compute_composite

    def _capture(records):
        captured.append(list(records))
        return _orig(records)

    report_mod.compute_composite = _capture
    try:
        state = _state_from_env()
        for name, fn in (("Ingest", ingest_run), ("Index", index_run), ("Eval", eval_run)):
            state = fn(state)
            if state.error:
                raise RuntimeError(f"{name} 실패: {state.error}")
        if state.eval_cache_hit:
            # 프로세스를 나눈 이유가 이것이다. 그래도 hit 이 났다면 재측정이 아니다.
            raise RuntimeError("eval_cache_hit=True - 재측정이 아니라 캐시 복원이다")
    finally:
        report_mod.compute_composite = _orig
    if not captured:
        raise RuntimeError("compute_composite 가 불리지 않았다 - 점수 재료를 못 잡았다")
    return captured[-1], state


def _dump_record(rec) -> dict:
    """probe 1개의 점수 재료. scoring 이 읽는 필드만 남긴다(집계 때 그대로 되살린다)."""
    from agents.eval.types import RAGAS_WEIGHTS

    return {
        "probe_id": rec.probe.probe_id,
        "has_gold": bool(rec.probe.ground_truth),
        "answer_exists": rec.probe.answer_exists,
        "ragas": {k: rec.ragas.get(k) for k in RAGAS_WEIGHTS},
        "recall_at_k": rec.recall_at_k,
        "answer_score": rec.answer_score,
        "retrieval_axis": rec.retrieval_axis,
        "findings": len(rec.findings),
        # 라벨은 σ 계산에 안 쓴다. 다음 측정에서 층화 기준으로 쓰려고 남긴다.
        "labels": sorted({f.label for f in rec.findings if f.label}),
    }


def _worker(args) -> int:
    """반복 1회. 부분표집한 probe 파일을 made 소스로 읽어 평가하고 결과를 덤프한다."""
    os.environ["EVAL_PROBE_SOURCE"] = "made"      # 부분표집 파일을 그대로 쓴다
    os.environ["EVAL_PROBE_STORE"] = args._subset  # probe_store 경로 지정

    records, state = _run_pipeline(args.dry_run)
    composite = (state.report.composite_score or {}) if state.report else {}
    out = {
        "run": args._worker,
        "composite_total": composite.get("total"),   # production 표시 스케일(0~100 정수)
        "index_config": state.index_config,
        "eval_mode": os.getenv("EVAL_MODE", ""),
        "judge_model": os.getenv("EVAL_JUDGE_MODEL", ""),
        "records": [_dump_record(r) for r in records],
    }
    path = Path(args.outdir) / f"run_{args._worker:02d}.json"
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[σ] 회차 {args._worker + 1}: probe {len(out['records'])}개, "
          f"표시 종합점수 {out['composite_total']} → {path}", flush=True)
    return 0


def _build_pool(args) -> int:
    """probe pool 을 만든다(LLM 없이 파일 로드+gold 재동기화 경로). 부분표집의 모집단."""
    from agents.eval.probe_gen import generate_probes
    from agents.eval.probe_store import save_probes
    from agents.ingest.agent import run as ingest_run
    from agents.index.agent import run as index_run

    state = _state_from_env()
    for name, fn in (("Ingest", ingest_run), ("Index", index_run)):
        state = fn(state)
        if state.error:
            print(f"[σ] {name} 실패: {state.error}")
            return 1
    probes = generate_probes(state)
    path = str(Path(args.outdir) / POOL_NAME)
    save_probes(probes, "sigma-pool", path)
    print(f"[σ] probe pool {len(probes)}개 → {path}", flush=True)
    return 0


# ══════════════════════════════════════════════════════════════════
#  부모 프로세스 - 표집 · 반복 실행 · 집계
# ══════════════════════════════════════════════════════════════════

def _load_probe_file(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    probes = data.get("probes") if isinstance(data, dict) else data
    if not probes:
        raise SystemExit(f"[σ] probe 가 없다: {path}")
    return probes


def _labels_from_log(path: Path) -> dict[str, str]:
    """기존 파이프라인 로그에서 probe 별 진단 라벨을 긁는다(공짜 층화 기준).

    왜 필요한가: 1차 표집은 라벨을 모르니 probe 속성(qtype·gold 유무)으로 층을 나눌 수밖에
    없는데, 그 속성은 **점수가 얼마나 흔들리는지와 상관이 없다.** 정답을 늘 맞히는 probe 는
    천장에 붙어 거의 안 흔들리고 실패 계열은 크게 흔들린다. 통과/실패 비율이 우연히 치우친
    표본을 뽑으면 σ 가 통째로 편향되는데, 이미 돌려 둔 전체 실행 로그가 있으면 그 비율을
    공짜로 맞출 수 있다."""
    labels: dict[str, str] = {}
    current = ""
    head = re.compile(r"^\s*\[\d+/\d+\]\s+(\S+)")
    finding = re.compile(r"^\s*!\s*([a-z_]+)")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = head.match(line)
        if m:
            current = m.group(1)
            labels.setdefault(current, "pass")   # finding 줄이 뒤따르면 덮인다
            continue
        m = finding.match(line)
        if m and current and labels.get(current) == "pass":
            labels[current] = m.group(1)
    return labels


def _labels_from_previous(outdir: Path) -> dict[str, str]:
    """이전 회차 덤프가 있으면 probe 별 라벨을 층화 기준으로 재사용한다(없으면 빈 dict)."""
    labels: dict[str, str] = {}
    for path in sorted(outdir.glob("run_*.json")):
        for rec in json.loads(path.read_text(encoding="utf-8")).get("records", []):
            labels.setdefault(rec["probe_id"], rec["labels"][0] if rec.get("labels") else "pass")
    return labels


def _stratum(probe: dict, labels: dict[str, str]) -> str:
    """층화 기준. 관측된 진단 라벨이 있으면 그걸 쓰고(가장 좋은 기준), 없으면 probe 속성."""
    pid = probe.get("probe_id", "")
    if labels.get(pid):
        return labels[pid]
    if probe.get("answer_exists") is False:
        return "no_answer"
    return probe.get("qtype") or ("gold" if probe.get("gold_spans") else "nogold")


def _stratified_sample(probes: list[dict], m: int, labels: dict[str, str],
                       seed: int) -> list[dict]:
    """층별 비례 배분(최대잔여) + 층 안에서는 시드 셔플. 앞에서 M개 자르지 않는다."""
    if m >= len(probes):
        return list(probes)
    rng = Random(seed)
    strata: dict[str, list[dict]] = {}
    for p in probes:
        strata.setdefault(_stratum(p, labels), []).append(p)
    total = len(probes)
    quota, remainder = {}, []
    for key, group in strata.items():
        exact = m * len(group) / total
        quota[key] = min(len(group), int(exact))
        remainder.append((exact - int(exact), key))
    for _, key in sorted(remainder, reverse=True):
        if sum(quota.values()) >= m:
            break
        if quota[key] < len(strata[key]):
            quota[key] += 1
    picked: list[dict] = []
    for key in sorted(strata):
        group = list(strata[key])
        rng.shuffle(group)
        picked.extend(group[:quota[key]])
    rng.shuffle(picked)
    return picked[:m]


def _confirm_cost(subset_size: int, args) -> None:
    """실측 회차를 돌리기 전 예상 비용을 보이고 확인받는다.

    이 가드가 있는 이유: 이 스크립트의 무거운 경로(반복 실행)와 공짜 경로(--aggregate-only)가
    같은 명령어라, 인자 하나가 비면 조용히 유료 경로로 떨어진다. 실제로 그렇게 한 번 돌았다."""
    if args.dry_run:
        return
    estimate = subset_size * args.repeat * args.cost_per_probe
    print(f"\n[σ] 예상 비용: probe {subset_size} × {args.repeat}회 × "
          f"${args.cost_per_probe:.4f} ≈ ${estimate:.2f} (인덱스 임베딩 별도)", flush=True)
    if args.yes:
        return
    try:
        answer = input("[σ] 진행할까? [y/N] ").strip().lower()
    except (EOFError, OSError):
        # 입력을 못 읽는 실행(파이프·비대화형)에서 '확인 없이 과금'으로 새지 않게 막는다.
        raise SystemExit("[σ] 입력을 읽을 수 없다 - 진행하려면 --yes 를 붙일 것")
    if answer not in ("y", "yes"):
        raise SystemExit("[σ] 중단")


def _spawn(argv: list[str]) -> None:
    result = subprocess.run([sys.executable, os.path.abspath(__file__), *argv])
    if result.returncode != 0:
        raise SystemExit(f"[σ] 서브프로세스 실패(exit {result.returncode}): {' '.join(argv)}")


def _digest(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


# ── 집계 ─────────────────────────────────────────────────────────

class _StubProbe:
    __slots__ = ("ground_truth", "answer_exists")

    def __init__(self, rec: dict):
        self.ground_truth = "gold" if rec["has_gold"] else None
        self.answer_exists = rec["answer_exists"]


class _StubRecord:
    """scoring 이 읽는 필드만 가진 가짜 record.

    점수식을 여기서 다시 구현하지 않는 이유: 조화평균이든 성분 구성이든 scoring.py 가
    바뀌면 이 스크립트의 σ 도 같이 바뀌어야 한다. 진짜 함수를 그대로 부르면 그 동기화가
    공짜다(COMPONENTS 레지스트리에 성분이 추가돼도 손댈 게 없다)."""
    __slots__ = ("probe", "ragas", "recall_at_k", "answer_score", "retrieval_axis", "findings")

    def __init__(self, rec: dict):
        self.probe = _StubProbe(rec)
        self.ragas = {k: v for k, v in rec["ragas"].items() if v is not None}
        self.recall_at_k = rec["recall_at_k"]
        self.answer_score = rec["answer_score"]
        self.retrieval_axis = rec["retrieval_axis"]
        self.findings = [1] * rec["findings"]   # 유무만 본다(_probe_reliability)


def _composite(stubs: list) -> float | None:
    """정규화 composite(0~1). production 과 같은 성분·같은 결합식."""
    from agents.eval import scoring

    values = [v for v in (fn(stubs) for _, _, fn in scoring.COMPONENTS) if v is not None]
    return scoring.combine(values) if values else None


def _quantized(value: float) -> float:
    """production 이 judge 에 넘기는 값. compute_composite 이 round(x*100) 으로 정수화하고
    history._read_score 가 100 으로 나눈다 - 판정이 보는 눈금은 0.01 이다."""
    return round(value * 100) / 100


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def _to_grid(value: float) -> float:
    """판정 눈금(0.01)에 맞춰 위로 올린다. 그보다 촘촘한 마진은 의미가 없다."""
    return max(0.01, round(value + 0.004999, 2))


def aggregate(outdir: Path, args) -> dict:
    runs = []
    for path in sorted(outdir.glob("run_*.json")):
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    if len(runs) < 2:
        raise SystemExit(f"[σ] 회차가 {len(runs)}개다 - 최소 2회 필요")

    # 모든 회차에 공통으로 있는 probe 만 쓴다(회차마다 빠진 probe 가 있으면 비교가 어긋난다).
    common = set(r["probe_id"] for r in runs[0]["records"])
    for run in runs[1:]:
        common &= set(r["probe_id"] for r in run["records"])
    dropped = len(runs[0]["records"]) - len(common)
    obs: dict[str, list[dict]] = {pid: [] for pid in sorted(common)}
    for run in runs:
        for rec in run["records"]:
            if rec["probe_id"] in common:
                obs[rec["probe_id"]].append(rec)

    pids = sorted(obs)
    stubs_cache = {pid: [_StubRecord(rec) for rec in obs[pid]] for pid in pids}

    # ① 실측: 회차별 composite(부분표집 M개 기준). 가공 없는 관측치다.
    measured = [_composite([stubs_cache[pid][i] for pid in pids]) for i in range(len(runs))]
    measured = [v for v in measured if v is not None]

    # ② 몬테카를로: 목표 N 으로 환산.
    #    2단계인 이유 - probe 를 매 실행 새로 뽑으면 'probe 가 달라서 생기는 차이'까지
    #    run-to-run 노이즈에 섞인다. 실제로는 매 실행 같은 probe 를 쓰므로, 먼저 가상의
    #    N-probe 세트를 고정하고(바깥 루프) 그 세트 안에서만 회차를 다시 뽑는다(안쪽 루프).
    meta = {}
    meta_path = outdir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # 환산 목표는 **운영 probe 수**다. 부분표집한 M 을 기본값으로 쓰면 "25개짜리 실행의 σ"가
    # 나와서, 100개로 도는 실제 판정에는 과대평가된 마진을 물리게 된다.
    target_n = args.target_n or meta.get("pool_size") or len(pids)
    rng = Random(args.seed)
    deltas: list[float] = []
    q_deltas: list[float] = []
    set_sigmas: list[float] = []
    best_of: dict[int, list[float]] = {t: [] for t in BEST_OF_T}
    for _ in range(args.mc_sets):
        chosen = [rng.choice(pids) for _ in range(target_n)]
        pools = [stubs_cache[pid] for pid in chosen]

        def _one_run() -> float | None:
            return _composite([rng.choice(pool) for pool in pools])

        values = []
        for _ in range(args.mc_pairs):
            a, b = _one_run(), _one_run()
            if a is None or b is None:
                continue
            values.extend((a, b))
            deltas.append(a - b)
            q_deltas.append(_quantized(a) - _quantized(b))
            # sweep 경로: 같은 config 로 만든 후보 T개 중 최고 vs baseline.
            # 전부 baseline 과 같은 config 이므로 진짜 개선은 0 이고, 여기서 나오는
            # 상승폭은 전부 노이즈다 - 그 분포가 sweep 이 통과시키면 안 될 값의 분포다.
            candidates = [a] + [v for v in (_one_run() for _ in range(max(BEST_OF_T) - 1))
                                if v is not None]
            for t in BEST_OF_T:
                if len(candidates) >= t:
                    best_of[t].append(max(candidates[:t]) - b)
        if len(values) > 1:
            set_sigmas.append(statistics.stdev(values))
    if not deltas:
        raise SystemExit("[σ] composite 을 하나도 계산하지 못했다 - 덤프를 확인할 것")

    sigma_run = statistics.median(set_sigmas) if set_sigmas else 0.0
    sigma_delta = statistics.stdev(deltas)
    p95 = _percentile([abs(d) for d in deltas], 0.95)
    q_p95 = _percentile([abs(d) for d in q_deltas], 0.95)
    raw_margin = max(args.k * sigma_delta, p95)
    # 눈금(0.01)에 올린 값만 보면 후보 수에 따른 차이가 반올림에 묻힌다 - 원값을 같이 남긴다.
    sweep = {
        str(t): {"raw": round(_percentile(best_of[t], 0.95), 5),
                 "grid": _to_grid(_percentile(best_of[t], 0.95))}
        for t in BEST_OF_T if best_of[t]
    }

    report = {
        "measured": {
            "repeats": len(runs),
            "probes_used": len(pids),
            "probes_dropped": dropped,
            "composite_per_run": [round(v, 5) for v in measured],
            "spread": round(max(measured) - min(measured), 5) if measured else None,
            "display_total_per_run": [r.get("composite_total") for r in runs],
        },
        "extrapolated": {
            "target_n": target_n,
            "sigma_run": round(sigma_run, 5),
            "sigma_delta": round(sigma_delta, 5),
            "p95_abs_delta": round(p95, 5),
            "p95_abs_delta_quantized": round(q_p95, 5),
        },
        "margin": {
            "k": args.k,
            "raw": round(raw_margin, 5),
            "suggested": _to_grid(raw_margin),          # judge(쌍대비교)용
            "suggested_display": round(_to_grid(raw_margin) * 100, 1),
            "sweep_best_of_t": sweep,                    # internal sweep(후보 T개)용
            "current": 0.02,
        },
        "conditions": {
            "eval_mode": runs[0].get("eval_mode"),
            "judge_model": runs[0].get("judge_model"),
            "index_config": runs[0].get("index_config"),
            "source_type": os.getenv("SOURCE_TYPE", "file"),
            "source_url": os.getenv("SOURCE_URL", ""),
            "measured_at": time.strftime("%Y-%m-%d"),
            "subset_size": meta.get("subset_size"),
            "strata": meta.get("strata"),
            "dry_run": meta.get("dry_run"),
        },
        "seed": args.seed,
    }
    (outdir / "sigma_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_report(report, outdir)
    return report


def _print_report(rep: dict, outdir: Path) -> None:
    m, e, g = rep["measured"], rep["extrapolated"], rep["margin"]
    print("\n" + "=" * 60)
    print("  종합점수 노이즈 측정 결과 (#102)")
    print("=" * 60)
    print(f"  실측    : {m['repeats']}회 반복 · probe {m['probes_used']}개"
          + (f" (회차 간 불일치로 {m['probes_dropped']}개 제외)" if m["probes_dropped"] else ""))
    print(f"            회차별 composite {[f'{v:.4f}' for v in m['composite_per_run']]}")
    print(f"            표시 종합점수    {m['display_total_per_run']}")
    print(f"            관측 편차(최대-최소) {m['spread']}")
    print(f"\n  환산    : probe {e['target_n']}개 기준(몬테카를로)")
    print(f"            σ (한 번 측정)        {e['sigma_run']:.4f}")
    print(f"            σ_Δ (두 측정의 차이)  {e['sigma_delta']:.4f}")
    print(f"            |Δ| 95 분위          {e['p95_abs_delta']:.4f}"
          f"  (판정 눈금 반영 {e['p95_abs_delta_quantized']:.4f})")
    print(f"\n  마진    : judge(1:1 비교)  k={g['k']} → max(k·σ_Δ, P95) = {g['raw']:.4f}"
          f" → 제안 {g['suggested']:.2f} (표시 {g['suggested_display']}점)")
    if g["sweep_best_of_t"]:
        detail = " · ".join(f"후보 {t}개 → {v['raw']:.4f}({v['grid']:.2f})"
                            for t, v in g["sweep_best_of_t"].items())
        print(f"            sweep(후보 중 최고) {detail}")
        print(f"            ↑ 후보가 많을수록 최고값이 위로 치우친다(승자의 저주). "
              f"두 경로가 한 상수를 공유한다면 더 큰 쪽이 기준이다")
    print(f"            현재 값 {g['current']:.2f}"
          f" → {'올려야 한다' if g['suggested'] > g['current'] else '유지해도 된다'}")
    print(f"\n  ⚠ 이 값은 위 조건(코퍼스·QA셋·probe 수·judge 모델)에 딸린 값이다.")
    print(f"  상세: {outdir / 'sigma_report.json'}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════

def main() -> int:
    force_utf8_stdio()
    args = _build_parser().parse_args()

    # 자식 프로세스 경로 - .env 는 여기서만 읽는다(부모가 미리 읽으면 회차마다 다른 env 가 샌다).
    if args._build_pool or args._worker >= 0:
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except ImportError:
            pass
        for item in args.overrides:
            key, _, value = item.partition("=")
            os.environ[key.strip()] = value.strip()
        applied = apply_embedding_args(args)
        if applied:
            print(f"[σ] 임베딩 설정: {applied}", flush=True)
        return _build_pool(args) if args._build_pool else _worker(args)

    if args.aggregate_only:
        aggregate(Path(args.aggregate_only), args)
        return 0

    outdir = Path(args.outdir or DEFAULT_OUT_ROOT / time.strftime("%Y%m%d_%H%M%S"))
    outdir.mkdir(parents=True, exist_ok=True)
    passthrough: list[str] = []
    for flag, value in (("--embed", args.embed), ("--query-embed", args.query_embed),
                        ("--rerank", args.rerank)):
        if value:
            passthrough += [flag, value]
    for item in args.overrides:
        passthrough += ["--set", item]
    if args.dry_run:
        passthrough.append("--dry-run")

    # 비용 확인은 pool 생성(인덱스 구축=임베딩 과금)보다 **먼저** 한다.
    _confirm_cost(args.probes, args)

    # ① probe pool
    pool_path = Path(args.pool) if args.pool else outdir / POOL_NAME
    if not pool_path.exists():
        print("[σ] probe pool 생성 (LLM 없이 로드+gold 재동기화)", flush=True)
        _spawn(["--_build-pool", "--outdir", str(outdir), *passthrough])
    pool = _load_probe_file(pool_path)

    # ② 층화 부분표집
    labels = _labels_from_previous(outdir)
    if args.labels_from_log:
        for pid, label in _labels_from_log(Path(args.labels_from_log)).items():
            labels.setdefault(pid, label)
    subset = _stratified_sample(pool, args.probes, labels, args.seed)
    subset_path = outdir / SUBSET_NAME
    subset_path.write_text(
        json.dumps({"version": "sigma-subset", "probes": subset}, ensure_ascii=False),
        encoding="utf-8")
    strata: dict[str, int] = {}
    for p in subset:
        key = _stratum(p, labels)
        strata[key] = strata.get(key, 0) + 1
    (outdir / "meta.json").write_text(json.dumps({
        "pool_size": len(pool),          # = 운영 probe 수. 몬테카를로 환산의 기본 목표 N
        "subset_size": len(subset),
        "repeat": args.repeat,
        "strata": strata,
        "labels_source": ("log" if args.labels_from_log else
                          ("previous_runs" if labels else "probe_attrs")),
        "dry_run": args.dry_run,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[σ] pool {len(pool)}개 → 부분표집 {len(subset)}개 "
          f"(층: {', '.join(f'{k}×{v}' for k, v in sorted(strata.items()))})", flush=True)
    if not labels:
        print("[σ]   ⚠ 층화 기준이 probe 속성이다(라벨 미관측). 통과/실패 비율이 우연히 "
              "치우치면 σ 가 편향된다 - 기존 실행 로그가 있으면 --labels-from-log 를 줄 것",
              flush=True)

    # ③ 반복 실행 - 회차마다 별도 프로세스(캐시 복원 방지)
    baseline = _digest(subset_path)
    original = subset_path.read_bytes()
    for i in range(args.repeat):
        print(f"\n[σ] ── 회차 {i + 1}/{args.repeat} ──", flush=True)
        _spawn(["--_worker", str(i), "--_subset", str(subset_path),
                "--outdir", str(outdir), *passthrough])
        # Eval 은 bad_gold 로 확정된 probe 를 재생성해 **probe 파일에 덮어쓴다**
        # (_maybe_regenerate_bad_gold → save_probes). 그대로 두면 2회차부터 다른
        # 질문을 재는 셈이라 '같은 조건 반복'이 아니게 된다. 되돌리고 알린다.
        if _digest(subset_path) != baseline:
            subset_path.write_bytes(original)
            print(f"[σ] ⚠ 회차 {i + 1} 에서 probe 파일이 바뀌었다(bad_gold 재생성) → 되돌렸다. "
                  f"해당 probe 는 점수 집합에서 빠지므로 집계에서도 제외된다", flush=True)

    # ④ 집계
    aggregate(outdir, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
