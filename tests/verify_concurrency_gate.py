# 머지 게이트: 병렬화(EVAL_LLM_CONCURRENCY) 결과 동일성 실증 스크립트.
#
# 배경: temperature=0 이어도 LLM 은 완벽 결정론이 아니라, "병렬 시 점수 불변"은
# 코드 구조만으론 증명되지 않는다. 이 스크립트는 같은 mock 코퍼스에 대해
#   A) 동시성 1 (probe 생성 포함, 워밍업 겸 기준)
#   B) 동시성 1 (probe 캐시 재사용 — 고유 비결정성 노이즈 플로어)
#   C) 동시성 4 (probe 캐시 재사용 — 병렬 효과 측정)
# 를 실행해 리포트 diff 와 LLM 호출 수를 비교한다.
#
# 판정 기준:
#   - B vs C 호출 수 동일  → 병렬화가 호출을 추가/누락하지 않음 (필수)
#   - A vs C 리포트 차이  ≤  A vs B 차이(노이즈 플로어)  → 병렬로 인한 추가 변동 없음
#
# 실행(실제 API 키·비용 필요, mock 5청크 + EVAL_TESTSET_SIZE 만큼의 probe):
#   EVAL_MODE=deep EVAL_ENABLE_LLM=1 EVAL_TESTSET_SIZE=10 python tests/verify_concurrency_gate.py
# 키가 없으면 안내 후 종료한다. eval_probes.json 을 건드리지 않도록 임시 폴더에서 돈다.

import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ.setdefault("EVAL_MODE", "deep")
os.environ.setdefault("EVAL_ENABLE_LLM", "1")
os.environ.setdefault("EVAL_TESTSET_SIZE", "10")

from agents.eval import llm_provider  # noqa: E402

if not llm_provider.has_key():
    print("[Gate] 활성 provider 의 API 키가 없어 실행 불가 — .env 에 키를 넣고 재실행하세요.")
    print("       (EVAL_LLM_PROVIDER=openai|gemini|github 에 맞는 키 필요)")
    sys.exit(1)

from core import llm_usage  # noqa: E402
from core.schema import Chunk  # noqa: E402
from core.state import AgentDoctorState  # noqa: E402
from agents.index.qdrant_store import embed  # noqa: E402
from agents.eval.agent import run  # noqa: E402

RAW = [
    ("doc_001_chunk_000", "재택근무는 주 2일까지 가능하며 팀장 승인 후 사용합니다."),
    ("doc_001_chunk_001", "재택근무 신청은 전날 오후 6시까지 슬랙으로 제출해야 합니다."),
    ("doc_002_chunk_000", "신입사원 온보딩 기간은 2주이며 첫 주는 교육입니다."),
    ("doc_002_chunk_001", "연차는 15일이고 반차는 4시간 기준입니다."),
    ("doc_002_chunk_002", "경조사 휴가는 별도 규정을 따르며 최대 5일입니다."),
]


def _report_dict(report) -> dict:
    d = asdict(report)
    d.pop("report_id", None)   # 실행마다 다른 식별자/시각은 비교에서 제외
    d.pop("created_at", None)
    return d


def _diff_paths(a, b, prefix="") -> list[str]:
    """두 dict/list 트리에서 값이 다른 경로 목록."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += _diff_paths(a.get(k), b.get(k), f"{prefix}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{prefix}(len {len(a)} vs {len(b)})"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _diff_paths(x, y, f"{prefix}[{i}]")
        return out
    return [] if a == b else [f"{prefix}: {a!r} vs {b!r}"]


def _run_once(label: str, concurrency: int, chunks) -> tuple[dict, dict]:
    os.environ["EVAL_LLM_CONCURRENCY"] = str(concurrency)
    before = dict(llm_usage._totals)
    state = AgentDoctorState()
    state.chunks = chunks
    result = run(state)
    if result.status == "error":
        print(f"[Gate] {label} 실행 오류: {result.error}")
        sys.exit(1)
    after = llm_usage._totals
    usage = {k: after[k] - before[k] for k in ("calls", "prompt", "output")}
    print(f"\n[Gate] {label}: 동시성={concurrency}, LLM {usage['calls']}회, "
          f"overall={result.report.overall_score}")
    return _report_dict(result.report), usage


def main() -> int:
    chunks = [
        Chunk(chunk_id=cid, doc_id=cid.rsplit("_chunk_", 1)[0], text=text, embedding=embed(text))
        for cid, text in RAW
    ]

    # eval_probes.json / output 산출물이 사용자 작업본을 덮지 않게 임시 폴더에서 실행
    workdir = tempfile.mkdtemp(prefix="concurrency_gate_")
    origin = os.getcwd()
    os.chdir(workdir)
    try:
        rep_a, use_a = _run_once("A(기준, 순차·probe 생성)", 1, chunks)
        rep_b, use_b = _run_once("B(노이즈 플로어, 순차·캐시)", 1, chunks)
        rep_c, use_c = _run_once("C(병렬·캐시)", 4, chunks)

        for name, rep in (("A", rep_a), ("B", rep_b), ("C", rep_c)):
            path = os.path.join(workdir, f"report_{name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rep, f, ensure_ascii=False, indent=2, default=str)

        noise = _diff_paths(rep_a, rep_b, "report")
        effect = _diff_paths(rep_a, rep_c, "report")

        print("\n" + "=" * 60)
        print(f"[Gate] A vs B 차이(고유 비결정성 노이즈): {len(noise)}곳")
        for line in noise[:20]:
            print(f"    {line}")
        print(f"[Gate] A vs C 차이(병렬 포함 변동):       {len(effect)}곳")
        for line in effect[:20]:
            print(f"    {line}")
        print(f"[Gate] 호출 수 — A:{use_a['calls']} (probe 생성 포함) / "
              f"B:{use_b['calls']} / C:{use_c['calls']}")
        print(f"[Gate] 리포트 JSON 저장: {workdir}")

        calls_ok = use_b["calls"] == use_c["calls"]
        effect_ok = len(effect) <= len(noise)
        print("\n[Gate] 판정:")
        print(f"  - B vs C 호출 수 동일(병렬이 호출을 추가/누락 안 함): {'PASS' if calls_ok else 'FAIL'}")
        print(f"  - 병렬 변동 ≤ 노이즈 플로어:                        {'PASS' if effect_ok else 'FAIL'}")
        return 0 if (calls_ok and effect_ok) else 1
    finally:
        os.chdir(origin)
        # 리포트 JSON 은 남긴다(수동 확인용) — probe 캐시 등 임시 부산물만 있는 폴더라
        # 필요 없으면 통째로 지워도 된다.


if __name__ == "__main__":
    sys.exit(main())
