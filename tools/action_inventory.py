"""
tools/action_inventory.py
rules.py 선언을 실제 실행 가능한 canonical action으로 집계한다.

[왜 필요한가]
  rules.py에 "선언된 것"과 파이프라인이 "실제로 실행할 수 있는 것"은 다르다.
  patch 키가 flat/canonical로 섞여 있고(canonicalize_path가 흡수), state mapping·
  capability·backend 지원을 모두 통과해야 실행된다. 그래서 처방 목록을 눈으로 세면
  틀린다.

  Action-Centered Optimizer 전환 계획서(ACTION_CENTERED_OPTIMIZER_IMPLEMENTATION_PLAN.md)
  는 이 집계 결과를 §2 현황표와 §6.1 catalog 등록 목록의 근거로 쓴다. 그런데 origin/main
  이 빠르게 움직여 손으로 유지한 표가 네 번 낡았다. 그래서 표를 문서에 박지 않고 이
  스크립트로 뽑는다.

[사용]
  python3 tools/action_inventory.py            # 사람이 읽는 표
  python3 tools/action_inventory.py --json     # 기계가 읽는 스냅샷(테스트 fixture 비교용)

[판정 기준]  **action_catalog 를 그대로 읽는다.**
  action key 조립·operation 유도·차단 사유·재색인·sweep 가능 여부는 전부 catalog 가
  단일 진실 원천이다. 이 스크립트가 같은 규칙을 다시 구현하면, 표가 낡는 문제를
  고치려다 **catalog 와 어긋날 자리를 새로 만든다**(PR #75 리뷰 지적).

  여기서 더하는 것은 catalog 가 들고 있지 않은 집계뿐이다 —
  어떤 라벨이 그 action 을 지지하는가, tier 는 무엇인가, 후보값이 몇 개인가,
  rules.py 가 flat 키로 선언했는가.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 한글 Windows 기본 콘솔(cp949)에서 '—' 같은 문자를 print 하면 UnicodeEncodeError 가
# 난다. graph.py 와 같은 방식으로 stdio 를 UTF-8 로 고정한다.
from core.console import force_utf8_stdio

force_utf8_stdio()

from agents.optimize import action_catalog
from agents.optimize import optimizer as _optimizer
from agents.optimize.rules import LABEL_TO_PRESCRIPTIONS


def collect() -> dict:
    """catalog 의 action 정의에 rules 쪽 지지 집계를 붙인다."""
    # 지지 라벨·tier·후보값·flat 키는 catalog 에 없다 — rules 선언에서 모은다.
    support: dict[str, dict] = {}
    for label, rule in LABEL_TO_PRESCRIPTIONS.items():
        if rule.get("status") != "ready":
            continue
        group = rule.get("group")
        for prescription in rule.get("prescriptions") or []:
            for raw_path, value in (prescription.get("patch") or {}).items():
                key = action_catalog.build_action_key(raw_path, value)
                entry = support.setdefault(key, {
                    "supporters": [],       # (group, label)
                    "values": set(),
                    "raw_paths": set(),
                })
                if (group, label) not in entry["supporters"]:
                    entry["supporters"].append((group, label))
                entry["values"].add(repr(value))
                entry["raw_paths"].add(raw_path)

    executable: list[dict] = []
    blocked: list[dict] = []
    for action in action_catalog.all_actions():
        entry = support.get(action.key, {"supporters": [], "values": set(), "raw_paths": set()})
        record = {
            "action_key": action.key,
            "canonical_path": action.canonical_path,
            "operation": action.operation,
            "tiers": sorted({g for g, _ in entry["supporters"] if g}),
            "supporting_labels": sorted(label for _, label in entry["supporters"]),
            "support_count": len(entry["supporters"]),
            "candidate_value_count": len(entry["values"]),
            "reindex_required": action.reindex_required,
            "flat_keys": sorted(p for p in entry["raw_paths"] if "." not in p),
        }
        if action.is_blocked:
            # catalog 는 사유와 상세를 나눠 들고 있다. 표시는 계획서 §2 형식을 따른다.
            record["blocked_reason"] = (
                f"{action.blocked_reason}({action.blocked_detail})"
                if action.blocked_reason == "capability_off"
                else action.blocked_reason
            )
            blocked.append(record)
        else:
            record["backend"] = (
                "internal" if action_catalog.is_sweepable(action.key) else "rules"
            )
            executable.append(record)

    statuses: dict[str, int] = {}
    for rule in LABEL_TO_PRESCRIPTIONS.values():
        status = rule.get("status")
        statuses[status] = statuses.get(status, 0) + 1

    return {
        "label_total": len(LABEL_TO_PRESCRIPTIONS),
        "label_status": dict(sorted(statuses.items())),
        "executable_count": len(executable),
        "shared_count": sum(1 for a in executable if a["support_count"] > 1),
        "blocked_count": len(blocked),
        "sweep_axes": sorted(_optimizer.BACKEND_SUPPORTED_PATHS.get("internal", set())),
        "executable": executable,
        "blocked": blocked,
    }


def _competing_axes(executable: list[dict]) -> dict[str, list[str]]:
    """같은 canonical 축에 둘 이상의 action이 있는 경우(경쟁 가능성)."""
    by_axis: dict[str, list[str]] = {}
    for action in executable:
        by_axis.setdefault(action["canonical_path"], []).append(action["action_key"])
    return {axis: keys for axis, keys in sorted(by_axis.items()) if len(keys) > 1}


def render(snapshot: dict) -> str:
    lines: list[str] = []
    add = lines.append

    add(f"라벨 {snapshot['label_total']}개  {snapshot['label_status']}")
    add(
        f"실행 가능 action {snapshot['executable_count']}개 "
        f"(공유 {snapshot['shared_count']}개) / 차단 {snapshot['blocked_count']}개"
    )
    add("")
    add(f"{'action':46s} {'지지':>2s} {'tier':6s} {'재색인':4s} backend")
    add("-" * 82)
    for action in snapshot["executable"]:
        add(
            f"{action['action_key']:46s} "
            f"{action['support_count']:>2d} "
            f"{'·'.join(action['tiers']):6s} "
            f"{'Y' if action['reindex_required'] else '-':4s} "
            f"{action['backend']}"
            + (
                f"  (후보값 {action['candidate_value_count']}개)"
                if action["candidate_value_count"] > 1
                else ""
            )
        )

    add("")
    add("차단:")
    for action in snapshot["blocked"]:
        add(f"  {action['action_key']:44s} {action['blocked_reason']}")

    competing = _competing_axes(snapshot["executable"])
    if competing:
        add("")
        add("같은 축에 여러 action (경쟁 가능성 — 계획서 §4.4):")
        for axis, keys in competing.items():
            add(f"  {axis}: {', '.join(k.split(':')[-1] for k in keys)}")

    add("")
    add(f"internal sweep 축: {', '.join(snapshot['sweep_axes'])}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="기계가 읽는 스냅샷을 출력한다(테스트 fixture 비교용).",
    )
    args = parser.parse_args()

    snapshot = collect()
    if args.json:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render(snapshot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
