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

[판정 기준]  optimizer.py가 실행 직전에 쓰는 것과 같은 정책을 그대로 사용한다.
  1. canonicalize_path 로 정규화
  2. STATE_MAPPABLE_PATHS 통과      (config_mapper가 state로 바꿀 수 있는가)
  3. PATH_CAPABILITIES → DEFAULT_CAPABILITIES 통과  (파이프라인이 실제로 소비하는가)
  둘 중 하나라도 막히면 blocked이며, 사유를 구분해 기록한다.
    not_state_mappable : mapper 계약 부재      → 영구적, mapper+소비 노드 추가 필요
    capability_off     : 소비 경로는 있으나 미검증 → 조건부, capability 값만 바꾸면 열림
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

from agents.optimize import optimizer as _optimizer
from agents.optimize.config_mapper import canonicalize_path
from agents.optimize.rules import LABEL_TO_PRESCRIPTIONS


# patch 값 → action operation. 계획서 §3.2의 action key 규칙과 같은 분류다.
#   방향(increase/decrease)  : key에 방향을 담는다
#   boolean(enable/disable)  : 값은 현재 상태가 결정하므로 key에 담지 않는다
#   replace                  : 고정값 교체. 값은 후보로 넘긴다(같은 축의 여러 값이
#                              경쟁하면 지지 집합이 같아 영구 교착이 생긴다)
#   adjust                   : 방향·폭이 진단 실측으로 정해진다(shift_to_favored_channel)
_SYMBOLIC_ADJUST_PREFIX = "shift_to"


def _operation(value) -> str:
    if value in ("increase", "decrease"):
        return value
    if value is True:
        return "enable"
    if value is False:
        return "disable"
    if isinstance(value, str) and value.startswith(_SYMBOLIC_ADJUST_PREFIX):
        return "adjust"
    return "replace"


def _blocked_reason(path: str) -> str | None:
    """실행을 막는 사유. 없으면 None(=실행 가능)."""
    if path not in _optimizer.STATE_MAPPABLE_PATHS:
        return "not_state_mappable"
    capability = _optimizer.PATH_CAPABILITIES.get(path)
    if capability and not _optimizer.DEFAULT_CAPABILITIES.get(capability, False):
        return f"capability_off({capability})"
    return None


def collect() -> dict:
    """ready 라벨의 처방을 canonical action으로 집계한다."""
    actions: dict[str, dict] = {}

    for label, rule in LABEL_TO_PRESCRIPTIONS.items():
        if rule.get("status") != "ready":
            continue
        group = rule.get("group")
        for prescription in rule.get("prescriptions") or []:
            reindex = bool(prescription.get("reindex"))
            for raw_path, value in (prescription.get("patch") or {}).items():
                path = canonicalize_path(raw_path)
                key = f"{path}:{_operation(value)}"
                entry = actions.setdefault(key, {
                    "action_key": key,
                    "canonical_path": path,
                    "operation": _operation(value),
                    "supporters": [],       # (group, label)
                    "values": set(),
                    "reindex_required": reindex,
                    "raw_paths": set(),
                })
                if (group, label) not in entry["supporters"]:
                    entry["supporters"].append((group, label))
                entry["values"].add(repr(value))
                entry["raw_paths"].add(raw_path)

    executable: list[dict] = []
    blocked: list[dict] = []
    for key in sorted(actions):
        entry = actions[key]
        path = entry["canonical_path"]
        reason = _blocked_reason(path)
        record = {
            "action_key": key,
            "canonical_path": path,
            "operation": entry["operation"],
            "tiers": sorted({g for g, _ in entry["supporters"] if g}),
            "supporting_labels": sorted(label for _, label in entry["supporters"]),
            "support_count": len(entry["supporters"]),
            "candidate_value_count": len(entry["values"]),
            "reindex_required": entry["reindex_required"],
            "flat_keys": sorted(p for p in entry["raw_paths"] if "." not in p),
        }
        if reason:
            record["blocked_reason"] = reason
            blocked.append(record)
        else:
            record["backend"] = (
                "internal"
                if path in _optimizer.BACKEND_SUPPORTED_PATHS.get("internal", set())
                else "rules"
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
