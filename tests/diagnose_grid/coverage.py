"""
tests/diagnose_grid/coverage.py
격자가 31개 라벨 중 몇 개를 덮는지 센다.

**왜 도구로 두나.** 격자의 값어치는 케이스 수가 아니라 "어느 라벨이 한 번도 검증된 적
없는가"다. 그건 손으로 세면 금방 낡는다 — 케이스를 추가할 때마다 다시 세야 하고, 그러다
안 세면 "많이 늘렸으니 괜찮겠지"로 넘어간다. 진척을 숫자로 남기려고 도구로 뽑는다.

정본은 LABELS.md 다. 진단 코드(diagnose.py)의 라벨 목록이 아니라 **사람이 케이스를 쓸 때
보는 표**를 기준으로 센다 — 코드에만 있고 사전에 없는 라벨은 애초에 케이스를 쓸 수 없고,
그 불일치 자체가 드러나야 한다(아래 '사전에 없는데 코드에 있는 라벨' 절).

    python -m tests.diagnose_grid.coverage
"""
from __future__ import annotations

import pathlib
import re

_HERE = pathlib.Path(__file__).resolve().parent
LABELS_MD = _HERE / "LABELS.md"

# LABELS.md 의 그룹 헤더와 라벨 행. 표 형식이 바뀌면 여기가 0 개를 세므로 조용히 통과하지
# 않도록 호출부에서 빈 결과를 오류로 다룬다.
_GROUP_RE = re.compile(r"^## ([A-D]) — (.+)$")
_LABEL_RE = re.compile(r"^\| `([a-z_]+)` \|")


def labels_by_group() -> dict[str, list[str]]:
    """LABELS.md 의 그룹 → 라벨 목록. 사전이 정본이다."""
    text = LABELS_MD.read_text(encoding="utf-8")
    groups: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        header = _GROUP_RE.match(line)
        if header:
            current = f"{header.group(1)} — {header.group(2)}"
            groups.setdefault(current, [])
            continue
        label = _LABEL_RE.match(line)
        if label and current:
            groups[current].append(label.group(1))
    return groups


def _labels_of(value) -> set[str]:
    """Case.expect / known_gap_labels 는 {슬롯: 라벨} 또는 라벨 목록으로 쓰인다."""
    if not value:
        return set()
    if isinstance(value, dict):
        return {v for v in value.values() if isinstance(v, str) and v}
    if isinstance(value, (list, tuple, set)):
        return {v for v in value if isinstance(v, str) and v}
    return {value} if isinstance(value, str) else set()


def covered_labels(cases) -> set[str]:
    """케이스가 기대하는 라벨 집합.

    known_gap 케이스도 센다 — 아직 진단이 못 내는 라벨이지만 **상황은 적혀 있고** 고쳐지면
    바로 검증되는 자리다. 덮인 것으로 세되 아래 리포트에서 따로 표시한다.
    """
    found: set[str] = set()
    for case in cases:
        found |= _labels_of(getattr(case, "expect", None))
        found |= _labels_of(getattr(case, "known_gap_labels", None))
    return found


def gap_labels(cases) -> set[str]:
    """known_gap 으로만 등장하는 라벨 — 케이스는 있는데 진단이 아직 못 내는 것."""
    gaps: set[str] = set()
    for case in cases:
        if getattr(case, "known_gap", None):
            gaps |= _labels_of(getattr(case, "expect", None))
    return gaps


def report(cases) -> str:
    groups = labels_by_group()
    if not groups:
        raise SystemExit(
            "LABELS.md 에서 라벨을 하나도 못 읽었습니다 — 표 형식이 바뀌었으면 "
            "coverage.py 의 정규식도 같이 고쳐야 합니다."
        )
    covered = covered_labels(cases)
    gaps = gap_labels(cases)
    total = sum(len(v) for v in groups.values())

    lines = [
        f"격자 케이스 {len(cases)}개 · 라벨 {len(covered)}/{total} 덮음"
        f"{f' (그중 known_gap {len(gaps)}개)' if gaps else ''}",
        "",
    ]
    for group, labels in groups.items():
        have = [x for x in labels if x in covered]
        lines.append(f"[{group}]  {len(have)}/{len(labels)}")
        for label in labels:
            if label in gaps:
                mark = "!"      # 케이스는 있는데 진단이 아직 못 냄
            elif label in covered:
                mark = "O"
            else:
                mark = "."      # 한 번도 검증된 적 없음
            lines.append(f"    {mark} {label}")
        lines.append("")

    unknown = covered - {x for v in groups.values() for x in v}
    if unknown:
        lines += [
            "사전(LABELS.md)에 없는데 케이스가 기대하는 라벨 — 둘 중 하나가 틀렸다:",
            *(f"    ? {x}" for x in sorted(unknown)),
            "",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    from tests.diagnose_grid.cases_g3 import CASES

    print(report(CASES))
