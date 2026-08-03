"""
agents/optimize/action_aggregator.py
Finding 을 action 후보로 집계하고 순위를 매기는 계층.

[흐름]
  라벨별 Finding 묶음
    → ActionSupport   (한 라벨이 어떤 action 을 왜 지지하는가)
    → ActionCandidate (같은 실제 config 변경을 하나로 통합)
    → eligibility 필터 (실행 불가 action 을 점수 계산 전에 제외)
    → 충돌 정리 → 정렬 → 최상위 1개

[이 계층이 지키는 세 가지]
  1. **고유 probe 단위 투표.** Eval 은 Finding 을 probe 마다 만들므로 라벨 수를 세면
     같은 probe 가 여러 표가 된다. probe 하나는 action 하나에 최대 한 번 기여한다.
  2. **실행 가능성 우선.** 실행할 수 없는 action 이 표를 받으면 실행 가능한 action 이
     밀려난다. 점수를 매기기 전에 거른다.
  3. **A > C > B 인과.** 검색이 새는 상태에서 생성 처방을 적용하면 garbage-in 이므로,
     그룹 순서가 점수보다 먼저다. B그룹이 ready 로 승격돼 실제로 A 와 경쟁한다.

[진단하지 않는다]
  Finding.metadata 를 다시 해석하지 않는다. 후보값 계산은 candidate_values 소관이고,
  여기서는 그 결과를 모아 통합할 뿐이다.
"""
from __future__ import annotations

from typing import Any, Iterable

from core.schema import Finding
from core.state import AgentDoctorState
from agents.optimize import action_catalog, eligibility, rules
from agents.optimize.schemas import (
    ActionCandidate,
    ActionDefinition,
    ActionSupport,
)


# 그룹 1차 우선순위. 값이 작을수록 먼저 처리한다.
# D(데이터 문제)는 manual 이라 자동 처방 대상이 아니므로 실질 순서는 A > C > B.
GROUP_ORDER: dict[str, int] = {"A": 0, "C": 1, "B": 2, "D": 3}

# rules.py 의 diagnosis_confidence 가 전부 None 이라 쓰는 기본값.
# 실측 confidence 가 생산되면 이 폴백이 사라지고 점수가 실제로 갈리기 시작한다.
DEFAULT_CONFIDENCE = 1.0

# 반대 방향 action 중 하나를 고르려면 이만큼 앞서야 한다(계획서 §4.4).
# 상대 조건만 두면 3:2 가 33% 로 통과하는데, probe 1개 차이는 측정 노이즈에 묻힌다.
CONFLICT_MARGIN_RATIO = 0.20
CONFLICT_MARGIN_PROBES = 2


# ── 1. Finding → ActionSupport ────────────────────────────────────

def build_action_supports(
    grouped_findings: dict[str, list[Finding]],
    state: AgentDoctorState,
    *,
    search_space_for: Any = None,
) -> list[ActionSupport]:
    """라벨 묶음마다 그 라벨이 지지하는 action 의 support 를 만든다.

    search_space_for(label, findings, changes) 는 후보값 계산 함수다. planner 가
    candidate_values 기반 구현을 주입한다(이 모듈이 후보값 수학을 다시 만들지 않는다).
    """
    supports: list[ActionSupport] = []

    for label, findings in grouped_findings.items():
        rule = rules.get_rule(label)
        if not rule or rule.get("status") != "ready":
            continue

        probes = {p for f in findings for p in f.affected_probes}
        confidence = rule.get("diagnosis_confidence")
        confidence_source = "rule" if confidence is not None else "default"
        if confidence is None:
            confidence = DEFAULT_CONFIDENCE

        for prescription in rule.get("prescriptions") or []:
            changes = dict(prescription.get("patch") or {})
            if not changes:
                continue

            space: dict[str, list[Any]] = {}
            grounding: dict[str, Any] = {}
            if search_space_for is not None:
                space, grounding = search_space_for(label, findings, changes)

            for raw_path, value in changes.items():
                action_key = action_catalog.build_action_key(raw_path, value)
                definition = action_catalog.get_action(action_key)
                if definition is None:
                    continue

                values = _candidate_values_for(space, definition.canonical_path, value)
                supports.append(
                    ActionSupport(
                        action_key=action_key,
                        label=label,
                        group=rule.get("group"),
                        finding_ids=[f.finding_id for f in findings],
                        affected_probes=set(probes),
                        confidence=float(confidence),
                        confidence_source=confidence_source,
                        target_metrics=list(rule.get("target_metrics") or []),
                        candidate_values=values,
                        applies_when=dict(prescription.get("applies_when") or {}),
                        reason=findings[0].description if findings else "",
                        grounding_metadata=dict(grounding or {}),
                    )
                )
    return supports


def _candidate_values_for(
    space: dict[str, list[Any]],
    canonical_path: str,
    raw_value: Any,
) -> list[Any]:
    """계산된 search space 에서 이 경로의 후보를 꺼낸다. 없으면 원시 값을 그대로."""
    for key, values in (space or {}).items():
        if key == canonical_path:
            return list(values)
    return [] if raw_value in ("increase", "decrease") else [raw_value]


# ── 2. ActionSupport → ActionCandidate ────────────────────────────

def aggregate_action_candidates(
    supports: Iterable[ActionSupport],
    state: AgentDoctorState,
    *,
    backend: str = "rules",
    exclusions: set[str] | None = None,
) -> list[ActionCandidate]:
    """같은 action key 를 지지하는 support 를 하나의 후보로 통합한다.

    실행 가능성 판정은 여기서 하지 않는다(filter_ineligible_actions 가 한다).
    통합과 판정을 나눠야 "왜 제외됐는지"를 리포트에 남길 수 있다.
    """
    grouped: dict[str, list[ActionSupport]] = {}
    for support in supports:
        grouped.setdefault(support.action_key, []).append(support)

    candidates: list[ActionCandidate] = []
    for action_key in sorted(grouped):
        group = grouped[action_key]
        definition = action_catalog.get_action(action_key)
        if definition is None:
            continue

        labels: list[str] = []
        probes: set[str] = set()
        metrics: list[str] = []
        values: list[Any] = []
        provenance: dict[str, list[str]] = {}

        for support in group:
            if support.label not in labels:
                labels.append(support.label)
            probes |= support.affected_probes
            for metric in support.target_metrics:
                if metric not in metrics:
                    metrics.append(metric)
            for value in support.candidate_values:
                if value not in values:
                    values.append(value)
                provenance.setdefault(str(value), []).append(support.label)

        candidate = ActionCandidate(
            action_key=action_key,
            definition=definition,
            supports=list(group),
            supporting_labels=labels,
            supporting_probes=probes,
            search_space={definition.canonical_path: values} if values else {},
            target_metrics=metrics,
            metadata={"candidate_support": provenance},
        )
        candidate.score, candidate.score_breakdown = score_candidate(candidate)
        if exclusions and action_key in exclusions:
            candidate.status = "blocked"
            candidate.reason = "excluded"
        candidates.append(candidate)

    _annotate_opposition(candidates)
    return candidates


def score_candidate(candidate: ActionCandidate) -> tuple[float, dict[str, Any]]:
    """고유 probe 기반 가중 투표 (계획서 §4.1).

        probe_support(p) = 그 probe 에서 이 action 을 지지한 support 중 최대 confidence
        coverage_weight  = Σ probe_support(p)
        score            = coverage_weight / base_cost

    probe 단위로 한 번만 세는 것이 핵심이다. 라벨 수로 세면 taxonomy 를 잘게 쪼갤수록
    점수가 커지는 왜곡이 생긴다.
    """
    per_probe: dict[str, float] = {}
    for support in candidate.supports:
        for probe in support.affected_probes:
            previous = per_probe.get(probe, 0.0)
            if support.confidence > previous:
                per_probe[probe] = support.confidence

    coverage = sum(per_probe.values())
    cost = candidate.definition.base_cost or 1.0
    score = coverage / cost

    grounded = sum(1 for s in candidate.supports if s.is_grounded)
    sources = {s.confidence_source for s in candidate.supports}
    breakdown = {
        "supporting_label_count": len(candidate.supporting_labels),
        "supporting_probe_count": len(per_probe),
        "weighted_probe_support": round(coverage, 6),
        "grounded_support_count": grounded,
        "target_metric_count": len(candidate.target_metrics),
        "base_cost": cost,
        "cost_source": "reindex_flag",
        "confidence_source": "rule" if sources == {"rule"} else "default",
        "reindex_required": candidate.definition.reindex_required,
    }
    return score, breakdown


def _annotate_opposition(candidates: list[ActionCandidate]) -> None:
    """같은 config 축을 공유하는 다른 action 을 기록한다.

    기록만 하고 배제하지 않는다 — boolean 축은 현재 상태가 한쪽을 no-op 으로 만들어
    eligibility 단계에서 자동으로 사라지므로, 여기서 미리 충돌로 처리하면 안 된다.
    """
    by_axis: dict[str, list[ActionCandidate]] = {}
    for candidate in candidates:
        by_axis.setdefault(candidate.definition.canonical_path, []).append(candidate)

    for peers in by_axis.values():
        if len(peers) < 2:
            continue
        for candidate in peers:
            for peer in peers:
                if peer.action_key == candidate.action_key:
                    continue
                candidate.opposing_action_keys.append(peer.action_key)
                for label in peer.supporting_labels:
                    if label not in candidate.opposing_labels:
                        candidate.opposing_labels.append(label)


# ── 3. 실행 가능성 필터 ───────────────────────────────────────────

def filter_ineligible_actions(
    candidates: list[ActionCandidate],
    state: AgentDoctorState,
    *,
    backend: str = "rules",
    capabilities: dict[str, Any] | None = None,
    runtime_capabilities: dict[str, Any] | None = None,
    constraints: dict[str, Any] | None = None,
) -> tuple[list[ActionCandidate], list[ActionCandidate]]:
    """실행 가능한 후보와 제외된 후보를 나눠 반환한다.

    **점수 계산 전이 아니라 정렬 전에 호출한다.** 점수는 이미 계산돼 있어도 되지만,
    실행 불가 action 이 정렬에 들어가면 실행 가능한 action 을 밀어낸다.
    """
    eligible: list[ActionCandidate] = []
    rejected: list[ActionCandidate] = []

    for candidate in candidates:
        if candidate.status == "blocked":
            rejected.append(candidate)
            continue

        values = candidate.search_space.get(candidate.definition.canonical_path, [])
        verdict = eligibility.evaluate_action(
            candidate.definition,
            values,
            state.index_config,
            backend=backend,
            capabilities=capabilities,
            runtime_capabilities=runtime_capabilities,
            constraints=constraints,
        )
        if verdict.eligible:
            candidate.search_space = verdict.search_space
            candidate.metadata["eligibility"] = verdict.detail
            eligible.append(candidate)
        else:
            candidate.status = "blocked"
            candidate.reason = verdict.reason or "ineligible"
            candidate.metadata["eligibility"] = verdict.detail
            rejected.append(candidate)

    return eligible, rejected


# ── 4. 충돌 정리 ──────────────────────────────────────────────────

def resolve_action_conflicts(
    candidates: list[ActionCandidate],
) -> tuple[list[ActionCandidate], list[dict[str, Any]]]:
    """같은 축에 남은 반대 방향 action 을 정리한다 (계획서 §4.4).

    **eligibility 통과 후에 호출해야 한다.** 대부분의 충돌은 그 전에 사라진다 —
    boolean 축은 no-op 필터가, 방향 축은 근거값의 direction_conflict 가 한쪽을
    제거한다. 여기 남는 것은 근거 계산이 실패해 방향 키워드로 폴백한 경우다.

    판정: tier → grounded → 점수차(상대 20% 이상 그리고 절대 probe 2 이상).
    미달이면 그 축을 이번 방문에서 보류한다(왕복 방지).
    """
    by_axis: dict[str, list[ActionCandidate]] = {}
    for candidate in candidates:
        by_axis.setdefault(candidate.definition.canonical_path, []).append(candidate)

    kept: list[ActionCandidate] = []
    deferred: list[dict[str, Any]] = []

    for axis, peers in sorted(by_axis.items()):
        if len(peers) == 1:
            kept.append(peers[0])
            continue

        ranked = sorted(peers, key=_conflict_sort_key)
        top, runner_up = ranked[0], ranked[1]

        if _tier_of(top) != _tier_of(runner_up):
            kept.append(top)
            continue
        if _is_grounded(top) != _is_grounded(runner_up):
            kept.append(top)
            continue

        if _margin_satisfied(top, runner_up):
            kept.append(top)
            continue

        for peer in peers:
            peer.status = "conflicted"
            peer.reason = "conflict_margin_unmet"
        deferred.append({
            "axis": axis,
            "reason": "conflict_margin_unmet",
            "candidates": [
                {
                    "action_key": p.action_key,
                    "score": p.score,
                    "probe_count": len(p.supporting_probes),
                    "supporting_labels": list(p.supporting_labels),
                }
                for p in ranked
            ],
        })

    return kept, deferred


def _margin_satisfied(top: ActionCandidate, runner_up: ActionCandidate) -> bool:
    """우세 판정: 상대 20% 이상 **그리고** 절대 probe 2개 이상 차이."""
    if top.score <= 0:
        return False
    relative = (top.score - runner_up.score) / top.score
    absolute = len(top.supporting_probes) - len(runner_up.supporting_probes)
    return relative >= CONFLICT_MARGIN_RATIO and absolute >= CONFLICT_MARGIN_PROBES


def _tier_of(candidate: ActionCandidate) -> int:
    return GROUP_ORDER.get(candidate.causal_rank_group or "D", 99)


def _is_grounded(candidate: ActionCandidate) -> bool:
    return any(s.is_grounded for s in candidate.supports)


def _conflict_sort_key(candidate: ActionCandidate) -> tuple:
    """충돌 판정용 정렬 — tier → grounded → 점수 순 (계획서 §4.4).

    **grounded 가 점수보다 먼저다.** 반대 방향이 맞붙는 상황은 근거값 계산이 실패해
    방향 키워드로 폴백한 경우인데, 한쪽만 실측 근거가 있다면 probe 수가 적더라도
    그쪽이 맞을 가능성이 높다. 점수를 먼저 보면 추측이 실측을 이긴다.
    """
    return (
        _tier_of(candidate),
        0 if _is_grounded(candidate) else 1,
        -candidate.score,
        candidate.action_key,
    )


# ── 5. 정렬 ───────────────────────────────────────────────────────

def rank_action_candidates(
    candidates: list[ActionCandidate],
) -> list[ActionCandidate]:
    """최종 정렬 (계획서 §4.2).

        1. causal_rank (A, C, B)   ← 불변조건. 점수보다 먼저다
        2. score 내림차순
        3. grounded support 수 내림차순
        4. base_cost 오름차순
        5. action_key 사전순         ← 결정성 보장

    5번이 있어야 같은 입력이 항상 같은 선택을 낸다. 기존 planner 는 dict 삽입 순서에
    의존해 동점에서 결과가 흔들릴 수 있었다.
    """
    return sorted(
        candidates,
        key=lambda c: (
            _tier_of(c),
            -c.score,
            -sum(1 for s in c.supports if s.is_grounded),
            c.definition.base_cost,
            c.action_key,
        ),
    )


def select_action(candidates: list[ActionCandidate]) -> ActionCandidate | None:
    """정렬 후 최상위 하나. 후보가 없으면 None."""
    ranked = rank_action_candidates(candidates)
    return ranked[0] if ranked else None
