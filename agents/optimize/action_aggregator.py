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

from core.schema import Finding, UNMEASURED_SIGNAL
from core.state import AgentDoctorState
from agents.optimize import action_catalog, eligibility, rules
from agents.optimize.schemas import (
    ActionAttemptKey,
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


# ── applies_when 신호 소비 스위치 — 현재 OFF (관측용 신호로만 유지) ──
# planner 에서 이관했다. 계획서 §3.3 "applies_when 을 만족하지 않는 support 는
# 생성하지 않는다"가 이 계층의 책임이기 때문이다.
#
# 신호 생산(Eval)·대조 로직·테스트는 모두 배선돼 있으나 소비는 의도적으로 꺼 둔다.
#   1) 신호가 고르는 유일한 처방 swap_embedding_model 은 embedding.model action 이
#      capability_off 로 차단돼 있어 분기가 config 적용까지 이어지지 않는다.
#   2) 임계값(rules.py TOPIC_CLUSTER_*_RATIO)이 캘리브레이션 전 임의값이고, 실측상
#      추정량 분산이 구간 폭보다 커 신호 없는 회차도 spread/concentrated 로 튄다.
#      소비를 켜면 그 노이즈가 비싼 재색인을 잘못 발동시킨다.
# 임베딩 교체 실행과 임계값 캘리브레이션이 준비된 뒤 True 로 켠다.
CONSUME_APPLIES_WHEN_SIGNAL = False


def _finding_signal(findings: list[Finding], key: str) -> Any:
    """findings metadata 에서 신호값 하나를 읽는다.

    같은 라벨의 findings 는 Eval 이 라벨 단위로 같은 신호를 실으므로 첫 값을 대표로
    쓴다. 없으면 None → 호출부가 '미측정 = 통과'로 처리한다.
    """
    for finding in findings:
        value = finding.metadata.get(key)
        if value is not None:
            return value
    return None


def _is_unmeasured(signal: Any) -> bool:
    """이 신호값이 '못 쟀다'인가 — 키 부재(None)와 측정 실패 sentinel 을 함께 본다.

    측정부가 실패를 알리는 방법이 둘이다: 키를 아예 안 싣거나(None), 재려다 실패했다는
    사실을 UNMEASURED_SIGNAL 로 명시하거나. 후자를 놓치면 '못 쟀다'가 '허용 리스트에
    없는 값'으로 읽혀 처방이 **탈락**한다 — 미측정이 통과여야 하는 것과 정반대다.
    """
    return signal is None or signal == UNMEASURED_SIGNAL


def _prescription_applies(prescription: dict, findings: list[Finding]) -> bool:
    """처방의 applies_when 조건을 finding metadata 와 대조한다.

    계약(rules.py 주석):
      - applies_when 이 없으면            → 항상 적용(신호 무관 처방)
      - 신호 키는 있는데 값이 없으면       → 적용(미측정 = 순차 fallback)
      - 값이 있으면 허용 리스트 membership → 포함될 때만 적용
    키가 여러 개면 전부(AND) 만족해야 한다.

    '미측정'은 값의 부재(None)뿐 아니라 UNMEASURED_SIGNAL 도 포함한다(_is_unmeasured).
    측정부는 '쟀는데 해당 없음'(예: topic_cluster "none")과 '아예 못 쟀음'을 다른 값으로
    구분해 보내는데, 후자를 일반 값처럼 membership 검사에 넣으면 어느 허용 리스트에도
    없어 처방이 탈락한다 — 근거가 없을수록 더 많이 걸러내는 셈이라 계약과 반대다.
    """
    for key, allowed in (prescription.get("applies_when") or {}).items():
        signal = _finding_signal(findings, key)
        if _is_unmeasured(signal):
            continue                    # 미측정 → 이 조건은 통과
        if signal not in allowed:
            return False
    return True


def _applicable_prescriptions(
    rule: dict,
    findings: list[Finding],
) -> list[dict]:
    """신호 조건을 통과한 처방만. 전부 걸러지면 조건을 완화한다.

    **완화가 핵심이다.** 신호는 처방 순서를 '선호'하게 만들 뿐 라벨을 통째로 막아선
    안 된다. 걸러서 0개가 되면 그 라벨은 이번 진단에서 아무 action 도 지지하지 못하고
    사라지는데, 임계값이 아직 캘리브레이션 전 임의값이라 더욱 — 신호 때문에 고칠
    기회를 잃는 쪽보다 덜 맞는 처방이라도 시도하는 쪽이 안전하다.
    """
    prescriptions = rule.get("prescriptions") or []
    if not CONSUME_APPLIES_WHEN_SIGNAL:
        return list(prescriptions)
    preferred = [p for p in prescriptions if _prescription_applies(p, findings)]
    return preferred or list(prescriptions)


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

        for prescription in _applicable_prescriptions(rule, findings):
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
                        prescription_id=str(prescription.get("id") or ""),
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
        values = merge_candidate_values(group, definition.canonical_path, state)
        # 채택된 값에 대해서만 출처를 남긴다 — 버려진 폴백 값의 provenance 가 남으면
        # 리포트가 시험하지 않은 값을 후보로 설명한다.
        kept = set(map(str, values))
        for support in group:
            for value in support.candidate_values:
                if str(value) in kept:
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


def merge_candidate_values(
    supports: list[ActionSupport],
    canonical_path: str,
    state: AgentDoctorState,
) -> list[Any]:
    """여러 support 의 후보값을 하나로 합친다 (계획서 §4.5).

    두 가지를 지킨다. 둘 다 **입력 순서와 무관해야** 한다 — Eval 이 Finding 을 내는
    순서가 실행 경로를 바꾸면 같은 진단이 방문마다 다른 config 를 적용한다.

    1. **실측 근거가 있으면 추측값을 버린다.** 같은 축을 여러 라벨이 지지할 때 한쪽은
       gold span 으로 목표값을 계산하고(예: 400·600) 다른 쪽은 근거가 없어 방향
       키워드로 폴백한다(800÷2=400). 둘을 합치면 근거 없는 값이 sweep 에 섞여
       파이프라인 전체 재평가를 한 번 더 태운다. 실측이 있는데 추측을 시험할 이유가 없다.
    2. **support 를 라벨 사전순으로 훑는다.** 순서가 흔들리는 원인은 Eval 이 Finding 을
       내는 순서다 — 그건 라벨 **간** 순서다. 같은 라벨 안의 처방 순서는 rules.py 의
       정적 선언이라 이미 결정적이므로 건드리지 않는다(정렬은 stable).
       rules backend 는 첫 값을 그대로 적용하므로 이 순서가 곧 **적용 결과**다.

    ⚠️ support **안쪽** 값 순서는 건드리지 않는다. 계획서 §4.5 는 "baseline 변화량이
    작은 값부터"라고 적었지만, 그 순서는 candidate_values 가 도메인 논리로 이미
    정한 것이다(무릎 분석의 `[8, 12, 15]`, chunk 의 `path_fractions` 단계값).
    거리순으로 다시 세우면 그 의도가 뒤집힌다 — 여기서 고치려는 것은 support 간
    순서지 후보값 자체의 의미가 아니다.
    """
    grounded_values: list[Any] = []
    fallback_values: list[Any] = []
    for support in sorted(supports, key=lambda s: s.label):
        target = grounded_values if support.is_grounded else fallback_values
        for value in support.candidate_values:
            if value not in target:
                target.append(value)
    return grounded_values or fallback_values


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

REASON_BLOCKED_ATTEMPT = "blocked_action_attempt"


def _drop_blocked_attempts(
    candidate: ActionCandidate,
    baseline_config: dict[str, Any],
    runtime_capabilities: dict[str, Any] | None,
    blocked_attempts: set[Any] | None,
) -> list[Any]:
    """이미 품질 실패로 확인된 **정확한 전이**만 후보값에서 뺀다 (계획서 §5.1).

    action 통째로 막지 않는 것이 핵심이다. top_k 7 이 나빴다는 사실은 top_k 9 에 대해
    아무것도 말해주지 않고, baseline 이 바뀌면 7 조차 다시 볼 가치가 있다. 축을 닫으면
    "상류를 고친 뒤 하류를 다시 본다"는 A>C>B 설계 의도와 충돌한다.
    """
    from agents.optimize import history

    path = candidate.definition.canonical_path
    values = list(candidate.search_space.get(path, []))
    if not blocked_attempts or not values:
        return values

    baseline = history.baseline_fingerprint(
        baseline_config, candidate.action_key, runtime_capabilities
    )
    kept: list[Any] = []
    for value in values:
        attempt = ActionAttemptKey(
            action_key=candidate.action_key,
            baseline_fingerprint=baseline,
            candidate_fingerprint=history.candidate_fingerprint({path: value}),
        )
        if attempt not in blocked_attempts:
            kept.append(value)
    return kept


def filter_ineligible_actions(
    candidates: list[ActionCandidate],
    state: AgentDoctorState,
    *,
    backend: str = "rules",
    capabilities: dict[str, Any] | None = None,
    runtime_capabilities: dict[str, Any] | None = None,
    constraints: dict[str, Any] | None = None,
    blocked_attempts: set[Any] | None = None,
) -> tuple[list[ActionCandidate], list[ActionCandidate]]:
    """실행 가능한 후보와 제외된 후보를 나눠 반환한다.

    **점수 계산 전이 아니라 정렬 전에 호출한다.** 점수는 이미 계산돼 있어도 되지만,
    실행 불가 action 이 정렬에 들어가면 실행 가능한 action 을 밀어낸다.

    ``blocked_attempts`` 는 품질 실패로 확인된 ActionAttemptKey 집합이다. 후보값
    단위로만 걸러내고, 남는 값이 없을 때에 한해 action 을 제외한다.
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
        if not verdict.eligible:
            candidate.status = "blocked"
            candidate.reason = verdict.reason or "ineligible"
            candidate.metadata["eligibility"] = verdict.detail
            rejected.append(candidate)
            continue

        candidate.search_space = verdict.search_space
        candidate.metadata["eligibility"] = verdict.detail

        # eligibility 를 통과한 최종 후보값에서 차단된 전이를 뺀다. 순서가 중요하다 —
        # constraint·no-op 필터 전 값으로 fingerprint 를 만들면 실제 적용된 값과
        # 어긋나 차단이 빗나간다.
        path = candidate.definition.canonical_path
        remaining = _drop_blocked_attempts(
            candidate, state.index_config, runtime_capabilities, blocked_attempts
        )
        if not remaining:
            candidate.status = "blocked"
            candidate.reason = REASON_BLOCKED_ATTEMPT
            candidate.metadata["eligibility"] = verdict.detail
            rejected.append(candidate)
            continue
        candidate.search_space = {path: remaining}
        eligible.append(candidate)

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
