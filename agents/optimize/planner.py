"""
agents/optimize/planner.py
Optimize 모듈의 "결정" 계층.

[역할]
  Eval의 진단 리포트(state.report.findings)를 받아서
    1) 자동 처방 대상인지 3분류하고 (manual / actionable / 스킵)
    2) 최적화 흐름을 결정하고 (OptimizeDecision: 제안/적용/유지/수동)
    3) 같은 라벨의 finding 을 묶어 최상위 우선순위 라벨 하나를 골라 처방 후보를 만들어
    4) OptimizationRequest 로 묶어 optimizer 에게 넘긴다.

  Eval 은 Finding 을 probe 마다 따로 만든다(같은 원인이 probe 10개에서 터지면
  Finding 도 10개). 처방은 라벨 단위이므로 planner 는 먼저 라벨로 묶고,
  그 묶음 전체의 측정값으로 점수와 근거값을 계산한다.

  "무엇을, 어느 순서로" 까지가 planner 의 책임이다.
  "그 처방을 실제 config 값으로 바꾸는 일"은 config_mapper/adapters,
  "적용 후 좋아졌는지 판단·롤백"은 optimizer 결과 + history 소관이다.

[읽는 것]  state.report(Finding 목록), state.index_config, state.iteration, blacklist
[쓰는 것]  (state 를 직접 수정하지 않음. agent.py 가 반환값을 받아 반영한다.)

[MVP 결정 사항]  (planner 설계 시 확정, 나중에 재검토 가능)
  - 우선순위: 1차로 그룹(A>C>B) 정렬, 2차로 점수 정렬.
    D그룹은 manual 이라 자동처방 대상에서 빠진다.
  - 예비(confirmed=False) finding 은 자동 처방 대상에서 제외한다(_split_findings 참고).
  - 점수 = 빈도 × 진단신뢰도 ÷ 처방비용
      빈도   = 라벨 묶음이 영향을 준 probe 수 (최소 1)
      신뢰도 = rules.py diagnosis_confidence (None 이면 1.0 fallback)
      비용   = rules.py cost 가 None 이므로 reindex 로 유도 (런타임=1, 재색인=3)
  - 후보값은 진단 측정값에서 계산한다(_GROUNDED_VALUES). 근거가 없는 라벨만
    방향 키워드(×2/÷2) 추측으로 폴백한다.
  - target_metrics 는 rules.py 라벨의 target_metrics 를 읽어 실어 보낸다.
    guardrail 은 폐기 — 롤백은 전역 하한선 체크 + 점수 비교로 대체(history.py).
  - propose_only(제안만) 모드는 뼈대만. 현재 기본은 apply_optimize.
"""
from __future__ import annotations

import uuid
from typing import Any

from core.state import AgentDoctorState
from core.schema import Finding
from agents.optimize import rules
from agents.optimize.config_mapper import canonicalize_path
from agents.optimize.schemas import (
    ConfigPatch,
    OptimizationRequest,
    OptimizeDecision,
    PrescriptionCandidate,
)


# ── 상수 ──────────────────────────────────────────────────────────
# 그룹 1차 우선순위. 값이 작을수록 먼저 처리.
# D(데이터 문제)는 manual 이라 자동처방 대상이 아니므로 실질 순서는 A > C > B.
_GROUP_ORDER: dict[str, int] = {"A": 0, "C": 1, "B": 2, "D": 3}

# MVP fallback 상수 (rules.py 값이 None 일 때 사용)
_DEFAULT_CONFIDENCE = 1.0
_COST_RUNTIME = 1
_COST_REINDEX = 3

# search_space 변환용 상수
# 방향 키워드("increase"/"decrease")를 구체 숫자로 바꿀 때 쓰는 배수.
# increase = 현재값 × STEP, decrease = 현재값 ÷ STEP.
# min/max 안전 검사는 optimizer 소관이라 여기서는 현재값만 알면 된다(계층 분리).
# 근거값 계산이 가능한 라벨은 이 추측 대신 측정값을 쓴다(_GROUNDED_VALUES 참고).
_DIRECTION_STEP = 2

# 무릎(knee) 분석 임계값: "probe 1개를 더 커버하려고 파라미터를 이만큼 넘게
# 올리지는 않는다". 커버리지를 넓히면 노이즈·비용이 늘고 too_long_context /
# lost_in_the_middle 을 유발하므로, 넘치는 쪽에 실제 벌점이 있다.
# 이 값은 노이즈와 커버리지의 교환비에 대한 '추측'이므로 최종 답이 아니라 sweep
# 구간의 시작점을 고르는 데만 쓴다 — 맞았는지는 실측(overall_score)이 판정한다.
_MAX_STEP_PER_PROBE = 2.0

# sweep 후보 상한. 후보 1개당 파이프라인 전체 재평가(LLM 호출 다수)가 들어가므로
# 무릎에서 위로 이만큼만 시도한다. (OPTIMIZER_IMPLEMENTATION_PLAN.md §2.3)
_MAX_SWEEP_CANDIDATES = 3

# 방향 키워드를 계산할 때 baseline_config 에 해당 키가 없을 경우의 기본 현재값.
_DEFAULT_CURRENT: dict[str, int] = {
    "top_k": 5,
    "chunk_size": 512,
    "chunk_overlap": 50,
    "rerank_candidates": 20,
}


# ── 진입점 ────────────────────────────────────────────────────────

def plan(
    state: AgentDoctorState,
    blacklist: set[tuple[str, str]] | None = None,
) -> tuple[OptimizationRequest | None, OptimizeDecision]:
    """
    진단 리포트를 보고 (최적화 요청, 흐름 결정)을 만든다.

    Args:
        state: 공유 상태. state.report 가 있어야 한다.
        blacklist: 이미 실패해 재시도 금지된 (label, prescription_id) 조합.
            history.py 가 나중에 채워 넘긴다. None 이면 빈 집합.

    Returns:
        (request, decision)
          - decision.mode != "apply_optimize" 이면 request 는 None.
          - apply_optimize 이면 request 에 처방 후보가 담긴다.
    """
    blacklist = blacklist or set() # set은 blacklist가 None인 경우 필요

    if state.report is None:
        return None, OptimizeDecision(
            mode="use_current",
            status="skipped",
            requires_user_confirmation=False,
            next_route="serve",
            reason="진단 리포트가 없음 — 최적화 스킵",
        )

    manual, actionable = _split_findings(state.report.findings) #draft 부분은 사라짐
    decision = _decide_mode(state, actionable, manual)

    if decision.mode != "apply_optimize":
        return None, decision

    ranked = _rank_groups(_group_by_label(actionable))
    picked = _pick_top(ranked, blacklist)
    if picked is None:
        # 점수는 났지만 후보가 전부 블랙리스트 → 처방할 게 없음
        return None, OptimizeDecision(
            mode="use_current",
            status="skipped",
            requires_user_confirmation=False,
            next_route="serve",
            reason="처방 후보가 모두 블랙리스트에 걸림",
            manual_labels=decision.manual_labels,
        )

    label, findings, rule, _score_val = picked
    candidates = _build_candidates(
        label, findings, rule, blacklist, state.index_config
    )
    request = _build_request(label, findings, rule, candidates, ranked, state)
    decision.request_id = request.request_id
    return request, decision


# ── 1. 분류 ───────────────────────────────────────────────────────

def _split_findings(
    findings: list[Finding],
) -> tuple[list[Finding], list[Finding]]:
    """
    finding 을 (manual, actionable) 로 나눈다.
      - manual     : is_manual (D그룹) → 자동처방 불가, reporter 로 넘어감
      - actionable : is_actionable (ready + 처방 있음) + 확정(confirmed) → 점수 경쟁 대상
      - 나머지     : draft/unassigned/라벨없음/예비 → 지금은 실행 불가, 스킵

    예비(confirmed=False)는 Eval 이 자원(진단 모드/tier) 부족으로 확신하지 못한
    의심 원인이다. 처방 1회는 파이프라인 전체 재평가(LLM 호출 다수)를 유발하므로,
    확신 없는 진단에는 그 비용을 쓰지 않는다. 더 깊은 EVAL_MODE 에서 확정되면
    그때 처방 대상이 된다. (manual 은 비용을 쓰지 않으므로 예비여도 사용자에게
    알리기 위해 그대로 넘긴다.)
    """
    manual: list[Finding] = []
    actionable: list[Finding] = []
    for f in findings:
        label = f.label
        if not label:
            continue  # 세분화 라벨이 없으면 rules.py 매핑 불가
        if rules.is_manual(label):
            manual.append(f)
        elif rules.is_actionable(label) and f.confirmed:
            actionable.append(f)
        # draft/unassigned/예비 는 의도적으로 스킵
    return manual, actionable


# ── 2. 흐름 결정 (3-way / 4-way) ──────────────────────────────────

def _decide_mode(
    state: AgentDoctorState,
    actionable: list[Finding],
    manual: list[Finding],
) -> OptimizeDecision:
    """
    최적화 흐름을 결정한다.
      - 이미 임계값 통과            → use_current(already_optimal) → serve
      - 자동처방 없음 + manual 있음 → manual_required → serve (사람 개입)
      - 자동처방 없음 + manual 없음 → use_current(skipped) → serve
      - 자동처방 있음               → apply_optimize → index
    """
    manual_labels = [f.label for f in manual if f.label]

    report = state.report
    if report is not None and report.pass_threshold:
        return OptimizeDecision(
            mode="use_current",
            status="already_optimal",
            requires_user_confirmation=False,
            next_route="serve",
            reason="모든 임계값 달성 — 최적화 불필요",
            manual_labels=manual_labels,
        )

    if not actionable:
        if manual:
            return OptimizeDecision(
                mode="manual_required",
                status="manual_required",
                requires_user_confirmation=True,
                next_route="serve",
                reason="자동 처방 가능한 라벨 없음 — 사람 개입 필요(D그룹)",
                manual_labels=manual_labels,
            )
        return OptimizeDecision(
            mode="use_current",
            status="skipped",
            requires_user_confirmation=False,
            next_route="serve",
            reason="처방 가능한 finding 없음",
            manual_labels=manual_labels,
        )

    return OptimizeDecision(
        mode="apply_optimize",
        status="proposed",
        requires_user_confirmation=False,
        next_route="index",
        reason="처방 가능한 finding 존재 → 최적화 진행",
        manual_labels=manual_labels,
    )


# ── 3. 우선순위 점수 ──────────────────────────────────────────────

def _derive_cost(prescription: dict) -> int:
    """처방비용을 reindex 로 유도한다 (런타임=1, 재색인=3).
    rules.py 의 cost 가 아직 전부 None 이라 MVP 는 reindex 플래그로 계산."""
    return _COST_REINDEX if prescription.get("reindex") else _COST_RUNTIME


def _label_cost(rule: dict) -> int:
    """라벨 점수 계산용 비용. 처방 리스트의 첫(가장 가벼운) 처방 기준."""
    prescriptions = rule.get("prescriptions") or []
    if not prescriptions:
        return _COST_RUNTIME
    return _derive_cost(prescriptions[0])


def _group_by_label(actionable: list[Finding]) -> dict[str, list[Finding]]:
    """같은 라벨의 finding 을 묶는다.

    Eval 은 Finding 을 probe 마다 따로 만든다(affected_probes 는 항상 1개).
    같은 원인이 probe 10개에서 터지면 Finding 객체도 10개가 된다. 처방은 라벨
    단위이므로, 점수·근거값 계산 모두 이 묶음을 대상으로 해야 한다.
    """
    groups: dict[str, list[Finding]] = {}
    for f in actionable:
        groups.setdefault(f.label, []).append(f)
    return groups


def _score(findings: list[Finding], rule: dict) -> float:
    """우선순위점수 = 빈도 × 진단신뢰도 ÷ 처방비용.

    빈도는 이 라벨이 영향을 준 probe 수다. Finding 하나당 probe 하나이므로
    묶음 전체의 affected_probes 를 합쳐야 실제 빈도가 나온다(중복 제거).
    """
    probes = {p for f in findings for p in f.affected_probes}
    frequency = max(len(probes), 1)
    confidence = rule.get("diagnosis_confidence")
    if confidence is None:
        confidence = _DEFAULT_CONFIDENCE
    cost = _label_cost(rule)
    return (frequency * confidence) / cost


def _rank_groups(
    groups: dict[str, list[Finding]],
) -> list[tuple[str, list[Finding], dict, float]]:
    """라벨 묶음을 (그룹순서, 점수내림차순)으로 정렬.
    1차 키 = 그룹(A>C>B), 2차 키 = 점수 높은 순."""
    ranked: list[tuple[str, list[Finding], dict, float]] = []
    for label, findings in groups.items():
        rule = rules.get_rule(label)
        if not rule:
            continue
        ranked.append((label, findings, rule, _score(findings, rule)))
    ranked.sort(
        key=lambda item: (
            _GROUP_ORDER.get(item[2].get("group"), 99),
            -item[3],
        )
    )
    return ranked


# ── 4. 최상위 선택 + 블랙리스트 ───────────────────────────────────

def _available_prescriptions(
    rule: dict, label: str, blacklist: set[tuple[str, str]]
) -> list[dict]:
    """블랙리스트에 걸리지 않은 처방만 순서대로 반환."""
    return [
        p
        for p in rule.get("prescriptions", [])
        if (label, p["id"]) not in blacklist
    ]


def _pick_top(
    ranked: list[tuple[str, list[Finding], dict, float]],
    blacklist: set[tuple[str, str]],
) -> tuple[str, list[Finding], dict, float] | None:
    """정렬된 목록에서 '아직 시도할 처방이 남은' 최상위 라벨 묶음을 고른다."""
    for label, findings, rule, score in ranked:
        if _available_prescriptions(rule, label, blacklist):
            return label, findings, rule, score
    return None


# ── 5. 근거값 계산 (진단 측정 → 파라미터 목표값) ──────────────────
# 방향 키워드(×2/÷2)는 "대충 이만큼 늘리면 되겠지"라는 추측이다. 진단이 이미 잰
# 숫자가 있으면 그 값에서 목표를 계산한다 — 그게 이 설계의 핵심이다.
# (설계 배경: agents/optimize/PARAM_TUNING_PROPOSAL.md)

def _knee(required: list[int]) -> int:
    """'필요값' 목록에서 한계비용이 급등하기 직전 지점을 고른다.

    커버리지 곡선(= 경험적 CDF)을 훑으며 "probe 1개를 더 커버하는 데 드는 값
    상승분"을 보고, 그 비용이 _MAX_STEP_PER_PROBE 를 넘으면 멈춘다.

    예) 필요값 [3,4,4,5,6,7,8,12,15,100] →
        3→8 구간은 probe 1개당 값 1 안팎으로 싸다(7/10 커버).
        8→12 는 probe 1개에 값 4, 15→100 은 값 85 — 밑지는 장사라 멈춘다. → 8

    이상치(100)가 배제되는 이유가 "통계적으로 이상해서"가 아니라 "하나 더
    커버하려고 값을 크게 올리는 대가가 이득보다 커서"라는 점이 중요하다.
    평균/최댓값과 달리 이상치 하나에 끌려가지 않는다.
    """
    candidates = sorted(set(required))
    best = candidates[0]
    covered = sum(1 for r in required if r <= best)
    for nxt in candidates[1:]:
        gain = sum(1 for r in required if r <= nxt) - covered
        if gain <= 0:
            continue
        if (nxt - best) / gain > _MAX_STEP_PER_PROBE:
            break
        best, covered = nxt, covered + gain
    return best


def _knee_candidates(required: list[int]) -> list[int]:
    """무릎과 그 위 지점들을 sweep 후보로 낸다(싼 것부터, 최대 _MAX_SWEEP_CANDIDATES 개).

    무릎 아래는 후보로 내지 않는다 — 그 구간은 값을 1 올릴 때마다 probe 를 1개쯤
    회수하므로 무릎이 지배한다(커버리지는 더 높고 노이즈 차이는 작다).
    무릎 위는 _MAX_STEP_PER_PROBE 라는 '추측'이 밑진다고 본 구간이라, 그 추측이
    맞았는지 실측으로 확인할 가치가 있다 → sweep 대상.

    예) [3,4,4,5,6,7,8,12,15,100] → 무릎 8 → [8, 12, 15]
        (100 은 상한 초과라 optimizer 안전범위가 걸러낸다.)
    후보가 1개면 sweep 할 게 없어 optimizer 는 rules 로 1회만 검증한다.
    """
    values = sorted(set(required))
    knee = _knee(required)
    start = values.index(knee)
    return values[start:start + _MAX_SWEEP_CANDIDATES]


def _ground_enumeration_top_k(
    findings: list[Finding], baseline_config: dict
) -> list[int] | None:
    """나열형 누락(retrieval_incomplete_enumeration) → top_k 후보.

    근거: probe 마다 정답이 흩어져 있는 gold 청크 개수(= len(affected_chunks)).
    top_k 가 그 개수보다 작으면 구조적으로 다 가져올 수 없다 — 이건 추측이
    아니라 Eval 이 실측한 값이다(diagnose._finding 이 gold_chunk_ids 를 실어준다).
    """
    counts = [len(f.affected_chunks) for f in findings if f.affected_chunks]
    if not counts:
        return None  # 측정값 없음 → 방향 키워드 폴백
    return _knee_candidates(counts)


# 라벨 → 근거값 계산 함수. 여기 없는 라벨은 방향 키워드(추측)로 폴백한다.
# 계산 공식(집계·이상치 처리)은 planner 소유다. Eval 은 원시 측정치만 준다.
# TODO: gold_rank 가 Finding.metadata 로 들어오면 retrieval_low_rank /
#   retrieval_missing_gold 의 top_k 근거값도 여기 추가한다(PARAM_TUNING_PROPOSAL.md §8).
_GROUNDED_VALUES: dict[str, dict[str, Any]] = {
    "retrieval_incomplete_enumeration": {"top_k": _ground_enumeration_top_k},
}


def _grounded_search_space(
    label: str, findings: list[Finding], baseline_config: dict
) -> dict[str, list]:
    """이 라벨에서 측정값으로 계산 가능한 {config키: [후보값]} 을 만든다.
    계산할 근거가 없는 키는 담지 않는다(호출부가 방향 키워드로 폴백)."""
    space: dict[str, list] = {}
    for key, compute in _GROUNDED_VALUES.get(label, {}).items():
        values = compute(findings, baseline_config)
        if values:
            space[key] = values
    return space


# ── 6. 처방 후보 생성 (rules.py → PrescriptionCandidate) ──────────

def _concrete_values(
    key: str, patch_value: Any, baseline_config: dict
) -> list[Any] | None:
    """
    rules.py patch 값 하나를 optimizer 가 쓸 구체 후보값 리스트로 변환한다.
      - "increase"/"decrease" : 현재값 × 또는 ÷ _DIRECTION_STEP (정수)
      - 그 외(True, 숫자, "recursive_sentence" 등) : 그대로 [값]
    현재값이 숫자가 아니거나 없어 계산이 불가하면 None(→ 해당 키 제외).
    """
    if patch_value in ("increase", "decrease"):
        current = baseline_config.get(key, _DEFAULT_CURRENT.get(key))
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            return None  # 현재값을 숫자로 알 수 없으면 방향 계산 불가
        if patch_value == "increase":
            return [int(round(current * _DIRECTION_STEP))]
        return [int(round(current / _DIRECTION_STEP))]
    # 방향 키워드가 아니면 이미 구체값으로 본다.
    return [patch_value]


def _build_search_space(changes: dict, baseline_config: dict) -> dict[str, list]:
    """patch 변경 묶음을 방향 키워드 기준 search_space({경로: [구체값]})로 변환한다.
    이건 근거가 없을 때 쓰는 최후 폴백이다(_finding_search_space 의 우선순위 참고).
    변환 불가한 키(방향 계산 실패 등)는 제외한다."""
    space: dict[str, list] = {}
    for key, patch_value in changes.items():
        values = _concrete_values(key, patch_value, baseline_config)
        if values is not None:
            space[key] = values
    return space


def _supplied_candidates(findings: list[Finding]) -> dict[str, list]:
    """Eval 이 Finding.metadata 로 직접 넘긴 후보를 canonical 경로별로 모은다.

    합의된 임시 입력 키는 ``Finding.metadata['parameter_candidates']``다.
    값은 ``{canonical_path: [후보값...]}`` 형식이며, 이 계약이 정식 Eval
    필드로 승격되기 전까지 metadata 확장점으로 유지한다.
    같은 라벨의 finding 여러 개가 후보를 주면 먼저 나온 것을 쓴다.
    """
    supplied: dict[str, list] = {}
    for finding in findings:
        raw = finding.metadata.get("parameter_candidates")
        if not isinstance(raw, dict):
            continue
        for path, values in raw.items():
            if not isinstance(path, str):
                continue
            if isinstance(values, (list, tuple)) and values:
                supplied.setdefault(canonicalize_path(path), list(values))
    return supplied


def _finding_search_space(
    findings: list[Finding],
    changes: dict,
    baseline_config: dict,
    grounded: dict[str, list],
) -> dict[str, list]:
    """이 처방이 바꿀 키들의 최종 후보값을 정한다.

    우선순위:
      1. Eval 이 Finding.metadata 로 직접 넘긴 후보(_supplied_candidates)
      2. planner 가 진단 측정값에서 계산한 근거값(_grounded_search_space)
      3. rules.py 방향 키워드를 현재값 기준으로 환산한 추측(_build_search_space)

    1이 2보다 앞서는 이유는 Eval 이 planner 보다 많은 원시 신호를 갖고 있어,
    후보 산출을 Eval 쪽으로 옮기더라도 planner 를 고치지 않게 하기 위해서다.
    """
    fallback = _build_search_space(changes, baseline_config)
    supplied = _supplied_candidates(findings)

    resolved: dict[str, list] = {}
    for raw_path, fallback_values in fallback.items():
        path = canonicalize_path(raw_path)
        values = supplied.get(path) or grounded.get(raw_path) or grounded.get(path)
        resolved[path] = list(values) if values else list(fallback_values)
    return resolved


def _build_candidates(
    label: str,
    findings: list[Finding],
    rule: dict,
    blacklist: set[tuple[str, str]],
    baseline_config: dict,
) -> list[PrescriptionCandidate]:
    """
    rules.py 의 raw dict 처방들을 PrescriptionCandidate 객체로 변환한다.
    rules.py 에 적힌 순서(가벼운 것 먼저)를 그대로 유지한다.
    블랙리스트에 걸린 처방은 제외한다.
    """
    candidates: list[PrescriptionCandidate] = []
    target_metrics = list(rule.get("target_metrics", []))
    # 진단 측정값에서 계산한 목표값(있으면 방향 키워드 추측보다 우선).
    grounded = _grounded_search_space(label, findings, baseline_config)
    reason = findings[0].description if findings else ""
    for pres in _available_prescriptions(rule, label, blacklist):
        changes = dict(pres.get("patch", {}))
        patch = ConfigPatch(
            changes=changes,
            reindex_required=bool(pres.get("reindex")),
            description=f"{label} → {pres['id']}",
            metadata={"prescription_id": pres["id"]},
        )
        candidates.append(
            PrescriptionCandidate(
                id=pres["id"],
                failure_label=label,
                group=rule.get("group"),
                status=rule.get("status"),
                patch=patch,
                # optimizer 가 소비할 구체 후보값.
                # 우선순위: Finding.metadata 후보 > 근거값 계산 > 방향 키워드 추측.
                search_space=_finding_search_space(
                    findings, changes, baseline_config, grounded
                ),
                cost=float(_derive_cost(pres)),
                priority=0.0,          # 후보 개별 우선순위는 MVP 미사용
                target_metrics=list(target_metrics),  # rules.py 라벨의 target_metrics
                # 신호기반 택1(retrieval_semantic_mismatch 등). 없으면 빈 dict
                # → optimizer 가 순서대로 순차 시도(fallback).
                applies_when=dict(pres.get("applies_when", {})),
                reason=reason,
            )
        )
    return candidates


# ── 6. 요청서 포장 ────────────────────────────────────────────────

def _build_request(
    label: str,
    findings: list[Finding],
    rule: dict,
    candidates: list[PrescriptionCandidate],
    ranked: list[tuple[str, list[Finding], dict, float]],
    state: AgentDoctorState,
) -> OptimizationRequest:
    """선택된 라벨과 처방 후보를 OptimizationRequest 로 묶는다."""
    related = [lbl for lbl, _fs, _rule, _s in ranked if lbl != label]
    probes = {p for f in findings for p in f.affected_probes}
    selected_space = candidates[0].search_space if candidates else {}
    candidate_count = (
        len(next(iter(selected_space.values())))
        if len(selected_space) == 1
        and isinstance(next(iter(selected_space.values())), (list, tuple))
        else 1
    )
    use_internal = candidate_count > 1
    metadata: dict[str, Any] = {
        # 후보별 trade-off의 최종 심판은 항상 Eval의 단일 overall_score다.
        "primary_metric": "overall_score",
        "study_baseline_config": dict(state.index_config),
        "baseline_metrics": _report_metrics(state),
        "trial_results": [],
    }
    if use_internal and _space_path(selected_space) in {
        "chunker.chunk_size",
        "chunker.chunk_overlap",
    }:
        metadata["chunk_precheck_context"] = _chunk_precheck_context(state)
    return OptimizationRequest(
        request_id=str(uuid.uuid4()),
        iteration=state.iteration,
        baseline_config=dict(state.index_config),
        failure_label=label,
        related_failure_labels=related,
        candidates=candidates,
        search_space={
            path: list(values) if isinstance(values, (list, tuple)) else [values]
            for path, values in selected_space.items()
        },
        target_metrics=list(rule.get("target_metrics", [])),  # 라벨의 target_metrics
        target_profile="balanced",
        # 후보가 여러 개면 internal backend 가 방문에 걸쳐 sweep 한다.
        optimizer="internal" if use_internal else "rules",
        max_trials=candidate_count,
        reason=f"우선순위 최상위 라벨: {label} (probe {len(probes)}개 영향)",
        propose_only=False,
        metadata=metadata,
    )


def _space_path(search_space: dict[str, Any]) -> str | None:
    """단일 축 search space의 canonical 경로를 반환한다."""

    if len(search_space) != 1:
        return None
    path = next(iter(search_space))
    return canonicalize_path(path) if isinstance(path, str) else None


def _report_metrics(state: AgentDoctorState) -> dict[str, Any]:
    """Internal Adapter가 고정 baseline으로 사용할 Eval 지표를 복사한다."""

    if state.report is None:
        return {}
    metrics: dict[str, Any] = dict(state.report.ragas_scores)
    if state.report.overall_score is not None:
        metrics["overall_score"] = state.report.overall_score
    metrics["pass_threshold"] = state.report.pass_threshold
    return metrics


def _chunk_precheck_context(state: AgentDoctorState) -> dict[str, Any]:
    """Chunk 사전검증에 필요한 원문과 이미 준비된 gold span만 전달한다."""

    gold_spans = [
        dict(span)
        for probe in state.probes
        for span in probe.gold_spans
        if isinstance(span, dict)
    ]
    affected_doc_ids = {
        span.get("doc_id")
        for span in gold_spans
        if isinstance(span.get("doc_id"), str)
    }
    documents = [
        document
        for document in state.documents
        if not affected_doc_ids or document.doc_id in affected_doc_ids
    ]
    return {
        "documents": documents,
        "gold_spans": gold_spans,
        "chunk_strategy": state.index_config.get(
            "chunk_strategy",
            state.index_config.get("chunk_stage", "markdown_recursive"),
        ),
    }
