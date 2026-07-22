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

import math
import uuid
from typing import Any

from core.state import AgentDoctorState
from core.schema import Finding
from agents.optimize import rules
from agents.optimize import gate
from agents.optimize.config_mapper import canonicalize_path, get_current_value
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
    candidates = _build_candidates(label, findings, rule, blacklist, state)
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
    if gate.passes_report(report):
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
    if not required:
        raise ValueError("무릎 분석에는 하나 이상의 필요값이 있어야 합니다.")
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


def _probe_required_top_k(finding: Finding) -> int | None:
    """probe 하나가 gold 를 다 담으려면 필요한 최소 top_k.

    우선순위:
      1. gold 순위(Finding.metadata["gold_ranks"], Eval tier2 실측): 가장 늦게 나오는
         gold 의 순위 = 필요 top_k. 개수와 달리 흩어짐(multi-hop/나열형)을 반영한다.
         예) gold 가 3·13·20위면 필요 top_k 는 20(개수 3이 아니라).
         wide_n 밖 gold(순위 None)는 top_k 로 도달 불가라 제외 — 남은 gold 기준의
         '이 probe 에서 top_k 가 기여 가능한 최대치'를 쓴다.
      2. gold 개수(len(affected_chunks)): 순위 미측정(FAST 모드 등) 시 폴백.
         "top_k 가 gold 개수보다 작으면 구조적으로 다 못 가져온다"는 하한 근사.
    둘 다 없으면 None(→ 방향 키워드 폴백).
    """
    ranks = finding.metadata.get("gold_ranks")
    if isinstance(ranks, dict):
        present = [r for r in ranks.values() if isinstance(r, int)]
        if present:
            return max(present)
    if finding.affected_chunks:
        return len(finding.affected_chunks)
    return None


def _ground_top_k_from_gold(
    findings: list[Finding], state: AgentDoctorState, _direction: Any
) -> tuple[list[int] | None, dict[str, Any] | None]:
    """gold 를 다 담으려면 필요한 top_k 후보 — 검색 실패 라벨 공용.

    top_k 를 키우면 고쳐지는 라벨들(나열형 누락 / missing_gold)은 근거가 같다:
    "가장 늦게 나오는 gold 의 순위 = 필요 top_k". 그래서 계산을 한 함수로
    공유한다. probe 마다 필요 top_k(순위 실측 우선, 없으면 개수 근사)를 뽑아
    무릎 분석에 넣는다. Eval 이 실측한 값이므로 방향 키워드 추측(×2)이 아니다.
    (low_rank 는 제외 — 정석 처방이 리랭커라 top_k 근거값이 안 쓰인다. 아래
    _GROUNDED_VALUES 등록부 참고.)
    """
    required = [
        r for r in (_probe_required_top_k(f) for f in findings) if r is not None
    ]
    if not required:
        return None, None  # 측정값 없음 → 방향 키워드 폴백
    return _knee_candidates(required), None


def _valid_gold_spans(
    state: AgentDoctorState,
    findings: list[Finding],
) -> list[dict[str, Any]]:
    """유효한 exact span을 우선하고, 없을 때만 fallback span을 반환한다."""

    affected_probe_ids = {
        probe_id
        for finding in findings
        for probe_id in finding.affected_probes
        if isinstance(probe_id, str)
    }
    document_lengths = {
        document.doc_id: len(document.content) for document in state.documents
    }
    exact_spans: list[dict[str, Any]] = []
    fallback_spans: list[dict[str, Any]] = []
    exact_seen: set[tuple[str, int, int]] = set()
    fallback_seen: set[tuple[str, int, int]] = set()

    for probe in state.probes:
        if affected_probe_ids and probe.probe_id not in affected_probe_ids:
            continue
        if not affected_probe_ids and not probe.answer_exists:
            continue
        grounding = probe.metadata.get("span_grounding", {})
        if not isinstance(grounding, dict):
            grounding = {}
        raw_qualities = grounding.get("span_qualities")
        qualities = raw_qualities if isinstance(raw_qualities, list) else []
        status = grounding.get("status")

        for index, span in enumerate(probe.gold_spans):
            if not isinstance(span, dict):
                continue
            doc_id = span.get("doc_id")
            start = span.get("start")
            end = span.get("end")
            if (
                not isinstance(doc_id, str)
                or doc_id not in document_lengths
                or isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end <= start
                or end > document_lengths[doc_id]
            ):
                continue
            identity = (doc_id, start, end)
            quality = qualities[index] if index < len(qualities) else None
            if quality not in {"exact", "chunk_fallback"}:
                if status == "chunk_fallback" or status == "partial":
                    quality = "chunk_fallback"
                else:
                    # 사람이 넣은 taxonomy/gold span과 새 exact Probe는 기본적으로 신뢰한다.
                    quality = "exact"
            target = exact_spans if quality == "exact" else fallback_spans
            seen = exact_seen if quality == "exact" else fallback_seen
            if identity in seen:
                continue
            seen.add(identity)
            target.append({"doc_id": doc_id, "start": start, "end": end})
    return exact_spans or fallback_spans


def _percentile_nearest_rank(values: list[int], quantile: float) -> int:
    """외부 통계 의존성 없이 nearest-rank 백분위 값을 계산한다."""

    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _chunk_candidate_policy(
    state: AgentDoctorState,
) -> tuple[dict[str, Any] | None, str | None]:
    """상태에서 chunk 후보 정책을 읽고 계산에 안전한지 검증한다."""

    policy = state.index_config.get("chunk_candidate_policy")
    if not isinstance(policy, dict):
        return None, "chunk_candidate_policy가 dict가 아님"

    target_quantile = policy.get("target_quantile")
    margin_ratio = policy.get("margin_ratio")
    rounding_step = policy.get("rounding_step")
    path_fractions = policy.get("path_fractions")
    candidate_count = policy.get("candidate_count")
    min_span_count = policy.get("min_span_count")
    valid = (
        isinstance(target_quantile, (int, float))
        and not isinstance(target_quantile, bool)
        and 0 < target_quantile <= 1
        and isinstance(margin_ratio, (int, float))
        and not isinstance(margin_ratio, bool)
        and margin_ratio >= 0
        and isinstance(rounding_step, int)
        and not isinstance(rounding_step, bool)
        and rounding_step > 0
        and isinstance(path_fractions, list)
        and path_fractions
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0 < value <= 1
            for value in path_fractions
        )
        and isinstance(candidate_count, int)
        and not isinstance(candidate_count, bool)
        and candidate_count > 1
        and isinstance(min_span_count, int)
        and not isinstance(min_span_count, bool)
        and min_span_count > 0
    )
    if not valid:
        return None, "chunk_candidate_policy 값이 유효하지 않음"
    return {
        "target_quantile": float(target_quantile),
        "margin_ratio": float(margin_ratio),
        "rounding_step": rounding_step,
        "path_fractions": [float(value) for value in path_fractions],
        "candidate_count": candidate_count,
        "min_span_count": min_span_count,
    }, None


def _chunk_overlap_candidate_policy(
    state: AgentDoctorState,
) -> tuple[dict[str, Any] | None, str | None]:
    """상태에서 chunk_overlap 후보 정책을 읽고 안전 범위를 검증한다."""

    policy = state.index_config.get("chunk_overlap_candidate_policy")
    if not isinstance(policy, dict):
        return None, "chunk_overlap_candidate_policy가 dict가 아님"

    target_quantiles = policy.get("target_quantiles")
    rounding_step = policy.get("rounding_step")
    candidate_count = policy.get("candidate_count")
    min_crossing_span_count = policy.get("min_crossing_span_count")
    max_ratio = policy.get("max_ratio")
    max_overlap = policy.get("max_overlap")
    valid = (
        isinstance(target_quantiles, list)
        and target_quantiles
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0 < value <= 1
            for value in target_quantiles
        )
        and isinstance(rounding_step, int)
        and not isinstance(rounding_step, bool)
        and rounding_step > 0
        and isinstance(candidate_count, int)
        and not isinstance(candidate_count, bool)
        and candidate_count > 0
        and isinstance(min_crossing_span_count, int)
        and not isinstance(min_crossing_span_count, bool)
        and min_crossing_span_count > 0
        and isinstance(max_ratio, (int, float))
        and not isinstance(max_ratio, bool)
        and 0 < max_ratio <= 0.40
        and isinstance(max_overlap, int)
        and not isinstance(max_overlap, bool)
        and 0 < max_overlap <= 300
    )
    if not valid:
        return None, "chunk_overlap_candidate_policy 값이 유효하지 않음"
    return {
        "target_quantiles": [float(value) for value in target_quantiles],
        "rounding_step": rounding_step,
        "candidate_count": candidate_count,
        "min_crossing_span_count": min_crossing_span_count,
        "max_ratio": float(max_ratio),
        "max_overlap": max_overlap,
    }, None


def _round_to_step(value: float, step: int) -> int:
    """후보를 사람이 읽기 쉬운 단위로 반올림한다."""

    return max(step, int(round(value / step) * step))


def _ground_chunk_size_candidates(
    findings: list[Finding],
    state: AgentDoctorState,
    direction: Any,
) -> tuple[list[int] | None, dict[str, Any]]:
    """gold span 길이 P85와 현재값 사이에서 chunk_size 후보를 만든다."""

    spans = _valid_gold_spans(state, findings)
    if not spans:
        return None, {"status": "missing_gold_spans"}

    policy, error = _chunk_candidate_policy(state)
    if policy is None:
        return None, {"status": "invalid_policy", "reason": error}

    current = get_current_value(state.index_config, "chunker.chunk_size")
    if isinstance(current, bool) or not isinstance(current, (int, float)) or current <= 0:
        return None, {"status": "invalid_current_value"}
    if direction not in {"increase", "decrease"}:
        return None, {"status": "unsupported_direction", "direction": direction}

    lengths = [span["end"] - span["start"] for span in spans]
    if len(lengths) < policy["min_span_count"]:
        return None, {
            "status": "insufficient_spans",
            "source": "gold_spans",
            "span_count": len(lengths),
            "min_span_count": policy["min_span_count"],
        }
    p50 = _percentile_nearest_rank(lengths, 0.50)
    p85 = _percentile_nearest_rank(lengths, 0.85)
    p95 = _percentile_nearest_rank(lengths, 0.95)
    target_span = _percentile_nearest_rank(lengths, policy["target_quantile"])
    raw_target = target_span * (1 + policy["margin_ratio"])
    step = policy["rounding_step"]
    target = max(step, int(math.ceil(raw_target / step) * step))
    current_int = int(round(current))

    metadata: dict[str, Any] = {
        "status": "grounded",
        "source": "gold_spans",
        "span_count": len(lengths),
        "min": min(lengths),
        "p50": p50,
        "p85": p85,
        "p95": p95,
        "max": max(lengths),
        "target_quantile": policy["target_quantile"],
        "margin_ratio": policy["margin_ratio"],
        "current_chunk_size": current_int,
        "target_chunk_size": target,
        "direction": direction,
    }
    if (direction == "decrease" and target >= current_int) or (
        direction == "increase" and target <= current_int
    ):
        metadata["status"] = "direction_conflict"
        return None, metadata

    candidates: list[int] = []
    for fraction in policy["path_fractions"]:
        value = current_int + ((target - current_int) * fraction)
        rounded = _round_to_step(value, step)
        if direction == "decrease" and not target <= rounded < current_int:
            continue
        if direction == "increase" and not current_int < rounded <= target:
            continue
        if rounded not in candidates:
            candidates.append(rounded)
        if len(candidates) >= policy["candidate_count"]:
            break

    if len(candidates) < 2:
        metadata["status"] = "insufficient_candidates"
        return None, metadata
    metadata["generated_candidates"] = list(candidates)
    return candidates, metadata


def _chunk_positions_by_doc(
    state: AgentDoctorState,
) -> tuple[dict[str, list[tuple[int, int]]], int]:
    """현재 청크의 원문 좌표를 문서별로 정렬한다."""

    positions_by_doc: dict[str, list[tuple[int, int]]] = {}
    missing_position_count = 0
    for chunk in state.chunks:
        raw = chunk.char_span
        if raw is None and isinstance(chunk.metadata, dict):
            raw = chunk.metadata.get("char_span")
        if (
            not isinstance(raw, (list, tuple))
            or len(raw) != 2
            or isinstance(raw[0], bool)
            or isinstance(raw[1], bool)
            or not isinstance(raw[0], int)
            or not isinstance(raw[1], int)
            or raw[0] < 0
            or raw[1] <= raw[0]
        ):
            missing_position_count += 1
            continue
        positions_by_doc.setdefault(chunk.doc_id, []).append((raw[0], raw[1]))
    for positions in positions_by_doc.values():
        positions.sort()
    return positions_by_doc, missing_position_count


def _ground_chunk_overlap_candidates(
    findings: list[Finding],
    state: AgentDoctorState,
    direction: Any,
) -> tuple[list[int] | None, dict[str, Any]]:
    """경계에 걸린 gold span에서 필요한 총 chunk_overlap 후보를 계산한다.

    경계 ``b``를 기준으로 왼쪽 청크에 들어간 정답 길이 ``b - start``가
    다음 청크 시작점을 정답 시작점까지 당겨야 하는 최소 overlap이다. 정답의
    오른쪽 길이와 전체 길이도 함께 검사해 chunk_size 고정 상태에서 회복 가능한
    단일 경계 사례만 백분위 계산에 넣는다. 실제 회복 여부는 prescreener가 현재
    청커를 dry-run해 다시 검증한다.
    """

    spans = _valid_gold_spans(state, findings)
    if not spans:
        return None, {"status": "missing_gold_spans"}
    if direction != "increase":
        return None, {"status": "unsupported_direction", "direction": direction}

    policy, error = _chunk_overlap_candidate_policy(state)
    if policy is None:
        return None, {"status": "invalid_policy", "reason": error}

    current = get_current_value(state.index_config, "chunker.chunk_overlap")
    chunk_size = get_current_value(state.index_config, "chunker.chunk_size")
    if (
        isinstance(current, bool)
        or not isinstance(current, (int, float))
        or current < 0
        or isinstance(chunk_size, bool)
        or not isinstance(chunk_size, (int, float))
        or chunk_size <= 0
    ):
        return None, {"status": "invalid_current_value"}
    current_int = int(round(current))
    chunk_size_int = int(round(chunk_size))
    max_allowed = min(
        policy["max_overlap"],
        int(math.floor(chunk_size_int * policy["max_ratio"])),
        chunk_size_int - 1,
    )
    if max_allowed <= current_int:
        return None, {
            "status": "at_safe_limit",
            "current_chunk_overlap": current_int,
            "max_allowed_overlap": max_allowed,
        }

    positions_by_doc, missing_position_count = _chunk_positions_by_doc(state)
    required: list[int] = []
    right_needs: list[int] = []
    contained_count = 0
    irregular_count = 0
    span_too_long_count = 0
    limit_exceeded_count = 0
    geometry_conflict_count = 0

    for span in spans:
        start, end = span["start"], span["end"]
        positions = positions_by_doc.get(span["doc_id"], [])
        if any(c_start <= start and c_end >= end for c_start, c_end in positions):
            contained_count += 1
            continue
        if end - start > chunk_size_int:
            span_too_long_count += 1
            continue

        # 시작을 덮는 왼쪽 청크와 끝을 덮는 바로 다음 청크의 경계를 찾는다.
        pairs: list[tuple[int, int]] = []
        for index in range(len(positions) - 1):
            left_start, boundary = positions[index]
            right_start, right_end = positions[index + 1]
            if (
                left_start <= start < boundary < end
                and start < right_start < end <= right_end
            ):
                pairs.append((boundary - start, end - boundary))
        if not pairs:
            irregular_count += 1
            continue

        # 같은 span에 후보 경계가 여러 개면 가장 작은 안전 overlap을 택하고,
        # 실제 청커 사전검증이 그 선택이 맞는지 확인한다.
        left_need, right_need = min(pairs, key=lambda pair: pair[0])
        if left_need <= current_int:
            geometry_conflict_count += 1
            continue
        if left_need > max_allowed:
            limit_exceeded_count += 1
            continue
        required.append(left_need)
        right_needs.append(right_need)

    metadata: dict[str, Any] = {
        "status": "grounded",
        "source": "gold_span_boundary_geometry",
        "span_count": len(spans),
        "recoverable_crossing_count": len(required),
        "contained_count": contained_count,
        "irregular_or_multi_boundary_count": irregular_count,
        "span_too_long_count": span_too_long_count,
        "limit_exceeded_count": limit_exceeded_count,
        "geometry_conflict_count": geometry_conflict_count,
        "missing_chunk_position_count": missing_position_count,
        "current_chunk_overlap": current_int,
        "chunk_size": chunk_size_int,
        "max_allowed_overlap": max_allowed,
        "target_quantiles": list(policy["target_quantiles"]),
    }
    if len(required) < policy["min_crossing_span_count"]:
        metadata["status"] = (
            "no_recoverable_crossings" if not required else "insufficient_crossings"
        )
        metadata["min_crossing_span_count"] = policy["min_crossing_span_count"]
        return None, metadata

    p50 = _percentile_nearest_rank(required, 0.50)
    p85 = _percentile_nearest_rank(required, 0.85)
    p95 = _percentile_nearest_rank(required, 0.95)
    metadata.update({
        "min_required_overlap": min(required),
        "p50": p50,
        "p85": p85,
        "p95": p95,
        "max_required_overlap": max(required),
        "max_right_need": max(right_needs),
    })

    step = policy["rounding_step"]
    candidates: list[int] = []
    for quantile in policy["target_quantiles"]:
        raw_target = _percentile_nearest_rank(required, quantile)
        rounded = int(math.ceil(raw_target / step) * step)
        rounded = min(rounded, max_allowed)
        if current_int < rounded <= max_allowed and rounded not in candidates:
            candidates.append(rounded)

    # 표본이 적어 백분위들이 같은 값이면 바로 위 값을 함께 dry-run한다.
    # P95보다 조금 큰 값의 중복비용까지 비교해야 가장 작은 회복값을 고를 수 있다.
    anchor = max(candidates, default=current_int)
    while len(candidates) < policy["candidate_count"] and anchor + step <= max_allowed:
        anchor += step
        if anchor not in candidates:
            candidates.append(anchor)
    candidates = sorted(set(candidates))[: policy["candidate_count"]]
    if not candidates:
        metadata["status"] = "insufficient_candidates"
        return None, metadata
    metadata["generated_candidates"] = list(candidates)
    return candidates, metadata


# 라벨 → 근거값 계산 함수. 여기 없는 라벨은 방향 키워드(추측)로 폴백한다.
# 계산 공식(집계·이상치 처리)은 planner 소유다. Eval 은 원시 측정치만 준다.
# top_k 를 키워야 gold 를 담는 두 라벨은 gold 순위(diagnose 가 metadata 로 실어줌)로
# 필요 top_k 를 계산한다 — 근거가 같아 한 함수를 공유한다.
# retrieval_low_rank 는 제외: gold 가 후보엔 있고 순위만 낮은 문제라 정석 처방이
#   리랭커(rules.py enable_reranker)다. top_k 증가는 노이즈(too_long/lost_in_middle)를
#   키우는 열등한 차선책이라 rules.py 도 top_k 를 처방하지 않는다.
_GROUNDED_VALUES: dict[str, dict[str, Any]] = {
    "retrieval_incomplete_enumeration": {"top_k": _ground_top_k_from_gold},
    "retrieval_missing_gold": {
        "top_k": _ground_top_k_from_gold,
        "chunk_overlap": _ground_chunk_overlap_candidates,
    },
    "chunking_context_mismatch": {
        "chunk_overlap": _ground_chunk_overlap_candidates,
    },
    "too_long_context": {"chunk_size": _ground_chunk_size_candidates},
}


def _grounded_search_space(
    label: str,
    findings: list[Finding],
    state: AgentDoctorState,
    changes: dict,
) -> tuple[dict[str, list], dict[str, Any] | None]:
    """이 라벨에서 측정값으로 계산 가능한 {config키: [후보값]} 을 만든다.
    계산할 근거가 없는 키는 담지 않는다(호출부가 방향 키워드로 폴백)."""
    space: dict[str, list] = {}
    grounding_metadata: dict[str, Any] | None = None
    for key, compute in _GROUNDED_VALUES.get(label, {}).items():
        if key not in changes:
            continue
        values, metadata = compute(findings, state, changes.get(key))
        if values:
            space[key] = values
        if metadata is not None:
            grounding_metadata = metadata
    return space, grounding_metadata


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
    state: AgentDoctorState,
) -> tuple[dict[str, list], dict[str, Any] | None]:
    """이 처방이 바꿀 키들의 최종 후보값을 정한다.

    우선순위:
      1. Eval 이 Finding.metadata 로 직접 넘긴 후보(_supplied_candidates)
      2. planner 가 진단 측정값에서 계산한 근거값(_grounded_search_space)
      3. rules.py 방향 키워드를 현재값 기준으로 환산한 추측(_build_search_space)

    1이 2보다 앞서는 이유는 Eval 이 planner 보다 많은 원시 신호를 갖고 있어,
    후보 산출을 Eval 쪽으로 옮기더라도 planner 를 고치지 않게 하기 위해서다.
    """
    fallback = _build_search_space(changes, state.index_config)
    supplied = _supplied_candidates(findings)
    grounded, grounding_metadata = _grounded_search_space(
        findings[0].label if findings else "", findings, state, changes
    )

    resolved: dict[str, list] = {}
    for raw_path, patch_value in changes.items():
        path = canonicalize_path(raw_path)
        fallback_values = fallback.get(raw_path) or fallback.get(path) or []
        supplied_values = supplied.get(path)
        grounded_values = grounded.get(raw_path) or grounded.get(path)
        evidence_values = supplied_values or grounded_values
        values = list(evidence_values) if evidence_values else []
        current = get_current_value(state.index_config, path)
        if (
            path == "retriever.top_k"
            and patch_value in ("increase", "decrease")
            and isinstance(current, (int, float))
            and not isinstance(current, bool)
        ):
            values = [
                value
                for value in values
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and (
                    (patch_value == "increase" and value > current)
                    or (patch_value == "decrease" and value < current)
                )
            ]
        label = findings[0].label if findings else ""
        blocks_symbolic_fallback = (
            label == "chunking_context_mismatch"
            and path == "chunker.chunk_overlap"
            and isinstance(grounding_metadata, dict)
            and grounding_metadata.get("status") != "grounded"
        )
        if values:
            resolved[path] = list(values)
        elif not evidence_values and not blocks_symbolic_fallback:
            resolved[path] = list(fallback_values)
        if supplied_values and path in {
            "chunker.chunk_size",
            "chunker.chunk_overlap",
        }:
            grounding_metadata = {
                "status": "explicit_candidates",
                "source": "finding.metadata.parameter_candidates",
                "generated_candidates": list(supplied_values),
            }
    return resolved, grounding_metadata


def _build_candidates(
    label: str,
    findings: list[Finding],
    rule: dict,
    blacklist: set[tuple[str, str]],
    state: AgentDoctorState,
) -> list[PrescriptionCandidate]:
    """
    rules.py 의 raw dict 처방들을 PrescriptionCandidate 객체로 변환한다.
    rules.py 에 적힌 순서(가벼운 것 먼저)를 그대로 유지한다.
    블랙리스트에 걸린 처방은 제외한다.
    """
    candidates: list[PrescriptionCandidate] = []
    target_metrics = list(rule.get("target_metrics", []))
    reason = findings[0].description if findings else ""
    for pres in _available_prescriptions(rule, label, blacklist):
        changes = dict(pres.get("patch", {}))
        search_space, grounding_metadata = _finding_search_space(
            findings, changes, state
        )
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
                search_space=search_space,
                cost=float(_derive_cost(pres)),
                priority=0.0,          # 후보 개별 우선순위는 MVP 미사용
                target_metrics=list(target_metrics),  # rules.py 라벨의 target_metrics
                # 신호기반 택1(retrieval_semantic_mismatch 등). 없으면 빈 dict
                # → optimizer 가 순서대로 순차 시도(fallback).
                applies_when=dict(pres.get("applies_when", {})),
                reason=reason,
                metadata=(
                    {"candidate_grounding": grounding_metadata}
                    if grounding_metadata is not None
                    else {}
                ),
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
    if candidates and "candidate_grounding" in candidates[0].metadata:
        metadata["candidate_grounding"] = dict(
            candidates[0].metadata["candidate_grounding"]
        )
    if use_internal and _space_path(selected_space) in {
        "chunker.chunk_size",
        "chunker.chunk_overlap",
    }:
        metadata["chunk_precheck_context"] = _chunk_precheck_context(state, findings)
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
    metrics["pass_threshold"] = gate.passes_report(state.report)
    return metrics


def _chunk_precheck_context(
    state: AgentDoctorState,
    findings: list[Finding],
) -> dict[str, Any]:
    """Chunk 사전검증에 필요한 원문과 이미 준비된 gold span만 전달한다."""

    gold_spans = _valid_gold_spans(state, findings)
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
