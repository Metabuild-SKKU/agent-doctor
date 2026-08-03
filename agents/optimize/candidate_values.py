"""
agents/optimize/candidate_values.py
"방향 키워드"를 실제 후보 숫자로 바꾸는 계층.

[이 파일의 역할]
  rules.py 는 "top_k 를 늘려라" 같은 방향만 말한다. 얼마나 늘릴지는 진단이 이미 잰
  숫자에서 계산한다 — gold 순위, gold span 길이 분포, 청크 경계 좌표, 우세 검색 채널
  같은 것들이다. 그 계산이 전부 여기 모여 있다.

  근거가 없을 때만 방향 키워드 추측(현재값 ×2 / ÷2)으로 폴백한다. 어떤 라벨이 어떤
  경로에서 폴백을 허용하는지는 _SYMBOLIC_FALLBACK_ALLOWED 가 정한다.

[왜 planner 에서 분리했나]
  planner 는 "무엇을 어느 순서로 처방할지"를 정하는 결정 계층인데, 후보값 수학이
  1000줄 넘게 섞여 있어 두 관심사가 한 파일에 있었다. Action-Centered 전환에서
  planner 는 얇은 orchestration 으로 줄어들고 후보값 계산은 그대로 유지되므로,
  전환 전에 미리 갈라둔다.

  **계산 자체는 옮기면서 하나도 바꾸지 않았다.** 결과가 달라지면 회귀다
  (tests/test_planner.py 와 chunk 관련 테스트가 그대로 통과해야 한다).

[읽는 것]  Finding.metadata(진단 실측), state.index_config(현재값·정책), state.chunks,
           state.documents, state.probes
[쓰는 것]  없음. {canonical 경로: [후보값...]} 과 근거 metadata 를 반환할 뿐이다.

[호환]  planner 가 이 모듈의 이름을 그대로 re-export 한다. 기존 코드와 테스트가
        planner._knee 처럼 접근하던 경로를 깨지 않기 위해서다.
"""
from __future__ import annotations

import math
from typing import Any

from core.state import AgentDoctorState
from core.schema import Document, Finding
from agents.optimize import rules
from agents.optimize.config_mapper import canonicalize_path, get_current_value
from agents.optimize.evidence_window import build_evidence_windows


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

# Chunk 축의 방향 추측 폴백은 상태 이름으로만 결정한다. ``source`` 키 존재 여부는
# metadata 표현이 조금만 바뀌어도 안전 정책을 뒤집으므로 제어 신호로 쓰지 않는다.
_SYMBOLIC_FALLBACK_ALLOWED: dict[str, frozenset[str]] = {
    "chunker.chunk_size": frozenset({
        "missing_gold_spans",
        "missing_evidence_windows",
        "invalid_policy",
        "invalid_evidence_window_policy",
    }),
    "chunker.chunk_overlap": frozenset({
        "missing_gold_spans",
        "invalid_policy",
    }),
}

_EvidenceAnalysis = tuple[list[dict[str, Any]], dict[str, Any] | None]

# 방향 키워드를 계산할 때 baseline_config 에 해당 키가 없을 경우의 기본 현재값.
_DEFAULT_CURRENT: dict[str, int] = {
    "top_k": 5,
    "chunk_size": 512,
    "chunk_overlap": 50,
    "rerank_candidates": 20,
    "reranker.candidate_count": 20,
}

# 리랭커 후보 수 상한의 폴백(index_config["rerank_candidate_policy"]["max_candidates"]).
# 후보 하나가 곧 cross-encoder 추론 1쌍이라 검색 지연에 선형으로 실린다.
_DEFAULT_MAX_RERANK_CANDIDATES = 50

# 융합 가중치 조정 정책. 방향(어느 채널이 우세한가)은 Eval 실측이고, 폭은 여기 정책값이다 —
# 채널별 점수 분포가 넘어오면 폭도 근거화할 수 있다(그때 이 상수는 사라진다).
# 융합 가중치 안전 범위. optimizer.DEFAULT_CONSTRAINTS 와 같은 값이어야 한다.
_WEIGHT_MIN, _WEIGHT_MAX = 0.1, 0.9

_WEIGHT_STEPS = (0.1, 0.2)

_DEFAULT_HYBRID_DENSE_WEIGHT = 0.7

# rules.py 가 "이 값은 근거값 계산으로만 정해진다"고 표시하는 자리표시자.
# 근거 계산이 실패하면 방향 키워드처럼 추측으로 때우지 않고 그 키를 통째로 뺀다
# (문자열이 그대로 config 에 박히는 것을 막는다).
_GROUNDED_ONLY = "shift_to_favored_channel"



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
    findings: list[Finding],
    state: AgentDoctorState,
    _direction: Any,
    _evidence_analysis: _EvidenceAnalysis | None = None,
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
    candidates = _knee_candidates(required)
    # 실측에서 나온 후보임을 밝힌다. action 선택이 "근거 있는 후보"를 방향 키워드
    # 추측보다 우선하는데, 이 표시가 없으면 무릎 분석 결과가 추측과 동급으로 취급된다.
    return candidates, {
        "status": "grounded",
        "source": "gold_rank_knee",
        "probe_count": len(required),
        "required_min": min(required),
        "required_max": max(required),
        "generated_candidates": list(candidates),
    }


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
    max_step_ratio = policy.get("max_step_ratio", 0.25)
    min_chunk_size = policy.get("min_chunk_size", 200)
    max_chunk_size = policy.get("max_chunk_size", 1500)
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
        and candidate_count > 0
        and isinstance(min_span_count, int)
        and not isinstance(min_span_count, bool)
        and min_span_count > 0
        and isinstance(max_step_ratio, (int, float))
        and not isinstance(max_step_ratio, bool)
        and 0 < max_step_ratio <= 0.50
        and isinstance(min_chunk_size, int)
        and not isinstance(min_chunk_size, bool)
        and min_chunk_size > 0
        and isinstance(max_chunk_size, int)
        and not isinstance(max_chunk_size, bool)
        and max_chunk_size >= min_chunk_size
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
        "max_step_ratio": float(max_step_ratio),
        "min_chunk_size": min_chunk_size,
        "max_chunk_size": max_chunk_size,
    }, None


def _evidence_windows(
    state: AgentDoctorState,
    findings: list[Finding],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """영향받은 probe의 gold span을 구조 기반 evidence window로 확장한다."""

    spans = _valid_gold_spans(state, findings)
    if not spans:
        return [], {"status": "missing_gold_spans"}
    try:
        windows = build_evidence_windows(
            state.documents,
            spans,
            state.index_config.get("evidence_window_policy"),
        )
    except ValueError as exc:
        return [], {
            "status": "invalid_evidence_window_policy",
            "reason": str(exc),
        }
    if not windows:
        return [], {"status": "missing_evidence_windows"}
    kinds: dict[str, int] = {}
    for window in windows:
        kind = str(window.get("kind", "unknown"))
        kinds[kind] = kinds.get(kind, 0) + 1
    return windows, {
        "status": "grounded",
        "source": "structural_evidence_windows",
        "gold_span_count": len(spans),
        "evidence_window_count": len(windows),
        "evidence_window_kinds": kinds,
    }


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
    evidence_analysis: _EvidenceAnalysis | None = None,
) -> tuple[list[int] | None, dict[str, Any]]:
    """구조 기반 evidence window 길이와 현재값 사이에서 후보 범위를 만든다."""

    windows, evidence_metadata = (
        evidence_analysis
        if evidence_analysis is not None
        else _evidence_windows(state, findings)
    )
    if not windows:
        return None, evidence_metadata or {"status": "missing_evidence_windows"}

    policy, error = _chunk_candidate_policy(state)
    if policy is None:
        return None, {"status": "invalid_policy", "reason": error}

    current = get_current_value(state.index_config, "chunker.chunk_size")
    if isinstance(current, bool) or not isinstance(current, (int, float)) or current <= 0:
        return None, {"status": "invalid_current_value"}
    if direction not in {"increase", "decrease"}:
        return None, {"status": "unsupported_direction", "direction": direction}

    current_int = int(round(current))
    min_chunk_size = int(policy["min_chunk_size"])
    max_chunk_size = int(policy["max_chunk_size"])
    limit_metadata = {
        **(evidence_metadata or {}),
        "current_chunk_size": current_int,
        "min_chunk_size": min_chunk_size,
        "max_chunk_size": max_chunk_size,
        "direction": direction,
    }
    if (direction == "decrease" and current_int <= min_chunk_size) or (
        direction == "increase" and current_int >= max_chunk_size
    ):
        return None, {
            **limit_metadata,
            "status": "at_safe_limit",
        }

    lengths = [window["end"] - window["start"] for window in windows]
    if len(lengths) < policy["min_span_count"]:
        return None, {
            "status": "insufficient_spans",
            "source": "structural_evidence_windows",
            "span_count": len(lengths),
            "min_span_count": policy["min_span_count"],
        }
    p50 = _percentile_nearest_rank(lengths, 0.50)
    p85 = _percentile_nearest_rank(lengths, 0.85)
    p95 = _percentile_nearest_rank(lengths, 0.95)
    target_span = _percentile_nearest_rank(lengths, policy["target_quantile"])
    raw_target = target_span * (1 + policy["margin_ratio"])
    step = policy["rounding_step"]
    evidence_target = max(step, int(math.ceil(raw_target / step) * step))
    metadata: dict[str, Any] = {
        **(evidence_metadata or {}),
        "status": "grounded",
        "source": "structural_evidence_windows",
        "span_count": len(lengths),
        "min": min(lengths),
        "p50": p50,
        "p85": p85,
        "p95": p95,
        "max": max(lengths),
        "target_quantile": policy["target_quantile"],
        "margin_ratio": policy["margin_ratio"],
        "current_chunk_size": current_int,
        "evidence_target_chunk_size": evidence_target,
        "max_step_ratio": policy["max_step_ratio"],
        "direction": direction,
    }
    if (direction == "decrease" and evidence_target >= current_int) or (
        direction == "increase" and evidence_target <= current_int
    ):
        metadata["status"] = "direction_conflict"
        return None, metadata

    limits = _chunk_candidate_limits(
        current_int,
        step,
        policy,
    )
    if limits is None:
        boundary = _chunk_safety_boundary(
            current_int,
            direction,
            min_chunk_size,
            max_chunk_size,
        )
        if boundary is not None:
            return _clamped_chunk_candidate(metadata, *boundary)
        return None, {
            **limit_metadata,
            "status": "no_safe_candidate_within_bounds",
            "max_step_ratio": policy["max_step_ratio"],
        }
    lower_limit, upper_limit = limits
    if direction == "decrease":
        upper_limit = min(upper_limit, ((current_int - 1) // step) * step)
    else:
        lower_limit = max(lower_limit, ((current_int // step) + 1) * step)
    if lower_limit > upper_limit:
        boundary = _chunk_safety_boundary(
            current_int,
            direction,
            min_chunk_size,
            max_chunk_size,
        )
        if boundary is not None:
            return _clamped_chunk_candidate(metadata, *boundary)
        return None, {
            **limit_metadata,
            "status": "no_safe_candidate_within_bounds",
            "candidate_lower_limit": lower_limit,
            "candidate_upper_limit": upper_limit,
            "max_step_ratio": policy["max_step_ratio"],
        }
    target = min(max(evidence_target, lower_limit), upper_limit)
    metadata.update({
        "target_chunk_size": target,
        "candidate_lower_limit": lower_limit,
        "candidate_upper_limit": upper_limit,
        "max_step_ratio_applied": True,
    })

    candidates: list[int] = []
    for fraction in policy["path_fractions"]:
        value = current_int + ((target - current_int) * fraction)
        rounded = _round_to_step(value, step)
        if not lower_limit <= rounded <= upper_limit:
            continue
        if direction == "decrease" and not target <= rounded < current_int:
            continue
        if direction == "increase" and not current_int < rounded <= target:
            continue
        if rounded not in candidates:
            candidates.append(rounded)
        if len(candidates) >= policy["candidate_count"]:
            break

    if not candidates:
        metadata["status"] = "insufficient_candidates"
        return None, metadata
    metadata["generated_candidates"] = list(candidates)
    return candidates, metadata


def _chunk_safety_boundary(
    current: int,
    direction: str,
    min_chunk_size: int,
    max_chunk_size: int,
) -> tuple[int, str] | None:
    """범위 밖 baseline을 처방 방향과 일치하는 가장 가까운 경계로 복구한다."""

    if direction == "decrease" and current > max_chunk_size:
        return max_chunk_size, "max_chunk_size"
    if direction == "increase" and current < min_chunk_size:
        return min_chunk_size, "min_chunk_size"
    return None


def _clamped_chunk_candidate(
    metadata: dict[str, Any],
    target: int,
    boundary_name: str,
) -> tuple[list[int], dict[str, Any]]:
    """비율·절대 범위의 교집합이 없을 때만 안전 경계 후보를 만든다."""

    metadata.update({
        "target_chunk_size": target,
        "candidate_lower_limit": target,
        "candidate_upper_limit": target,
        "safety_bound_clamp": boundary_name,
        "max_step_ratio_applied": False,
        "generated_candidates": [target],
    })
    return [target], metadata


def _chunk_candidate_limits(
    current: int,
    step: int,
    policy: dict[str, Any],
) -> tuple[int, int] | None:
    """절대 안전 범위와 1회 변경 비율의 교집합을 계산한다."""

    lower_limit = max(
        int(policy["min_chunk_size"]),
        int(math.ceil(
            current * (1 - float(policy["max_step_ratio"])) / step
        ) * step),
    )
    upper_limit = min(
        int(policy["max_chunk_size"]),
        int(math.floor(
            current * (1 + float(policy["max_step_ratio"])) / step
        ) * step),
    )
    if lower_limit > upper_limit:
        return None
    return lower_limit, upper_limit


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
    _evidence_analysis: _EvidenceAnalysis | None = None,
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


def _probe_required_candidates(finding: Finding, max_allowed: int) -> int | None:
    """probe 하나가 gold 를 리랭커 후보에 담으려면 필요한 최소 후보 수(상한 안쪽만).

    _probe_required_top_k 와 달리 상한 밖 gold 순위를 **먼저** 버리고 최댓값을 뽑는다.
    나중에 거르면 도달 불가 gold 하나가 그 probe 의 도달 가능한 근거까지 끌고 나간다.
    """
    ranks = finding.metadata.get("gold_ranks")
    if isinstance(ranks, dict):
        reachable = [
            r for r in ranks.values()
            if isinstance(r, int) and not isinstance(r, bool) and r <= max_allowed
        ]
        return max(reachable) if reachable else None
    required = _probe_required_top_k(finding)     # 순위 미측정 → 개수 근사 폴백
    return required if required is not None and required <= max_allowed else None


def _ground_rerank_candidates(
    findings: list[Finding],
    state: AgentDoctorState,
    _direction: Any,
    _evidence_analysis: _EvidenceAnalysis | None = None,
) -> tuple[list[int] | None, dict[str, Any] | None]:
    """리랭커 후보창을 얼마나 넓혀야 gold 가 후보에 들어오나 — 실측 순위의 무릎.

    top_k 근거값과 계산이 같다("가장 늦게 나오는 gold 의 순위 = 필요한 창 크기").
    다른 건 상한이다: 후보 수는 매 검색마다 cross-encoder 추론 쌍 수라, 정책상 상한
    (rerank_candidate_policy.max_candidates)을 넘겨선 안 된다.
    현재값 이하 후보는 넓히는 처방이 아니므로 버린다.

    상한 밖 순위는 필요값 집계에서 아예 제외한다 — 창을 그만큼 넓힐 수 없으니 '도달 불가'이고,
    이건 wide_n 밖 gold(순위 None)를 top_k 근거에서 빼는 것과 같은 이유다. 안 빼면 창 밖
    gold 하나가 무릎을 상한 위로 끌어올려, 실제로 도달 가능한 gold 들의 근거값까지 통째로
    날아가고 방향 키워드 추측(×2)으로 내려간다.

    ⚠ 제외는 **gold 순위 단위**로 해야 한다. probe 단위(= 그 probe 의 최대 순위)로 거르면
    한 probe 안에 30위(도달 가능)와 90위(불가)가 섞였을 때 max=90 이라 그 probe 가 통째로
    빠지고, 30위라는 멀쩡한 근거까지 같이 날아간다 — 막으려던 현상이 probe 안에서 재현된다.
    """
    policy = state.index_config.get("rerank_candidate_policy") or {}
    max_allowed = int(policy.get("max_candidates", _DEFAULT_MAX_RERANK_CANDIDATES))
    required = [
        r for r in (_probe_required_candidates(f, max_allowed) for f in findings)
        if r is not None
    ]
    if not required:
        return None, None
    current = get_current_value(state.index_config, "reranker.candidate_count")
    current_int = (
        int(current) if isinstance(current, (int, float)) and not isinstance(current, bool)
        else _DEFAULT_CURRENT["rerank_candidates"]
    )
    candidates = [
        value for value in _knee_candidates(required)
        if current_int < value <= max_allowed
    ]
    if not candidates:
        return None, {"status": "insufficient_candidates",
                      "source": "gold_rank_knee", "max_allowed": max_allowed}
    return candidates, {"status": "grounded", "source": "gold_rank_knee",
                        "generated_candidates": list(candidates)}


def _ground_hybrid_dense_weight(
    findings: list[Finding],
    state: AgentDoctorState,
    _direction: Any,
    _evidence_analysis: _EvidenceAnalysis | None = None,
) -> tuple[list[float] | None, dict[str, Any] | None]:
    """융합 가중치를 어느 쪽으로 옮길지 — 방향은 실측, 폭은 정책.

    Eval 이 "어느 채널이 gold 를 상위에 뒀나"(favored_channel)를 실측해 넘긴다. 그 채널
    쪽으로 _WEIGHT_STEPS 만큼 옮긴 값을 후보로 낸다(dense 우세면 올리고, lexical 우세면 내림).
    폭까지 실측하려면 채널별 점수 분포가 필요한데 현재 순위만 넘어오므로, 폭은 정책값이다.
    채널이 섞여 있으면(우세 채널이 여럿) 다수결로 정하고, 동수면 근거가 없어 폴백한다.
    """
    votes = {"dense": 0, "lexical": 0}
    for finding in findings:
        channel = finding.metadata.get("favored_channel")
        if channel in votes:
            votes[channel] += 1
    if votes["dense"] == votes["lexical"]:
        return None, None                # 방향 근거 없음 → 방향 키워드 폴백
    favored = "dense" if votes["dense"] > votes["lexical"] else "lexical"

    current = get_current_value(state.index_config, "retriever.hybrid_dense_weight")
    if not isinstance(current, (int, float)) or isinstance(current, bool):
        current = _DEFAULT_HYBRID_DENSE_WEIGHT
    sign = 1 if favored == "dense" else -1
    candidates: list[float] = []
    for step in _WEIGHT_STEPS:
        value = round(min(_WEIGHT_MAX, max(_WEIGHT_MIN, float(current) + sign * step)), 2)
        if value != round(float(current), 2) and value not in candidates:
            candidates.append(value)
    if not candidates:
        return None, None
    return candidates, {"status": "grounded", "source": "channel_rank_advantage",
                        "favored_channel": favored,
                        "generated_candidates": list(candidates)}


# 라벨 → 근거값 계산 함수. 여기 없는 라벨은 방향 키워드(추측)로 폴백한다.
# 계산 공식(집계·이상치 처리)은 planner 소유다. Eval 은 원시 측정치만 준다.
# top_k 를 키워야 gold 를 담는 두 라벨은 gold 순위(diagnose 가 metadata 로 실어줌)로
# 필요 top_k 를 계산한다 — 근거가 같아 한 함수를 공유한다.
# retrieval_low_rank 는 제외: gold 가 후보엔 있고 순위만 낮은 문제라 정석 처방이
#   리랭커(rules.py enable_reranker)다. top_k 증가는 노이즈(too_long/lost_in_middle)를
#   키우는 열등한 차선책이라 rules.py 도 top_k 를 처방하지 않는다.
_GROUNDED_VALUES: dict[str, dict[str, Any]] = {
    # 순위 원인 4형제: 창 크기는 실측 순위의 무릎, 융합 가중치는 실측 우세 채널 방향.
    "retrieval_rerank_candidate_miss": {
        "reranker.candidate_count": _ground_rerank_candidates,
    },
    "retrieval_rank_fusion_loss": {
        "retriever.hybrid_dense_weight": _ground_hybrid_dense_weight,
    },
    # semantic mismatch 중 topic_cluster="none"은 rules.py에서 청크 희석으로
    # 분류해 chunk_size 축소를 처방한다. 이 경우에도 고정 비율 추측 대신 같은
    # 구조적 evidence window 분포에서 안전한 축소 후보를 계산한다.
    "retrieval_semantic_mismatch": {
        "chunk_size": _ground_chunk_size_candidates,
    },
    "retrieval_incomplete_enumeration": {"top_k": _ground_top_k_from_gold},
    "retrieval_missing_gold": {
        "top_k": _ground_top_k_from_gold,
        "chunk_overlap": _ground_chunk_overlap_candidates,
        "chunk_size": _ground_chunk_size_candidates,
    },
    "chunking_context_mismatch": {
        "chunk_overlap": _ground_chunk_overlap_candidates,
        "chunk_size": _ground_chunk_size_candidates,
    },
    # gold span 이 청크보다 길어 겹침으로는 못 담는 경우 — 필요한 크기를 span 길이에서 계산한다
    # (없으면 _concrete_values 의 방향 폴백으로 현재값 배수 추측이 된다).
    "chunking_overchunking": {"chunk_size": _ground_chunk_size_candidates},
    "chunking_underchunking": {"chunk_size": _ground_chunk_size_candidates},
    "too_long_context": {"chunk_size": _ground_chunk_size_candidates},
}


def _grounded_search_space(
    label: str,
    findings: list[Finding],
    state: AgentDoctorState,
    changes: dict,
    evidence_analysis: _EvidenceAnalysis | None = None,
) -> tuple[dict[str, list], dict[str, Any] | None]:
    """이 라벨에서 측정값으로 계산 가능한 {config키: [후보값]} 을 만든다.
    계산할 근거가 없는 키는 담지 않는다(호출부가 방향 키워드로 폴백)."""
    space: dict[str, list] = {}
    grounding_metadata: dict[str, Any] | None = None
    for key, compute in _GROUNDED_VALUES.get(label, {}).items():
        # rules.py 는 같은 축을 flat("chunk_size")으로도 canonical 로도 선언한다.
        # 문자 그대로 비교하면 표기가 다른 순간 근거 계산이 조용히 건너뛰어지고
        # 방향 키워드 추측으로 내려간다.
        change_key = key if key in changes else next(
            (
                raw_path
                for raw_path in changes
                if canonicalize_path(raw_path) == canonicalize_path(key)
            ),
            None,
        )
        if change_key is None:
            continue
        values, metadata = compute(
            findings,
            state,
            changes.get(change_key),
            evidence_analysis,
        )
        if values:
            space[change_key] = values
        if metadata is not None:
            grounding_metadata = metadata
    return space, grounding_metadata


# ── 6. 후보값 변환 (rules.py patch 값 → 구체 후보값) ──────────────

def _concrete_values(
    key: str, patch_value: Any, baseline_config: dict
) -> list[Any] | None:
    """
    rules.py patch 값 하나를 optimizer 가 쓸 구체 후보값 리스트로 변환한다.
      - "increase"/"decrease" : 현재값 × 또는 ÷ _DIRECTION_STEP.
          정수 knob(top_k·chunk_size 등)은 정수로, 실수 knob(temperature 등)은
          실수(소수 3자리)로 유지한다 — float 을 int 로 캐스팅하면 0.15→0 처럼
          뭉개져 온도 미세조정이 불가능해진다.
      - _GROUNDED_ONLY        : 추측 폴백 금지 → None (근거값이 없으면 키를 뺀다)
      - 그 외(True, 숫자, "recursive_sentence" 등) : 그대로 [값]
    현재값이 숫자가 아니거나 없어 계산이 불가하면 None(→ 해당 키 제외).
    """
    if patch_value == _GROUNDED_ONLY:
        return None
    if patch_value in ("increase", "decrease"):
        canonical_key = canonicalize_path(key)
        current = get_current_value(baseline_config, canonical_key)
        if current is None:
            current = _DEFAULT_CURRENT.get(
                key,
                _DEFAULT_CURRENT.get(canonical_key),
            )
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            return None  # 현재값을 숫자로 알 수 없으면 방향 계산 불가
        raw = (current * _DIRECTION_STEP if patch_value == "increase"
               else current / _DIRECTION_STEP)
        # 정수 knob 은 정수 유지, 실수 knob 은 소수로 유지(타입 보존).
        return [int(round(raw)) if isinstance(current, int) else round(raw, 3)]
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


def _has_grounding_calculator(label: str, raw_path: str) -> bool:
    """이 라벨이 이 축의 근거 계산기를 등록해 뒀는가."""
    registered = _GROUNDED_VALUES.get(label, {})
    path = canonicalize_path(raw_path)
    return any(canonicalize_path(key) == path for key in registered)


def _unregistered_chunk_grounding(
    label: str,
    raw_path: str,
    patch_value: Any,
) -> dict[str, Any] | None:
    """근거 계산기를 아예 등록하지 않은 chunk 축을 **드러낸다**.

    등록 누락과 "계산했는데 근거가 부족했다"는 다른 상황인데, 둘 다 근거값이 없다는
    같은 모습으로 나타난다. 구분하지 않으면 라벨이 조용히 현재값 배수 추측으로
    내려가고 아무도 알아채지 못한다 — chunk 축은 재색인을 유발하므로 대가가 크다.
    """
    path = canonicalize_path(raw_path)
    if path not in _SYMBOLIC_FALLBACK_ALLOWED:
        return None
    if patch_value not in {"increase", "decrease", _GROUNDED_ONLY}:
        return None
    if _has_grounding_calculator(label, raw_path):
        return None
    return {
        "status": "grounding_unregistered",
        "source": "candidate_values._GROUNDED_VALUES",
        "label": label,
        "path": path,
        "raw_path": raw_path,
        "direction": patch_value,
    }


def _finding_search_space(
    findings: list[Finding],
    changes: dict,
    state: AgentDoctorState,
    evidence_analysis: _EvidenceAnalysis | None = None,
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
        findings[0].label if findings else "",
        findings,
        state,
        changes,
        evidence_analysis,
    )

    resolved: dict[str, list] = {}
    # 축마다 근거 상태가 다를 수 있으므로 반환용 metadata 를 따로 들고 간다.
    candidate_grounding_metadata = grounding_metadata
    label = findings[0].label if findings else ""
    for raw_path, patch_value in changes.items():
        path = canonicalize_path(raw_path)
        fallback_values = fallback.get(raw_path) or fallback.get(path) or []
        supplied_values = supplied.get(path)
        grounded_values = grounded.get(raw_path) or grounded.get(path)
        evidence_values = supplied_values or grounded_values
        values = list(evidence_values) if evidence_values else []
        unregistered_grounding = (
            _unregistered_chunk_grounding(label, raw_path, patch_value)
            if not evidence_values
            else None
        )
        path_grounding_metadata = unregistered_grounding or grounding_metadata
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
        allows_symbolic_fallback = _allows_symbolic_fallback(
            path,
            path_grounding_metadata,
        )
        if values:
            resolved[path] = list(values)
        elif fallback_values and not evidence_values and allows_symbolic_fallback:
            # 폴백까지 비면 그 키는 후보값을 못 만든 것이다. 빈 리스트를 남기면
            # optimizer 가 '축은 있는데 값이 없는' search_space 를 받게 되므로 아예 뺀다.
            resolved[path] = list(fallback_values)
        if unregistered_grounding is not None:
            candidate_grounding_metadata = unregistered_grounding
        if supplied_values and path in {
            "chunker.chunk_size",
            "chunker.chunk_overlap",
        }:
            candidate_grounding_metadata = {
                "status": "explicit_candidates",
                "source": "finding.metadata.parameter_candidates",
                "generated_candidates": list(supplied_values),
            }
    return resolved, candidate_grounding_metadata


def _allows_symbolic_fallback(
    path: str,
    grounding_metadata: dict[str, Any] | None,
) -> bool:
    """Chunk 방향 추측의 허용 여부를 명시적인 grounding status로 결정한다."""

    allowed_statuses = _SYMBOLIC_FALLBACK_ALLOWED.get(path)
    if allowed_statuses is None:
        return True
    # 근거가 아예 없는 것을 "허용"으로 읽으면 chunk 축이 조용히 현재값 배수 추측으로
    # 내려간다 — 이 축은 재색인을 유발하므로 추측의 대가가 크다.
    if grounding_metadata is None:
        return False
    status = grounding_metadata.get("status")
    return isinstance(status, str) and status in allowed_statuses


