"""
agents/optimize/history.py
Optimize 모듈의 "판정·기록" 계층.

[역할]
  planner 가 고른 처방이 실제로 적용된 뒤(Index→Eval 재측정 후) 다시 Optimize 로
  돌아왔을 때, 지난 처방이 좋았는지 나빴는지 판정하고 이력 항목을 만들어 준다.
    1) check_floor         : 모든 지표가 하한선 위인지 검사 (위반 지표 반환)
    2) judge               : 하한선 위반 → 무조건 롤백, 아니면 점수 비교로 유지/롤백
    3) create_pending_item : 처방 적용 시점의 pending 이력 항목 생성(before 만 채움)
    4) finalize_item       : 다음 방문에서 판정 결과로 pending 항목 확정(after+verdict)
    5) find_pending        : 판정 대기 중(pending)인 최근 이력 항목 조회

  판정이 다음 Eval 재측정 후에야 가능하므로(시점 문제), 이력은 두 단계로 기록한다.
  적용 시점엔 before 만 담은 pending 항목을 만들고, 다음 방문에서 확정한다.
  "판정과 이력 항목 생성"까지가 history 의 책임이다. state 를 직접 수정하지 않고
  값만 반환하며, 실제 반영(블랙리스트 추가·이력 append·config 롤백)은 agent.py 가 한다.
  (planner 가 (request, decision) 만 반환하고 state 는 안 건드린 것과 같은 방식.)
  "무엇을 처방할지"는 planner, "그 처방을 config 로 바꾸는 일"은 config_mapper 소관.

[읽는 것]  before/after 리포트(DiagnosticReport.overall_score, ragas_scores), state.iteration
[쓰는 것]  없음 — 반환값을 agent.py 가 state.optimization_history / state.blacklist 에 반영

[MVP 결정 사항]  (나중에 재검토 가능)
  - 단일 점수(overall_score)는 Eval 이 계산한다. history 는 그 값을 읽어 비교만 한다.
    (여기서 점수 공식을 다시 만들지 않는다 — Eval 과 다른 값이 나오면 안 되므로.)
  - FLOORS(하한선)는 Optimize 고유의 롤백 안전망이라 여기서 관리한다. 지금은 임시
    더미값이며, Eval 임계값이 확정되면 그에 맞춰 조정한다.
  - 롤백 판정은 원값(반올림 전)으로 한다. 사용자 표시용 반올림은 reporter 소관.
"""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from typing import Any

from core.schema import DiagnosticReport
from core.state import AgentDoctorState
from agents.optimize.config_mapper import canonical_config_view, canonicalize_path
from agents.optimize.schemas import (
    ActionAttemptKey,
    ActionStudyKey,
    OptimizationHistoryItem,
    OptimizationRequest,
    Verdict,
)


# ── 하한선 기준값 (Optimize 롤백 안전망) ──────────────────────────
# RAGAS 6개 지표. noise_sensitivity 만 "낮을수록 좋음".
_LOWER_IS_BETTER: set[str] = {"noise_sensitivity"}

# 하한선: 이 밑(낮을수록 좋은 지표는 이 위)으로 떨어지면 무조건 롤백.
# "통과 기준"과 별개인 '절대 넘으면 안 되는 안전선'이라 Optimize 가 관리한다.
# TODO: ⚠️ 전부 임시 더미값. Eval 임계값 확정 시 그에 맞춰 조정.
_FLOORS: dict[str, float] = {
    "faithfulness": 0.50,
    "answer_relevancy": 0.50,
    "context_recall": 0.45,
    "context_precision": 0.45,
    "context_utilization": 0.40,
    "noise_sensitivity": 0.60,   # 낮을수록 좋음 → 0.60 초과면 위반
}


# ── 개선 판정 최소 마진 ───────────────────────────────────────────
# 개선으로 인정할 최소 상승폭(정규화 composite 0~1 기준).
# 표시 점수 2점 = 0.02. Eval 노이즈(LLM judge 편차·표본 오차) 안에 묻히는 상승을
# "개선"으로 확정하지 않기 위한 값이다.
#
# ⚠️ 이 값은 **잠정치다. 노이즈 분포를 측정해서 나온 값이 아니다.**
#    유일한 관측 근거는 같은 config 재평가가 82↔78(표시 4점 폭)로 흔들린 사례
#    하나뿐이고(PR #66 리뷰), 그게 사실이면 2점은 노이즈보다 작아 제 역할을 못 한다.
#    같은 config 반복 Eval 로 σ 를 재고 마진을 k·σ 로 다시 정해야 한다.
#    보정에 쓸 관측값은 finalize_item 이 이력에 남긴다
#    (margin_rejected·score_delta·improvement_margin).
#
# ⚠️ 스케일 주의: judge 도 sweep objective 도 정규화 composite(0~1)를 쓴다.
#    표시용 composite_total(0~100)과 혼동하면 마진이 100배 어긋난다.
#
# judge(유지/롤백)와 internal sweep(best 후보 선정)이 같은 값을 써야 한다. 두 경로는
# 각각 독립적으로 "개선했는가"를 판정하고 둘 다 사용자 리포트로 나가므로, 임계가
# 다르면 같은 점수 변화가 경로에 따라 다르게 보고된다.
# internal_adapter 는 history 를 import 하지 않으므로 planner 가 이 값을
# OptimizationRequest.metadata["min_delta"] 로 중계한다(planner._build_request).
MIN_IMPROVEMENT_MARGIN: float = 0.02


def meets_improvement_margin(
    improvement: float,
    margin: float = MIN_IMPROVEMENT_MARGIN,
) -> bool:
    """상승폭이 마진을 만족하는가. (경계값 포함)

    internal_adapter._meets_min_delta 와 **동작이 같아야** 한다. 두 모듈은 각각
    독립적으로 "개선했는가"를 판정하며(judge 는 rules 처방을, sweep 은 후보 묶음을),
    둘 다 사용자 리포트로 나간다. 임계가 다르면 같은 점수 변화가 어느 경로를 탔는지에
    따라 다르게 보고된다. 그래서 비교식(`> margin` 또는 math.isclose)까지 맞춘다.

    agent._relax_reranker_precision_floor 도 이 함수를 쓴다 — floor 위반 경로는
    judge 가 먼저 반환해 마진을 안 거치므로, 그쪽 점수 관문을 여기에 맞춰야 한다.
    """
    return improvement > 0 and (
        improvement > margin
        or math.isclose(improvement, margin, rel_tol=1e-12, abs_tol=1e-12)
    )


# ── 0. action 식별자 (구현계획 §5.1 / §8.2) ───────────────────────
# 실행 이력과 차단의 단위를 (label, prescription_id) 에서 "실제 config 전이" 로 옮긴다.
#
#   ActionAttemptKey = (action_key, baseline_fingerprint, candidate_fingerprint)
#   ActionStudyKey   = (action_key, baseline_fingerprint, search_space_fingerprint)
#
# ⚠️ 이 전환은 차단 **강화가 아니라 baseline 별 완화**다. 기존 튜플은 baseline 과
# 무관한 영구 차단이라 "top_k 를 고쳐 검색을 개선한 뒤 리랭커를 다시 본다" 같은
# 정당한 재평가까지 막았다. fingerprint 에 baseline 을 넣으면 config 가 바뀐 뒤의
# 같은 action 은 새 시도가 된다. 무한 반복은 정확 전이 차단 + 후보 소진 +
# 예산(iteration/visit) + 개선 마진이 막는다.
_FINGERPRINT_LENGTH = 16


def _digest(payload: Any) -> str:
    """비교 가능한 payload 를 결정적 짧은 해시로 만든다.

    set 이나 dict 순서 때문에 같은 입력이 다른 해시를 내면 차단이 무력화되므로
    sort_keys 로 정렬하고, 직렬화 불가 타입은 repr 로 고정한다.
    """
    encoded = json.dumps(payload, sort_keys=True, default=repr, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:_FINGERPRINT_LENGTH]


def _capability_identity(
    action_key: str,
    runtime_capabilities: dict[str, Any] | None,
) -> dict[str, Any]:
    """이 action 의 실행 가능 여부를 좌우하는 runtime 관측만 뽑는다.

    reranker 처럼 Index 가 실제 로드해 봐야 아는 optional 모델이 대상이다. 모델이
    바뀌거나 verified 로 승격되면 같은 config 전이라도 결과가 달라지므로 baseline
    정체성에 포함한다. 무관한 capability 까지 넣으면 관계없는 변화에 차단이 풀린다.
    """
    if not action_key.startswith("reranker.") or not isinstance(
        runtime_capabilities, dict
    ):
        return {}
    capability = runtime_capabilities.get("reranker")
    if not isinstance(capability, dict):
        return {}
    return {
        "reranker": {
            "status": capability.get("status"),
            "model": capability.get("model"),
        }
    }


def baseline_fingerprint(
    baseline_config: dict[str, Any],
    action_key: str = "",
    runtime_capabilities: dict[str, Any] | None = None,
) -> str:
    """"지금 어느 baseline 위에 서 있는가" 의 지문."""
    return _digest(
        {
            "config": canonical_config_view(baseline_config or {}),
            "capability": _capability_identity(action_key, runtime_capabilities),
        }
    )


def candidate_fingerprint(changes: dict[str, Any] | None) -> str:
    """적용하려는(또는 적용한) 구체 config 변경의 지문."""
    return _digest(
        {
            canonicalize_path(path): value
            for path, value in sorted((changes or {}).items())
        }
    )


def search_space_fingerprint(search_space: dict[str, Any] | None) -> str:
    """탐색 범위 전체의 지문. 후보 하나가 아니라 묶음을 식별한다."""
    return _digest(
        {
            canonicalize_path(path): list(values)
            if isinstance(values, (list, tuple))
            else [values]
            for path, values in sorted((search_space or {}).items())
        }
    )


def build_attempt_key(
    action_key: str,
    baseline_config: dict[str, Any],
    changes: dict[str, Any] | None,
    runtime_capabilities: dict[str, Any] | None = None,
) -> ActionAttemptKey:
    """품질 실패로 차단할 '정확한 전이' 하나."""
    return ActionAttemptKey(
        action_key=action_key,
        baseline_fingerprint=baseline_fingerprint(
            baseline_config, action_key, runtime_capabilities
        ),
        candidate_fingerprint=candidate_fingerprint(changes),
    )


def build_study_key(
    action_key: str,
    baseline_config: dict[str, Any],
    search_space: dict[str, Any] | None,
    runtime_capabilities: dict[str, Any] | None = None,
) -> ActionStudyKey:
    """한 baseline 에서 결론이 난 탐색 하나."""
    return ActionStudyKey(
        action_key=action_key,
        baseline_fingerprint=baseline_fingerprint(
            baseline_config, action_key, runtime_capabilities
        ),
        search_space_fingerprint=search_space_fingerprint(search_space),
    )


def study_key_for_request(
    request: OptimizationRequest,
    runtime_capabilities: dict[str, Any] | None = None,
) -> ActionStudyKey | None:
    """요청 하나가 시작하려는 study 의 식별자. action 이 없으면 None."""
    if not request.action_key:
        return None
    return build_study_key(
        request.action_key,
        request.baseline_config,
        request.search_space,
        runtime_capabilities,
    )


# ── 0-b. 결과 config 기억 (같은 답 재측정 방지) ────────────────────
# attempt/study 차단의 단위는 **전이**(action, baseline, 후보값)라, 도착 지점이 같은
# 다른 전이는 막지 못한다. reranker 끄기가 그 예다 — baseline 의 후보창이 20 이든
# 22 든 끄고 나면 검색 동작은 하나로 수렴하는데, baseline 지문이 달라 차단이 풀리고
# 이미 실패한 config 를 다시 잰다.
#
# 그래서 전이가 아니라 **도착 config** 를 기억한다. 이미 재 봤고 그 점수가 지금
# baseline 대비 개선 마진을 못 넘었다면, 다시 적용해도 judge 는 반드시 롤백한다 —
# iteration 하나를 아는 답 재확인에 쓰는 셈이다.

# 다른 축의 값 때문에 파이프라인이 아예 읽지 않게 되는 축.
#   {비활성화되는 축: 그것을 켜는 축}
# 리랭커가 꺼져 있으면 후보창 크기는 검색 동작에 아무 영향이 없으므로, config
# 정체성에서 빼야 "같은 동작"을 같은 config 로 인식한다. 여기 빠진 축이 있으면
# 기억이 헐거워질 뿐(중복 측정)이고, 반대로 잘못 넣으면 서로 다른 동작을 같은
# config 로 오인해 정당한 시도를 막는다 — 확신이 있는 축만 넣는다.
_INERT_WHEN_OFF: dict[str, str] = {
    "reranker.candidate_count": "reranker.enabled",
}

# 되돌리기 견제(reversal guard)용 방향 쌍. 여기 없는 operation(replace/adjust)은
# "되돌린다"를 정의할 수 없으므로 견제 대상이 아니다.
_REVERSE_OPERATIONS: dict[str, str] = {
    "enable": "disable",
    "disable": "enable",
    "increase": "decrease",
    "decrease": "increase",
}

_MISSING = object()


def effective_config_view(config: dict[str, Any]) -> dict[str, Any]:
    """파이프라인 동작에 실제로 영향을 주는 축만 남긴 config 뷰.

    게이트 축이 config 에 없으면 아무것도 빼지 않는다 — 기본값을 추측해 비활성으로
    단정하면, 서로 다른 동작의 config 두 개가 같은 지문을 갖게 된다.
    """
    view = canonical_config_view(config or {})
    for path, gate in _INERT_WHEN_OFF.items():
        if path in view and gate in view and not view[gate]:
            view.pop(path)
    return view


def effective_config_fingerprint(config: dict[str, Any]) -> str:
    """'이 config 로 돌리면 같은 동작인가' 의 지문."""
    return _digest(effective_config_view(config))


def measured_config_scores(
    optimization_history: list[OptimizationHistoryItem],
) -> dict[str, float]:
    """이번 실행에서 실제로 Eval 을 돌려 본 config 와 그 점수(정규화 0~1).

    한 config 를 여러 번 쟀으면 **가장 좋았던** 관측을 남긴다. 이 값은 재적용을
    막는 근거로 쓰이므로 후보에게 가장 유리한 관측으로 판단해야, 노이즈로 낮게
    나온 한 번의 측정이 멀쩡한 축을 닫지 않는다.

    unjudgeable 항목은 건너뛴다 — 점수가 있어도 그 측정이 성립하지 않았다는 뜻이라
    (리포트 부재·리랭커 미실행) 재적용 차단의 근거로 쓸 수 없다.
    """
    scores: dict[str, float] = {}

    def record(config: dict[str, Any] | None, score: Any) -> None:
        if not config:
            return
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            return
        key = effective_config_fingerprint(config)
        if key not in scores or float(score) > scores[key]:
            scores[key] = float(score)

    for item in optimization_history:
        if item.metadata.get("pending") or item.metadata.get("unjudgeable"):
            continue
        record(item.before_config, item.metadata.get("before_score"))
        record(item.after_config, item.metadata.get("after_score"))
    return scores


def no_progress_config_fingerprints(
    optimization_history: list[OptimizationHistoryItem],
    index_config: dict[str, Any],
    report: DiagnosticReport | None = None,
    margin: float = MIN_IMPROVEMENT_MARGIN,
) -> set[str]:
    """다시 적용해도 judge 가 반드시 롤백할 config 들의 지문.

    비교 기준(현재 점수)은 **현재 config 의 관측값을 우선**한다. 롤백 직후 방문에서
    state.report 는 되돌리기 전의 열화된 Eval 이라, 그걸 기준으로 삼으면 문턱이
    낮아져 이미 실패한 config 가 다시 통과한다. 복원된 config 의 점수는 이력에
    남아 있으므로 그쪽을 먼저 본다.
    """
    scores = measured_config_scores(optimization_history)
    if not scores:
        return set()

    current = scores.get(effective_config_fingerprint(index_config or {}))
    if current is None:
        if report is None:
            return set()
        current = _read_score(report)

    return {
        fingerprint
        for fingerprint, score in scores.items()
        if not meets_improvement_margin(score - current, margin)
    }


def _reverse_action_key(action_key: str) -> str | None:
    """이 action 을 되돌리는 action 의 key. 방향이 없는 축이면 None."""
    path, _, operation = action_key.partition(":")
    reverse = _REVERSE_OPERATIONS.get(operation)
    return f"{path}:{reverse}" if reverse else None


def reversal_guard_thresholds(
    optimization_history: list[OptimizationHistoryItem],
    index_config: dict[str, Any],
) -> dict[str, float]:
    """되돌리기를 견제할 action 과, 그것을 허용하기 위해 필요한 지지 크기.

    유지(KEEP) 판정은 "이 변경이 **전체적으로** 이득이었다"는 실측이다. 그런데
    변경 하나가 probe 여럿을 살리면서 한둘을 새로 망가뜨리는 일은 흔하고, 그
    부작용에 붙은 라벨이 곧바로 역방향 처방을 지지한다. 지지 크기를 비교하지 않으면
    probe 1개짜리 부작용이 probe 7개짜리 이득을 되돌린다.
    ⚠️ 이건 영구 금지가 아니라 **문턱**이다. 되돌리기가 원래 처방보다 넓은 지지를
    받으면 통과한다 — 진단이 실제로 뒤집힌 경우까지 막지는 않는다.

    반환은 {되돌리는 action key: 필요한 지지 크기}. 그 축을 뒤에 다른 처방이 덮어써
    현재 config 에 남아 있지 않으면 보호하지 않는다(지킬 이득이 이미 없다).
    """
    current = canonical_config_view(index_config or {})
    thresholds: dict[str, float] = {}

    for item in optimization_history:
        # status=applied 는 KEEP 이다(롤백은 finalize 에서 failed 로 바뀐다).
        # pending 은 아직 판정 전이라 지킬 근거가 없다.
        if item.metadata.get("pending") or item.status != "applied":
            continue
        if not item.action_key:
            continue
        reverse_key = _reverse_action_key(item.action_key)
        if reverse_key is None:
            continue

        axis = item.action_key.partition(":")[0]
        applied = canonical_config_view(item.after_config or {}).get(axis, _MISSING)
        if applied is _MISSING or current.get(axis, _MISSING) != applied:
            continue

        support = item.metadata.get("weighted_probe_support")
        if not isinstance(support, (int, float)) or isinstance(support, bool):
            # 구버전 이력엔 가중 지지가 없다. probe 수가 confidence=1.0 일 때의
            # 같은 값이므로 근사치로 쓴다.
            support = float(len(item.supporting_probes))
        thresholds[reverse_key] = max(thresholds.get(reverse_key, 0.0), float(support))

    return thresholds


# ── 1. 하한선 검사 ────────────────────────────────────────────────

def check_floor(metrics: dict[str, float]) -> list[str]:
    """모든 지표를 하한선과 비교해 위반한 지표명을 반환. (비면 통과)"""
    violations: list[str] = []
    for metric, value in metrics.items():
        floor = _FLOORS.get(metric)
        if floor is None:
            continue
        if metric in _LOWER_IS_BETTER:
            if value > floor:       # 낮을수록 좋음 → 상한 초과가 위반
                violations.append(metric)
        else:
            if value < floor:       # 높을수록 좋음 → 하한 미달이 위반
                violations.append(metric)
    return violations


# ── 2. 유지/롤백 판정 ─────────────────────────────────────────────

def _read_score(report: DiagnosticReport) -> float:
    """탐색 심판(keep/rollback)이 쓰는 단일 점수 — 정규화 composite(0~1).
    composite 은 품질×신뢰도라 검색 실패에 민감해, overall(검색에 둔감)이 오르는 동안에도
    신뢰도가 무너지면 하락한다 → composite 이 내려가면 롤백이 걸린다(overall 만 보던 때의
    '오서빙' 회귀를 여기서 차단). composite 미측정(평가 신호 없음)이면 overall 로 폴백하고,
    그마저 없으면 0.0 으로 보수적으로 취급한다."""
    total = (report.composite_score or {}).get("total")
    if total is not None:
        return float(total) / 100.0
    return report.overall_score if report.overall_score is not None else 0.0


def _read_composite(report: DiagnosticReport | None) -> float | None:
    """설계 종합점수(composite_score.total, 0~100)를 읽는다. 표시·게이트용.
    (판정은 overall 로 하고 이 값은 리포트에 함께 실어주기 위한 것 — 없으면 None.)"""
    if report is None:
        return None
    total = (report.composite_score or {}).get("total")
    return float(total) if total is not None else None


def judge(
    before_report: DiagnosticReport,
    after_report: DiagnosticReport,
) -> Verdict:
    """
    처방 전후 Eval 리포트를 비교해 유지/롤백을 판정한다. (CONTEXT.md 5번)
      ① after 가 하한선을 위반하면 → 무조건 롤백 (지표별 원값으로 검사)
      ② Eval 단일 점수가 MIN_IMPROVEMENT_MARGIN 이상 올랐으면 → 유지, 아니면 → 롤백
    단일 점수는 정규화 composite(품질×신뢰도, 0~1)를 쓴다(_read_score, 여기서 재계산 안 함).
    하한선 검사가 마진보다 **먼저**다 — 하한선 위반은 점수가 얼마나 올랐든 무조건 롤백이다.

    ── 왜 마진이 필요한가 ────────────────────────────────────────────
    Eval 점수에는 LLM judge 편차·표본 노이즈가 섞여 있다. "0.0001 상승도 개선"으로
    보면 노이즈로 우연히 오른 config 가 롤백 안전망을 통과해 유지되고, 같은 축을
    늘렸다 줄였다 하는 왕복까지 양쪽 다 "개선"이 된다. max_iterations 가 몇 회뿐인
    예산을 노이즈 추적에 태우지 않도록 최소 상승폭을 요구한다.

    ── 탐색 신호를 composite 으로 통일한다 (과거엔 overall 이었음) ──────────────
    표시·게이트(gate.py)도, 탐색(이 judge)도 이제 같은 composite 을 쓴다.
    과거엔 탐색을 overall(품질 단일축)로 분리했는데, 그 이유는 "composite 의 신뢰도 축이
    이진 통과/실패 카운트라 계단 함수 → 저신뢰도 구간에서 평평해 탐색 신호가 죽는다"는
    것이었다. 그 근본 원인(이진 신뢰도)이 scoring.reliability_score 의 연속화로 제거되면서,
    composite 은 저신뢰도 구간에서 오히려 기울기가 큰 매끄러운 신호가 됐다(조화평균은
    약한 축을 강하게 반영 → 등반 방향을 또렷하게 가리킴).
    핵심 효과: overall 은 검색 실패에 둔감해, 코퍼스가 커져 검색이 새는데도(신뢰도↓)
    답변 품질만 보고 상승 → 그 사이 composite(=실제 표시·서빙 점수)이 나빠지는 config 를
    "개선"으로 오인해 유지·서빙했다(오서빙 회귀). composite 으로 심판하면 신뢰도 하락이
    곧 점수 하락이라 그런 처방이 롤백된다. "탐색·게이트·표시를 하나의 정직한 종합점수로
    통일"하는 것이 이 설계 변경의 핵심이다.
    """
    before_score = _read_score(before_report)
    after_score = _read_score(after_report)
    # 판정도 표시도 모두 composite 기준. 심판용(0~1)과 표시용(0~100)을 함께 싣는다.
    before_composite = _read_composite(before_report)
    after_composite = _read_composite(after_report)

    violations = check_floor(after_report.ragas_scores)
    if violations:
        return Verdict(
            keep=False,
            before_score=before_score,
            after_score=after_score,
            before_composite=before_composite,
            after_composite=after_composite,
            floor_violations=violations,
            reason=f"하한선 위반 {violations} → 무조건 롤백",
        )

    improvement = after_score - before_score
    if meets_improvement_margin(improvement):
        return Verdict(
            keep=True,
            before_score=before_score,
            after_score=after_score,
            before_composite=before_composite,
            after_composite=after_composite,
            reason=(
                f"종합점수 상승 {before_score:.3f}→{after_score:.3f} "
                f"(+{improvement:.3f} ≥ 마진 {MIN_IMPROVEMENT_MARGIN:.3f}) → 유지"
            ),
        )

    # 올랐지만 마진 미만이면 '노이즈로 오른 것'과 구분되지 않는다 → 개선으로 보지 않는다.
    if improvement > 0:
        return Verdict(
            keep=False,
            before_score=before_score,
            after_score=after_score,
            before_composite=before_composite,
            after_composite=after_composite,
            margin_rejected=True,
            reason=(
                f"종합점수 상승폭 부족 {before_score:.3f}→{after_score:.3f} "
                f"(+{improvement:.3f}, 필요 +{MIN_IMPROVEMENT_MARGIN:.3f}) → 롤백"
            ),
        )

    return Verdict(
        keep=False,
        before_score=before_score,
        after_score=after_score,
        before_composite=before_composite,
        after_composite=after_composite,
        reason=f"종합점수 미상승 {before_score:.3f}→{after_score:.3f} → 롤백",
    )


# ── 3. 이력 항목: pending 생성 → 확정 (2단계) ──────────────────────
# 처방이 좋았는지는 다음 Eval 재측정 후에야 알 수 있어(시점 문제) 이력을 두 단계로
# 나눈다. 적용 시점엔 create_pending_item(before 만), 다음 방문에서 finalize_item
# (after+verdict)으로 확정한다. state.optimization_history 에 append/수정하는 것은
# agent.py 가 한다(판정만 하고 반영은 조율자가 — planner 와 같은 규칙).

def create_pending_item(
    state: AgentDoctorState,
    request: OptimizationRequest,
    prescription_id: str,
    before_config: dict,
    before_report: DiagnosticReport | None,
    *,
    applied_changes: dict[str, Any] | None = None,
) -> OptimizationHistoryItem:
    """처방 적용 시점의 'pending' 이력 항목을 만든다.
    after/verdict 는 아직 없다 — 다음 방문에서 finalize_item 으로 채운다.
    before_report 는 다음 방문의 judge 에 쓰려고 metadata 에 잠시 보관한다.

    ``applied_changes`` 는 실제로 적용한 config 변경이다. attempt fingerprint 의
    입력이라 반드시 '이번에 적용한 값'이어야 한다 — 탐색 범위 전체를 넣으면 sweep
    중간 후보가 서로를 차단한다.

    action 스냅샷(지지 라벨·probe)을 함께 남기는 이유는 구현계획 §5.4 다. "이 action 이
    지지받았다"와 "그 라벨이 실제로 해결됐다"는 다른 사실이라, 다음 Eval 의 남은
    라벨과 비교하려면 적용 당시 지지 집합이 이력에 있어야 한다.
    """
    before_scores = dict(before_report.ragas_scores) if before_report else {}
    runtime_capabilities = request.metadata.get("runtime_capabilities")
    attempt_key = None
    study_key = None
    if request.action_key:
        attempt_key = build_attempt_key(
            request.action_key,
            before_config,
            applied_changes,
            runtime_capabilities,
        )
        study_key = build_study_key(
            request.action_key,
            before_config,
            request.search_space,
            runtime_capabilities,
        )
    return OptimizationHistoryItem(
        trial_id=str(uuid.uuid4()),
        request_id=request.request_id,
        iteration=state.iteration,
        failure_labels=list(request.supporting_labels),
        optimizer=request.optimizer,
        status="applied",   # config 적용됨, 다음 Eval 검증 대기 (AGENTS.md §8)
        selected_prescription_id=prescription_id,
        before_config=dict(before_config),
        after_config={},           # 미확정
        before_metrics=before_scores,
        after_metrics={},          # 비어있음
        target_metrics=list(request.target_metrics),
        reason=request.reason,
        rollback_reason=None,
        action_key=request.action_key,
        supporting_labels=list(request.supporting_labels),
        supporting_probes=list(request.supporting_probes),
        action_attempt_key=attempt_key,
        action_study_key=study_key,
        metadata={
            "pending": True,               # 아직 판정되지 않음
            "before_report": before_report,  # judge 용(MVP: 객체 참조 보관)
            "before_composite": _read_composite(before_report),  # 표시용 baseline 종합점수
            "applied_changes": dict(applied_changes or {}),
            # 이 처방이 받은 지지 크기. 유지 판정이 난 뒤 되돌리기를 견제할 때
            # 문턱으로 쓴다(reversal_guard_thresholds).
            "weighted_probe_support": request.action_score_breakdown.get(
                "weighted_probe_support"
            ),
        },
    )


def find_pending(
    history: list[OptimizationHistoryItem],
) -> OptimizationHistoryItem | None:
    """판정 대기 중(pending)인 가장 최근 이력 항목을 찾는다. 없으면 None."""
    for item in reversed(history):
        if item.metadata.get("pending"):
            return item
    return None


def find_active_study(
    history: list[OptimizationHistoryItem],
) -> OptimizationHistoryItem | None:
    """후보 sweep를 진행 중인 가장 최근 라벨 study를 찾는다.

    별도 ActiveStudy 스키마를 추가하지 않고 기존 이력 항목의 metadata를
    사용한다. ``pending``은 다음 Eval을 기다린다는 뜻이고,
    ``active_study``는 개별 후보가 아니라 전체 후보 묶음의 판정이 아직
    끝나지 않았다는 뜻이다.
    """
    for item in reversed(history):
        if item.metadata.get("pending") and item.metadata.get("active_study"):
            return item
    return None


def last_action_key(
    history: list[OptimizationHistoryItem],
) -> str | None:
    """가장 최근에 실제 적용을 시작한 action key. 구버전 이력이면 None."""
    for item in reversed(history):
        if item.action_key:
            return item.action_key
    return None


def last_action_study_key(
    history: list[OptimizationHistoryItem],
) -> ActionStudyKey | None:
    """가장 최근에 시작한 ActionStudy 의 식별자.

    iteration 은 "새 ActionStudy 를 적용한 횟수"이므로(구현계획 §5.3) 증가 판정은
    이 값과 새 요청의 study key 를 비교해 한다. action key 만 비교하면 같은 action 을
    **다른 baseline 에서** 다시 탐색하는 경우가 예산을 소비하지 않아, 롤백↔재적용
    왕복이 무한히 예산을 피해 간다.
    """
    for item in reversed(history):
        if item.action_study_key is not None:
            return item.action_study_key
    return None


def last_failure_label(
    history: list[OptimizationHistoryItem],
) -> str | None:
    """가장 최근에 실제 적용을 시작한 대표 라벨.

    ⚠️ DEPRECATED — 실행 제어는 last_action_key/last_action_study_key 가 한다.
    구버전 이력을 읽는 리포트 경로만 이 함수를 쓴다.
    """
    for item in reversed(history):
        if item.failure_labels:
            return item.failure_labels[0]
    return None


def _confirmed_labels(report: DiagnosticReport | None) -> set[str]:
    """리포트에서 확정(confirmed) 라벨 집합을 뽑는다."""
    if report is None:
        return set()
    return {
        finding.label
        for finding in report.findings
        if finding.confirmed and finding.label
    }


def attribute_result(
    item: OptimizationHistoryItem,
    before_report: DiagnosticReport | None,
    after_report: DiagnosticReport | None,
) -> dict[str, list[str]]:
    """지지받은 라벨 중 실제로 사라진 것과 남은 것을 가른다 (구현계획 §5.4).

    "이 action 이 지지받았다"와 "그 라벨이 해결됐다"는 다른 사실이다. 지지 라벨을
    그대로 성과로 보고하면 리포트가 실제보다 좋게 읽힌다. before/after 의 확정
    Finding 차집합으로만 resolved 를 계산한다 — 리포트가 없으면 계산하지 않는다
    (측정 없이 해결을 주장하지 않는다).
    """
    supporting = list(item.supporting_labels)
    if not supporting or before_report is None or after_report is None:
        return {"resolved_labels": [], "remaining_labels": list(supporting)}

    before_labels = _confirmed_labels(before_report)
    after_labels = _confirmed_labels(after_report)
    resolved = [
        label
        for label in supporting
        if label in before_labels and label not in after_labels
    ]
    remaining = [label for label in supporting if label not in resolved]
    return {"resolved_labels": resolved, "remaining_labels": remaining}


def finalize_item(
    item: OptimizationHistoryItem,
    verdict: Verdict,
    after_config: dict,
    after_report: DiagnosticReport | None,
) -> None:
    """pending 항목을 판정 결과로 확정한다(제자리 수정).
    유지면 status='applied', 롤백이면 'failed'+rollback_reason 을 기록한다.
    after_config 는 롤백 전의 '실제 적용된(=측정된)' config 다."""
    item.after_config = dict(after_config)
    item.after_metrics = dict(after_report.ragas_scores) if after_report else {}
    item.status = "applied" if verdict.keep else "failed"
    item.rollback_reason = None if verdict.keep else verdict.reason
    item.metadata["pending"] = False
    item.metadata["before_score"] = verdict.before_score
    item.metadata["after_score"] = verdict.after_score
    # 측정 자체가 성립하지 않은 처방은 품질 blacklist와 구분해 기록한다.
    # Optimize Agent가 이 값을 사용해 같은 실행 안에서 동일 no-progress 전이를
    # 다시 열지 않으며, 새 파이프라인 실행에서는 이력이 초기화되어 재시도할 수 있다.
    item.metadata["unjudgeable"] = bool(verdict.unjudgeable)
    # 마진 사후 조정용 기록. 마진이 노이즈보다 작으면 제 역할을 못 하고, 과도하게 크면
    # 진짜 작은 개선까지 버린다. 어느 쪽인지 판단하려면 탈락한 시도뿐 아니라 **유지된
    # 시도의 상승폭 분포**도 필요하므로 세 값을 모든 확정 항목에 남긴다.
    item.metadata["score_delta"] = verdict.after_score - verdict.before_score
    item.metadata["margin_rejected"] = bool(verdict.margin_rejected)
    item.metadata["improvement_margin"] = MIN_IMPROVEMENT_MARGIN
    # 표시·게이트용 종합점수(0~100). verdict 가 실어줬으면 그걸, 아니면 리포트에서 직접.
    before_report = item.metadata.get("before_report")
    item.metadata["before_composite"] = (
        verdict.before_composite if verdict.before_composite is not None
        else _read_composite(before_report)
    )
    item.metadata["after_composite"] = (
        verdict.after_composite if verdict.after_composite is not None
        else _read_composite(after_report)
    )
    item.metadata["floor_violations"] = verdict.floor_violations
    # 지지 라벨과 실제 해결 라벨을 구분해 남긴다(§5.4). before_report 를 지우기 전에.
    #
    # ⚠️ **롤백이면 해결을 주장하지 않는다.** config 를 되돌렸으므로 사라졌던 라벨의
    # 개선은 지금 설정에 남아 있지 않다. 특히 margin_rejected 롤백(점수는 올랐지만
    # 상승폭이 마진 미달)에서는 확정 Finding 이 실제로 줄어 차집합이 채워지는데,
    # 그 값을 그대로 저장하면 "되돌렸는데 해결됨"이라는 리포트가 나온다.
    #
    # 판정을 **저장 시점에** 적용하는 이유: 소비처(reporter·report_view·web_api)가
    # 여럿이고 각자 raw metadata 를 읽으므로, 표시 계층마다 같은 가드를 반복하면
    # 하나만 빠져도 경로별로 다른 값이 보인다(실제로 그렇게 새어 나갔다).
    attribution = attribute_result(item, before_report, after_report)
    if not verdict.keep:
        attribution = {
            "resolved_labels": [],
            "remaining_labels": list(item.supporting_labels or item.failure_labels),
        }
    item.metadata.update(attribution)
    item.metadata.pop("before_report", None)   # 판정 끝 — 무거운 참조 제거
