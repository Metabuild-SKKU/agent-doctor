"""
agents/optimize/schemas.py
Optimize 모듈에서 공통으로 사용하는 데이터 모델.

이 파일은 planner, optimizer, adapters, config_mapper, history, reporter가
서로 주고받는 데이터의 형태를 정의한다. 실행 로직은 넣지 않고, 순환 참조를
막기 위해 optimize 내부의 다른 모듈도 import하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


# ConfigPatch가 어느 state config 영역을 수정하는지 나타낸다.
ConfigTarget = Literal["index_config", "generation_config", "serve_config"]

# 진단 라벨의 큰 분류. A=검색, B=생성, C=context 구조, D=데이터/평가 문제.
FailureGroup = Literal["A", "B", "C", "D"]

# 처방 규칙의 실행 가능 상태. ready만 자동 적용 대상이다.
PrescriptionStatus = Literal["ready", "draft", "unassigned", "manual"]

# 사용자가 어떤 최적화 성향을 우선하는지 나타낸다.
TargetProfile = Literal["accuracy", "speed", "cost", "balanced"]

# optimizer.py가 선택할 수 있는 최적화 backend 종류.
OptimizerBackend = Literal["internal", "ragbuilder", "autorag"]

# optimize를 제안만 할지, 실제 적용할지, 수동 처리할지 결정하는 모드.
DecisionMode = Literal[
    "propose_only",
    "apply_optimize",
    "use_current",
    "manual_required",
]

# optimize 이후 graph/agent가 이동할 다음 단계.
NextRoute = Literal["index", "serve", "end"]

# 최적화 요청, 결과, 이력, 리포트에서 공유하는 처리 상태.
OptimizationStatus = Literal[
    "proposed",
    "applied",
    "already_optimal",
    "manual_required",
    "failed",
    "skipped",
]


@dataclass
class ConfigPatch:
    """
    처방이 만들어낸 config 변경 조각.

    Attributes:
        changes: 실제로 바꿀 key-value 목록. 예: {"top_k": "increase"}.
        target: 변경 대상 config 영역. 현재는 주로 index_config를 사용한다.
        reindex_required: 이 변경을 적용한 뒤 재색인이 필요한지 여부.
        description: 사용자나 로그에 보여줄 변경 설명.
        warnings: 아직 state에 없는 key, 재색인 필요 같은 주의사항 목록.
    """

    changes: dict[str, Any]
    target: ConfigTarget = "index_config"
    reindex_required: bool = False
    description: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class PrescriptionCandidate:
    """
    하나의 진단 라벨에 대해 시도할 수 있는 처방 후보.

    rules.py의 raw dict 처방을 optimizer/adapter가 쓰기 쉬운 형태로 감싼다.
    한 라벨에 여러 후보가 있을 수 있으며, planner나 optimizer는 priority,
    cost, patch.reindex_required 등을 보고 어떤 후보를 먼저 적용할지 결정한다.

    Attributes:
        id: 처방 식별자. 예: "enable_reranker", "increase_top_k".
        failure_label: 이 처방이 대응하는 진단 라벨.
        group: 라벨 그룹. A=검색, B=생성, C=context 구조, D=데이터/평가 문제.
        status: ready, draft, manual 등 현재 처방 실행 가능 상태.
        patch: 실제 config 변경 내용. 수동 조치 라벨이면 None일 수 있다.
        search_space: RAGBuilder/AutoRAG에 넘길 탐색 후보 영역.
        cost: 처방 비용. 낮을수록 우선 시도하기 쉽다.
        priority: planner가 계산한 후보 우선순위 점수.
        target_metrics: 이 처방으로 개선하려는 주요 지표 목록.
        guardrails: 악화 여부를 감시해야 하는 보호 지표 목록.
        reason: 이 처방을 제안한 이유.
        tradeoffs: latency, cost, precision 하락 등 예상되는 부작용.
        metadata: 실험적 신호나 원본 rule dict 같은 확장 정보.
    """

    id: str
    failure_label: str
    group: FailureGroup
    status: PrescriptionStatus
    patch: ConfigPatch | None = None
    search_space: dict[str, Any] = field(default_factory=dict)
    cost: float | None = None
    priority: float = 0.0
    target_metrics: list[str] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)
    reason: str = ""
    tradeoffs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationRequest:
    """
    planner가 만들고 optimizer/adapter가 소비하는 최적화 요청.

    Eval 결과에서 우선순위가 가장 높은 failure label 하나를 고른 뒤,
    현재 baseline config, 처방 후보, search space를 하나로 묶은 wrapper다.
    internal adapter는 candidates를
    직접 사용하고, RAGBuilder/AutoRAG adapter는 search_space와 fixed_config를
    외부 도구 입력 형식으로 변환한다.

    Attributes:
        request_id: 최적화 요청 고유 ID.
        iteration: 현재 optimize 반복 회차.
        baseline_config: 변경 전 기준 config.
        failure_label: 이번 요청에서 해결하려는 대표 진단 라벨.
        related_failure_labels: 같은 Eval report에서 함께 관찰된 관련 라벨 목록.
        candidates: planner가 선정한 처방 후보 목록.
        search_space: optimizer가 탐색할 수 있는 config 후보 범위.
        fixed_config: 최적화 중 고정해야 하는 config 값.
        target_metrics: 개선해야 하는 목표 지표 목록.
        guardrails: 성능 악화를 막기 위해 감시할 지표 목록.
        target_profile: 사용자의 최적화 성향. 예: accuracy, speed, cost, balanced.
        optimizer: 사용할 backend. 예: internal, ragbuilder, autorag.
        max_trials: 최대 탐색/시도 횟수.
        reason: 요청 생성 이유.
        propose_only: True이면 실제 적용하지 않고 제안만 생성한다.
        metadata: adapter별 추가 입력이나 실험 정보를 담는 확장 필드.
    """

    request_id: str
    iteration: int
    baseline_config: dict[str, Any]
    failure_label: str
    related_failure_labels: list[str] = field(default_factory=list)
    candidates: list[PrescriptionCandidate] = field(default_factory=list)
    search_space: dict[str, Any] = field(default_factory=dict)
    fixed_config: dict[str, Any] = field(default_factory=dict)
    target_metrics: list[str] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)
    target_profile: TargetProfile = "balanced"
    optimizer: OptimizerBackend = "internal"
    max_trials: int = 1
    reason: str = ""
    propose_only: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """
    optimizer/adapter가 반환하는 표준 결과.

    config_mapper는 config_patch 또는 best_config를 사용해 state에 반영하고,
    reporter는 status/message/tradeoff 정보를 사용자 요약으로 바꾸며,
    history는 before/after 정보를 저장한다.

    Attributes:
        request_id: 이 결과가 대응하는 OptimizationRequest ID.
        status: proposed, applied, manual_required, failed 등 결과 상태.
        optimizer: 실제 실행한 backend 이름.
        selected_candidate: 최종 선택된 처방 후보.
        config_patch: 현재 config에 병합할 변경 조각.
        best_config: 외부 optimizer가 반환한 전체 최적 config.
        before_metrics: 적용 전 평가 지표.
        after_metrics: 적용 후 평가 지표.
        improved: 목표 지표가 개선됐는지 여부. 아직 평가 전이면 None.
        needs_reindex: 결과 적용 후 Index 단계 재실행이 필요한지 여부.
        message: 사용자/로그용 요약 메시지.
        error: 실패한 경우의 에러 메시지.
        metadata: adapter 원본 응답 등 확장 정보.
    """

    request_id: str
    status: OptimizationStatus
    optimizer: OptimizerBackend
    selected_candidate: PrescriptionCandidate | None = None
    config_patch: ConfigPatch | None = None
    best_config: dict[str, Any] | None = None
    before_metrics: dict[str, Any] = field(default_factory=dict)
    after_metrics: dict[str, Any] = field(default_factory=dict)
    improved: bool | None = None
    needs_reindex: bool = False
    message: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigDiff:
    """
    config_mapper가 만든 config 적용 전후 차이.

    OptimizationResult의 config_patch 또는 best_config를 실제 state config에
    적용했을 때 무엇이 바뀌었고, 무엇이 무시됐는지 기록한다. reporter는 이
    정보를 사용자에게 보여주고, history는 rollback을 위해 before/after config를
    저장한다.

    Attributes:
        before_config: 변경 적용 전 config.
        after_config: 변경 적용 후 config.
        changed_keys: 값이 바뀐 config key 목록.
        added_keys: 새로 추가된 config key 목록.
        removed_keys: 제거된 config key 목록. MVP에서는 거의 사용하지 않는다.
        ignored_keys: state가 아직 지원하지 않아 적용하지 않은 key 목록.
        warnings: 적용 중 발생한 주의사항. 예: unknown key, requires_reindex.
        metadata: mapper 내부 판단이나 adapter 원본 정보를 담는 확장 필드.
    """

    before_config: dict[str, Any]
    after_config: dict[str, Any]
    changed_keys: list[str] = field(default_factory=list)
    added_keys: list[str] = field(default_factory=list)
    removed_keys: list[str] = field(default_factory=list)
    ignored_keys: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizeDecision:
    """
    optimize 이후 흐름을 결정하는 분기 정보.

    사용자가 제안만 받을지, 실제 optimize를 적용할지, 현재 설정을 유지할지,
    수동 조치가 필요한지를 명시한다. agent.py나 graph 조건부 분기에서
    next_route를 참고해 index, serve, end 중 다음 흐름을 선택할 수 있다.

    Attributes:
        mode: propose_only, apply_optimize, use_current, manual_required 중 하나.
        status: 현재 결정에 따른 결과 상태.
        requires_user_confirmation: 사용자의 명시적 확인이 필요한지 여부.
        request_id: 연결된 OptimizationRequest ID. 없을 수 있다.
        next_route: 다음 그래프 흐름. index, serve, end 중 하나.
        reason: 이 결정을 내린 이유.
    """

    mode: DecisionMode
    status: OptimizationStatus
    requires_user_confirmation: bool
    request_id: str | None = None
    next_route: NextRoute = "serve"
    reason: str = ""


@dataclass
class OptimizationReport:
    """
    reporter가 생성하는 사용자용 처방 요약.

    최적화 결과를 그대로 노출하지 않고, 문제 원인, 적용/제안된 처방,
    config 변경점, 예상 trade-off, 수동 조치가 필요한 항목을 사람이 읽기 쉬운
    구조로 정리한다. CLI/API/UI 어디로 내보내든 같은 구조를 사용할 수 있다.

    Attributes:
        report_id: 처방 리포트 고유 ID.
        request_id: 연결된 OptimizationRequest ID.
        status: 리포트가 설명하는 최적화 결과 상태.
        summary: 한두 문장짜리 전체 요약.
        problem: 진단된 핵심 문제 원인 설명.
        selected_prescription: 선택된 처방 ID 또는 이름.
        config_changes: 사용자에게 보여줄 config 변경 요약.
        expected_tradeoffs: latency, cost, precision 등 예상되는 영향.
        manual_actions: 사용자가 직접 해야 하는 조치 목록.
        next_steps: 이후 흐름 안내. 예: reindex, serve, manual review.
        diff: config 적용 전후 차이. 제안만 한 경우 None일 수 있다.
        metadata: UI 표시용 세부 정보나 원본 result 정보.
        created_at: 리포트 생성 시각.
    """

    report_id: str
    request_id: str
    status: OptimizationStatus
    summary: str
    problem: str = ""
    selected_prescription: str | None = None
    config_changes: list[str] = field(default_factory=list)
    expected_tradeoffs: list[str] = field(default_factory=list)
    manual_actions: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    diff: ConfigDiff | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationHistoryItem:
    """
    최적화 1회 시도에 대한 이력 기록.

    같은 처방을 반복 적용하지 않도록 하거나, guardrail 위반 시 rollback할 때
    필요한 정보를 저장한다. 초기 구현에서는 state에 dict로 저장하더라도,
    이 모델을 기준으로 history.py에서 직렬화하면 된다.

    Attributes:
        trial_id: 최적화 시도 고유 ID.
        request_id: 연결된 OptimizationRequest ID.
        iteration: 최적화 반복 회차.
        failure_labels: 이 시도에서 대상으로 삼은 진단 라벨 목록.
        optimizer: 사용한 backend 이름.
        status: 해당 시도의 결과 상태.
        selected_prescription_id: 적용하거나 제안한 처방 ID.
        before_config: 처방 적용 전 config.
        after_config: 처방 적용 후 config.
        before_metrics: 처방 적용 전 평가 지표.
        after_metrics: 처방 적용 후 평가 지표.
        target_metrics: 개선 목표였던 지표 목록.
        guardrail_metrics: rollback 판단에 사용한 보호 지표 목록.
        reason: 해당 시도를 수행한 이유.
        rollback_reason: rollback했다면 그 이유.
        created_at: 이력 생성 시각.
        metadata: blacklist, adapter 응답 등 확장 정보.
    """

    trial_id: str
    request_id: str
    iteration: int
    failure_labels: list[str]
    optimizer: OptimizerBackend
    status: OptimizationStatus
    selected_prescription_id: str | None = None
    before_config: dict[str, Any] = field(default_factory=dict)
    after_config: dict[str, Any] = field(default_factory=dict)
    before_metrics: dict[str, Any] = field(default_factory=dict)
    after_metrics: dict[str, Any] = field(default_factory=dict)
    target_metrics: list[str] = field(default_factory=list)
    guardrail_metrics: list[str] = field(default_factory=list)
    reason: str = ""
    rollback_reason: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
