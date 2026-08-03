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

# action이 config 축에 가하는 변경의 종류.
#   increase/decrease : 방향이 고정. action key에 방향을 담는다.
#   enable/disable    : boolean 축. 현재 상태가 한쪽을 no-op으로 만들어 자동 배타된다.
#   replace           : 고정값 교체. 값은 key가 아니라 후보값으로 넘긴다 —
#                       같은 축의 여러 값을 key로 쪼개면 지지 label 집합이 같아져
#                       점수가 영원히 동률이 되고 그 축이 선택되지 않는다(starvation).
#   adjust            : 방향과 폭이 진단 실측으로 정해진다(예: shift_to_favored_channel).
ActionOperation = Literal[
    "increase",
    "decrease",
    "enable",
    "disable",
    "replace",
    "adjust",
]

# action이 실행되지 못하는 사유. 해제 조건이 다르므로 구분해 기록한다.
#   not_state_mappable : config_mapper 계약 부재. mapper와 소비 노드가 함께 필요하다.
#   capability_off     : 소비 경로는 있으나 검증된 후보/구현이 없다. capability 값만
#                        바꾸면 열린다.
#   runtime_unavailable: 이번 실행의 runtime capability 미검증. 품질 실패와 구분한다.
ActionBlockedReason = Literal[
    "not_state_mappable",
    "capability_off",
    "runtime_unavailable",
]

# aggregate 단계에서 action이 놓인 상태.
ActionCandidateStatus = Literal["ready", "blocked", "conflicted"]

# 사용자가 어떤 최적화 성향을 우선하는지 나타낸다.
TargetProfile = Literal["accuracy", "speed", "cost", "balanced"]

# optimizer.py가 선택할 수 있는 최적화 backend 종류.
OptimizerBackend = Literal["rules", "internal", "ragbuilder", "autorag"]

# 자체 optimizer가 사용하는 목적함수 방향과 실행 상태.
ObjectiveDirection = Literal["maximize", "minimize"]
InternalAdapterStatus = Literal[
    "needs_evaluation",
    "completed",
    "failed",
    "skipped",
]
InternalTrialStatus = Literal[
    "completed",
    "failed",
    "inconclusive",
    "rejected",
]

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
        metadata: prescription_ids 같은 mapper/adapter용 확장 정보.
    """

    changes: dict[str, Any]
    target: ConfigTarget = "index_config"
    reindex_required: bool = False
    description: str = ""
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectedAction:
    """optimizer 가 실행 직전 검증까지 마치고 준비한 **변경 하나**.

    `PrescriptionCandidate`(라벨별 처방 후보 목록)를 대체한다. 선택 단위가 action 이
    된 뒤 planner 는 이미 하나를 고른 상태로 요청을 보내므로, optimizer 가 후보
    목록을 순회하며 "어떤 처방을 먼저 시도할지" 다시 정하는 계층이 사라졌다.
    남은 책임은 "이 변경이 지금 실행 가능한가"의 재검증뿐이다.

    Attributes:
        action_key: 실제 config 변경의 canonical 식별자.
        prescription_id: 이 변경을 선언한 rules.py 처방 id. 리포트가 선언으로
            되짚기 위한 표시용이며 실행 제어에는 쓰지 않는다.
        description: 사람이 읽는 변경 설명.
        reindex_required: 적용 후 재색인이 필요한지.
        search_space: constraint·no-op 필터를 통과한 최종 후보값. 단일 축이다.
    """

    action_key: str | None = None
    prescription_id: str | None = None
    description: str = ""
    reindex_required: bool = False
    search_space: dict[str, list[Any]] = field(default_factory=dict)


# ── Action 중심 모델 ──────────────────────────────────────────────
# 선택 단위를 failure label에서 실제 config action으로 옮기기 위한 모델이다.
# label은 진단 근거·영향 probe·목표 metric을 제공하고, action이 우선순위 경쟁·
# 후보값·적용·이력·차단의 중심이 된다.
# (설계: agents/optimize/ACTION_CENTERED_OPTIMIZER_IMPLEMENTATION_PLAN.md)


@dataclass(frozen=True)
class ActionDefinition:
    """실제 config 변경 하나의 정적 정의. label과 독립적으로 한 번만 선언한다.

    같은 config 변경이 여러 label에서 다른 이름으로 선언되던 것을 여기로 모은다.
    reindex/cost/capability/conflict family의 단일 진실 원천이다.

    key 규칙은 ``<canonical_path>:<operation>``이며 label이나 기존 prescription id를
    넣지 않는다. 고정값도 넣지 않는다(replace 주석 참고).
    """

    key: str
    canonical_path: str
    operation: ActionOperation
    description: str
    # 재색인 필요 여부. base_cost의 판정 근거이며 optimizer.REINDEX_PATHS와 일치해야 한다.
    reindex_required: bool = False
    # 우선순위 점수의 분모. 현재는 재색인 여부로만 유도한다(재색인 3 / 런타임 1).
    # rules.py의 cost가 전부 None이라 실측 근거가 없기 때문이며, 근거 없는 숫자를
    # 새로 만들지 않는다. 실측 기반 세분화는 confidence 생산과 함께 별도 작업이다.
    base_cost: float = 1.0
    # 이 경로가 요구하는 pipeline capability(optimizer.PATH_CAPABILITIES 기준).
    capability: str | None = None
    # 같은 축을 공유해 서로 경쟁할 수 있는 action 묶음. 보통 canonical_path와 같다.
    conflict_family: str = ""
    # 실행 전에 만족해야 하는 조건(예: candidate_count는 reranker가 켜져 있어야 한다).
    prerequisites: tuple[str, ...] = ()
    # 실행 불가 사유. None이면 실행 가능하다.
    blocked_reason: ActionBlockedReason | None = None
    # 차단 사유의 상세(어느 capability인지 등). 리포트와 catalog 검증에 쓴다.
    blocked_detail: str = ""
    tradeoffs: tuple[str, ...] = ()

    @property
    def is_blocked(self) -> bool:
        return self.blocked_reason is not None


@dataclass
class ActionSupport:
    """한 label 묶음이 특정 action에 제공하는 런타임 근거.

    Eval은 Finding을 probe마다 따로 만든다(affected_probes는 항상 1개). 같은 label의
    Finding 여러 개를 먼저 support 하나로 묶은 뒤, 같은 action을 지지하는 support들을
    ActionCandidate로 통합한다.
    """

    action_key: str
    label: str
    group: FailureGroup | None = None
    # 이 support 를 만든 rules.py 처방 id. action 이 선택 단위가 된 뒤에도 리포트와
    # 하위 호환 필드를 채우려면 "어느 선언에서 왔는지"를 알아야 한다.
    prescription_id: str = ""
    finding_ids: list[str] = field(default_factory=list)
    # 이 label이 영향을 준 고유 probe. 점수는 label 수가 아니라 이 집합으로 센다.
    affected_probes: set[str] = field(default_factory=set)
    # rules.py diagnosis_confidence. 현재 전부 None이라 1.0으로 채워진다.
    confidence: float = 1.0
    # confidence의 출처. 나중에 실측 confidence가 생산되면 이력에서 구분할 수 있다.
    confidence_source: str = "default"
    target_metrics: list[str] = field(default_factory=list)
    # 이 support가 제안하는 구체 후보값. 근거값 계산 결과이거나 방향 키워드 폴백이다.
    candidate_values: list[Any] = field(default_factory=list)
    applies_when: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    # 후보값이 어떻게 나왔는지(status, source, 분포 등). 리포트와 디버깅용.
    grounding_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_grounded(self) -> bool:
        """측정값에 근거한 후보인지(방향 키워드 추측이 아닌지)."""
        return self.grounding_metadata.get("status") in {
            "grounded",
            "explicit_candidates",
        }


@dataclass
class ActionCandidate:
    """같은 action key를 지지하는 support를 통합한 실제 선택 단위."""

    action_key: str
    definition: ActionDefinition
    supports: list[ActionSupport] = field(default_factory=list)
    # 이 action을 지지하는 label과 고유 probe. 점수와 설명의 근거다.
    supporting_labels: list[str] = field(default_factory=list)
    supporting_probes: set[str] = field(default_factory=set)
    # 같은 축에서 반대 방향을 지지하는 action과 그 label.
    opposing_action_keys: list[str] = field(default_factory=list)
    opposing_labels: list[str] = field(default_factory=list)
    # {canonical_path: [후보값...]}. 단일 축만 담는다.
    search_space: dict[str, list[Any]] = field(default_factory=dict)
    target_metrics: list[str] = field(default_factory=list)
    score: float = 0.0
    # 점수 구성 요소. 사용자에게 선택 이유를 설명하는 데 쓴다.
    score_breakdown: dict[str, Any] = field(default_factory=dict)
    status: ActionCandidateStatus = "ready"
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def causal_rank_group(self) -> FailureGroup | None:
        """이 action을 지지하는 label 중 가장 높은 우선순위 그룹.

        A > C > B 인과는 점수보다 먼저 적용되는 1차 정렬 키다. 검색이 새는 상태에서
        생성 처방을 먼저 적용하면 garbage-in tuning이 되기 때문이다.
        """
        order: dict[str, int] = {"A": 0, "C": 1, "B": 2, "D": 3}
        groups = [s.group for s in self.supports if s.group]
        if not groups:
            return None
        return min(groups, key=lambda g: order.get(g, 99))


@dataclass(frozen=True)
class ActionAttemptKey:
    """정확한 config 전이 하나. 품질 실패로 차단할 단위다.

    ``(label, prescription_id)``와 달리 baseline과 candidate를 포함하므로, 같은
    action이라도 다른 baseline에서는 다시 시도할 수 있다. 즉 차단을 강화하는 것이
    아니라 baseline별로 완화하는 식별자다.
    """

    action_key: str
    baseline_fingerprint: str
    candidate_fingerprint: str


@dataclass(frozen=True)
class ActionStudyKey:
    """한 baseline에서 특정 search space를 이미 탐색했음을 나타낸다.

    sweep을 정상 완료한 뒤 같은 탐색을 다시 시작하지 않기 위한 식별자다.
    """

    action_key: str
    baseline_fingerprint: str
    search_space_fingerprint: str


@dataclass
class SkippedAction:
    """실행 단계에서 제외된 action과 그 사유."""

    action_key: str
    reason: str
    target: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkippedPrescription:
    """
    config_mapper가 실행 가능한 patch로 변환하지 못한 처방.

    속성:
        prescription_id: rule 수준의 prescription id.
        reason: 짧은 machine-readable skip 사유.
        target: 알 수 있는 경우, 변경하려던 내부 config path.
        metadata: log, adapter, report에서 쓸 수 있는 추가 정보.
    """

    prescription_id: str
    reason: str
    target: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigMappingResult:
    """
    config_mapper가 prescription을 patch로 변환한 뒤 반환하는 결과.

    속성:
        patches: 실행 가능한 AgentDoctor 표준 config patch 후보.
        search_space: optimizer/adapter가 사용할 path별 후보값 목록.
        skipped: 지원하지 않거나, 유효하지 않거나, 이미 만족되어 건너뛴 처방.
        warnings: 치명적이지 않은 mapping warning 목록.
        metadata: 선택적인 debug/trace 정보.
    """

    patches: list[ConfigPatch] = field(default_factory=list)
    search_space: dict[str, list[Any]] = field(default_factory=dict)
    skipped: list[SkippedPrescription] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGBuilderTrialResult:
    """
    RAGBuilder가 실행한 단일 trial 결과.

    이 값은 surrogate pipeline에서의 최적화 결과일 뿐이다. 실제 사용자
    pipeline에 적용하려면 ConfigPatch 후보로 변환한 뒤 eval 검증을 거쳐야 한다.
    """

    trial_id: str
    config: dict[str, Any]
    score: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    status: Literal["completed", "failed", "rejected", "unsupported"] = "completed"
    unsupported_reasons: list[str] = field(default_factory=list)
    raw_trial: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGBuilderResult:
    """
    RAGBuilder adapter가 반환하는 표준화된 외부 최적화 결과.

    RAGBuilderResult는 surrogate RAGBuilder pipeline에서 유망했던 config 조합을
    설명한다. 사용자 pipeline에서 실제 개선이 있었는지 판단하는 최종
    ValidationResult가 아니다.
    """

    request_id: str
    best_config: dict[str, Any] | None
    best_score: float | None
    trial_results: list[RAGBuilderTrialResult] = field(default_factory=list)
    optimized_stage: str = "retrieval"
    search_space: dict[str, list[Any]] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    raw_result: dict[str, Any] = field(default_factory=dict)
    status: Literal["completed", "failed", "skipped"] = "completed"
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InternalTrialResult:
    """
    AgentDoctor 자체 optimizer가 관리하는 단일 trial 결과.

    아직 평가하지 않은 후보는 이 모델에 넣지 않는다. ``config``는 canonical
    경로로 표현한 단일 축 변경이며, baseline trial은 빈 dict와
    ``is_baseline=True``를 사용한다.
    """

    trial_id: str
    config: dict[str, Any]
    score: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    status: InternalTrialStatus = "completed"
    is_baseline: bool = False
    fingerprint: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InternalAdapterResult:
    """
    자체 search-space optimizer의 표준 결과.

    ``next_config``는 아직 실제 Eval이 필요한 후보이고, ``best_config``는
    평가가 끝난 trial 중에서 선택된 설정이다. 두 값을 분리해 미평가 후보를
    최적 설정으로 오인하지 않게 한다.
    """

    request_id: str
    status: InternalAdapterStatus
    next_config: dict[str, Any] | None = None
    best_config: dict[str, Any] | None = None
    best_score: float | None = None
    trial_results: list[InternalTrialResult] = field(default_factory=list)
    # 실제 탐색 objective 는 planner 가 primary_metric=composite_score(정규화 0~1)로 구동한다
    # — 신뢰도 축이 연속값이 된 뒤 composite 이 매끄러워져 표시·게이트와 같은 지표로 통일됨
    # (history.judge · scoring.reliability_score 주석 참고). 이 필드 기본값은 result 가
    # 항상 실제 objective 로 덮어쓰므로 하위호환 안전망일 뿐이다.
    objective_metric: str = "overall_score"
    direction: ObjectiveDirection = "maximize"
    search_space: dict[str, list[Any]] = field(default_factory=dict)
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationRequest:
    """
    planner가 만들고 optimizer/adapter가 소비하는 최적화 요청.

    Eval 결과에서 우선순위가 가장 높은 failure label 하나를 고른 뒤,
    현재 baseline config, 처방 후보, search space를 하나로 묶은 wrapper다.
    rules backend는 candidates와 search_space에서 검증된 단일 변경을 고르고,
    RAGBuilder/AutoRAG adapter는 search_space와 fixed_config를 외부 도구 입력
    형식으로 변환한다. internal은 자체 evaluator 또는 이전 trial 관측을 이용해
    다음 후보와 평가 완료된 best config를 선택한다.

    Attributes:
        request_id: 최적화 요청 고유 ID.
        iteration: 현재 optimize 반복 회차.
        baseline_config: 변경 전 기준 config.
        action_key: 이번 요청이 적용하려는 실제 config 변경. 선택의 단위다.
        supporting_labels: 그 변경을 지지한 진단 라벨 전체.
        search_space: optimizer가 탐색할 수 있는 config 후보 범위. 단일 축이다.
        fixed_config: 최적화 중 고정해야 하는 config 값.
        target_metrics: 개선해야 하는 목표 지표 목록.
        target_profile: 사용자의 최적화 성향. 예: accuracy, speed, cost, balanced.
        optimizer: 사용할 backend. 예: rules, internal, ragbuilder, autorag.
        max_trials: 최대 탐색/시도 횟수.
        reason: 요청 생성 이유.
        propose_only: True이면 실제 적용하지 않고 제안만 생성한다.
        metadata: adapter별 추가 입력이나 실험 정보를 담는 확장 필드.
        prescription_id: 이 변경을 선언한 rules.py 처방 id(표시용).
    """

    request_id: str
    iteration: int
    baseline_config: dict[str, Any]
    search_space: dict[str, Any] = field(default_factory=dict)
    fixed_config: dict[str, Any] = field(default_factory=dict)
    target_metrics: list[str] = field(default_factory=list)
    target_profile: TargetProfile = "balanced"
    optimizer: OptimizerBackend = "rules"
    max_trials: int = 1
    reason: str = ""
    propose_only: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── action 중심 필드 ──────────────────────────────────────────
    # 이 요청이 적용하려는 실제 config 변경. 선택의 단위이자 이력·차단의 identity 다.
    action_key: str | None = None
    action: ActionCandidate | None = None
    # 이 action 을 지지한 라벨과 고유 probe. 대표 라벨 하나가 아니라 전체다.
    supporting_labels: list[str] = field(default_factory=list)
    supporting_probes: list[str] = field(default_factory=list)
    # 같은 config 축에서 반대 방향을 지지한 라벨(있다면). 리포트가 충돌을 설명한다.
    opposing_labels: list[str] = field(default_factory=list)
    # 선택 근거. 사용자에게 "왜 이걸 골랐는지" 설명하는 데 쓴다.
    action_score: float | None = None
    action_score_breakdown: dict[str, Any] = field(default_factory=dict)
    # 이 변경을 선언한 rules.py 처방 id. 리포트가 선언으로 되짚기 위한 표시용이며
    # 실행 제어에는 쓰지 않는다(구현계획 §8.2 완료 조건).
    prescription_id: str | None = None


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
        selected_action: optimizer 가 실행 직전 검증을 마치고 준비한 변경 하나.
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
    selected_action: SelectedAction | None = None
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
        manual_labels: 이번 진단에서 함께 발견된 D그룹(manual) 라벨들.
            apply_optimize로 자동 처방이 진행되는 경우에도, 별도로 사람이
            확인해야 할 문제가 있으면 여기 담아 reporter가 사용자에게 알린다.
        metadata: 결정의 부가 근거. 특히 실행 가능한 action이 하나도 남지 않아
            request가 None인 경우, 어떤 action이 왜 제외됐는지(rejected_actions)와
            어떤 축이 충돌로 보류됐는지(deferred_axes)를 여기 담는다. request가
            없으면 그 설명을 실을 곳이 여기뿐이다 — 없으면 "처방 없음"만 남고
            사용자는 이유를 알 수 없다.
    """

    mode: DecisionMode
    status: OptimizationStatus
    requires_user_confirmation: bool
    request_id: str | None = None
    next_route: NextRoute = "serve"
    reason: str = ""
    manual_labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Verdict:
    """
    history.judge가 내리는 처방 유지/롤백 판정 결과.

    OptimizeDecision이 planner의 흐름 결정 봉투인 것과 대응된다. 한 처방을
    적용해 재측정한 뒤, 그 처방을 유지할지 되돌릴지를 담는다. agent.py가 이
    값을 보고 실제 롤백(config 복원)·블랙리스트 등록을 수행한다.

    Attributes:
        keep: True면 유지, False면 롤백.
        before_score: 처방 전 단일 점수(Eval overall_score). 최적화 탐색 신호(0~1).
        after_score: 처방 후 단일 점수(Eval overall_score). 최적화 탐색 신호(0~1).
        before_composite: 처방 전 설계 종합점수(composite_score.total, 0~100).
        after_composite: 처방 후 설계 종합점수(composite_score.total, 0~100).
            유지/롤백 판정은 overall(탐색 신호)로 하되, 표시·게이트용으로 composite 를
            함께 실어 리포트가 사용자에게 정직한 종합점수를 보여줄 수 있게 한다.
        floor_violations: 하한선을 위반한 지표명 목록. 있으면 무조건 롤백.
        reason: 이 판정을 내린 이유(사람이 읽는 설명).
        unjudgeable: 리포트 부재로 '측정 자체가 없어' 롤백한 경우 True.
            처방이 나빴다는 증거가 아니라 판정이 불가했다는 뜻이므로, config 복원은
            하되 블랙리스트 등록은 건너뛴다(무죄추정). 정상 판정(유지/롤백)은 False.
        margin_rejected: 점수가 오르긴 했으나 상승폭이
            history.MIN_IMPROVEMENT_MARGIN 미만이라 롤백한 경우 True. 마진 값이
            노이즈보다 과도하게 큰지 사후 검증하기 위한 기록이며, 판정 자체는
            일반 롤백과 같다(하락으로 롤백한 경우는 False).
    """

    keep: bool
    before_score: float
    after_score: float
    before_composite: float | None = None
    after_composite: float | None = None
    floor_violations: list[str] = field(default_factory=list)
    reason: str = ""
    unjudgeable: bool = False
    margin_rejected: bool = False


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
            ⚠️ DEPRECATED — 실제 선택 단위는 action_key다. rules.py 선언으로 되짚기
            위한 표시용으로 남는다.
        config_changes: 사용자에게 보여줄 config 변경 요약.
        expected_tradeoffs: latency, cost, precision 등 예상되는 영향.
        manual_actions: 사용자가 직접 해야 하는 조치 목록.
        next_steps: 이후 흐름 안내. 예: reindex, serve, manual review.
        diff: config 적용 전후 차이. 제안만 한 경우 None일 수 있다.
        metadata: UI 표시용 세부 정보나 원본 result 정보.
        created_at: 리포트 생성 시각.

        ── action 중심 설명 필드 ──────────────────────────────────
        action_key: 이번에 바꾼(또는 바꾸려는) 실제 config 변경.
        supporting_labels: 그 변경을 지지한 진단 라벨 전체. 대표 하나가 아니다 —
            여러 라벨이 같은 변경을 지지했다는 사실이 선택 근거이기 때문이다.
        opposing_labels: 같은 축에서 반대 방향을 지지한 라벨. 왜 이쪽을 골랐는지
            설명하려면 반대편도 보여야 한다.
        resolved_labels / remaining_labels: 지지받은 라벨 중 실제로 사라진 것과
            남은 것. "지지받았다"와 "해결됐다"는 다른 사실이라 구분해 보고한다.
        score_breakdown: 선택 점수의 구성 요소(고유 probe 수·가중 지지·비용 출처 등).
        deferred_axes: 근소한 차이로 이번 방문에서 보류한 축과 그 이유.
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

    action_key: str | None = None
    supporting_labels: list[str] = field(default_factory=list)
    opposing_labels: list[str] = field(default_factory=list)
    resolved_labels: list[str] = field(default_factory=list)
    remaining_labels: list[str] = field(default_factory=list)
    score_breakdown: dict[str, Any] = field(default_factory=dict)
    deferred_axes: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class OptimizationHistoryItem:
    """
    최적화 1회 시도에 대한 이력 기록.

    같은 처방을 반복 적용하지 않도록 하거나, 전역 하한선 위반/점수 하락으로
    rollback할 때 필요한 정보를 저장한다. 초기 구현에서는 state에 dict로 저장하더라도,
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
        reason: 해당 시도를 수행한 이유.
        rollback_reason: rollback했다면 그 이유.
        created_at: 이력 생성 시각.
        metadata: blacklist, adapter 응답 등 확장 정보.
    """

    trial_id: str
    request_id: str
    iteration: int
    # DEPRECATED: 선택 단위가 action 으로 옮겨졌다. 이 목록의 첫 원소를 "대표 라벨"로
    #   읽던 실행 제어는 action_key 로 대체됐고, 여기는 설명·구버전 호환용으로 남는다.
    #   지지 라벨 전체는 supporting_labels 를 읽어야 한다.
    failure_labels: list[str]
    optimizer: OptimizerBackend
    status: OptimizationStatus
    # DEPRECATED: 실행 제어는 action_key/attempt·study key 가 소유한다(구현계획 §8.2).
    selected_prescription_id: str | None = None
    before_config: dict[str, Any] = field(default_factory=dict)
    after_config: dict[str, Any] = field(default_factory=dict)
    before_metrics: dict[str, Any] = field(default_factory=dict)
    after_metrics: dict[str, Any] = field(default_factory=dict)
    target_metrics: list[str] = field(default_factory=list)
    reason: str = ""
    rollback_reason: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── action 중심 필드 (구현계획 §5.4 결과 귀속) ────────────────────
    # 이 시도가 실제로 바꾼 config 변경. 실행·이력·차단 identity 의 정본이다.
    action_key: str | None = None
    # 적용 당시 이 action 을 지지한 라벨과 고유 probe 스냅샷. 다음 Eval 의 남은
    # 라벨과 비교해 "지지받았다"와 "실제로 해결됐다"를 구분한다.
    supporting_labels: list[str] = field(default_factory=list)
    supporting_probes: list[str] = field(default_factory=list)
    # 정확한 config 전이와 탐색 범위의 식별자. 재선택 차단에 쓴다.
    action_attempt_key: ActionAttemptKey | None = None
    action_study_key: ActionStudyKey | None = None
