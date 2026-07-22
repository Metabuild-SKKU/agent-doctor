"""
AgentDoctor의 최적화 요청을 RAGBuilder 실행 형식으로 변환하는 어댑터.

현재 진행 상황(2026-07-14):

[구현 완료]
- RAGBuilder 0.1.6 기준 request 검증, config 변환, payload 생성 경계를 구현했다.
- mock, 주입 client, 실제 RAGBuilder의 세 실행 경로를 지원한다.
- retrieval 실행 전 baseline 설정으로 surrogate data ingest/index를 생성한다.
- RAGBuilder의 process-global store를 요청마다 초기화하고 native 실행을 직렬화한다.
- retriever type과 reranker OFF/ON을 별도 scenario로 실행하고 전체 trial budget을
  나눠 사용한다.
- strict hybrid는 단일 custom retriever 내부에서 dense+BM25를 항상 조합한다.
- context_precision/context_recall evaluator 매핑과 embedding 호환 정책을 적용한다.
- 외부 best config/result를 RAGBuilderResult와 RAGBuilderTrialResult로 정규화한다.
- Python 3.11, RAGBuilder 0.1.6 CPU Docker 이미지에서 adapter unit/contract
  테스트를 통과했다.

[현재 지원하는 search space]
- retriever.top_k
- retriever.search_type
- reranker.enabled
- chunker.chunk_size
- chunker.chunk_overlap
- chunker.strategy

[의도적으로 제한한 기능]
- RAGBuilder 0.1.6 공개 결과에는 전체 Optuna trial이 없어 native 실행은 best
  config만 사용한다. 전체 trial/상위 N개 추출은 별도 확장 지점만 마련했다.
- 외부 vectorstore 재사용은 RAGBuilder 내부 ConfigStore/DocumentStore 등록 계약이
  필요해 현재 명시적으로 unsupported 처리한다.
- reranker.candidate_count, embedding model 탐색, context compression, generation
  prompt/temperature/citation 최적화는 현재 지원하지 않는다.
- 실제 corpus, eval dataset, API key를 사용하는 native integration test는 아직
  수행하지 않았다.

[strict hybrid 구현]
- retriever.search_type=["hybrid"]는 vector similarity와 BM25 후보 두 개가 아니라
  StrictHybridRetriever custom 후보 하나로 전달한다.
- custom retriever는 surrogate ingest와 동일한 chunks로 BM25를 만들고 기존
  vectorstore retriever와 EnsembleRetriever로 결합한다.
- retriever type별 scenario를 분리해 mixed search space에서도 hybrid+dense 같은
  요청하지 않은 중첩 ensemble을 만들지 않는다.
- 반환된 retriever type이 요청 search space 밖이면 outside_search_space로 거부한다.

[모듈 책임 경계]
1. optimizer가 검증한 search space로 surrogate pipeline payload를 만든다.
2. 주입 client, 실제 RAGBuilder 또는 명시적인 mock 실행 경로를 호출한다.
3. 외부 결과를 AgentDoctor 표준 RAGBuilderResult로 정규화한다.

처방 해석, 사용자 pipeline capability 검증, trial 우선순위 선택,
ConfigPatch 생성과 적용/rollback 판단은 optimizer와 internal_adapter의 책임이다.
"""
from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from threading import Lock
from typing import Any, Literal

from agents.optimize.config_mapper import canonicalize_path, get_current_value
from agents.optimize.schemas import (
    ConfigMappingResult,
    OptimizationRequest,
    RAGBuilderResult,
    RAGBuilderTrialResult,
)


# AgentDoctor 표준 경로와 RAGBuilder 실행 경로 사이의 변환 계약이다.
RAGBUILDER_KEY_MAP: dict[str, str] = {
    "retriever.top_k": "retrieval.top_k",
    "retriever.search_type": "retrieval.retriever_type",
    "reranker.enabled": "retrieval.reranker.enabled",
    "chunker.chunk_size": "data_ingest.chunk_size",
    "chunker.chunk_overlap": "data_ingest.chunk_overlap",
    "chunker.strategy": "data_ingest.chunking_strategy",
}

# RAGBuilder 버전에 따라 결과 키가 짧은 이름으로 반환되는 경우를 흡수한다.
RESULT_KEY_ALIASES: dict[str, str] = {
    "top_k": "retriever.top_k",
    "retriever_type": "retriever.search_type",
    "reranker_enabled": "reranker.enabled",
    "chunk_size": "chunker.chunk_size",
    "chunk_overlap": "chunker.chunk_overlap",
    "chunking_strategy": "chunker.strategy",
}

DEFAULT_RERANKER = "BAAI/bge-reranker-base"
DEFAULT_RETRIEVER_K = 20
STRICT_HYBRID_CLASS = (
    "agents.optimize.adapters.ragbuilder_hybrid.StrictHybridRetriever"
)

# RAGBuilder 0.1.6은 stage별 평가 방식을 고정해서 사용한다. 아래 매핑에 없는
# AgentDoctor metric을 기본 evaluator로 조용히 대체하면 다른 목적을 최적화하게
# 되므로 실행 전에 unsupported로 거부한다.
RAGBUILDER_METRIC_MAP: dict[str, dict[str, str]] = {
    "data_ingest": {
        "context_precision": "similarity",
        "context_recall": "similarity",
    },
    "retrieval": {
        "context_precision": "ragas",
        "context_recall": "ragas",
    },
}


@dataclass(frozen=True)
class _ValidationIssue:
    """어댑터 입력 경계에서 발견한 오류 또는 경고."""

    code: str
    message: str
    severity: Literal["error", "warning"] = "error"


class _AdapterExecutionError(RuntimeError):
    """optimizer가 fallback 근거로 사용할 수 있는 코드형 실행 오류."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class RAGBuilderAdapter:
    """RAGBuilder 외부 실행기와 AgentDoctor 최적화 흐름 사이의 경계."""

    # RAGBuilder 0.1.6은 process-global store를 사용하므로 native 실행을 직렬화한다.
    _native_execution_lock = Lock()

    # 전체 실행 흐름 ---------------------------------------------------------
    def run(self, request: OptimizationRequest) -> RAGBuilderResult:
        """요청 검증, payload 생성, 외부 실행, 결과 정규화를 순서대로 수행한다."""

        mapping = self.build_mapping(request)
        issues = self.validate_request(request, mapping)
        warnings = [issue.message for issue in issues if issue.severity == "warning"]
        mapping.warnings.extend(warnings)

        errors = [issue for issue in issues if issue.severity == "error"]
        if errors:
            first_error = errors[0]
            status = "skipped" if first_error.code == "empty_search_space" else "failed"
            return self._failure_result(
                request=request,
                mapping=mapping,
                status=status,
                error_code=first_error.code,
                error=first_error.message,
                issues=issues,
            )

        payload = self.build_payload(request, mapping)

        try:
            # mock은 테스트에서 명시한 경우에만 사용하며 운영 fallback으로 사용하지 않는다.
            if self._use_mock(request):
                raw_result = self._mock_result(payload)
            else:
                raw_result = self.execute(payload, request)
            return self.normalize_result(raw_result, request, payload, mapping)
        except _AdapterExecutionError as exc:
            return self._failure_result(
                request=request,
                mapping=mapping,
                payload=payload,
                error_code=exc.code,
                error=str(exc),
            )
        except Exception as exc:  # 외부 라이브러리 경계의 예외를 표준 결과로 감싼다.
            return self._failure_result(
                request=request,
                mapping=mapping,
                payload=payload,
                error_code="ragbuilder_execution_failed",
                error=str(exc),
            )

    # 입력 계약과 사전 검증 -------------------------------------------------
    def build_mapping(self, request: OptimizationRequest) -> ConfigMappingResult:
        """optimizer가 전달한 명시적 search space를 canonical 경로로 정리한다."""

        result = ConfigMappingResult(metadata={"source": "request.search_space"})
        for path, values in request.search_space.items():
            canonical_path = canonicalize_path(path)
            value_list = values if isinstance(values, list) else [values]
            result.search_space[canonical_path] = self._dedupe_values(value_list)
        return result

    def validate_request(
        self,
        request: OptimizationRequest,
        mapping: ConfigMappingResult,
    ) -> list[_ValidationIssue]:
        """RAGBuilder 형식으로 변환하고 실행할 수 있는 요청인지 검사한다."""

        issues: list[_ValidationIssue] = []
        if not mapping.search_space:
            issues.append(
                _ValidationIssue(
                    code="empty_search_space",
                    message="optimizer가 검증한 search_space가 비어 있습니다.",
                )
            )
            return issues

        for path, values in mapping.search_space.items():
            if path not in RAGBUILDER_KEY_MAP:
                issues.append(
                    _ValidationIssue(
                        code="unsupported_config_path",
                        message=f"RAGBuilder로 변환할 수 없는 config 경로입니다: {path}",
                    )
                )
            if not values:
                issues.append(
                    _ValidationIssue(
                        code="empty_candidate_values",
                        message=f"후보값이 비어 있는 config 경로입니다: {path}",
                    )
                )

        stages = self._stages_for_paths(mapping.search_space)
        if len(stages) > 1:
            issues.append(
                _ValidationIssue(
                    code="mixed_optimization_stage",
                    message=(
                        "한 요청에 여러 최적화 stage가 섞여 있습니다: "
                        + ", ".join(sorted(stages))
                    ),
                )
            )

        if request.max_trials < 1:
            issues.append(
                _ValidationIssue(
                    code="invalid_trial_budget",
                    message="max_trials는 1 이상이어야 합니다.",
                )
            )

        reranker_values = mapping.search_space.get("reranker.enabled", [])
        if len({bool(value) for value in reranker_values}) > 1 and request.max_trials < 2:
            issues.append(
                _ValidationIssue(
                    code="insufficient_trial_budget",
                    message="reranker OFF/ON 분리 실험에는 max_trials가 2 이상 필요합니다.",
                )
            )

        fixed_paths = {canonicalize_path(path) for path in request.fixed_config}
        conflicts = sorted(fixed_paths & set(mapping.search_space))
        if conflicts:
            issues.append(
                _ValidationIssue(
                    code="fixed_search_space_conflict",
                    message="고정값과 탐색값이 동시에 지정된 경로입니다: " + ", ".join(conflicts),
                )
            )

        if not self._get_input_source(request):
            issues.append(
                _ValidationIssue(
                    code="missing_input_source",
                    message="surrogate pipeline을 구성할 input_source가 없습니다.",
                )
            )

        if not self._get_eval_dataset(request):
            issues.append(
                _ValidationIssue(
                    code="missing_eval_dataset",
                    message="eval dataset이 없어 surrogate 결과의 전이 신뢰도가 낮아질 수 있습니다.",
                    severity="warning",
                )
            )

        if not request.target_metrics:
            issues.append(
                _ValidationIssue(
                    code="missing_objective_metric",
                    message="target_metrics가 없어 RAGBuilder 기본 objective를 사용합니다.",
                    severity="warning",
                )
            )
        else:
            stage = self._infer_optimized_stage(mapping.search_space)
            supported_metrics = RAGBUILDER_METRIC_MAP.get(stage, {})
            unsupported_metrics = [
                metric
                for metric in request.target_metrics
                if metric not in supported_metrics
            ]
            if unsupported_metrics:
                issues.append(
                    _ValidationIssue(
                        code="unsupported_objective_metric",
                        message=(
                            "RAGBuilder objective로 매핑할 수 없는 metric입니다: "
                            + ", ".join(unsupported_metrics)
                        ),
                    )
                )
        return issues

    # RAGBuilder payload 생성 ------------------------------------------------
    def build_payload(
        self,
        request: OptimizationRequest,
        mapping: ConfigMappingResult,
    ) -> dict[str, Any]:
        """검증된 canonical 설정을 RAGBuilder 실행 payload로 변환한다."""

        search_space = self._to_ragbuilder_config(mapping.search_space)
        fixed_config = self._to_ragbuilder_config(request.fixed_config, strict=False)
        optimized_stage = self._infer_optimized_stage(mapping.search_space)
        budget = dict(request.metadata.get("budget", {}))
        budget.update({"max_trials": request.max_trials, "n_trials": request.max_trials})

        return {
            "request_id": request.request_id,
            "optimized_stage": optimized_stage,
            "input_source": self._get_input_source(request),
            "eval_dataset": self._get_eval_dataset(request),
            "objective": {
                "metrics": list(request.target_metrics),
                "target_profile": request.target_profile,
                "direction": request.metadata.get("optimization_direction", "maximize"),
            },
            "budget": budget,
            "surrogate_pipeline": self._build_surrogate_pipeline(request, search_space),
            "search_space": search_space,
            "agentdoctor_search_space": mapping.search_space,
            "fixed_config": fixed_config,
            "agentdoctor_fixed_config": dict(request.fixed_config),
            "metadata": {
                "failure_label": request.failure_label,
                "related_failure_labels": list(request.related_failure_labels),
                "reason": request.reason,
                "mapping_warnings": list(mapping.warnings),
                "evaluator_kwargs": dict(
                    request.metadata.get("evaluator_kwargs", {})
                ),
                "metric_mapping": self._metric_mapping(
                    optimized_stage,
                    request.target_metrics,
                ),
            },
        }

    def _build_surrogate_pipeline(
        self,
        request: OptimizationRequest,
        search_space: dict[str, list[Any]],
    ) -> dict[str, Any]:
        """사용자 pipeline의 핵심 구성과 가까운 surrogate 기준 설정을 만든다."""

        metadata = request.metadata
        return {
            "input_source": self._get_input_source(request),
            "eval_dataset": self._get_eval_dataset(request),
            "vectorstore_type": metadata.get("vectorstore_type", "ragbuilder_default"),
            "chunking": {
                "chunk_size": self._baseline_value(request, "chunker.chunk_size", 512),
                "chunk_overlap": self._baseline_value(request, "chunker.chunk_overlap", 50),
                "strategy": self._baseline_value(
                    request, "chunker.strategy", "CharacterTextSplitter"
                ),
            },
            "embedding_model": self._baseline_value(
                request, "embedding.model", metadata.get("embedding_model")
            ),
            "retrieval": {
                "retriever_type": self._baseline_value(
                    request, "retriever.search_type", "dense"
                ),
                "top_k": self._baseline_value(request, "retriever.top_k", 3),
                "retriever_k": self._baseline_value(
                    request, "retriever.retriever_k", DEFAULT_RETRIEVER_K
                ),
            },
            "reranker": {
                "enabled": bool(
                    self._baseline_value(request, "reranker.enabled", False)
                ),
                "model": metadata.get("reranker_model", DEFAULT_RERANKER),
            },
            "search_space": search_space,
        }

    # 외부 실행 경계 ---------------------------------------------------------
    def execute(
        self,
        payload: dict[str, Any],
        request: OptimizationRequest,
    ) -> dict[str, Any]:
        """주입 client가 있으면 사용하고, 없으면 설치된 RAGBuilder를 실행한다."""

        client = request.metadata.get("ragbuilder_client")
        if client is not None:
            if hasattr(client, "optimize"):
                return self._ensure_dict(client.optimize(payload))
            if callable(client):
                return self._ensure_dict(client(payload))
            raise _AdapterExecutionError(
                "invalid_ragbuilder_client",
                "ragbuilder_client는 callable이거나 optimize()를 제공해야 합니다.",
            )

        try:
            from ragbuilder import RAGBuilder
        except ImportError as exc:
            raise _AdapterExecutionError(
                "ragbuilder_not_installed",
                "ragbuilder 패키지가 설치되어 있지 않습니다.",
            ) from exc

        with self._native_execution_lock:
            self._reset_ragbuilder_runtime_state()
            builder = self._create_builder(RAGBuilder, payload, request)
            return self._ensure_dict(self._run_builder(builder, payload, request))

    def _create_builder(
        self,
        ragbuilder_cls: Any,
        payload: dict[str, Any],
        request: OptimizationRequest,
    ) -> Any:
        """감지한 RAGBuilder 생성 API 하나만 사용해 인스턴스를 만든다."""

        data_config = self._build_data_ingest_config(payload, request)
        constructor_values = {
            "data_ingest_config": data_config,
            "default_llm": request.metadata.get("default_llm"),
            "default_embeddings": request.metadata.get("default_embeddings"),
            "n_trials": payload["budget"].get("n_trials"),
        }
        constructor_kwargs = {
            key: value
            for key, value in constructor_values.items()
            if value is not None and self._accepts_parameter(ragbuilder_cls, key)
        }

        # 생성자가 data ingest config를 공식적으로 받는 버전이면 오류를 숨기지 않는다.
        if data_config is not None and "data_ingest_config" in constructor_kwargs:
            return ragbuilder_cls(**constructor_kwargs)

        # data ingest 탐색은 세부 config를 전달할 수 없으면 요청 범위를 보장할 수 없다.
        if payload["optimized_stage"] == "data_ingest":
            raise _AdapterExecutionError(
                "ragbuilder_api_incompatible",
                "RAGBuilder에 제한된 data ingest search space를 전달할 수 없습니다.",
            )

        factory = getattr(ragbuilder_cls, "from_source_with_defaults", None)
        if factory is None:
            raise _AdapterExecutionError(
                "ragbuilder_api_incompatible",
                "지원할 수 있는 RAGBuilder 생성 API를 찾지 못했습니다.",
            )

        factory_kwargs = self._filter_kwargs(
            factory,
            {
                "input_source": payload["input_source"],
                "test_dataset": payload.get("eval_dataset"),
                "n_trials": payload["budget"].get("n_trials"),
            },
        )
        return factory(**factory_kwargs)

    def _run_builder(
        self,
        builder: Any,
        payload: dict[str, Any],
        request: OptimizationRequest,
    ) -> Any:
        """검증된 단일 optimized stage에 대응하는 RAGBuilder API를 실행한다."""

        stage = payload["optimized_stage"]
        if stage == "data_ingest":
            method = getattr(builder, "optimize_data_ingest", None)
            if method is None:
                raise _AdapterExecutionError(
                    "ragbuilder_api_incompatible",
                    "RAGBuilder가 data_ingest 최적화 API를 제공하지 않습니다.",
                )
            return method()

        if stage != "retrieval":
            raise _AdapterExecutionError(
                "ragbuilder_api_incompatible",
                f"RAGBuilder가 {stage} 최적화 API를 제공하지 않습니다.",
            )

        method = getattr(builder, "optimize_retrieval", None)
        if method is None or not self._accepts_positional_argument(method):
            raise _AdapterExecutionError(
                "ragbuilder_api_incompatible",
                "RAGBuilder retrieval API가 config 인자를 받지 않습니다.",
            )

        external_vectorstore = request.metadata.get("surrogate_vectorstore")
        if external_vectorstore is not None:
            # 향후 외부 index 재사용 경로다. RAGBuilder 0.1.6은 vectorstore 외에도
            # 전역 ConfigStore의 best ingest config를 요구하므로, 현재는 호출자가
            # 두 객체를 함께 준비한 경우에만 이 경로를 허용한다.
            external_ingest_config = request.metadata.get(
                "surrogate_data_ingest_config"
            )
            if external_ingest_config is None:
                raise _AdapterExecutionError(
                    "missing_surrogate_ingest_config",
                    "외부 vectorstore 재사용에는 surrogate_data_ingest_config도 필요합니다.",
                )
            self._register_external_surrogate(
                builder,
                external_vectorstore,
                external_ingest_config,
            )
        else:
            # MVP에서는 adapter가 baseline ingest를 한 번 실행해 동일 corpus 기반의
            # surrogate index를 만든다. 추후에는 위 외부 vectorstore 경로를 통해
            # 반복 요청 간 index를 재사용할 수 있다.
            ingest_method = getattr(builder, "optimize_data_ingest", None)
            if ingest_method is None:
                raise _AdapterExecutionError(
                    "ragbuilder_api_incompatible",
                    "retrieval 전에 surrogate index를 만들 API가 없습니다.",
                )
            ingest_method()

        scenarios = self._retrieval_scenarios(payload)
        scenario_budgets = self._split_trial_budget(
            int(payload["budget"].get("n_trials") or 1),
            len(scenarios),
        )
        scenario_results: list[dict[str, Any]] = []
        for (retriever_type, reranker_enabled), trial_budget in zip(
            scenarios,
            scenario_budgets,
        ):
            config = self._build_retrieval_config(
                payload,
                reranker_enabled,
                trial_budget,
                retriever_type,
            )
            if config is None:
                raise _AdapterExecutionError(
                    "ragbuilder_api_incompatible",
                    "RAGBuilder에 제한된 retrieval search space를 전달할 수 없습니다.",
                )
            reranker_name = "reranker-on" if reranker_enabled else "reranker-off"
            scenario_name = f"{retriever_type}-{reranker_name}"
            config_overrides = {
                "retriever_type": retriever_type,
                "reranker_enabled": reranker_enabled,
            }
            try:
                result = method(config)
                scenario_results.append(
                    self._compact_module_result(
                        result,
                        trial_id=f"ragbuilder-{scenario_name}",
                        config_overrides=config_overrides,
                    )
                )
            except Exception as exc:
                # ON/OFF 중 한 경로만 실패해도 다른 경로의 유효한 후보는 보존한다.
                scenario_results.append(
                    {
                        "trial_id": f"ragbuilder-{scenario_name}",
                        "config": config_overrides,
                        "score": None,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

        completed = [
            trial
            for trial in scenario_results
            if trial.get("status") == "completed" and trial.get("score") is not None
        ]
        if not completed:
            errors = [trial.get("error") for trial in scenario_results if trial.get("error")]
            raise _AdapterExecutionError(
                "ragbuilder_execution_failed",
                "; ".join(errors) or "모든 retrieval scenario가 실패했습니다.",
            )

        selector = (
            min
            if str(payload["objective"]["direction"]).lower() == "minimize"
            else max
        )
        best = selector(completed, key=lambda trial: trial["score"])
        return {
            "status": "completed",
            "best_config": best["config"],
            "best_score": best["score"],
            "trial_results": scenario_results,
            "metadata": {
                "trial_collection": "best_only_per_retriever_reranker_scenario",
                "native_trial_results_available": False,
            },
        }

    # RAGBuilder config 객체 생성 -------------------------------------------
    def _build_data_ingest_config(
        self,
        payload: dict[str, Any],
        request: OptimizationRequest,
    ) -> Any | None:
        """chunking 계열 search space를 RAGBuilder data ingest config로 만든다."""

        config_cls = self._load_ragbuilder_config_class("DataIngestOptionsConfig")
        if config_cls is None:
            return None

        search_space = payload["search_space"]
        chunking = payload["surrogate_pipeline"]["chunking"]
        stage = payload["optimized_stage"]
        chunk_size_values = (
            search_space.get("data_ingest.chunk_size")
            if stage == "data_ingest"
            else None
        )
        chunk_overlap_values = (
            search_space.get("data_ingest.chunk_overlap")
            if stage == "data_ingest"
            else None
        )
        strategy_values = (
            search_space.get("data_ingest.chunking_strategy")
            if stage == "data_ingest"
            else None
        )
        kwargs = {
            "input_source": payload.get("input_source"),
            "chunk_size": self._chunk_size_config(
                chunk_size_values,
                chunking["chunk_size"],
            ),
            "chunk_overlap": self._integer_candidates(
                chunk_overlap_values,
                chunking["chunk_overlap"],
            ),
            "chunking_strategies": self._chunking_strategies(
                strategy_values or [chunking["strategy"]]
            ),
            "embedding_models": self._embedding_model_options(request, payload),
            "optimization": {
                # retrieval 선행 ingest는 고정 surrogate index 하나만 만들면 된다.
                "n_trials": payload["budget"].get("n_trials")
                if stage == "data_ingest"
                else 1,
                "optimization_direction": payload["objective"]["direction"],
            },
            "evaluation_config": self._evaluation_config(payload, "data_ingest"),
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        return config_cls(**self._filter_kwargs(config_cls, kwargs))

    def _build_retrieval_config(
        self,
        payload: dict[str, Any],
        reranker_enabled: bool,
        trial_budget: int | None = None,
        retriever_type: str | None = None,
    ) -> Any | None:
        """retrieval 계열 search space를 RAGBuilder retrieval config로 만든다."""

        config_cls = self._load_ragbuilder_config_class("RetrievalOptionsConfig")
        if config_cls is None:
            return None

        search_space = payload["search_space"]
        surrogate = payload["surrogate_pipeline"]
        retriever_types = [
            retriever_type
            or surrogate["retrieval"]["retriever_type"]
        ]
        # 빈 목록은 RAGBuilder apply_defaults가 기본 BGE reranker를 추가하지 않게
        # 하는 명시적인 OFF 표현이다. None을 사용하면 OFF 실험이 ON으로 바뀐다.
        rerankers = (
            [{"type": surrogate["reranker"]["model"]}]
            if reranker_enabled
            else []
        )

        kwargs = {
            "retrievers": self._retriever_options(
                retriever_types,
                surrogate["retrieval"]["retriever_k"],
            ),
            "rerankers": rerankers,
            "top_k": search_space.get("retrieval.top_k")
            or [surrogate["retrieval"]["top_k"]],
            "optimization": {
                "n_trials": trial_budget or payload["budget"].get("n_trials"),
                "optimization_direction": payload["objective"]["direction"],
            },
            "evaluation_config": self._evaluation_config(payload, "retrieval"),
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        return config_cls(**self._filter_kwargs(config_cls, kwargs))

    def _load_ragbuilder_config_class(self, name: str) -> Any | None:
        """RAGBuilder config class를 optional dependency로 안전하게 불러온다."""

        try:
            module = __import__("ragbuilder.config", fromlist=[name])
        except ImportError:
            return None
        return getattr(module, name, None)

    def _evaluation_config(
        self,
        payload: dict[str, Any],
        stage: str,
    ) -> dict[str, Any]:
        """AgentDoctor objective를 RAGBuilder 0.1.6 evaluator 설정으로 변환한다."""

        mapped = self._metric_mapping(stage, payload["objective"]["metrics"])
        evaluator_type = mapped.get("evaluator") or (
            "ragas" if stage == "retrieval" else "similarity"
        )
        evaluator_kwargs = dict(payload.get("metadata", {}).get("evaluator_kwargs", {}))
        if stage == "data_ingest":
            evaluator_kwargs.setdefault(
                "top_k",
                payload["surrogate_pipeline"]["retrieval"]["top_k"],
            )

        # RAGBuilder 0.1.6 retrieval evaluator는 context precision/recall의 F1을
        # 고정 objective로 사용한다. 단일 metric을 요청해도 완전히 같은 목적함수가
        # 되지 않아 surrogate 결과의 전이성이 낮아질 수 있으며 최종 Eval 검증이 필수다.
        return {
            "type": evaluator_type,
            "test_dataset": payload.get("eval_dataset"),
            "evaluator_kwargs": evaluator_kwargs,
        }

    def _metric_mapping(
        self,
        stage: str,
        metrics: list[str],
    ) -> dict[str, Any]:
        """지원 metric 목록과 실제 RAGBuilder evaluator를 추적 가능한 형태로 반환한다."""

        stage_mapping = RAGBUILDER_METRIC_MAP.get(stage, {})
        mapped_metrics = {
            metric: stage_mapping[metric]
            for metric in metrics
            if metric in stage_mapping
        }
        evaluators = self._dedupe_values(list(mapped_metrics.values()))
        return {
            "requested_metrics": list(metrics),
            "mapped_metrics": mapped_metrics,
            "evaluator": evaluators[0] if len(evaluators) == 1 else None,
            "approximate": stage == "data_ingest" and bool(metrics),
        }

    def _embedding_model_options(
        self,
        request: OptimizationRequest,
        payload: dict[str, Any],
    ) -> list[Any] | None:
        """동일 모델 우선, 명시된 호환 모델만 허용해 embedding config를 만든다."""

        metadata = request.metadata
        configured = metadata.get("embedding_models")
        if configured:
            configured_values = configured if isinstance(configured, list) else [configured]
            resolved: list[Any] = []
            for model in configured_values:
                resolved.extend(self._embedding_configs_for_model(model, request))
            return resolved

        # 초기화된 embedding 객체가 있으면 RAGBuilder ConfigStore가 같은 객체를
        # 사용하므로 별도 EmbeddingConfig 후보를 만들지 않는다.
        if metadata.get("default_embeddings") is not None:
            return None

        model = payload["surrogate_pipeline"].get("embedding_model")
        return self._embedding_configs_for_model(model, request)

    def _embedding_configs_for_model(
        self,
        model: Any,
        request: OptimizationRequest,
    ) -> list[Any]:
        """embedding model 하나를 동일 모델 또는 명시된 호환 config로 변환한다."""

        metadata = request.metadata
        if isinstance(model, dict):
            return [model]
        if not isinstance(model, str) or not model:
            raise _AdapterExecutionError(
                "missing_embedding_model",
                "surrogate pipeline에 사용할 embedding model 정보가 없습니다.",
            )

        explicit_mapping = metadata.get("embedding_model_compatibility", {})
        if model in explicit_mapping:
            mapped = explicit_mapping[model]
            # 호환 모델 fallback은 실제 사용자 embedding과 벡터 공간이 달라져
            # 최적 config의 전이성을 낮출 수 있다. 따라서 호출자가 명시한 매핑만 허용한다.
            return mapped if isinstance(mapped, list) else [mapped]

        provider = metadata.get("embedding_provider")
        if provider:
            model_key = "model_name" if provider == "huggingface" else "model"
            return [{"type": provider, "model_kwargs": {model_key: model}}]

        # AgentDoctor 기본 config의 provider://model 표기는 provider와 동일 모델을
        # 모두 명시하므로 추측 없이 RAGBuilder EmbeddingConfig로 변환할 수 있다.
        if "://" in model:
            uri_provider, model_name = model.split("://", 1)
            provider_aliases = {
                "sentence-transformers": "huggingface",
                "hf": "huggingface",
            }
            uri_provider = provider_aliases.get(uri_provider, uri_provider)
            supported_providers = {
                "openai",
                "azure_openai",
                "huggingface",
                "ollama",
                "cohere",
                "vertexai",
                "bedrock",
                "jina",
            }
            if uri_provider in supported_providers and model_name:
                model_key = (
                    "model_name" if uri_provider == "huggingface" else "model"
                )
                return [
                    {
                        "type": uri_provider,
                        "model_kwargs": {model_key: model_name},
                    }
                ]

        # HuggingFace repository 형태는 provider를 안전하게 식별할 수 있어 동일
        # 모델 config로 자동 변환한다. API provider 모델은 명시 정보 없이 추측하지 않는다.
        huggingface_prefixes = (
            "sentence-transformers/",
            "BAAI/",
            "intfloat/",
        )
        if model.startswith(huggingface_prefixes) or model.startswith("all-"):
            return [
                {
                    "type": "huggingface",
                    "model_kwargs": {"model_name": model},
                }
            ]

        raise _AdapterExecutionError(
            "unsupported_embedding_model",
            "embedding_provider 또는 embedding_model_compatibility 매핑이 필요합니다: "
            + model,
        )

    def _chunk_size_config(
        self,
        values: list[Any] | None,
        default: Any,
    ) -> dict[str, int]:
        """chunk size 후보를 RAGBuilder ChunkSizeConfig 범위로 변환한다."""

        candidates = self._integer_candidates(values, default)
        if any(candidate < 1 for candidate in candidates):
            raise _AdapterExecutionError(
                "invalid_numeric_candidate",
                "chunk_size는 1 이상이어야 합니다.",
            )
        if len(candidates) == 1:
            return {
                "min": candidates[0],
                "max": candidates[0],
                "stepsize": 1,
            }
        diffs = [
            candidates[index + 1] - candidates[index]
            for index in range(len(candidates) - 1)
        ]
        if len(set(diffs)) != 1:
            raise _AdapterExecutionError(
                "unsupported_candidate_values",
                "RAGBuilder 0.1.6 chunk_size는 등간격 후보만 지원합니다.",
            )
        return {
            "min": candidates[0],
            "max": candidates[-1],
            "stepsize": diffs[0],
        }

    def _integer_candidates(
        self,
        values: list[Any] | None,
        default: Any,
    ) -> list[int]:
        """RAGBuilder 숫자 목록 필드에 전달할 양의 정수 후보를 검증한다."""

        raw_values = values or [default]
        candidates: list[int] = []
        for value in raw_values:
            if isinstance(value, bool):
                raise _AdapterExecutionError(
                    "invalid_numeric_candidate",
                    f"bool은 숫자 후보로 사용할 수 없습니다: {value}",
                )
            if isinstance(value, float) and not value.is_integer():
                raise _AdapterExecutionError(
                    "invalid_numeric_candidate",
                    f"소수 후보를 정수로 절삭할 수 없습니다: {value}",
                )
            try:
                candidate = int(value)
            except (TypeError, ValueError) as exc:
                raise _AdapterExecutionError(
                    "invalid_numeric_candidate",
                    f"정수로 변환할 수 없는 후보입니다: {value}",
                ) from exc
            if candidate < 0:
                raise _AdapterExecutionError(
                    "invalid_numeric_candidate",
                    f"음수 후보는 사용할 수 없습니다: {value}",
                )
            candidates.append(candidate)
        return sorted(set(candidates))

    def _reranker_scenarios(self, payload: dict[str, Any]) -> list[bool]:
        """reranker OFF와 ON을 서로 독립적인 RAGBuilder 실행으로 분리한다."""

        values = payload["search_space"].get("retrieval.reranker.enabled")
        if values is None:
            values = [payload["surrogate_pipeline"]["reranker"]["enabled"]]
        return self._dedupe_values([bool(value) for value in values])

    def _retriever_scenarios(self, payload: dict[str, Any]) -> list[str]:
        """retriever type을 고정된 독립 RAGBuilder 실행 단위로 분리한다."""

        values = payload["search_space"].get("retrieval.retriever_type")
        if values is None:
            values = [payload["surrogate_pipeline"]["retrieval"]["retriever_type"]]

        aliases = {
            "vector": "dense",
            "vector_similarity": "dense",
        }
        normalized = [aliases.get(str(value), str(value)) for value in values]
        return self._dedupe_values(normalized)

    def _retrieval_scenarios(self, payload: dict[str, Any]) -> list[tuple[str, bool]]:
        """retriever type과 reranker 여부의 직교 조합을 실행 scenario로 만든다."""

        return [
            (retriever_type, reranker_enabled)
            for retriever_type in self._retriever_scenarios(payload)
            for reranker_enabled in self._reranker_scenarios(payload)
        ]

    def _split_trial_budget(self, total: int, scenario_count: int) -> list[int]:
        """전체 trial budget을 scenario별로 합계가 유지되도록 분배한다."""

        if scenario_count < 1 or total < scenario_count:
            raise _AdapterExecutionError(
                "insufficient_trial_budget",
                "scenario 수보다 trial budget이 작습니다.",
            )
        base, remainder = divmod(total, scenario_count)
        return [base + (1 if index < remainder else 0) for index in range(scenario_count)]

    def _compact_module_result(
        self,
        result: Any,
        *,
        trial_id: str,
        config_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """native 결과에서 pipeline/vectorstore를 제외하고 best 정보만 추출한다."""

        best_config = getattr(result, "best_config", None)
        best_score = self._coerce_float(getattr(result, "best_score", None))
        if isinstance(result, dict):
            best_config = result.get("best_config", best_config)
            best_score = self._coerce_float(result.get("best_score", best_score))
        config = self._model_dump(best_config)
        if not isinstance(config, dict) or best_score is None:
            raise _AdapterExecutionError(
                "invalid_result_shape",
                "RAGBuilder module 결과에 best_config 또는 best_score가 없습니다.",
            )
        config.update(config_overrides or {})
        return {
            "trial_id": trial_id,
            "config": config,
            "score": best_score,
            "metrics": {
                "avg_latency": getattr(result, "avg_latency", None),
                "error_rate": getattr(result, "error_rate", None),
            },
            "status": "completed",
        }

    def _extract_native_trials(self, result: Any) -> list[dict[str, Any]]:
        """향후 Optuna/SQLite trial 추출을 연결할 확장 지점이다.

        RAGBuilder 0.1.6의 공개 결과에는 전체 study가 포함되지 않는다. 내부 SQLite나
        private optimizer에 결합하면 버전 변경에 취약하므로 MVP에서는 best config만
        사용하고 빈 목록을 반환한다. 공개 callback 계약이 확정되면 여기서 상위 N개를
        표준 trial 형식으로 변환한다.
        """

        return []

    def _register_external_surrogate(
        self,
        builder: Any,
        vectorstore: Any,
        ingest_config: Any,
    ) -> None:
        """향후 외부 surrogate index 재사용을 위한 명시적 확장 지점이다."""

        # RAGBuilder 0.1.6 retrieval optimizer는 builder.optimized_store 외에도
        # process-global ConfigStore/DocumentStore 상태를 요구한다. 불완전하게 주입해
        # 잘못된 index를 쓰는 것보다 공식 등록 계약을 구현할 때까지 명시적으로 막는다.
        raise _AdapterExecutionError(
            "external_surrogate_not_supported",
            "RAGBuilder 0.1.6 외부 vectorstore 등록 계약은 아직 지원하지 않습니다.",
        )

    def _reset_ragbuilder_runtime_state(self) -> None:
        """요청 간 corpus/index가 섞이지 않도록 RAGBuilder 전역 최적화 상태를 비운다."""

        try:
            from ragbuilder.core.config_store import ConfigStore
            from ragbuilder.core.document_store import DocumentStore
        except ImportError as exc:
            raise _AdapterExecutionError(
                "ragbuilder_api_incompatible",
                "RAGBuilder runtime store를 불러올 수 없습니다.",
            ) from exc

        DocumentStore.clear()
        # ConfigStore에는 공개 clear API가 없어 0.1.6의 내부 저장 필드만 초기화한다.
        # 버전이 바뀌면 이 계약을 contract test로 다시 확인해야 한다.
        ConfigStore._configs.clear()
        ConfigStore._metadata.clear()
        ConfigStore._best_data_ingest_pipeline = None
        ConfigStore._best_retriever_pipeline = None
        ConfigStore._best_generator_pipeline = None
        ConfigStore._default_llm = None
        ConfigStore._default_embeddings = None
        ConfigStore._default_n_trials = None

    # 외부 결과 정규화 -------------------------------------------------------
    def normalize_result(
        self,
        raw_result: dict[str, Any],
        request: OptimizationRequest,
        payload: dict[str, Any],
        mapping: ConfigMappingResult,
    ) -> RAGBuilderResult:
        """외부 결과를 후보 선택 전 단계인 RAGBuilderResult로 정규화한다."""

        raw_result = self._ensure_dict(raw_result)
        if str(raw_result.get("status", "")).lower() in {"failed", "error"}:
            return self._failure_result(
                request=request,
                mapping=mapping,
                payload=payload,
                raw_result=raw_result,
                error_code=str(raw_result.get("error_code") or "ragbuilder_failed"),
                error=str(raw_result.get("error") or "RAGBuilder 실행이 실패했습니다."),
            )

        trials = self._normalize_trials(raw_result, mapping.search_space)
        best_config = self._extract_best_config(
            raw_result,
            trials,
            mapping.search_space,
            payload["objective"]["direction"],
        )
        best_score = self._extract_best_score(
            raw_result,
            trials,
            mapping.search_space,
            payload["objective"]["direction"],
        )
        if not trials and best_config is None:
            return self._failure_result(
                request=request,
                mapping=mapping,
                payload=payload,
                raw_result=raw_result,
                error_code="invalid_result_shape",
                error="RAGBuilder 결과에 best_config 또는 trial_results가 없습니다.",
            )

        return RAGBuilderResult(
            request_id=request.request_id,
            best_config=best_config,
            best_score=best_score,
            trial_results=trials,
            optimized_stage=payload["optimized_stage"],
            search_space=mapping.search_space,
            payload=payload,
            raw_result=raw_result,
            status="completed",
            warnings=list(mapping.warnings),
            metadata={
                "is_mock": bool(raw_result.get("mock")),
                "execution_mode": self._execution_mode(request),
                "mapping": self._to_plain_dict(mapping),
                "native_metadata": self._ensure_mapping(raw_result.get("metadata")),
            },
        )

    def _normalize_trials(
        self,
        raw_result: dict[str, Any],
        expected_space: dict[str, list[Any]],
    ) -> list[RAGBuilderTrialResult]:
        """RAGBuilder의 다양한 trial 표현을 순서를 유지한 표준 목록으로 바꾼다."""

        raw_trials = self._first_present(
            raw_result, ("trial_results", "trials", "results"), []
        )
        if hasattr(raw_trials, "to_dict"):
            raw_trials = raw_trials.to_dict("records")
        if not isinstance(raw_trials, (list, tuple)):
            raw_trials = [raw_trials]

        trials = [
            self._normalize_trial(raw_trial, index, expected_space)
            for index, raw_trial in enumerate(raw_trials)
        ]
        if trials:
            return trials

        # trial 목록 없이 best_config만 반환하는 API도 단일 trial로 보존한다.
        raw_best = self._first_present(raw_result, ("best_config", "config"), None)
        if raw_best is None:
            return []
        return [
            self._normalize_trial(
                {
                    "trial_id": "ragbuilder-best",
                    "config": raw_best,
                    "score": self._first_present(
                        raw_result, ("best_score", "score"), None
                    ),
                    "metrics": raw_result.get("metrics") or {},
                },
                0,
                expected_space,
            )
        ]

    def _normalize_trial(
        self,
        raw_trial: Any,
        index: int,
        expected_space: dict[str, list[Any]],
    ) -> RAGBuilderTrialResult:
        """단일 raw trial을 canonical config와 표준 상태로 변환한다."""

        trial = self._to_plain_dict(raw_trial)
        if not isinstance(trial, dict):
            trial = {"value": trial}

        raw_config = self._first_present(
            trial, ("config", "params", "best_config"), None
        )
        config, unsupported_reasons = self._to_agentdoctor_config(
            raw_config, expected_space
        )
        unsupported_reasons.extend(
            self._outside_search_space_reasons(config, expected_space)
        )
        status = self._normalize_trial_status(trial.get("status"))
        if raw_config is None or not config:
            status = "rejected"
            unsupported_reasons.append("invalid_or_empty_config")
        elif unsupported_reasons and status == "completed":
            status = "unsupported"

        return RAGBuilderTrialResult(
            trial_id=str(
                self._first_present(
                    trial,
                    ("trial_id", "id", "number"),
                    f"ragbuilder-trial-{index + 1}",
                )
            ),
            config=config,
            score=self._coerce_float(
                self._first_present(
                    trial, ("score", "value", "best_score", "metric"), None
                )
            ),
            metrics=self._ensure_mapping(trial.get("metrics")),
            status=status,
            unsupported_reasons=self._dedupe_values(unsupported_reasons),
            raw_trial=trial,
        )

    def _extract_best_config(
        self,
        raw_result: dict[str, Any],
        trials: list[RAGBuilderTrialResult],
        expected_space: dict[str, list[Any]],
        direction: str,
    ) -> dict[str, Any] | None:
        """RAGBuilder가 명시한 best config를 보존하고 없을 때만 trial에서 보완한다."""

        raw_best = self._first_present(raw_result, ("best_config", "config"), None)
        if raw_best is not None:
            best_config, reasons = self._to_agentdoctor_config(raw_best, expected_space)
            reasons.extend(
                self._outside_search_space_reasons(best_config, expected_space)
            )
            if best_config and not reasons:
                return best_config

        completed = [trial for trial in trials if trial.status == "completed"]
        scored = [trial for trial in completed if trial.score is not None]
        if scored:
            selector = min if str(direction).lower() == "minimize" else max
            return selector(scored, key=lambda trial: trial.score).config
        return completed[0].config if completed else None

    def _extract_best_score(
        self,
        raw_result: dict[str, Any],
        trials: list[RAGBuilderTrialResult],
        expected_space: dict[str, list[Any]],
        direction: str,
    ) -> float | None:
        """요청 범위 안의 외부 best score를 보존하고 없으면 trial에서 보완한다."""

        raw_score = self._first_present(raw_result, ("best_score", "score"), None)
        score = self._coerce_float(raw_score)
        raw_best = self._first_present(raw_result, ("best_config", "config"), None)
        if score is not None and raw_best is not None:
            best_config, reasons = self._to_agentdoctor_config(raw_best, expected_space)
            reasons.extend(
                self._outside_search_space_reasons(best_config, expected_space)
            )
            if best_config and not reasons:
                return score
        scores = [
            trial.score
            for trial in trials
            if trial.status == "completed" and trial.score is not None
        ]
        if not scores:
            return None
        return min(scores) if str(direction).lower() == "minimize" else max(scores)

    def _to_agentdoctor_config(
        self,
        config: Any,
        expected_space: dict[str, list[Any]],
    ) -> tuple[dict[str, Any], list[str]]:
        """외부 config에서 이번 요청의 최적화 축만 canonical 경로로 복원한다."""

        plain = self._to_plain_dict(config)
        if not isinstance(plain, dict):
            return {}, ["config_is_not_mapping"]

        plain = self._normalize_native_config_shape(plain)
        flattened = self._flatten_dict(plain)
        converted: dict[str, Any] = {}
        missing: list[str] = []
        for canonical_path in expected_space:
            external_path = RAGBUILDER_KEY_MAP[canonical_path]
            aliases = (
                external_path,
                canonical_path,
                *(
                    alias
                    for alias, target in RESULT_KEY_ALIASES.items()
                    if target == canonical_path
                ),
            )
            found = False
            for alias in aliases:
                if alias in flattened:
                    converted[canonical_path] = flattened[alias]
                    found = True
                    break
            if not found:
                missing.append(f"missing_optimized_path:{canonical_path}")
        return converted, missing

    def _outside_search_space_reasons(
        self,
        config: dict[str, Any],
        expected_space: dict[str, list[Any]],
    ) -> list[str]:
        """반환된 retriever type이 요청한 후보 범위를 벗어나면 거부 사유를 만든다."""

        path = "retriever.search_type"
        if path not in config or path not in expected_space:
            return []
        aliases = {
            "vector": "dense",
            "vector_similarity": "dense",
        }
        actual = aliases.get(str(config[path]), str(config[path]))
        expected = {
            aliases.get(str(value), str(value))
            for value in expected_space[path]
        }
        if actual in expected:
            return []
        return [f"outside_search_space:{path}:{actual}"]

    def _normalize_native_config_shape(
        self,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """RAGBuilder 0.1.6 module config를 canonical alias가 있는 평면 구조로 보강한다."""

        normalized = dict(config)
        retrievers = config.get("retrievers")
        if isinstance(retrievers, list):
            has_strict_hybrid = any(
                self._is_strict_hybrid_component(item) for item in retrievers
            )
            retriever_types = [
                self._component_type(item)
                for item in retrievers
                if self._component_type(item)
            ]
            type_set = set(retriever_types)
            if has_strict_hybrid:
                normalized["retriever_type"] = "hybrid"
            elif "vector_similarity" in type_set and "bm25" in type_set:
                normalized["retriever_type"] = "hybrid"
            elif "bm25" in type_set:
                normalized["retriever_type"] = "bm25"
            elif type_set & {"vector_similarity", "vector_mmr"}:
                normalized["retriever_type"] = "dense"
            elif retriever_types:
                normalized["retriever_type"] = retriever_types[0]

        if "rerankers" in config:
            normalized["reranker_enabled"] = bool(config.get("rerankers"))

        chunking_strategy = config.get("chunking_strategy")
        if isinstance(chunking_strategy, dict):
            normalized["chunking_strategy"] = chunking_strategy.get("type")

        embedding_model = config.get("embedding_model")
        if isinstance(embedding_model, dict):
            model_kwargs = embedding_model.get("model_kwargs") or {}
            normalized["embedding_model"] = (
                model_kwargs.get("model")
                or model_kwargs.get("model_name")
                or embedding_model.get("type")
            )
        return normalized

    def _is_strict_hybrid_component(self, component: Any) -> bool:
        """RAGBuilder 결과의 custom component가 strict hybrid인지 판별한다."""

        plain = self._to_plain_dict(component)
        if not isinstance(plain, dict):
            return False
        return (
            self._component_type(plain) == "custom"
            and plain.get("custom_class") == STRICT_HYBRID_CLASS
        )

    def _component_type(self, component: Any) -> str | None:
        """RAGBuilder component 객체나 dict에서 type 문자열을 추출한다."""

        plain = self._to_plain_dict(component)
        if isinstance(plain, dict):
            value = plain.get("type")
        else:
            value = getattr(component, "type", None)
        if hasattr(value, "value"):
            value = value.value
        return str(value) if value is not None else None

    # 실패 결과와 상태 코드 --------------------------------------------------
    def _failure_result(
        self,
        request: OptimizationRequest,
        mapping: ConfigMappingResult,
        *,
        error_code: str,
        error: str,
        status: Literal["failed", "skipped"] = "failed",
        payload: dict[str, Any] | None = None,
        raw_result: dict[str, Any] | None = None,
        issues: list[_ValidationIssue] | None = None,
    ) -> RAGBuilderResult:
        """모든 경계 오류를 optimizer가 해석 가능한 동일한 결과로 만든다."""

        stage = self._infer_optimized_stage(mapping.search_space)
        return RAGBuilderResult(
            request_id=request.request_id,
            best_config=None,
            best_score=None,
            optimized_stage=stage,
            search_space=mapping.search_space,
            payload=payload or {},
            raw_result=raw_result or {},
            status=status,
            error=error,
            warnings=list(mapping.warnings),
            metadata={
                "error_code": error_code,
                "execution_mode": self._execution_mode(request),
                "issues": [self._to_plain_dict(issue) for issue in issues or []],
                "mapping": self._to_plain_dict(mapping),
            },
        )

    # 경로와 stage 변환 유틸 -------------------------------------------------
    def _to_ragbuilder_config(
        self,
        config: dict[str, Any],
        *,
        strict: bool = True,
    ) -> dict[str, Any]:
        """canonical config 묶음을 RAGBuilder dotted path 묶음으로 변환한다."""

        converted: dict[str, Any] = {}
        for path, value in config.items():
            canonical_path = canonicalize_path(path)
            external_path = RAGBUILDER_KEY_MAP.get(canonical_path)
            if external_path is None:
                if strict:
                    raise _AdapterExecutionError(
                        "unsupported_config_path",
                        f"RAGBuilder로 변환할 수 없는 config 경로입니다: {path}",
                    )
                continue
            converted[external_path] = list(value) if isinstance(value, list) else value
        return converted

    def _stages_for_paths(self, config: dict[str, Any]) -> set[str]:
        """canonical config 경로들이 속한 최적화 stage 집합을 반환한다."""

        stages: set[str] = set()
        for path in config:
            if path.startswith("chunker.") or path.startswith("embedding."):
                stages.add("data_ingest")
            elif path.startswith(("retriever.", "reranker.", "context.")):
                stages.add("retrieval")
            else:
                stages.add("unknown")
        return stages

    def _infer_optimized_stage(self, config: dict[str, Any]) -> str:
        """단일 stage 요청의 stage를 반환하고 빈 요청은 retrieval로 표시한다."""

        stages = self._stages_for_paths(config)
        return next(iter(stages)) if len(stages) == 1 else "retrieval"

    # surrogate 기준값 조회 --------------------------------------------------
    def _baseline_value(
        self,
        request: OptimizationRequest,
        path: str,
        default: Any,
    ) -> Any:
        """fixed config를 우선하고 baseline config를 다음 순서로 조회한다."""

        sentinel = object()
        fixed_value = get_current_value(request.fixed_config, path, sentinel)
        if fixed_value is not sentinel:
            return fixed_value
        return get_current_value(request.baseline_config, path, default)

    def _get_input_source(self, request: OptimizationRequest) -> Any:
        """metadata, fixed config, baseline config 순서로 corpus 입력을 찾는다."""

        return (
            request.metadata.get("input_source")
            or request.metadata.get("source_url")
            or request.fixed_config.get("input_source")
            or request.baseline_config.get("input_source")
        )

    def _get_eval_dataset(self, request: OptimizationRequest) -> Any:
        """RAGBuilder surrogate 평가에 사용할 dataset 입력을 찾는다."""

        return request.metadata.get("eval_dataset") or request.metadata.get(
            "test_dataset"
        )

    # mock과 RAGBuilder 옵션 변환 -------------------------------------------
    def _use_mock(self, request: OptimizationRequest) -> bool:
        """개발과 단위 테스트에서 명시된 mock 실행 여부만 확인한다."""

        return bool(request.metadata.get("use_mock") or request.metadata.get("mock"))

    def _execution_mode(self, request: OptimizationRequest) -> str:
        """결과 추적용 실행 경로 이름을 반환한다."""

        if self._use_mock(request):
            return "mock"
        if request.metadata.get("ragbuilder_client") is not None:
            return "injected_client"
        return "ragbuilder"

    def _mock_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        """변환과 정규화 테스트를 위한 결정적인 가짜 trial 결과를 만든다."""

        trials: list[dict[str, Any]] = []
        max_trials = int(payload["budget"].get("n_trials") or 1)
        trial_configs = self._first_trial_configs(
            payload["search_space"],
            max_trials,
        )
        for index, config in enumerate(trial_configs):
            score = round(1.0 - index * 0.05, 4)
            trials.append(
                {
                    "trial_id": f"mock-ragbuilder-{index + 1}",
                    "config": config,
                    "score": score,
                    "metrics": {"mock_objective": score},
                    "status": "completed",
                }
            )
        best = trials[0] if trials else None
        return {
            "mock": True,
            "status": "completed",
            "best_config": best["config"] if best else {},
            "best_score": best["score"] if best else None,
            "trial_results": trials,
            "payload": payload,
        }

    def _first_trial_configs(
        self,
        search_space: dict[str, list[Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """mock 후보 조합을 만들되 각 단계에서 trial budget만큼만 유지한다."""

        configs: list[dict[str, Any]] = [{}]
        for path, values in search_space.items():
            configs = [
                {**config, path: value}
                for config in configs
                for value in values
            ][:limit]
        return configs[:limit]

    def _chunking_strategies(self, values: list[Any]) -> list[dict[str, Any]]:
        """canonical chunking strategy를 RAGBuilder splitter 옵션으로 바꾼다."""

        type_map = {
            "recursive_sentence": "RecursiveCharacterTextSplitter",
            "recursive": "RecursiveCharacterTextSplitter",
            "character": "CharacterTextSplitter",
            "semantic": "SemanticChunker",
        }
        return [{"type": type_map.get(str(value), str(value))} for value in values]

    def _retriever_options(
        self,
        values: list[Any],
        retriever_k: int,
    ) -> list[dict[str, Any]]:
        """canonical retriever 이름을 RAGBuilder retriever 옵션으로 바꾼다."""

        options: list[dict[str, Any]] = []
        for value in values:
            normalized = str(value)
            if normalized == "hybrid":
                # RAGBuilder가 dense/BM25 중 하나만 고르지 못하도록 두 검색기를
                # 내부에서 항상 결합하는 custom retriever 하나로 전달한다.
                options.append(
                    {
                        "type": "custom",
                        "custom_class": STRICT_HYBRID_CLASS,
                        "retriever_k": [retriever_k],
                        "retriever_kwargs": {"retriever_k": retriever_k},
                    }
                )
            elif normalized in {"dense", "vector", "vector_similarity"}:
                options.append(
                    {"type": "vector_similarity", "retriever_k": [retriever_k]}
                )
            else:
                options.append({"type": normalized, "retriever_k": [retriever_k]})
        return options

    # 범용 정규화 유틸 -------------------------------------------------------
    def _normalize_trial_status(
        self,
        status: Any,
    ) -> Literal["completed", "failed", "rejected", "unsupported"]:
        """외부 trial 상태를 AgentDoctor가 이해하는 네 상태로 제한한다."""

        normalized = str(status or "completed").lower()
        if normalized in {"complete", "completed", "success", "succeeded"}:
            return "completed"
        if normalized in {"unsupported"}:
            return "unsupported"
        if normalized in {"rejected", "pruned", "skipped"}:
            return "rejected"
        return "failed"

    def _flatten_dict(
        self,
        value: dict[str, Any],
        prefix: str = "",
    ) -> dict[str, Any]:
        """nested 외부 config를 dotted path로 평탄화한다."""

        flattened: dict[str, Any] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, dict):
                flattened.update(self._flatten_dict(item, path))
            else:
                flattened[path] = item
        return flattened

    def _first_present(
        self,
        mapping: dict[str, Any],
        keys: tuple[str, ...],
        default: Any,
    ) -> Any:
        """0이나 빈 문자열도 유효값으로 보존하면서 첫 번째 존재 key를 읽는다."""

        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
        return default

    def _dedupe_values(self, values: list[Any]) -> list[Any]:
        """입력 순서를 유지하면서 중복값을 제거한다."""

        deduped: list[Any] = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        return deduped

    def _filter_kwargs(
        self,
        callable_obj: Any,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """RAGBuilder 버전별 signature가 받는 keyword 인자만 남긴다."""

        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return kwargs
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            return kwargs
        return {key: value for key, value in kwargs.items() if key in signature.parameters}

    def _accepts_parameter(self, callable_obj: Any, name: str) -> bool:
        """callable이 특정 keyword 인자를 공식적으로 받는지 확인한다."""

        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return False
        return name in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def _accepts_positional_argument(self, callable_obj: Any) -> bool:
        """bound method가 config 위치 인자를 받을 수 있는지 확인한다."""

        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return False
        return any(
            parameter.kind
            in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
            for parameter in signature.parameters.values()
        ) or any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in signature.parameters.values()
        )

    def _ensure_dict(self, value: Any) -> dict[str, Any]:
        """외부 실행 결과가 mapping이 아니면 명확한 형식 오류로 바꾼다."""

        plain = self._to_plain_dict(value)
        if not isinstance(plain, dict):
            raise _AdapterExecutionError(
                "invalid_result_shape",
                "RAGBuilder 결과가 dict 형태가 아닙니다.",
            )
        return plain

    def _ensure_mapping(self, value: Any) -> dict[str, Any]:
        """metrics 같은 선택 필드를 안전한 dict로 정규화한다."""

        plain = self._to_plain_dict(value or {})
        return plain if isinstance(plain, dict) else {}

    def _model_dump(self, value: Any) -> Any:
        """Pydantic 결과에서 실행 객체를 제외한 serializable 필드만 추출한다."""

        if hasattr(value, "model_dump"):
            try:
                return value.model_dump(
                    exclude={"best_pipeline", "best_index"},
                    mode="python",
                )
            except TypeError:
                return value.model_dump(exclude={"best_pipeline", "best_index"})
        return self._to_plain_dict(value)

    def _to_plain_dict(self, value: Any, _seen: set[int] | None = None) -> Any:
        """dataclass, 객체, pandas-like 값을 plain dict/list로 재귀 변환한다."""

        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Enum):
            return value.value

        seen = _seen if _seen is not None else set()
        object_id = id(value)
        if object_id in seen:
            return "<circular-reference>"
        seen.add(object_id)

        if is_dataclass(value):
            return self._to_plain_dict(asdict(value), seen)
        if hasattr(value, "model_dump"):
            try:
                dumped = value.model_dump(
                    exclude={"best_pipeline", "best_index"},
                    mode="python",
                )
            except TypeError:
                dumped = value.model_dump(exclude={"best_pipeline", "best_index"})
            return self._to_plain_dict(dumped, seen)
        if isinstance(value, dict):
            return {
                key: self._to_plain_dict(item, seen)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._to_plain_dict(item, seen) for item in value]
        if hasattr(value, "to_dict"):
            try:
                return self._to_plain_dict(value.to_dict("records"), seen)
            except TypeError:
                return self._to_plain_dict(value.to_dict(), seen)
        if hasattr(value, "__dict__") and not isinstance(value, type):
            return {
                key: self._to_plain_dict(item, seen)
                for key, item in vars(value).items()
                if not key.startswith("_")
                and key not in {"best_pipeline", "best_index"}
            }
        return value

    def _coerce_float(self, value: Any) -> float | None:
        """bool을 제외한 score 값을 안전하게 float로 변환한다."""

        if value is None or isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def run(request: OptimizationRequest) -> RAGBuilderResult:
    """optimizer가 사용할 함수형 adapter 진입점."""

    return RAGBuilderAdapter().run(request)
