"""
AgentDoctor optimize에서 RAGBuilder를 외부 최적화 실행기로 감싸는 어댑터.

이 모듈의 책임은 의도적으로 좁다.
  1. OptimizationRequest를 RAGBuilder surrogate pipeline payload로 변환한다.
  2. RAGBuilder를 실행하거나, 명시적으로 요청된 개발/테스트용 mock 실행을 수행한다.
  3. 외부 실행 결과를 AgentDoctor 표준 RAGBuilderResult로 정규화한다.

이 모듈은 후보 config의 accept/apply/rollback 여부를 결정하지 않는다.
그 판단은 사용자 실제 pipeline에서 검증한 뒤 optimizer와 internal_adapter가 담당한다.
"""
from __future__ import annotations

import inspect
from dataclasses import asdict, is_dataclass
from typing import Any

from agents.optimize.config_mapper import get_current_value, map_prescriptions_to_config
from agents.optimize.schemas import (
    ConfigMappingResult,
    OptimizationRequest,
    RAGBuilderResult,
    RAGBuilderTrialResult,
)


# ---------------------------------------------------------------------------
# Config path 변환 테이블
# ---------------------------------------------------------------------------
# config_mapper는 AgentDoctor 내부 표준 path를 만든다. RAGBuilderAdapter는
# 그 path를 RAGBuilder payload에서 쓰는 path로 바꾸고, 결과를 받을 때는
# 다시 AgentDoctor path로 되돌린다.
RAGBUILDER_KEY_MAP: dict[str, str] = {
    "retriever.top_k": "retrieval.top_k",
    "retriever.search_type": "retrieval.retriever_type",
    "reranker.enabled": "retrieval.reranker.enabled",
    "reranker.candidate_count": "retrieval.reranker.top_n",
    "context.compression.enabled": "retrieval.context_compression.enabled",
    "chunker.chunk_size": "data_ingest.chunk_size",
    "chunker.chunk_overlap": "data_ingest.chunk_overlap",
    "chunker.strategy": "data_ingest.chunking_strategy",
}

AGENTDOCTOR_KEY_MAP: dict[str, str] = {
    external: internal for internal, external in RAGBUILDER_KEY_MAP.items()
}

# ---------------------------------------------------------------------------
# MVP 기본 capability
# ---------------------------------------------------------------------------
# 현재 AgentDoctor pipeline이 실제로 지원하는 옵션만 기본 True로 둔다.
# reranker, hybrid, chunking_strategy 등은 RAGBuilder가 지원하더라도
# 사용자 pipeline에 적용할 수 없으면 search space에 넣지 않는다.
DEFAULT_CAPABILITIES: dict[str, bool] = {
    "retriever.top_k": True,
    "hybrid_search": False,
    "reranker": False,
    "chunking": True,
    "chunking_strategy": False,
    "context_compression": False,
}

DEFAULT_RERANKER = "BAAI/bge-reranker-base"
DEFAULT_RETRIEVER_K = 20


class RAGBuilderAdapter:
    """RAGBuilder 외부 optimizer와 AgentDoctor optimize 흐름 사이의 경계."""

    def run(self, request: OptimizationRequest) -> RAGBuilderResult:
        """전체 어댑터 실행 흐름: mapping -> payload -> execute -> normalize."""

        mapping = self.build_mapping(request)
        if not mapping.search_space:
            # 탐색 가능한 config가 없으면 RAGBuilder를 호출하지 않는다.
            # 이는 실패가 아니라 "이번 request에서 최적화할 축이 없음"에 가깝다.
            return RAGBuilderResult(
                request_id=request.request_id,
                best_config=None,
                best_score=None,
                optimized_stage=self._infer_optimized_stage({}),
                search_space={},
                status="skipped",
                error="empty_search_space",
                warnings=list(mapping.warnings),
                metadata={
                    "skipped_prescriptions": [
                        self._to_plain_dict(item) for item in mapping.skipped
                    ],
                    "mapping": self._to_plain_dict(mapping),
                },
            )

        payload = self.build_payload(request, mapping)

        try:
            # mock은 명시적으로 요청된 경우에만 사용한다. 운영 경로에서는 실제
            # RAGBuilder 또는 주입된 ragbuilder_client를 사용해야 한다.
            if self._use_mock(request):
                raw_result = self._mock_result(payload)
            else:
                raw_result = self.execute(payload, request)
            return self.normalize_result(
                raw_result=raw_result,
                request=request,
                payload=payload,
                mapping=mapping,
            )
        except Exception as exc:
            # RAGBuilder는 외부 런타임 경계다. 예외를 밖으로 터뜨리기보다
            # 실패한 RAGBuilderResult로 감싸서 optimizer가 fallback/보고를
            # 결정할 수 있게 한다.
            return RAGBuilderResult(
                request_id=request.request_id,
                best_config=None,
                best_score=None,
                optimized_stage=payload["optimized_stage"],
                search_space=mapping.search_space,
                payload=payload,
                raw_result={},
                status="failed",
                error=str(exc),
                warnings=list(mapping.warnings),
                metadata={
                    "skipped_prescriptions": [
                        self._to_plain_dict(item) for item in mapping.skipped
                    ],
                    "mapping": self._to_plain_dict(mapping),
                },
            )

    def build_mapping(self, request: OptimizationRequest) -> ConfigMappingResult:
        """request를 AgentDoctor canonical search space로 변환한다."""

        if request.search_space:
            # planner/optimizer가 이미 search_space를 명시했다면 처방 id 재해석을
            # 하지 않고 그대로 사용한다.
            result = ConfigMappingResult()
            result.search_space = {
                path: list(values) if isinstance(values, list) else [values]
                for path, values in request.search_space.items()
            }
            result.metadata["source"] = "request.search_space"
            return result

        prescription_ids = self._extract_prescription_ids(request)
        capabilities = dict(DEFAULT_CAPABILITIES)
        capabilities.update(request.metadata.get("capabilities", {}))

        # prescription id -> canonical config path/value 후보 변환은
        # config_mapper의 책임이다. adapter는 그 결과를 소비만 한다.
        return map_prescriptions_to_config(
            prescription_ids=prescription_ids,
            current_config=request.baseline_config,
            capabilities=capabilities,
            constraints=request.metadata.get("constraints", {}),
        )

    def build_payload(
        self,
        request: OptimizationRequest,
        mapping: ConfigMappingResult,
    ) -> dict[str, Any]:
        """canonical mapping 결과를 RAGBuilder 실행 payload로 변환한다."""

        search_space = self._to_ragbuilder_search_space(mapping.search_space)
        optimized_stage = self._infer_optimized_stage(mapping.search_space)
        input_source = self._get_input_source(request)
        eval_dataset = request.metadata.get("eval_dataset") or request.metadata.get(
            "test_dataset"
        )

        return {
            "request_id": request.request_id,
            "optimized_stage": optimized_stage,
            "input_source": input_source,
            "eval_dataset": eval_dataset,
            "objective": {
                "metrics": list(request.target_metrics),
                "guardrails": list(request.guardrails),
                "target_profile": request.target_profile,
            },
            "budget": {
                "max_trials": request.max_trials,
                "n_trials": request.max_trials,
                **request.metadata.get("budget", {}),
            },
            "surrogate_pipeline": self._build_surrogate_pipeline(
                request=request,
                search_space=search_space,
            ),
            # RAGBuilder용 search_space와 AgentDoctor 원본 search_space를 둘 다
            # 보존한다. 결과 해석과 debugging에서 path 혼선을 줄이기 위함이다.
            "search_space": search_space,
            "agentdoctor_search_space": mapping.search_space,
            "fixed_config": dict(request.fixed_config),
            "metadata": {
                "failure_label": request.failure_label,
                "related_failure_labels": list(request.related_failure_labels),
                "reason": request.reason,
                "mapping_warnings": list(mapping.warnings),
                "skipped_prescriptions": [
                    self._to_plain_dict(item) for item in mapping.skipped
                ],
            },
        }

    def execute(
        self,
        payload: dict[str, Any],
        request: OptimizationRequest,
    ) -> dict[str, Any]:
        """
        RAGBuilder 실행 경계.

        request.metadata["ragbuilder_client"]가 있으면 테스트나 별도 실행기에
        위임한다. 없으면 설치된 ragbuilder 패키지를 사용한다.
        """

        client = request.metadata.get("ragbuilder_client")
        if client is not None:
            # 단위 테스트와 실험 환경에서 실제 RAGBuilder 설치 없이도 adapter
            # 배관을 검증할 수 있게 하는 주입 지점이다.
            if hasattr(client, "optimize"):
                return self._to_plain_dict(client.optimize(payload))
            if callable(client):
                return self._to_plain_dict(client(payload))
            raise TypeError("ragbuilder_client must be callable or expose optimize()")

        try:
            from ragbuilder import RAGBuilder
        except ImportError as exc:
            raise RuntimeError(
                "ragbuilder is not installed. Install requirements or set "
                "request.metadata['use_mock']=True for development tests."
            ) from exc

        builder = self._create_builder(RAGBuilder, payload, request)
        result = self._run_builder(builder, payload)
        return self._to_plain_dict(result)

    def normalize_result(
        self,
        raw_result: dict[str, Any],
        request: OptimizationRequest,
        payload: dict[str, Any],
        mapping: ConfigMappingResult,
    ) -> RAGBuilderResult:
        """RAGBuilder/raw client 결과를 AgentDoctor 표준 schema로 정규화한다."""

        raw_result = self._to_plain_dict(raw_result)

        if raw_result.get("status") == "failed":
            # 외부 실행 실패도 표준 schema로 감싸서 reporter/optimizer가 같은
            # 방식으로 처리할 수 있게 한다.
            return RAGBuilderResult(
                request_id=request.request_id,
                best_config=None,
                best_score=None,
                optimized_stage=payload["optimized_stage"],
                search_space=mapping.search_space,
                payload=payload,
                raw_result=raw_result,
                status="failed",
                error=str(raw_result.get("error") or "ragbuilder_failed"),
                warnings=list(mapping.warnings),
                metadata={
                    "skipped_prescriptions": [
                        self._to_plain_dict(item) for item in mapping.skipped
                    ],
                    "mapping": self._to_plain_dict(mapping),
                },
            )

        trials = self._normalize_trials(raw_result)
        best_config = self._extract_best_config(raw_result, trials)
        best_score = self._extract_best_score(raw_result, trials)

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
            error=None,
            warnings=list(mapping.warnings),
            metadata={
                "is_mock": bool(raw_result.get("mock")),
                "skipped_prescriptions": [
                    self._to_plain_dict(item) for item in mapping.skipped
                ],
                "mapping": self._to_plain_dict(mapping),
            },
        )

    def _extract_prescription_ids(self, request: OptimizationRequest) -> list[str]:
        """candidate와 metadata에서 처방 id를 중복 없이 추출한다."""

        ids: list[str] = []
        for candidate in request.candidates:
            if candidate.id and candidate.id not in ids:
                ids.append(candidate.id)

        metadata_ids = request.metadata.get("prescription_ids", [])
        for prescription_id in metadata_ids:
            if prescription_id not in ids:
                ids.append(prescription_id)
        return ids

    def _to_ragbuilder_search_space(
        self,
        search_space: dict[str, list[Any]],
    ) -> dict[str, list[Any]]:
        """AgentDoctor canonical search_space를 RAGBuilder path로 변환한다."""

        external: dict[str, list[Any]] = {}
        for path, values in search_space.items():
            external_path = RAGBUILDER_KEY_MAP.get(path, path)
            external[external_path] = list(values)
        return external

    def _build_surrogate_pipeline(
        self,
        request: OptimizationRequest,
        search_space: dict[str, list[Any]],
    ) -> dict[str, Any]:
        """사용자 pipeline과 전이 가능성이 높은 surrogate pipeline 정보를 만든다."""

        current = request.baseline_config
        metadata = request.metadata

        return {
            "input_source": self._get_input_source(request),
            "eval_dataset": metadata.get("eval_dataset")
            or metadata.get("test_dataset"),
            "vectorstore_type": metadata.get("vectorstore_type", "ragbuilder_default"),
            "chunking": {
                "chunk_size": get_current_value(current, "chunker.chunk_size", 512),
                "chunk_overlap": get_current_value(
                    current, "chunker.chunk_overlap", 50
                ),
                "strategy": get_current_value(
                    current, "chunker.strategy", "CharacterTextSplitter"
                ),
            },
            "embedding_model": get_current_value(
                current,
                "embedding_model",
                metadata.get("embedding_model"),
            ),
            "retrieval": {
                "retriever_type": get_current_value(
                    current, "retriever.search_type", "dense"
                ),
                "top_k": get_current_value(current, "retriever.top_k", 3),
                "retriever_k": get_current_value(
                    current,
                    "retriever.retriever_k",
                    DEFAULT_RETRIEVER_K,
                ),
            },
            "reranker": {
                "enabled": bool(get_current_value(current, "reranker.enabled", False)),
                "model": metadata.get("reranker_model", DEFAULT_RERANKER),
                "top_n": get_current_value(current, "reranker.candidate_count", None),
            },
            "search_space": search_space,
        }

    def _create_builder(
        self,
        ragbuilder_cls: Any,
        payload: dict[str, Any],
        request: OptimizationRequest,
    ) -> Any:
        """RAGBuilder 인스턴스를 생성한다.

        가능한 경우 세부 config class를 사용하고, 버전/API 차이로 실패하면
        from_source_with_defaults 경로로 fallback한다.
        """

        data_config = self._build_data_ingest_config(payload, request)

        default_llm = request.metadata.get("default_llm")
        default_embeddings = request.metadata.get("default_embeddings")
        builder_kwargs = {
            "data_ingest_config": data_config,
            "default_llm": default_llm,
            "default_embeddings": default_embeddings,
        }
        builder_kwargs = {
            key: value for key, value in builder_kwargs.items() if value is not None
        }

        if builder_kwargs:
            try:
                return ragbuilder_cls(**builder_kwargs)
            except TypeError:
                # RAGBuilder 버전에 따라 생성자 signature가 다를 수 있다.
                # 아래의 더 보수적인 factory 경로로 다시 시도한다.
                pass

        input_source = payload.get("input_source")
        if not input_source:
            raise ValueError("RAGBuilder requires input_source in request.metadata")

        factory = getattr(ragbuilder_cls, "from_source_with_defaults", None)
        if factory is None:
            raise TypeError("RAGBuilder has no supported constructor")

        factory_kwargs = self._filter_kwargs(
            factory,
            {
                "input_source": input_source,
                "test_dataset": payload.get("eval_dataset"),
                "n_trials": payload["budget"].get("n_trials"),
            },
        )
        return factory(**factory_kwargs)

    def _run_builder(self, builder: Any, payload: dict[str, Any]) -> Any:
        """optimized_stage에 맞춰 RAGBuilder module-level optimization을 실행한다."""

        stage = payload["optimized_stage"]

        if stage in {"data_ingest", "full"} and hasattr(
            builder, "optimize_data_ingest"
        ):
            builder.optimize_data_ingest()

        if stage in {"retrieval", "full"} and hasattr(builder, "optimize_retrieval"):
            retrieval_config = self._build_retrieval_config(payload)
            if retrieval_config is not None:
                builder.optimize_retrieval(retrieval_config)
            else:
                builder.optimize_retrieval()

        if stage == "generation" and hasattr(builder, "optimize_generation"):
            generation_config = self._build_generation_config(payload)
            if generation_config is not None:
                builder.optimize_generation(generation_config)
            else:
                builder.optimize_generation()

        if hasattr(builder, "optimization_results"):
            return builder.optimization_results

        if hasattr(builder, "optimize"):
            return builder.optimize()

        return builder

    def _build_data_ingest_config(
        self,
        payload: dict[str, Any],
        request: OptimizationRequest,
    ) -> Any | None:
        """chunking/embedding 계열 RAGBuilder data ingest config를 만든다."""

        config_cls = self._load_ragbuilder_config_class("DataIngestOptionsConfig")
        if config_cls is None:
            return None

        search_space = payload["search_space"]
        surrogate = payload["surrogate_pipeline"]
        chunking = surrogate["chunking"]

        chunk_size_values = search_space.get("data_ingest.chunk_size")
        chunk_overlap_values = search_space.get("data_ingest.chunk_overlap")
        strategy_values = search_space.get("data_ingest.chunking_strategy")

        kwargs = {
            "input_source": payload.get("input_source"),
            "chunk_size": self._numeric_space_or_value(
                chunk_size_values,
                chunking["chunk_size"],
            ),
            "chunk_overlap": self._numeric_space_or_value(
                chunk_overlap_values,
                chunking["chunk_overlap"],
            ),
            "chunking_strategies": self._chunking_strategies(
                strategy_values or [chunking["strategy"]]
            ),
            "embedding_models": request.metadata.get("embedding_models"),
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        return config_cls(**self._filter_kwargs(config_cls, kwargs))

    def _build_retrieval_config(self, payload: dict[str, Any]) -> Any | None:
        """retriever/reranker/top_k 계열 RAGBuilder retrieval config를 만든다."""

        config_cls = self._load_ragbuilder_config_class("RetrievalOptionsConfig")
        if config_cls is None:
            return None

        search_space = payload["search_space"]
        surrogate = payload["surrogate_pipeline"]

        retriever_types = search_space.get("retrieval.retriever_type")
        if not retriever_types:
            retriever_types = [surrogate["retrieval"]["retriever_type"]]

        top_k = search_space.get("retrieval.top_k") or [
            surrogate["retrieval"]["top_k"]
        ]

        rerank_enabled = search_space.get("retrieval.reranker.enabled")
        rerankers = None
        if True in (rerank_enabled or [surrogate["reranker"]["enabled"]]):
            rerankers = [{"type": surrogate["reranker"]["model"]}]

        kwargs = {
            "retrievers": self._retriever_options(retriever_types),
            "rerankers": rerankers,
            "top_k": top_k,
            "optimization": {
                "n_trials": payload["budget"].get("n_trials"),
                "optimization_direction": "maximize",
            },
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        return config_cls(**self._filter_kwargs(config_cls, kwargs))

    def _build_generation_config(self, payload: dict[str, Any]) -> Any | None:
        """generation 계열 최적화를 위한 최소 RAGBuilder config를 만든다."""

        config_cls = self._load_ragbuilder_config_class("GenerationOptionsConfig")
        if config_cls is None:
            return None

        kwargs = {
            "optimization": {
                "n_trials": payload["budget"].get("n_trials"),
                "optimization_direction": "maximize",
            }
        }
        return config_cls(**self._filter_kwargs(config_cls, kwargs))

    def _load_ragbuilder_config_class(self, name: str) -> Any | None:
        """RAGBuilder config class를 optional dependency처럼 안전하게 로드한다."""

        try:
            module = __import__("ragbuilder.config", fromlist=[name])
        except ImportError:
            return None
        return getattr(module, name, None)

    def _normalize_trials(self, raw_result: dict[str, Any]) -> list[RAGBuilderTrialResult]:
        """RAGBuilder가 반환한 다양한 trial 형태를 표준 trial list로 맞춘다."""

        raw_trials = (
            raw_result.get("trial_results")
            or raw_result.get("trials")
            or raw_result.get("results")
            or []
        )

        if hasattr(raw_trials, "to_dict"):
            raw_trials = raw_trials.to_dict("records")

        trials: list[RAGBuilderTrialResult] = []
        for index, raw_trial in enumerate(raw_trials):
            trial = self._to_plain_dict(raw_trial)
            if not isinstance(trial, dict):
                trial = {"value": trial}

            config = (
                trial.get("config")
                or trial.get("params")
                or trial.get("best_config")
                or {}
            )
            config = self._to_agentdoctor_config(config)
            score = self._coerce_float(
                trial.get("score")
                or trial.get("value")
                or trial.get("best_score")
                or trial.get("metric")
            )
            trial_id = str(
                trial.get("trial_id")
                or trial.get("id")
                or trial.get("number")
                or f"ragbuilder-trial-{index + 1}"
            )

            trials.append(
                RAGBuilderTrialResult(
                    trial_id=trial_id,
                    config=config,
                    score=score,
                    metrics=self._to_plain_dict(trial.get("metrics") or {}),
                    status=trial.get("status", "completed"),
                    unsupported_reasons=list(trial.get("unsupported_reasons") or []),
                    raw_trial=trial,
                )
            )

        if not trials:
            # 일부 RAGBuilder 결과는 trial 목록 없이 best_config만 줄 수 있다.
            # downstream은 trial_results를 기대하므로 단일 trial로 감싸 둔다.
            best_config = self._to_agentdoctor_config(
                raw_result.get("best_config") or raw_result.get("config") or {}
            )
            if best_config:
                trials.append(
                    RAGBuilderTrialResult(
                        trial_id="ragbuilder-best",
                        config=best_config,
                        score=self._coerce_float(raw_result.get("best_score")),
                        metrics=self._to_plain_dict(raw_result.get("metrics") or {}),
                        raw_trial=raw_result,
                    )
                )
        return trials

    def _extract_best_config(
        self,
        raw_result: dict[str, Any],
        trials: list[RAGBuilderTrialResult],
    ) -> dict[str, Any] | None:
        """raw result의 best_config를 우선 사용하고, 없으면 최고 점수 trial을 고른다."""

        best = raw_result.get("best_config") or raw_result.get("config")
        best = self._to_agentdoctor_config(best or {})
        if best:
            return best
        if not trials:
            return None
        completed = [trial for trial in trials if trial.status == "completed"]
        candidates = completed or trials
        return max(candidates, key=lambda trial: trial.score or float("-inf")).config

    def _extract_best_score(
        self,
        raw_result: dict[str, Any],
        trials: list[RAGBuilderTrialResult],
    ) -> float | None:
        """raw result 또는 trial 목록에서 best score를 추출한다."""

        score = self._coerce_float(raw_result.get("best_score") or raw_result.get("score"))
        if score is not None:
            return score
        scored = [trial.score for trial in trials if trial.score is not None]
        return max(scored) if scored else None

    def _to_agentdoctor_config(self, config: Any) -> dict[str, Any]:
        """RAGBuilder path로 온 config를 AgentDoctor canonical path로 되돌린다."""

        plain = self._to_plain_dict(config)
        if not isinstance(plain, dict):
            return {}

        converted: dict[str, Any] = {}
        for key, value in plain.items():
            canonical = AGENTDOCTOR_KEY_MAP.get(key, key)
            converted[canonical] = value
        return converted

    def _infer_optimized_stage(self, search_space: dict[str, Any]) -> str:
        """search_space에 포함된 path를 보고 최적화 stage를 추론한다."""

        if not search_space:
            return "retrieval"
        keys = set(search_space)
        has_data = any(key.startswith("chunker.") for key in keys)
        has_retrieval = any(
            key.startswith(("retriever.", "reranker.", "context.")) for key in keys
        )
        if has_data and has_retrieval:
            return "full"
        if has_data:
            return "data_ingest"
        return "retrieval"

    def _get_input_source(self, request: OptimizationRequest) -> Any:
        """RAGBuilder surrogate pipeline에 사용할 input_source를 찾는다."""

        metadata = request.metadata
        return (
            metadata.get("input_source")
            or metadata.get("source_url")
            or request.fixed_config.get("input_source")
            or request.baseline_config.get("input_source")
        )

    def _use_mock(self, request: OptimizationRequest) -> bool:
        """개발/테스트용 mock 실행 여부를 확인한다."""

        return bool(request.metadata.get("use_mock") or request.metadata.get("mock"))

    def _mock_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        """명시적으로 요청된 경우에만 사용하는 가짜 RAGBuilder 결과를 만든다."""

        trials = []
        best_config = None
        best_score = None

        for index, config in enumerate(self._first_trial_configs(payload["search_space"])):
            score = round(1.0 - (index * 0.05), 4)
            if best_score is None or score > best_score:
                best_score = score
                best_config = config
            trials.append(
                {
                    "trial_id": f"mock-ragbuilder-{index + 1}",
                    "config": config,
                    "score": score,
                    "metrics": {"mock_objective": score},
                    "status": "completed",
                }
            )

        return {
            "mock": True,
            "status": "completed",
            "best_config": best_config or {},
            "best_score": best_score,
            "trial_results": trials,
            "payload": payload,
        }

    def _first_trial_configs(self, search_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
        """mock 실행에서 search_space를 단일 변경 trial들로 펼친다."""

        configs: list[dict[str, Any]] = []
        for path, values in search_space.items():
            for value in values[:3]:
                configs.append({path: value})
        return configs[:5]

    def _chunking_strategies(self, values: list[Any]) -> list[dict[str, Any]]:
        """AgentDoctor chunking strategy 값을 RAGBuilder splitter option으로 바꾼다."""

        strategies = []
        for value in values:
            splitter_type = {
                "recursive_sentence": "RecursiveCharacterTextSplitter",
                "recursive": "RecursiveCharacterTextSplitter",
                "character": "CharacterTextSplitter",
                "semantic": "SemanticChunker",
            }.get(str(value), str(value))
            strategies.append({"type": splitter_type})
        return strategies

    def _retriever_options(self, values: list[Any]) -> list[dict[str, Any]]:
        """AgentDoctor retriever 값을 RAGBuilder retriever option으로 바꾼다."""

        options: list[dict[str, Any]] = []
        for value in values:
            normalized = str(value)
            if normalized == "hybrid":
                options.extend(
                    [
                        {"type": "vector_similarity", "retriever_k": [DEFAULT_RETRIEVER_K]},
                        {"type": "bm25", "retriever_k": [DEFAULT_RETRIEVER_K]},
                    ]
                )
            elif normalized in {"dense", "vector", "vector_similarity"}:
                options.append(
                    {"type": "vector_similarity", "retriever_k": [DEFAULT_RETRIEVER_K]}
                )
            elif normalized == "bm25":
                options.append({"type": "bm25", "retriever_k": [DEFAULT_RETRIEVER_K]})
            else:
                options.append({"type": normalized, "retriever_k": [DEFAULT_RETRIEVER_K]})
        return options

    def _numeric_space_or_value(
        self,
        values: list[Any] | None,
        default: Any,
    ) -> Any:
        """숫자 후보 목록을 RAGBuilder가 선호하는 range 형태로 줄일 수 있으면 줄인다."""

        if not values:
            return default
        numeric_values = [value for value in values if isinstance(value, (int, float))]
        if len(numeric_values) != len(values):
            return values
        return self._values_to_range(numeric_values)

    def _values_to_range(self, values: list[int | float]) -> dict[str, Any] | list[Any]:
        """등간격 숫자 후보를 min/max/stepsize range로 압축한다."""

        values = sorted(set(values))
        if not values:
            return []
        if len(values) == 1:
            return {"min": values[0], "max": values[0], "stepsize": 1}
        diffs = [values[index + 1] - values[index] for index in range(len(values) - 1)]
        if len(set(diffs)) == 1:
            return {"min": values[0], "max": values[-1], "stepsize": diffs[0]}
        return values

    def _filter_kwargs(self, callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        """RAGBuilder 버전별 signature 차이를 피하기 위해 지원 kwargs만 남긴다."""

        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return kwargs

        params = signature.parameters
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
            return kwargs
        return {key: value for key, value in kwargs.items() if key in params}

    def _to_plain_dict(self, value: Any) -> Any:
        """dataclass, 객체, pandas-like 결과 등을 plain dict/list로 변환한다."""

        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, dict):
            return {key: self._to_plain_dict(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_plain_dict(item) for item in value]
        if isinstance(value, tuple):
            return [self._to_plain_dict(item) for item in value]
        if hasattr(value, "to_dict"):
            try:
                return value.to_dict()
            except TypeError:
                pass
        if hasattr(value, "__dict__") and not isinstance(value, type):
            return {
                key: self._to_plain_dict(item)
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        return value

    def _coerce_float(self, value: Any) -> float | None:
        """score처럼 들어온 값을 안전하게 float로 변환한다."""

        if value is None or isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def run(request: OptimizationRequest) -> RAGBuilderResult:
    """다른 AgentDoctor 모듈과 맞춘 함수형 진입점."""

    return RAGBuilderAdapter().run(request)
