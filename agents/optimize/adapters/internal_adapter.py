"""
agents/optimize/adapters/internal_adapter.py

AgentDoctor 자체 search-space 최적화 backend.

읽기: OptimizationRequest의 baseline_config, search_space, target_metrics, metadata
쓰기: AgentDoctorState를 직접 수정하지 않고 InternalAdapterResult만 반환

두 실행 방식을 같은 계약으로 지원한다.
  1. evaluator가 주입되면 후보를 순서대로 실제 평가하고 best config를 반환한다.
  2. evaluator가 없으면 metadata의 기존 trial 결과를 읽고 다음 미평가 후보를
     반환한다. LangGraph의 Index→Eval 루프가 이 후보를 평가한 뒤 다시 넘길 수 있다.

미평가 후보(next_config)와 평가 완료 후보(best_config)를 분리하며, 후보별 오류는
전체 탐색을 중단시키지 않는다. 검색·임베딩·진단은 이 모듈에서 직접 수행하지 않고
주입 evaluator 또는 AgentDoctor의 기존 Index/Eval 경계를 사용한다.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any, Callable

from agents.optimize.config_mapper import (
    canonicalize_path,
    get_current_value,
    map_changes_to_index_config,
)
from agents.optimize.schemas import (
    InternalAdapterResult,
    InternalTrialResult,
    ObjectiveDirection,
    OptimizationRequest,
)


TrialEvaluator = Callable[[dict[str, Any], OptimizationRequest], Any]

_VALID_DIRECTIONS = {"maximize", "minimize"}
_VALID_TRIAL_STATUSES = {"completed", "failed", "inconclusive", "rejected"}
_MINIMIZE_OBJECTIVES = {
    "noise_sensitivity",
    "latency",
    "latency_ms",
    "error_rate",
    "cost",
}
_OBJECTIVE_ALIASES: dict[str, tuple[str, ...]] = {
    "context_recall": ("context_recall", "mean_recall_at_k"),
    "answer_relevancy": ("answer_relevancy", "response_relevancy"),
}


class InternalAdapter:
    """결정론적 단일 축 search-space optimizer."""

    def __init__(self, evaluator: TrialEvaluator | None = None):
        self.evaluator = evaluator

    def run(self, request: OptimizationRequest) -> InternalAdapterResult:
        """요청을 평가해 다음 후보 또는 평가 완료된 best config를 반환한다."""

        try:
            search_space = self._normalize_search_space(request)
            if not search_space:
                return self._result(
                    request,
                    status="skipped",
                    error="search_space가 비어 있거나 현재값 외 후보가 없습니다.",
                    search_space={},
                )

            max_trials = self._max_trials(request.max_trials)
            objective = self._objective_metric(request)
            direction = self._direction(request, objective)
            min_delta = self._min_delta(request)
            path, values = next(iter(search_space.items()))
            self._validate_fixed_config(request, path)
        except (TypeError, ValueError) as exc:
            return self._result(
                request,
                status="failed",
                error=str(exc),
                search_space={},
            )

        warnings: list[str] = []
        try:
            trials = self._load_trials(
                request,
                search_space,
                objective,
                warnings,
            )
            trials = self._limit_trials(trials, max_trials, warnings)
            candidates = [
                self._candidate(request, path, value)
                for value in values
            ]
        except (TypeError, ValueError) as exc:
            return self._result(
                request,
                status="failed",
                error=str(exc),
                objective=objective,
                direction=direction,
                search_space=search_space,
                warnings=warnings,
            )
        known = {trial.fingerprint for trial in trials if trial.fingerprint}

        baseline = next((trial for trial in trials if trial.is_baseline), None)
        baseline_needs_evaluation = baseline is None or (
            not self._is_completed(baseline) and not self._trial_passed(baseline)
        )
        if self.evaluator is not None and baseline_needs_evaluation:
            evaluated_baseline = self._evaluate(
                request,
                config={},
                objective=objective,
                is_baseline=True,
                warnings=warnings,
            )
            if baseline is not None:
                trials = [trial for trial in trials if trial is not baseline]
            baseline = evaluated_baseline
            trials.append(evaluated_baseline)
            known.add(evaluated_baseline.fingerprint)

        budget_used = self._budget_used(trials)
        early_stopped = self._has_passing_trial(trials)

        if self.evaluator is not None and not early_stopped:
            for candidate in candidates:
                if candidate["fingerprint"] in known:
                    continue
                if budget_used >= max_trials:
                    break

                trial = self._evaluate(
                    request,
                    config=candidate["config"],
                    objective=objective,
                    is_baseline=False,
                    warnings=warnings,
                )
                trials.append(trial)
                known.add(trial.fingerprint)
                budget_used += 1
                if self._trial_passed(trial):
                    early_stopped = True
                    break

        try:
            direction = self._direction(request, objective, trials)
        except ValueError as exc:
            return self._result(
                request,
                status="failed",
                error=str(exc),
                trials=trials,
                objective=objective,
                direction=direction,
                search_space=search_space,
                warnings=warnings,
            )

        unseen = [
            candidate
            for candidate in candidates
            if candidate["fingerprint"] not in known
        ]

        if (
            self.evaluator is None
            and not early_stopped
            and budget_used < max_trials
            and unseen
        ):
            best_trial = self._best_completed_trial(trials, direction)
            return self._result(
                request,
                status="needs_evaluation",
                next_config=dict(unseen[0]["config"]),
                best_config=(
                    None
                    if best_trial is None
                    else (
                        self._baseline_axis_config(request, search_space)
                        if best_trial.is_baseline
                        else dict(best_trial.config)
                    )
                ),
                best_score=best_trial.score if best_trial else None,
                trials=trials,
                objective=objective,
                direction=direction,
                search_space=search_space,
                warnings=warnings,
                metadata={
                    "budget_used": budget_used,
                    "max_trials": max_trials,
                    "stop_reason": "candidate_requires_evaluation",
                    "best_is_baseline": bool(best_trial and best_trial.is_baseline),
                },
            )

        return self._complete(
            request=request,
            trials=trials,
            objective=objective,
            direction=direction,
            min_delta=min_delta,
            search_space=search_space,
            warnings=warnings,
            budget_used=budget_used,
            max_trials=max_trials,
        )

    # 요청 검증 -------------------------------------------------------------
    def _normalize_search_space(
        self,
        request: OptimizationRequest,
    ) -> dict[str, list[Any]]:
        raw = request.search_space
        if not isinstance(raw, dict):
            raise TypeError("search_space는 dict여야 합니다.")
        if not raw:
            return {}

        normalized: dict[str, list[Any]] = {}
        for raw_path, raw_values in raw.items():
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError("search_space 경로는 비어 있지 않은 문자열이어야 합니다.")
            if isinstance(raw_values, dict):
                raise TypeError(f"후보값은 scalar 또는 list여야 합니다: {raw_path}")

            values = (
                list(raw_values)
                if isinstance(raw_values, (list, tuple))
                else [raw_values]
            )
            path = canonicalize_path(raw_path)
            normalized.setdefault(path, []).extend(values)

        if len(normalized) != 1:
            raise ValueError("internal optimizer는 한 번에 config 축 하나만 탐색합니다.")

        path, values = next(iter(normalized.items()))
        current = get_current_value(self._study_baseline_config(request), path)
        deduped: list[Any] = []
        for value in values:
            if value == current or value in deduped:
                continue
            deduped.append(deepcopy(value))
        return {path: deduped} if deduped else {}

    def _max_trials(self, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("max_trials는 1 이상의 정수여야 합니다.")
        return value

    def _objective_metric(self, request: OptimizationRequest) -> str:
        configured = request.metadata.get("primary_metric")
        if configured is None:
            configured = request.metadata.get("objective_metric")
        if configured is None and request.target_metrics:
            configured = request.target_metrics[0]
        if configured is None:
            configured = "overall_score"
        if not isinstance(configured, str) or not configured.strip():
            raise ValueError("primary_metric은 비어 있지 않은 문자열이어야 합니다.")
        return configured.strip()

    def _direction(
        self,
        request: OptimizationRequest,
        objective: str,
        trials: list[InternalTrialResult] | None = None,
    ) -> ObjectiveDirection:
        comparison_metrics = {
            self._comparison_metric(trial, objective)
            for trial in (trials or [])
            if self._is_completed(trial)
        }
        comparison_metrics.discard(None)
        if len(comparison_metrics) > 1:
            raise ValueError(
                "trial들이 서로 다른 목적 지표로 평가되어 비교할 수 없습니다: "
                + ", ".join(sorted(comparison_metrics))
            )
        effective_metric = next(iter(comparison_metrics), objective)

        configured = request.metadata.get("optimization_direction")
        if configured is None:
            configured = request.metadata.get("objective_direction")
        if configured is not None:
            if effective_metric != objective:
                raise ValueError(
                    "명시한 optimization_direction을 fallback 지표에 적용할 수 없습니다: "
                    f"{objective} -> {effective_metric}"
                )
            raw = configured
        else:
            raw = "minimize" if effective_metric in _MINIMIZE_OBJECTIVES else "maximize"
        if raw not in _VALID_DIRECTIONS:
            raise ValueError("optimization_direction은 maximize 또는 minimize여야 합니다.")
        return raw

    def _min_delta(self, request: OptimizationRequest) -> float:
        raw = request.metadata.get("min_delta", 0.0)
        if not self._is_finite_number(raw) or float(raw) < 0:
            raise ValueError("min_delta는 0 이상의 유한한 숫자여야 합니다.")
        return float(raw)

    def _comparison_metric(
        self,
        trial: InternalTrialResult,
        objective: str,
    ) -> str | None:
        """alias는 같은 목적 지표로 묶고 실제 fallback 지표는 구분한다."""

        used = trial.metadata.get("used_metric")
        if not isinstance(used, str) or not used:
            return objective
        if used in _OBJECTIVE_ALIASES.get(objective, (objective,)):
            return objective
        return used

    def _validate_fixed_config(self, request: OptimizationRequest, path: str) -> None:
        fixed_paths = {
            canonicalize_path(key)
            for key in request.fixed_config
            if isinstance(key, str)
        }
        if path in fixed_paths:
            raise ValueError(f"fixed_config와 search_space가 충돌합니다: {path}")

    # trial 생성·정규화 -----------------------------------------------------
    def _load_trials(
        self,
        request: OptimizationRequest,
        search_space: dict[str, list[Any]],
        objective: str,
        warnings: list[str],
    ) -> list[InternalTrialResult]:
        raw_trials: list[Any] = []
        # 명시적인 baseline_trial이 baseline_metrics 요약보다 우선한다.
        baseline_trial = request.metadata.get("baseline_trial")
        if baseline_trial is not None:
            if isinstance(baseline_trial, dict):
                raw_trials.append({**baseline_trial, "is_baseline": True})
            elif isinstance(baseline_trial, InternalTrialResult):
                copied_baseline = deepcopy(baseline_trial)
                copied_baseline.is_baseline = True
                raw_trials.append(copied_baseline)
            else:
                raw_trials.append(baseline_trial)

        baseline_metrics = request.metadata.get("baseline_metrics")
        if isinstance(baseline_metrics, dict):
            raw_trials.append(
                {
                    "trial_id": f"{request.request_id}:baseline",
                    "config": {},
                    "metrics": baseline_metrics,
                    "is_baseline": True,
                }
            )

        supplied = request.metadata.get("trial_results", [])
        if isinstance(supplied, (list, tuple)):
            raw_trials.extend(supplied)
        elif supplied:
            raise TypeError("metadata.trial_results는 list여야 합니다.")

        path, allowed_values = next(iter(search_space.items()))
        trials: list[InternalTrialResult] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_trials):
            trial = self._normalize_trial(
                request=request,
                raw=raw,
                index=index,
                path=path,
                allowed_values=allowed_values,
                objective=objective,
                warnings=warnings,
            )
            if trial.fingerprint in seen:
                continue
            seen.add(trial.fingerprint)
            trials.append(trial)
        return trials

    def _limit_trials(
        self,
        trials: list[InternalTrialResult],
        max_trials: int,
        warnings: list[str],
    ) -> list[InternalTrialResult]:
        """baseline/잘못된 관측은 보존하고 실제 후보 관측은 budget까지만 사용한다."""

        limited: list[InternalTrialResult] = []
        used = 0
        for trial in trials:
            in_space = bool(trial.metadata.get("in_search_space", False))
            if trial.is_baseline or not in_space:
                limited.append(trial)
                continue
            if used >= max_trials:
                self._append_warning(
                    warnings,
                    "max_trials를 초과한 trial observation을 무시했습니다.",
                )
                continue
            limited.append(trial)
            used += 1
        return limited

    def _normalize_trial(
        self,
        *,
        request: OptimizationRequest,
        raw: Any,
        index: int,
        path: str,
        allowed_values: list[Any],
        objective: str,
        warnings: list[str],
    ) -> InternalTrialResult:
        if isinstance(raw, InternalTrialResult):
            data = {
                "trial_id": raw.trial_id,
                "config": deepcopy(raw.config),
                "score": raw.score,
                "metrics": deepcopy(raw.metrics),
                "status": raw.status,
                "is_baseline": raw.is_baseline,
                "error": raw.error,
                "metadata": deepcopy(raw.metadata),
            }
        elif isinstance(raw, dict):
            data = dict(raw)
        else:
            raise TypeError("trial_results 항목은 dict 또는 InternalTrialResult여야 합니다.")

        is_baseline = bool(data.get("is_baseline", False))
        config = data.get("config") or {}
        if not isinstance(config, dict):
            raise TypeError("trial config는 dict여야 합니다.")

        normalized_config: dict[str, Any] = {}
        status = data.get("status", "completed")
        error = data.get("error")
        in_search_space = is_baseline
        if status not in _VALID_TRIAL_STATUSES:
            status = "rejected"
            error = error or "지원하지 않는 trial status입니다."
        if not is_baseline:
            if len(config) != 1:
                status = "rejected"
                error = error or "trial config는 단일 축이어야 합니다."
            else:
                raw_path, value = next(iter(config.items()))
                canonical_path = canonicalize_path(raw_path)
                normalized_config = {canonical_path: value}
                in_search_space = canonical_path == path and value in allowed_values
                if not in_search_space:
                    status = "rejected"
                    error = error or "trial config가 요청 search_space 밖에 있습니다."

        if not is_baseline and not in_search_space:
            effective = deepcopy(self._study_baseline_config(request))
            fingerprint = self._fingerprint(
                {
                    "baseline_config": effective,
                    "rejected_config": normalized_config or config,
                }
            )
        else:
            effective = self._effective_config(request, normalized_config)
            fingerprint = self._fingerprint(effective)
        metrics = self._normalize_metrics(data.get("metrics", {}))
        if "pass_threshold" in data:
            metrics["pass_threshold"] = bool(data["pass_threshold"])

        score = data.get("score")
        used_metric = None
        if status == "completed" and not self._is_finite_number(score):
            score, used_metric = self._extract_score(
                metrics,
                objective,
                allow_fallback=bool(
                    request.metadata.get("allow_overall_fallback", True)
                ),
            )
            if score is None:
                status = "inconclusive"
                error = error or f"목적 지표를 찾을 수 없습니다: {objective}"
        elif self._is_finite_number(score):
            score = float(score)

        if used_metric and used_metric != objective:
            self._append_warning(
                warnings,
                f"목적 지표 {objective!r} 대신 {used_metric!r}을 사용했습니다.",
            )

        metadata = dict(data.get("metadata") or {})
        metadata.setdefault("effective_config", effective)
        metadata["in_search_space"] = in_search_space
        if used_metric:
            metadata.setdefault("used_metric", used_metric)

        return InternalTrialResult(
            trial_id=str(data.get("trial_id") or f"{request.request_id}:observed:{index}"),
            config=normalized_config,
            score=float(score) if self._is_finite_number(score) else None,
            metrics=metrics,
            status=status,
            is_baseline=is_baseline,
            fingerprint=fingerprint,
            error=str(error) if error is not None else None,
            metadata=metadata,
        )

    def _candidate(
        self,
        request: OptimizationRequest,
        path: str,
        value: Any,
    ) -> dict[str, Any]:
        config = {path: deepcopy(value)}
        effective = self._effective_config(request, config)
        return {
            "config": config,
            "effective_config": effective,
            "fingerprint": self._fingerprint(effective),
        }

    def _evaluate(
        self,
        request: OptimizationRequest,
        *,
        config: dict[str, Any],
        objective: str,
        is_baseline: bool,
        warnings: list[str],
    ) -> InternalTrialResult:
        effective = self._effective_config(request, config)
        fingerprint = self._fingerprint(effective)
        trial_id = f"{request.request_id}:{'baseline' if is_baseline else fingerprint[:12]}"
        try:
            raw = self.evaluator(deepcopy(effective), request)  # type: ignore[misc]
            metrics = self._normalize_metrics(raw)
            score, used_metric = self._extract_score(
                metrics,
                objective,
                allow_fallback=bool(
                    request.metadata.get("allow_overall_fallback", True)
                ),
            )
            if score is None:
                return InternalTrialResult(
                    trial_id=trial_id,
                    config=dict(config),
                    metrics=metrics,
                    status="inconclusive",
                    is_baseline=is_baseline,
                    fingerprint=fingerprint,
                    error=f"목적 지표를 찾을 수 없습니다: {objective}",
                    metadata={
                        "effective_config": effective,
                        "in_search_space": not is_baseline,
                    },
                )

            if used_metric != objective:
                self._append_warning(
                    warnings,
                    f"목적 지표 {objective!r} 대신 {used_metric!r}을 사용했습니다.",
                )
            return InternalTrialResult(
                trial_id=trial_id,
                config=dict(config),
                score=score,
                metrics=metrics,
                status="completed",
                is_baseline=is_baseline,
                fingerprint=fingerprint,
                metadata={
                    "effective_config": effective,
                    "used_metric": used_metric,
                    "in_search_space": not is_baseline,
                },
            )
        except Exception as exc:
            return InternalTrialResult(
                trial_id=trial_id,
                config=dict(config),
                status="failed",
                is_baseline=is_baseline,
                fingerprint=fingerprint,
                error=f"trial 평가 실패: {exc}",
                metadata={
                    "effective_config": effective,
                    "in_search_space": not is_baseline,
                },
            )

    # best 선택 -------------------------------------------------------------
    def _complete(
        self,
        *,
        request: OptimizationRequest,
        trials: list[InternalTrialResult],
        objective: str,
        direction: ObjectiveDirection,
        min_delta: float,
        search_space: dict[str, list[Any]],
        warnings: list[str],
        budget_used: int,
        max_trials: int,
    ) -> InternalAdapterResult:
        passing = [trial for trial in trials if self._trial_passed(trial)]
        if passing:
            scored_passing = [trial for trial in passing if self._is_completed(trial)]
            selected_pass = (
                self._best_completed_trial(scored_passing, direction)
                if scored_passing
                else passing[0]
            )
            return self._result(
                request,
                status="completed",
                best_config=self._trial_config(
                    request,
                    search_space,
                    selected_pass,
                ),
                best_score=selected_pass.score,
                trials=trials,
                objective=objective,
                direction=direction,
                search_space=search_space,
                warnings=warnings,
                metadata={
                    "best_trial_id": selected_pass.trial_id,
                    "best_is_baseline": selected_pass.is_baseline,
                    "improved": False if selected_pass.is_baseline else None,
                    "min_delta": min_delta,
                    "budget_used": budget_used,
                    "max_trials": max_trials,
                    "stop_reason": "pass_threshold_reached",
                    "unscored_pass": selected_pass.score is None,
                },
            )

        baseline = next(
            (
                trial
                for trial in trials
                if trial.is_baseline and self._is_completed(trial)
            ),
            None,
        )
        candidates = [
            trial
            for trial in trials
            if not trial.is_baseline and self._is_completed(trial)
        ]
        best_candidate = self._best_completed_trial(candidates, direction)

        if best_candidate is None:
            if baseline is not None:
                return self._result(
                    request,
                    status="completed",
                    best_config=self._baseline_axis_config(request, search_space),
                    best_score=baseline.score,
                    trials=trials,
                    objective=objective,
                    direction=direction,
                    search_space=search_space,
                    warnings=warnings,
                    metadata={
                        "best_is_baseline": True,
                        "improved": False,
                        "budget_used": budget_used,
                        "max_trials": max_trials,
                        "stop_reason": "no_successful_candidate",
                    },
                )
            return self._result(
                request,
                status="failed",
                error="평가 가능한 completed trial이 없습니다.",
                trials=trials,
                objective=objective,
                direction=direction,
                search_space=search_space,
                warnings=warnings,
                metadata={
                    "budget_used": budget_used,
                    "max_trials": max_trials,
                    "stop_reason": "no_scorable_trial",
                },
            )

        if baseline is None and min_delta > 0:
            return self._result(
                request,
                status="failed",
                error="min_delta 비교에 필요한 scorable baseline trial이 없습니다.",
                trials=trials,
                objective=objective,
                direction=direction,
                search_space=search_space,
                warnings=warnings,
                metadata={
                    "budget_used": budget_used,
                    "max_trials": max_trials,
                    "stop_reason": "missing_scorable_baseline",
                },
            )

        improved: bool | None = None
        best_is_baseline = False
        selected = best_candidate
        if baseline is not None:
            improvement = self._improvement(
                best_candidate.score,
                baseline.score,
                direction,
            )
            improved = improvement > 0 and (
                improvement > min_delta
                or math.isclose(improvement, min_delta, rel_tol=1e-12, abs_tol=1e-12)
            )
            if not improved:
                best_is_baseline = True
                selected = baseline

        return self._result(
            request,
            status="completed",
            best_config=self._trial_config(request, search_space, selected),
            best_score=selected.score,
            trials=trials,
            objective=objective,
            direction=direction,
            search_space=search_space,
            warnings=warnings,
            metadata={
                "best_trial_id": selected.trial_id,
                "best_is_baseline": best_is_baseline,
                "improved": improved,
                "min_delta": min_delta,
                "budget_used": budget_used,
                "max_trials": max_trials,
                "stop_reason": "search_budget_finished",
            },
        )

    def _trial_config(
        self,
        request: OptimizationRequest,
        search_space: dict[str, list[Any]],
        trial: InternalTrialResult,
    ) -> dict[str, Any]:
        if trial.is_baseline:
            return self._baseline_axis_config(request, search_space)
        return dict(trial.config)

    def _best_completed_trial(
        self,
        trials: list[InternalTrialResult],
        direction: ObjectiveDirection,
    ) -> InternalTrialResult | None:
        completed = [trial for trial in trials if self._is_completed(trial)]
        if not completed:
            return None
        if direction == "minimize":
            return min(completed, key=lambda trial: trial.score)  # type: ignore[arg-type]
        return max(completed, key=lambda trial: trial.score)  # type: ignore[arg-type]

    def _budget_used(self, trials: list[InternalTrialResult]) -> int:
        return len(
            {
                trial.fingerprint
                for trial in trials
                if not trial.is_baseline
                and bool(trial.metadata.get("in_search_space", False))
            }
        )

    def _has_passing_trial(self, trials: list[InternalTrialResult]) -> bool:
        return any(self._trial_passed(trial) for trial in trials)

    def _trial_passed(self, trial: InternalTrialResult) -> bool:
        return (
            trial.status in {"completed", "inconclusive"}
            and trial.metrics.get("pass_threshold") is True
        )

    def _is_completed(self, trial: InternalTrialResult) -> bool:
        return trial.status == "completed" and self._is_finite_number(trial.score)

    def _improvement(
        self,
        candidate: float | None,
        baseline: float | None,
        direction: ObjectiveDirection,
    ) -> float:
        if candidate is None or baseline is None:
            return float("-inf")
        if direction == "minimize":
            return baseline - candidate
        return candidate - baseline

    # metric/config 유틸 ----------------------------------------------------
    def _effective_config(
        self,
        request: OptimizationRequest,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        effective = deepcopy(self._study_baseline_config(request))
        for changes in (request.fixed_config, candidate):
            mapped, ignored, _warnings = map_changes_to_index_config(changes)
            if ignored:
                raise ValueError(
                    "state index_config로 변환할 수 없는 경로: "
                    + ", ".join(sorted(ignored))
                )
            effective.update(mapped)
        return effective

    def _study_baseline_config(
        self,
        request: OptimizationRequest,
    ) -> dict[str, Any]:
        """교차 회차에도 바뀌지 않는 study 최초 config를 반환한다."""

        configured = request.metadata.get("study_baseline_config")
        if configured is None:
            configured = request.baseline_config
        if not isinstance(configured, dict):
            raise TypeError("metadata.study_baseline_config는 dict여야 합니다.")
        baseline = deepcopy(configured)
        baseline.pop("_optimization", None)
        return baseline

    def _baseline_axis_config(
        self,
        request: OptimizationRequest,
        search_space: dict[str, list[Any]],
    ) -> dict[str, Any]:
        """현재 탐색 축의 study baseline 값을 canonical patch로 반환한다."""

        path = next(iter(search_space))
        sentinel = object()
        value = get_current_value(self._study_baseline_config(request), path, sentinel)
        if value is sentinel:
            return {}
        return {path: deepcopy(value)}

    def _normalize_metrics(self, raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        if self._is_finite_number(raw):
            return {"overall_score": float(raw)}

        if isinstance(raw, dict):
            metrics: dict[str, Any] = {}
            nested = raw.get("metrics")
            if isinstance(nested, dict):
                metrics.update(nested)
            ragas = raw.get("ragas_scores")
            if isinstance(ragas, dict):
                metrics.update(ragas)
            report = raw.get("report")
            if report is not None:
                metrics.update(self._normalize_metrics(report))
            for key, value in raw.items():
                if key in {"metrics", "ragas_scores", "report"}:
                    continue
                if self._is_finite_number(value) or isinstance(value, bool):
                    metrics[key] = float(value) if self._is_finite_number(value) else value
            if "overall_score" not in metrics and self._is_finite_number(raw.get("score")):
                metrics["overall_score"] = float(raw["score"])
            return metrics

        metrics = {}
        ragas_scores = getattr(raw, "ragas_scores", None)
        if isinstance(ragas_scores, dict):
            metrics.update(ragas_scores)
        overall = getattr(raw, "overall_score", None)
        if self._is_finite_number(overall):
            metrics["overall_score"] = float(overall)
        pass_threshold = getattr(raw, "pass_threshold", None)
        if isinstance(pass_threshold, bool):
            metrics["pass_threshold"] = pass_threshold
        return metrics

    def _extract_score(
        self,
        metrics: dict[str, Any],
        objective: str,
        *,
        allow_fallback: bool,
    ) -> tuple[float | None, str | None]:
        for metric in _OBJECTIVE_ALIASES.get(objective, (objective,)):
            value = metrics.get(metric)
            if self._is_finite_number(value):
                return float(value), metric
        if allow_fallback and objective != "overall_score":
            overall = metrics.get("overall_score")
            if self._is_finite_number(overall):
                return float(overall), "overall_score"
        return None, None

    def _fingerprint(self, config: dict[str, Any]) -> str:
        payload = json.dumps(
            config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _is_finite_number(self, value: Any) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        )

    def _append_warning(self, warnings: list[str], warning: str) -> None:
        if warning not in warnings:
            warnings.append(warning)

    # 결과 빌더 -------------------------------------------------------------
    def _result(
        self,
        request: OptimizationRequest,
        *,
        status: str,
        next_config: dict[str, Any] | None = None,
        best_config: dict[str, Any] | None = None,
        best_score: float | None = None,
        trials: list[InternalTrialResult] | None = None,
        objective: str = "overall_score",
        direction: ObjectiveDirection = "maximize",
        search_space: dict[str, list[Any]] | None = None,
        error: str | None = None,
        warnings: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> InternalAdapterResult:
        return InternalAdapterResult(
            request_id=request.request_id,
            status=status,  # type: ignore[arg-type]
            next_config=deepcopy(next_config),
            best_config=deepcopy(best_config),
            best_score=best_score,
            trial_results=list(trials or []),
            objective_metric=objective,
            direction=direction,
            search_space=deepcopy(search_space or {}),
            error=error,
            warnings=list(warnings or []),
            metadata=dict(metadata or {}),
        )


def run(
    request: OptimizationRequest,
    *,
    evaluator: TrialEvaluator | None = None,
) -> InternalAdapterResult:
    """모듈 공개 진입점. optimizer backend runner로 바로 주입할 수 있다."""

    return InternalAdapter(evaluator=evaluator).run(request)
