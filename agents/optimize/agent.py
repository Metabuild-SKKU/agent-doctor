"""
agents/optimize/agent.py
Optimize 노드의 진입점(오케스트레이션 계층).

[역할]
  Eval 진단 리포트를 받아 planner → optimizer → config_mapper → history 를 엮는다.
  각 방문은 2단계다:
    (1) 지난 방문에 적용한 처방을 판정(judge)한다. 나빠졌으면 config 를 되돌리고
        (label, prescription_id)를 blacklist 에 넣는다. (방문 간 롤백)
    (2) 새 처방 하나를 골라 적용하고 pending 이력을 남긴다.
  모든 경로에서 같은 state 를 반환한다(AGENTS.md 2절 계약).

[읽는 것]  state.report, state.index_config, state.iteration, state.max_iterations,
           state.blacklist, state.optimization_history
[쓰는 것]  state.index_config, state.iteration, state.status, state.error,
           state.current_agent, state.blacklist, state.optimization_history,
           state.optimization_report

[state.status 신호]  (graph 라우팅이 참고)
  - "applied"      : 새 처방을 적용함 → 재색인 필요(Index)
  - "rolled_back"  : 롤백으로 config 를 되돌림 → 재색인 필요(Index)
  - 그 외(skipped/manual_required/already_optimal/verified/error) → 변경 없음(Serve)

[사용자 리포트]  매 방문마다 state.optimization_report 에 이번 방문 결과를 번역해
  저장한다(reporter). 방문마다 덮어써지며 Serve 가 마지막 방문 리포트를 읽는다.
  리포트가 설명하는 처방과 점수가 어긋나지 않도록, '판정'은 이력 항목 기반
  trial 리포트로, '새 적용/수동/유지'는 decision 기반 리포트로 나눠 만든다.
"""
from __future__ import annotations

from core.state import AgentDoctorState
from agents.optimize import planner, optimizer, config_mapper, history, reporter
from agents.optimize.schemas import OptimizationHistoryItem, Verdict


def run(state: AgentDoctorState) -> AgentDoctorState:
    """Optimize 노드 진입점. 성공·스킵·수동·오류 어느 경로든 같은 state 를 반환한다."""
    state.current_agent = "optimize"
    try:
        # (1) 지난 처방 판정 + (나빴으면) 롤백/블랙리스트
        judged_item, verdict = _judge_pending_trial(state)
        rolled_back = judged_item is not None and not verdict.keep

        # (2) 반복 예산 소진: 판정만 하고 신규 처방은 시도하지 않는다.
        #     (롤백했으면 재색인이 필요하므로 rolled_back 신호를 남긴다.)
        if state.iteration >= state.max_iterations:
            state.status = "rolled_back" if rolled_back else "verified"
            if judged_item is not None:
                # 마지막 방문의 headline = 방금 내린 판정(유지/롤백).
                state.optimization_report = reporter.build_trial_report(judged_item, verdict)
            return state

        # (3) 새 처방 선택
        request, decision = planner.plan(state, blacklist=state.blacklist)
        if decision.mode != "apply_optimize" or request is None:
            state.status = "rolled_back" if rolled_back else decision.status
            # 롤백이 있었으면 그게 headline, 아니면 흐름 결정(수동/유지)을 보고.
            state.optimization_report = (
                reporter.build_trial_report(judged_item, verdict)
                if rolled_back
                else reporter.build_report(decision, request)
            )
            return state

        result = optimizer.run(request)
        if result.status != "proposed" or result.config_patch is None:
            # optimizer 가 적용 가능한 patch 를 못 만듦(skipped/failed)
            state.status = "rolled_back" if rolled_back else result.status
            if result.status == "failed":
                state.error = result.error or result.message
            # 롤백만 리포트로 남긴다. 적용 실패 자체는 status/error 로 전달(MVP 한계).
            if rolled_back:
                state.optimization_report = reporter.build_trial_report(judged_item, verdict)
            return state

        # 적용 직전 스냅샷(롤백이 있었다면 이미 되돌려진 config 가 before 가 된다)
        before_config = dict(state.index_config)
        before_report = state.report

        # 검증된 처방을 실제 index_config 에 반영(canonical→flat 변환은 mapper 담당)
        config_mapper.apply_config_patch(state.index_config, result.config_patch)

        # pending 이력 생성(다음 방문에서 finalize) + iteration 1회 증가
        prescription_id = (
            result.selected_candidate.id if result.selected_candidate else None
        )
        item = history.create_pending_item(
            state, request, prescription_id, before_config, before_report
        )
        state.optimization_history.append(item)
        state.iteration += 1
        state.status = "applied"
        # 새 처방을 적용함 → "적용, 다음 검증 대기" 리포트(verdict 없음).
        state.optimization_report = reporter.build_report(decision, request)
        return state

    except Exception as exc:  # 예외를 밖으로 전파하지 않고 state 에 기록(AGENTS.md 2절)
        state.status = "error"
        state.error = f"optimize 실행 실패: {exc}"
        return state


def _judge_pending_trial(
    state: AgentDoctorState,
) -> tuple[OptimizationHistoryItem | None, Verdict | None]:
    """직전에 적용한 처방(pending)을 판정한다. 나빴으면 config 롤백 + blacklist.
    판정한 이력 항목과 Verdict 를 반환한다(판정할 게 없으면 (None, None)).
    호출부가 이 둘로 롤백 여부(not verdict.keep) 판단 + 사용자 리포트를 만든다."""
    pending = history.find_pending(state.optimization_history)
    if pending is None:
        return None, None

    before_report = pending.metadata.get("before_report")
    after_report = state.report

    if before_report is None or after_report is None:
        # 비교할 점수가 없어 판정 불가 → 보수적으로 유지 처리하고 확정만.
        verdict = Verdict(
            keep=True, before_score=0.0, after_score=0.0,
            reason="판정 불가(리포트 없음) — 유지",
        )
    else:
        verdict = history.judge(before_report, after_report)

    # 롤백 전의 '실제 적용되어 측정된' config 를 이력에 남긴다.
    after_config = dict(state.index_config)

    if not verdict.keep:
        restored = dict(pending.before_config)  # config 되돌리기(롤백)
        # 임베딩 모델을 바꿨던 처방을 되돌릴 때는 컬렉션 차원도 원래대로
        # 재생성해야 한다. before_config의 플래그는 적용 전 스냅샷(False)이라
        # 그대로 복원하면 영구 Qdrant에서 dimension mismatch로 죽는다.
        if restored.get("embedding_model") != after_config.get("embedding_model"):
            restored["recreate_collection_on_dimension_mismatch"] = True
        state.index_config = restored
        label = pending.failure_labels[0] if pending.failure_labels else ""
        if label and pending.selected_prescription_id:
            state.blacklist.add((label, pending.selected_prescription_id))

    history.finalize_item(pending, verdict, after_config, after_report)
    return pending, verdict
