"""
agents/optimize/agent.py
Optimize 노드의 진입점(오케스트레이션 계층).

[역할]
  Eval 진단 리포트를 받아 planner → optimizer → config_mapper 를 순서대로 호출하고,
  선택된 처방을 실제 state.index_config 에 적용한다. 모든 경로에서 같은 state 를
  반환한다(AGENTS.md 2절 계약).

[읽는 것]  state.report, state.index_config, state.iteration, state.blacklist
[쓰는 것]  state.index_config, state.iteration, state.status, state.error,
           state.current_agent

[Phase 1 범위]  (현재)
  - 순방향 실행만: 최상위 처방 하나를 골라 적용하고 iteration 을 1 증가시킨다.
  - iteration 은 "실제 처방을 적용한" 경로에서만 정확히 한 번 증가한다.
    (제안만/현재 유지/수동 조치 경로는 반복 횟수를 소비하지 않는다.)
  - 롤백·블랙리스트, 이력 기록, 사용자 리포트 저장은 Phase 2 로 미룬다.
      · 판정(history.judge)은 처방 적용 후 '다음 Eval 재측정'이 있어야 가능하고,
      · 리포트를 담을 state 정식 필드는 팀 합의가 필요하다(AGENTS.md 11절).
"""
from __future__ import annotations

from core.state import AgentDoctorState
from agents.optimize import planner, optimizer, config_mapper


def run(state: AgentDoctorState) -> AgentDoctorState:
    """Optimize 노드 진입점. 성공·스킵·수동·오류 어느 경로든 같은 state 를 반환한다."""
    state.current_agent = "optimize"
    try:
        request, decision = planner.plan(state, blacklist=state.blacklist)

        # 자동 처방 경로가 아니면 config/iteration 을 건드리지 않는다.
        # (already_optimal / skipped / manual_required / propose_only)
        if decision.mode != "apply_optimize" or request is None:
            state.status = decision.status
            return state

        result = optimizer.run(request)

        # optimizer 가 적용 가능한 patch 를 만들지 못하면(skipped/failed) 그대로 종료.
        # 실제 적용이 없으므로 iteration 을 소비하지 않는다.
        if result.status != "proposed" or result.config_patch is None:
            state.status = result.status
            if result.status == "failed":
                state.error = result.error or result.message
            return state

        # 검증된 처방을 실제 index_config 에 반영한다. (canonical→flat 변환은 mapper 담당)
        config_mapper.apply_config_patch(state.index_config, result.config_patch)
        state.iteration += 1
        state.status = "applied"
        return state

    except Exception as exc:  # 예외를 밖으로 전파하지 않고 state 에 기록(AGENTS.md 2절)
        state.status = "error"
        state.error = f"optimize 실행 실패: {exc}"
        return state
