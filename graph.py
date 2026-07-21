"""
graph.py
Agent Doctor v2 메인 LangGraph 그래프

팀원별 담당:
  [Ingest Agent]   → agents/ingest/agent.py
  [Index Agent]    → agents/index/agent.py
  [Eval Agent]     → agents/eval/agent.py 
  [Optimize Agent] → agents/optimize/agent.py
  [Serve Agent]    → agents/serve/agent.py

각 팀원은 agents/agent.py의 run(state) 함수 구현
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from core.state import AgentDoctorState
from agents.ingest.agent import run as ingest_run
from agents.index.agent import run as index_run
from agents.eval.agent import run as eval_run
from agents.optimize.agent import run as optimize_run
from agents.optimize import history   # 판정 대기(pending) 처방 조회용
from agents.optimize import gate      # serve/optimize 게이트 정책(점수 + 검색 바닥선)
from agents.serve.agent import run as serve_run


def route_after_eval(state: AgentDoctorState) -> str:
    """
    Eval 결과를 보고 다음 에이전트 결정.
      품질 통과                       → Serve
      반복 예산 소진 + 판정 대기 처방  → Optimize (마지막 처방 판정 기회)
      반복 예산 소진 + 대기 없음       → Serve
      품질 미달(예산 남음)            → Optimize
    """
    if gate.passes_report(state.report):
        print(f"[Orchestrator] 품질 통과 ({state.report.overall_score}점) → Serve")
        return "serve"

    if state.iteration >= state.max_iterations:
        # 예산은 다 썼지만 마지막 처방이 아직 판정(유지/롤백) 안 됐으면 한 번 더 보낸다.
        if history.find_pending(state.optimization_history):
            print("[Orchestrator] 반복 예산 소진 — 마지막 처방 판정 위해 → Optimize")
            return "optimize"
        print(f"[Orchestrator] 최대 반복 {state.max_iterations}회 도달 → Serve")
        return "serve"

    print(f"[Orchestrator] 품질 미달 → Optimize (반복 {state.iteration}/{state.max_iterations})")
    return "optimize"


def route_after_optimize(state: AgentDoctorState) -> str:
    """
    Optimize 결과를 보고 다음 흐름 결정.
      config 변경(새 처방 적용 또는 롤백) → Index (재색인 후 재검증)
      변경 없음(제안/유지/수동/스킵)      → Serve
    """
    if state.status in ("applied", "rolled_back"):
        print(f"[Orchestrator] config 변경({state.status}) → Index 재색인")
        return "index"
    print(f"[Orchestrator] config 변경 없음({state.status}) → Serve")
    return "serve"


def build_graph():
    graph = StateGraph(AgentDoctorState)

    graph.add_node("ingest",   ingest_run)
    graph.add_node("index",    index_run)
    graph.add_node("eval",     eval_run)
    graph.add_node("optimize", optimize_run)
    graph.add_node("serve",    serve_run)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest",   "index")
    graph.add_edge("index",    "eval")
    graph.add_edge("serve",    END)

    # Optimize 후: config가 바뀌었으면 Index 재색인, 아니면 바로 Serve.
    graph.add_conditional_edges(
        "optimize",
        route_after_optimize,
        {"index": "index", "serve": "serve"},
    )

    graph.add_conditional_edges(
        "eval",
        route_after_eval,
        {"optimize": "optimize", "serve": "serve"},
    )

    return graph.compile()


def run_pipeline(
    source_url: str,
    source_type: str = "notion",
    user_questions: list[str] = None,
) -> AgentDoctorState:
    """
    Agent Doctor v2 파이프라인 실행.

    Args:
        source_url:     데이터 소스 URL
        source_type:    "notion" | "gdrive" | "file" | "slack"
        user_questions: 테스트 질문 (없으면 자동 생성)
    """
    graph = build_graph()

    initial_state = AgentDoctorState(
        source_url=source_url,
        source_type=source_type,
        user_questions=user_questions or [],
        status="running",
    )

    print("=" * 60)
    print("Agent Doctor v2 시작")
    print(f"  소스: {source_url} ({source_type})")
    print("=" * 60)

    final_state = graph.invoke(initial_state)

    print("=" * 60)
    print("완료")
    if final_state.mcp_endpoint:
        print(f"  MCP 엔드포인트: {final_state.mcp_endpoint}")
    if final_state.report:
        print(f"  최종 품질 점수: {final_state.report.overall_score}")
    print("=" * 60)

    return final_state


if __name__ == "__main__":
    run_pipeline(
        source_url="https://notion.so/example-page",
        source_type="notion",
        user_questions=["휴가 정책이 어떻게 돼?", "신입사원 온보딩 기간은?"],
    )
