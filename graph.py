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

import os

# .env 를 '다른 모든 import 보다 먼저' 로드한다 — 아래 모듈들이 import 시점에 읽는 최상위 env
# (metrics_ragas.EVAL_RELEVANCY_STRICTNESS, serve.AGENT_DOCTOR_API_* 등)까지 .env 값이 반영되도록.
# override=True: 셸/OS 에 이미 있는 값보다 .env 를 우선(수정한 .env 가 항상 반영되게).
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

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


def _langsmith_run_config(state: AgentDoctorState) -> dict:
    """이번 실행의 index_config 를 LangSmith tags/metadata 로 만든다(2단계: 실행 간 비교).

    graph.invoke(state, config=...) 에 넘기면 이 실행에서 생성되는 모든 하위 트레이스
    (gemini_chat 등)에 자동 전파된다 → UI 에서 "reranker=off 태그만 필터" 같은 before/after
    비교가 된다. 트레이싱이 꺼져 있으면 이 config 는 langsmith 가 그냥 무시하므로 무해하다.

    태그는 파이프라인 시작 시점의 config 기준이다 — Optimize 가 실행 중 config 를 바꿔도
    태그는 초기값을 유지한다. before/after 비교는 use_reranker 등을 바꿔 '따로 실행'하는
    방식을 전제로 하므로 이 정도로 충분하다(각 실행은 처음부터 끝까지 config 고정)."""
    cfg = state.index_config
    # 필터에 쓰기 좋은 핵심 축만 태그로. 값은 소문자 문자열로 정규화.
    tag_keys = ("use_reranker", "use_hybrid", "chunk_strategy", "chunk_size", "top_k")
    short = {"use_reranker": "reranker", "use_hybrid": "hybrid",
             "chunk_strategy": "strategy", "chunk_size": "chunk", "top_k": "topk"}
    tags = [f"{short[k]}={str(cfg.get(k)).lower()}" for k in tag_keys if k in cfg]
    return {
        "tags": tags,
        "metadata": {
            "source_type": state.source_type,
            "index_config": cfg,  # config 전체 — 상세에서 정확한 값 확인용
        },
    }


def run_pipeline(
    source_url: str,
    source_type: str = "notion",
    user_questions: list[str] = None,
    index_config_overrides: dict = None,
) -> AgentDoctorState:
    """
    Agent Doctor v2 파이프라인 실행.

    Args:
        source_url:     데이터 소스 URL
        source_type:    "notion" | "gdrive" | "file" | "slack"
        user_questions: 테스트 질문 (없으면 자동 생성)
        index_config_overrides: 초기 index_config 에 덮어쓸 키(예: {"use_reranker": True}).
            2단계 before/after 비교용 — 같은 소스를 config 만 바꿔 따로 실행할 때 쓴다.
    """
    from core.run_logger import setup_run_logging
    setup_run_logging(prefix="pipeline")  # 이후 모든 print 를 콘솔+로그파일에 동시 출력

    graph = build_graph()

    initial_state = AgentDoctorState(
        source_url=source_url,
        source_type=source_type,
        user_questions=user_questions or [],
        status="running",
    )
    if index_config_overrides:
        initial_state.index_config.update(index_config_overrides)

    print("=" * 60)
    print("Agent Doctor v2 시작")
    print(f"  소스: {source_url} ({source_type})")
    # LangSmith 트레이싱 상태 안내(선택). .env 의 LANGSMITH_TRACING 이 이미 로드돼 있으면
    # langsmith SDK 가 자동으로 트레이스를 전송한다 — 여기선 켜졌는지만 알린다.
    if os.getenv("LANGSMITH_TRACING", "").strip().lower() in ("1", "true", "yes"):
        project = os.getenv("LANGSMITH_PROJECT", "(default)")
        print(f"  LangSmith 트레이싱: ON (project={project})")
    print("=" * 60)

    # config=... : 이 실행의 index_config 를 tags/metadata 로 하위 트레이스에 전파(2단계).
    # 트레이싱 OFF 여도 LangGraph 는 이 config 를 정상 처리하므로(langsmith 만 무시) 무해.
    final_state = graph.invoke(initial_state, config=_langsmith_run_config(initial_state))
    # LangGraph 는 dataclass state 를 dict 로 반환한다 → 속성 접근 위해 dataclass 로 복원
    # (노드 안에서는 AgentDoctorState 객체지만 invoke() 최종 반환은 dict).
    if isinstance(final_state, dict):
        final_state = AgentDoctorState(**final_state)

    print("=" * 60)
    print("완료")
    if final_state.mcp_endpoint:
        print(f"  MCP 엔드포인트: {final_state.mcp_endpoint}")
    if final_state.report:
        print(f"  최종 품질 점수: {final_state.report.overall_score}")
    print("=" * 60)

    return final_state


if __name__ == "__main__":
    # 소스는 env 로 받는다(run_local_pipeline.py 와 동일 계약: SOURCE_TYPE / SOURCE_URL).
    #   SOURCE_TYPE=file    SOURCE_URL=sample_docs/hr_policy.md python graph.py   # 기본
    #   SOURCE_TYPE=korquad SOURCE_URL=data/corpus.jsonl python graph.py
    #   SOURCE_TYPE=notion  SOURCE_URL=https://notion.so/... python graph.py
    # run_local_pipeline.py 와 달리 Serve 까지 띄우고, 품질 미달 시 재색인·재평가
    # 루프(route_after_eval/optimize)를 예산 소진까지 반복한다.
    # 기본값은 레포에 포함된 sample_docs/hr_policy.md 로 둔다 — korquad 는 data/ 파일을
    # 별도 준비해야 하므로(gitignore), 클론 직후 바로 실행되려면 file 이어야 한다.
    source_type = os.getenv("SOURCE_TYPE", "file").strip().lower()
    defaults = {
        "korquad": "data/corpus.jsonl",
        "file": "sample_docs/hr_policy.md",
        "notion": "https://notion.so/example-page",
    }
    source_url = os.getenv("SOURCE_URL", defaults.get(source_type, ""))
    user_questions: list[str] = []

    if source_type == "korquad":
        os.environ.setdefault("EVAL_PROBE_SOURCE", "taxonomy")  # qa 를 taxonomy 로
        # 규모 제한(스모크). .env·shell 값이 있으면 그쪽이 우선(setdefault).
        # 전체를 쓰려면 두 값을 0/미설정으로.
        os.environ.setdefault("KORQUAD_MAX_DOCS", "20")
        os.environ.setdefault("KORQUAD_QA_LIMIT", "50")
    elif source_type == "file":
        user_questions = [
            "재택근무 며칠까지 가능해?",
            "연차는 며칠이야?",
            "성과급은 언제 나와?",
        ]

    # USE_RERANKER=1 python graph.py → reranker ON 으로 실행(2단계 before/after 비교용).
    #   미설정/0 이면 기본값(OFF). 같은 소스를 이 env 만 바꿔 두 번 돌리면 LangSmith 에
    #   'reranker=false' / 'reranker=true' 태그가 붙은 LangGraph 루트 트레이스 2개가 쌓인다.
    overrides: dict = {}
    if os.getenv("USE_RERANKER", "").strip().lower() in ("1", "true", "yes"):
        overrides["use_reranker"] = True

    run_pipeline(
        source_url,
        source_type=source_type,
        user_questions=user_questions,
        index_config_overrides=overrides,
    )
