"""
serve_module.py
Serve 모듈 — 색인된 문서를 MCP 서버로 감싸 외부 AI(Claude 등)에 검색을 제공.

사용 방식 두 가지:
  1. LangGraph 노드:  run(state) -> state
       파이프라인에서 호출된다. LangGraph는 노드를 run(state) 형태로 호출하고
       state를 돌려받으므로(graph.py 의 add_node/add_edge 참고), Serve 함수는
       RAG 객체 존재 여부와 무관하게 항상 state를 받고 state를 반환해야 한다.
  2. 독립 실행:       ServeModule(client).start()
       Claude Desktop 등이 이 파일을 stdio 프로세스로 직접 띄울 때.

NOTE(임시): 최종 선택된 RAG 객체(rag.invoke(question))는 아직 없다.
            그 객체가 생기기 전까지는 실재하는 qdrant_store.search()로 직접 검색한다.
            나중에 RAG 객체가 만들어지면 ServeModule._answer() 만 교체하면 된다.
"""
from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from agents.index import qdrant_store
from core.state import AgentDoctorState

SERVER_NAME = "agent-doctor-rag-server"
TOP_K = int(os.getenv("AGENT_DOCTOR_TOP_K", "3"))


class ServeModule:
    def __init__(self, client, server_name: str = SERVER_NAME, top_k: int = TOP_K,
                 vector_dim: int | None = None):
        """
        Parameters
        ----------
        client : 이미 색인된 Qdrant client (qdrant_store.build_client 결과).
                 최종 RAG 객체가 생기기 전까지 검색은 이 client 로 직접 수행한다.
        server_name : MCP 서버 이름.
        top_k : ask_docs 가 돌려줄 검색 결과 개수.
        vector_dim : 색인에 쓰인 임베딩 차원. 질의 임베딩 차원을 맞추는 데 쓴다(없으면 기본값).
        """
        self.client = client
        self.top_k = top_k
        self.vector_dim = vector_dim
        self.mcp = FastMCP(server_name)   # server_name 을 가진 MCP 서버 생성

    def _answer(self, question: str) -> str:
        """질문 → Qdrant 검색 → 사람이 읽을 수 있는 답변 문자열.

        (추후 최종 RAG 객체가 생기면 이 메서드 내부를 rag.invoke(question) 으로 교체)
        """
        query_vec = qdrant_store.embed(question, vector_dim=self.vector_dim)
        results = qdrant_store.search(self.client, query_vec, top_k=self.top_k)
        if not results:
            return "관련 문서를 찾지 못했습니다."

        parts = []
        for i, r in enumerate(results, 1):
            title = r.get("metadata", {}).get("title", "")
            source = f"\n출처: {title}" if title else ""
            parts.append(f"[{i}] {r.get('text', '')}{source}")
        return "\n\n".join(parts)

    def create_tools(self) -> None:
        """MCP Tool 등록."""

        @self.mcp.tool()
        def ask_docs(question: str) -> str:
            """질문을 받아 문서를 검색하고, 검색 결과를 반환."""
            try:
                print(f"[Serve] 질문 수신: {question}")   # 로깅

                answer = self._answer(question)

                print("[Serve] 답변 생성 완료")
                return answer

            except Exception as e:
                # 검색 내부 오류를 사용자에게 그대로 노출하지 않고, 원인은 터미널에만 남긴다.
                print(f"[Serve] 질문 처리 실패 '{question}': {e}")
                return "질문 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."

    def start(self) -> None:
        """MCP 서버(stdio) 실행. Claude Desktop 등이 프로세스로 띄울 때 사용 — 블로킹."""
        self.create_tools()   # 내부 함수들을 MCP tool 로 등록
        self.mcp.run()        # MCP 서버 실행 → 외부 AI 연결 가능


def _build_index(state: AgentDoctorState):
    """state.chunks 를 Qdrant 에 올리고 (client, vector_dim) 을 돌려준다.

    임베딩된 청크가 하나도 없으면 (None, None).
    """
    embedded = [c for c in state.chunks if getattr(c, "embedding", None)]
    if not embedded:
        return None, None

    vector_dim = len(embedded[0].embedding)
    client = qdrant_store.build_client(
        url=os.getenv("QDRANT_URL", ":memory:"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    qdrant_store.ensure_collection(client, vector_dim=vector_dim, recreate_on_mismatch=True)
    qdrant_store.upsert_chunks(client, embedded)
    return client, vector_dim


def run(state: AgentDoctorState) -> AgentDoctorState:
    """
    Serve Agent 진입점 — LangGraph 노드.

    LangGraph 가 노드를 run(state) 로 호출하고 state 를 돌려받는 구조이므로,
    RAG 객체가 있든 없든 시그니처는 항상 state -> state 여야 한다.

    읽기: state.chunks
    쓰기: state.mcp_endpoint, state.status, state.error
    """
    state.current_agent = "serve"
    print(f"[Serve] 청크 {len(state.chunks)}개 처리 중")

    if not state.chunks:
        state.status = "error"
        state.error = "청크가 없습니다. Index Agent가 완료됐는지 확인하세요."
        return state

    try:
        client, vector_dim = _build_index(state)
        if client is None:
            state.status = "error"
            state.error = "임베딩된 청크가 없습니다. Index Agent 임베딩 단계를 확인하세요."
            return state

        # MCP 서버 구성(툴 등록)까지만 하고, 블로킹 실행(start)은 하지 않는다.
        # 실제 stdio 서버는 Claude Desktop 등이 별도 프로세스로 띄운다.
        module = ServeModule(client, vector_dim=vector_dim)
        module.create_tools()

        state.mcp_endpoint = f"stdio://{SERVER_NAME}"
        state.status = "done"
        print(f"[Serve] MCP 서버 준비 완료 → {state.mcp_endpoint}")

    except Exception as e:
        state.status = "error"
        state.error = f"Serve 실패: {e}"
        print(f"[Serve] 오류: {e}")

    return state


if __name__ == "__main__":
    # 독립 실행: 이미 색인된 Qdrant(QDRANT_URL)에 붙어 MCP stdio 서버를 띄운다.
    _client = qdrant_store.build_client(
        url=os.getenv("QDRANT_URL", ":memory:"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    ServeModule(_client).start()
