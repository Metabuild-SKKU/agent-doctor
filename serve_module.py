from mcp.server.fastmcp import FastMCP


class ServeModule:
    def __init__(self, rag, server_name="agent-doctor-rag-server"):
        """
        Parameters
        ----------
        self: 현재 만들어지고 있는 ServeModule 자체. ServeModule에 저장해야 할 것들을 담아두는 용도
        rag : 최종 선택된 RAG 객체(최적화 완료한). invoke(question)를 지원한다고 가정
        server_name: MCP server 이름
        """
        self.rag = rag                    # server.rag = optimized_rag 와 같은 의미
        self.mcp = FastMCP(server_name)   # server_name 을 가진 MCP server 를 만들어 self.mcp 에 저장

    def create_tools(self):
        """MCP Tool 등록"""

        @self.mcp.tool()
        def ask_docs(question: str) -> str:
            """질문을 받아서 RAG에 전달하고, RAG에서 준 결과를 반환"""
            try:
                print(f"[INFO] Question received: {question}")   # 로깅

                answer = self.rag.invoke(question)

                print("[INFO] Answer generated successfully")

                return answer

            except Exception as e:
                # 서버는 RAG 내부 구조를 모르므로 오류를 처리할 수는 없다(문제 발생을 알리는 게 유일).
                # 오류 원인은 터미널에만 출력해 사용자에게 내부 정보 노출을 막는다.
                print(f"[ERROR] Failed to process question '{question}': {e}")

                return "질문 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."   # 사용자가 보는 화면

    def run(self):
        """MCP 서버 실행"""
        self.create_tools()   # create_tools() 실행 시 내부 함수들이 MCP tool 로 등록됨
        self.mcp.run()        # MCP server 실행 → 외부 AI 연결 가능
