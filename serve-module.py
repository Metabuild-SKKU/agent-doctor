from mcp.server.fastmcp import FastMCP

class ServeModule:
def __init__(self, rag, server_name="agent-doctor-rag-server"):
"""
Parameters
----------
self: 현재 만들어지고 있는 servemodule 자체, servemodule에 저장해야 할 것들을 저장해두는 용도
rag : 최종 선택된 RAG 객체(최적화 완료한)
server_name: mcp server 이름 정하기
invoke(question)를 지원한다고 가정
"""
self.rag = rag    #server.rag = optimized_rag랑 같은 의미
self.mcp = FastMCP(server_name) #server_name을 가진 MCP server를 만들어서 self.mcp에 저장

def create_tools(self):
    """
    MCP Tool 등록
    """

    @self.mcp.tool()
def ask_docs(question: str) -> str:
    """
    질문을 받아서 RAG에 전달하고,
    RAG에서 준 결과를 반환
    """

    try:
        print(f"[INFO] Question received: {question}") #로깅

        answer = self.rag.invoke(question)

        print(f"[INFO] Answer generated successfully")

        return answer

    except Exception as e: #예외 처리
                           # 서버는 RAG 내 구체적인 구조를 모르기 때문에 오류를 처리할 수                              는 없음(문제 발생 시 오류를 알리는 게 유일한 것)
        print(
            f"[ERROR] Failed to process question " #오류 원인은 터미널에만 띄워서 사용자에게 내부 정보 노출 막음
            f"'{question}': {e}"
        )

        return (
            "질문 처리 중 오류가 발생했습니다. " #사용자가 보는 화면
            "잠시 후 다시 시도해주세요."
        )

def run(self): #serve 기능 시작 시점
    """
    MCP 서버 실행
    """
    self.create_tools() #create_tools()가 실행하면서 그 안의 함수들이 mcp tool들로 들어감
    self.mcp.run() #mcp server 실행 -> 외부 AI 연결 가능