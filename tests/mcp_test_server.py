"""MCP 연결 테스트용 서버. 플랫폼의 기능이 아니라 MCP 클라이언트(core/tools/mcp)를 시험하기 위한 접속 대상이다.

테스트와 `chat_walkthrough.py --scenario mcp`가 stdio 자식 프로세스로 띄운다. 오류, 지연, 긴 결과, 프로세스 정리를 시험하는 Tool이 있다.

환경 변수 TEST_PIDFILE이 있으면 시작할 때 자기 프로세스 번호를 그 파일에 적는다(종료 후 프로세스가 남지 않았는지 확인용).
"""

import os
import time

from mcp.server import MCPServer

if os.environ.get("TEST_PIDFILE"):
    with open(os.environ["TEST_PIDFILE"], "w") as f:
        f.write(str(os.getpid()))

server = MCPServer("test")


@server.tool()
def echo(text: str) -> str:
    """받은 문자열을 그대로 돌려준다."""
    return text


@server.tool()
def fail() -> str:
    """항상 실패한다(Tool 실행 오류 시험)."""
    raise RuntimeError("의도된 실패")


@server.tool()
def sleep(seconds: float) -> str:
    """지정한 시간 동안 멈춘다(시간 제한 시험)."""
    time.sleep(seconds)
    return "깼다"


@server.tool()
def long_text(chars: int) -> str:
    """지정한 글자 수의 긴 문자열을 돌려준다(결과 길이 제한 시험)."""
    return "가" * chars


@server.tool(name="dotted.name")
def dotted() -> str:
    """이름에 점이 든 Tool(이름 변환 시험)."""
    return "점"


if __name__ == "__main__":
    server.run()
