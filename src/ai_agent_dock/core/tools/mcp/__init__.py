"""MCP 클라이언트: 설정 파일에 적은 MCP 서버에 접속해 그 서버의 Tool을 우리 Tool 형식으로 가져온다.

에이전트 루프는 MCP를 모른다. 여기서 만든 Tool의 func가 MCP 클라이언트를 통해 서버를 부를 뿐이다.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from ...llm import Tool
from .config import ServerSpec, load_config
from .connection import McpConnection
from .convert import model_tool_name, result_to_text

__all__ = ["ServerSpec", "connect_mcp_servers", "load_config"]


@contextmanager
def connect_mcp_servers(
    servers: dict[str, ServerSpec],
    *,
    call_timeout: float,
    start_timeout: float,
    warn: Callable[[str], None] = print,
) -> Iterator[dict[str, list[Tool]]]:
    """서버마다 연결하고 서버 이름 → Tool 목록을 내준다. with를 벗어나면 모든 연결과 서버 프로세스를 정리한다.

    연결에 실패한 서버는 경고를 내고 건너뛴다(나머지 서버로 계속한다).
    """
    connections: list[McpConnection] = []
    tools: dict[str, list[Tool]] = {}
    try:
        for name, spec in servers.items():
            connection = McpConnection(name, spec, call_timeout=call_timeout)
            try:
                definitions = connection.start(start_timeout)
            except Exception as e:
                command = " ".join([spec.command, *spec.args])
                warn(f"MCP 서버 {name!r}({command}) 연결 실패, 건너뜁니다: {type(e).__name__}: {e}")
                continue
            connections.append(connection)
            tools[name] = [_to_tool(connection, definition) for definition in definitions]
        yield tools
    finally:
        for connection in connections:
            connection.close()


def _to_tool(connection: McpConnection, definition: Any) -> Tool:
    def func(**arguments: Any) -> str:
        return result_to_text(connection.call_tool(definition.name, arguments))

    # 읽기 전용 힌트와 서버 신뢰를 각각 옮긴다. 자동 실행 판단은 루프가 두 값을 함께 본다.
    annotations = getattr(definition, "annotations", None)
    return Tool(
        name=model_tool_name(connection.name, definition.name),
        description=definition.description or definition.title or "",
        parameters=definition.input_schema,
        func=func,
        read_only=bool(getattr(annotations, "read_only_hint", False)),
        trusted=connection.spec.trusted,
    )
