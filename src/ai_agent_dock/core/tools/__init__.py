"""Tool 계층: 기본 제공 Tool, 레지스트리(이름 중복 방지, 인자 검증, 결과 길이 제한), MCP 클라이언트(.mcp 하위 패키지).

MCP 클라이언트는 SDK를 가져오는 비용이 있어 여기서 import하지 않는다. 필요한 곳에서 `from ai_agent_dock.core.tools.mcp import ...`로 쓴다.
"""

from .builtin import BUILTIN_TOOLS, calculate, get_current_time
from .errors import ToolArgumentError, ToolError
from .registry import DEFAULT_MAX_RESULT_CHARS, ToolRegistry

__all__ = [
    "BUILTIN_TOOLS",
    "DEFAULT_MAX_RESULT_CHARS",
    "ToolArgumentError",
    "ToolError",
    "ToolRegistry",
    "calculate",
    "get_current_time",
]
