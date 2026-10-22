"""에이전트(C12): agents/<이름>/AGENT.md로 에이전트를 정의하고, 메인 에이전트가 하위 에이전트에게 일을 맡긴다(delegate)."""

from .loader import AGENT_INDEX_HEAD, MAIN, build_agent_index, load_agents, select_tools
from .system_prompt import build_system_prompt
from .team import DELEGATE, MAX_DEPTH, AgentTeam
from .types import AgentSpec, AgentFormatError

__all__ = [
    "AGENT_INDEX_HEAD",
    "DELEGATE",
    "MAIN",
    "MAX_DEPTH",
    "AgentSpec",
    "AgentFormatError",
    "AgentTeam",
    "build_agent_index",
    "build_system_prompt",
    "load_agents",
    "select_tools",
]
