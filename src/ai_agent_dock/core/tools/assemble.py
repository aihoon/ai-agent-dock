"""Tool 조립과 결정(C17 Q20~Q23): 에이전트가 쓸 Tool은 여기서만 정한다.

pool(build_pool): 내장 Tool + mcp.json의 켜진 서버(connect_mcp, 시작 때 한 번 연결)
  → safety(with_safety, pool 전체에 한 번): deny는 남겨 두고 실행 때 막는다. ask와 경로는 실행 때 본다(Q23)
  → tools_for(에이전트와 노드마다): 에이전트 패턴 → 하위 에이전트면 ask_user·load_skill 빼기 → only로 좁히기 → 노드 규칙
delegate 붙이기(core/agents), 출처를 모으는 search_docs와 결과 모으기(workflows)는 빼기가 아니라 더하기라 각자 자리에 둔다.

의존 방향(순환 방지): 이 파일만 skills, retrieval, hitl을 import한다. 그 모듈들은 core/tools를 import하지 않는다(tests/test_assemble.py).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

from ...env import env_float, env_int
from ..hitl import make_ask_user_tool
from ..llm import Tool
from ..retrieval import Knowledge, make_search_docs_tool
from ..skills import SkillMeta, make_load_skill_tool
from .builtin import BUILTIN_TOOLS
from .registry import DEFAULT_MAX_RESULT_CHARS, ToolRegistry

DELEGATE = "delegate"  # 맡기기 Tool. 만드는 곳은 core/agents(맡길 에이전트를 알아야 한다)
HANDOFF = "handoff"  # 넘기기 Tool(C21). 만드는 곳은 core/agents. 세션의 대화 에이전트만 받는다
PARENT_ONLY = {"ask_user", "load_skill"}  # 되묻기와 스킬은 main만 쓴다. 하위 에이전트는 사용자를 볼 수 없다(C12).
NO_ASKING = {"ask_user", DELEGATE, HANDOFF}  # 워크플로 노드 안에서는 되묻지 않고(체크포인트에 남지 않는다) 다시 맡기지 않는다(C17 Q19).
WEB = "web__"
GLOB = "*?["  # 패턴 문자


def make_builtin_tools(
    *,
    skills: list[SkillMeta] | None = None,
    knowledge: Knowledge | None = None,
    input_fn: Callable[[str], str] = input,
) -> list[tuple[str, Tool]]:
    """(출처, Tool) 목록을 돌려준다. 출처는 /tools와 오류 메시지에 쓰는 이름이다.

    스킬이 없으면 load_skill을, 문서 색인이 열리지 않았으면 search_docs를 넣지 않는다.
    ask_user는 항상 넣는다. input_fn은 되묻기의 입력 함수다(테스트에서 바꿀 수 있다).
    """
    entries = [("builtin", tool) for tool in BUILTIN_TOOLS]
    entries.append(("builtin", make_ask_user_tool(input_fn)))  # 되묻기 Tool. 승인 대상이 아니다.
    if skills:
        entries.append(("skills", make_load_skill_tool(skills)))
    if knowledge:
        entries.append(("knowledge", make_search_docs_tool(knowledge.store, knowledge.seen, calls=knowledge.calls)))
    return entries


def connect_mcp(
    stack: ExitStack,
    mcp_config: str | Path | None,
    *,
    skip_servers: Iterable[str] = (),
    announce: Callable[[int], None] | None = None,
) -> dict[str, list[Tool]]:
    """mcp_config 파일의 켜진 서버에 연결해 서버 이름 → Tool 목록을 돌려준다. 파일이 없으면 {}(Q21).

    서버 프로세스를 띄우므로 시작 때 한 번만 부른다. 연결은 stack이 닫힐 때 정리된다.
    skip_servers는 빼고 연결한다(평가는 web을 뺀다, C12 Q13). announce(서버 수)는 연결 전에 부른다(처음 연결은 오래 걸린다).
    """
    if not mcp_config:
        return {}
    from .mcp import connect_mcp_servers, load_config  # MCP SDK는 설정했을 때만 가져온다.

    skip = set(skip_servers)
    servers = {name: spec for name, spec in load_config(mcp_config).items() if name not in skip}
    if announce:
        announce(len(servers))
    return stack.enter_context(
        connect_mcp_servers(servers, call_timeout=env_float("MCP_TOOL_TIMEOUT", 30.0), start_timeout=env_float("MCP_START_TIMEOUT", 60.0))
    )


def build_pool(
    *,
    skills: list[SkillMeta] | None = None,
    knowledge: Knowledge | None = None,
    mcp_tools: dict[str, list[Tool]] | None = None,
    input_fn: Callable[[str], str] = input,
) -> ToolRegistry:
    """설치된 Tool 전체(pool): 내장 Tool과 연결된 MCP 서버의 Tool을 레지스트리 하나에(Q21).

    평가는 케이스마다 문서 색인과 되묻기 답이 달라 케이스마다 다시 만든다. MCP 연결(connect_mcp)은 한 번만 하고 나눠 쓴다.
    """
    registry = ToolRegistry(max_result_chars=env_int("TOOL_RESULT_MAX_CHARS", DEFAULT_MAX_RESULT_CHARS))
    for source, tool in make_builtin_tools(skills=skills, knowledge=knowledge, input_fn=input_fn):
        registry.add(tool, source=source)
    for server, tools in (mcp_tools or {}).items():
        registry.add_all(tools, source=f"mcp:{server}")
    return registry


def matches(patterns: Iterable[str], name: str) -> bool:
    """Tool 이름이 패턴에 맞는지. 하나라도 포함 패턴에 맞고, 제외 패턴(!)에는 하나도 맞지 않아야 한다."""
    patterns = list(patterns)
    include = [p for p in patterns if not p.startswith("!")]
    exclude = [p[1:] for p in patterns if p.startswith("!")]
    return any(fnmatchcase(name, p) for p in include) and not any(fnmatchcase(name, p) for p in exclude)


def select_tools(patterns: Iterable[str], tools: list[Tool]) -> list[Tool]:
    patterns = list(patterns)
    return [t for t in tools if matches(patterns, t.name)]


@dataclass(frozen=True)
class Node:
    """워크플로 노드의 Tool 규칙(C17 Q22). 되묻기와 맡기기는 늘 뺀다."""

    branch: bool = False  # 동시에 도는 가지: 승인이 필요 없는 Tool만(C15 5.3). 두 가지가 동시에 사람에게 묻지 않게 한다.
    web: tuple[str, ...] = ()  # main은 웹 Tool을 빼지만(C14) 노드가 웹도 찾게 할 때 이 패턴(web 에이전트의 tools)에 맞는 pool의 웹 Tool을 더한다(C17 Q8-1, Q26).


def tools_for(
    pool: list[Tool],
    *,
    patterns: Iterable[str] = ("*",),
    depth: int = 0,
    only: Iterable[str] | None = None,
    node: Node | None = None,
) -> list[Tool]:
    """에이전트 하나(와 노드)가 쓸 Tool(Q22). pool은 safety를 거친 설치된 Tool 전체다.

    patterns: AGENT.md의 tools. depth: 0이면 main, 1 이상이면 하위 에이전트. only: 경로 표나 메뉴가 더 좁히는 패턴(교집합, C14 Q2-3).
    """
    tools = pool if depth == 0 else [t for t in pool if t.name not in PARENT_ONLY]
    tools = select_tools(patterns, tools)
    if only is not None:
        tools = select_tools(only, tools)
    if node is None:
        return tools
    tools = [t for t in tools if t.name not in NO_ASKING and (not node.branch or (t.read_only and t.trusted))]
    if node.web:
        names = {t.name for t in tools}
        safe = lambda t: not node.branch or (t.read_only and t.trusted)  # 동시 가지는 웹 Tool도 승인이 필요 없는 것만(C90 리뷰 A4)
        tools += [t for t in select_tools(node.web, pool) if t.name.startswith(WEB) and t.name not in names and safe(t)]
    return tools
