"""Tool 조립과 결정 테스트(C5, C17 Q20~Q23): 내장 Tool 목록, pool, 에이전트와 노드의 Tool, 의존 방향(순환 방지)."""

import re
from contextlib import ExitStack
from pathlib import Path

from ai_agent_dock.core.retrieval import Knowledge
from ai_agent_dock.core.skills import SkillMeta
from ai_agent_dock.core.llm import Tool
from ai_agent_dock.core.tools import Node, ToolRegistry, build_pool, connect_mcp, make_builtin_tools, tools_for

CORE = Path(__file__).resolve().parent.parent / "src" / "ai_agent_dock" / "core"
SKILL = SkillMeta(name="요약", description="글을 짧게 줄인다", path=Path("요약/SKILL.md"))


def pairs(entries):
    return [(source, tool.name) for source, tool in entries]


def test_builtin_only_when_there_are_no_skills_and_no_index():
    assert pairs(make_builtin_tools()) == [
        ("builtin", "get_current_time"),
        ("builtin", "calculate"),
        ("builtin", "ask_user"),
    ]


def test_skill_and_docs_tools_come_with_their_sources():
    knowledge = Knowledge(store=object(), seen=set(), files=1, chunks=1, calls=[])

    result = pairs(make_builtin_tools(skills=[SKILL], knowledge=knowledge))

    assert ("skills", "load_skill") in result
    assert ("knowledge", "search_docs") in result


def test_all_entries_register_without_a_name_clash():
    knowledge = Knowledge(store=object(), seen=set(), files=1, chunks=1, calls=[])
    registry = ToolRegistry()

    for source, tool in make_builtin_tools(skills=[SKILL], knowledge=knowledge):
        registry.add(tool, source=source)

    assert len(registry) == 5


def test_only_assemble_imports_the_feature_modules():
    """의존 방향: core/tools 안에서는 assemble만 기능 모듈(skills, retrieval, hitl)을 import하고, 기능 모듈과 llm은 core/tools를 import하지 않는다."""
    feature = re.compile(r"from \.\.(?:\.)?(skills|retrieval|hitl)\b|import ai_agent_dock\.core\.(skills|retrieval|hitl)")
    for path in (CORE / "tools").rglob("*.py"):
        if path.name != "assemble.py":
            assert not feature.search(path.read_text(encoding="utf-8")), path
    for folder in ["skills", "retrieval", "hitl", "llm"]:
        for path in (CORE / folder).rglob("*.py"):
            assert not re.search(r"from \.\.tools\b|core\.tools\b|\btoolkit\b", path.read_text(encoding="utf-8")), path


def test_build_pool_without_mcp_config_has_only_builtin_tools():
    with ExitStack() as stack:
        assert connect_mcp(stack, None) == {}
    pool = build_pool(skills=[SKILL])
    assert [t.name for t in pool.tools()] == ["get_current_time", "calculate", "ask_user", "load_skill"]
    assert pool.source_of("load_skill") == "skills"


def tool(name, read_only=True, trusted=True):
    return Tool(name=name, description=name, parameters={"type": "object"}, func=lambda **kw: "ok", read_only=read_only, trusted=trusted)


POOL = [tool("calculate"), tool("ask_user"), tool("load_skill"), tool("search_docs"), tool("files__write_file", read_only=False), tool("delegate"), tool("web__tavily_search")]


def names(tools):
    return [t.name for t in tools]


def test_tools_for_picks_by_agent_patterns_and_sub_agents_lose_asking_and_skills():  # Q22
    assert names(tools_for(POOL, patterns=["*", "!web__*"])) == ["calculate", "ask_user", "load_skill", "search_docs", "files__write_file", "delegate"]
    assert names(tools_for(POOL, patterns=["*", "!web__*"], depth=1)) == ["calculate", "search_docs", "files__write_file", "delegate"]
    assert names(tools_for(POOL, patterns=["*"], only=["search_docs", "calculate"])) == ["calculate", "search_docs"]  # 교집합
    assert tools_for(POOL, patterns=[]) == []


def test_tools_for_node_rules_drop_asking_and_delegating_keep_branches_safe_and_add_web():  # Q19, Q22
    main = ["*", "!web__*"]
    assert names(tools_for(POOL, patterns=main, node=Node())) == ["calculate", "load_skill", "search_docs", "files__write_file"]
    assert names(tools_for(POOL, patterns=main, node=Node(branch=True))) == ["calculate", "load_skill", "search_docs"]
    pool = [*POOL, tool("web__tavily_research"), tool("get_current_time")]
    web = ("web__tavily_search", "get_current_time")  # web 에이전트의 tools: 웹 Tool만 더한다(Q26)
    assert names(tools_for(pool, patterns=main, node=Node(web=web)))[-1] == "web__tavily_search"
    assert "web__tavily_research" not in names(tools_for(pool, patterns=main, node=Node(web=web)))
