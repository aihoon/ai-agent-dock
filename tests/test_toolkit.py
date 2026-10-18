"""빌트인 Tool 조립 테스트: 목록의 구성과 출처, 그리고 의존 방향(순환 방지)."""

import re
from pathlib import Path

from ai_agent_dock.core.retrieval import Knowledge
from ai_agent_dock.core.skills import SkillMeta
from ai_agent_dock.core.toolkit import make_builtin_tools
from ai_agent_dock.core.tools import ToolRegistry

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


def test_only_the_toolkit_imports_the_feature_modules():
    """의존 방향: toolkit만 기능 모듈을 import한다. 기능 모듈과 tools, llm은 toolkit을 import하지 않는다."""
    for folder in ["tools", "skills", "retrieval", "hitl", "llm"]:
        for path in (CORE / folder).rglob("*.py"):
            assert not re.search(r"\btoolkit\b", path.read_text(encoding="utf-8")), path
