"""시스템 프롬프트 조립 테스트: 조각이 정해진 순서로 들어가고, 조건(Tool 목록)에 맞을 때만 들어간다."""

from pathlib import Path

from ai_agent_dock.core.agents import AGENT_INDEX_HEAD, AgentSpec, build_system_prompt
from ai_agent_dock.core.llm import Tool
from ai_agent_dock.core.retrieval.tool import CITATION_RULE
from ai_agent_dock.core.safety import UNTRUSTED_RULE
from ai_agent_dock.core.skills import SkillMeta

BASE = "기본 지침"  # AGENT.md 본문 자리(C12 Q11). 코드에는 기본 문장이 없다.


def tool(name, untrusted_output=False):
    return Tool(name, "", {"type": "object", "properties": {}}, lambda: "", untrusted_output=untrusted_output)


def test_base_only_when_nothing_else():
    assert build_system_prompt(BASE, [tool("calculate")]) == BASE


def test_parts_are_joined_in_fixed_order():
    skill = SkillMeta(name="요약", description="글을 짧게 줄인다", path=Path("요약/SKILL.md"))
    researcher = AgentSpec(name="researcher", description="조사한다", tools=("search_docs",), body="", path=Path("x"))
    tools = [tool("load_skill"), tool("search_docs", untrusted_output=True)]

    text = build_system_prompt(BASE, tools, skills=[skill], memory=["반려견은 콩"], targets=[researcher])

    order = [BASE, "반려견은 콩", "요약", AGENT_INDEX_HEAD, CITATION_RULE, UNTRUSTED_RULE]
    assert [text.index(part) for part in order] == sorted(text.index(part) for part in order)
    assert text.endswith(UNTRUSTED_RULE)  # 외부 내용 규칙은 마지막(C11 D9)


def test_each_part_follows_the_tools():
    skill = SkillMeta(name="요약", description="글을 짧게 줄인다", path=Path("x"))
    assert build_system_prompt(BASE, [], skills=[skill]) == BASE  # load_skill이 없으면 스킬 목록도 없다
    assert build_system_prompt(BASE, [tool("search_docs")]) == f"{BASE}\n\n{CITATION_RULE}"
    assert build_system_prompt(BASE, [tool("files__read", untrusted_output=True)]) == f"{BASE}\n\n{UNTRUSTED_RULE}"
    assert build_system_prompt(BASE, [], targets=[]) == BASE  # 맡길 에이전트가 없으면 목록도 없다
