"""Tool 레지스트리 테스트: 이름 중복 거부, 인자 스키마 검증, 결과 길이 제한, 루프와의 연결(루프는 고치지 않았다)."""

import pytest
from fakes import TEST_CONFIG, FakeLLM, final, tool_call

from ai_agent_dock.core.llm import Tool
from ai_agent_dock.core.runtime.loop import run_agent
from ai_agent_dock.core.runtime.types import AgentSettings
from ai_agent_dock.core.tools import BUILTIN_TOOLS, ToolArgumentError, ToolRegistry

ADD_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "mode": {"type": "string", "enum": ["fast", "slow"]}},
    "required": ["a"],
}


def add_tool(name="add", func=None):
    return Tool(name, "더하기", ADD_SCHEMA, func or (lambda a, mode="fast": a + 1))


def test_registers_tools_and_remembers_their_sources():
    registry = ToolRegistry()
    registry.add_all(BUILTIN_TOOLS)
    registry.add(add_tool(), source="mcp:demo")

    assert [t.name for t in registry.tools()] == ["get_current_time", "calculate", "add"]
    assert registry.source_of("calculate") == "builtin"
    assert registry.source_of("add") == "mcp:demo"
    assert len(registry) == 3


def test_duplicate_name_is_rejected_naming_both_sources():
    registry = ToolRegistry()
    registry.add(add_tool(), source="builtin")

    with pytest.raises(ValueError, match=r"add.*builtin.*mcp:demo"):
        registry.add(add_tool(), source="mcp:demo")


def test_invalid_json_schema_is_rejected_when_registering():
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="broken"):
        registry.add(Tool("broken", "", {"type": "object", "properties": {"a": {"type": "no-such-type"}}}, lambda: 1))


def test_valid_arguments_reach_the_function_and_result_is_a_string():
    registry = ToolRegistry()
    registry.add(add_tool())
    tool = registry.tools()[0]

    assert tool.func(a=1, mode="slow") == "2"  # 루프도 str()로 바꾸므로 결과 문자열은 달라지지 않는다.


@pytest.mark.parametrize(
    ("arguments", "mentions"),
    [
        ({}, "'a' is a required property"),  # 필수 인자 누락
        ({"a": "하나"}, "a:"),  # 타입 불일치: 어느 인자가 틀렸는지 경로가 나온다
        ({"a": 1, "mode": "medium"}, "mode:"),  # enum에 없는 값
    ],
)
def test_invalid_arguments_are_rejected_before_the_function_runs(arguments, mentions):
    calls = []
    registry = ToolRegistry()
    registry.add(add_tool(func=lambda **kw: calls.append(kw)))

    with pytest.raises(ToolArgumentError, match="add") as error:
        registry.tools()[0].func(**arguments)

    assert mentions in str(error.value)
    assert calls == []  # Tool 함수는 호출되지 않았다.


def test_all_argument_errors_are_reported_together():
    registry = ToolRegistry()
    registry.add(add_tool())

    with pytest.raises(ToolArgumentError) as error:
        registry.tools()[0].func(a="x", mode="medium")

    assert "a:" in str(error.value) and "mode:" in str(error.value)


def test_long_result_is_cut_with_a_notice_that_says_how_much_was_cut():
    registry = ToolRegistry(max_result_chars=10)
    registry.add(Tool("long", "", {"type": "object", "properties": {}}, lambda: "가" * 25))

    result = registry.tools()[0].func()

    assert result.startswith("가" * 10 + "\n")
    assert "전체 25자 중 앞 10자" in result


def test_result_within_the_limit_is_unchanged():
    registry = ToolRegistry(max_result_chars=10)
    registry.add(Tool("exact", "", {"type": "object", "properties": {}}, lambda: "나" * 10))

    assert registry.tools()[0].func() == "나" * 10


def test_builtin_tools_keep_working_through_the_registry():
    registry = ToolRegistry()
    registry.add_all(BUILTIN_TOOLS)
    by_name = {t.name: t for t in registry.tools()}

    assert by_name["calculate"].func(expression="17*23") == "391"
    with pytest.raises(ToolArgumentError):
        by_name["calculate"].func()  # 필수 인자 누락은 이제 TypeError가 아니라 인자 오류다.


def test_loop_reports_argument_errors_to_the_model_as_error_observations():
    """루프는 고치지 않았다: Tool이 던진 인자 오류가 그대로 오류 관찰(is_error)이 되어 모델에 전달된다."""
    registry = ToolRegistry()
    registry.add(add_tool())
    llm = FakeLLM([tool_call("c1", "add", a="하나"), final("고쳐서 다시 부르겠습니다")])

    result = run_agent(AgentSettings(config=TEST_CONFIG, tools=registry.tools(), llm=llm), "더해 줘")

    tool_message = next(m for m in result.history if m.role == "tool")
    assert tool_message.is_error
    assert "add의 인자가 형식에 맞지 않는다" in tool_message.content
    assert result.status == "completed"
