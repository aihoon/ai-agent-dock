"""MCP 연결 테스트. 시험용 서버(tests/mcp_test_server.py)를 stdio 자식 프로세스로 띄운다(로컬 프로세스이고 네트워크는 쓰지 않는다)."""

import os
import sys
import time
from pathlib import Path

import pytest
from fakes import TEST_CONFIG, FakeLLM, final, tool_call

from ai_agent_dock.core.runtime.loop import run_agent
from ai_agent_dock.core.runtime.types import AgentConfig
from ai_agent_dock.core.session import Session
from ai_agent_dock.core.tools import ToolArgumentError, ToolError, ToolRegistry
from ai_agent_dock.core.tools.mcp import ServerSpec, connect_mcp_servers

SERVER = str(Path(__file__).parent / "mcp_test_server.py")


def spec(**env):
    return ServerSpec(sys.executable, [SERVER], env)


def connect(servers=None, call_timeout=10, start_timeout=30, warn=print):
    return connect_mcp_servers(servers or {"test": spec()}, call_timeout=call_timeout, start_timeout=start_timeout, warn=warn)


def registry_of(connected, **kwargs):
    registry = ToolRegistry(**kwargs)
    for server, tools in connected.items():
        registry.add_all(tools, source=f"mcp:{server}")
    return registry


def test_lists_all_tools_with_prefixed_and_sanitized_names():
    with connect() as connected:
        names = {t.name for t in connected["test"]}

    assert names == {"test__echo", "test__fail", "test__sleep", "test__long_text", "test__dotted_name"}


def test_tool_definitions_carry_description_and_schema():
    with connect() as connected:
        echo = next(t for t in connected["test"] if t.name == "test__echo")

    assert "그대로" in echo.description
    assert echo.parameters["required"] == ["text"]


def test_calls_a_tool_on_the_server_and_returns_its_text():
    with connect() as connected:
        by_name = {t.name: t for t in registry_of(connected).tools()}

        assert by_name["test__echo"].func(text="안녕") == "안녕"
        assert by_name["test__dotted_name"].func() == "점"  # 이름이 바뀌어도 서버에는 원래 이름(dotted.name)으로 호출된다.


def test_server_side_tool_failure_becomes_tool_error():
    with connect() as connected:
        by_name = {t.name: t for t in registry_of(connected).tools()}

        with pytest.raises(ToolError, match="fail"):
            by_name["test__fail"].func()


def test_call_that_exceeds_the_time_limit_becomes_tool_error():
    with connect(call_timeout=0.5) as connected:
        by_name = {t.name: t for t in registry_of(connected).tools()}

        started = time.monotonic()
        with pytest.raises(ToolError, match="0.5초"):
            by_name["test__sleep"].func(seconds=5)

        assert time.monotonic() - started < 3  # 서버가 끝나기를 기다리지 않고 제한 시간에 돌아온다.


def test_registry_validates_arguments_before_calling_the_server():
    with connect() as connected:
        by_name = {t.name: t for t in registry_of(connected).tools()}

        with pytest.raises(ToolArgumentError, match="text"):  # 필수 인자 누락: 서버를 부르기 전에 거부된다.
            by_name["test__echo"].func()


def test_long_result_is_cut_by_the_registry():
    with connect() as connected:
        by_name = {t.name: t for t in registry_of(connected, max_result_chars=40).tools()}

        result = by_name["test__long_text"].func(chars=500)

    assert result.startswith("가" * 40) and "전체 500자 중 앞 40자" in result


def test_server_that_cannot_start_is_skipped_and_the_others_still_work():
    warnings = []
    servers = {"broken": ServerSpec("no-such-command-xyz"), "test": spec()}

    with connect(servers, warn=warnings.append) as connected:
        assert list(connected) == ["test"]
        assert len(connected["test"]) == 5

    assert len(warnings) == 1 and "broken" in warnings[0] and "no-such-command-xyz" in warnings[0]


def test_server_that_exits_immediately_is_reported_with_its_real_cause():
    warnings = []
    exits = ServerSpec(sys.executable, ["-c", "import sys; sys.exit(3)"])

    with connect({"gone": exits}, warn=warnings.append) as connected:
        assert connected == {}

    assert "Connection closed" in warnings[0]  # 'Event loop is closed' 같은 부수 오류가 원인을 가리지 않는다.


def test_server_that_never_answers_times_out_and_is_skipped():
    warnings = []
    silent = ServerSpec(sys.executable, ["-c", "import time; time.sleep(60)"])

    with connect({"silent": silent}, start_timeout=1, warn=warnings.append) as connected:
        assert connected == {}

    assert "1초 안에" in warnings[0]


def test_server_process_is_gone_after_leaving_the_with_block(tmp_path):
    pidfile = tmp_path / "pid"

    with connect({"test": spec(TEST_PIDFILE=str(pidfile))}):
        pid = int(pidfile.read_text())
        os.kill(pid, 0)  # 실행 중이면 예외가 없다.

    for _ in range(30):  # 정리가 끝나기를 최대 3초 기다린다.
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    pytest.fail(f"서버 프로세스 {pid}가 종료 후에도 남아 있다")


def test_calling_after_close_gives_a_clear_error():
    with connect() as connected:
        echo = next(t for t in registry_of(connected).tools() if t.name == "test__echo")

    with pytest.raises(ToolError, match="연결되어 있지 않다"):
        echo.func(text="늦음")


def test_agent_loop_uses_an_mcp_tool_like_any_other_tool():
    """루프는 MCP를 모른다: 내장 Tool과 같은 방식으로 호출되고 결과가 이력에 들어간다."""
    with connect() as connected:
        tools = registry_of(connected).tools()
        llm = FakeLLM([tool_call("c1", "test__echo", text="반사"), final("끝")])

        result = run_agent(AgentConfig(llm_config=TEST_CONFIG, tools=tools, llm=llm), "따라 해")

    tool_message = next(m for m in result.history if m.role == "tool")
    assert (tool_message.content, tool_message.is_error) == ("반사", False)


def test_agent_loop_reports_mcp_failures_as_error_observations_and_continues():
    with connect() as connected:
        tools = registry_of(connected).tools()
        llm = FakeLLM([tool_call("c1", "test__fail"), final("실패했지만 계속합니다")])

        result = run_agent(AgentConfig(llm_config=TEST_CONFIG, tools=tools, llm=llm), "실패시켜")

    tool_message = next(m for m in result.history if m.role == "tool")
    assert tool_message.is_error and "fail" in tool_message.content
    assert result.status == "completed" and result.answer == "실패했지만 계속합니다"


def test_streaming_works_in_a_turn_that_uses_an_mcp_tool():
    with connect() as connected:
        tools = registry_of(connected).tools()
        session = Session(AgentConfig(llm_config=TEST_CONFIG, tools=tools, llm=FakeLLM([tool_call("c1", "test__echo", text="x"), final("완료했습니다")])))
        seen = []

        session.ask("해 줘", on_text_delta=seen.append)

    assert "".join(seen) == "완료했습니다"
