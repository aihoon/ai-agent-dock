"""사람 개입(HITL) 테스트: 위험 등급, 승인 게이트, 번호 메뉴, 되묻기, 감사 로그. 네트워크는 쓰지 않는다."""

import json
from dataclasses import replace
from types import SimpleNamespace

from fakes import TEST_CONFIG, FakeLLM, final, tool_call

from ai_agent_dock.core.hitl import Approval, AuditLog, ask_in_terminal, make_ask_user_tool
from ai_agent_dock.core.llm import Tool, ToolCall
from ai_agent_dock.core.runtime.loop import run_agent
from ai_agent_dock.core.runtime.types import AgentSettings
import pytest

from ai_agent_dock.core.tools.mcp import _to_tool
from ai_agent_dock.core.tools.mcp.config import load_config

EMPTY_SCHEMA = {"type": "object", "properties": {}}
WRITE_CALL = ToolCall("c1", "files__write_file", {"path": "a.md"})


def write_tool(calls: list) -> Tool:
    def write(**arguments):
        calls.append(arguments)
        return "기록함"

    return Tool("files__write_file", "파일을 쓴다", EMPTY_SCHEMA, write)


def approver(*decisions, seen: list | None = None):
    items = list(decisions)

    def approve(call):
        if seen is not None:
            seen.append(call.name)
        return items.pop(0)

    return approve


def _menu(answers):
    it = iter(answers)
    return lambda prompt: next(it)


# --- 읽기 전용 선언 (위험 등급은 Tool이 스스로 알린다) ---


def test_tool_is_untrusted_by_default_and_mcp_adapter_sets_it_from_config():
    assert Tool("x", "", EMPTY_SCHEMA, lambda: "").trusted is False  # 빠뜨린 툴이 승인 없이 실행되지 않도록 기본은 믿지 않는다
    untrusted = SimpleNamespace(name="files", call_tool=None, spec=SimpleNamespace(trusted=False))
    read = SimpleNamespace(name="read_text_file", description="읽기", title=None, input_schema=EMPTY_SCHEMA, annotations=SimpleNamespace(read_only_hint=True))
    assert _to_tool(untrusted, read).trusted is False


def test_tools_need_approval_unless_declared_read_only():
    assert Tool("x", "", EMPTY_SCHEMA, lambda: "").read_only is False
    assert Tool("x", "", EMPTY_SCHEMA, lambda: "", read_only=True).read_only is True


def test_mcp_read_only_hint_and_server_trust_become_the_tools_flags():
    trusted = SimpleNamespace(name="files", call_tool=None, spec=SimpleNamespace(trusted=True))
    untrusted = SimpleNamespace(name="files", call_tool=None, spec=SimpleNamespace(trusted=False))
    read = SimpleNamespace(name="read_text_file", description="읽기", title=None, input_schema=EMPTY_SCHEMA, annotations=SimpleNamespace(read_only_hint=True))
    write = SimpleNamespace(name="write_file", description="쓰기", title=None, input_schema=EMPTY_SCHEMA, annotations=SimpleNamespace(read_only_hint=False))
    unknown = SimpleNamespace(name="mystery", description="", title=None, input_schema=EMPTY_SCHEMA, annotations=None)

    assert (_to_tool(trusted, read).read_only, _to_tool(trusted, read).trusted) == (True, True)
    assert _to_tool(trusted, write).read_only is False
    assert _to_tool(trusted, unknown).read_only is False  # 힌트가 없으면 읽기 전용이 아니다
    assert _to_tool(untrusted, read).trusted is False


# --- 자동 실행 규칙: 읽기 전용 힌트 true AND 서버 신뢰 true ---


def test_auto_run_needs_both_read_only_hint_and_trust():
    asked = []
    read_untrusted = Tool("files__read_text_file", "", EMPTY_SCHEMA, lambda **kwargs: "내용", read_only=True, trusted=False)
    llm = FakeLLM([tool_call("c1", "files__read_text_file"), final("끝")])
    settings = AgentSettings(config=TEST_CONFIG, tools=[read_untrusted], llm=llm, approve=approver(Approval.approve(), seen=asked))

    run_agent(settings, "읽어 줘")

    assert asked == ["files__read_text_file"]  # 읽기 전용이라도 신뢰하지 않는 서버면 묻는다


def test_trusted_tool_without_read_only_hint_still_asks():
    asked = []
    write_trusted = Tool("files__write_file", "", EMPTY_SCHEMA, lambda **kwargs: "기록함", read_only=False, trusted=True)
    llm = FakeLLM([tool_call("c1", "files__write_file", path="a.md"), final("끝")])
    settings = AgentSettings(config=TEST_CONFIG, tools=[write_trusted], llm=llm, approve=approver(Approval.approve(), seen=asked))

    run_agent(settings, "써 줘")

    assert asked == ["files__write_file"]  # 신뢰하는 서버라도 쓰기 툴은 묻는다


# --- mcp.json의 trusted ---


def test_config_trusted_defaults_to_false_and_reads_true(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps({"servers": {"time": {"command": "uvx", "args": ["mcp-server-time"], "trusted": True}, "files": {"command": "npx"}}}),
        encoding="utf-8",
    )

    specs = load_config(path)

    assert specs["time"].trusted is True
    assert specs["files"].trusted is False


def test_config_trusted_must_be_true_or_false(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": {"time": {"command": "uvx", "trusted": "yes"}}}), encoding="utf-8")

    with pytest.raises(ValueError, match="trusted"):
        load_config(path)


# --- 승인 게이트 ---


def test_high_risk_tool_runs_only_after_approval():
    calls, asked = [], []
    llm = FakeLLM([tool_call("c1", "files__write_file", path="a.md", content="x"), final("끝")])
    settings = AgentSettings(config=TEST_CONFIG, tools=[write_tool(calls)], llm=llm, approve=approver(Approval.approve(), seen=asked))

    result = run_agent(settings, "메모 써 줘")

    assert calls == [{"path": "a.md", "content": "x"}]
    assert asked == ["files__write_file"]
    assert result.answer == "끝"


def test_denied_tool_does_not_run_and_the_denial_goes_back_to_the_model():
    calls = []
    llm = FakeLLM([tool_call("c1", "files__write_file", path="a.md", content="x"), final("저장하지 않았다")])
    settings = AgentSettings(config=TEST_CONFIG, tools=[write_tool(calls)], llm=llm, approve=approver(Approval.deny("경로가 틀렸다")))

    result = run_agent(settings, "메모 써 줘")

    assert calls == []
    tool_message = next(m for m in result.history if m.role == "tool")
    assert tool_message.is_error is True
    assert "거부" in tool_message.content and "경로가 틀렸다" in tool_message.content


def test_same_turn_retry_after_denial_is_blocked_without_asking_again():
    calls, asked = [], []
    llm = FakeLLM([
        tool_call("c1", "files__write_file", path="a.md", content="x"),
        tool_call("c2", "files__write_file", path="a.md", content="y"),  # 같은 턴의 재시도
        final("거부되어 저장하지 않았다"),
    ])
    settings = AgentSettings(config=TEST_CONFIG, tools=[write_tool(calls)], llm=llm, approve=approver(Approval.deny(), seen=asked))

    result = run_agent(settings, "메모 써 줘")

    assert calls == [] and asked == ["files__write_file"]  # 두 번째 호출은 묻지 않고 막힌다
    second = [m for m in result.history if m.role == "tool"][1]
    assert second.is_error is True and "이미 거부" in second.content


def test_denial_does_not_carry_over_to_the_next_turn():
    calls, asked = [], []
    llm = FakeLLM([
        tool_call("c1", "files__write_file", path="a.md", content="x"),
        final("거부됨"),
        tool_call("c2", "files__write_file", path="a.md", content="y"),
        final("저장했다"),
    ])
    settings = AgentSettings(config=TEST_CONFIG, tools=[write_tool(calls)], llm=llm, approve=approver(Approval.deny(), Approval.approve(), seen=asked))

    first = run_agent(settings, "메모 써 줘")
    run_agent(settings, "다시 써 줘", history=first.history)

    assert asked == ["files__write_file", "files__write_file"]  # 다음 턴에서는 다시 묻는다
    assert calls == [{"path": "a.md", "content": "y"}]


def test_revise_returns_the_users_request_without_running():
    calls = []
    llm = FakeLLM([tool_call("c1", "files__write_file", path="a.md", content="x"), final("다시 하겠다")])
    settings = AgentSettings(config=TEST_CONFIG, tools=[write_tool(calls)], llm=llm, approve=approver(Approval.revise("b.md에 써 줘")))

    result = run_agent(settings, "메모 써 줘")

    assert calls == []
    tool_message = next(m for m in result.history if m.role == "tool")
    assert "b.md에 써 줘" in tool_message.content


def test_low_risk_tool_is_never_asked():
    asked = []
    llm = FakeLLM([tool_call("c1", "calculate", expression="1+1"), final("끝")])
    calc = Tool("calculate", "", EMPTY_SCHEMA, lambda **kwargs: "2", read_only=True, trusted=True)
    settings = AgentSettings(config=TEST_CONFIG, tools=[calc], llm=llm, approve=approver(seen=asked))

    run_agent(settings, "계산해 줘")

    assert asked == []


def test_without_an_approver_high_risk_tools_run_as_before():
    calls = []
    llm = FakeLLM([tool_call("c1", "files__write_file", path="a.md", content="x"), final("끝")])

    run_agent(AgentSettings(config=TEST_CONFIG, tools=[write_tool(calls)], llm=llm), "써 줘")

    assert len(calls) == 1


# --- 번호 메뉴 ---


def test_terminal_menu_maps_numbers_to_decisions():
    quiet = lambda message: None  # noqa: E731
    assert ask_in_terminal(WRITE_CALL, interactive=True, input_fn=_menu(["1"]), print_fn=quiet) == Approval.approve()
    assert ask_in_terminal(WRITE_CALL, interactive=True, input_fn=_menu(["2"]), print_fn=quiet) == Approval.deny()
    assert ask_in_terminal(WRITE_CALL, interactive=True, input_fn=_menu(["3", "b.md에 써 줘"]), print_fn=quiet) == Approval.revise("b.md에 써 줘")


def test_terminal_menu_asks_again_on_invalid_choice():
    messages = []

    decision = ask_in_terminal(WRITE_CALL, interactive=True, input_fn=_menu(["9", "1"]), print_fn=messages.append)

    assert decision == Approval.approve()
    assert any("1, 2, 3" in message for message in messages)


def test_non_interactive_run_is_denied_without_asking():
    def no_input(prompt):
        raise AssertionError("입력을 받으면 안 된다")

    decision = ask_in_terminal(WRITE_CALL, interactive=False, input_fn=no_input)

    assert decision.kind == "deny"


# --- 되묻기 ---


def test_ask_user_returns_the_answer_and_shows_the_question():
    asked = []
    tool = make_ask_user_tool(input_fn=lambda prompt: asked.append(prompt) or "b.md")

    assert tool.func(question="어느 파일?") == "b.md"
    assert "어느 파일?" in asked[0]


def test_model_asks_the_user_and_continues_with_the_answer():
    tool = make_ask_user_tool(input_fn=lambda prompt: "b.md")
    llm = FakeLLM([tool_call("c1", "ask_user", question="어느 파일?"), final("b.md에 쓰겠다")])

    result = run_agent(AgentSettings(config=TEST_CONFIG, tools=[tool], llm=llm), "써 줘")

    tool_message = next(m for m in result.history if m.role == "tool")
    assert tool_message.content == "b.md"
    assert result.answer == "b.md에 쓰겠다"


# --- 감사 로그 ---


def test_audit_log_records_every_execution_with_short_arguments(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.begin_turn()
    calc = Tool("calculate", "", EMPTY_SCHEMA, lambda **kwargs: "2", read_only=True, trusted=True)
    calls = []
    llm = FakeLLM([
        tool_call("c1", "calculate", expression="1+1"),
        tool_call("c2", "files__write_file", path="a.md", content="가" * 50),
        final("끝"),
    ])
    settings = AgentSettings(
        config=TEST_CONFIG, tools=[calc, write_tool(calls)], llm=llm, approve=approver(Approval.deny()), audit=log
    )

    run_agent(settings, "해 줘")

    lines = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["tool"], r["decision"]) for r in lines] == [("calculate", "auto"), ("files__write_file", "deny")]
    assert lines[1]["args"]["content"] == "가" * 20  # 인자 값은 20자까지만 남는다
    assert lines[1]["turn"] == 1 and lines[1]["is_error"] is True


def test_trusted_time_server_runs_automatically_and_untrusted_files_server_asks(tmp_path):
    """mcp.json에서 시간 서버는 trusted: true, 파일 서버는 trusted: false일 때 자동 실행과 승인이 갈린다."""
    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps({"servers": {
            "time": {"command": "uvx", "args": ["mcp-server-time"], "trusted": True},
            "files": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "sandbox"], "trusted": False},
        }}),
        encoding="utf-8",
    )
    specs = load_config(config)
    time_def = SimpleNamespace(name="get_current_time", description="현재 시각", title=None, input_schema=EMPTY_SCHEMA, annotations=SimpleNamespace(read_only_hint=True))
    read_def = SimpleNamespace(name="read_text_file", description="파일 읽기", title=None, input_schema=EMPTY_SCHEMA, annotations=SimpleNamespace(read_only_hint=True))
    time_tool = _to_tool(SimpleNamespace(name="time", call_tool=None, spec=specs["time"]), time_def)
    files_tool = _to_tool(SimpleNamespace(name="files", call_tool=None, spec=specs["files"]), read_def)

    ran, asked = [], []
    time_tool = replace(time_tool, func=lambda **kwargs: ran.append("time") or "2026-10-16")
    files_tool = replace(files_tool, func=lambda **kwargs: ran.append("files") or "파일 내용")
    llm = FakeLLM([
        tool_call("c1", "time__get_current_time"),
        tool_call("c2", "files__read_text_file", path="a.md"),
        final("끝"),
    ])
    settings = AgentSettings(config=TEST_CONFIG, tools=[time_tool, files_tool], llm=llm, approve=approver(Approval.approve(), seen=asked))

    run_agent(settings, "시각과 파일을 알려 줘")

    assert ran == ["time", "files"]  # 둘 다 실행되지만
    assert asked == ["files__read_text_file"]  # 승인을 받은 것은 신뢰하지 않는 파일 서버뿐이다



def test_disabled_server_is_left_out_of_the_table_but_the_rest_still_load(tmp_path):
    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps({"servers": {"time": {"command": "uvx", "enabled": False}, "files": {"command": "npx"}}}),
        encoding="utf-8",
    )

    assert list(load_config(config)) == ["files"]


def test_enabled_must_be_true_or_false(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": {"time": {"command": "uvx", "enabled": "no"}}}), encoding="utf-8")

    with pytest.raises(ValueError, match="enabled"):
        load_config(path)
