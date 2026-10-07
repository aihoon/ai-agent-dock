"""OpenAI 어댑터 테스트. 네트워크 없이 형식 변환과 .env 설정 반영만 확인한다."""

from types import SimpleNamespace

from ai_agent_dock.core.llm import Message, Tool, ToolCall
from ai_agent_dock.core.llm.providers.openai import DEFAULT_MODEL, build_request, parse, to_input


def test_to_input_rebuilds_items_when_raw_is_empty():
    history = [
        Message("user", "질문"),
        Message("assistant", None, [ToolCall("c1", "add", {"a": 1})]),
        Message("tool", "3", tool_call_id="c1"),
    ]

    assert to_input(history) == [
        {"role": "user", "content": "질문"},
        {"type": "function_call", "call_id": "c1", "name": "add", "arguments": '{"a": 1}'},
        {"type": "function_call_output", "call_id": "c1", "output": "3"},
    ]


def test_parse_extracts_tool_calls_and_flags_broken_arguments():
    ok = SimpleNamespace(type="function_call", call_id="c1", name="add", arguments='{"a": 1}')
    broken = SimpleNamespace(type="function_call", call_id="c2", name="add", arguments="{broken")
    response = SimpleNamespace(output=[ok, broken], output_text="")

    parsed = parse(response)

    assert parsed.text is None
    assert parsed.tool_calls[0].arguments == {"a": 1} and parsed.tool_calls[0].error is None
    assert parsed.tool_calls[1].error is not None
    assert parsed.raw == [ok, broken]


def test_parse_flags_tool_calls_when_response_is_incomplete():
    call = SimpleNamespace(type="function_call", call_id="c1", name="add", arguments='{"a": 1}')

    parsed = parse(SimpleNamespace(output=[call], output_text="", status="incomplete"))

    assert parsed.tool_calls[0].error is not None


def test_build_request_uses_defaults_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MAX_OUTPUT_TOKENS", raising=False)

    request = build_request([Message("user", "q")], [])

    assert request["model"] == DEFAULT_MODEL
    assert "max_output_tokens" not in request and "tools" not in request and "instructions" not in request


def test_build_request_reflects_env_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "my-model")
    monkeypatch.setenv("OPENAI_MAX_OUTPUT_TOKENS", "1234")
    tool = Tool("t", "설명", {"type": "object", "properties": {}}, lambda: None)

    request = build_request([Message("user", "q")], [tool], system_prompt="시스템")

    assert request["model"] == "my-model"
    assert request["max_output_tokens"] == 1234
    assert request["instructions"] == "시스템"
    assert request["tools"][0]["name"] == "t"
