"""Anthropic 어댑터 테스트. 네트워크 없이 형식 변환과 .env 설정 반영만 확인한다."""

from types import SimpleNamespace

import pytest

from ai_agent_dock.core.llm import Message, Tool, ToolCall
from ai_agent_dock.core.llm.providers.anthropic import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    build_request,
    parse,
    to_messages,
    to_tools,
)


def test_consecutive_tool_results_are_merged_into_one_user_message():
    history = [
        Message("user", "질문"),
        Message("assistant", None, [ToolCall("a", "x", {}), ToolCall("b", "y", {})]),
        Message("tool", "결과A", tool_call_id="a"),
        Message("tool", "고장", tool_call_id="b", is_error=True),
    ]

    messages = to_messages(history)

    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"] == [
        {"type": "tool_use", "id": "a", "name": "x", "input": {}},
        {"type": "tool_use", "id": "b", "name": "y", "input": {}},
    ]
    assert messages[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "a", "content": "결과A"},
        {"type": "tool_result", "tool_use_id": "b", "content": "고장", "is_error": True},
    ]


def test_assistant_raw_blocks_are_replayed_unchanged():
    thinking = SimpleNamespace(type="thinking")  # 원본 블록은 손대지 않고 그대로 돌려줘야 한다.
    history = [Message("user", "q"), Message("assistant", "a", [], raw=[thinking])]

    assert to_messages(history)[1]["content"] == [thinking]


def test_to_tools_uses_input_schema():
    schema = {"type": "object", "properties": {}}

    assert to_tools([Tool("t", "설명", schema, lambda: None)]) == [
        {"name": "t", "description": "설명", "input_schema": schema}
    ]


def test_parse_extracts_text_and_tool_use():
    blocks = [
        SimpleNamespace(type="text", text="생각 중"),
        SimpleNamespace(type="tool_use", id="c1", name="add", input={"a": 1}),
    ]

    parsed = parse(SimpleNamespace(content=blocks, stop_reason="tool_use"))

    assert parsed.text == "생각 중"
    assert parsed.tool_calls == [ToolCall("c1", "add", {"a": 1})]
    assert parsed.raw == blocks


def test_parse_flags_tool_calls_when_output_was_truncated():
    blocks = [SimpleNamespace(type="tool_use", id="c1", name="add", input={})]

    parsed = parse(SimpleNamespace(content=blocks, stop_reason="max_tokens"))

    assert parsed.tool_calls[0].error is not None


def test_parse_raises_on_refusal():
    with pytest.raises(RuntimeError, match="refusal"):
        parse(SimpleNamespace(content=[], stop_reason="refusal"))


def test_build_request_uses_defaults_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_MAX_TOKENS", raising=False)

    request = build_request([Message("user", "q")], [])

    assert request["model"] == DEFAULT_MODEL
    assert request["max_tokens"] == DEFAULT_MAX_TOKENS
    assert "tools" not in request and "system" not in request


def test_build_request_reflects_env_settings(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "my-model")
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "2048")

    request = build_request([Message("user", "q")], [], system="시스템")

    assert request["model"] == "my-model"
    assert request["max_tokens"] == 2048
    assert request["system"] == "시스템"
