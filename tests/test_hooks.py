"""Hook 테스트(C11 D20~D23): hooks.json 읽기, 외부 명령 실행, 실패 처리, /hooks와 /safety 명령."""

import json
import sys

import pytest
from fakes import TEST_CONFIG, FakeLLM, final, tool_call

from ai_agent_dock.core.hooks import load_hooks
from ai_agent_dock.core.llm import Tool, ToolCall
from ai_agent_dock.core.runtime.loop import HOOK_BLOCKED, run_agent
from ai_agent_dock.core.runtime.types import AgentConfig

SCHEMA = {"type": "object", "properties": {"path": {"type": "string"}}}
READ = Tool("files__read", "read", SCHEMA, lambda **kw: "원문", read_only=True, trusted=True)


def write_hooks(tmp_path, hooks, script):
    (tmp_path / "hook.py").write_text(script, encoding="utf-8")
    (tmp_path / "hooks.json").write_text(json.dumps({"hooks": hooks}), encoding="utf-8")
    return tmp_path / "hooks.json"


def entry(match="*"):
    return {"match": match, "command": [sys.executable, "hook.py"]}


def test_external_hook_can_block_change_result_and_answer(tmp_path):
    script = """import json, sys
e = json.load(sys.stdin)
out = {"before_tool": {"block": "읽기 금지"} if e.get("arguments", {}).get("path") == "secret" else {},
       "after_tool": {"result": e.get("result", "") + " +hook"},
       "after_answer": {"answer": e.get("answer", "") + "!"}}[e["event"]]
print(json.dumps(out, ensure_ascii=False))
"""
    path = write_hooks(tmp_path, {"before_tool": [entry()], "after_tool": [entry("files__*")], "after_answer": [{"command": [sys.executable, "hook.py"]}]}, script)
    records = []
    hooks = load_hooks(path, record=lambda **f: records.append(f))
    llm = FakeLLM([tool_call("c1", "files__read", path="a"), tool_call("c2", "files__read", path="secret"), final("답")])
    result = run_agent(AgentConfig(llm_config=TEST_CONFIG, tools=[READ], llm=llm, hooks=hooks), "읽어")
    first, second = (m.content for m in result.history if m.role == "tool")
    assert first == "원문 +hook"
    assert second.startswith(HOOK_BLOCKED) and "읽기 금지" in second
    assert result.answer == "답!"
    assert {r["event"] for r in records} == {"before_tool", "after_tool", "after_answer"}


def test_failing_hook_blocks_before_tool_and_keeps_values_elsewhere(tmp_path):
    path = write_hooks(tmp_path, {"before_tool": [entry()], "after_tool": [entry()]}, "import sys; sys.exit(3)")
    warnings = []
    hooks = load_hooks(path, warn=warnings.append)
    assert "실행하지 못해 막았다" in hooks.before_tool[0](ToolCall("1", "files__read", {}), READ)
    assert hooks.after_tool[0](ToolCall("1", "files__read", {}), READ, "원문") == "원문" and warnings


def test_slow_hook_times_out_and_match_skips_other_tools(tmp_path):
    path = write_hooks(tmp_path, {"before_tool": [entry("files__*")]}, "import time; time.sleep(5)")
    hooks = load_hooks(path, timeout=0.5)
    assert "끝나지 않았다" in hooks.before_tool[0](ToolCall("1", "files__read", {}), READ)
    other = Tool("calculate", "calc", SCHEMA, lambda **kw: "1")
    assert hooks.before_tool[0](ToolCall("1", "calculate", {}), other) is None  # match가 안 맞으면 실행하지 않는다


def test_unknown_event_and_bad_command_are_errors(tmp_path):
    with pytest.raises(ValueError, match="없는 Hook 이벤트"):
        load_hooks(write_hooks(tmp_path, {"before_answer": [entry()]}, ""))
    with pytest.raises(ValueError, match="command"):
        load_hooks(write_hooks(tmp_path, {"before_tool": [{"command": "python hook.py"}]}, ""))


def test_example_hooks_file_loads():
    from pathlib import Path

    hooks = load_hooks(Path(__file__).resolve().parent.parent / "hooks.example.json")
    assert [s.event for s in hooks.specs] == ["before_tool"]


def test_hooks_and_safety_commands(tmp_path):
    from ai_agent_dock.core.commands import CommandContext, find_command
    from ai_agent_dock.core.memory import MemoryStore
    from ai_agent_dock.core.safety import load_safety_policy, with_safety
    from ai_agent_dock.core.session import Session
    from ai_agent_dock.core.session.store import SessionStore
    from ai_agent_dock.core.tools import ToolRegistry

    (tmp_path / "safety.json").write_text(json.dumps({"tools": {"files__read": {"mode": "ask", "paths": ["*.md"]}}}), encoding="utf-8")
    read = Tool("files__read", "read", SCHEMA, lambda **kw: "", read_only=True, trusted=True, untrusted_output=True)
    config, _ = with_safety(AgentConfig(llm_config=TEST_CONFIG, tools=[read]), policy=load_safety_policy(tmp_path / "safety.json"))
    ctx = CommandContext(session=Session(config), registry=ToolRegistry(), skills=[], store=SessionStore(tmp_path / "s"),
                         memory=MemoryStore(tmp_path / "m.md"), make_prompt=lambda: "", summarize=lambda t: "")

    def run(line):
        command, arg = find_command(line)
        return command.run(ctx, arg)

    assert "사용자 Hook: 없음" in run("/hooks")
    config.hooks = load_hooks(write_hooks(tmp_path, {"before_tool": [entry("files__*")]}, ""))
    assert "before_tool" in run("/hooks") and "files__*" in run("/hooks")
    text = run("/safety")
    assert "항상 켜짐" in text and "감쌀 Tool 1개: files__read" in text and "files__read" in text and "ask, 경로 *.md" in text
