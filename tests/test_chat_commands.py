"""명령 테스트(C9 D25): core/commands.py의 명령을 가짜 모델로 실행하고, /help 목록이 처리하는 명령과 맞는지 본다."""

import json
import re
import sys
from pathlib import Path

from fakes import TEST_CONFIG, FakeLLM, final

from ai_agent_dock.core.commands import COMMANDS, CommandContext, find_command, help_text
from ai_agent_dock.core.memory import MemoryStore
from ai_agent_dock.core.runtime.types import AgentConfig
from ai_agent_dock.core.session import Session
from ai_agent_dock.core.session.store import SessionStore
from ai_agent_dock.core.tools import BUILTIN_TOOLS, ToolRegistry

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def make_ctx(tmp_path, answers=("답",)):
    registry = ToolRegistry()
    registry.add_all(BUILTIN_TOOLS, source="builtin")
    memory = MemoryStore(tmp_path / "memory.md")
    session = Session(AgentConfig(llm_config=TEST_CONFIG, tools=registry.tools(), llm=FakeLLM([final(a) for a in answers])))
    return CommandContext(
        session=session,
        registry=registry,
        skills=[],
        store=SessionStore(tmp_path / "sessions"),
        memory=memory,
        make_prompt=lambda: "지침 " + " ".join(memory.items()),
        summarize=lambda text: "요약",
    )


def run(ctx, line):
    command, arg = find_command(line)
    return command.run(ctx, arg)


def test_command_names_are_unique_and_found_with_arguments():
    names = [c.name for c in COMMANDS]
    assert len(names) == len(set(names))
    command, arg = find_command("/memory-add  반려견은 콩 ")
    assert (command.name, arg) == ("/memory-add", "반려견은 콩")
    assert find_command("/skill")[0].name == "/skill" and find_command("/없는명령") is None


def test_chat_handles_only_app_commands_itself_and_lists_them_in_help():
    sys.path.insert(0, str(EXAMPLES))
    import chat

    source = (EXAMPLES / "chat.py").read_text(encoding="utf-8")
    handled = set(re.findall(r'line == "(/[a-z-]+)"', source))
    assert handled == {name for name, _ in chat.APP_COMMANDS}
    assert all(c.usage in chat.HELP for c in COMMANDS) and "/exit" in chat.HELP


def test_after_turn_saves_and_sessions_resume_by_number(tmp_path):
    ctx = make_ctx(tmp_path, answers=("첫 답", "둘째 답"))
    ctx.session.ask("첫 질문")
    assert ctx.after_turn("첫 질문") is None
    first_id = ctx.record.id

    assert run(ctx, "/session-new") == "새 세션을 시작합니다." and ctx.session.history == []
    ctx.session.ask("둘째 질문")
    ctx.after_turn("둘째 질문")

    listing = run(ctx, "/sessions")
    assert "1." in listing and "2." in listing and "← 지금" in listing
    number = next(n for n, r in enumerate(ctx.store.list(), start=1) if r.id == first_id)
    assert "이어 갑니다(1턴)" in run(ctx, f"/session {number}")
    assert ctx.record.id == first_id and ctx.session.history[0].content == "첫 질문"
    assert run(ctx, "/session 99").startswith("사용법: /session 번호")


def test_memory_commands_reload_the_prompt_and_remove_asks_first(tmp_path):
    ctx = make_ctx(tmp_path)
    run(ctx, "/memory-add 반려견은 콩")
    assert "반려견은 콩" in ctx.session.config.system_prompt

    command, arg = find_command("/memory-delete 1")
    assert command.confirm(ctx, arg) == "'반려견은 콩'을 지울까요?"
    assert command.confirm(ctx, "5") is None  # 없는 번호는 묻지 않고 사용법을 보여 준다.
    assert run(ctx, "/memory-delete 1") == "1번을 지웠습니다. 다음 턴부터 반영합니다."
    assert "반려견은 콩" not in ctx.session.config.system_prompt


def test_tokens_and_compact_report_sizes(tmp_path):
    ctx = make_ctx(tmp_path)
    assert "이력 크기" in run(ctx, "/tokens")
    assert run(ctx, "/compact") == "줄일 이력이 없습니다."


def test_help_puts_app_commands_first():
    text = help_text([("/exit", "종료")])
    assert text.splitlines()[1].strip().startswith("/exit")


def test_runtime_and_session_do_not_import_commands():
    """의존 방향: commands가 session과 runtime을 쓰고, 반대로는 import하지 않는다."""
    core = Path(__file__).resolve().parent.parent / "src" / "ai_agent_dock" / "core"
    for folder in ["runtime", "session", "memory", "llm"]:
        for path in (core / folder).rglob("*.py"):
            assert not re.search(r"\bcommands\b", path.read_text(encoding="utf-8")), path


def test_switched_sessions_get_their_own_session_start(tmp_path):
    from ai_agent_dock.core.observability import Tracer

    ctx = make_ctx(tmp_path, answers=("첫 답", "둘째 답", "셋째 답"))
    ctx.session.config = ctx.session.config.__class__(**{**ctx.session.config.__dict__, "tracer": Tracer(tmp_path / "trace")})
    ctx.session.ask("첫 질문")
    ctx.after_turn("첫 질문")
    first = ctx.tracer.path

    run(ctx, "/session-new")
    assert not ctx.tracer.path.exists()  # 새 세션은 질문 전에는 파일이 없다.
    ctx.session.ask("둘째 질문")
    ctx.after_turn("둘째 질문")
    number = next(n for n, r in enumerate(ctx.store.list(), start=1) if r.id == first.stem)
    assert "이어 갑니다" in run(ctx, f"/session {number}")
    ctx.session.ask("셋째 질문")

    for path in tmp_path.glob("trace/*.jsonl"):
        kinds = [json.loads(line)["kind"] for line in path.read_text(encoding="utf-8").splitlines()]
        assert "session_start" in kinds, path
    kinds = [json.loads(line)["kind"] for line in first.read_text(encoding="utf-8").splitlines()]
    assert kinds.count("session_start") == 2 and "session_resume" in kinds
