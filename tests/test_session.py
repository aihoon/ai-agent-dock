"""대화 세션 테스트. 가짜 모델로 이력 이어짐, 실패 시 롤백, max_steps 턴, 초기화, 설정 검증을 확인한다."""

import pytest
from fakes import FakeLLM, final, tool_call

from ai_agent_dock.core.llm import Tool
from ai_agent_dock.core.session import DEFAULT_SYSTEM_PROMPT, Session

EMPTY_SCHEMA = {"type": "object", "properties": {}}


def ping_tool():
    return Tool("ping", "", EMPTY_SCHEMA, lambda: "pong")


def flaky(*items):
    """차례로 응답을 돌려주되, 항목이 예외이면 그 예외를 던지는 가짜 모델."""
    queue = list(items)

    def llm(history, tools, system_prompt):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return llm


def test_second_turn_sees_first_turn():
    """기억: 두 번째 턴에서 모델이 받은 이력에 첫 번째 턴의 질문과 답이 들어 있다."""
    llm = FakeLLM([final("A1"), final("A2")])
    session = Session([], llm=llm)

    session.ask("Q1")
    session.ask("Q2")

    seen_by_turn2 = [(m.role, m.content) for m in llm.calls[1]]
    assert seen_by_turn2 == [("user", "Q1"), ("assistant", "A1"), ("user", "Q2")]


def test_conversation_continues_after_a_tool_turn():
    """Tool을 쓴 턴 뒤의 턴도 Tool 요청과 결과를 이력으로 가지고 이어진다."""
    llm = FakeLLM([tool_call("c1", "ping"), final("A1"), final("A2")])
    session = Session([ping_tool()], llm=llm)

    session.ask("Q1")
    session.ask("Q2")

    turn2_first_call = llm.calls[2]  # 세 번째 모델 호출 = 두 번째 턴의 첫 스텝
    assert [m.role for m in turn2_first_call] == ["user", "assistant", "tool", "assistant", "user"]
    assert turn2_first_call[2].content == "pong"


def test_failed_turn_is_rolled_back():
    """모델 호출이 예외로 실패하면 예외가 전달되고, 세션 이력은 그 턴 이전 그대로이며 다음 턴이 정상 동작한다."""
    session = Session([], llm=flaky(final("A1"), RuntimeError("네트워크 오류"), final("A3")))
    session.ask("Q1")
    before = list(session.history)

    with pytest.raises(RuntimeError, match="네트워크"):
        session.ask("Q2")

    assert session.history == before
    result = session.ask("Q3")
    assert result.answer == "A3"
    assert [m.content for m in session.history if m.role == "user"] == ["Q1", "Q3"]


def test_failure_in_the_middle_of_a_turn_leaves_no_partial_history():
    """Tool을 실행한 뒤 다음 모델 호출이 실패해도, 그 턴의 Tool 요청과 결과가 이력에 반쯤 남지 않는다."""
    session = Session([ping_tool()], llm=flaky(tool_call("c1", "ping"), RuntimeError("중간 실패")))

    with pytest.raises(RuntimeError):
        session.ask("Q1")

    assert session.history == []


def test_max_steps_turn_keeps_history_and_the_next_turn_continues():
    """최대 스텝에 도달한 턴의 이력은 유지되고, 다음 턴은 Tool 결과 바로 뒤에 사용자 입력을 이어 받는다."""
    llm = FakeLLM([tool_call("c1", "ping"), tool_call("c2", "ping"), final("계속했다")])
    session = Session([ping_tool()], max_steps=2, llm=llm)

    first = session.ask("Q1")
    second = session.ask("계속해")

    assert first.status == "max_steps"
    assert second.status == "completed" and second.answer == "계속했다"
    seen = llm.calls[2]
    assert [m.role for m in seen][-2:] == ["tool", "user"]  # Tool 결과 바로 뒤에 새 사용자 입력


def test_reset_clears_history_but_keeps_configuration():
    llm = FakeLLM([final("A1"), final("A2")])
    session = Session([ping_tool()], system_prompt="지침", max_steps=3, llm=llm)
    session.ask("Q1")

    session.reset()
    session.ask("Q2")

    assert [m.content for m in llm.calls[1]] == ["Q2"]  # 초기화 뒤에는 앞 대화를 모른다.
    assert session.system_prompt == "지침" and session.max_steps == 3 and [t.name for t in session.tools] == ["ping"]


def test_system_prompt_is_passed_every_call_but_not_stored_in_history():
    seen_system = []

    def llm(history, tools, system_prompt):
        seen_system.append(system_prompt)
        return final("답")

    session = Session([], llm=llm)
    session.ask("Q1")
    session.ask("Q2")

    assert seen_system == [DEFAULT_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT]
    assert all(m.role != "system" for m in session.history)


def test_llm_and_provider_cannot_be_combined():
    with pytest.raises(ValueError, match="provider"):
        Session([], provider="openai", llm=FakeLLM([]))
