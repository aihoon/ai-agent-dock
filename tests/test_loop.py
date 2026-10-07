"""Agent Loop 테스트. 가짜 모델로 연속 호출, 오류 처리, 최대 턴을 확인한다."""

from fakes import FakeLLM, final, tool_call

from ai_agent_dock.core.llm import LLMResponse, Tool, ToolCall
from ai_agent_dock.core.runtime.loop import run_agent

EMPTY_SCHEMA = {"type": "object", "properties": {}}


def test_consecutive_tool_calls():
    """Tool을 2번 연속 호출하고, 첫 결과가 다음 모델 호출에 전달된다."""
    executed = []

    def add(a, b):
        executed.append((a, b))
        return a + b

    tool = Tool("add", "두 수를 더한다", EMPTY_SCHEMA, add)
    llm = FakeLLM([tool_call("c1", "add", a=1, b=2), tool_call("c2", "add", a=3, b=4), final("끝")])

    result = run_agent("질문", [tool], llm=llm)

    assert executed == [(1, 2), (3, 4)]
    assert (result.status, result.answer, result.steps) == ("completed", "끝", 3)
    # 두 번째 모델 호출은 첫 번째 Tool 결과를 이력에서 볼 수 있어야 한다.
    assert [m.content for m in llm.calls[1] if m.role == "tool"] == ["3"]


def test_tool_error_does_not_stop_loop():
    """Tool이 예외를 던져도 루프는 계속되고, 오류 내용이 모델에 전달된다."""

    def boom():
        raise ValueError("고장")

    llm = FakeLLM([tool_call("c1", "boom"), final("복구했다")])

    result = run_agent("질문", [Tool("boom", "항상 실패", EMPTY_SCHEMA, boom)], llm=llm)

    assert (result.status, result.answer) == ("completed", "복구했다")
    seen = [m for m in llm.calls[1] if m.role == "tool"]
    assert seen[0].is_error and "고장" in seen[0].content


def test_unknown_tool_and_bad_arguments_are_reported():
    """존재하지 않는 Tool 이름과 해석 실패한 인자도 오류로 모델에 알리고 계속한다."""
    bad_args = LLMResponse(text=None, tool_calls=[ToolCall("c2", "ping", {}, error="잘못된 Tool 인자")])
    llm = FakeLLM([tool_call("c1", "nope"), bad_args, final("정리했다")])
    ping = Tool("ping", "", EMPTY_SCHEMA, lambda: "pong")

    result = run_agent("질문", [ping], llm=llm)

    assert result.status == "completed"
    errors = [m.content for m in result.history if m.role == "tool" and m.is_error]
    assert "알 수 없는 Tool: nope" in errors[0]
    assert "잘못된 Tool 인자" in errors[1]


def test_max_steps_stops_infinite_loop():
    """모델이 끝없이 Tool만 요청해도 최대 스텝에서 반드시 종료된다."""
    count = []
    ping = Tool("ping", "", EMPTY_SCHEMA, lambda: count.append(1) or "pong")
    llm = FakeLLM([tool_call("c1", "ping")], repeat_last=True)

    result = run_agent("질문", [ping], max_steps=3, llm=llm)

    assert (result.status, result.steps) == ("max_steps", 3)
    assert len(llm.calls) == 3 and len(count) == 3


def test_run_agent_continues_from_prior_history_without_mutating_it():
    """이전 이력을 받으면 모델이 그 이력을 보고, 넘긴 목록은 호출 뒤에도 바뀌지 않으며, 결과에 전체 이력이 담긴다."""
    first_llm = FakeLLM([final("A1")])
    first = run_agent("Q1", [], llm=first_llm)
    prior = list(first.history)

    second_llm = FakeLLM([final("A2")])
    second = run_agent("Q2", [], history=first.history, llm=second_llm)

    assert [(m.role, m.content) for m in second_llm.calls[0]] == [("user", "Q1"), ("assistant", "A1"), ("user", "Q2")]
    assert first.history == prior  # 넘긴 이력은 그대로
    assert [m.content for m in second.history] == ["Q1", "A1", "Q2", "A2"]


def test_run_agent_without_history_starts_fresh():
    """history를 생략하는 C1 방식의 호출은 그대로 빈 이력에서 시작한다."""
    llm = FakeLLM([final("A")])

    run_agent("Q", [], llm=llm)

    assert [m.content for m in llm.calls[0]] == ["Q"]


def test_max_steps_comes_from_env_when_not_passed(monkeypatch):
    """max_steps 인자가 없으면 환경 변수 AGENT_MAX_STEPS를 쓰고, 인자가 있으면 인자가 우선한다."""
    ping = Tool("ping", "", EMPTY_SCHEMA, lambda: "pong")
    monkeypatch.setenv("AGENT_MAX_STEPS", "2")

    from_env = run_agent("질문", [ping], llm=FakeLLM([tool_call("c1", "ping")], repeat_last=True))
    explicit = run_agent("질문", [ping], max_steps=4, llm=FakeLLM([tool_call("c1", "ping")], repeat_last=True))

    assert from_env.steps == 2
    assert explicit.steps == 4
