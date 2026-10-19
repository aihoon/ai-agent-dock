"""추적 기록과 토큰 테스트(가짜 모델): 세션 파일, 줄 순서와 묶음, 세션 설정 1회, 끄기, 토큰 출처 태그."""

import json

from fakes import TEST_CONFIG, FakeLLM, final, tool_call

from ai_agent_dock.core.llm import LLMRequest, LLMResponse, ToolCall, UserMessage
from ai_agent_dock.core.llm.tokens import PROVIDER, TIKTOKEN, fill_tokens
from ai_agent_dock.core.observability import Tracer
from ai_agent_dock.core.runtime.loop import run_agent
from ai_agent_dock.core.runtime.types import AgentSettings
from ai_agent_dock.core.session import Session
from ai_agent_dock.core.tools import BUILTIN_TOOLS


def lines(tracer):
    return [json.loads(line) for line in tracer.path.read_text(encoding="utf-8").splitlines()]


def calc_turn():
    return [tool_call("c1", "calculate", expression="2+3"), final("5입니다")]


def test_one_turn_writes_ordered_lines_with_the_same_trace_id(tmp_path):
    tracer = Tracer(tmp_path)
    settings = AgentSettings(config=TEST_CONFIG, tools=BUILTIN_TOOLS, llm=FakeLLM(calc_turn()), tracer=tracer)

    run_agent(settings, "2+3은?")

    rows = lines(tracer)
    assert [r["kind"] for r in rows] == ["turn_start", "llm", "tool", "llm", "turn_end"]
    assert {r["trace_id"] for r in rows} == {rows[0]["trace_id"]}
    assert {r["session"] for r in rows} == {tracer.session_id}
    assert all("." in r["time"] for r in rows)  # 밀리초까지 찍는다
    assert rows[2]["name"] == "calculate" and rows[2]["result"] == "5"
    assert rows[-1]["duration_ms"] is not None and rows[-1]["answer"] == "5입니다"
    assert tracer.path == tmp_path / f"{tracer.session_id}.jsonl"  # 세션별 파일


def test_session_start_is_written_once_per_session(tmp_path):
    tracer = Tracer(tmp_path)
    llm = FakeLLM([final("안녕"), final("또 안녕")])
    session = Session(AgentSettings(config=TEST_CONFIG, tools=BUILTIN_TOOLS, llm=llm, tracer=tracer))

    session.ask("안녕")
    session.ask("또")

    kinds = [r["kind"] for r in lines(tracer)]
    assert kinds.count("session_start") == 1 and kinds[0] == "session_start"
    first = lines(tracer)[0]
    assert first["model"] == "fake-model" and {t["name"] for t in first["tools"]} == {"get_current_time", "calculate"}


def test_tracing_off_writes_nothing(tmp_path):
    tracer = Tracer(None)  # TRACE_ENABLED=false일 때 chat이 이렇게 만든다
    run_agent(AgentSettings(config=TEST_CONFIG, tools=BUILTIN_TOOLS, llm=FakeLLM(calc_turn()), tracer=tracer), "2+3은?")
    assert tracer.path is None and list(tmp_path.iterdir()) == []


def test_llm_line_carries_tokens_and_their_source(tmp_path):
    tracer = Tracer(tmp_path)
    response = LLMResponse(text="안녕", input_tokens=12, output_tokens=3, input_tokens_source=PROVIDER, output_tokens_source=PROVIDER)
    run_agent(AgentSettings(config=TEST_CONFIG, llm=FakeLLM([response]), tracer=tracer), "안녕")

    llm = next(r for r in lines(tracer) if r["kind"] == "llm")
    assert (llm["input_tokens"], llm["output_tokens"], llm["input_tokens_source"]) == (12, 3, PROVIDER)


def test_provider_values_are_kept_and_tagged_provider():
    request = LLMRequest(config=TEST_CONFIG, messages=[UserMessage(content="안녕")])
    response = fill_tokens(request, LLMResponse(text="반가워", input_tokens=7, output_tokens=2))
    assert (response.input_tokens, response.input_tokens_source, response.output_tokens_source) == (7, PROVIDER, PROVIDER)


def test_missing_values_are_counted_with_tiktoken_and_tagged():
    request = LLMRequest(config=TEST_CONFIG, messages=[UserMessage(content="17*23은 얼마야?")], system_prompt="지침", tools=BUILTIN_TOOLS)
    response = fill_tokens(request, LLMResponse(text=None, tool_calls=[ToolCall(id="c", name="calculate", arguments={"expression": "17*23"})]))
    assert response.input_tokens > 0 and response.output_tokens > 0
    assert (response.input_tokens_source, response.output_tokens_source) == (TIKTOKEN, TIKTOKEN)


def test_no_trace_file_until_the_first_turn(tmp_path):
    tracer = Tracer(tmp_path)
    Session(AgentSettings(config=TEST_CONFIG, tools=BUILTIN_TOOLS, llm=FakeLLM([]), tracer=tracer))
    assert not tracer.path.exists()  # 질문 없이 끝나면 "턴 0" 세션이 남지 않는다(C9 D27).
