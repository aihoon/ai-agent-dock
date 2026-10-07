"""스트리밍 테스트. 가짜 스트림 객체로 어댑터를, 가짜 모델로 세션을 시험한다(네트워크 없음).

확인하는 것: 글자 조각이 순서대로 전달된다 / 최종 응답은 스트리밍 여부와 같은 형식이다 /
스트림이 끊기면 예외가 전달되고 세션 이력은 롤백된다 / on_text_delta를 안 주면 기존 경로를 쓴다.
"""

from types import SimpleNamespace

import pytest
from fakes import FakeLLM, final, tool_call

from ai_agent_dock.core.llm import LLMResponse, Tool, ToolCall
from ai_agent_dock.core.llm import client as llm_client
from ai_agent_dock.core.llm.providers import anthropic as anthropic_provider
from ai_agent_dock.core.llm.providers import openai as openai_provider
from ai_agent_dock.core.session import Session

EMPTY_SCHEMA = {"type": "object", "properties": {}}


class FakeStream:
    """SDK의 스트림 객체 흉내. with 문과 이벤트 순회, 최종 응답 조립을 제공한다."""

    def __init__(self, events, final_response, text_stream=(), error=None):
        self.events = events
        self.final_response = final_response
        self.text_stream = iter(text_stream)  # Anthropic 방식
        self.error = error
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def __iter__(self):  # OpenAI 방식
        yield from self.events
        if self.error:
            raise self.error

    def get_final_response(self):
        return self.final_response

    def get_final_message(self):
        return self.final_response


def openai_response(text):
    return SimpleNamespace(output=[], output_text=text, status="completed")


def anthropic_message(text):
    return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])


def delta(text):
    return SimpleNamespace(type="response.output_text.delta", delta=text)


# --- OpenAI 어댑터 ---


def test_openai_stream_forwards_text_deltas_in_order_and_parses_final_response(monkeypatch):
    stream = FakeStream(
        [SimpleNamespace(type="response.created"), delta("안"), delta("녕"), SimpleNamespace(type="response.completed")],
        openai_response("안녕"),
    )
    client = SimpleNamespace(responses=SimpleNamespace(stream=lambda **kw: stream))
    monkeypatch.setattr(openai_provider, "_client", lambda: client)
    seen = []

    response = openai_provider.call([], [], on_text_delta=seen.append)

    assert seen == ["안", "녕"]  # 글자 조각이 아닌 이벤트는 건너뛴다.
    assert response == openai_provider.parse(openai_response("안녕"))  # 비스트리밍과 같은 형식
    assert stream.closed


def test_openai_without_callback_uses_create_not_stream(monkeypatch):
    def forbidden(**kw):
        raise AssertionError("스트리밍을 요청하지 않았는데 stream이 호출됐다")

    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: openai_response("끝"), stream=forbidden))
    monkeypatch.setattr(openai_provider, "_client", lambda: client)

    assert openai_provider.call([], []).text == "끝"


def test_openai_stream_error_propagates_and_closes_the_stream(monkeypatch):
    stream = FakeStream([delta("안")], openai_response("안녕"), error=ConnectionError("끊김"))
    client = SimpleNamespace(responses=SimpleNamespace(stream=lambda **kw: stream))
    monkeypatch.setattr(openai_provider, "_client", lambda: client)

    with pytest.raises(ConnectionError):
        openai_provider.call([], [], on_text_delta=lambda t: None)

    assert stream.closed


# --- Anthropic 어댑터 ---


def test_anthropic_stream_forwards_text_deltas_in_order_and_parses_final_message(monkeypatch):
    stream = FakeStream([], anthropic_message("안녕"), text_stream=["안", "녕"])
    client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: stream))
    monkeypatch.setattr(anthropic_provider, "_client", lambda: client)
    seen = []

    response = anthropic_provider.call([], [], on_text_delta=seen.append)

    assert seen == ["안", "녕"]
    assert response == anthropic_provider.parse(anthropic_message("안녕"))
    assert stream.closed


def test_anthropic_without_callback_uses_create_not_stream(monkeypatch):
    def forbidden(**kw):
        raise AssertionError("스트리밍을 요청하지 않았는데 stream이 호출됐다")

    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: anthropic_message("끝"), stream=forbidden))
    monkeypatch.setattr(anthropic_provider, "_client", lambda: client)

    assert anthropic_provider.call([], []).text == "끝"


def test_anthropic_stream_with_tool_use_keeps_complete_tool_call(monkeypatch):
    """Tool 요청은 조각이 아니라 SDK가 조립한 완성된 최종 응답에서 ToolCall이 된다."""
    message = SimpleNamespace(
        stop_reason="tool_use",
        content=[SimpleNamespace(type="tool_use", id="c1", name="add", input={"a": 1})],
    )
    stream = FakeStream([], message, text_stream=[])
    client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: stream))
    monkeypatch.setattr(anthropic_provider, "_client", lambda: client)

    response = anthropic_provider.call([], [], on_text_delta=lambda t: None)

    assert response.text is None
    assert [(c.id, c.name, c.arguments, c.error) for c in response.tool_calls] == [("c1", "add", {"a": 1}, None)]


# --- call_llm ---


def test_call_llm_passes_callback_to_adapter_only_when_given(monkeypatch):
    seen = []
    adapter = SimpleNamespace(call=lambda *args: seen.append(args) or "응답")
    monkeypatch.setattr(llm_client, "_provider", lambda name: adapter)
    callback = lambda text: None  # noqa: E731

    llm_client.call_llm([], [], "시스템")
    llm_client.call_llm([], [], "시스템", on_text_delta=callback)

    assert len(seen[0]) == 3  # 콜백이 없으면 기존 호출 모양 그대로
    assert seen[1][3] is callback


# --- Session ---


def test_session_streams_text_of_every_step_and_keeps_same_history_as_batch():
    tools = [Tool("ping", "", EMPTY_SCHEMA, lambda: "pong")]

    def script():
        return FakeLLM([tool_call("c1", "ping"), final("완료했습니다")])

    seen = []

    streamed = Session(tools, llm=script())
    result = streamed.ask("확인해", on_text_delta=seen.append)
    batch = Session(tools, llm=script())
    batch.ask("확인해")

    assert "".join(seen) == "완료했습니다"
    assert result.answer == "완료했습니다"
    assert streamed.history == batch.history  # 스트리밍이어도 이력은 같다.


def test_session_streams_text_that_comes_before_a_tool_call():
    """Tool을 부르기 전에 모델이 쓴 설명도 도착하는 대로 전달된다."""
    explain = LLMResponse(text="확인해 볼게요", tool_calls=[ToolCall("c1", "ping", {})])
    session = Session([Tool("ping", "", EMPTY_SCHEMA, lambda: "pong")], llm=FakeLLM([explain, final("끝")]))
    seen = []

    session.ask("확인해", on_text_delta=seen.append)

    assert "".join(seen) == "확인해 볼게요끝"


def test_session_rolls_back_when_stream_breaks_midway():
    """스트림이 도중에 끊겨 예외가 나면 그 턴은 이력에 남지 않는다. 이미 전달한 조각은 되돌릴 수 없다."""
    seen = []

    def breaking(history, tools, system_prompt=None, on_text_delta=None):
        on_text_delta("안")
        raise ConnectionError("스트림 끊김")

    session = Session([], llm=breaking)

    with pytest.raises(ConnectionError):
        session.ask("질문", on_text_delta=seen.append)

    assert seen == ["안"]
    assert session.history == []


def test_session_without_callback_does_not_pass_on_text_delta():
    """콜백이 없으면 llm에 on_text_delta 키워드를 넘기지 않는다(기존 llm 함수와 호환)."""

    def old_style(history, tools, system_prompt):
        return final("답")

    assert Session([], llm=old_style).ask("질문").answer == "답"
