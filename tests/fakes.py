"""테스트용 가짜 모델. 미리 정한 응답을 차례로 돌려준다."""

from __future__ import annotations

from collections.abc import Callable

from ai_agent_dock.core.llm import LLMResponse, Message, Tool, ToolCall

DELTA_SIZE = 3  # 가짜 스트리밍에서 한 번에 전달하는 글자 수


class FakeLLM:
    """call_llm과 같은 모양으로 호출되며, 호출마다 받은 이력을 calls에 기록한다."""

    def __init__(self, responses: list[LLMResponse], repeat_last: bool = False):
        self._responses = list(responses)
        self._repeat_last = repeat_last  # True면 응답이 바닥난 뒤에도 마지막 응답을 계속 돌려준다.
        self.calls: list[list[Message]] = []

    def __call__(
        self,
        history: list[Message],
        tools: list[Tool],
        system_prompt: str | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        self.calls.append(list(history))  # 그 시점의 이력을 복사해 둔다(이후 변경 방지).
        n = len(self.calls)
        if n <= len(self._responses):
            response = self._responses[n - 1]
        elif self._repeat_last:
            response = self._responses[-1]
        else:
            raise AssertionError("가짜 모델에 준비된 응답을 모두 사용했다")
        if on_text_delta is not None and response.text:
            for i in range(0, len(response.text), DELTA_SIZE):  # 스트리밍처럼 글자 조각으로 나눠 전달한다.
                on_text_delta(response.text[i : i + DELTA_SIZE])
        return response


def tool_call(call_id: str, name: str, **arguments) -> LLMResponse:
    """Tool 하나를 요청하는 응답."""
    return LLMResponse(text=None, tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])


def final(text: str) -> LLMResponse:
    """Tool 없이 끝나는 최종 답변."""
    return LLMResponse(text=text)
