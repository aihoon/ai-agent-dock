"""테스트용 가짜 모델. 미리 정한 응답을 차례로 돌려준다."""

from __future__ import annotations

from ai_agent_dock.core.llm import LLMResponse, Message, Tool, ToolCall


class FakeLLM:
    """call_llm과 같은 모양으로 호출되며, 호출마다 받은 이력을 calls에 기록한다."""

    def __init__(self, responses: list[LLMResponse], repeat_last: bool = False):
        self._responses = list(responses)
        self._repeat_last = repeat_last  # True면 응답이 바닥난 뒤에도 마지막 응답을 계속 돌려준다.
        self.calls: list[list[Message]] = []

    def __call__(self, history: list[Message], tools: list[Tool], system: str | None = None) -> LLMResponse:
        self.calls.append(list(history))  # 그 시점의 이력을 복사해 둔다(이후 변경 방지).
        n = len(self.calls)
        if n <= len(self._responses):
            return self._responses[n - 1]
        if self._repeat_last:
            return self._responses[-1]
        raise AssertionError("가짜 모델에 준비된 응답을 모두 사용했다")


def tool_call(call_id: str, name: str, **arguments) -> LLMResponse:
    """Tool 하나를 요청하는 응답."""
    return LLMResponse(text=None, tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])


def final(text: str) -> LLMResponse:
    """Tool 없이 끝나는 최종 답변."""
    return LLMResponse(text=text)
