"""Agent Loop: 모델 호출 → Tool 실행 → 결과 관찰을 반복한다.

조절 가능한 환경 변수: AGENT_MAX_TURNS (모델 호출 횟수의 상한, 기본 10)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from ai_agent_dock.env import env_int

from ..llm import LLMResponse, Message, Tool, ToolCall, call_llm

log = logging.getLogger(__name__)

DEFAULT_MAX_TURNS = 10


@dataclass
class AgentResult:
    """run_agent의 결과."""

    answer: str | None
    status: str  # "completed"(최종 답변으로 종료) 또는 "max_turns"(최대 턴 도달로 종료)
    turns: int  # 모델을 호출한 횟수
    history: list[Message]


def run_agent(
    question: str,
    tools: list[Tool],
    *,
    system: str | None = None,
    max_turns: int | None = None,
    llm: Callable[[list[Message], list[Tool], str | None], LLMResponse] = call_llm,
) -> AgentResult:
    """질문 하나를 처리한다. max_turns는 모델을 호출하는 횟수의 상한이다.

    max_turns 우선순위: 인자 > 환경 변수 AGENT_MAX_TURNS > 기본값 10.
    """
    if max_turns is None:
        max_turns = env_int("AGENT_MAX_TURNS", DEFAULT_MAX_TURNS)
    tools_by_name = {t.name: t for t in tools}
    history = [Message(role="user", content=question)]
    answer: str | None = None

    for turn in range(1, max_turns + 1):
        response = llm(history, tools, system)
        # 응답을 이력에 그대로 이어 붙여야 모델이 다음 호출에서 자신의 이전 판단을 본다.
        history.append(
            Message(role="assistant", content=response.text, tool_calls=response.tool_calls, raw=response.raw)
        )
        answer = response.text

        if not response.tool_calls:
            log.info("turn %d: 최종 답변", turn)
            return AgentResult(answer=answer, status="completed", turns=turn, history=history)

        for call in response.tool_calls:
            content, is_error = _execute(tools_by_name, call)
            history.append(Message(role="tool", content=content, tool_call_id=call.id, is_error=is_error))

    # 모델이 끝까지 Tool만 요청했다. 마지막 응답 텍스트(없을 수 있음)와 함께 종료 사유를 알린다.
    log.warning("최대 턴(%d)에 도달해 종료", max_turns)
    return AgentResult(answer=answer, status="max_turns", turns=max_turns, history=history)


def _execute(tools_by_name: dict[str, Tool], call: ToolCall) -> tuple[str, bool]:
    """Tool 하나를 실행해 (결과 문자열, 오류 여부)를 돌려준다. 어떤 실패도 예외로 새어 나가지 않는다."""
    if call.error:
        log.warning("Tool %s 건너뜀: %s", call.name, call.error)
        return call.error, True
    tool = tools_by_name.get(call.name)
    if tool is None:
        log.warning("알 수 없는 Tool: %s", call.name)
        return f"알 수 없는 Tool: {call.name}", True
    log.info("Tool 실행: %s", call.name)
    log.debug("Tool 인자: %s", call.arguments)
    try:
        result = tool.func(**call.arguments)
    except Exception as e:  # Tool 오류는 루프를 멈추지 않고 모델에게 알려 다음 판단을 맡긴다.
        log.warning("Tool %s 실패: %s: %s", call.name, type(e).__name__, e)
        return f"{type(e).__name__}: {e}", True
    return str(result), False
