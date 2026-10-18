"""Agent Loop: 모델 호출 → Tool 실행 → 결과 관찰을 반복한다.

용어: 사용자 입력 하나와 최종 답변 하나를 "턴(문답)"이라 하고, 한 턴 안에서 모델을 한 번 호출하는
루프 한 바퀴를 "스텝"이라 한다. run_agent는 턴 하나를 처리하며, 그 안에서 스텝을 반복한다.

조절 가능한 환경 변수: AGENT_MAX_STEPS (한 턴에서 모델을 호출하는 횟수의 상한, 기본 10)
"""

from __future__ import annotations

import logging
import time

from ai_agent_dock.env import env_int

from ..llm import AssistantMessage, LLMRequest, Message, Tool, ToolCall, ToolMessage, UserMessage
from ..observability import Tracer
from .types import AgentResult, AgentSettings

log = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 10

# 실행하지 않은 Tool의 관찰 문장 앞부분. 평가(core/evals)가 "실행했다 실패한 Tool"과 구분할 때 쓴다.
BLOCKED = "이 턴에서 사용자가"
DENIED = "사용자가 실행을 거부했다."
REVISED = "사용자가 다르게 요청했다:"
NOT_RUN_PREFIXES = (BLOCKED, DENIED, REVISED)


def run_agent(settings: AgentSettings, question: str, *, history: list[Message] | None = None) -> AgentResult:
    """턴 하나(질문 하나)를 처리한다. settings.max_steps는 모델을 호출하는 횟수의 상한이다.

    settings는 세션 내내 같은 설정이고, 모든 스텝의 요청에 그대로 실린다. history는 이전 턴들의 이력이다.
    넘긴 목록은 바꾸지 않고 복사해서 쓰며, 이번 턴이 반영된 전체 이력을 결과(AgentResult.history)로 돌려준다.
    생략하면 빈 이력으로 시작한다(이전 턴을 기억하지 않는다).

    max_steps 우선순위: settings.max_steps > 환경 변수 AGENT_MAX_STEPS > 기본값 10.
    """
    max_steps = settings.max_steps
    if max_steps is None:
        max_steps = env_int("AGENT_MAX_STEPS", DEFAULT_MAX_STEPS)
    tools_by_name = {t.name: t for t in settings.tools}
    history = [*(history or []), UserMessage(content=question)]
    answer: str | None = None
    denied: set[str] = set()  # 이번 턴에서 사용자가 거부한 Tool 이름. 같은 턴에서는 다시 묻지 않는다.
    tracer = settings.tracer
    turn_started = time.perf_counter()
    if tracer is not None:
        tracer.begin_turn()
        tracer.record("turn_start", None, question=question)

    for step in range(1, max_steps + 1):
        started = time.perf_counter()
        response = settings.llm(
            LLMRequest(config=settings.config, messages=history, tools=settings.tools, system_prompt=settings.system_prompt)
        )
        if tracer is not None:
            tracer.record(
                "llm",
                step,
                _ms_since(started),
                text=response.text,
                tool_calls=[{"name": c.name, "arguments": c.arguments} for c in response.tool_calls],
                input_tokens=response.input_tokens,
                input_tokens_source=response.input_tokens_source,
                output_tokens=response.output_tokens,
                output_tokens_source=response.output_tokens_source,
            )
        # 응답을 이력에 그대로 이어 붙여야 모델이 다음 호출에서 자신의 이전 판단을 본다.
        history.append(
            AssistantMessage(content=response.text, tool_calls=response.tool_calls, raw=response.raw)
        )
        answer = response.text

        if not response.tool_calls:
            log.info("step %d: 최종 답변", step)
            return _finish(tracer, turn_started, AgentResult(answer=answer, status="completed", steps=step, history=history))

        for call in response.tool_calls:
            started = time.perf_counter()
            content, is_error = _run_call(settings, tools_by_name, call, denied)
            if tracer is not None:
                tracer.record(
                    "tool",
                    step,
                    _ms_since(started),
                    name=call.name,
                    arguments=call.arguments,
                    result=content,
                    is_error=is_error,
                )
            history.append(ToolMessage(tool_call_id=call.id, content=content, is_error=is_error))

    # 모델이 끝까지 Tool만 요청했다. 마지막 응답 텍스트(없을 수 있음)와 함께 종료 사유를 알린다.
    log.warning("최대 스텝(%d)에 도달해 종료", max_steps)
    return _finish(tracer, turn_started, AgentResult(answer=answer, status="max_steps", steps=max_steps, history=history))


def _ms_since(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _finish(tracer: Tracer | None, turn_started: float, result: AgentResult) -> AgentResult:
    """턴 끝 줄(턴 전체 걸린 시간 포함)을 쓰고 결과를 돌려준다. 추적을 끈 경우에도 결과는 그대로 돌려준다."""
    if tracer is not None:
        tracer.record("turn_end", result.steps, _ms_since(turn_started), answer=result.answer, status=result.status)
    return result


def _run_call(
    settings: AgentSettings, tools_by_name: dict[str, Tool], call: ToolCall, denied: set[str]
) -> tuple[str, bool]:
    """Tool 하나를 실행하기 전에, 읽기 전용이 아니면 승인을 받는다. 거부하면 실행하지 않고 그 사실을 관찰로 돌려준다.

    이번 턴에서 거부된 Tool은 승인을 다시 묻지 않고 막는다(denied에 이름을 모은다).
    """
    tool = tools_by_name.get(call.name)
    needs_approval = tool is not None and not (tool.read_only and tool.trusted)
    decision = "auto"
    if needs_approval and call.name in denied:
        decision = "blocked"
        content, is_error = f"{BLOCKED} {call.name} 실행을 이미 거부했다. 같은 요청을 다시 하지 말고 거부된 사실을 사용자에게 알려라.", True
    elif settings.approve is not None and needs_approval:
        approval = settings.approve(call)
        decision = approval.kind
        if approval.kind == "deny":
            denied.add(call.name)
            reason = f" 이유: {approval.reason}" if approval.reason else ""
            content, is_error = f"{DENIED}{reason}", True
        elif approval.kind == "revise":
            content, is_error = f"{REVISED} {approval.reason}", True
        else:
            content, is_error = _execute(tools_by_name, call)
    else:
        content, is_error = _execute(tools_by_name, call)
    if settings.audit is not None:
        settings.audit.record(
            tool=call.name,
            risk="high" if needs_approval else "low",
            decision=decision,
            arguments=call.arguments,
            result_chars=len(content),
            is_error=is_error,
        )
    return content, is_error


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
