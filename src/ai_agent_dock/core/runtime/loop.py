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
from .types import AgentResult, AgentConfig, Usage

log = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 10

# 실행하지 않은 Tool의 관찰 문장 앞부분. 평가(core/evals)가 "실행했다 실패한 Tool"과 구분할 때 쓴다.
BLOCKED = "이 턴에서 사용자가"
DENIED = "사용자가 실행을 거부했다."
REVISED = "사용자가 다르게 요청했다:"
SAFETY_BLOCKED = "안전 정책으로 막았다:"  # Safety가 막았다(C11 D19)
HOOK_BLOCKED = "Hook이 막았다:"  # 사용자 Hook이 막았다(C11 D20)
NOT_RUN_PREFIXES = (BLOCKED, DENIED, REVISED, SAFETY_BLOCKED, HOOK_BLOCKED)


def run_agent(config: AgentConfig, question: str, *, history: list[Message] | None = None) -> AgentResult:
    """턴 하나(질문 하나)를 처리한다. config.max_steps는 모델을 호출하는 횟수의 상한이다.

    settings는 세션 내내 같은 설정이고, 모든 스텝의 요청에 그대로 실린다. history는 이전 턴들의 이력이다.
    넘긴 목록은 바꾸지 않고 복사해서 쓰며, 이번 턴이 반영된 전체 이력을 결과(AgentResult.history)로 돌려준다.
    생략하면 빈 이력으로 시작한다(이전 턴을 기억하지 않는다).

    max_steps 우선순위: config.max_steps > 환경 변수 AGENT_MAX_STEPS > 기본값 10.
    """
    max_steps = config.max_steps
    if max_steps is None:
        max_steps = env_int("AGENT_MAX_STEPS", DEFAULT_MAX_STEPS)
    tools_by_name = {t.name: t for t in config.tools}
    history = [*(history or []), UserMessage(content=question)]
    answer: str | None = None
    denied: set[str] = set()  # 이번 턴에서 사용자가 거부한 Tool 이름. 같은 턴에서는 다시 묻지 않는다.
    tracer = config.tracer
    usage = Usage()  # 이번 턴의 토큰 합계(한 턴에 LLM 호출이 여러 번이면 더한다)
    turn_started = time.perf_counter()
    if tracer is not None:
        tracer.begin_turn()
        tracer.record("turn_start", None, question=question)

    for step in range(1, max_steps + 1):
        started = time.perf_counter()
        response = config.llm(
            LLMRequest(config=config.llm_config, messages=history, tools=config.tools, system_prompt=config.system_prompt)
        )
        if tracer is not None:
            tracer.record(
                "llm",
                step,
                _ms_since(started),
                model=f"{config.llm_config.provider}:{config.llm_config.model}",  # 에이전트마다 모델이 다를 수 있다(C14 Q9).
                text=response.text,
                tool_calls=[{"name": c.name, "arguments": c.arguments} for c in response.tool_calls],
                input_tokens=response.input_tokens,
                input_tokens_source=response.input_tokens_source,
                output_tokens=response.output_tokens,
                output_tokens_source=response.output_tokens_source,
                cached_input_tokens=response.cached_input_tokens,
            )
        usage.add(response)
        # 응답을 이력에 그대로 이어 붙여야 모델이 다음 호출에서 자신의 이전 판단을 본다.
        history.append(
            AssistantMessage(content=response.text, tool_calls=response.tool_calls, raw=response.raw)
        )
        answer = response.text

        if not response.tool_calls:
            filtered = _filter_answer(config, answer)
            if filtered != answer:  # 바뀐 답(예: 키를 가림)을 이력에 둔다. 원본이 남은 raw는 버린다.
                answer = filtered
                history[-1] = AssistantMessage(content=answer)
            log.info("step %d: 최종 답변", step)
            return _finish(tracer, turn_started, AgentResult(answer=answer, status="completed", steps=step, history=history, usage=usage))

        for call in response.tool_calls:
            started = time.perf_counter()
            content, is_error = _run_call(config, tools_by_name, call, denied)
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
    return _finish(tracer, turn_started, AgentResult(answer=answer, status="max_steps", steps=max_steps, history=history, usage=usage))


def _ms_since(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _finish(tracer: Tracer | None, turn_started: float, result: AgentResult) -> AgentResult:
    """턴 끝 줄(턴 전체 걸린 시간 포함)을 쓰고 결과를 돌려준다. 추적을 끈 경우에도 결과는 그대로 돌려준다."""
    if tracer is not None:
        tracer.record("turn_end", result.steps, _ms_since(turn_started), answer=result.answer, status=result.status)
    return result


def _run_call(
    config: AgentConfig, tools_by_name: dict[str, Tool], call: ToolCall, denied: set[str]
) -> tuple[str, bool]:
    """Tool 하나를 실행하기 전에, 읽기 전용이 아니면 승인을 받는다. 거부하면 실행하지 않고 그 사실을 관찰로 돌려준다.

    이번 턴에서 거부된 Tool은 승인을 다시 묻지 않고 막는다(denied에 이름을 모은다).
    """
    tool = tools_by_name.get(call.name)
    needs_approval = tool is not None and not (tool.read_only and tool.trusted)
    decision = "auto"
    blocked = _check_before(config, call, tool)
    if blocked:  # Safety나 Hook이 막으면 승인을 묻지도 실행하지도 않는다(C11 D22).
        decision, prefix, reason = blocked
        content, is_error = f"{prefix} {reason} 같은 요청을 다시 하지 말고 막힌 사실을 사용자에게 알려라.", True
    elif needs_approval and call.name in denied:
        decision = "blocked"
        content, is_error = f"{BLOCKED} {call.name} 실행을 이미 거부했다. 같은 요청을 다시 하지 말고 거부된 사실을 사용자에게 알려라.", True
    elif config.approve is not None and needs_approval:
        approval = config.approve(call)
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
    if decision in ("auto", "approve") and tool is not None:  # Hook 다음 Safety가 마지막(C11 D22)
        for hook in config.hooks.after_tool:
            content = hook(call, tool, content)
        if config.safety is not None:
            content = config.safety.after_tool(call, tool, content)
    if config.audit is not None:
        config.audit.record(
            tool=call.name,
            risk="high" if needs_approval else "low",
            decision=decision,
            arguments=call.arguments,
            result_chars=len(content),
            is_error=is_error,
        )
    return content, is_error


def _check_before(config: AgentConfig, call: ToolCall, tool: Tool | None) -> tuple[str, str, str] | None:
    """Safety를 먼저, 그다음 Hook을 차례로 부른다. 막으면 (감사 로그 결정, 관찰 머리말, 이유)를 돌려준다."""
    if tool is None:
        return None
    if config.safety is not None:
        reason = config.safety.before_tool(call, tool)
        if reason:
            return "safety_block", SAFETY_BLOCKED, reason
    for hook in config.hooks.before_tool:
        reason = hook(call, tool)
        if reason:
            return "hook_block", HOOK_BLOCKED, reason
    return None


def _filter_answer(config: AgentConfig, answer: str | None) -> str | None:
    if answer is None:
        return None
    for hook in config.hooks.after_answer:
        answer = hook(answer)
    if config.safety is not None:  # Safety가 마지막이라 Hook이 답을 바꿔도 키 가리기가 빠지지 않는다.
        answer = config.safety.after_answer(answer)
    return answer


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
