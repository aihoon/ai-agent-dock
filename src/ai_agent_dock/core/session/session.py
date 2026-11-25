"""대화 세션: 한 대화의 이력을 보관하고, 턴(질문 하나)마다 Agent Loop를 돌려 이력을 이어 준다.

세션은 이력과 에이전트 설정(AgentConfig)만 소유한다. 턴 처리는 run_agent가 하고, 세션은 그 결과의 이력을 다음 턴으로 넘길 뿐이다.

Handoff(C21): agents(에이전트 이름 → 설정)를 주면 세션이 "지금 에이전트"를 든다. 지금 에이전트가 handoff Tool을 부르면
세션이 이력에 넘김 알림(사용자 메시지, handoff_note)을 더하고 같은 턴에서 넘겨받은 에이전트가 이어 답하고, 다음 턴부터도 그 에이전트가 답한다.
한 턴에 넘기기는 MAX_HANDOFFS번까지다. 그 뒤에 넘겨받은 에이전트는 handoff Tool 없이 돌아 답을 쓴다(Q11).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial

from ..llm import AssistantMessage, Message, UserMessage
from ..runtime.loop import run_agent
from ..llm import LLMRequest
from ..llm.tokens import request_tokens
from ..runtime.types import AgentResult, AgentConfig, Usage
from ..tools import HANDOFF

__all__ = ["MAX_HANDOFFS", "Session", "handoff_note", "read_handoff"]

MAX_HANDOFFS = 3  # 한 턴에 넘기기 상한(C21 Q11). main과 전문 에이전트가 서로 계속 넘기는 핑퐁을 막는다.
MAIN = "main"  # core/agents의 MAIN과 같다. 세션은 에이전트 정의를 모른다(이름만 든다).


def handoff_note(source: str, target: str, reason: str, *, last: bool) -> str:
    """넘겨받은 에이전트에게 주는 알림(C21 Q25). 이력의 assistant 말을 자기 말로 읽어 다시 넘기거나 빈 답을 내지 않게 한다."""
    note = (
        f"[넘김] {source}이(가) 이 대화를 {target}에게 넘겼다(이유: {reason}). 위 대화에서 assistant가 한 말과 Tool 호출은 앞 에이전트의 것이다. "
        f"당신은 {target}로서 사용자의 마지막 요청을 이어 처리하고 사용자에게 직접 답한다."
    )
    return note + (" 이번 턴에는 더 넘기지 말고 직접 답한다." if last else "")


def read_handoff(history: list[Message]) -> tuple[str, str] | None:
    """마지막 모델 응답의 handoff 호출에서 (넘길 에이전트, 이유). 없으면 None."""
    last = next((m for m in reversed(history) if isinstance(m, AssistantMessage)), None)
    call = next((c for c in reversed(last.tool_calls) if c.name == HANDOFF), None) if last else None
    return (str(call.arguments.get("agent", "")), str(call.arguments.get("reason", ""))) if call else None


class Session:
    """여러 턴의 대화를 이어 가는 세션. 이력은 메모리에만 있고 프로그램을 끄면 사라진다."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        agents: Callable[[str], AgentConfig] | None = None,
        agent: str = MAIN,
        on_handoff: Callable[[str, str, str], None] | None = None,
    ):
        self.config = config  # 지금 에이전트의 설정(모델, Tool, 지침, 최대 스텝). 넘기기가 없으면 모든 턴이 같은 설정이다.
        self.agents = agents  # 에이전트 이름 → 설정(C21). None이면 넘기기를 하지 않는다.
        self.agent = agent  # 지금 대화 중인 에이전트
        self.on_handoff = on_handoff  # (넘긴 에이전트, 넘겨받은 에이전트, 이유). 응용이 화면에 보인다(Q13).
        self.history: list[Message] = []
        self.last_usage = Usage()  # 마지막 턴의 토큰(/tokens)
        self.total_usage = Usage()  # 이번 세션의 누적 토큰(/tokens)
        self.trace_started = False  # 추적 기록에 세션 설정(session_start)을 썼는지. 첫 턴에 쓴다(C9 D27).

    def record_start(self) -> None:
        """세션 설정은 세션의 첫 턴 직전에 한 번만 기록한다(C8 D18). 질문 없이 끝난 세션은 추적 파일을 만들지 않는다."""
        tracer = self.config.tracer
        if tracer is None or self.trace_started:
            return
        tracer.record(
            "session_start",
            None,
            system_prompt=self.config.system_prompt,
            tools=[{"name": t.name, "description": t.description, "parameters": t.parameters} for t in self.config.tools],
            model=self.config.llm_config.model,
            max_steps=self.config.max_steps,
        )
        self.trace_started = True

    def ask(self, question: str, on_text_delta: Callable[[str], None] | None = None) -> AgentResult:
        """턴 하나를 처리한다. 모델 호출이 예외로 실패하면 예외가 그대로 전달되고 이력은 바뀌지 않는다.

        on_text_delta를 주면 모든 스텝의 답 글자 조각을 도착하는 대로 전달한다(스트리밍).
        이때 config.llm은 on_text_delta 키워드 인자를 받아야 한다(call_llm은 받는다). 이력 처리는 스트리밍 여부와 같다.
        """
        if on_text_delta is not None:
            config = replace(self.config, llm=partial(self.config.llm, on_text_delta=on_text_delta))
        else:
            config = self.config
        self.record_start()
        tracer, agent, usage = config.tracer, self.agent, Usage()
        result = run_agent(self._traced(config, agent, first=True), question, history=self.history)
        usage.add(result.usage)
        handoffs = 0
        while result.status == "handoff" and (target := read_handoff(result.history)):
            (to, reason), source = target, agent
            handoffs += 1
            if tracer is not None:
                tracer.record("handoff", None, source=agent, target=to, reason=reason)
            if self.on_handoff:
                self.on_handoff(agent, to, reason)
            agent = to
            config = self.agents(to)
            if handoffs >= MAX_HANDOFFS:  # 더 넘기지 못하게 handoff Tool을 빼고 이 에이전트가 답한다(Q11).
                config = replace(config, tools=[t for t in config.tools if t.name != HANDOFF])
            if on_text_delta is not None:
                config = replace(config, llm=partial(config.llm, on_text_delta=on_text_delta))
            note = handoff_note(source, to, reason, last=handoffs >= MAX_HANDOFFS)
            result = run_agent(self._traced(config, agent, first=False), note, history=result.history)
            usage.add(result.usage)
        # 예외가 나면 이 줄에 도달하지 않으므로 실패한 턴은 이력에 남지 않는다(넘기기 중이어도 지금 에이전트는 바뀌지 않는다).
        # max_steps로 끝난 턴도 모든 Tool 요청에 결과가 붙어 있어 이력이 일관되므로 그대로 유지한다.
        if agent != self.agent:
            self.switch(agent)
        self.history = result.history
        self.last_usage = usage
        self.total_usage.add(usage)
        return replace(result, usage=usage)

    def switch(self, agent: str) -> None:
        """지금 에이전트를 바꾼다(넘기기, /agent). 이력은 그대로 둔다."""
        if self.agents is None:
            raise ValueError("이 세션은 에이전트를 바꿀 수 없다")
        self.config = self.agents(agent)
        self.agent = agent

    def refresh(self) -> None:
        """지금 에이전트의 설정을 다시 만든다(장기 메모리가 바뀌었을 때, C9 D20)."""
        if self.agents is not None:
            self.config = self.agents(self.agent)

    def _traced(self, config: AgentConfig, agent: str, *, first: bool) -> AgentConfig:
        """추적 줄에 에이전트 이름을 붙인다(Q15). 넘겨받아 이어 도는 실행은 새 턴을 열지 않고 같은 턴에 남는다."""
        tracer = config.tracer
        if tracer is None or (first and agent == MAIN):
            return config
        if first:  # 다른 에이전트가 턴을 시작한다: 턴 번호는 여기서 세고, 실행은 같은 턴의 줄로 남긴다.
            tracer.begin_turn()
        name = None if agent == MAIN else agent
        return replace(config, tracer=replace(tracer, agent=name, parent="handoff"))

    def history_tokens(self) -> int:
        """지금 이력 크기: 다음 호출에 보낼 시스템 프롬프트, 이력, Tool 정의의 토큰 수."""
        request = LLMRequest(config=self.config.llm_config, messages=self.history, system_prompt=self.config.system_prompt, tools=self.config.tools)
        return request_tokens(request)

    def add_user_message(self, text: str) -> None:
        """모델을 부르지 않고 사용자 메시지를 이력에 넣는다(사용자가 스킬을 직접 지정할 때 쓴다)."""
        self.history.append(UserMessage(content=text))

    def reset(self) -> None:
        """이력을 비운다. 넘기기를 쓰는 세션이면 main으로 돌아온다(C21 4.2절). 에이전트 설정은 유지한다."""
        self.history = []
        if self.agents is not None and self.agent != MAIN:
            self.switch(MAIN)
