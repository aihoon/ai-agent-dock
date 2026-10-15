"""대화 세션: 한 대화의 이력을 보관하고, 턴(질문 하나)마다 Agent Loop를 돌려 이력을 이어 준다.

세션은 이력과 에이전트 설정(AgentSettings)만 소유한다. 턴 처리는 run_agent가 하고, 세션은 그 결과의 이력을 다음 턴으로 넘길 뿐이다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial

from ..llm import Message, UserMessage
from ..runtime.loop import run_agent
from ..runtime.types import DEFAULT_SYSTEM_PROMPT, AgentResult, AgentSettings

__all__ = ["DEFAULT_SYSTEM_PROMPT", "Session"]  # DEFAULT_SYSTEM_PROMPT는 기존 import 경로를 위해 여기서도 내보낸다.


class Session:
    """여러 턴의 대화를 이어 가는 세션. 이력은 메모리에만 있고 프로그램을 끄면 사라진다."""

    def __init__(self, settings: AgentSettings):
        self.settings = settings  # 모든 턴이 같은 설정(모델, Tool, 지침, 최대 스텝)을 쓴다.
        self.history: list[Message] = []

    def ask(self, question: str, on_text_delta: Callable[[str], None] | None = None) -> AgentResult:
        """턴 하나를 처리한다. 모델 호출이 예외로 실패하면 예외가 그대로 전달되고 이력은 바뀌지 않는다.

        on_text_delta를 주면 모든 스텝의 답 글자 조각을 도착하는 대로 전달한다(스트리밍).
        이때 settings.llm은 on_text_delta 키워드 인자를 받아야 한다(call_llm은 받는다). 이력 처리는 스트리밍 여부와 같다.
        """
        if on_text_delta is not None:
            settings = replace(self.settings, llm=partial(self.settings.llm, on_text_delta=on_text_delta))
        else:
            settings = self.settings
        result = run_agent(settings, question, history=self.history)
        # 예외가 나면 이 줄에 도달하지 않으므로 실패한 턴은 이력에 남지 않는다.
        # max_steps로 끝난 턴도 모든 Tool 요청에 결과가 붙어 있어 이력이 일관되므로 그대로 유지한다.
        self.history = result.history
        return result

    def add_user_message(self, text: str) -> None:
        """모델을 부르지 않고 사용자 메시지를 이력에 넣는다(사용자가 스킬을 직접 지정할 때 쓴다)."""
        self.history.append(UserMessage(content=text))

    def reset(self) -> None:
        """이력만 비운다. 에이전트 설정(모델, Tool, 지침)은 유지한다."""
        self.history = []
