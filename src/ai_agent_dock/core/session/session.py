"""대화 세션: 한 대화의 이력을 보관하고, 턴(질문 하나)마다 Agent Loop를 돌려 이력을 이어 준다.

세션은 이력만 소유한다. 턴 처리는 run_agent가 하고, 세션은 그 결과의 이력을 다음 턴으로 넘길 뿐이다.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

from ..llm import LLMResponse, Message, Tool, call_llm
from ..runtime.loop import AgentResult, run_agent

# 일반 대화형 어시스턴트의 최소 기본 지침. 인자로 덮어쓸 수 있고, 이력에는 넣지 않고 호출마다 따로 전달한다.
DEFAULT_SYSTEM_PROMPT = (
    "당신은 도움이 되는 어시스턴트다. 사용자의 언어로 답한다. "
    "모르거나 확인할 수 없는 것은 지어내지 말고 모른다고 답한다. "
    "계산, 시간 조회처럼 Tool로 확인할 수 있는 것은 추측하지 말고 Tool을 사용한다."
)


class Session:
    """여러 턴의 대화를 이어 가는 세션. 이력은 메모리에만 있고 프로그램을 끄면 사라진다."""

    def __init__(
        self,
        tools: list[Tool],
        *,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        provider: str | None = None,
        max_steps: int | None = None,
        llm: Callable[[list[Message], list[Tool], str | None], LLMResponse] | None = None,
    ):
        if llm is not None and provider is not None:
            raise ValueError("llm과 provider는 함께 쓸 수 없다(llm을 직접 넘기면 제공자 선택은 그 함수의 몫이다)")
        self.tools = list(tools)
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        # 제공자 선택: provider 인자 > 환경 변수 LLM_PROVIDER > 기본값(call_llm이 정한다).
        self.llm = llm or partial(call_llm, provider=provider)
        self.history: list[Message] = []

    def ask(self, question: str, on_text_delta: Callable[[str], None] | None = None) -> AgentResult:
        """턴 하나를 처리한다. 모델 호출이 예외로 실패하면 예외가 그대로 전달되고 이력은 바뀌지 않는다.

        on_text_delta를 주면 모든 스텝의 답 글자 조각을 도착하는 대로 전달한다(스트리밍).
        이때 self.llm은 on_text_delta 키워드 인자를 받아야 한다(call_llm은 받는다). 이력 처리는 스트리밍 여부와 같다.
        """
        llm = self.llm if on_text_delta is None else partial(self.llm, on_text_delta=on_text_delta)
        result = run_agent(
            question,
            self.tools,
            history=self.history,
            system_prompt=self.system_prompt,
            max_steps=self.max_steps,
            llm=llm,
        )
        # 예외가 나면 이 줄에 도달하지 않으므로 실패한 턴은 이력에 남지 않는다.
        # max_steps로 끝난 턴도 모든 Tool 요청에 결과가 붙어 있어 이력이 일관되므로 그대로 유지한다.
        self.history = result.history
        return result

    def reset(self) -> None:
        """이력만 비운다. 시스템 프롬프트, Tool, 제공자 설정은 유지한다."""
        self.history = []
