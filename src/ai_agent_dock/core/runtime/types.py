"""턴 처리에 쓰는 데이터 타입: 에이전트 설정(AgentConfig)과 턴 결과(AgentResult).

동작은 loop.py의 run_agent가 한다. 여기에는 모양만 정의한다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from ..hitl import Approval, AuditLog
from ..hooks import HookSet
from ..llm import LLMConfig, LLMRequest, LLMResponse, Message, Tool, ToolCall, call_llm
from ..observability import Tracer


@dataclass
class Usage:
    """토큰 사용량 합계. 입력은 캐시분을 포함한다."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    def add(self, response: LLMResponse | Usage) -> None:
        self.input_tokens += response.input_tokens or 0
        self.output_tokens += response.output_tokens or 0
        self.cached_input_tokens += response.cached_input_tokens or 0


@dataclass
class AgentResult:
    """run_agent의 결과."""

    answer: str | None
    status: str  # "completed"(최종 답변으로 종료) 또는 "max_steps"(최대 스텝 도달로 종료)
    steps: int  # 모델을 호출한 횟수
    history: list[Message]
    usage: Usage = field(default_factory=Usage)  # 이번 턴의 토큰 합계(/tokens가 쓴다)


@dataclass
class AgentConfig:
    """에이전트 한 개의 정적 설정. 턴마다 바뀌지 않는 값을 모아 run_agent와 Session에 함께 넘긴다."""

    llm_config: LLMConfig  # 모델 설정. 모든 스텝의 요청에 실린다.
    tools: list[Tool] = field(default_factory=list)
    system_prompt: str | None = None  # 진입점이 에이전트 정의(AGENT.md)로 조립해 넣는다(C12, core/agents의 AgentTeam)
    max_steps: int | None = None  # 한 턴에서 모델을 호출하는 횟수 상한. None이면 환경 변수, 없으면 기본값
    llm: Callable[[LLMRequest], LLMResponse] = call_llm  # 모델 호출 함수. 테스트에서는 가짜 모델을 넣는다.
    approve: Callable[[ToolCall], Approval] | None = None  # 높음 등급 Tool을 실행하기 전에 묻는 함수. None이면 묻지 않는다.
    audit: AuditLog | None = None  # 실행 기록(감사 로그). None이면 기록하지 않는다.
    tracer: Tracer | None = None  # 추적 기록(세션 설정, 턴, 스텝, LLM 호출, Tool 실행). None이면 기록하지 않는다.
    safety: SafetyChecks | None = None  # 항상 거는 고정 검사(C11 D19). 진입점이 core/safety의 with_safety로 넣는다.
    hooks: HookSet = field(default_factory=HookSet)  # 사용자가 설치한 Hook(C11 D20). hooks.json 또는 코드로 더한다.


class SafetyChecks(Protocol):
    """Safety가 가져야 할 함수 모양. runtime은 core/safety를 import하지 않고 이 모양만 안다(C11 D19).

    루프는 before_tool에서 Safety를 Hook보다 먼저, after_tool과 after_answer에서는 Hook 뒤 마지막에 부른다(C11 D22).
    """

    def before_tool(self, call: ToolCall, tool: Tool) -> str | None: ...  # 막을 이유. 통과면 None
    def after_tool(self, call: ToolCall, tool: Tool, content: str) -> str: ...  # 모델에게 넘길 결과
    def after_answer(self, answer: str) -> str: ...  # 이력에 넣을 답
