"""워크플로들이 함께 쓰는 도구(C15 Q20). 데이터(경로 표, 메뉴)는 두지 않는다. 표는 워크플로마다 자기 파일에 둔다.

워크플로끼리는 서로 import하지 않고 이 파일만 가져다 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..core.agents import MAIN
from ..core.llm import model_config
from ..core.orchestration import WorkflowContext
from ..core.runtime.types import AgentConfig


@dataclass(frozen=True)
class AgentSlot:
    """에이전트 자리 하나: 어떤 에이전트를 어떤 Tool과 모델로 쓸지. routing의 경로 표와 sectioning의 메뉴가 쓴다."""

    name: str
    description: str  # 고르는 에이전트(router, selector)에게 보여 줄 설명
    agent: str  # agents/<agent>/AGENT.md. 본문이 시스템 프롬프트가 된다.
    model: str  # "제공자:모델". AGENT.md의 model보다 우선한다(C14 Q3). ""이면 AGENT.md의 model
    tools: tuple[str, ...]  # AGENT.md의 Tool 중 더 좁힐 패턴(교집합, C14 Q2-3). ()이면 에이전트의 Tool 그대로(AGENT.md의 []와 달리 "Tool 없음"이 아니다)


def slot_config(ctx: WorkflowContext, slot: AgentSlot) -> AgentConfig:
    """자리의 에이전트 설정: 에이전트, Tool, 모델."""
    if slot.agent == MAIN and not slot.tools:
        config = ctx.config  # 채팅의 main 그대로(스킬, 메모리, delegate 포함)
    else:
        config = ctx.team.config_for(slot.agent, ctx.base, only=slot.tools or None)
    if slot.model:
        config = replace(config, llm_config=model_config(slot.model, ctx.base.llm_config))
    return config


def branch_config(ctx: WorkflowContext, config: AgentConfig, name: str, workflow: str) -> AgentConfig:
    """동시에 도는 가지 하나의 설정. 추적 줄과 응용의 진행 줄에 가지 이름(docs, judge1 등)을 붙인다(C15 Q25).

    추적은 턴 번호를 새로 세지 않고 워크플로의 턴 안에 남는다(parent에 워크플로 이름).
    """
    tracer = config.tracer.child(name, workflow) if config.tracer else None
    llm = ctx.team.wrap_llm(name, ctx.base.llm) if ctx.team.wrap_llm else config.llm
    return replace(config, tracer=tracer, llm=llm)


def clean_label(label: str) -> str:
    """에이전트가 낸 이름 하나를 정리한다. 대소문자, 앞뒤 공백, 따옴표, 마침표만 정리한다(C14 Q4)."""
    return label.strip(" \t\n\"'`.").lower()
