"""워크플로들이 함께 쓰는 도구(C15 Q20). 데이터(경로 표, 메뉴)는 두지 않는다. 표는 워크플로마다 자기 파일에 둔다.

워크플로끼리는 서로 import하지 않고 이 파일만 가져다 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..core.agents import MAIN
from ..core.llm import Tool, model_config
from ..core.orchestration import WorkflowContext
from ..core.retrieval import make_search_docs_tool
from ..core.runtime.types import AgentConfig
from ..core.tools import Node, tools_for


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
    """동시에 도는 가지 하나의 설정. 추적 줄과 응용의 call message에 가지 이름(docs, judge1 등)을 붙인다(C15 Q25).

    추적은 턴 번호를 새로 세지 않고 워크플로의 턴 안에 남는다(parent에 워크플로 이름).
    """
    tracer = config.tracer.child(name, workflow) if config.tracer else None
    llm = ctx.team.wrap_llm(name, ctx.base.llm) if ctx.team.wrap_llm else config.llm
    return replace(config, tracer=tracer, llm=llm)


def docs_tool(ctx: WorkflowContext, seen: set[str]) -> Tool:
    """검색한 출처를 seen에 모으는 search_docs. 출처 검사를 하는 노드가 main의 search_docs 대신 쓴다.
    새로 만든 Tool이라 응용의 call message(ctx.wrap_tool)을 여기서 붙인다(C17 Q27)."""
    return ctx.wrap_tool(make_search_docs_tool(ctx.knowledge.store, seen))


def keep_results(tools: list[Tool], sink: list[tuple[str, str]]) -> list[Tool]:
    """Tool을 감싸 실행할 때마다 (Tool 이름, 결과)를 sink에 모은다. 다음 단계나 평가자가 근거로 본다(C13, C17 Q19)."""

    def keep(tool: Tool) -> Tool:
        def run(**arguments):
            result = tool.func(**arguments)
            sink.append((tool.name, str(result)))
            return result

        return replace(tool, func=run)

    return [keep(t) for t in tools]


def writer_tools(ctx: WorkflowContext, evidence: list[tuple[str, str]], seen: set[str]) -> list[Tool]:
    """답 전체를 쓰는 노드의 Tool. evaluator-optimizer(C17)와 reflexion(C19)이 함께 쓴다.

    main의 Tool에 web 에이전트의 웹 Tool을 더한다(C17 Q8-1, Q26). 문서 색인이 있으면 검색한 출처를 seen에 모으는 search_docs로 바꾼다.
    부른 Tool의 결과는 evidence에 모인다(평가자의 근거).
    """
    web = ctx.team.agents.get("web")
    tools = tools_for(ctx.base.tools, patterns=ctx.team.agents[MAIN].tools, node=Node(web=tuple(web.tools) if web else ()))
    if ctx.knowledge:
        search = docs_tool(ctx, seen)
        tools = [t for t in tools if t.name != search.name] + [search]
    return keep_results(tools, evidence)


def clean_label(label: str) -> str:
    """에이전트가 낸 이름 하나를 정리한다. 대소문자, 앞뒤 공백, 따옴표, 마침표만 정리한다(C14 Q4)."""
    return label.strip(" \t\n\"'`.").lower()


# --- worker 목록과 가지 이름. orchestrator-workers(C16)와 plan-and-execute(C18)가 함께 쓴다(C18 Q13, Q27). ---

FALLBACK = "general"  # 목록에 없는 worker 이름의 작업을 맡고(C16 Q14), 작업을 못 읽으면 원래 질문에 답한다.
# 같은 Tool이 두 worker에 겹치지 않게 researcher에서 웹 Tool을 뺐다(C16 Q12).
WORKERS = [
    AgentSlot("researcher", "이 프로젝트 문서를 조사한다(필요하면 샌드박스 파일도 읽는다)", agent="researcher", model="", tools=("search_docs", "files__read_*")),
    AgentSlot("web", "최신 정보, 외부 웹사이트를 조사한다", agent="web", model="", tools=()),
    AgentSlot("utility", "계산, 지금 시각과 날짜", agent=MAIN, model="openai:gpt-6-luna", tools=("calculate", "get_current_time")),
]
MAX_TASKS = 5
ONLY_MINE = "원래 질문은 목적을 알려 주려고 넘긴다. 맡은 작업만 하고, 나머지는 다른 worker가 맡는다."  # C16 Q22


def branch_name(task: dict) -> str:
    """가지 이름: worker-작업 번호(researcher-1, web-2). 추적, progress message, 합치기 프롬프트에 쓴다(C16 Q15)."""
    return f"{task['worker']}-{task['n']}"
