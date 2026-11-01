"""sectioning 워크플로(C15): selector 에이전트가 경로 메뉴에서 1~3개를 고르고 경로마다 하위 질문을 쓰면,
고른 경로 에이전트들이 동시에 답하고, aggregator 에이전트가 한 답으로 합친다.

메뉴(MENU)는 이 파일의 표다. routing의 경로 표와 따로 두고 서로 import하지 않는다(Q20).
항목마다 에이전트, Tool, 모델을 정하고 하위 질문 한 줄만 넘긴다(Q5, Q15, Q18).
몇 개를 고를지 미리 모르므로 LangGraph의 Send로 펼친다(Q4). 한 번 나누고 끝난다(다시 나누기는 C16).
가지는 승인이 필요 없는 Tool만 쓴다. 두 가지가 동시에 사람에게 묻지 않게 한다(5.3).
"""

from __future__ import annotations

import operator
from dataclasses import replace
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..core.agents import MAIN
from ..core.orchestration import Workflow, WorkflowContext, ask_agent
from ..core.retrieval import check_citations
from .common import AgentSlot, branch_config, clean_label, slot_config

SELECTOR = "selector"
AGGREGATOR = "aggregator"
FALLBACK = "general"  # 아무것도 못 골랐을 때 채팅의 main 그대로 답한다.
# chat은 Tool이 없어 나눌 이유가 없다. 가지는 승인을 묻지 않는 Tool만 받으므로 files는 읽기만 한다(5.3).
MENU = [
    AgentSlot("docs", "이 프로젝트 문서에서 찾을 부분", agent=MAIN, model="", tools=("search_docs",)),
    AgentSlot("web", "최신 정보, 뉴스, 외부 웹사이트에서 찾을 부분", agent="web", model="", tools=()),
    AgentSlot("utility", "계산, 지금 시각과 날짜가 필요한 부분", agent=MAIN, model="openai:gpt-6-luna", tools=("calculate", "get_current_time")),
    AgentSlot("files", "샌드박스 파일을 읽거나 목록을 볼 부분(쓰기는 하지 않는다)", agent=MAIN, model="", tools=("files__*",)),
]
MAX_ROUTES = 3


class State(TypedDict, total=False):
    workflow: str
    question: str
    started: str
    status: str  # running, completed
    label: str  # selector가 실제로 낸 답
    routes: list[str]  # 고른 경로(못 골랐으면 [general])
    assigned: list[dict]  # 경로마다 맡긴 하위 질문 {route, question}
    sections: Annotated[list[dict], operator.add]  # 경로마다 {route, question, answer}. 동시에 들어오는 결과를 모은다.
    answer: str


def selector_prompt(question: str) -> str:
    menu = "\n".join(f"- {s.name}: {s.description}" for s in MENU)
    return f"경로 메뉴:\n{menu}\n\n질문: {question}"


def read_sections(label: str, question: str) -> list[tuple[str, str]]:
    """selector의 답을 (경로, 하위 질문) 목록으로. 메뉴에 없는 이름과 같은 이름의 두 번째 줄은 버리고 3개까지만(Q18)."""
    names = {s.name for s in MENU}
    chosen: dict[str, str] = {}
    for line in label.splitlines():
        name, _, sub = line.partition(":")
        name = clean_label(name)
        if name in names and name not in chosen:
            chosen[name] = sub.strip() or question  # 질문을 비우면 원래 질문을 넘긴다.
    return list(chosen.items())[:MAX_ROUTES]


def aggregator_prompt(question: str, sections: list[dict]) -> str:
    parts = [f"[{s['route']}] 맡은 질문: {s['question']}\n답:\n{s['answer']}" for s in sections]
    return "\n\n".join([f"원래 질문: {question}", *parts])


def build(ctx: WorkflowContext) -> StateGraph:
    slots = {s.name: s for s in MENU}

    def select(state: State) -> State:
        ctx.progress(f"고르기: selector가 메뉴 {len(MENU)}개 중에서 고른다")
        try:
            label = ask_agent(ctx.team.config_for(SELECTOR, ctx.base), selector_prompt(state["question"]))
        except Exception as e:  # 고르기에 실패해도 general로 답한다.
            label = f"오류: {type(e).__name__}"
        chosen = read_sections(label, state["question"])
        if not chosen:
            ctx.progress(f'고르기 실패("{label}") → {FALLBACK}')
            return {"label": label, "routes": [FALLBACK]}
        listed = ", ".join(f"{name}({sub})" for name, sub in chosen)
        if len(chosen) == 1:  # Q22
            ctx.progress(f"고른 경로 1개: {listed} → 합치지 않고 이 답을 쓴다")
        else:
            ctx.progress(f"고른 경로 {len(chosen)}개: {listed}")
            ctx.progress("동시 실행: " + ", ".join(name for name, _ in chosen))
        return {"label": label, "routes": [name for name, _ in chosen], "assigned": [{"route": n, "question": q} for n, q in chosen]}

    def fan_out(state: State):
        if state["routes"] == [FALLBACK]:
            return "general"
        return [Send("run_route", section) for section in state["assigned"]]

    def run_route(section: dict) -> State:
        route = slots[section["route"]]
        config = slot_config(ctx, route)
        tools = [t for t in config.tools if t.read_only and t.trusted]  # 승인을 묻는 Tool은 뺀다(5.3).
        try:
            answer = ask_agent(branch_config(ctx, replace(config, tools=tools), route.name, "sectioning"), section["question"])
            ctx.progress(f"{route.name} 완료")
        except Exception as e:  # 한 가지가 실패해도 합치기를 계속한다(Q7).
            answer = f"실패: {type(e).__name__}: {e}"
            ctx.progress(f"{route.name} 실패({type(e).__name__}: {e})")
        return {"sections": [{**section, "answer": answer}]}

    def aggregate(state: State) -> State:
        done = sorted(state["sections"], key=lambda s: state["routes"].index(s["route"]))  # 고른 순서대로(가지는 끝나는 순서대로 들어온다)
        if len(done) == 1:  # 하나만 골랐으면 그 답이 최종 답이다.
            return {"status": "completed", "answer": done[0]["answer"]}
        ctx.progress("합치기: aggregator 에이전트")
        answer = ask_agent(ctx.team.config_for(AGGREGATOR, ctx.base), aggregator_prompt(state["question"], done))
        unknown = check_citations(answer, ctx.knowledge.seen).unknown if ctx.knowledge else []  # URL 출처는 보지 않는다(5.1).
        if unknown:
            ctx.progress(f"※ 검색 결과에 없는 출처: {', '.join(unknown)}")
        return {"status": "completed", "answer": answer}

    def general(state: State) -> State:
        ctx.progress(f"{FALLBACK} 경로 실행")
        return {"status": "completed", "answer": ask_agent(ctx.config, state["question"])}

    graph = StateGraph(State)
    graph.add_node("select", select)
    graph.add_node("run_route", run_route)
    graph.add_node("aggregate", aggregate)
    graph.add_node("general", general)
    graph.add_edge(START, "select")
    graph.add_conditional_edges("select", fan_out, ["run_route", "general"])
    graph.add_edge("run_route", "aggregate")
    graph.add_edge("aggregate", END)
    graph.add_edge("general", END)
    return graph


def unavailable(ctx: WorkflowContext) -> str | None:
    if ctx.team is None or ctx.base is None:
        return "에이전트 정의가 없어 sectioning을 쓸 수 없습니다."
    needed = {SELECTOR, AGGREGATOR, MAIN, *(s.agent for s in MENU)}
    missing = sorted(needed - set(ctx.team.agents))
    if missing:
        return f"sectioning에 필요한 에이전트 정의가 없습니다: {', '.join(missing)} (agents/이름/AGENT.md)"
    return None


WORKFLOW = Workflow(
    name="sectioning",
    description="selector가 경로를 1~3개 고르고 하위 질문을 쓰면, 경로들이 동시에 답하고 aggregator가 합친다",
    build=build,
    unavailable=unavailable,
)
