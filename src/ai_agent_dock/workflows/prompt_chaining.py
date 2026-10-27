"""prompt-chaining 워크플로(C13): 개요 작성 → gate(코드 검사) → 본문 작성.

순서는 이 그래프가 정한다. LLM은 단계마다 에이전트(run_agent)로 부르고, 단계마다 다른 것은 사용자 프롬프트와 Tool이다(C13 Q2).
gate는 Tool이 아니라 노드다. 모델이 건너뛸 수 없다. 실패하면 고쳐 쓰지 않고 멈춘다(Q4). 사람 승인은 없다(Q5).
본문 단계는 Tool을 받지 않는다. gate가 확인한 개요와 검색 결과 밖으로 나가지 못한다(Q2-1).
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ..core.orchestration import Workflow, WorkflowContext, ask_agent
from ..core.retrieval import check_citations, make_search_docs_tool
from .answer_check import EXCLUDED_TOOLS, unavailable

MIN_ITEMS, MAX_ITEMS = 2, 5  # 개요 항목 수(Q3). 짧은 문서는 2개가 자연스러워 C14에서 3을 2로 내렸다
ITEM = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(.+)$")  # 줄 머리의 -, *, 1. 을 항목으로 센다


class State(TypedDict, total=False):
    workflow: str
    question: str
    started: str
    status: str  # running, completed, gate_failed
    answer: str
    outline: str
    evidence: str  # 개요 단계의 search_docs 결과. 본문 단계의 근거다.
    seen: list[str]  # 개요 단계에서 검색한 출처
    problem: str  # gate 실패 이유. 비어 있으면 통과


def outline_prompt(question: str) -> str:
    return (
        f"질문: {question}\n\n"
        f"search_docs로 문서를 찾아 이 질문에 답할 개요를 {MIN_ITEMS}~{MAX_ITEMS}개 항목으로 써라. "
        "한 줄에 항목 하나, '- 항목 내용 [문서 제목, 위치]' 형식이다. 항목마다 검색 결과의 출처를 붙인다. "
        "개요만 쓰고 본문, 머리말, 맺음말은 쓰지 않는다. 근거를 찾지 못하면 찾지 못했다고만 쓴다."
    )


def write_prompt(state: State) -> str:
    return (
        f"질문: {state['question']}\n\n[개요]\n{state.get('outline', '')}\n\n[근거]\n{state.get('evidence', '')}\n\n"
        "위 개요의 항목 순서대로 질문에 대한 답을 써라. 근거에 있는 내용만 쓰고, 항목에 붙은 출처를 그대로 붙인다. "
        "개요나 근거에 대한 설명, 사용자에게 되묻는 문장은 쓰지 않는다."
    )


def outline_items(text: str) -> list[str]:
    return [m.group(1) for line in text.splitlines() if (m := ITEM.match(line))]


def gate_problem(outline: str, seen: set[str]) -> str:
    """gate(Q3). 통과하면 빈 문자열."""
    items = outline_items(outline)
    if not MIN_ITEMS <= len(items) <= MAX_ITEMS:
        return f"개요 항목 {len(items)}개({MIN_ITEMS}~{MAX_ITEMS}개여야 한다)"
    uncited = [item for item in items if not check_citations(item, seen).cited]
    if uncited:
        return f"출처 없는 항목 {len(uncited)}개"
    unknown = check_citations(outline, seen).unknown
    if unknown:
        return f"검색에 없던 출처: {', '.join(unknown)}"
    return ""


def build(ctx: WorkflowContext) -> StateGraph:
    def outline(state: State) -> State:
        ctx.progress("개요 작성")
        seen: set[str] = set()
        results: list[str] = []
        search = make_search_docs_tool(ctx.knowledge.store, seen)

        def search_and_keep(query: str) -> str:  # 검색 결과 글을 본문 단계의 근거로 모아 둔다.
            results.append(search.func(query=query))
            return results[-1]

        tools = [t for t in ctx.config.tools if t.name not in EXCLUDED_TOOLS | {search.name}] + [replace(search, func=search_and_keep)]
        text = ask_agent(replace(ctx.config, tools=tools), outline_prompt(state["question"]))
        return {"outline": text, "evidence": "\n\n".join(results), "seen": sorted(seen)}

    def gate(state: State) -> State:
        outline_text = state.get("outline", "")
        problem = gate_problem(outline_text, set(state.get("seen", [])))
        ctx.progress(f"gate 실패: {problem}" if problem else f"gate 통과 (항목 {len(outline_items(outline_text))}개)")
        if problem:
            return {"problem": problem, "status": "gate_failed", "answer": f"개요가 검사를 통과하지 못해 본문을 쓰지 않았습니다: {problem}"}
        return {"problem": ""}

    def write(state: State) -> State:
        ctx.progress("본문 작성")
        return {"status": "completed", "answer": ask_agent(replace(ctx.config, tools=[]), write_prompt(state))}

    graph = StateGraph(State)
    for name, node in [("outline", outline), ("gate", gate), ("write", write)]:
        graph.add_node(name, node)
    graph.add_edge(START, "outline")
    graph.add_edge("outline", "gate")
    graph.add_conditional_edges("gate", lambda state: END if state.get("problem") else "write", ["write", END])
    graph.add_edge("write", END)
    return graph


WORKFLOW = Workflow(
    name="prompt-chaining",
    description="문서를 찾아 개요를 쓰고, 코드가 개요를 검사(gate)한 뒤 통과하면 본문을 쓴다",
    build=build,
    unavailable=unavailable,
)
