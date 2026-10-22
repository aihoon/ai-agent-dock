"""answer-check 워크플로(C10 D3): 답 작성 → 출처 검사 → 실패하면 고쳐 쓰기(상한까지) → 사람 승인.

순서와 되돌리기는 이 그래프가 정한다. 에이전트(독)는 '답 작성' 노드 하나다.
상태에는 글자와 숫자만 담는다(C10 D12). 출처 집합(seen)은 실행마다 새로 모은다(C10 D20).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..core.orchestration import Workflow, WorkflowContext, ask_agent
from ..core.retrieval import check_citations, make_search_docs_tool

NO_EVIDENCE = "관련 문서에서 근거를 찾지 못했습니다."
# 되묻기는 노드 안에서 일어나 체크포인트에 남지 않는다. 방향은 승인 단계의 다르게 요청으로 바꾼다(C10 D23).
# 맡기기(delegate, C12)도 뺀다. 맡은 에이전트의 검색은 이 노드의 출처 집합(seen)에 남지 않아 출처 검사가 틀어진다.
EXCLUDED_TOOLS = {"ask_user", "delegate"}


class State(TypedDict, total=False):
    workflow: str
    question: str
    started: str
    status: str  # running, approved, rejected, no_evidence
    answer: str
    draft: str
    seen: list[str]  # 이번 답을 쓰며 검색한 출처
    problem: str  # 출처 검사 실패 이유나 사람의 다르게 요청 이유. 비어 있으면 통과
    feedback: list[str]  # 지금까지 고쳐 쓰게 한 이유
    revisions: int
    decision: str  # 승인 단계의 답: approve, reject, revise


def prompt_for(state: State) -> str:
    """처음에는 질문만, 고쳐 쓸 때는 질문, 이전 답, 고칠 점을 한 메시지로 넘긴다(C10 D12)."""
    if not state.get("feedback"):
        return state["question"]
    reasons = "\n".join(f"- {r}" for r in state["feedback"])
    return (
        f"{state['question']}\n\n[이전 답]\n{state.get('draft', '')}\n\n[고칠 점]\n{reasons}\n\n"
        "문서를 다시 검색해서 출처를 붙여 질문에 대한 답을 새로 써라. 답만 쓴다. "
        "고쳤다는 말, 이전 답이나 고칠 점에 대한 설명, 사용자에게 되묻는 문장은 쓰지 않는다."
    )


def citation_problem(answer: str, seen: set[str]) -> str:
    """출처 검사(C10 D20). 통과하면 빈 문자열."""
    check = check_citations(answer, seen)
    if not check.cited:
        return "출처 없음"
    if check.unknown:
        return f"검색에 없던 출처: {', '.join(check.unknown)}"
    return ""


def build(ctx: WorkflowContext) -> StateGraph:
    def write(state: State) -> State:
        feedback = state.get("feedback")
        ctx.progress(f"답 다시 작성 (고칠 점: {feedback[-1]})" if feedback else "답 작성")
        seen: set[str] = set()
        search = make_search_docs_tool(ctx.knowledge.store, seen)
        tools = [t for t in ctx.config.tools if t.name not in EXCLUDED_TOOLS | {search.name}] + [search]
        draft = ask_agent(replace(ctx.config, tools=tools), prompt_for(state))
        return {"draft": draft, "seen": sorted(seen)}

    def check(state: State) -> State:
        seen = set(state.get("seen", []))
        problem = citation_problem(state.get("draft", ""), seen)
        ctx.progress(f"출처 검사: {problem or '통과 → 사람 승인'} (검색한 출처 {len(seen)}개)")
        return {"problem": problem}

    def revise(state: State) -> State:
        revisions = state.get("revisions", 0) + 1
        ctx.progress(f"고쳐 쓰기 {revisions}/{ctx.max_revisions}")
        return {"revisions": revisions, "feedback": [*state.get("feedback", []), state["problem"]]}

    def give_up(state: State) -> State:
        ctx.progress(f"고쳐 쓰기 상한({ctx.max_revisions}번)에 이르러 근거 없음으로 끝냅니다")
        return {"status": "no_evidence", "answer": f"{NO_EVIDENCE} ({state.get('problem', '')})"}

    def approve(state: State) -> State:
        # interrupt는 노드 맨 앞에 둔다. 재개하면 이 노드를 처음부터 다시 실행하기 때문이다(C10 V7).
        answer = interrupt({"draft": state.get("draft", ""), "revisions": state.get("revisions", 0)})
        kind, _, reason = str(answer).partition(":")
        if kind == "approve":
            return {"decision": kind, "status": "approved", "answer": state.get("draft", "")}
        if kind == "reject":
            return {"decision": kind, "status": "rejected"}
        return {"decision": "revise", "problem": f"사용자 요청: {reason.strip() or '다르게 써 달라'}"}

    def can_revise(state: State) -> bool:
        return state.get("revisions", 0) < ctx.max_revisions

    def after_check(state: State) -> str:
        if not state.get("problem"):
            return "approve"
        return "revise" if can_revise(state) else "give_up"

    def after_approve(state: State) -> str:
        if state.get("decision") != "revise":
            return END
        return "revise" if can_revise(state) else "give_up"

    graph = StateGraph(State)
    for name, node in [("write", write), ("check", check), ("revise", revise), ("give_up", give_up), ("approve", approve)]:
        graph.add_node(name, node)
    graph.add_edge(START, "write")
    graph.add_edge("write", "check")
    graph.add_conditional_edges("check", after_check, ["approve", "revise", "give_up"])
    graph.add_edge("revise", "write")
    graph.add_edge("give_up", END)
    graph.add_conditional_edges("approve", after_approve, ["revise", "give_up", END])
    return graph


def unavailable(ctx: WorkflowContext) -> str | None:
    if ctx.knowledge is None:
        return "문서 색인이 없습니다. uv run python scripts/index_knowledge.py 를 먼저 실행하세요."
    return None


WORKFLOW = Workflow(
    name="answer-check",
    description="문서를 검색해 출처를 붙여 답하고, 출처가 틀리면 고쳐 쓴 뒤 사람 승인을 받는다",
    build=build,
    unavailable=unavailable,
)
