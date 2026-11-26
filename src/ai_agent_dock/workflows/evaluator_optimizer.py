"""evaluator-optimizer 워크플로(C17): main이 답을 쓰고, evaluator 에이전트가 기준표로 평가하고, 실패한 기준의 이유로 고쳐 쓴다.

모든 기준을 통과하거나 고쳐 쓰기 상한(WORKFLOW_MAX_REVISIONS)에 닿으면 끝난다. 사람 승인은 없다(Q5).
answer-check(C10)와 같은 루프 모양이고, 평가자가 코드가 아니라 LLM이다. 코드 출처 검사는 참고로만 남긴다(Q11).
평가자는 작성자가 부른 모든 Tool의 결과를 근거로 본다(Q8-1). 기준표와 평가자 도구는 core/evals/rubric.py에 있다(Q14).
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ..core.evals.rubric import EVALUATOR, FAIL, Verdict, evaluator_prompt, failed, read_verdicts, verdict_line
from ..core.orchestration import Workflow, WorkflowContext, ask_agent
from ..core.retrieval import check_citations
from .common import writer_tools

NAME = "evaluator-optimizer"


class State(TypedDict, total=False):
    workflow: str
    question: str
    started: str
    status: str  # running, passed, limit, eval_failed
    draft: str
    evidence: list[dict]  # 작성자가 부른 Tool의 {tool, result}
    seen: list[str]  # 이번 답을 쓰며 검색한 문서 출처(코드 출처 검사용)
    verdicts: list[dict]  # 마지막 평가의 {criterion, result, reason}
    code_check: str  # 코드 출처 검사 결과(참고)
    feedback: list[str]  # 이번에 고칠 점("기준: 이유")
    revisions: int
    answer: str


def write_prompt(state: State) -> str:
    """처음에는 질문만, 고쳐 쓸 때는 질문, 이전 답, 고칠 점(answer-check와 같은 모양, 3.6절)."""
    if not state.get("feedback"):
        return state["question"]
    reasons = "\n".join(f"- {r}" for r in state["feedback"])
    return (
        f"{state['question']}\n\n[이전 답]\n{state.get('draft', '')}\n\n[고칠 점]\n{reasons}\n\n"
        "필요하면 다시 찾아서 질문에 대한 답을 새로 써라. 답만 쓴다. "
        "고쳤다는 말, 이전 답이나 고칠 점에 대한 설명, 되묻는 문장은 쓰지 않는다."
    )


def code_check(answer: str, seen: set[str]) -> str:
    check = check_citations(answer, seen)
    if not check.cited:
        return "출처 없음"
    return f"검색에 없던 출처: {', '.join(check.unknown)}" if check.unknown else "통과"


def build(ctx: WorkflowContext) -> StateGraph:
    def write(state: State) -> State:
        feedback = state.get("feedback")
        ctx.progress(f"답 다시 작성 (고칠 점: {', '.join(r.split(':')[0] for r in feedback)})" if feedback else "답 작성")
        seen: set[str] = set()  # 검색한 출처(코드 출처 검사용)
        evidence: list[tuple[str, str]] = []
        draft = ask_agent(replace(ctx.config, tools=writer_tools(ctx, evidence, seen)), write_prompt(state))  # 부른 Tool의 결과가 근거다.
        return {"draft": draft, "evidence": [{"tool": tool, "result": result} for tool, result in evidence], "seen": sorted(seen)}

    def evaluate(state: State) -> State:
        draft = state.get("draft", "")
        evidence = [(e["tool"], e["result"]) for e in state.get("evidence", [])]
        try:
            label = ask_agent(ctx.team.config_for(EVALUATOR, ctx.base), evaluator_prompt(state["question"], draft, evidence))
        except Exception as e:  # 평가하지 못하면 고쳐 쓰지 않고 지금 답으로 끝낸다(Q12).
            ctx.progress(f"평가 실패({type(e).__name__}: {e}) → 지금 답으로 끝냅니다")
            return {"status": "eval_failed", "answer": draft}
        verdicts = read_verdicts(label)
        ctx.progress(f"평가: {verdict_line(verdicts)}")
        if not ctx.knowledge:
            checked = "문서 색인 없음(참고)"
        elif not any(tool == "search_docs" for tool, _ in evidence):  # 웹, 시간 답에는 볼 문서 출처가 없다(Q33).
            checked = "문서 검색 안 함(건너뜀)"
        else:
            checked = f"{code_check(draft, set(state.get('seen', [])))}(참고)"
        ctx.progress(f"코드 출처 검사: {checked}")
        return {"verdicts": [asdict(v) for v in verdicts], "code_check": checked}

    def revise(state: State) -> State:
        revisions = state.get("revisions", 0) + 1
        ctx.progress(f"고쳐 쓰기 {revisions}/{ctx.max_revisions}")
        return {"revisions": revisions, "feedback": [f"{v.criterion}: {v.reason}" for v in failed(verdicts_of(state))]}

    def passed(state: State) -> State:
        return {"status": "passed", "answer": state.get("draft", "")}

    def limit(state: State) -> State:
        ctx.progress(f"고쳐 쓰기 상한({ctx.max_revisions}번)에 이르러 지금 답으로 끝냅니다")
        rest = "\n".join(f"- {v.criterion}: {v.reason}" for v in failed(verdicts_of(state)))
        return {"status": "limit", "answer": f"{state.get('draft', '')}\n\n아직 통과하지 못한 기준:\n{rest}"}

    def after_evaluate(state: State) -> str:
        if state.get("status") == "eval_failed":
            return END
        if not any(v["result"] == FAIL for v in state.get("verdicts", [])):
            return "passed"
        return "revise" if state.get("revisions", 0) < ctx.max_revisions else "limit"

    graph = StateGraph(State)
    for name, node in [("write", write), ("evaluate", evaluate), ("revise", revise), ("passed", passed), ("limit", limit)]:
        graph.add_node(name, node)
    graph.add_edge(START, "write")
    graph.add_edge("write", "evaluate")
    graph.add_conditional_edges("evaluate", after_evaluate, ["passed", "revise", "limit", END])
    graph.add_edge("revise", "write")
    graph.add_edge("passed", END)
    graph.add_edge("limit", END)
    return graph


def verdicts_of(state: State) -> list[Verdict]:
    return [Verdict(**v) for v in state.get("verdicts", [])]


def unavailable(ctx: WorkflowContext) -> str | None:
    if ctx.team is None or ctx.base is None:
        return f"에이전트 정의가 없어 {NAME}를 쓸 수 없습니다."
    if EVALUATOR not in ctx.team.agents:
        return f"{NAME}에 필요한 에이전트 정의가 없습니다: {EVALUATOR} (agents/이름/AGENT.md)"
    return None


WORKFLOW = Workflow(
    name=NAME,
    description="main이 답을 쓰고 evaluator가 기준표로 평가해, 모든 기준을 통과하거나 상한에 닿을 때까지 고쳐 쓴다",
    build=build,
    unavailable=unavailable,
    count_label="고쳐 쓴 횟수",
)
