"""orchestrator-workers 워크플로(C16): orchestrator 에이전트가 질문을 보고 하위 작업의 내용, 개수, 맡을 worker를 정하면,
worker들이 동시에 일하고, aggregator 에이전트가 한 답으로 합친다. 예제 작업은 research-report다.

sectioning(C15)과 구조가 같고, 나누는 쪽이 메뉴에서 고르기(selector)가 아니라 작업 새로 쓰기(orchestrator)다(Q3-1).
같은 worker를 여러 작업에 쓸 수 있다(Q2-1). 한 번 나누고 끝난다(다시 나누기는 C18).
가지는 승인이 필요 없는 Tool만 쓰고, 되묻거나 다시 맡기지 않는다(Q14).
"""

from __future__ import annotations

import operator
import re
from dataclasses import replace
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..core.agents import DELEGATE, MAIN
from ..core.orchestration import Workflow, WorkflowContext, ask_agent
from ..core.retrieval import check_citations
from .common import AgentSlot, branch_config, clean_label, slot_config

NAME = "orchestrator-workers"
ORCHESTRATOR = "orchestrator"
AGGREGATOR = "aggregator"
FALLBACK = "general"  # 목록에 없는 worker 이름의 작업을 맡고(Q14), 작업을 못 읽으면 원래 질문에 답한다.
# 같은 Tool이 두 worker에 겹치지 않게 researcher에서 웹 Tool을 뺐다(Q12).
WORKERS = [
    AgentSlot("researcher", "이 프로젝트 문서를 조사한다(필요하면 샌드박스 파일도 읽는다)", agent="researcher", model="", tools=("search_docs", "files__read_*")),
    AgentSlot("web", "최신 정보, 외부 웹사이트를 조사한다", agent="web", model="", tools=()),
    AgentSlot("utility", "계산, 지금 시각과 날짜", agent=MAIN, model="openai:gpt-6-luna", tools=("calculate", "get_current_time")),
]
MAX_TASKS = 5
NO_ASKING = {"ask_user", DELEGATE}  # 동시에 도는 가지가 사람에게 묻거나 다시 맡기지 않는다.
FORMAT = "질문에 맞는 형식으로 합친다. 질문이 보고서를 원하면 제목, 요약(3~5줄), 작업마다 절 하나로 쓴다."
ONLY_MINE = "원래 질문은 목적을 알려 주려고 넘긴다. 맡은 작업만 하고, 나머지는 다른 worker가 맡는다."  # Q22
LEADING = re.compile(r"^\s*(?:[-*•]|\d+[.)])?\s*")  # 줄 앞의 목록 기호와 번호
WORD = re.compile(r"[a-z_]+")


class State(TypedDict, total=False):
    workflow: str
    question: str
    started: str
    status: str  # running, completed
    label: str  # orchestrator가 실제로 낸 답
    tasks: list[dict]  # 작업마다 {n, worker, task}
    workers: list[str]  # 작업 순서대로 worker 이름(평가가 expect_workers와 비교, Q17). 못 나눴으면 []
    results: Annotated[list[dict], operator.add]  # 가지마다 {n, worker, task, answer, ok}. 동시에 들어오는 결과를 모은다.
    answer: str


def plan_prompt(question: str) -> str:
    workers = "\n".join(f"- {w.name}: {w.description}" for w in WORKERS)
    return f"worker 목록:\n{workers}\n\n질문: {question}"


def read_tasks(label: str) -> list[tuple[str, str]]:
    """orchestrator의 답을 (worker 이름, 작업) 목록으로. 이름은 영어 소문자 한 단어만 받고(설명 줄을 작업으로 읽지 않게),
    목록에 없는 이름도 남긴다(그래프가 general에게 맡긴다). 5개까지."""
    tasks = []
    for line in label.splitlines():
        name, colon, task = LEADING.sub("", line).partition(":")
        name, task = clean_label(name), task.strip()
        if colon and task and WORD.fullmatch(name):
            tasks.append((name, task))
    return tasks[:MAX_TASKS]


def branch_name(task: dict) -> str:
    """가지 이름: worker-작업 번호(researcher-1, web-2). 추적, 진행 줄, 합치기 프롬프트에 쓴다(Q15)."""
    return f"{task['worker']}-{task['n']}"


def worker_prompt(question: str, task: str) -> str:
    return f"원래 질문: {question}\n맡은 작업: {task}\n{ONLY_MINE}"


def aggregator_prompt(question: str, results: list[dict]) -> str:
    parts = [f"[{branch_name(r)}] 맡은 작업: {r['task']}\n결과:\n{r['answer']}" for r in results]
    return "\n\n".join([f"원래 질문: {question}", *parts, FORMAT])


def build(ctx: WorkflowContext) -> StateGraph:
    slots = {w.name: w for w in WORKERS}

    def plan(state: State) -> State:
        ctx.progress(f"나누기: orchestrator가 worker {len(WORKERS)}종으로 작업을 정한다")
        try:
            label = ask_agent(ctx.team.config_for(ORCHESTRATOR, ctx.base), plan_prompt(state["question"]))
        except Exception as e:  # 나누기에 실패해도 general로 답한다.
            label = f"오류: {type(e).__name__}"
        tasks = []
        for n, (name, task) in enumerate(read_tasks(label), 1):
            if name not in slots:
                ctx.progress(f"알 수 없는 worker {name} → {FALLBACK}")
            tasks.append({"n": n, "worker": name if name in slots else FALLBACK, "task": task})
        if not tasks:
            ctx.progress(f'나누기 실패("{label}") → {FALLBACK}')
            return {"label": label, "tasks": [], "workers": []}
        ctx.progress("\n".join([f"작업 {len(tasks)}개:", *(f"  {branch_name(t)}: {t['task']}" for t in tasks)]))  # 작업마다 한 줄, 들여 쓴다
        if len(tasks) > 1:
            ctx.progress("동시 실행: " + ", ".join(branch_name(t) for t in tasks))
        return {"label": label, "tasks": tasks, "workers": [t["worker"] for t in tasks]}

    def fan_out(state: State):
        if not state["tasks"]:
            return "general"
        return [Send("run_worker", {**task, "question": state["question"]}) for task in state["tasks"]]

    def run_worker(task: dict) -> State:
        branch = branch_name(task)
        config = ctx.config if task["worker"] == FALLBACK else slot_config(ctx, slots[task["worker"]])
        tools = [t for t in config.tools if t.read_only and t.trusted and t.name not in NO_ASKING]
        try:
            answer, ok = ask_agent(branch_config(ctx, replace(config, tools=tools), branch, NAME), worker_prompt(task["question"], task["task"])), True
            ctx.progress(f"{branch} 완료")
        except Exception as e:  # 한 worker가 실패해도 나머지로 합친다(Q9).
            answer, ok = f"조사 실패: {type(e).__name__}: {e}", False
            ctx.progress(f"{branch} 실패({type(e).__name__}: {e})")
        return {"results": [{"n": task["n"], "worker": task["worker"], "task": task["task"], "answer": answer, "ok": ok}]}

    def aggregate(state: State) -> State:
        done = sorted(state["results"], key=lambda r: r["n"])  # 가지는 끝나는 순서대로 들어온다.
        if not any(r["ok"] for r in done):  # 모두 실패하면 합치지 않는다(Q9).
            return {"status": "completed", "answer": "\n".join(f"{branch_name(r)}: {r['answer']}" for r in done)}
        ctx.progress(f"합치기: {AGGREGATOR}")
        answer = ask_agent(ctx.team.config_for(AGGREGATOR, ctx.base), aggregator_prompt(state["question"], done))
        unknown = check_citations(answer, ctx.knowledge.seen).unknown if ctx.knowledge else []  # URL 출처는 보지 않는다.
        if unknown:
            ctx.progress(f"※ 검색 결과에 없는 출처: {', '.join(unknown)}")
        return {"status": "completed", "answer": answer}

    def general(state: State) -> State:
        ctx.progress(f"{FALLBACK} 경로 실행")
        return {"status": "completed", "answer": ask_agent(ctx.config, state["question"])}

    graph = StateGraph(State)
    graph.add_node("plan", plan)
    graph.add_node("run_worker", run_worker)
    graph.add_node("aggregate", aggregate)
    graph.add_node("general", general)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", fan_out, ["run_worker", "general"])
    graph.add_edge("run_worker", "aggregate")
    graph.add_edge("aggregate", END)
    graph.add_edge("general", END)
    return graph


def unavailable(ctx: WorkflowContext) -> str | None:
    if ctx.team is None or ctx.base is None:
        return f"에이전트 정의가 없어 {NAME}를 쓸 수 없습니다."
    missing = sorted({ORCHESTRATOR, AGGREGATOR, MAIN, *(w.agent for w in WORKERS)} - set(ctx.team.agents))
    if missing:
        return f"{NAME}에 필요한 에이전트 정의가 없습니다: {', '.join(missing)} (agents/이름/AGENT.md)"
    return None


WORKFLOW = Workflow(
    name=NAME,
    description="orchestrator가 하위 작업과 맡을 worker를 정하면, worker들이 동시에 일하고 aggregator가 합친다",
    build=build,
    unavailable=unavailable,
)
