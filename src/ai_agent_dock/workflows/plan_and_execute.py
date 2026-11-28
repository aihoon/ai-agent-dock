"""plan-and-execute 워크플로(C18): planner 에이전트가 단계 계획을 만들고, worker가 한 단계씩 차례로 실행하고,
단계마다 planner가 결과를 보고 남은 계획을 유지하거나 고치거나 끝낸다(replan). 끝나면 aggregator가 단계 결과를 합친다.

C16과 같은 worker 목록을 쓰지만, 동시에 한 번 돌지 않고 차례로 돌며 뒤 단계가 앞 단계의 결과를 받는다(Q10).
사람 승인은 없다(Q9). 단계는 C16 가지처럼 승인이 필요 없는 Tool만 쓰고, 되묻거나 다시 맡기지 않는다(Q20).
planner의 출력은 JSON(finished, steps)이고, 다시 계획할 때의 입력도 State로 만든 JSON이다(Q22~Q24). 형식 설명은 agents/planner/AGENT.md에 있다.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ..core.agents import MAIN
from ..core.orchestration import Workflow, WorkflowContext, ask_agent
from ..core.retrieval import check_citations
from ..core.tools import Node, tools_for
from .common import FALLBACK, MAX_TASKS, aggregator_prompt, ONLY_MINE, WORKERS, branch_config, branch_name, clean_label, missing_agents, slot_config

NAME = "plan-and-execute"
PLANNER = "planner"
AGGREGATOR = "aggregator"
MAX_RUNS = 8  # 실행 상한(Q8). 계획 단계 수 상한 5(MAX_TASKS)는 read_plan이 지킨다(Q15).
FORMAT = "질문에 맞는 형식으로 답한다. 단계는 차례로 실행되었고, 뒤 단계는 앞 단계의 결과를 받아 썼다."


class State(TypedDict, total=False):
    workflow: str
    question: str
    started: str
    status: str  # running, completed, limit
    label: str  # planner가 마지막으로 낸 답(JSON 원문, Q26)
    plan: list[dict]  # 처음 계획 {worker, task}
    remaining: list[dict]  # 남은 계획 {worker, task}
    done: list[dict]  # 끝난 단계 {n, worker, task, result, ok}. n은 실행 순서. replan 입력 JSON의 done과 같은 모양
    workers: list[str]  # 실행한 단계의 worker(평가가 expect_workers와 비교)
    revisions: int  # replan이 계획을 고친 횟수
    answer: str


def worker_list() -> str:
    return "\n".join(f"- {w.name}: {w.description}" for w in WORKERS)


def results_text(done: list[dict]) -> str:
    return "\n\n".join(f"[{branch_name(d)}] 단계: {d['task']}\n결과:\n{d['result']}" for d in done)


def plan_prompt(question: str) -> str:
    return f"worker 목록:\n{worker_list()}\n\n질문: {question}"


def replan_input(state: State) -> dict:
    """다시 계획할 때 planner에게 줄 JSON(Q24). 필드 이름은 AGENT.md의 "받는 것"과 같다."""
    done = state.get("done", [])
    return {
        "question": state["question"],
        "plan": state["plan"],
        "done": [{"n": d["n"], "worker": d["worker"], "task": d["task"], "ok": d["ok"], "result": d["result"]} for d in done],
        "remaining": state.get("remaining", []),
        "runs_left": MAX_RUNS - len(done),
    }


def replan_prompt(state: State) -> str:
    return f"worker 목록:\n{worker_list()}\n\n{json.dumps(replan_input(state), ensure_ascii=False, indent=2)}"


def read_plan(label: str) -> tuple[bool, list[tuple[str, str]]] | None:
    """planner의 출력 JSON을 (finished, [(worker 이름, 단계)])로 읽는다. 코드 울타리와 앞뒤 글은 버리고
    첫 { 부터 마지막 } 까지 읽는다(Q25). JSON이 아니거나 모양이 틀리면 None. 단계는 5개까지."""
    try:
        data = json.loads(label[label.index("{") : label.rindex("}") + 1])
        finished, steps = data["finished"], data["steps"]
        if not isinstance(finished, bool) or not isinstance(steps, list):
            return None
        pairs = [(clean_label(s["worker"]), s["task"].strip()) for s in steps]
    except (ValueError, KeyError, TypeError, AttributeError):  # JSON이 아님, 칸이 없음, 칸의 값이 문자열이 아님
        return None
    if not all(name and task for name, task in pairs):
        return None
    return finished, pairs[:MAX_TASKS]


def step_prompt(question: str, done: list[dict], task: str) -> str:
    """Q10: 원래 질문, 앞 단계 결과 전체, 맡은 단계, 맡은 것만 한다는 문장."""
    before = f"앞 단계 결과:\n{results_text(done)}\n\n" if done else ""
    return f"원래 질문: {question}\n\n{before}맡은 단계: {task}\n{ONLY_MINE}"


def build(ctx: WorkflowContext) -> StateGraph:
    slots = {w.name: w for w in WORKERS}

    def ask_planner(prompt: str) -> str:
        try:
            return ask_agent(ctx.team.config_for(PLANNER, ctx.base), prompt)
        except Exception as e:  # 계획에 실패하면 general(plan)이나 끝(replan)으로 간다(Q11).
            return f"오류: {type(e).__name__}"

    def to_steps(pairs: list[tuple[str, str]]) -> list[dict]:
        steps = []
        for name, task in pairs:
            if name not in slots:
                ctx.progress(f"알 수 없는 worker {name} → {FALLBACK}")
            steps.append({"worker": name if name in slots else FALLBACK, "task": task})
        return steps

    def plan(state: State) -> State:
        ctx.progress(f"워크플로 시작: {NAME}")
        ctx.progress("Plan 시작")
        label = ask_planner(plan_prompt(state["question"]))
        parsed = read_plan(label)
        steps = to_steps(parsed[1]) if parsed and not parsed[0] else []  # 처음 계획의 finished: true는 못 읽음과 같다(Q23).
        if not steps:
            ctx.progress(f'Plan 완료: 실패("{label}") → {FALLBACK}')
            return {"label": label, "plan": [], "remaining": [], "workers": []}
        ctx.progress("\n".join([f"Plan 완료: {len(steps)}단계", *(f"  {s['worker']}: {s['task']}" for s in steps)]))
        return {"label": label, "plan": steps, "remaining": steps, "done": [], "workers": []}

    def execute(state: State) -> State:
        step, done = state["remaining"][0], state.get("done", [])
        task = {"n": len(done) + 1, **step}
        branch = branch_name(task)
        run = f"Execute {task['n']}/{MAX_RUNS}"
        ctx.progress(f"{run} 시작: {branch}\n  맡은 단계: {step['task']}")
        config = ctx.config if step["worker"] == FALLBACK else slot_config(ctx, slots[step["worker"]])
        tools = tools_for(config.tools, node=Node(branch=True))  # 되묻거나 다시 맡기지 않고, 승인이 필요 없는 Tool만(Q20)
        try:
            result, ok = ask_agent(branch_config(ctx, replace(config, tools=tools), branch, NAME), step_prompt(state["question"], done, step["task"])), True
            ctx.progress(f"{run} 완료: {branch}")
        except Exception as e:  # 실패도 결과로 남기고 replan이 다음을 정한다(Q11).
            result, ok = f"실행 실패: {type(e).__name__}: {e}", False
            ctx.progress(f"{run} 실패: {branch} ({type(e).__name__}: {e})")
        return {"remaining": state["remaining"][1:], "done": [*done, {**task, "result": result, "ok": ok}], "workers": [*state["workers"], step["worker"]]}

    def replan(state: State) -> State:
        ctx.progress("Replan 시작")
        label = ask_planner(replan_prompt(state))
        parsed = read_plan(label)
        if parsed and parsed[0]:
            ctx.progress("Replan 완료: 끝")
            return {"label": label, "remaining": []}
        steps = to_steps(parsed[1]) if parsed else []
        if not steps:  # 못 읽으면 지금까지로 답한다(Q25).
            ctx.progress(f'Replan 완료: 읽기 실패("{label}") → 끝')
            return {"label": label, "remaining": []}
        if steps == state["remaining"]:
            ctx.progress("Replan 완료: 유지")
            return {"label": label}
        ctx.progress("\n".join(["Replan 완료: 고침", *(f"  {s['worker']}: {s['task']}" for s in steps)]))
        return {"label": label, "remaining": steps, "revisions": state.get("revisions", 0) + 1}

    def aggregate(state: State) -> State:
        done, rest = state["done"], state.get("remaining", [])
        limit = len(done) >= MAX_RUNS
        if limit:
            ctx.progress(f"Execute 상한({MAX_RUNS}번)에 이르러 지금까지로 답합니다")
        if not any(d["ok"] for d in done):  # 모두 실패하면 합치지 않는다(C16과 같음).
            answer = "\n".join(f"{branch_name(d)}: {d['result']}" for d in done)
        else:
            ctx.progress(f"Aggregate 시작: {AGGREGATOR}")
            answer = ask_agent(ctx.team.config_for(AGGREGATOR, ctx.base), aggregator_prompt(state["question"], [(branch_name(d), d["task"], d["result"]) for d in done], form=FORMAT))
            unknown = check_citations(answer, ctx.knowledge.seen).unknown if ctx.knowledge else []  # URL 출처는 보지 않는다.
            if unknown:
                ctx.progress(f"※ 검색 결과에 없는 출처: {', '.join(unknown)}")
        if limit and rest:
            answer += "\n\n실행 상한에 이르러 하지 못한 단계:\n" + "\n".join(f"- {s['worker']}: {s['task']}" for s in rest)
        return {"status": "limit" if limit else "completed", "answer": answer}

    def general(state: State) -> State:
        ctx.progress(f"{FALLBACK} 경로 실행")
        return {"status": "completed", "answer": ask_agent(ctx.config, state["question"])}

    graph = StateGraph(State)
    for name, node in [("plan", plan), ("execute", execute), ("replan", replan), ("aggregate", aggregate), ("general", general)]:
        graph.add_node(name, node)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", lambda s: "execute" if s["remaining"] else "general", ["execute", "general"])
    graph.add_conditional_edges("execute", lambda s: "aggregate" if len(s["done"]) >= MAX_RUNS else "replan", ["replan", "aggregate"])
    graph.add_conditional_edges("replan", lambda s: "execute" if s["remaining"] else "aggregate", ["execute", "aggregate"])
    graph.add_edge("aggregate", END)
    graph.add_edge("general", END)
    return graph


def unavailable(ctx: WorkflowContext) -> str | None:
    return missing_agents(ctx, NAME, {PLANNER, AGGREGATOR, MAIN, *(w.agent for w in WORKERS)})


WORKFLOW = Workflow(
    name=NAME,
    description="planner가 단계 계획을 만들고, worker가 한 단계씩 차례로 실행하며, 단계마다 남은 계획을 다시 보고 고친다",
    build=build,
    unavailable=unavailable,
    count_label="계획 고친 횟수",
)
