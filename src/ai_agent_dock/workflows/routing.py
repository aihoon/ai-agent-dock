"""routing 워크플로(C14): router 에이전트가 질문을 분류하고, 코드가 경로를 골라, 그 경로의 에이전트가 답한다.

경로 목록은 이 파일의 경로 표(ROUTES)가 정한다. router의 시스템 프롬프트(agents/router/AGENT.md)에는 역할과 규칙만 두고,
경로 목록은 코드가 만들어 사용자 프롬프트에 넣는다. 고른 경로 에이전트의 답이 최종 답이다(한 턴, 하위 에이전트도 handoff도 아니다).
경로 에이전트에는 질문만 넘긴다(Q11). 복합 질문은 나누지 않고 general로 보낸다(나누기는 C16, C18).
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ..core.agents import MAIN
from ..core.orchestration import Workflow, WorkflowContext, ask_agent
from .common import AgentSlot, clean_label, slot_config

ROUTER = "router"
FALLBACK = "general"


# 기본값을 두지 않는다. 경로마다 agent, model, tools를 모두 적고, 기본을 쓰면 ""와 ()로 비워 둔다.
ROUTES = [  # 경로 표. sectioning의 메뉴(MENU)와 따로 둔다(C15 Q20).
    AgentSlot("chat", "인사, 일반 지식, 글 다듬기 같은 가벼운 대화. Tool이 필요 없다", agent="chat", model="", tools=()),
    AgentSlot("docs", "이 프로젝트 문서에 있는 내용을 묻는 질문", agent=MAIN, model="", tools=("search_docs",)),
    AgentSlot("web", "최신 정보, 뉴스, 외부 웹사이트 내용", agent="web", model="", tools=()),
    AgentSlot("utility", "계산, 지금 시각과 날짜", agent=MAIN, model="openai:gpt-6-luna", tools=("calculate", "get_current_time")),
    AgentSlot("files", "샌드박스 파일 읽기, 목록, 쓰기", agent=MAIN, model="", tools=("files__*",)),
    # main 그대로(ask_user, delegate 포함, Q10)
    AgentSlot(FALLBACK, "여러 일이 섞였거나 어느 경로인지 분명하지 않은 질문", agent=MAIN, model="", tools=()),
]


class State(TypedDict, total=False):
    workflow: str
    question: str
    started: str
    status: str  # running, completed
    label: str  # router가 실제로 낸 답
    route: str  # 고른 경로
    answer: str


def router_prompt(question: str) -> str:
    routes = "\n".join(f"- {r.name}: {r.description}" for r in ROUTES)
    return f"경로 목록:\n{routes}\n\n질문: {question}"


def read_route(label: str) -> str | None:
    """router의 답을 경로 이름으로. 대소문자, 앞뒤 공백, 따옴표, 마침표만 정리한다. 그 밖의 답은 None(Q4)."""
    name = clean_label(label)
    return name if any(r.name == name for r in ROUTES) else None


def build(ctx: WorkflowContext) -> StateGraph:
    routes = {r.name: r for r in ROUTES}

    def classify(state: State) -> State:
        ctx.progress(f"분류: router 에이전트가 경로 {len(ROUTES)}개 중 하나를 고른다")
        try:
            label = ask_agent(ctx.team.config_for(ROUTER, ctx.base), router_prompt(state["question"]))
        except Exception as e:  # 분류에 실패해도 general로 답한다(다시 부르지 않는다, Q4).
            label = f"오류: {type(e).__name__}"
        route = read_route(label)
        ctx.progress(f"분류 결과: {route}" if route else f'분류 실패("{label}") → {FALLBACK}')
        return {"label": label, "route": route or FALLBACK}

    def run_route(state: State) -> State:
        route = routes[state["route"]]
        ctx.progress(f"{route.name} 경로 실행")
        return {"status": "completed", "answer": ask_agent(slot_config(ctx, route), state["question"])}

    graph = StateGraph(State)
    graph.add_node("classify", classify)
    graph.add_node("run_route", run_route)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "run_route")
    graph.add_edge("run_route", END)
    return graph


def unavailable(ctx: WorkflowContext) -> str | None:
    if ctx.team is None or ctx.base is None:
        return "에이전트 정의가 없어 routing을 쓸 수 없습니다."
    missing = sorted({ROUTER, *(r.agent for r in ROUTES)} - set(ctx.team.agents))
    if missing:
        return f"routing에 필요한 에이전트 정의가 없습니다: {', '.join(missing)} (agents/이름/AGENT.md)"
    return None


WORKFLOW = Workflow(
    name="routing",
    description="router가 질문을 분류하고, 고른 경로의 에이전트(프롬프트, Tool, 모델)가 답한다",
    build=build,
    unavailable=unavailable,
)
