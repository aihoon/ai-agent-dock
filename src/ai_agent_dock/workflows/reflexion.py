"""reflexion 워크플로(C19): main이 답을 쓰고(act), evaluator가 판정하고(evaluate), 실패하면 reflector가 교훈 한 줄을 쓴다(reflect).

다시 쓸 때는 이전 답 없이 질문과 교훈만 받아 처음부터 쓴다(Q11, C17과 다른 점). 반성을 거쳐 통과하면 그 교훈을
data/reflections.md에 저장해 다음 실행(다른 질문, 다른 세션)의 첫 시도부터 넣는다(Q3). 논문의 Actor, Evaluator, Self-Reflection이
act, evaluate, reflect다(Shinn 외, 2023).

기준표는 C17의 4개에 작성자는 모르는 팀 규칙 2개를 더한다(4.3절). 교훈으로 배울 것이 분명히 보이게 하려는 시연용이다.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ..core.evals.rubric import EVALUATOR, FAIL, LEADING, RUBRIC, Criterion, Verdict, evaluator_prompt, failed, read_verdicts, verdict_line
from ..core.memory import MemoryStore
from ..core.orchestration import Workflow, WorkflowContext, ask_agent
from ..core.safety import wrap_untrusted
from .common import writer_tools

NAME = "reflexion"
REFLECTOR = "reflector"
MAX_SAVED = 20  # 저장하는 교훈 수 상한. 넘으면 오래된 것부터 지운다(Q4).
TEAM_RULES = [  # 작성자 프롬프트에는 없고 평가자만 아는 팀 규칙(Q7)
    Criterion("결론 먼저", '첫 줄은 "결론:"으로 시작하는 한 문장이다', False),
    Criterion("확인한 곳", '마지막 줄은 "확인한 곳:" 뒤에 답을 쓰며 부른 Tool 이름을 쓴다. Tool을 부르지 않았으면 "확인한 곳: 없음"', False),
]
CRITERIA = [*RUBRIC, *TEAM_RULES]


class State(TypedDict, total=False):
    workflow: str
    question: str
    started: str
    status: str  # running, passed, limit, eval_failed
    saved: list[str]  # 시작할 때 읽은 저장된 교훈
    lessons: list[dict]  # 이번 실행의 교훈 {text, keep}. keep이 false면 저장하지 않는다(reflector 대신 쓴 교훈, Q14).
    draft: str
    evidence: list[dict]  # 작성자가 부른 Tool의 {tool, result}
    verdicts: list[dict]  # 마지막 평가의 {criterion, result, reason}
    revisions: int  # 다시 쓴 횟수
    answer: str


def all_lessons(state: State) -> list[str]:
    return [*state.get("saved", []), *(lesson["text"] for lesson in state.get("lessons", []))]


def act_prompt(question: str, lessons: list[str]) -> str:
    """교훈이 있으면 질문 앞에 목록으로 넣는다(Q12). 이전 답과 고칠 점은 넣지 않는다(Q11)."""
    if not lessons:
        return question
    return "[지난 실패에서 배운 교훈]\n" + "\n".join(f"- {lesson}" for lesson in lessons) + f"\n\n[질문]\n{question}"


def reflect_prompt(question: str, draft: str, verdicts: list[Verdict], tools: list[str], lessons: list[str]) -> str:
    """질문, 실패한 답, 실패한 기준과 이유, 부른 Tool 이름, 이미 있는 교훈(Q9). Tool 결과 글은 넘기지 않는다."""
    reasons = "\n".join(f"- {v.criterion}: {v.reason}" for v in failed(verdicts))
    known = "\n".join(f"- {lesson}" for lesson in lessons) or "없음"
    return (
        f"질문: {question}\n\n실패한 답:\n{wrap_untrusted('answer', draft)}\n\n실패한 기준과 이유:\n{reasons}\n\n"
        f"답을 쓰며 부른 Tool: {', '.join(tools) or '없음'}\n\n이미 있는 교훈:\n{known}\n\n다음 시도에 쓸 교훈 한 줄만 쓴다."
    )


def read_lesson(label: str) -> str:
    """첫 줄만, 목록 기호와 따옴표를 뗀다. 없으면 빈 문자열."""
    line = next((line for line in label.splitlines() if line.strip()), "")
    return LEADING.sub("", line).strip(" \"'`")


def save_lessons(store: MemoryStore, lessons: list[str]) -> int:
    """저장소에 없는 교훈만 맨 끝에 더하고, MAX_SAVED를 넘으면 오래된 것부터 지운다(Q4). 더한 수를 돌려준다."""
    new = [text for text in dict.fromkeys(" ".join(t.split()) for t in lessons) if text and text not in store.items()]
    for text in new:
        store.add(text)
    while len(store.items()) > MAX_SAVED:
        store.remove(1)
    return len(new)


def build(ctx: WorkflowContext) -> StateGraph:
    attempts = ctx.max_revisions + 1  # 처음 1번 + 다시 쓰기(Q13)

    def act(state: State) -> State:
        update: State = {}
        if "saved" not in state:  # 첫 act: 저장된 교훈을 읽는다.
            update["saved"] = ctx.reflections.items()
            ctx.progress(f"워크플로 시작: {NAME} (저장된 교훈 {len(update['saved'])}개)")
        lessons = all_lessons({**state, **update})
        run = f"Act {state.get('revisions', 0) + 1}/{attempts}"
        ctx.progress(f"{run} 시작" + (f" (교훈 {len(lessons)}개)" if lessons else ""))
        evidence: list[tuple[str, str]] = []
        draft = ask_agent(replace(ctx.config, tools=writer_tools(ctx, evidence, set())), act_prompt(state["question"], lessons))
        ctx.progress(f"{run} 완료")
        return {**update, "draft": draft, "evidence": [{"tool": tool, "result": result} for tool, result in evidence]}

    def evaluate(state: State) -> State:
        ctx.progress("Evaluate 시작")
        draft = state.get("draft", "")
        evidence = [(e["tool"], e["result"]) for e in state.get("evidence", [])]
        try:
            label = ask_agent(ctx.team.config_for(EVALUATOR, ctx.base), evaluator_prompt(state["question"], draft, evidence, CRITERIA))
        except Exception as e:  # 평가하지 못하면 지금 답으로 끝낸다(Q14, C17과 같음).
            ctx.progress(f"Evaluate 실패({type(e).__name__}: {e}) → 지금 답으로 끝냅니다")
            return {"status": "eval_failed", "answer": draft}
        verdicts = read_verdicts(label, CRITERIA)
        ctx.progress(f"Evaluate 완료: {verdict_line(verdicts)}")
        return {"verdicts": [asdict(v) for v in verdicts]}

    def reflect(state: State) -> State:
        ctx.progress("Reflect 시작")
        verdicts = verdicts_of(state)
        tools = list(dict.fromkeys(e["tool"] for e in state.get("evidence", [])))
        prompt = reflect_prompt(state["question"], state.get("draft", ""), verdicts, tools, all_lessons(state))
        try:
            lesson, keep = read_lesson(ask_agent(ctx.team.config_for(REFLECTOR, ctx.base), prompt)), True
            if not lesson:
                raise ValueError("빈 답")
            ctx.progress(f"Reflect 완료\n  {lesson}")
        except Exception as e:  # 실패한 기준의 이유를 이 시도의 교훈으로 쓰고 저장하지 않는다(Q14).
            lesson, keep = "; ".join(f"{v.criterion}: {v.reason}" for v in failed(verdicts)), False
            ctx.progress(f"Reflect 실패({type(e).__name__}: {e}) → 실패 이유를 교훈으로\n  {lesson}")
        return {"lessons": [*state.get("lessons", []), {"text": lesson, "keep": keep}], "revisions": state.get("revisions", 0) + 1}

    def passed(state: State) -> State:
        """통과한 main의 답을 그대로 최종 답으로 둔다. 반성을 거쳐 통과했으면 교훈을 저장한다(Q3)."""
        keep = [lesson["text"] for lesson in state.get("lessons", []) if lesson["keep"]]
        added = save_lessons(ctx.reflections, keep) if keep else 0
        if added:
            ctx.progress(f"교훈 저장: {added}개 (모두 {len(ctx.reflections.items())}개)")
        return {"status": "passed", "answer": state.get("draft", "")}

    def limit(state: State) -> State:
        ctx.progress(f"Act 상한({attempts}번)에 이르러 지금 답으로 끝냅니다")
        rest = "\n".join(f"- {v.criterion}: {v.reason}" for v in failed(verdicts_of(state)))
        return {"status": "limit", "answer": f"{state.get('draft', '')}\n\n아직 통과하지 못한 기준:\n{rest}"}

    def after_evaluate(state: State) -> str:
        if state.get("status") == "eval_failed":
            return END
        if not any(v["result"] == FAIL for v in state.get("verdicts", [])):
            return "passed"
        return "reflect" if state.get("revisions", 0) < ctx.max_revisions else "limit"

    graph = StateGraph(State)
    for name, node in [("act", act), ("evaluate", evaluate), ("reflect", reflect), ("passed", passed), ("limit", limit)]:
        graph.add_node(name, node)
    graph.add_edge(START, "act")
    graph.add_edge("act", "evaluate")
    graph.add_conditional_edges("evaluate", after_evaluate, ["passed", "reflect", "limit", END])
    graph.add_edge("reflect", "act")
    graph.add_edge("passed", END)
    graph.add_edge("limit", END)
    return graph


def verdicts_of(state: State) -> list[Verdict]:
    return [Verdict(**v) for v in state.get("verdicts", [])]


def unavailable(ctx: WorkflowContext) -> str | None:
    if ctx.team is None or ctx.base is None:
        return f"에이전트 정의가 없어 {NAME}를 쓸 수 없습니다."
    missing = [name for name in (EVALUATOR, REFLECTOR) if name not in ctx.team.agents]
    if missing:
        return f"{NAME}에 필요한 에이전트 정의가 없습니다: {', '.join(missing)} (agents/이름/AGENT.md)"
    if ctx.reflections is None:
        return f"교훈 저장소가 없어 {NAME}를 쓸 수 없습니다."
    return None


WORKFLOW = Workflow(
    name=NAME,
    description="main이 답을 쓰고 evaluator가 판정해, 실패하면 reflector가 교훈을 쓰고 다시 쓴다. 통과에 도움이 된 교훈은 다음 실행에도 쓴다",
    build=build,
    unavailable=unavailable,
    count_label="다시 쓴 횟수",
)
