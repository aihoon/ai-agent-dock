"""debate 워크플로(C20): 토론자 3명이 따로 답하고(propose), 라운드마다 서로의 답을 보고 고친다(debate).

합의형 multi-agent debate(Du 외, 2023)를 따른다. 세 토론자는 같은 정의(agents/debater/AGENT.md)로 동시에 돌고(Send),
라운드끼리는 차례로 돈다. 토론자는 C17 작성자와 같은 Tool(웹 포함)을 쓰되 가지라 승인이 필요한 Tool은 받지 않는다(Q26).

결론 줄이 글자까지 모두 같으면 바로 끝낸다(agreed, LLM 호출 없음). 고치기 상한까지 갈리면 aggregator가 마지막 답들을 읽고
같은 뜻인지 판단해 최종 답을 쓴다(decided, Q25). 코드 비교는 같은 뜻을 다르다고 볼 수는 있어도, 다른 뜻을 같다고 보지는 않는다.
"""

from __future__ import annotations

import operator
import re
from dataclasses import replace
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..core.orchestration import Workflow, WorkflowContext, ask_agent
from ..core.safety import wrap_untrusted
from .common import branch_config, missing_agents, writer_tools

NAME = "debate"
DEBATER = "debater"
AGGREGATOR = "aggregator"
DEBATERS = 3
MAX_ROUNDS = 4  # 고치기 라운드 상한(Q4). 첫 답까지 최대 5라운드
CONCLUSION = "결론:"
REVIEW = "[다른 답 검토]"  # debate 라운드 답의 맨 끝 절. 다른 토론자는 받고, 최종 답에서는 잘라 낸다(Q33).
ENDINGS = ("입니다", "이다", "다")


class State(TypedDict, total=False):
    workflow: str
    question: str
    started: str
    status: str  # running, agreed, decided, failed
    answers: Annotated[list[dict], operator.add]  # 토론자 답 {round, n, conclusion, text}. 실패면 conclusion이 None
    next: str  # tally가 정한 다음 길: agreed, debate, decide, failed
    revisions: int  # 지난 debate 라운드 수
    answer: str


def read_conclusion(text: str) -> str:
    """첫 줄의 "결론:" 뒤. 첫 줄이 "결론:"으로 시작하지 않으면 첫 줄 전체(4.2절)."""
    line = next((line for line in text.splitlines() if line.strip()), "")
    line = line.strip().strip("*").strip()
    return line[len(CONCLUSION) :].strip() if line.startswith(CONCLUSION) else line


def normalize(conclusion: str) -> str:
    """비교할 글자: 굵게, 따옴표, 문장 부호, 숫자의 쉼표, 공백을 떼고, 끝의 "입니다/이다/다"를 하나 뗀 뒤 소문자(4.3절)."""
    # 숫자의 부호(-3), 소수점(3.5), 분수(1/2)는 남긴다. 지우면 다른 값을 같다고 본다(C90 리뷰 A3).
    text = re.sub(r"(?<=\d),(?=\d)", "", conclusion)
    text = re.sub(r"(?!-(?=\d)|(?<=\d)[./](?=\d))[\W_]", "", text).lower()
    for ending in ENDINGS:
        if text.endswith(ending) and len(text) > len(ending):
            return text[: -len(ending)]
    return text


def without_review(text: str) -> str:
    """[다른 답 검토] 절부터 끝까지를 뗀다. 절이 없으면 그대로(Q33)."""
    return text.split(REVIEW, 1)[0].rstrip()


def name_of(n: int) -> str:
    return f"{DEBATER}-{n}"


def debate_prompt(question: str, own: dict, others: list[dict]) -> str:
    """질문, 자기의 지난 답, 다른 토론자의 지난 답 전문(Q12). 다른 토론자의 Tool 결과는 넘기지 않는다(Q31)."""
    parts = [f"[질문]\n{question}", f"[당신의 지난 답]\n{own['text'] if own['conclusion'] is not None else '(답 없음)'}", "[다른 토론자의 지난 답]"]
    for other in others:
        body = wrap_untrusted(name_of(other["n"]), other["text"]) if other["conclusion"] is not None else "(답 없음)"
        parts.append(f"{name_of(other['n'])}:\n{body}")
    parts.append(
        "의심되는 주장은 직접 찾아 확인한 뒤, 답을 고치거나 유지한다. 첫 줄은 \"결론: 짧은 답\"이다(예/아니요 질문이면 예 또는 아니요만). "
        f"그 아래 근거를 쓰고, 다른 답의 틀린 곳은 맨 끝 {REVIEW} 절에만 짧게 쓴다."
    )
    return "\n\n".join(parts)


def decide_prompt(question: str, answers: list[dict]) -> str:
    """aggregator에게: 질문과 마지막 라운드의 토론자별 답 전문(4.2절)."""
    parts = [f"원래 질문: {question}"]
    for a in answers:
        parts.append(f"{name_of(a['n'])}의 답:\n{wrap_untrusted(name_of(a['n']), a['text'])}")
    parts.append(
        "토론자들이 서로의 답을 보며 고친 마지막 답이다. 같은 뜻의 결론은 같은 것으로 보고, 과반의 답을 근거와 함께 쓴다. "
        "과반이 없으면 결론들을 나란히 적고 갈렸다고 말한다. 답을 새로 풀지 않는다."
    )
    return "\n\n".join(parts)


def last_round(answers: list[dict]) -> list[dict]:
    """답이 하나라도 있는 가장 마지막 라운드의 답들(토론자 순서). 없으면 빈 목록."""
    for r in sorted({a["round"] for a in answers}, reverse=True):
        current = sorted((a for a in answers if a["round"] == r), key=lambda a: a["n"])
        if any(a["conclusion"] is not None for a in current):
            return current
    return []


def build(ctx: WorkflowContext) -> StateGraph:
    def config(n: int):
        base = ctx.team.config_for(DEBATER, ctx.base)
        return branch_config(ctx, replace(base, tools=writer_tools(ctx, [], set(), branch=True)), name_of(n), NAME)

    def speak(task: dict) -> State:
        try:
            text = ask_agent(config(task["n"]), task["prompt"])
            if not text.strip():
                raise ValueError("빈 답")
            conclusion = read_conclusion(text)
        except Exception as e:  # 그 라운드에서 이 토론자는 실패, 나머지로 계속한다(Q15).
            text, conclusion = f"실패: {type(e).__name__}: {e}", None
        return {"answers": [{"round": task["round"], "n": task["n"], "conclusion": conclusion, "text": text}]}

    def fan_out(state: State) -> list[Send]:
        ctx.progress(f"워크플로 시작: {NAME} (토론자 {DEBATERS}명, 고치기 최대 {MAX_ROUNDS}번)")
        ctx.progress("Propose 시작: " + ", ".join(name_of(n) for n in range(1, DEBATERS + 1)))
        return [Send("propose", {"round": 0, "n": n, "prompt": state["question"]}) for n in range(1, DEBATERS + 1)]

    def tally(state: State) -> State:
        r = max(a["round"] for a in state["answers"])
        current = sorted((a for a in state["answers"] if a["round"] == r), key=lambda a: a["n"])
        shown = " | ".join(a["conclusion"] if a["conclusion"] is not None else "실패" for a in current)
        answers = last_round(state["answers"])
        found = {normalize(a["conclusion"]) for a in answers if a["conclusion"] is not None}
        if not answers:
            step, mark = "failed", "답 없음"
        elif len(found) == 1:
            step, mark = "agreed", "모두 같음"
        else:
            step, mark = ("debate" if r < MAX_ROUNDS else "decide"), "갈림"
        ctx.progress(f"{'Propose' if r == 0 else f'Debate {r}/{MAX_ROUNDS}'} 완료: {shown} → {mark}")
        return {"next": step, "revisions": r}

    def route(state: State):
        step = state["next"]
        if step != "debate":
            return step
        r = state["revisions"] + 1
        answers = last_round(state["answers"])
        ctx.progress(f"Debate {r}/{MAX_ROUNDS} 시작")
        return [
            Send("debate", {"round": r, "n": a["n"], "prompt": debate_prompt(state["question"], a, [o for o in answers if o is not a])})
            for a in answers
        ]

    def failed(state: State) -> State:
        reasons = "; ".join(a["text"] for a in state["answers"])
        return {"status": "failed", "answer": f"토론자가 모두 답하지 못했습니다: {reasons}"}

    def agreed(state: State) -> State:
        first = next(a for a in last_round(state["answers"]) if a["conclusion"] is not None)
        return {"status": "agreed", "answer": without_review(first["text"])}

    def decide(state: State) -> State:
        answers = [a for a in last_round(state["answers"]) if a["conclusion"] is not None]
        ctx.progress(f"Decide 시작: {AGGREGATOR} (갈린 채 상한)")
        try:
            answer = ask_agent(ctx.team.config_for(AGGREGATOR, ctx.base), decide_prompt(state["question"], answers))
            if not answer.strip():
                raise ValueError("빈 답")
            ctx.progress("Decide 완료")
        except Exception as e:  # 정리하지 못하면 답이 있는 첫 토론자의 답(4.2절)
            ctx.progress(f"Decide 실패({type(e).__name__}: {e}) → {name_of(answers[0]['n'])}의 답으로 끝냅니다")
            answer = f"정리 실패: {name_of(answers[0]['n'])}의 답입니다.\n\n{answers[0]['text']}"
        return {"status": "decided", "answer": answer}

    graph = StateGraph(State)
    for name, node in [("propose", speak), ("debate", speak), ("tally", tally), ("agreed", agreed), ("decide", decide), ("failed", failed)]:
        graph.add_node(name, node)
    graph.add_conditional_edges(START, fan_out, ["propose"])
    graph.add_edge("propose", "tally")
    graph.add_edge("debate", "tally")
    graph.add_conditional_edges("tally", route, ["debate", "agreed", "decide", "failed"])
    for name in ("agreed", "decide", "failed"):
        graph.add_edge(name, END)
    return graph


def unavailable(ctx: WorkflowContext) -> str | None:
    return missing_agents(ctx, NAME, {DEBATER, AGGREGATOR})


WORKFLOW = Workflow(
    name=NAME,
    description="토론자 3명이 각자 찾아 답하고 서로의 답을 보며 고친다. 결론이 모두 같으면 끝, 상한까지 갈리면 aggregator가 정한다",
    build=build,
    unavailable=unavailable,
    count_label="토론 라운드",
)
