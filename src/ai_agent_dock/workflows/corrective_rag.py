"""corrective-rag 워크플로(C22): retrieve → grade(grader) → correct가 없으면 검색어를 재작성해 재검색(최대 2번) → generate → check.

CRAG(Yan 외 2024)를 따른다. 평가기는 학습한 모델이 아니라 grader 에이전트(LLM)이고, 재검색은 웹이 아니라 같은 색인이다(Q9).
조각 안의 문장 추리기(knowledge refinement)는 하지 않는다(Q8). generate는 Tool 없이 고른 조각(correct, ambiguous)만 받는다.
grader의 출력은 JSON {"grades", "rewrite"}다(Q12). 형식 설명은 agents/grader/AGENT.md에 있다.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ..core.orchestration import Workflow, WorkflowContext, ask_agent
from ..core.retrieval import check_citations
from ..core.safety import wrap_untrusted

NAME = "corrective-rag"
GRADER = "grader"
MAX_RETRIES = 2  # 재검색 상한(Q10). 검색은 모두 MAX_RETRIES + 1번
TOP_K = 5  # 한 번에 가져오는 조각 수(Q13). search_docs는 3개
GRADES = ("correct", "ambiguous", "incorrect")
USABLE = ("correct", "ambiguous")  # generate가 받는 조각
NO_EVIDENCE = "관련 문서에서 근거를 찾지 못했습니다."


class State(TypedDict, total=False):
    workflow: str
    question: str
    started: str
    status: str  # running, completed, no_evidence
    query: str  # 이번 검색어
    queries: list[str]  # 지금까지 쓴 검색어. grader가 같은 쪽으로 다시 찾지 않게 보여 준다
    new: list[dict]  # 이번 검색에서 새로 나온 조각 {cite, text}
    chunks: list[dict]  # 평가한 조각 {cite, text, grade}
    rewrite: str  # grader가 재작성한 검색어
    revisions: int  # 재검색 횟수
    answer: str


def read_grades(label: str, count: int) -> tuple[list[str], str] | None:
    """grader의 출력 JSON을 (평가 목록, 재작성한 검색어)로 읽는다. 첫 { 부터 마지막 } 까지 읽는다(C18 read_plan과 같음).
    JSON이 아니거나, grades가 조각 수와 다르거나, 모르는 평가가 있으면 None."""
    try:
        data = json.loads(label[label.index("{") : label.rindex("}") + 1])
        values = [g.strip().lower() for g in data["grades"]]
        rewrite = str(data.get("rewrite") or "").strip()
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    if not isinstance(data["grades"], list) or len(values) != count or any(g not in GRADES for g in values):
        return None
    return values, rewrite


def grader_prompt(question: str, queries: list[str], new: list[dict]) -> str:
    body = "\n\n".join(f"{n}. [{c['cite']}]\n{c['text']}" for n, c in enumerate(new, 1))
    used = "\n".join(f"- {q}" for q in queries)
    return f"질문: {question}\n지금까지 쓴 검색어(마지막이 이번 검색어):\n{used}\n\n조각 {len(new)}개:\n{wrap_untrusted('search', body)}"


def generate_prompt(question: str, chunks: list[dict]) -> str:
    evidence = "\n\n".join(f"[{c['cite']}]\n{c['text']}" for c in chunks)
    return (
        f"질문: {question}\n\n[근거]\n{wrap_untrusted('search', evidence)}\n\n"
        "근거에 있는 내용만으로 질문에 답하라. 쓴 내용의 근거마다 위 대괄호의 [문서 제목, 위치]를 글자 그대로 붙인다. "
        "근거로 답할 수 없는 부분은 문서에서 찾지 못했다고 쓴다. 근거에 대한 설명, 사용자에게 되묻는 문장은 쓰지 않는다."
    )


def build(ctx: WorkflowContext) -> StateGraph:
    def ask_grader(prompt: str, count: int) -> tuple[list[str], str]:
        """한 번 다시 묻고, 또 못 읽거나 호출이 실패하면 모두 ambiguous(Q12)."""
        config = ctx.team.config_for(GRADER, ctx.base)
        for _ in range(2):
            try:
                parsed = read_grades(ask_agent(config, prompt), count)
            except Exception as e:
                ctx.progress(f"grader 오류({type(e).__name__}) → 모두 ambiguous")
                break
            if parsed:
                return parsed
        else:
            ctx.progress("grader 답을 읽지 못함 → 모두 ambiguous")
        return ["ambiguous"] * count, ""

    def retrieve(state: State) -> State:
        query = state.get("query") or state["question"]
        round_ = state.get("revisions", 0) + 1
        if round_ == 1:
            ctx.progress(f"워크플로 시작: {NAME}")
        known = {(c["cite"], c["text"]) for c in state.get("chunks", [])}
        hits = ctx.knowledge.store.search(query, top_k=TOP_K)
        new = [{"cite": h.cite, "text": h.text} for h in hits if (h.cite, h.text) not in known]
        label = "조각" if round_ == 1 else "새 조각"
        ctx.progress(f"Retrieve {round_}/{MAX_RETRIES + 1} 완료: {label} {len(new)}개 (검색어: {query})")
        return {"query": query, "queries": [*state.get("queries", []), query], "new": new}

    def grade(state: State) -> State:
        new = state.get("new", [])
        if not new:
            ctx.progress(f"Grade 건너뜀: 새 조각 없음 → {after_grade(state)}")
            return {"rewrite": ""}
        values, rewrite = ask_grader(grader_prompt(state["question"], state["queries"], new), len(new))
        chunks = [*state.get("chunks", []), *({**c, "grade": g} for c, g in zip(new, values))]
        counts = " · ".join(f"{g} {values.count(g)}" for g in GRADES)
        ctx.progress(f"Grade 완료: {counts} → {after_grade({**state, 'chunks': chunks})}")
        return {"chunks": chunks, "rewrite": rewrite}

    def rewrite(state: State) -> State:
        query = state.get("rewrite", "")
        if not query or query in state.get("queries", []):
            query = state["question"]
        return {"query": query, "revisions": state.get("revisions", 0) + 1}

    def generate(state: State) -> State:
        chunks = [c for c in state.get("chunks", []) if c["grade"] in USABLE]
        ctx.progress(f"Generate 시작: main (조각 {len(chunks)}개)")
        answer = ask_agent(replace(ctx.config, tools=[]), generate_prompt(state["question"], chunks))
        result = check_citations(answer, {c["cite"] for c in chunks})
        if not result.cited:
            ctx.progress("출처 검사: ※ 출처 없음")
        elif result.unknown:
            ctx.progress(f"출처 검사: ※ 검색 결과에 없는 출처: {', '.join(result.unknown)}")
        else:
            ctx.progress("출처 검사: 통과")
        return {"status": "completed", "answer": answer}

    def give_up(state: State) -> State:
        ctx.progress(f"재검색 상한({MAX_RETRIES}번)까지 쓸 조각이 없어 근거 없음으로 끝냅니다")
        return {"status": "no_evidence", "answer": NO_EVIDENCE}

    def after_grade(state: State) -> str:
        found = {c["grade"] for c in state.get("chunks", [])}
        if "correct" in found:
            return "generate"
        if state.get("revisions", 0) < MAX_RETRIES:
            return "retrieve"
        return "generate" if "ambiguous" in found else "no_evidence"

    graph = StateGraph(State)
    for name, node in [("retrieve", retrieve), ("grade", grade), ("rewrite", rewrite), ("generate", generate), ("give_up", give_up)]:
        graph.add_node(name, node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    routes = {"generate": "generate", "retrieve": "rewrite", "no_evidence": "give_up"}
    graph.add_conditional_edges("grade", after_grade, routes)
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("give_up", END)
    return graph


def unavailable(ctx: WorkflowContext) -> str | None:
    if ctx.knowledge is None:
        return "문서 색인이 없습니다. uv run python scripts/index_knowledge.py 를 먼저 실행하세요."
    if ctx.team is None:
        return "에이전트 정의가 없어 grader를 부를 수 없습니다."
    return None


WORKFLOW = Workflow(
    name=NAME,
    description="문서를 찾아 grader가 조각마다 관련성을 평가하고, correct가 없으면 검색어를 재작성해 다시 찾은 뒤 고른 조각으로 답한다",
    build=build,
    unavailable=unavailable,
)
