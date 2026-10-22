"""평가 판정: 케이스의 규칙을 턴 결과와 대조한다. 규칙을 모두 만족하면 통과, 하나라도 어긋나면 실패다.

규칙 기반만 쓴다(LLM 채점은 나중에). 케이스 파일은 프로젝트 루트의 evals/cases.jsonl이다.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..agents import DELEGATE
from ..llm import AssistantMessage, ToolMessage
from ..runtime.loop import NOT_RUN_PREFIXES
from ..runtime.types import AgentResult

# 출처 형식: [문서 제목, 위치] (D15)
CITATION = re.compile(r"\[[^\[\]]+,\s*[^\[\]]+\]")


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    question: str
    expect_tools: list[str] = field(default_factory=list)
    forbid_tools: list[str] = field(default_factory=list)
    expect_contains: list[str] = field(default_factory=list)  # 답에 모두 있어야 할 단어
    expect_any: list[str] = field(default_factory=list)  # 답에 하나만 있으면 되는 표현(같은 뜻의 여러 말투)
    forbid_contains: list[str] = field(default_factory=list)  # 답에 있으면 실패인 표현(예: 지어낸 날씨 수치 °C)
    expect_citation: bool | None = None  # True면 출처가 있어야 하고, False면 없어야 하고, None이면 보지 않는다.
    approvals: dict[str, str] = field(default_factory=dict)  # 툴 이름 → 미리 적는 승인 답(approve 또는 deny). 적지 않은 툴은 승인한다. 실행기가 쓴다.
    user_inputs: list[str] = field(default_factory=list)  # 되묻기 케이스에서 사용자 답으로 미리 적는 값. 실행기가 쓴다.
    setup_turns: list[str] = field(default_factory=list)  # 마지막 질문 전에 보낼 사용자 질문들(memory 케이스)
    compact_before: bool = False  # 마지막 질문 전에 컴팩션을 실행한다
    memory_items: list[str] = field(default_factory=list)  # 평가 동안만 쓰는 장기 메모리(실제 data/memory.md는 건드리지 않는다)
    workflow: str | None = None  # 워크플로 이름. 있으면 질문을 워크플로로 실행한다(C10 D22).
    workflow_decisions: list[str] = field(default_factory=list)  # 승인 단계에서 차례로 줄 답: approve, reject, revise:이유. 다 쓰면 approve
    expect_status: str | None = None  # 워크플로 마지막 상태: approved, rejected, no_evidence
    expect_max_revisions: int | None = None  # 고쳐 쓴 횟수 상한
    forbid_executed: list[str] = field(default_factory=list)  # 실행되면 안 되는 툴. 요청은 해도 정책이나 승인으로 막혔으면 통과(C11)
    knowledge: str | None = None  # "fixtures"면 평가 전용 문서 색인을 쓴다(C11 D14). 실행기가 쓴다.
    expect_delegate: bool | None = None  # True면 main이 delegate를 불러야 하고, False면 부르면 안 된다(C12 F6). None이면 보지 않는다.


def load_cases(path: Path) -> list[Case]:
    """케이스 파일을 읽는다. 한 줄이 케이스 하나다(JSON Lines)."""
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(Case(**json.loads(line)))
    return cases


def called_tools(result: AgentResult) -> list[str]:
    """이번 턴에서 모델이 요청한 Tool 이름들(요청 순서대로)."""
    return [c.name for m in result.history if isinstance(m, AssistantMessage) for c in m.tool_calls]


def tool_errors(result: AgentResult) -> list[str]:
    """실행했는데 실패한 Tool 호출의 설명(D21). 거부나 다르게 요청해서 실행하지 않은 호출은 넣지 않는다."""
    names = {c.id: c.name for m in result.history if isinstance(m, AssistantMessage) for c in m.tool_calls}
    return [
        f"툴 {names.get(m.tool_call_id, '?')}이 오류를 냈다: {m.content[:80]}"
        for m in result.history
        if isinstance(m, ToolMessage) and m.is_error and not m.content.startswith(NOT_RUN_PREFIXES)
    ]


def executed_tools(result: AgentResult) -> list[str]:
    """실제로 실행된 Tool 이름들. 거부, 다르게 요청, 정책으로 막혀 실행하지 않은 호출은 뺀다."""
    names = {c.id: c.name for m in result.history if isinstance(m, AssistantMessage) for c in m.tool_calls}
    return [
        names.get(m.tool_call_id, "?")
        for m in result.history
        if isinstance(m, ToolMessage) and not m.content.startswith(NOT_RUN_PREFIXES)
    ]


def check(case: Case, result: AgentResult, children: Iterable[AgentResult] = ()) -> list[str]:
    """어긋난 규칙의 설명 목록을 돌려준다. 빈 목록이면 통과다.

    children은 이번 턴에 delegate로 맡긴 하위 에이전트들의 결과다(C12). 하위 에이전트가 부르거나 실행한 Tool도
    expect_tools, forbid_tools, forbid_executed, Tool 오류에 함께 센다. 답과 출처는 main의 최종 답으로만 본다.
    """
    children = list(children)
    failures = tool_errors(result) + [e for c in children for e in tool_errors(c)]
    main_tools = called_tools(result)
    tools = main_tools + [name for c in children for name in called_tools(c)]
    answer = result.answer or ""
    if case.expect_delegate is True and DELEGATE not in main_tools:
        failures.append("하위 에이전트에게 맡기지 않았다")
    if case.expect_delegate is False and DELEGATE in main_tools:
        failures.append("맡기지 않아야 할 질문을 하위 에이전트에게 맡겼다")
    for name in case.expect_tools:
        if name not in tools:
            failures.append(f"툴 {name}을 부르지 않았다")
    for name in case.forbid_tools:
        if name in tools:
            failures.append(f"부르면 안 되는 툴 {name}을 불렀다")
    executed = executed_tools(result) + [name for c in children for name in executed_tools(c)]
    for name in case.forbid_executed:
        if name in executed:
            failures.append(f"실행되면 안 되는 툴 {name}이 실행됐다")
    for word in case.expect_contains:
        if word not in answer:
            failures.append(f"답에 {word!r}가 없다")
    if case.expect_any and not any(word in answer for word in case.expect_any):
        failures.append(f"답에 {case.expect_any} 중 어느 표현도 없다")
    for word in case.forbid_contains:
        if word in answer:
            failures.append(f"답에 있으면 안 되는 {word!r}가 있다")
    if case.expect_citation is not None:
        has_citation = CITATION.search(answer) is not None
        if case.expect_citation and not has_citation:
            failures.append("답에 출처 [문서 제목, 위치]가 없다")
        if not case.expect_citation and has_citation:
            failures.append("근거가 없는데 출처를 붙였다")
    return failures


def check_workflow(case: Case, status: str, revisions: int) -> list[str]:
    """워크플로 케이스의 마지막 상태와 고쳐 쓴 횟수를 본다(C10 D22)."""
    failures = []
    if case.expect_status and status != case.expect_status:
        failures.append(f"워크플로가 {case.expect_status}가 아니라 {status}로 끝났다")
    if case.expect_max_revisions is not None and revisions > case.expect_max_revisions:
        failures.append(f"고쳐 쓴 횟수 {revisions}가 상한 {case.expect_max_revisions}를 넘었다")
    return failures


def regressions(previous_passed: set[str], current_failed: set[str]) -> set[str]:
    """직전 실행에서 통과했는데 이번에 실패한 케이스 id들. 회귀다(D8)."""
    return previous_passed & current_failed
