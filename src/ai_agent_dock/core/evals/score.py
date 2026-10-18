"""평가 판정: 케이스의 규칙을 턴 결과와 대조한다. 규칙을 모두 만족하면 통과, 하나라도 어긋나면 실패다.

규칙 기반만 쓴다(LLM 채점은 나중에). 케이스 파일은 프로젝트 루트의 evals/cases.jsonl이다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

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


def check(case: Case, result: AgentResult) -> list[str]:
    """어긋난 규칙의 설명 목록을 돌려준다. 빈 목록이면 통과다."""
    failures = tool_errors(result)
    tools = called_tools(result)
    answer = result.answer or ""
    for name in case.expect_tools:
        if name not in tools:
            failures.append(f"툴 {name}을 부르지 않았다")
    for name in case.forbid_tools:
        if name in tools:
            failures.append(f"부르면 안 되는 툴 {name}을 불렀다")
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


def regressions(previous_passed: set[str], current_failed: set[str]) -> set[str]:
    """직전 실행에서 통과했는데 이번에 실패한 케이스 id들. 회귀다(D8)."""
    return previous_passed & current_failed
