"""컴팩션: 이력을 줄인다(C9 D5~D9). /compact 명령으로만 실행한다.

1단계: 원문으로 둘 최근 턴을 뺀 오래된 턴에서 큰 Tool 결과를 앞부분만 남기고 자른다.
2단계: 오래된 턴을 요약 하나로 바꾼다. 요약은 남긴 첫 사용자 메시지 앞에 붙여서, 사용자 메시지가 연속되지 않게 한다.
턴 경계(사용자 메시지)에서만 자르므로 Tool 요청과 Tool 결과의 짝은 깨지지 않는다.
요약 함수는 밖에서 받는다(요약 모델 설정은 진입점이 정한다). 이 모듈은 runtime을 import하지 않는다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from ..llm import AssistantMessage, Message, ToolMessage, UserMessage

TOOL_RESULT_CHARS = 500  # 1단계에서 오래된 턴의 Tool 결과를 이만큼만 남긴다
SUMMARY_HEADER = "[이전 대화 요약]"
SUMMARY_INSTRUCTION = (
    "다음 대화를 이어서 대화할 수 있도록 요약하라. 이름, 숫자, 결정, 파일 이름, 사용자가 기억해 달라고 한 사실은 빼지 말고 그대로 적는다. "
    "지어내지 말고 대화에 있는 내용만 쓴다."
)


@dataclass
class CompactReport:
    before_tokens: int
    after_tokens: int
    summarized_turns: int
    trimmed_results: int


def split_turns(history: list[Message]) -> list[list[Message]]:
    """이력을 턴(사용자 메시지로 시작하는 묶음)으로 나눈다. 첫 사용자 메시지 앞의 항목은 첫 턴에 붙인다."""
    turns: list[list[Message]] = []
    for m in history:
        if isinstance(m, UserMessage) or not turns:
            turns.append([m])
        else:
            turns[-1].append(m)
    return turns


def _as_text(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        if isinstance(m, UserMessage):
            lines.append(f"사용자: {m.content}")
        elif isinstance(m, AssistantMessage):
            if m.content:
                lines.append(f"어시스턴트: {m.content}")
            lines += [f"어시스턴트가 Tool 요청: {c.name} {c.arguments}" for c in m.tool_calls]
        elif isinstance(m, ToolMessage):
            lines.append(f"Tool 결과: {m.content}")
    return "\n".join(lines)


def compact(
    history: list[Message],
    *,
    keep_turns: int,
    summarize: Callable[[str], str],
    count: Callable[[list[Message]], int],
) -> tuple[list[Message], CompactReport]:
    """줄인 이력과 보고를 돌려준다. 원래 이력은 바꾸지 않는다. 오래된 턴이 없으면 그대로 돌려준다."""
    keep_turns = max(1, keep_turns)
    before = count(history)
    turns = split_turns(history)
    if len(turns) <= keep_turns:
        return list(history), CompactReport(before, before, 0, 0)
    old, recent = turns[:-keep_turns], turns[-keep_turns:]

    trimmed = 0
    old_messages: list[Message] = []
    for m in (m for turn in old for m in turn):
        if isinstance(m, ToolMessage) and len(m.content) > TOOL_RESULT_CHARS:
            m = replace(m, content=m.content[:TOOL_RESULT_CHARS] + " …(컴팩션으로 잘림)")
            trimmed += 1
        old_messages.append(m)

    summary = summarize(f"{SUMMARY_INSTRUCTION}\n\n{_as_text(old_messages)}").strip()
    first, *rest = recent[0]
    first = replace(first, content=f"{SUMMARY_HEADER}\n{summary}\n\n{first.content}")
    new_history = [first, *rest, *(m for turn in recent[1:] for m in turn)]
    return new_history, CompactReport(before, count(new_history), len(old), trimmed)
