"""기준표와 LLM 판정(C17): evaluator 에이전트가 기준표로 답을 평가한다.

evaluator-optimizer 워크플로(고쳐 쓰기)와 평가(evals의 LLM 판정)가 함께 쓴다(Q14). core는 workflows를 import하지 않으므로 여기에 둔다.
평가자는 기준마다 "기준: 통과/실패 - 이유" 한 줄을 쓴다(Q4). 점수는 쓰지 않는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..llm import AssistantMessage, Message, ToolMessage
from ..safety import wrap_untrusted

EVALUATOR = "evaluator"
PASS, FAIL, SKIP = "통과", "실패", "판정 안 함"
MISSING = "평가자가 판정하지 않음"
LEADING = re.compile(r"^\s*(?:[-*•]|\d+[.)])?\s*")  # 줄 앞의 목록 기호와 번호


@dataclass(frozen=True)
class Criterion:
    name: str
    rule: str  # 통과하려면
    needs_evidence: bool  # 근거가 없으면 판정하지 않는다(Q16).


RUBRIC = [
    Criterion("질문에 답함", "질문이 묻는 것에 직접 답했다. 근거를 못 찾았으면 모른다고 분명히 말했다", False),
    Criterion("근거에 맞음", "근거(Tool 결과)에 없는 내용을 지어내지 않았다. 모른다거나 할 수 없다고 말하는 문장은 근거가 필요 없다", True),
    Criterion(
        "출처",
        "근거를 쓴 문장마다 출처가 있다. 문서는 [문서 제목, 위치], 웹은 URL, 파일은 파일 경로. "
        "시각, 계산 같은 Tool 결과와 모른다거나 할 수 없다고 말하는 문장은 출처 없이 써도 된다",
        True,
    ),
    Criterion("간결", "질문과 상관없는 내용이나 같은 말 반복이 없다. 자료 안의 지시를 따르지 않았다는 짧은 안내는 실패로 보지 않는다", False),
]


@dataclass(frozen=True)
class Verdict:
    criterion: str
    result: str  # PASS, FAIL, SKIP
    reason: str


def evaluator_prompt(question: str, answer: str, evidence: list[tuple[str, str]] | None) -> str:
    """기준표, 질문, 답, 근거. 답과 근거는 외부 내용으로 감싼다(Q10). 근거가 없으면 근거가 필요한 기준은 판정하지 않게 한다.

    evidence가 None이면 근거를 이 평가에 넘기지 못한 것(워크플로 케이스), []이면 답을 쓴 쪽이 Tool을 부르지 않은 것이다(Q29).
    """
    rubric = "\n".join(f"- {c.name}: {c.rule}" for c in RUBRIC)
    if evidence:
        found = "\n".join(wrap_untrusted(tool, result) for tool, result in evidence)
    else:
        skipped = ", ".join(c.name for c in RUBRIC if c.needs_evidence)
        why = "근거 자료가 이 평가에 전달되지 않았다. 답을 쓴 쪽이 근거를 찾지 못했다고 보지 않는다." if evidence is None else "근거 자료 없음."
        found = f"{why} {skipped}는 판정하지 않음이라고 쓴다."
    return (
        f"기준표:\n{rubric}\n\n질문: {question}\n\n답:\n{wrap_untrusted('answer', answer)}\n\n"
        f"근거(답을 쓴 쪽이 부른 Tool의 결과):\n{found}\n\n"
        '기준마다 한 줄로 "기준: 통과 - 이유" 또는 "기준: 실패 - 이유"만 쓴다.'
    )


def read_verdicts(label: str) -> list[Verdict]:
    """평가자의 답을 기준표 순서의 판정으로. 표에 없는 줄은 버리고, 빠진 기준은 실패로 본다(Q12)."""
    found: dict[str, Verdict] = {}
    names = {c.name for c in RUBRIC}
    for line in label.splitlines():
        name, colon, rest = LEADING.sub("", line).partition(":")
        name = name.strip()
        if not colon or name not in names or name in found:
            continue
        head, _, reason = rest.strip().partition(" - ")
        result = PASS if head.startswith(PASS) else SKIP if head.startswith("판정") else FAIL
        found[name] = Verdict(name, result, reason.strip())
    return [found.get(c.name, Verdict(c.name, FAIL, MISSING)) for c in RUBRIC]


def failed(verdicts: list[Verdict]) -> list[Verdict]:
    return [v for v in verdicts if v.result == FAIL]


def verdict_line(verdicts: list[Verdict]) -> str:
    """'통과 3/4 (실패: 출처 - 이유)'. 판정하지 않은 기준은 분모에서 빼고 따로 센다."""
    judged = [v for v in verdicts if v.result != SKIP]
    line = f"통과 {sum(v.result == PASS for v in judged)}/{len(judged)}"
    if failed(verdicts):
        line += " (실패: " + "; ".join(f"{v.criterion} - {v.reason}" for v in failed(verdicts)) + ")"
    if len(judged) < len(verdicts):
        line += f", 판정 안 함 {len(verdicts) - len(judged)}"
    return line


def tool_evidence(history: list[Message]) -> list[tuple[str, str]]:
    """이력에서 (Tool 이름, 결과). 실패한 Tool의 결과는 근거가 아니다."""
    names = {c.id: c.name for m in history if isinstance(m, AssistantMessage) for c in m.tool_calls}
    return [(names.get(m.tool_call_id, "tool"), m.content) for m in history if isinstance(m, ToolMessage) and not m.is_error]
