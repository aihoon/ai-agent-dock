"""평가 실행: 케이스를 실행해 판정하고, 결과를 data/evals/실행번호.json에 저장하고, 직전 실행과 비교해 회귀를 찾는다.

승인 메뉴와 되묻기는 사람이 답할 수 없으므로 케이스에 미리 적어 둔 답(approvals, user_inputs)을 차례로 쓴다.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from ..hitl import Approval
from ..llm import Message, ToolCall
from ..orchestration import WAITING, Outcome, WorkflowRunner
from ..runtime.types import AgentResult, AgentConfig
from ..session import Session
from .rubric import FAIL, evaluator_prompt, read_verdicts, tool_evidence, verdict_line
from .score import Case, called_tools, check, check_workflow


@dataclass
class CaseResult:
    id: str
    category: str
    runs: int
    passes: int
    failures: list[str] = field(default_factory=list)  # 마지막으로 실패한 실행의 이유
    tools: list[str] = field(default_factory=list)  # 마지막 실행에서 부른 Tool
    answer: str | None = None  # 마지막 실행의 답
    verdicts: list[str] = field(default_factory=list)  # voting 실행마다 "판정 (표 1, 표 2, 표 3)". 판정자 1번과 다수결의 흔들림을 비교한다(C15).
    llm_verdicts: list[str] = field(default_factory=list)  # llm_judge 실행마다 "통과 3/4 (실패: 출처 - 이유)" 또는 "오류: …"(C17 Q17)
    disagreements: int = 0  # 규칙 판정과 LLM 판정이 어긋난 실행 수. 오류와 판정 안 한 기준은 세지 않는다.

    @property
    def passed(self) -> bool:
        return self.passes == self.runs


def scripted_approver(decisions: dict[str, str]) -> Callable[[ToolCall], Approval]:
    """케이스의 approvals(툴 이름 → approve 또는 deny)로 답한다. 적지 않은 툴은 승인한다(평가에서만, 샌드박스 안).

    승인 메뉴가 뜨는 순서가 아니라 툴 이름으로 답한다. 모델이 다른 툴을 먼저 불러도 답이 엉뚱한 툴에 쓰이지 않는다(R6).
    """

    def approve(call: ToolCall) -> Approval:
        kind = decisions.get(call.name, "approve")
        return Approval.approve() if kind == "approve" else Approval.deny("평가 케이스에 적힌 답")

    return approve


def scripted_input(answers: list[str]) -> Callable[[str], str]:
    """되묻기(ask_user)에 케이스의 user_inputs를 차례로 답한다. 다 쓰면 빈 답을 준다."""
    remaining = list(answers)
    return lambda prompt: remaining.pop(0) if remaining else ""


def evaluate(
    case: Case,
    make_config: Callable[[Case], AgentConfig],
    repeat: int = 1,
    compactor: Callable[[list[Message]], list[Message]] | None = None,
    workflows: Callable[[AgentConfig], WorkflowRunner] | None = None,
    delegated: Callable[[], list[AgentResult]] | None = None,
    judge: Callable[[str], str] | None = None,
    session: Callable[[AgentConfig], Session] | None = None,
) -> CaseResult:
    """케이스를 repeat번 실행한다. 실행마다 설정을 새로 만든다(미리 적은 답이 실행마다 처음부터 쓰이도록).

    setup_turns가 있으면 앞선 턴을 먼저 보내고, compact_before면 마지막 질문 전에 compactor로 이력을 줄인다.
    판정은 마지막 질문의 턴만 본다(앞선 턴의 Tool 호출은 세지 않는다).
    workflow가 있으면 질문을 워크플로로 실행하고, 승인 단계에는 workflow_decisions를 차례로 답한다(C10 D22).
    delegated는 그동안 맡긴 하위 에이전트의 결과를 꺼내고 비우는 함수다(C12). 마지막 질문의 것만 판정에 쓴다.
    judge는 evaluator에게 평가자 프롬프트를 주고 답을 받는 함수다(C17). llm_judge 케이스에서 실행마다 부르고 기록만 한다.
    session은 main 설정으로 채팅과 같은 세션을 만드는 함수다(C21 Q20). 넘기기가 있으면 앞선 턴의 대화 에이전트가 마지막 질문으로 이어진다.
    판정의 Tool은 마지막 턴에서 돈 모든 에이전트의 호출을 센다. expect_agent는 마지막 턴이 끝났을 때 대화 중인 에이전트다.
    """
    take = delegated or (lambda: [])
    outcome = CaseResult(id=case.id, category=case.category, runs=repeat, passes=0)
    for _ in range(repeat):
        config = make_config(case)
        if case.workflow:
            failures, result = run_workflow_case(case, workflows(config))
            outcome.answer = result.answer
            if result.verdict:
                outcome.verdicts.append(f"{result.verdict} ({', '.join(result.ballots or [])})")
            if case.llm_judge and judge:  # 워크플로는 이력이 없어 근거 없이 본다(Q16). None은 "넘기지 못함"(Q29)
                record_llm_judge(outcome, judge, evaluator_prompt(case.question, result.answer or "", None), not failures)
            if failures:
                outcome.failures = failures
            else:
                outcome.passes += 1
            continue
        chat = (session or Session)(config)
        chat.trace_started = True  # 평가는 실행 처음에 세션 설정을 한 번만 기록한다.
        for question in case.setup_turns:
            chat.ask(question)
        if case.compact_before and compactor is not None:
            chat.history = compactor(chat.history)
        take()  # 앞선 턴에서 맡긴 것은 버린다.
        before = len(chat.history)
        full = chat.ask(case.question)
        result = replace(full, history=full.history[before:])
        children = take()
        failures = check(case, result, children)
        if case.expect_agent and chat.agent != case.expect_agent:
            failures.append(f"대화 에이전트: 기대 {case.expect_agent}, 실제 {chat.agent}")
        outcome.tools = called_tools(result) + [name for c in children for name in called_tools(c)]
        outcome.answer = result.answer
        if case.llm_judge and judge:
            evidence = tool_evidence(result.history) + [e for c in children for e in tool_evidence(c.history)]
            record_llm_judge(outcome, judge, evaluator_prompt(case.question, result.answer or "", evidence), not failures)
        if failures:
            outcome.failures = failures
        else:
            outcome.passes += 1
    return outcome


def record_llm_judge(outcome: CaseResult, judge: Callable[[str], str], prompt: str, rule_passed: bool) -> None:
    """LLM 판정 한 번을 기록하고, 규칙 판정과 어긋나면 센다. 평가자 호출이 실패하면 \"오류\"만 남긴다(Q18)."""
    try:
        verdicts = read_verdicts(judge(prompt))
    except Exception as e:
        outcome.llm_verdicts.append(f"오류: {type(e).__name__}")
        return
    outcome.llm_verdicts.append(verdict_line(verdicts))
    outcome.disagreements += rule_passed == any(v.result == FAIL for v in verdicts)


def run_workflow_case(case: Case, runner: WorkflowRunner) -> tuple[list[str], Outcome]:
    """(어긋난 규칙, 워크플로 결과). 판정은 최종 답과 마지막 상태, 고쳐 쓴 횟수, 고른 경로, 판정으로 한다."""
    decisions = list(case.workflow_decisions)
    result = runner.start(case.workflow, case.question)
    while result.status == WAITING:
        result = runner.resume(result.run_id, decisions.pop(0) if decisions else "approve")
    answer = AgentResult(answer=result.answer, status="completed", steps=0, history=[])
    failures = check_workflow(case, result.status, result.revisions, result.route, routes=result.routes, verdict=result.verdict, workers=result.workers)
    return check(case, answer) + failures, result


def last_passed(directory: Path) -> set[str] | None:
    """가장 최근 실행 결과에서 통과한 케이스 id들. 이전 실행이 없으면 None."""
    files = sorted(directory.glob("*.json")) if directory.is_dir() else []
    if not files:
        return None
    data = json.loads(files[-1].read_text(encoding="utf-8"))
    return {c["id"] for c in data["cases"] if c["passed"]}


def save_run(directory: Path, run_id: str, provider: str, model: str, repeat: int, results: list[CaseResult]) -> Path:
    """실행 결과를 data/evals/실행번호.json으로 저장한다."""
    directory.mkdir(parents=True, exist_ok=True)
    passed = sum(r.passed for r in results)
    data = {
        "run": run_id,
        "provider": provider,
        "model": model,
        "repeat": repeat,
        "pass_rate": passed / len(results) if results else 0.0,
        "cases": [{**asdict(r), "passed": r.passed} for r in results],
    }
    path = directory / f"{run_id}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def summarize(results: list[CaseResult], regressed: set[str]) -> str:
    """화면에 보여 줄 요약: 통과율, 카테고리별 결과, 실패한 케이스와 이유, 회귀."""
    passed = sum(r.passed for r in results)
    lines = [f"통과 {passed}/{len(results)} ({passed / len(results):.0%})" if results else "케이스가 없습니다."]
    for category in sorted({r.category for r in results}):
        group = [r for r in results if r.category == category]
        lines.append(f"  {category}: {sum(r.passed for r in group)}/{len(group)}")
    for r in results:
        if not r.passed:
            mark = " [회귀]" if r.id in regressed else ""
            lines.append(f"  실패 {r.id}{mark} ({r.passes}/{r.runs}): {'; '.join(r.failures)}")
    if regressed:
        lines.append(f"회귀 {len(regressed)}개: {', '.join(sorted(regressed))}")
    judged = [r for r in results if r.llm_verdicts]
    if judged:  # 기록만 한다. 통과율에는 넣지 않는다(C17 Q6, Q17).
        off = [r for r in judged if r.disagreements]
        lines.append(f"LLM 판정: {len(judged)}케이스, 규칙 판정과 어긋남 {sum(r.disagreements for r in off)}번 ({', '.join(r.id for r in off) or '없음'})")
    return "\n".join(lines)
