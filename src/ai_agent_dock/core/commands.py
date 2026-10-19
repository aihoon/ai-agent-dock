"""슬래시 명령(C9 D25): 명령의 이름, 사용법, 설명, 처리를 한 곳에 둔다.

응용(CLI, 나중의 브라우저)은 입력을 find_command로 찾아 실행하고 돌려받은 문장을 화면에 보여 주기만 한다.
처리 함수는 화면에 직접 출력하지 않고 보여 줄 문장을 돌려준다. 실행 전 확인이 필요한 명령은 confirm이 확인 문장을 주고,
확인은 응용이 받는다(터미널은 y/N, 브라우저는 대화상자). 응용에만 있는 명령(/exit)은 응용에 둔다.
의존 방향: 이 모듈은 session, memory, llm, observability, orchestration을 import한다. runtime과 session은 이 모듈을 import하지 않는다.
워크플로가 사람 승인에서 멈추면 waiting에 승인 요청을 두고, 응용이 메뉴로 답을 받아 answer_workflow를 부른다(C10 D6).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, replace

from .llm import AssistantMessage, LLMRequest, Message, Tool, UserMessage
from .llm.tokens import count as count_tokens, request_tokens
from .memory import MemoryStore
from .observability import new_session_id
from .orchestration import WAITING, Outcome, WorkflowRunner
from .retrieval import Knowledge
from .runtime.types import Usage
from .session import Session
from .session.compact import compact
from .session.store import SessionRecord, SessionStore, auto_name
from .skills import SkillMeta, read_skill
from .tools import ToolRegistry

SHOW = 80  # /history에서 한 항목을 이만큼만 보여 준다.


@dataclass
class CommandContext:
    """명령이 쓰는 것과, 명령 사이에 이어지는 상태(지금 세션의 저장 기록)."""

    session: Session
    registry: ToolRegistry
    skills: list[SkillMeta]
    store: SessionStore
    memory: MemoryStore
    make_prompt: Callable[[], str]  # 메모리가 바뀌면 시스템 프롬프트를 다시 조립한다(C9 D20).
    summarize: Callable[[str], str]  # 컴팩션 요약 함수(요약 모델은 응용이 정한다, C9 D8).
    knowledge: Knowledge | None = None
    keep_turns: int = 4  # COMPACT_KEEP_TURNS
    compact_warn_tokens: int = 100_000  # COMPACT_WARN_TOKENS
    memory_warn_tokens: int = 2_000  # MEMORY_WARN_TOKENS
    record: SessionRecord | None = None  # 지금 세션의 저장 기록. 첫 턴(또는 이름 정하기)에 만든다.
    workflows: WorkflowRunner | None = None  # 워크플로 실행기(C10). None이면 워크플로 명령을 쓸 수 없다.
    waiting: Outcome | None = None  # 사람 승인을 기다리는 워크플로. 응용이 메뉴를 보여 주고 answer_workflow로 답한다.

    @property
    def tracer(self):
        return self.session.settings.tracer

    def ensure_record(self, name: str) -> SessionRecord:
        if self.record is None:
            session_id = self.tracer.session_id if self.tracer else new_session_id()
            self.record = SessionRecord(id=session_id, name=self.store.unique_name(name), provider=self.session.settings.config.provider)
        return self.record

    def save(self) -> None:
        if self.record is not None:
            self.record.history = self.session.history
            self.store.save(self.record)

    def after_turn(self, question: str) -> str | None:
        """턴이 끝나면 자동 저장하고(C9 D11), 이력이 기준보다 크면 안내 문장을 돌려준다(C9 D5)."""
        self.ensure_record(auto_name(question))
        self.save()
        if self.session.history_tokens() > self.compact_warn_tokens:
            return "※ 이력이 기준보다 큽니다. /compact로 줄일 수 있습니다."
        return None

    def start_fresh(self, history: list[Message]) -> None:
        self.session.history = history
        self.session.last_usage, self.session.total_usage = Usage(), Usage()
        self.session.trace_started = False  # 바꾼 세션의 추적 파일에도 다음 턴 전에 세션 설정을 쓴다(C9 D27).
        if self.knowledge:
            self.knowledge.seen.clear()

    def reload_prompt(self) -> None:
        self.session.settings = replace(self.session.settings, system_prompt=self.make_prompt())

    def runner(self) -> WorkflowRunner:
        """워크플로 실행기. 노드 안 에이전트가 지금 설정(메모리를 반영한 시스템 프롬프트)을 쓰게 맞춘다."""
        self.workflows.context.settings = self.session.settings
        self.session.record_start()  # 워크플로가 이 세션의 첫 동작이어도 추적 파일에 세션 설정을 먼저 쓴다.
        return self.workflows

    def answer_workflow(self, decision: str) -> str:
        """승인 요청에 답한다: approve, reject, revise:이유(C10 D7)."""
        return _report(self, self.runner().resume(self.waiting.run_id, decision))


@dataclass(frozen=True)
class Command:
    usage: str  # 예: "/memory-add 내용". 첫 단어가 명령 이름이다.
    help: str
    run: Callable[[CommandContext, str], str]  # (문맥, 인자) → 보여 줄 문장
    confirm: Callable[[CommandContext, str], str | None] | None = None  # 실행 전 확인 문장. None을 돌려주면 확인 없이 실행한다.

    @property
    def name(self) -> str:
        return self.usage.split()[0]


def brief(message: Message) -> str:
    text = (message.content or "").replace("\n", " ")
    text = text if len(text) <= SHOW else text[:SHOW] + "…"
    if message.role == "assistant" and message.tool_calls:
        text += f" [Tool 요청: {', '.join(c.name for c in message.tool_calls)}]"
    if message.role == "tool" and message.is_error:
        text = "(오류) " + text
    return f"{message.role:9s}: {text}"


def describe_tool(tool: Tool, source: str) -> str:
    """Tool 하나를 `이름(인자, ...) [출처]`와 설명 두 줄로 설명한다. 필수 인자 뒤에는 *를 붙인다."""
    schema = tool.parameters
    required = set(schema.get("required", []))
    args = ", ".join(f"{name}{'*' if name in required else ''}" for name in schema.get("properties", {}))
    first_line = (tool.description.splitlines() or [""])[0]  # 외부 서버의 설명은 길 수 있어 첫 줄만 보여 준다.
    return f"{tool.name}({args})  [{source}]\n      {first_line[: 2 * SHOW]}"


def _reset(ctx: CommandContext, arg: str) -> str:
    ctx.session.reset()
    if ctx.knowledge:
        ctx.knowledge.seen.clear()  # 이력과 함께 검색 출처도 비운다.
    return "대화 이력을 비웠습니다."


def _history(ctx: CommandContext, arg: str) -> str:
    history = ctx.session.history
    return "\n".join([f"(메시지 {len(history)}개)", *(f"  {brief(m)}" for m in history)])


def _tools(ctx: CommandContext, arg: str) -> str:
    tools = ctx.session.settings.tools
    return "\n".join([f"(Tool {len(tools)}개, *는 필수 인자)", *(f"  {describe_tool(t, ctx.registry.source_of(t.name))}" for t in tools)])


def _skills(ctx: CommandContext, arg: str) -> str:
    if not ctx.skills:
        return "스킬이 없습니다(skills/ 폴더에 SKILL.md를 추가하세요)."
    return "\n".join(f"  {m.name}: {m.description}" for m in ctx.skills)


def _skill(ctx: CommandContext, arg: str) -> str:
    if not arg:
        return "사용법: /skill 이름 (목록은 /skills)"
    match = next((m for m in ctx.skills if m.name == arg), None)
    if match is None:
        return f"없는 스킬: {arg!r}. 목록은 /skills 로 보세요."
    ctx.session.add_user_message(f"다음 스킬의 지침을 따라 이후 답하라: {match.name}\n\n{read_skill(match).body}")
    return f"스킬 '{match.name}'의 지침을 대화에 넣었습니다. 이어서 질문하세요."


def _tokens(ctx: CommandContext, arg: str) -> str:
    last, total = ctx.session.last_usage, ctx.session.total_usage
    return "\n".join([
        f"마지막 턴: 입력 {last.input_tokens:,}(캐시 {last.cached_input_tokens:,}) / 출력 {last.output_tokens:,}",
        f"세션 누적: 입력 {total.input_tokens:,}(캐시 {total.cached_input_tokens:,}) / 출력 {total.output_tokens:,}",
        f"이력 크기: {ctx.session.history_tokens():,} / 기준 {ctx.compact_warn_tokens:,} (다음 호출에 보낼 토큰 수)",
    ])


def _compact(ctx: CommandContext, arg: str) -> str:
    session = ctx.session
    if not session.history:
        return "줄일 이력이 없습니다."

    def tokens_of(history: list[Message]) -> int:
        s = session.settings
        return request_tokens(LLMRequest(config=s.config, messages=history, system_prompt=s.system_prompt, tools=s.tools))

    try:
        new_history, report = compact(session.history, keep_turns=ctx.keep_turns, summarize=ctx.summarize, count=tokens_of)
    except Exception as e:  # 요약 모델 호출 실패 등. 이력은 그대로 둔다.
        return f"컴팩션 실패: {type(e).__name__}: {e}"
    if report.summarized_turns == 0:
        return f"최근 {ctx.keep_turns}턴뿐이라 줄일 것이 없습니다. (이력 크기 {report.before_tokens:,})"
    session.history = new_history
    ctx.save()
    if ctx.tracer:
        ctx.tracer.record("compact", None, before_tokens=report.before_tokens, after_tokens=report.after_tokens, summarized_turns=report.summarized_turns, trimmed_results=report.trimmed_results)
    return f"컴팩션: 오래된 {report.summarized_turns}턴을 요약했습니다(잘린 Tool 결과 {report.trimmed_results}개). 이력 크기 {report.before_tokens:,} → {report.after_tokens:,}"


def _sessions(ctx: CommandContext, arg: str) -> str:
    """저장된 세션 목록. 지금 세션에 표시를 붙인다(C9 D13)."""
    records = ctx.store.list()
    if not records:
        return "저장된 세션이 없습니다."
    lines = []
    for n, r in enumerate(records, start=1):
        mark = " ← 지금" if ctx.record is not None and r.id == ctx.record.id else ""
        lines.append(f"  {n}. {r.name}  ({r.updated.replace('T', ' ')}, {r.turns}턴){mark}")
    return "\n".join([*lines, "이어 가려면 /session 번호 를 입력하세요."])


def _session(ctx: CommandContext, arg: str) -> str:
    """/sessions 목록의 번호로 세션을 골라 이어 간다(C9 D13, D25)."""
    records = ctx.store.list()
    if not records:
        return "저장된 세션이 없습니다."
    if not arg.isdigit() or not 1 <= int(arg) <= len(records):
        return f"사용법: /session 번호 (1~{len(records)}, 목록은 /sessions)"
    ctx.save()  # 바꾸기 전 세션을 저장해 둔다.
    ctx.record = ctx.store.load(records[int(arg) - 1].id, ctx.session.settings.config.provider)
    ctx.start_fresh(ctx.record.history)
    if ctx.tracer:
        ctx.tracer.session_id, ctx.tracer.turn = ctx.record.id, ctx.record.turns  # 같은 파일에서 턴 번호가 겹치지 않게 이어 센다
        ctx.tracer.record("session_resume", None, name=ctx.record.name)
    return f"세션 '{ctx.record.name}'을 이어 갑니다({ctx.record.turns}턴)."


def _session_name(ctx: CommandContext, arg: str) -> str:
    if not arg:
        return "사용법: /session-name 새이름"
    old_name = ctx.record.name if ctx.record else None
    try:
        ctx.store.rename(ctx.ensure_record(arg), arg)
    except ValueError as e:  # 빈 이름, 겹치는 이름
        return str(e)
    if ctx.tracer:
        ctx.tracer.record("session_rename", None, old_name=old_name, new_name=arg)
    return f"세션 이름: {arg}"


def _session_new(ctx: CommandContext, arg: str) -> str:
    ctx.save()
    ctx.record = None
    ctx.start_fresh([])
    if ctx.tracer:
        ctx.tracer.session_id, ctx.tracer.turn = new_session_id(), 0
    return "새 세션을 시작합니다."


def _memory(ctx: CommandContext, arg: str) -> str:
    items = ctx.memory.items()
    if not items:
        return "장기 메모리가 비어 있습니다. /memory-add 내용 으로 더하세요."
    lines = [f"  {n}. {text}" for n, text in enumerate(items, start=1)]
    size = count_tokens("\n".join(items))
    if size > ctx.memory_warn_tokens:
        lines.append(f"※ 장기 메모리가 {size:,} 토큰입니다(기준 {ctx.memory_warn_tokens:,}). 매 호출마다 보내므로 줄이는 것이 좋습니다.")
    return "\n".join(lines)


def _memory_add(ctx: CommandContext, arg: str) -> str:
    try:
        n = ctx.memory.add(arg)
    except ValueError as e:
        return f"{e} 사용법: /memory-add 내용"
    ctx.reload_prompt()
    return f"장기 메모리 {n}번에 저장했습니다. 다음 턴부터 반영합니다."


def _memory_number(ctx: CommandContext, arg: str) -> int | None:
    return int(arg) if arg.isdigit() and 1 <= int(arg) <= len(ctx.memory.items()) else None


def _memory_delete_confirm(ctx: CommandContext, arg: str) -> str | None:
    n = _memory_number(ctx, arg)
    return None if n is None else f"'{ctx.memory.items()[n - 1]}'을 지울까요?"


def _memory_delete(ctx: CommandContext, arg: str) -> str:
    n = _memory_number(ctx, arg)
    if n is None:
        return f"사용법: /memory-delete 번호 (1~{len(ctx.memory.items())})"
    ctx.memory.remove(n)
    ctx.reload_prompt()
    return f"{n}번을 지웠습니다. 다음 턴부터 반영합니다."


STATUS = {"approved": "승인", "rejected": "거부", "no_evidence": "근거 없음"}


def _report(ctx: CommandContext, outcome: Outcome) -> str:
    """워크플로 결과를 문장으로. 승인 대기면 승인 요청을 두고, 끝났으면 결과를 채팅 이력에 한 턴으로 붙인다(C10 D5)."""
    if outcome.status == WAITING:
        ctx.waiting = outcome
        return f"[승인 요청] {outcome.workflow} (고쳐 쓴 횟수 {outcome.revisions})\n{outcome.draft}"
    ctx.waiting = None
    if outcome.answer:  # 승인과 근거 없음만 붙인다. 거부는 붙이지 않는다.
        ctx.session.history = [*ctx.session.history, UserMessage(content=outcome.question), AssistantMessage(content=outcome.answer)]
        ctx.ensure_record(auto_name(outcome.question))
        ctx.save()
    label = STATUS.get(outcome.status, outcome.status)
    return f"워크플로 끝: {label} (고쳐 쓴 횟수 {outcome.revisions})" + (f"\n{outcome.answer}" if outcome.answer else "")


def _no_workflows(ctx: CommandContext) -> str | None:
    return None if ctx.workflows else "워크플로를 쓸 수 없습니다."


def _workflows(ctx: CommandContext, arg: str) -> str:
    if reason := _no_workflows(ctx):
        return reason
    lines = ["워크플로:", *(f"  {w.name}: {w.description}" for w in ctx.workflows.workflows.values())]
    pending = ctx.runner().pending()
    lines.append("승인 대기:" if pending else "승인 대기: 없음")
    lines += [f"  {n}. {o.workflow}  {o.question[:30]}  ({o.started.replace('T', ' ')})" for n, o in enumerate(pending, start=1)]
    return "\n".join([*lines, "시작은 /workflow 이름 요청, 이어 가기는 /workflow-resume 번호"])


def _workflow(ctx: CommandContext, arg: str) -> str:
    if reason := _no_workflows(ctx):
        return reason
    name, _, question = arg.partition(" ")
    if not question.strip():
        return "사용법: /workflow 이름 요청 (목록은 /workflows)"
    try:
        return _report(ctx, ctx.runner().start(name, question.strip()))
    except ValueError as e:  # 없는 워크플로, 문서 색인 없음
        return str(e)


def _workflow_resume(ctx: CommandContext, arg: str) -> str:
    if reason := _no_workflows(ctx):
        return reason
    pending = ctx.runner().pending()
    if not arg.isdigit() or not 1 <= int(arg) <= len(pending):
        return f"사용법: /workflow-resume 번호 (1~{len(pending)}, 목록은 /workflows)" if pending else "승인 대기 중인 워크플로가 없습니다."
    return _report(ctx, pending[int(arg) - 1])


COMMANDS = [
    Command("/reset", "대화 이력 비우기", _reset),
    Command("/history", "이력 보기", _history),
    Command("/tools", "사용 중인 Tool 보기", _tools),
    Command("/skills", "스킬 목록 보기", _skills),
    Command("/skill 이름", "스킬 지침을 대화에 넣기", _skill),
    Command("/tokens", "마지막 턴, 세션 누적, 이력 크기 토큰 보기", _tokens),
    Command("/compact", "오래된 턴을 요약해 이력 줄이기", _compact),
    Command("/sessions", "저장된 세션 목록 보기", _sessions),
    Command("/session 번호", "목록의 번호로 세션을 골라 이어 가기", _session),
    Command("/session-name 새이름", "지금 세션 이름 바꾸기", _session_name),
    Command("/session-new", "새 세션 시작", _session_new),
    Command("/memory", "장기 메모리 보기", _memory),
    Command("/memory-add 내용", "장기 메모리에 더하기", _memory_add),
    Command("/memory-delete 번호", "장기 메모리에서 지우기(확인 후)", _memory_delete, confirm=_memory_delete_confirm),
    Command("/workflows", "워크플로 목록과 승인 대기 실행 보기", _workflows),
    Command("/workflow 이름 요청", "워크플로 시작(예: /workflow answer-check 질문)", _workflow),
    Command("/workflow-resume 번호", "승인 대기 중인 워크플로 이어 가기", _workflow_resume),
]


def find_command(line: str, commands: list[Command] = COMMANDS) -> tuple[Command, str] | None:
    """입력 줄에서 명령과 인자를 찾는다. 없는 명령이면 None."""
    name, _, arg = line.strip().partition(" ")
    command = next((c for c in commands if c.name == name), None)
    return None if command is None else (command, arg.strip())


def help_text(app_commands: list[tuple[str, str]] = ()) -> str:
    """명령 목록. 응용에만 있는 명령(사용법, 설명)을 앞에 붙인다."""
    rows = [*app_commands, *((c.usage, c.help) for c in COMMANDS)]
    return "명령 목록:\n" + "\n".join(f"  {usage}{' ' * (22 - _width(usage))}{text}" for usage, text in rows)


def _width(text: str) -> int:
    """터미널에 보이는 폭. 한글처럼 넓은 글자는 두 칸이다."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
