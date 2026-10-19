"""대화형 CLI. 여러 턴의 대화를 이어 가며, 앞의 질문과 답을 기억한다.

사용법:
  uv run python examples/chat.py                       # .env의 LLM_PROVIDER로 대화
  uv run python examples/chat.py --provider anthropic  # 제공자를 인자로 지정(환경 변수보다 우선)
  uv run python examples/chat.py --max-steps 5         # 한 턴에서 모델을 호출하는 횟수 상한
  uv run python examples/chat.py --no-stream           # 일괄 응답(생략하면 .env의 AGENT_STREAM, 없으면 스트리밍)
  uv run python examples/chat.py --mcp-config mcp.json # MCP 서버의 Tool도 함께 사용(생략하면 .env의 MCP_CONFIG)

명령: /help로 목록을 본다. /exit와 /help는 이 파일에, 나머지 명령은 core/commands.py에 있다.
스트리밍이면 답의 글자를 도착하는 대로 보여 주고, 일괄 응답이면 턴이 끝난 뒤 한 번에 보여 준다.
어느 쪽이든 진행 상황(모델 호출, Tool 실행)은 한 줄씩 보여 준다. 턴 도중 Ctrl+C는 그 턴만 취소한다.
"""

import argparse
import logging
import sys
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from ai_agent_dock.core.commands import CommandContext, find_command, help_text
from ai_agent_dock.core.hitl import AuditLog, ask_in_terminal
from ai_agent_dock.core.llm import LLMRequest, Tool, UserMessage, call_llm, config_from_env
from ai_agent_dock.core.memory import MemoryStore
from ai_agent_dock.core.observability import Tracer
from ai_agent_dock.core.retrieval import Knowledge, check_citations, open_knowledge
from ai_agent_dock.core.runtime.prompt import build_system_prompt
from ai_agent_dock.core.runtime.types import AgentSettings
from ai_agent_dock.core.session import Session
from ai_agent_dock.core.session.store import SessionStore
from ai_agent_dock.core.skills import SkillMeta, scan_skills
from ai_agent_dock.core.toolkit import make_builtin_tools
from ai_agent_dock.core.tools import DEFAULT_MAX_RESULT_CHARS, ToolRegistry
from ai_agent_dock.env import env_bool, env_float, env_int, env_str

AUDIT_PATH = Path(__file__).resolve().parent.parent / "logs" / "audit.jsonl"  # 실행 기록(감사 로그). logs/는 git에서 제외한다.
TRACE_DIR = Path(__file__).resolve().parent.parent / "logs" / "sessions"  # 추적 기록(세션별 파일). TRACE_ENABLED=false면 쓰지 않는다.
APP_COMMANDS = [("/help", "명령 목록 보기"), ("/exit", "종료")]  # 응용에만 있는 명령. 나머지 명령은 core/commands.py에 있다(C9 D25).
HELP = help_text(APP_COMMANDS)
SESSIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "sessions"  # 세션 저장(이어 하기용). data/는 git에서 제외한다.
MEMORY_PATH = Path(__file__).resolve().parent.parent / "data" / "memory.md"  # 장기 메모리. data/는 git에서 제외한다.
SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"  # 프로젝트 루트의 skills/
KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"  # 프로젝트 루트의 knowledge/(검색할 문서)
INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "index.sqlite"  # 문서 색인(문서에서 다시 만들 수 있다). data/는 git에서 제외한다.


class Progress:
    """모델 호출과 Tool 실행을 감싸서 진행 상황을 한 줄씩 출력한다. 루프 코드는 건드리지 않는다.

    스트리밍으로 글자 조각이 출력 중이면 줄이 끝나지 않은 상태(open)다. 다음 진행 줄은 그 줄을 먼저 끝내고 출력한다.
    """

    def __init__(self):
        self.step = 0
        self.open = False  # 글자 조각을 출력하다 줄바꿈을 아직 못 한 상태

    def begin_turn(self):
        self.step = 0

    def end_line(self):
        """출력 중인 글자 줄이 있으면 줄바꿈으로 끝낸다."""
        if self.open:
            print(flush=True)
            self.open = False

    def on_text_delta(self, text: str):
        print(text, end="", flush=True)
        self.open = True

    def _say(self, message: str):
        self.end_line()
        print(message, flush=True)

    def wrap_llm(self, llm):
        def traced(request, on_text_delta=None):
            self.step += 1
            self._say(f"  · 모델 호출 중 (스텝 {self.step})")
            return llm(request, on_text_delta=on_text_delta)

        return traced

    def wrap_tool(self, tool: Tool) -> Tool:
        def traced(**kwargs):
            self._say(f"  · Tool 실행: {tool.name}")
            return tool.func(**kwargs)

        return replace(tool, func=traced)


def main():
    parser = argparse.ArgumentParser(description="대화형 CLI")
    parser.add_argument("--provider", help="LLM 제공자(openai, anthropic). 생략하면 .env의 LLM_PROVIDER, 없으면 openai")
    parser.add_argument("--max-steps", type=int, help="한 턴에서 모델을 호출하는 횟수 상한(생략하면 .env의 AGENT_MAX_STEPS 또는 10)")
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        help="스트리밍(--stream) 또는 일괄 응답(--no-stream). 생략하면 .env의 AGENT_STREAM, 없으면 스트리밍",
    )
    parser.add_argument(
        "--mcp-config",
        help="연결할 MCP 서버 목록(JSON) 경로. 생략하면 .env의 MCP_CONFIG, 없으면 MCP를 쓰지 않음(mcp.example.json 참고)",
    )
    args = parser.parse_args()

    load_dotenv()  # 시작점에서 .env를 한 번만 읽는다.
    logging.basicConfig(level=logging.ERROR)  # 진행 표시와 섞이지 않게 로그는 오류만 보인다.
    try:
        # 우선순위: 인자 > 환경 변수 AGENT_STREAM > 기본값(스트리밍)
        stream = args.stream if args.stream is not None else env_bool("AGENT_STREAM", True)
        config = config_from_env(args.provider)  # 모델 설정은 시작할 때 한 번 만든다.
    except ValueError as e:
        raise SystemExit(f"설정 오류: {e}")

    progress = Progress()
    llm = progress.wrap_llm(call_llm)
    with ExitStack() as stack:  # MCP 서버 연결은 프로그램이 끝날 때 이 with가 정리한다.
        try:
            skills = scan_skills(SKILLS_DIR)
        except ValueError as e:  # 스킬 형식 오류 등
            raise SystemExit(f"설정 오류: {e}")
        if skills:
            print(f"스킬 {len(skills)}개를 찾았습니다.")
        knowledge = None
        try:
            knowledge = open_knowledge(INDEX_PATH)
        except Exception as e:  # 색인 파일이 깨졌거나 열 수 없으면 문서 검색만 끄고 계속한다.
            print(f"문서 검색을 끕니다: {type(e).__name__}: {e}")
        if knowledge:
            print(f"문서 검색을 켰습니다(문서 {knowledge.files}개, 청크 {knowledge.chunks}개).")
        elif any(KNOWLEDGE_DIR.glob("*.md")):
            print("문서 색인이 없습니다. uv run python scripts/index_knowledge.py 를 먼저 실행하세요.")
        try:
            registry = build_registry(args, stack, skills=skills, knowledge=knowledge)
        except ValueError as e:  # 설정 파일 오류, Tool 이름 겹침 등
            raise SystemExit(f"설정 오류: {e}")
        audit = AuditLog(AUDIT_PATH)
        memory = MemoryStore(MEMORY_PATH)

        def make_prompt() -> str:  # 메모리가 바뀌면 다시 조립한다(다음 턴부터 반영, C9 D20).
            return build_system_prompt(skills=skills, cite_docs=knowledge is not None, memory=memory.items())

        system_prompt = make_prompt()
        settings = AgentSettings(
            config=config,
            tools=[progress.wrap_tool(t) for t in registry.tools()],
            system_prompt=system_prompt,
            max_steps=args.max_steps,
            llm=llm,
            approve=lambda call: ask_in_terminal(call, interactive=sys.stdin.isatty()),
            audit=audit,
            tracer=Tracer(TRACE_DIR if env_bool("TRACE_ENABLED", True) else None),
        )
        session = Session(settings)
        ctx = CommandContext(
            session=session,
            registry=registry,
            skills=skills,
            store=SessionStore(SESSIONS_DIR),
            memory=memory,
            make_prompt=make_prompt,
            summarize=make_summarizer(config),
            knowledge=knowledge,
            keep_turns=env_int("COMPACT_KEEP_TURNS", 4),
            compact_warn_tokens=env_int("COMPACT_WARN_TOKENS", 100_000),
            memory_warn_tokens=env_int("MEMORY_WARN_TOKENS", 2_000),
        )
        repl(session, ctx, progress, stream)


def build_registry(args, stack: ExitStack, *, skills: list[SkillMeta], knowledge: Knowledge | None) -> ToolRegistry:
    """빌트인 Tool(core/toolkit가 조립)과, 설정 파일에 적은 MCP 서버의 Tool을 한 레지스트리에 모은다. 설정 파일이 없으면 빌트인 Tool만 쓴다."""
    registry = ToolRegistry(max_result_chars=env_int("TOOL_RESULT_MAX_CHARS", DEFAULT_MAX_RESULT_CHARS))
    for source, tool in make_builtin_tools(skills=skills, knowledge=knowledge):
        registry.add(tool, source=source)

    config_path = args.mcp_config or env_str("MCP_CONFIG", "")  # 우선순위: 인자 > 환경 변수 > 사용 안 함
    if config_path:
        from ai_agent_dock.core.tools.mcp import connect_mcp_servers, load_config  # MCP SDK는 설정했을 때만 가져온다.

        servers = load_config(config_path)
        print(f"MCP 서버 {len(servers)}개에 연결합니다(처음 실행은 패키지를 내려받느라 오래 걸릴 수 있습니다)...", flush=True)
        connected = stack.enter_context(
            connect_mcp_servers(
                servers,
                call_timeout=env_float("MCP_TOOL_TIMEOUT", 30.0),
                start_timeout=env_float("MCP_START_TIMEOUT", 60.0),
            )
        )
        for server, tools in connected.items():
            registry.add_all(tools, source=f"mcp:{server}")
            print(f"  MCP 서버 {server}: Tool {len(tools)}개 연결됨")
    return registry


def make_summarizer(config):
    """컴팩션 요약 함수. COMPACT_LLM_PROVIDER, COMPACT_LLM_MODEL이 비어 있으면 대화 모델을 쓴다(C9 D8)."""
    provider = env_str("COMPACT_LLM_PROVIDER", "") or config.provider
    compact_config = config_from_env(provider) if provider != config.provider else config
    model = env_str("COMPACT_LLM_MODEL", "")
    if model:
        compact_config = replace(compact_config, model=model)

    def summarize(text: str) -> str:
        return call_llm(LLMRequest(config=compact_config, messages=[UserMessage(content=text)])).text or ""

    return summarize


def report_citations(answer: str | None, seen: set[str], searched: bool) -> None:
    """답의 출처를 검색 결과와 대조해 지어낸 출처와 출처 누락을 알린다. 이번 턴에 검색했을 때만 누락을 알린다."""
    check = check_citations(answer or "", seen)
    if check.unknown:
        print(f"※ 검색 결과에 없는 출처가 있습니다: {', '.join(check.unknown)}")
    elif searched and not check.cited:
        print("※ 문서를 검색했지만 답에 출처가 없습니다.")


def repl(session: Session, ctx: CommandContext, progress: Progress, stream: bool) -> None:
    knowledge, audit = ctx.knowledge, session.settings.audit
    print(f"대화를 시작합니다({'스트리밍' if stream else '일괄 응답'}). 명령을 보려면 /help 를 입력하세요.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            if line == "/exit":
                break
            if line == "/help":
                print(HELP)
                continue
            found = find_command(line)
            if found is None:
                print(f"알 수 없는 명령입니다: {line}\n{HELP}")
                continue
            command, arg = found
            question = command.confirm(ctx, arg) if command.confirm else None
            if question and input(f"{question} (y/N): ").strip().lower() != "y":
                print("취소했습니다.")
                continue
            print(command.run(ctx, arg))
            continue

        progress.begin_turn()
        if audit:
            audit.begin_turn()
        calls_before = len(knowledge.calls) if knowledge else 0
        try:
            result = session.ask(line, on_text_delta=progress.on_text_delta if stream else None)
        except KeyboardInterrupt:  # Ctrl+C는 이 턴만 취소한다. 세션 이력은 이 턴 이전 그대로 남는다.
            progress.end_line()
            print("(중단됨: 이 턴은 이력에 반영되지 않았습니다)")
            continue
        except Exception as e:  # 모델 호출 실패, 스트림 끊김 등. 세션 이력은 이 턴 이전 그대로 남는다.
            progress.end_line()
            print(f"오류: {type(e).__name__}: {e}\n(이 턴은 이력에 반영되지 않았습니다)")
            continue
        progress.end_line()
        if not stream or not result.answer:  # 스트리밍이면 답은 이미 화면에 나왔으므로 다시 출력하지 않는다.
            print(result.answer or "(답변 없음)")
        if knowledge:
            report_citations(result.answer, knowledge.seen, len(knowledge.calls) > calls_before)
        notice = ctx.after_turn(line)  # 턴마다 자동 저장하고, 이력이 기준보다 크면 안내한다.
        if notice:
            print(notice)
        if result.status == "max_steps":
            print(f"※ 최대 스텝({result.steps})에 도달해 중단되었습니다. 이어서 '계속해'라고 말해 보세요.")


if __name__ == "__main__":
    main()
