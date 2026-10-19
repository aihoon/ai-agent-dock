"""여러 턴의 대화가 이어지는 과정을 한 단계씩 눈으로 확인하는 데모.

각 턴이 시작될 때 세션 이력에 무엇이 들어 있는지, 스텝마다 무슨 일이 일어나는지, 턴이 끝난 뒤 이력이
어떻게 바뀌는지 출력한다. (C1의 walkthrough.py가 "한 턴 안"을 보여 준다면, 이것은 "턴과 턴 사이"를 보여 준다.)

사용법:
  uv run python examples/chat_walkthrough.py --scenario memory     # 모델 없이: 앞 대화를 기억하는 모습
  uv run python examples/chat_walkthrough.py --scenario fail       # 모델 없이: 도중에 실패해도 이력이 안 깨지는 모습
  uv run python examples/chat_walkthrough.py --scenario maxsteps   # 모델 없이: 최대 스텝 뒤에도 대화가 이어지는 모습
  uv run python examples/chat_walkthrough.py --scenario mcp        # 모델 없이: MCP 서버의 Tool을 부르는 모습(서버는 실제로 띄움)
  uv run python examples/chat_walkthrough.py --scenario memory --stream   # 모델 없이: 글자 조각이 도착하는 모습(│가 조각 경계)
  uv run python examples/chat_walkthrough.py --scenario skill    # 모델 없이: 스킬 목록에서 본문을 load_skill로 불러 따르는 모습
  uv run python examples/chat_walkthrough.py --scenario rag      # 모델 없이: 문서를 검색해 출처를 붙여 답하는 모습(임베딩도 가짜)
  uv run python examples/chat_walkthrough.py --scenario workflow # 모델 없이: 워크플로가 고쳐 쓰기 → 승인 대기 → (껐다 켜고) 재개하는 모습
  uv run python examples/chat_walkthrough.py --stream "17*23은 얼마야?"    # 실제 모델로 스트리밍
  uv run python examples/chat_walkthrough.py "17*23은 얼마야?" "거기에 2를 곱해 줘"   # 실제 모델로 여러 턴
  uv run python examples/chat_walkthrough.py --provider anthropic "안녕" "내가 방금 뭐라고 했지?"

scenario는 미리 정한 응답을 쓰는 가짜 모델이라 API 호출과 비용이 없다. 질문을 여러 개 주면 한 세션에서 차례로 묻는다.
"""

import argparse
import json
import logging
import sqlite3
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from walkthrough import Tracer, ask, banner, describe, scripted  # C1 워크스루의 도구를 재사용한다.

from ai_agent_dock.core.hitl import Approval
from ai_agent_dock.core.llm import LLMConfig, LLMResponse, Tool, ToolCall, call_llm, config_from_env
from ai_agent_dock.core.observability import Tracer as TraceFile
from ai_agent_dock.core.orchestration import WorkflowContext, WorkflowRunner
from ai_agent_dock.core.retrieval import HashingEmbedder, Knowledge, KnowledgeIndex, make_search_docs_tool
from ai_agent_dock.core.runtime.types import AgentSettings
from ai_agent_dock.core.session import Session
from ai_agent_dock.core.skills import make_load_skill_tool, scan_skills
from ai_agent_dock.core.tools import BUILTIN_TOOLS
from ai_agent_dock.workflows import WORKFLOWS

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"  # 프로젝트 루트의 skills/
KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"  # 프로젝트 루트의 knowledge/


class TurnTracer(Tracer):
    """Tracer를 턴마다 스텝 번호를 다시 세도록, 그리고 스트리밍 콜백을 받아 둘 수 있도록 확장한다."""

    on_text_delta = None  # 이번 모델 호출에 쓸 글자 조각 콜백(스트리밍이 아니면 None)

    def begin_turn(self):
        self.step = 0

    def llm(self, request, on_text_delta=None):
        self.on_text_delta = on_text_delta
        return super().llm(request)


class DeltaPrinter:
    """글자 조각(델타)이 도착하는 모습을 보여 준다. 조각 경계를 │로 표시한다."""

    def __init__(self):
        self.count = 0

    def __call__(self, text):
        print(("       스트리밍 : " if self.count == 0 else "│") + text, end="", flush=True)
        self.count += 1

    def close(self):
        if self.count:
            print(f"  ← 글자 조각 {self.count}개가 도착하는 대로 출력됨")
        self.count = 0


def streaming(llm):
    """가짜 모델이 스트리밍처럼 동작하게 감싼다. 응답 텍스트를 세 글자씩 조각내 on_text_delta로 전달한다."""

    def wrapped(request, on_text_delta=None):
        response = llm(request)
        if on_text_delta and response.text:
            for i in range(0, len(response.text), 3):
                on_text_delta(response.text[i : i + 3])
        return response

    return wrapped


def final(text):
    return LLMResponse(text=text)


def failing_after_tool():
    """도중에 실패하는 가짜 모델: Tool 요청까지는 정상이다가 다음 호출에서 예외를 던진다."""
    items = [ask("c1", "calculate", expression="1+1"), RuntimeError("네트워크 오류(가상)")]
    state = {"n": 0}

    def llm(request):
        item = items[min(state["n"], len(items) - 1)]
        state["n"] += 1
        if isinstance(item, Exception):
            raise item
        return item

    return llm


# (질문 목록, 가짜 모델, 턴당 최대 스텝)
SCENARIOS = {
    "memory": (
        ["17*23은 얼마야?", "거기에 2를 곱해 줘"],
        scripted(
            ask("c1", "calculate", expression="17*23"),
            final("391입니다."),
            ask("c2", "calculate", expression="391*2"),
            final("782입니다."),  # 두 번째 턴이 첫 턴의 391을 이력에서 알고 있어야 나올 수 있는 답
        ),
        None,
    ),
    "fail": (
        ["1+1은?", "다시 한 번 1+1은?"],
        None,  # 아래에서 턴별로 다른 모델을 쓴다(첫 턴 중간 실패 → 두 번째 턴 정상)
        None,
    ),
    "maxsteps": (
        ["시간을 계속 알려 줘", "계속해"],
        scripted(
            ask("c1", "get_current_time"),
            ask("c2", "get_current_time"),
            final("그만 알려 드립니다."),
        ),
        2,
    ),
    # 승인 게이트: 쓰기 Tool은 실행 전에 승인을 받는다. 가짜 승인자가 첫 번째는 승인하고, 두 번째는 거부한다.
    "hitl": (
        ["notes.md에 메모를 써 줘", "그 파일을 다른 내용으로 고쳐 줘"],
        scripted(
            LLMResponse(text=None, tool_calls=[ToolCall("c1", "files__write_file", {"path": "notes.md", "content": "안녕"})]),
            final("notes.md에 저장했습니다."),
            LLMResponse(text=None, tool_calls=[ToolCall("c2", "files__write_file", {"path": "notes.md", "content": "바뀐 내용"})]),
            final("거부되어 저장하지 않았습니다."),
        ),
        None,
    ),
    # 문서 검색: 가짜 모델이 search_docs를 부르고, 검색 결과의 출처를 붙여 답한다. 임베딩은 네트워크를 쓰지 않는 가짜다.
    "rag": (
        ["Tool은 무엇으로 정의하나?"],
        scripted(
            LLMResponse(text=None, tool_calls=[ToolCall("c1", "search_docs", {"query": "Tool 정의 인자 스키마"})]),
            final("Tool은 이름, 설명, 인자 스키마, 실행 함수로 정의합니다 [Tool과 MCP, Tool 정의]."),
        ),
        None,
    ),
    # 스킬 본문은 모델이 load_skill로 불러온다. 가짜 모델이 그 Tool을 먼저 부르고, 본문을 받은 뒤 답한다.
    "skill": (
        ["버그를 찾는 절차대로 점검해 줘"],
        scripted(
            LLMResponse(text=None, tool_calls=[ToolCall("c1", "load_skill", {"name": "debug-checklist"})]),  # ask()는 name 인자와 겹쳐 직접 만든다
            final("체크리스트 순서대로 점검하겠습니다."),
        ),
        None,
    ),
    # MCP 서버(tests/mcp_test_server.py를 실제 자식 프로세스로 띄움)의 Tool을 부른다.
    # 턴 1은 정상 호출, 턴 2는 인자를 빠뜨려 서버를 부르기 전에 검증 오류를 받고 고쳐서 다시 부르며, 턴 3은 서버 쪽 실패를 받는다.
    "mcp": (
        ["'안녕'을 따라 말해 줘", "'반가워'를 따라 말해 줘", "실패하는 Tool을 불러 줘"],
        scripted(
            ask("c1", "demo__echo", text="안녕"),
            final("안녕"),
            ask("c2", "demo__echo"),  # text 인자를 빠뜨림 → 레지스트리가 서버를 부르기 전에 거부한다
            ask("c3", "demo__echo", text="반가워"),
            final("반가워"),
            ask("c4", "demo__fail"),  # 서버 쪽에서 실패 → 오류 관찰로 모델에 전달된다
            final("실패했습니다."),
        ),
        None,
    ),
}


def show_history(session, title):
    print(f"\n[{title}] 세션 이력 {len(session.history)}개")
    for i, m in enumerate(session.history, 1):
        print(f"   {i:2d}. {describe(m)}")


def scripted_approver(*decisions):
    """미리 정한 결정을 차례로 돌려주는 가짜 승인자. 실제 메뉴 대신 결정을 출력한다."""
    items = list(decisions)

    def approve(call):
        decision = items.pop(0)
        suffix = f" ({decision.reason})" if decision.reason else ""
        print(f"  · 승인 요청: {call.name} → 사용자 결정: {decision.kind}{suffix}")
        return decision

    return approve


def hitl_write_tool(tracer):
    """워크스루용 가짜 쓰기 Tool. 실제로는 아무 파일도 쓰지 않고 호출만 보여 준다."""

    def write(**arguments):
        print(f"       → 파일 기록(가짜): {arguments}")
        return f"{arguments.get('path')}에 기록함"

    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    }
    return tracer.wrap_tool(Tool("files__write_file", "파일을 쓴다(워크스루용 가짜, 실제로 쓰지 않는다)", schema, write))


def main():
    parser = argparse.ArgumentParser(description="여러 턴 대화의 단계별 출력 데모")
    parser.add_argument("questions", nargs="*", help="실제 모델에 차례로 보낼 질문들(생략하면 기본 두 질문)")
    parser.add_argument("--scenario", choices=[*SCENARIOS, "workflow"], help="모델 없이 시나리오 실행")
    parser.add_argument("--provider", help="LLM 제공자(openai, anthropic). 생략하면 .env의 LLM_PROVIDER")
    parser.add_argument("--max-steps", type=int, help="한 턴에서 모델을 호출하는 횟수 상한")
    parser.add_argument("--stream", action="store_true", help="스트리밍으로 받아 글자 조각이 도착하는 모습을 보여 준다")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.ERROR)
    if args.scenario == "workflow":
        return walk_workflow()

    if args.scenario:
        questions, model, scenario_steps = SCENARIOS[args.scenario]
        config = LLMConfig(provider="scenario", model="scenario", max_output_tokens=None)  # 가짜 모델은 설정을 읽지 않는다.
        mode = f"가짜 모델(시나리오: {args.scenario})"
    else:
        questions = args.questions or ["17*23은 얼마야?", "거기에 2를 곱해 줘"]
        model, scenario_steps = call_llm, None
        config = config_from_env(args.provider)
        mode = f"실제 모델(제공자: {args.provider or '.env의 LLM_PROVIDER'})"
    max_steps = args.max_steps or scenario_steps

    # fail 시나리오는 첫 턴에서 도중에 실패하고, 두 번째 턴은 정상으로 끝나도록 턴별 모델을 따로 만든다.
    turn_models = None
    if args.scenario == "fail":
        turn_models = [failing_after_tool(), scripted(ask("c9", "calculate", expression="1+1"), final("2입니다."))]

    delta = DeltaPrinter() if args.stream else None
    # 가짜 모델(시나리오)은 스트리밍을 모르므로, 스트리밍을 켜면 응답을 조각내 전달하도록 감싼다. 실제 모델은 그대로 쓴다.
    prepare = streaming if (delta and args.scenario) else (lambda m: m)
    model = prepare(model) if model else None
    turn_models = [prepare(m) for m in turn_models] if turn_models else None

    holder = {"llm": model}

    def call_model(request):
        callback = tracer.on_text_delta
        if callback is None:
            return holder["llm"](request)
        try:
            return holder["llm"](request, on_text_delta=callback)
        finally:
            delta.close()  # 조각을 출력하던 줄을 끝낸다.

    tracer = TurnTracer(call_model)
    with ExitStack() as stack:  # mcp 시나리오가 띄운 MCP 서버는 이 with가 끝날 때 정리된다.
        approve = scripted_approver(Approval.approve(), Approval.deny("경로가 틀렸다")) if args.scenario == "hitl" else None
        settings = AgentSettings(config=config, tools=build_tools(args.scenario, tracer, stack), max_steps=max_steps, llm=tracer.llm, approve=approve)
        session = Session(settings)
        walk(session, tracer, questions, turn_models, holder, delta, mode, max_steps)


def build_tools(scenario, tracer, stack):
    """시나리오에 쓸 Tool 목록. mcp 시나리오는 테스트용 MCP 서버(tests/mcp_test_server.py)를 실제로 띄워 그 Tool을 더한다."""
    if scenario == "hitl":
        return [hitl_write_tool(tracer)]
    if scenario == "rag":
        index = KnowledgeIndex(":memory:", HashingEmbedder())
        index.refresh(KNOWLEDGE_DIR)
        return [tracer.wrap_tool(t) for t in [*BUILTIN_TOOLS, make_search_docs_tool(index, set())]]
    if scenario == "skill":
        skills = make_load_skill_tool(scan_skills(SKILLS_DIR))
        return [tracer.wrap_tool(t) for t in [*BUILTIN_TOOLS, skills]]
    if scenario != "mcp":
        return [tracer.wrap_tool(t) for t in BUILTIN_TOOLS]
    from ai_agent_dock.core.tools import ToolRegistry
    from ai_agent_dock.core.tools.mcp import ServerSpec, connect_mcp_servers

    registry = ToolRegistry()
    registry.add_all(BUILTIN_TOOLS)
    demo = ServerSpec(sys.executable, [str(Path(__file__).parent.parent / "tests" / "mcp_test_server.py")])
    connected = stack.enter_context(connect_mcp_servers({"demo": demo}, call_timeout=30, start_timeout=60))
    for server, tools in connected.items():
        registry.add_all(tools, source=f"mcp:{server}")
    return [tracer.wrap_tool(t) for t in registry.tools()]


def walk(session, tracer, questions, turn_models, holder, delta, mode, max_steps):
    banner("시작")
    print(f"모드    : {mode}")
    print(f"질문 수 : {len(questions)}턴")
    print(f"Tool    : {', '.join(t.name for t in session.settings.tools)}")
    print(f"턴당 최대 스텝: {max_steps or '(.env의 AGENT_MAX_STEPS 또는 기본 10)'}")

    for turn, question in enumerate(questions, 1):
        if turn_models:
            holder["llm"] = turn_models[turn - 1]
        tracer.begin_turn()
        tracer._seen = len(session.history)  # 이번 턴 시작 전까지의 이력은 "이미 본 것"으로 취급한다.
        banner(f"턴 {turn}: {question}")
        print(f"턴 시작 전 세션 이력: {len(session.history)}개 (앞 턴의 대화를 이 이력으로 모델에 전달한다)")
        try:
            result = session.ask(question, on_text_delta=delta)
        except Exception as e:
            print(f"\n✗ 이 턴은 실패했다: {type(e).__name__}: {e}")
            print("   → 세션은 실패한 턴의 변경을 버리고, 턴 시작 전 이력 그대로 유지한다.")
            show_history(session, f"턴 {turn} 실패 뒤")
            continue
        print(f"\n✓ 턴 {turn} 종료: {result.status}, 스텝 {result.steps}회, 답변: {result.answer or '(없음)'}")
        show_history(session, f"턴 {turn} 종료 뒤")

    banner("끝")
    print(f"최종 세션 이력: {len(session.history)}개")


def walk_workflow():
    """answer-check 워크플로를 가짜 모델로 돌린다. 첫 답은 출처가 없어 코드가 고쳐 쓰게 하고, 둘째 답은 승인 단계에서 멈춘다.
    그 뒤 새 DB 연결과 새 실행기(프로그램을 껐다 켠 것과 같음)로 승인 대기 실행을 찾아 승인한다."""
    index = KnowledgeIndex(":memory:", HashingEmbedder())  # 가짜 임베딩으로 knowledge/를 색인한다(네트워크 없음).
    index.refresh(KNOWLEDGE_DIR)
    knowledge = Knowledge(store=index, seen=set(), files=1, chunks=1)
    llm = scripted(
        final("Tool은 이름과 설명으로 정의합니다."),  # 출처가 없다 → 출처 검사 실패 → 고쳐 쓰기
        ask("c1", "search_docs", query="Tool 정의 인자 스키마"),
        final("Tool은 이름, 설명, 인자 스키마, 실행 함수로 정의합니다 [Tool과 MCP, Tool 정의]."),
    )
    config = LLMConfig(provider="scenario", model="scenario", max_output_tokens=None)
    with tempfile.TemporaryDirectory() as tmp:
        trace = TraceFile(Path(tmp))
        settings = AgentSettings(config=config, tools=BUILTIN_TOOLS, llm=llm, tracer=trace)
        db = Path(tmp) / "checkpoints.sqlite"

        def runner(conn):
            return WorkflowRunner(WORKFLOWS, WorkflowContext(settings=settings, knowledge=knowledge), SqliteSaver(conn))

        banner("워크플로 시작: /workflow answer-check Tool은 무엇으로 정의하나?")
        conn = sqlite3.connect(db, check_same_thread=False)
        waiting = runner(conn).start("answer-check", "Tool은 무엇으로 정의하나?")
        show_workflow_trace(trace)
        print(f"\n→ 상태: {waiting.status}, 고쳐 쓴 횟수 {waiting.revisions}\n   초안: {waiting.draft}")
        conn.close()

        banner("프로그램을 껐다 켬: 새 DB 연결과 새 실행기")
        conn = sqlite3.connect(db, check_same_thread=False)
        restarted = runner(conn)
        [found] = restarted.pending()
        print(f"/workflows → 승인 대기 1개: {found.run_id} ({found.question})")
        done = restarted.resume(found.run_id, "approve")
        print(f"/workflow-resume 1 → 1) 승인\n→ 상태: {done.status}, 답: {done.answer}")
        print("   (재개할 때 모델을 다시 부르지 않았다. 체크포인트에 저장된 초안을 그대로 썼다)")
        conn.close()


def show_workflow_trace(trace):
    """추적 기록의 workflow_* 줄을 차례로 보여 준다."""
    for line in trace.path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["kind"] == "workflow_node":
            update = {k: v for k, v in row["update"].items() if k in ("problem", "revisions", "status")}
            print(f"  · 노드 {row['node']:<8} {update or ''}")
        elif row["kind"].startswith("workflow_"):
            print(f"  · {row['kind']}")


if __name__ == "__main__":
    main()
