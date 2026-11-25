"""Handoff 테스트(C21, 가짜 모델): main이 handoff Tool로 대화를 넘기면 같은 턴에 넘겨받은 에이전트가 이어 답하고,
다음 턴부터도 그 에이전트가 답한다. 넘겨받은 에이전트는 handoff(main)으로 돌려준다.

가짜 모델은 에이전트 이름(wrap_llm)으로 응답을 고른다(test_debate의 ByName).
"""

from fakes import TEST_CONFIG, final, tool_call
from test_debate import ByName
from test_parallelization import AGENTS, POOL, tool

from ai_agent_dock.core.agents import MAIN, AgentTeam, load_agents
from ai_agent_dock.core.commands import CommandContext, find_command
from ai_agent_dock.core.llm import AssistantMessage, UserMessage
from ai_agent_dock.core.memory import MemoryStore
from ai_agent_dock.core.observability import Tracer
from ai_agent_dock.core.runtime.types import AgentConfig
from ai_agent_dock.core.session import Session
from ai_agent_dock.core.session.session import MAX_HANDOFFS
from ai_agent_dock.core.session.store import SessionRecord, SessionStore
from ai_agent_dock.core.tools import ToolRegistry


def handoff(to, reason="파일 일", call_id="h1"):
    return tool_call(call_id, "handoff", agent=to, reason=reason)


def make(answers, tracer=None, memory=()):
    fake = ByName(answers)
    team = AgentTeam(load_agents(AGENTS), wrap_llm=fake.wrap)
    base = AgentConfig(llm_config=TEST_CONFIG, tools=[*POOL, tool("ask_user"), tool("files__read_text_file")], llm=lambda r, on_text_delta=None: None, tracer=tracer)
    moves = []
    agents = lambda name: team.config_for(name, base, memory=list(memory), handoff=True)
    session = Session(agents(MAIN), agents=agents, on_handoff=lambda *move: moves.append(move))
    return session, fake, moves, team


def test_main_hands_off_and_files_answers_this_turn_and_the_next():
    session, fake, moves, _ = make({MAIN: [handoff("files")], "files": [tool_call("l1", "files__list_directory", path="eval"), final("목록입니다"), final("다음 턴도 files")]})
    result = session.ask("eval 폴더 정리 도와줘")
    assert (result.status, result.answer, session.agent) == ("completed", "목록입니다", "files")
    assert moves == [(MAIN, "files", "파일 일")]
    first = fake.of("files")[0]
    users = [m.content for m in first.messages if isinstance(m, UserMessage)]
    assert users[0] == "eval 폴더 정리 도와줘" and len(users) == 2  # 질문은 한 번, 그 뒤에 넘김 알림(Q25)
    assert users[1].startswith("[넘김] main이(가) 이 대화를 files에게 넘겼다(이유: 파일 일)") and "더 넘기지 말고" not in users[1]
    assert any(isinstance(m, AssistantMessage) and m.tool_calls and m.tool_calls[0].name == "handoff" for m in first.messages)  # 이력 전체
    assert first.system_prompt.startswith("당신은 샌드박스 파일을 맡은")
    assert len(fake.of(MAIN)) == 1  # main은 넘기기만 하고 답을 쓰지 않는다

    assert session.ask("archive.md도 봐 줘").answer == "다음 턴도 files" and len(fake.of(MAIN)) == 1


def test_files_returns_to_main():
    session, fake, moves, _ = make({MAIN: [handoff("files"), final("네, 물어보세요")], "files": [final("파일 답"), handoff(MAIN, "파일 일 끝", "h2")]})
    session.ask("파일 정리")
    result = session.ask("고마워, 이제 3+4는?")
    assert (result.answer, session.agent) == ("네, 물어보세요", MAIN)
    assert moves[-1] == ("files", MAIN, "파일 일 끝")


def test_tools_and_prompts_follow_the_star_shape():
    session, fake, _, team = make({MAIN: [final("안녕")]}, memory=["사용자 이름은 훈이다."])
    main = {t.name for t in session.config.tools}
    assert "handoff" in main
    handoff_tool = next(t for t in session.config.tools if t.name == "handoff")
    assert handoff_tool.parameters["properties"]["agent"]["enum"] == ["files", "web"]
    assert "대화를 넘길 수 있는 에이전트 목록" in session.config.system_prompt and "- files:" in session.config.system_prompt
    assert "먼저 되묻지 말고 바로 넘긴다" in session.config.system_prompt  # 모호한 파일 요청에 main이 ask_user로 먼저 묻지 않게(T7 4차)
    files = session.agents("files")
    assert next(t for t in files.tools if t.name == "handoff").parameters["properties"]["agent"]["enum"] == [MAIN]
    assert {t.name for t in files.tools} == {"files__list_directory", "files__write_file", "files__read_text_file", "handoff"}  # 쓰기 Tool도(승인)
    assert "main에게 돌려준다" in files.system_prompt and "훈" in files.system_prompt  # 장기 메모리
    assert "스스로 돌려주지 않는다" in files.system_prompt  # 확인만 마치고 같은 턴에 돌려주지 않게(Q26)
    assert "files__write_file" not in main and {"files__list_directory", "files__read_text_file"} <= main  # 쓰기는 files에게 넘겨야만(Q22)
    workflow_main = team.config_for(MAIN, AgentConfig(llm_config=TEST_CONFIG, tools=list(POOL)))  # 워크플로용 main(handoff 인자 없음)
    assert "handoff" not in {t.name for t in workflow_main.tools} and "대화를 넘길 수" not in workflow_main.system_prompt
    assert "files__write_file" in {t.name for t in workflow_main.tools}


def test_handoff_limit_removes_the_tool_and_the_last_agent_answers():
    ping = [handoff("files", "a", f"m{i}") for i in range(3)]
    pong = [handoff(MAIN, "b", f"f{i}") for i in range(3)]
    session, fake, moves, _ = make({MAIN: [*ping, final("main이 답함")], "files": [*pong, final("files가 답함")]})
    result = session.ask("핑퐁")
    assert len(moves) == MAX_HANDOFFS and result.status == "completed"
    last = fake.requests[-1][1]
    assert "handoff" not in {t.name for t in last.tools}
    assert [m for m in last.messages if isinstance(m, UserMessage)][-1].content.endswith("이번 턴에는 더 넘기지 말고 직접 답한다.")


def test_unknown_target_is_a_tool_error_and_the_agent_continues():
    session, fake, moves, _ = make({MAIN: [handoff("researcher"), final("제가 답합니다")]})
    result = session.ask("조사해 줘")
    assert (result.answer, session.agent, moves) == ("제가 답합니다", MAIN, [])


def test_trace_records_the_handoff_and_the_agent_name(tmp_path):
    tracer = Tracer(tmp_path, session_id="s")
    session, _, _, _ = make({MAIN: [handoff("files")], "files": [final("파일 답")]}, tracer=tracer)
    session.ask("파일 정리")
    lines = [line for line in (tmp_path / "s.jsonl").read_text().splitlines()]
    import json

    rows = [json.loads(line) for line in lines]
    assert any(r["kind"] == "handoff" and r["source"] == MAIN and r["target"] == "files" for r in rows)
    files_llm = [r for r in rows if r["kind"] == "llm" and r.get("agent") == "files"]
    assert files_llm and {r["turn"] for r in rows if r["kind"] == "llm"} == {1}  # 넘겨받아도 같은 턴


def test_reset_returns_to_main_and_session_file_keeps_the_agent(tmp_path):
    session, _, _, team = make({MAIN: [handoff("files")], "files": [final("파일 답")]})
    session.ask("파일 정리")
    store = SessionStore(tmp_path)
    store.save(SessionRecord(id="a", name="a", provider="fake", history=session.history, agent=session.agent))
    assert store.load("a", "fake").agent == "files"
    (tmp_path / "old.json").write_text('{"id": "old", "name": "old", "provider": "fake", "created": "", "updated": "", "history": []}')
    assert store.load("old", "fake").agent == MAIN
    session.reset()
    assert session.agent == MAIN and session.history == []


def test_agent_command_shows_and_switches(tmp_path):
    session, _, _, team = make({MAIN: [final("안녕")]})
    ctx = CommandContext(
        session=session, registry=ToolRegistry(), skills=[], store=SessionStore(tmp_path), memory=MemoryStore(tmp_path / "m.md"),
        make_prompt=lambda: "", summarize=lambda t: "", agents=team,
    )
    run = lambda line: (lambda c, a: c.run(ctx, a))(*find_command(line))
    assert run("/agent") == "지금 대화 중인 에이전트: main (바꿀 수 있음: main, files, web)"
    assert run("/agent files") == "이제 files와 대화합니다." and session.agent == "files"
    assert run("/agent researcher").startswith("바꿀 수 없는 에이전트입니다: researcher")
    assert "files: " in run("/agents") and "← 지금 대화 중" in run("/agents")


def test_evaluate_runs_setup_turns_in_one_session_and_checks_the_agent():
    from ai_agent_dock.core.evals import evaluate
    from ai_agent_dock.core.evals.score import Case

    answers = {MAIN: [handoff("files")], "files": [final("목록입니다"), tool_call("r1", "files__read_text_file", path="eval/a.md"), final("화요일")]}
    session, _, _, _ = make(answers)
    case = Case(id="x", category="handoff", question="그 파일 읽어 줘", setup_turns=["파일 정리"], expect_tools=["files__read_text_file"], expect_agent="files")
    result = evaluate(case, lambda c: session.config, session=lambda config: session)
    assert result.passed and result.tools == ["files__read_text_file"]  # 마지막 턴의 Tool만 센다

    session, _, _, _ = make({MAIN: [final("2")]})
    result = evaluate(Case(id="y", category="handoff", question="1+1", expect_agent="files"), lambda c: session.config, session=lambda config: session)
    assert result.failures == ["대화 에이전트: 기대 files, 실제 main"]
