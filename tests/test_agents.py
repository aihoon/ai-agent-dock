"""서브에이전트 테스트(C12, 가짜 모델): 에이전트 정의, Tool 고르기, 컨텍스트 격리, 깊이와 순환, 쓰기 Tool 거부, 추적, 승인 표시, 평가 판정, MCP 설정."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fakes import TEST_CONFIG, final, tool_call

from ai_agent_dock.core.agents import AGENT_INDEX_HEAD, DELEGATE, MAIN, AgentFormatError, AgentTeam, load_agents, select_tools
from ai_agent_dock.core.evals import Case, check
from ai_agent_dock.core.hitl import Approval, ask_in_terminal
from ai_agent_dock.core.llm import AssistantMessage, Tool, ToolCall, ToolMessage, UserMessage
from ai_agent_dock.core.observability import Tracer
from ai_agent_dock.core.observability.report import all_stats, format_session
from ai_agent_dock.core.runtime.loop import run_agent
from ai_agent_dock.core.runtime.types import AgentResult, AgentConfig
from ai_agent_dock.core.safety import UNTRUSTED_RULE, with_safety
from ai_agent_dock.core.skills import SkillMeta
from ai_agent_dock.core.tools.mcp import load_config

AGENTS = Path(__file__).resolve().parent.parent / "agents"
SCHEMA = {"type": "object", "properties": {"q": {"type": "string"}}}


def tool(name, result="ok", read_only=True, trusted=True):
    return Tool(name=name, description=name, parameters=SCHEMA, func=lambda **kw: result, read_only=read_only, trusted=trusted, untrusted_output=True)


POOL = [tool("calculate"), tool("ask_user"), tool("load_skill"), tool("search_docs", "근거 [문서, 1]"), tool("files__read_text_file"), tool("web__tavily_search")]


def agent_file(root, name, tools, body="역할 지침"):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "AGENT.md").write_text(f"---\nname: {name}\ndescription: {name} 설명\ntools: {tools}\n---\n{body}\n", encoding="utf-8")


class Recorder:
    """요청을 그대로 남기는 가짜 모델. main과 하위 에이전트가 같은 모델 함수를 쓰므로 응답은 호출 순서대로 준다."""

    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def __call__(self, request, on_text_delta=None):
        self.requests.append(replace(request, messages=list(request.messages)))  # 그 시점의 이력을 복사해 둔다.
        return self.responses.pop(0)


def team_and_base(llm, tools=POOL, **team_args):
    return AgentTeam(load_agents(AGENTS), **team_args), AgentConfig(llm_config=TEST_CONFIG, tools=list(tools), llm=llm)


# --- 에이전트 정의(F1) ---


def test_project_agents_load_and_main_keeps_the_old_base_prompt():
    agents = load_agents(AGENTS)
    assert list(agents) == [MAIN, "researcher"]
    assert agents[MAIN].body.startswith("당신은 도움이 되는 어시스턴트다.")  # 예전 DEFAULT_SYSTEM_PROMPT를 글자 그대로 옮겼다(Q11).
    assert agents[MAIN].tools == ("*", "!web__*")
    assert agents["researcher"].max_steps == 6


def test_bad_definitions_fail_at_start(tmp_path):
    with pytest.raises(AgentFormatError, match="메인 에이전트"):
        load_agents(tmp_path)
    agent_file(tmp_path / "a", MAIN, "")
    with pytest.raises(AgentFormatError, match="tools"):
        load_agents(tmp_path / "a")
    agent_file(tmp_path / "b", MAIN, "*")
    (tmp_path / "b" / "other").mkdir()
    (tmp_path / "b" / "other" / "AGENT.md").write_text("---\nname: wrong\ndescription: x\ntools: *\n---\n본문\n", encoding="utf-8")
    with pytest.raises(AgentFormatError, match="폴더 이름"):
        load_agents(tmp_path / "b")


def test_tool_patterns_include_and_exclude():
    names = lambda patterns: [t.name for t in select_tools(patterns, POOL)]
    assert names(["*", "!web__*"]) == ["calculate", "ask_user", "load_skill", "search_docs", "files__read_text_file"]
    assert names(["search_docs", "files__read_*", "web__tavily_search"]) == ["search_docs", "files__read_text_file", "web__tavily_search"]


# --- main 설정과 시스템 프롬프트 ---


def test_main_gets_delegate_agent_list_memory_and_skills():
    team, base = team_and_base(Recorder([]))
    skill = SkillMeta(name="요약", description="글을 줄인다", path=Path("x"))
    main = team.config_for(MAIN, base, skills=[skill], memory=["반려견은 콩"])
    names = [t.name for t in main.tools]
    assert "web__tavily_search" not in names and names[-1] == DELEGATE  # 웹 검색은 researcher만 쓴다(Q13).
    assert main.tools[-1].parameters["properties"]["agent"]["enum"] == ["researcher"]
    assert main.system_prompt.startswith("당신은 도움이 되는 어시스턴트다.")
    for part in ["반려견은 콩", "요약", AGENT_INDEX_HEAD, "- researcher:"]:
        assert part in main.system_prompt
    assert main.system_prompt.endswith(UNTRUSTED_RULE)


def test_untrusted_rule_only_when_a_tool_wraps_its_result(tmp_path):
    agent_file(tmp_path, MAIN, "[calculate, search_docs]")
    plain = Tool(name="calculate", description="", parameters=SCHEMA, func=lambda **kw: "4")
    base = AgentConfig(llm_config=TEST_CONFIG, tools=[plain], llm=Recorder([]))
    team = AgentTeam(load_agents(tmp_path))
    assert team.config_for(MAIN, base).system_prompt == "역할 지침"  # 감싸는 Tool이 없으면 규칙도 없다
    wrapped = replace(base, tools=[plain, tool("search_docs")])
    assert team.config_for(MAIN, wrapped).system_prompt.endswith(UNTRUSTED_RULE)


# --- 위임과 컨텍스트 격리(F2, F3) ---


def test_delegate_runs_child_with_fresh_history_and_returns_only_the_answer():
    llm = Recorder([
        tool_call("c1", DELEGATE, agent="researcher", task="스킬 형식을 조사해"),
        tool_call("s1", "search_docs", q="스킬"),
        final("스킬은 SKILL.md다 [문서, 1]"),
        final("정리: 스킬은 SKILL.md다 [문서, 1]"),
    ])
    team, base = team_and_base(llm)
    base, _ = with_safety(base)
    result = run_agent(team.config_for(MAIN, base, memory=["비밀 메모"]), "스킬 형식이 뭐야?")

    child = llm.requests[1]
    assert child.messages == [UserMessage(content="스킬 형식을 조사해")]  # 부모 이력을 보지 않는다(Q4).
    assert "비밀 메모" not in child.system_prompt and AGENT_INDEX_HEAD not in child.system_prompt
    assert {t.name for t in child.tools} == {"search_docs", "files__read_text_file", "web__tavily_search"}  # ask_user, load_skill, delegate 없음
    tool_results = [m.content for m in result.history if isinstance(m, ToolMessage)]
    assert len(tool_results) == 1 and tool_results[0].startswith('<untrusted source="delegate">')  # 답만, 외부 내용으로 감싸서(Q6)
    assert "스킬은 SKILL.md다" in tool_results[0] and result.answer.startswith("정리")


def test_child_that_hits_its_step_limit_says_so():
    llm = Recorder([tool_call("c1", DELEGATE, agent="researcher", task="조사"), tool_call("s1", "search_docs", q="x"), tool_call("s2", "search_docs", q="y"), final("끝")])
    team, base = team_and_base(llm, max_steps=2)
    team.agents["researcher"] = replace(team.agents["researcher"], max_steps=None)  # SUBAGENT_MAX_STEPS를 쓴다.
    result = run_agent(team.config_for(MAIN, base), "조사해")
    assert "스텝 상한(2)" in result.history[2].content


def test_depth_and_cycles(tmp_path):
    agent_file(tmp_path, MAIN, '"*"')
    agent_file(tmp_path, "a", "calculate, delegate")
    agent_file(tmp_path, "b", "calculate, delegate")
    one = AgentTeam(load_agents(tmp_path), max_depth=1)
    two = AgentTeam(load_agents(tmp_path), max_depth=2)
    assert [a.name for a in one.targets(MAIN)] == ["a", "b"]
    assert one.targets("a", depth=1, parents=(MAIN,)) == []  # 깊이 상한 1: 맡은 에이전트는 다시 맡기지 못한다.
    assert [a.name for a in two.targets("a", depth=1, parents=(MAIN,))] == ["b"]  # 자기 자신과 main은 부르지 못한다.
    assert two.targets("b", depth=2, parents=(MAIN, "a")) == []
    with pytest.raises(ValueError, match="SUBAGENT_MAX_DEPTH"):
        AgentTeam(load_agents(tmp_path), max_depth=4)


def test_subagent_write_tool_is_refused_and_missing_names_warn(tmp_path):
    team = AgentTeam(load_agents(AGENTS))
    assert team.check(POOL[:4]) == ["에이전트 researcher: 연결되지 않은 Tool은 빼고 씁니다: web__tavily_search, web__tavily_extract"]
    agent_file(tmp_path, MAIN, "*")
    agent_file(tmp_path, "writer", "files__*")
    with pytest.raises(AgentFormatError, match="읽기 전용 Tool만"):
        AgentTeam(load_agents(tmp_path)).check([tool("files__write_file", read_only=False)])


def test_child_approval_carries_the_agent_name():
    asked = []
    read = tool("files__read_text_file", trusted=False)  # 파일 서버는 trusted false라 읽기도 승인을 받는다.
    llm = Recorder([tool_call("c1", DELEGATE, agent="researcher", task="읽어"), tool_call("r1", "files__read_text_file", q="a"), final("내용"), final("끝")])
    team, base = team_and_base(llm, tools=[read])
    team.approve_for = lambda name: lambda call: asked.append((name, call.name)) or Approval.approve()
    run_agent(team.config_for(MAIN, base), "읽어 줘")
    assert asked == [("researcher", "files__read_text_file")]
    printed = []
    ask_in_terminal(ToolCall("1", "files__read_text_file", {}), interactive=True, input_fn=lambda p: "1", print_fn=printed.append, agent="researcher")
    assert printed[0] == "[승인 요청] [researcher] files__read_text_file"


# --- 추적(F4) ---


def test_child_trace_lines_stay_in_the_parent_turn(tmp_path):
    llm = Recorder([tool_call("c1", DELEGATE, agent="researcher", task="조사"), final("결과"), final("끝")])
    team, base = team_and_base(llm)
    tracer = Tracer(tmp_path)
    run_agent(team.config_for(MAIN, replace(base, tracer=tracer)), "질문")
    lines = [json.loads(line) for line in tracer.path.read_text(encoding="utf-8").splitlines()]
    delegation = next(line for line in lines if line["kind"] == "delegate")
    child = [line for line in lines if line.get("agent") == "researcher"]
    assert {line["turn"] for line in lines} == {1}  # 하위 에이전트는 턴 번호를 새로 세지 않는다.
    assert child and all(line["parent"] == delegation["id"] for line in child)
    report = format_session(all_stats(tmp_path)[0])
    assert "위임: 1번" in report and "researcher: LLM 1" in report and "main: LLM 2" in report


# --- 평가 판정(F6) ---


def test_eval_counts_child_tools_and_checks_delegation():
    main = AgentResult(answer="답 [문서, 1]", status="completed", steps=2, history=[AssistantMessage(tool_calls=[ToolCall("c1", DELEGATE, {})])])
    child = AgentResult(answer="조사", status="completed", steps=2, history=[AssistantMessage(tool_calls=[ToolCall("s1", "search_docs", {})])])
    case = Case(id="x", category="subagent", question="q", expect_tools=["search_docs"], expect_delegate=True)
    assert check(case, main, [child]) == []
    assert check(case, main) == ["툴 search_docs을 부르지 않았다"]
    assert check(Case(id="y", category="subagent", question="q", expect_delegate=False), main) == ["맡기지 않아야 할 질문을 하위 에이전트에게 맡겼다"]


# --- MCP 설정: 키 치환과 읽기 전용 지정(Q13, V6) ---


def test_mcp_env_reference_and_read_only(tmp_path, monkeypatch):
    path = tmp_path / "mcp.json"
    server = {"command": "npx", "env": {"TAVILY_API_KEY": "${TAVILY_API_KEY}"}, "read_only": ["tavily_search"]}
    path.write_text(json.dumps({"servers": {"web": server, "off": {**server, "enabled": False, "env": {"K": "${NOPE}"}}}}), encoding="utf-8")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    specs = load_config(path)
    assert list(specs) == ["web"] and specs["web"].env == {"TAVILY_API_KEY": "tvly-test"} and specs["web"].read_only == ["tavily_search"]
    monkeypatch.delenv("TAVILY_API_KEY")
    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        load_config(path)
