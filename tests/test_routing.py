"""routing 워크플로 테스트(C14, 가짜 모델): 에이전트 정의의 tools: [], model, delegate, 경로 고르기, 경로별 프롬프트와 Tool과 모델, 추적의 model, 평가의 expect_route."""

from dataclasses import replace
from pathlib import Path

import pytest
from fakes import TEST_CONFIG, final, tool_call
from langgraph.checkpoint.memory import InMemorySaver

from ai_agent_dock.core.agents import DELEGATE, MAIN, AgentFormatError, AgentTeam, load_agents
from ai_agent_dock.core.evals import Case
from ai_agent_dock.core.evals.score import check_workflow
from ai_agent_dock.core.llm import LLMConfig, Tool, model_config
from ai_agent_dock.core.observability import Tracer
from ai_agent_dock.core.observability.report import all_stats, format_session
from ai_agent_dock.core.orchestration import WorkflowContext, WorkflowRunner
from ai_agent_dock.core.runtime.loop import run_agent
from ai_agent_dock.core.runtime.types import AgentConfig
from ai_agent_dock.workflows import WORKFLOWS
from ai_agent_dock.workflows.routing import ROUTES, pick, router_prompt

AGENTS = Path(__file__).resolve().parent.parent / "agents"
SCHEMA = {"type": "object", "properties": {"q": {"type": "string"}}}


def tool(name):
    return Tool(name=name, description=name, parameters=SCHEMA, func=lambda **kw: "ok", read_only=True, trusted=True)


POOL = [tool(n) for n in ["calculate", "get_current_time", "ask_user", "load_skill", "search_docs", "files__list_directory", "web__tavily_search"]]


def agent_file(root, name, head, body="역할 지침"):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "AGENT.md").write_text(f"---\nname: {name}\ndescription: {name} 설명\n{head}\n---\n{body}\n", encoding="utf-8")


class Recorder:
    """요청을 남기고 미리 정한 응답을 차례로 주는 가짜 모델."""

    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def __call__(self, request, on_text_delta=None):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def run(question, responses, tools=POOL):
    """(결과, 가짜 모델, 진행 줄)."""
    llm = Recorder(responses)
    team = AgentTeam(load_agents(AGENTS))
    base = AgentConfig(llm_config=TEST_CONFIG, tools=list(tools), llm=llm)
    lines = []
    context = WorkflowContext(config=team.config_for(MAIN, base), team=team, base=base, progress=lines.append)
    return WorkflowRunner(WORKFLOWS, context, InMemorySaver()).start("routing", question), llm, lines


# --- 에이전트 정의: tools: [], model, delegate (Q2-2, Q2-4, Q3) ---


def test_empty_tools_model_and_delegate_fields(tmp_path):
    agent_file(tmp_path, MAIN, "model:\ntools: *\ndelegate:\nmax_steps:")  # 빈 값: 대화 모델, 맡길 수 없음, 기본 스텝 상한
    agent_file(tmp_path, "quiet", "model: openai:gpt-6-luna\ntools: []\ndelegate: true\nmax_steps: 3")
    agents = load_agents(tmp_path)
    assert (agents[MAIN].model, agents[MAIN].delegate, agents[MAIN].max_steps) == (None, False, None)
    assert (agents["quiet"].tools, agents["quiet"].model, agents["quiet"].delegate, agents["quiet"].max_steps) == ((), "openai:gpt-6-luna", True, 3)


@pytest.mark.parametrize(
    "head, message",
    [
        ("description: x", "tools"),  # tools 줄이 없다
        ("tools:", "tools"),  # 비운 것과 [] 는 다르다
        ("tools: []\nmodel: gpt-6-luna", "제공자:모델"),
        ("tools: []\nmodel: nobody:x", "제공자:모델"),
        ("tools: []\ndelegate: maybe", "delegate"),
        ("tools: []\nmax_steps: 0", "max_steps"),
    ],
)
def test_bad_new_fields_fail_at_start(tmp_path, head, message):
    agent_file(tmp_path, MAIN, "tools: *")
    agent_file(tmp_path, "bad", head)
    with pytest.raises(AgentFormatError, match=message):
        load_agents(tmp_path)


def test_model_config_keeps_base_settings_for_the_same_provider():
    base = LLMConfig(provider="openai", model="gpt-5-mini", max_output_tokens=777)
    assert model_config("openai:gpt-6-luna", base) == replace(base, model="gpt-6-luna")
    assert model_config("anthropic:claude-haiku-5-5", base).provider == "anthropic"
    with pytest.raises(ValueError, match="제공자:모델"):
        model_config("gpt-6-luna", base)


def test_agents_with_delegate_false_are_not_targets_and_model_is_applied():
    team = AgentTeam(load_agents(AGENTS))
    assert [a.name for a in team.targets(MAIN)] == ["researcher"]  # router, web, chat은 맡길 수 없다
    base = AgentConfig(llm_config=TEST_CONFIG, tools=POOL)
    chat = team.config_for("chat", base)
    assert chat.tools == [] and (chat.llm_config.provider, chat.llm_config.model) == ("openai", "gpt-6-luna")
    team.agents[MAIN] = replace(team.agents[MAIN], model=None)
    assert team.config_for(MAIN, base).llm_config == TEST_CONFIG  # model을 비우면 대화 모델


def test_only_narrows_agent_tools_and_drops_delegate():
    team = AgentTeam(load_agents(AGENTS))
    base = AgentConfig(llm_config=TEST_CONFIG, tools=POOL)
    docs = team.config_for(MAIN, base, only=("search_docs", "web__*"))
    assert [t.name for t in docs.tools] == ["search_docs"]  # main은 web__*를 빼므로 교집합에 없다(Q2-3)
    assert DELEGATE not in docs.system_prompt and "researcher" not in docs.system_prompt


# --- 경로 고르기 (Q4) ---


def test_pick_normalizes_case_space_quotes_and_periods_only():
    assert pick(" Web.") == "web" and pick('"DOCS"') == "docs" and pick("utility\n") == "utility"
    for label in ["경로: web", "web 경로가 맞습니다", "search", ""]:
        assert pick(label) is None


def test_router_prompt_lists_all_routes_and_the_question():
    prompt = router_prompt("3의 12제곱은?")
    assert [r.name for r in ROUTES] == ["chat", "docs", "web", "utility", "files", "general"]
    assert all(f"- {r.name}: {r.description}" in prompt for r in ROUTES) and prompt.endswith("질문: 3의 12제곱은?")


# --- 워크플로 ---


def test_routes_question_to_the_chosen_agent_with_its_tools_and_model():
    result, llm, lines = run("3의 12제곱은?", [final("utility"), tool_call("c1", "calculate", q="3**12"), final("531441")])

    router, route = llm.requests[0], llm.requests[1]
    assert router.tools == [] and router.config.model == "gpt-6-luna" and "질문에 답하지 않는다" in router.system_prompt
    assert [t.name for t in route.tools] == ["calculate", "get_current_time"] and route.config.model == "gpt-6-luna"
    assert route.messages[0].content == "3의 12제곱은?"  # 질문만 넘긴다(Q11)
    assert route.system_prompt.startswith("당신은 도움이 되는 어시스턴트다.")  # utility는 main의 프롬프트
    assert (result.status, result.route, result.answer) == ("completed", "utility", "531441")
    assert lines == ["분류: router 에이전트가 경로 6개 중 하나를 고른다", "분류 결과: utility", "utility 경로 실행"]


def test_web_route_stays_without_web_tools_and_uses_web_prompt():
    result, llm, _ = run("오늘 뉴스", [final("web"), final("찾지 못했다")], tools=POOL[:-1])
    assert result.route == "web" and [t.name for t in llm.requests[1].tools] == ["get_current_time"]  # 웹 서버가 없어도 경로를 빼지 않는다(Q6)
    assert "웹 검색을 맡은" in llm.requests[1].system_prompt


def test_bad_label_or_router_error_falls_back_to_general_main():
    result, llm, lines = run("문서 찾아 저장해", [final("경로: docs"), final("main의 답")])
    assert (result.route, result.answer) == ("general", "main의 답")
    assert lines[1] == '분류 실패("경로: docs") → general'
    general = {t.name for t in llm.requests[1].tools}
    assert {"ask_user", "load_skill", DELEGATE} <= general and "web__tavily_search" not in general  # main 그대로(Q10)

    result, _, lines = run("x", [RuntimeError("연결 끊김"), final("답")])
    assert result.route == "general" and lines[1].startswith('분류 실패("오류: RuntimeError")')


def test_routing_is_unavailable_without_agent_team():
    base = AgentConfig(llm_config=TEST_CONFIG, tools=POOL)
    runner = WorkflowRunner(WORKFLOWS, WorkflowContext(config=base), InMemorySaver())
    with pytest.raises(ValueError, match="에이전트"):
        runner.start("routing", "x")


# --- 추적과 평가 (Q8, Q9) ---


def test_llm_trace_lines_carry_the_model_and_report_splits_by_model(tmp_path):
    tracer = Tracer(tmp_path)
    tracer.record("session_start", None, model="fake-model")
    nano = replace(TEST_CONFIG, provider="openai", model="gpt-6-luna")
    run_agent(AgentConfig(llm_config=nano, llm=Recorder([final("a")]), tracer=tracer), "q1")
    run_agent(AgentConfig(llm_config=TEST_CONFIG, llm=Recorder([final("b")]), tracer=tracer), "q2")

    report = format_session(all_stats(tmp_path)[0])
    assert "openai:gpt-6-luna: LLM 1" in report and "fake:fake-model: LLM 1" in report


def test_eval_checks_expected_route():
    case = Case(id="r", category="routing", question="x", workflow="routing", expect_route="web")
    assert check_workflow(case, "completed", 0, "web") == []
    assert check_workflow(case, "completed", 0, "general") == ["경로가 web이 아니라 general이었다"]


def test_team_wraps_each_agent_llm_with_its_name():
    calls = []
    team = AgentTeam(load_agents(AGENTS), wrap_llm=lambda name, llm: lambda request, on_text_delta=None: calls.append(name) or llm(request))
    base = AgentConfig(llm_config=TEST_CONFIG, tools=POOL, llm=Recorder([final("utility"), final("531441")]))
    context = WorkflowContext(config=team.config_for(MAIN, base), team=team, base=base)
    WorkflowRunner(WORKFLOWS, context, InMemorySaver()).start("routing", "3의 12제곱은?")
    assert calls == ["router", MAIN]  # 경로 utility는 main의 프롬프트를 쓴다
