"""평가 잣대 맞추기 테스트(C23, 가짜 모델): 노드 안 Tool 판정, LLM 판정의 근거, 비용 칸, 끝 줄의 횟수 이름."""

from dataclasses import replace

from fakes import TEST_CONFIG, FakeLLM, final, tool_call
from langgraph.checkpoint.memory import InMemorySaver

from ai_agent_dock.core.commands import outcome_line
from ai_agent_dock.core.evals import Case, evaluate
from ai_agent_dock.core.orchestration import Outcome, WorkflowContext, WorkflowRunner
from ai_agent_dock.core.retrieval import Hit, Knowledge
from ai_agent_dock.core.runtime.loop import run_agent
from ai_agent_dock.core.runtime.types import AgentConfig
from ai_agent_dock.core.tools import BUILTIN_TOOLS
from ai_agent_dock.workflows import WORKFLOWS

CITE = "에이전트 루프, 최대 스텝"
EVIDENCE = "최대 스텝은 10이다."


class FakeStore:
    def search(self, query, top_k=3):
        return [Hit(cite=CITE, section="최대 스텝", text=EVIDENCE, score=0.9)]


def searched(tokens=None):
    """노드(답 작성)가 search_docs를 부르고 출처를 붙여 답한다."""
    first = tool_call("s1", "search_docs", query="최대 스텝")
    return [replace(first, input_tokens=tokens, output_tokens=tokens), final(f"최대 스텝은 10이다 [{CITE}]")]


class Collector:
    """평가가 채우는 on_result와, 모은 결과를 꺼내고 비우는 함수(take)."""

    def __init__(self):
        self.results = []

    def take(self):
        taken = list(self.results)
        self.results.clear()  # on_result가 이 목록의 append를 쥐고 있어 새 목록으로 바꾸지 않는다.
        return taken


def setup(responses, collector):
    def make_config(_case):
        return AgentConfig(llm_config=TEST_CONFIG, tools=BUILTIN_TOOLS, llm=FakeLLM(responses), on_result=collector.results.append)

    def workflows(config):
        kb = Knowledge(store=FakeStore(), seen=set(), files=1, chunks=1)
        return WorkflowRunner(WORKFLOWS, WorkflowContext(config=config, knowledge=kb), InMemorySaver())

    return make_config, workflows


def answer_check(**rules):
    return Case(id="w", category="workflow", question="최대 스텝은?", workflow="answer-check", expect_status="approved", **rules)


# --- 모으기 ---


def test_run_agent_hands_its_result_to_on_result_once():
    got = []
    config = AgentConfig(llm_config=TEST_CONFIG, tools=BUILTIN_TOOLS, llm=FakeLLM([final("안녕")]), on_result=got.append)
    result = run_agent(config, "안녕")
    assert got == [result]
    run_agent(replace(config, llm=FakeLLM([final("안녕")]), on_result=None), "안녕")  # 비우면 부르지 않는다(채팅)
    assert len(got) == 1


# --- 노드 안 Tool 판정 ---


def test_workflow_case_is_judged_by_the_tools_its_nodes_called():
    collector = Collector()
    make_config, workflows = setup(searched(), collector)
    passed = evaluate(answer_check(expect_tools=["search_docs"]), make_config, workflows=workflows, finished=collector.take)
    assert passed.passed and passed.tools == ["search_docs"]

    collector = Collector()
    make_config, workflows = setup(searched(), collector)
    failed = evaluate(answer_check(expect_tools=["calculate"], forbid_tools=["search_docs"]), make_config, workflows=workflows, finished=collector.take)
    assert failed.failures == ["툴 calculate을 부르지 않았다", "부르면 안 되는 툴 search_docs을 불렀다"]


def test_a_tool_error_inside_a_node_fails_the_workflow_case():
    collector = Collector()
    make_config, workflows = setup([tool_call("c1", "calculate", expression="뭐야"), *searched()], collector)
    result = evaluate(answer_check(), make_config, workflows=workflows, finished=collector.take)
    assert len(result.failures) == 1 and result.failures[0].startswith("툴 calculate이 오류를 냈다")


def test_without_finished_the_workflow_case_is_judged_as_before():
    make_config, workflows = setup([tool_call("c1", "calculate", expression="뭐야"), *searched()], Collector())
    assert evaluate(answer_check(), make_config, workflows=workflows).passed


# --- LLM 판정의 근거와 비용 ---


def test_llm_judge_gets_node_evidence_and_its_own_calls_are_not_counted():
    collector = Collector()
    make_config, workflows = setup(searched(tokens=7), collector)
    prompts = []

    def judge(prompt):  # 평가처럼 에이전트로 판정한다. 이 호출도 on_result로 모이지만 비용에는 들어가지 않아야 한다.
        prompts.append(prompt)
        config = AgentConfig(llm_config=TEST_CONFIG, llm=FakeLLM([final("질문에 답함: 통과 - 됨")]), on_result=collector.results.append)
        return run_agent(config, prompt).answer

    case = answer_check(llm_judge=True)
    result = evaluate(case, make_config, workflows=workflows, finished=collector.take, judge=judge)
    assert EVIDENCE in prompts[0]
    assert (result.llm_calls, result.input_tokens, result.output_tokens) == (2, 7, 7)
    assert result.seconds >= 0


def test_costs_are_averaged_over_runs_for_main_cases_too():
    collector = Collector()

    def make_config(_case):
        llm = FakeLLM([replace(tool_call("c1", "calculate", expression="2+3"), input_tokens=10), replace(final("5"), input_tokens=20)])
        return AgentConfig(llm_config=TEST_CONFIG, tools=BUILTIN_TOOLS, llm=llm, on_result=collector.results.append)

    result = evaluate(Case(id="m", category="tool", question="2+3?"), make_config, repeat=2, finished=collector.take)
    assert (result.llm_calls, result.input_tokens, result.output_tokens) == (2, 30, 0)


def test_case_without_finished_has_zero_costs():
    make_config, workflows = setup(searched(), Collector())
    result = evaluate(answer_check(), make_config, workflows=workflows)
    assert (result.llm_calls, result.input_tokens) == (0, 0)


# --- 끝 줄의 횟수 이름(Q6) ---


def outcome(status, revisions):
    return Outcome(run_id="r", workflow="w", question="q", status=status, answer="답", revisions=revisions)


def test_outcome_line_uses_the_korean_status_and_the_count_label():
    assert outcome_line(outcome("completed", 1), "다시 찾은 횟수") == "▸ 워크플로 끝: 완료 (다시 찾은 횟수 1)"
    assert outcome_line(outcome("agreed", 2), "토론 라운드") == "▸ 워크플로 끝: 합의 (토론 라운드 2)"
    assert outcome_line(outcome("completed", 0), "") == "▸ 워크플로 끝: 완료"


def test_each_workflow_names_its_count():
    labels = {w.name: w.count_label for w in WORKFLOWS}
    assert labels == {
        "answer-check": "고쳐 쓴 횟수",
        "evaluator-optimizer": "고쳐 쓴 횟수",
        "plan-and-execute": "계획 고친 횟수",
        "reflexion": "다시 쓴 횟수",
        "debate": "토론 라운드",
        "corrective-rag": "다시 찾은 횟수",
        "prompt-chaining": "",
        "routing": "",
        "sectioning": "",
        "voting": "",
        "orchestrator-workers": "",
    }

