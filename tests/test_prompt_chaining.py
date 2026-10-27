"""prompt-chaining 워크플로 테스트(C13, 가짜 모델): gate 검사, 통과하면 본문, 실패하면 멈춤, 단계마다 다른 Tool과 프롬프트."""

from fakes import TEST_CONFIG, FakeLLM, final, tool_call
from langgraph.checkpoint.memory import InMemorySaver

from ai_agent_dock.core.hitl import make_ask_user_tool
from ai_agent_dock.core.orchestration import WorkflowContext, WorkflowRunner
from ai_agent_dock.core.retrieval import Hit, Knowledge
from ai_agent_dock.core.runtime.types import AgentConfig
from ai_agent_dock.core.tools import BUILTIN_TOOLS
from ai_agent_dock.workflows import WORKFLOWS
from ai_agent_dock.workflows.prompt_chaining import gate_problem

CITE = "에이전트 루프, 최대 스텝"
OUTLINE = f"- 최대 스텝은 상한이다 [{CITE}]\n- 기본값은 10이다 [{CITE}]\n- AGENT_MAX_STEPS로 바꾼다 [{CITE}]"


class FakeStore:
    def search(self, query, top_k=3):
        return [Hit(cite=CITE, section="최대 스텝", text="최대 스텝은 10이다.", score=0.9)]


def run(responses):
    """(결과, 모델이 호출마다 받은 Tool 이름 목록, 가짜 모델)."""
    llm = FakeLLM(responses)
    seen_tools = []

    def spy(request, on_text_delta=None):
        seen_tools.append({t.name for t in request.tools})
        return llm(request)

    tools = [*BUILTIN_TOOLS, make_ask_user_tool(lambda prompt: "")]
    config = AgentConfig(llm_config=TEST_CONFIG, tools=tools, llm=spy)
    knowledge = Knowledge(store=FakeStore(), seen=set(), files=1, chunks=1)
    runner = WorkflowRunner(WORKFLOWS, WorkflowContext(config=config, knowledge=knowledge), InMemorySaver())
    return runner.start("prompt-chaining", "최대 스텝은?"), seen_tools, llm


def test_gate_checks_item_count_citations_and_searched_sources():
    seen = {CITE}
    assert gate_problem(OUTLINE, seen) == ""
    assert gate_problem(f"- 하나 [{CITE}]\n- 둘 [{CITE}]", seen) == ""  # 2개는 통과
    assert gate_problem(f"- 하나 [{CITE}]", seen) == "개요 항목 1개(2~5개여야 한다)"
    assert gate_problem(f"- 하나 [{CITE}]\n- 둘\n1. 셋 [{CITE}]", seen) == "출처 없는 항목 1개"
    assert gate_problem(OUTLINE, set()) == f"검색에 없던 출처: {CITE}"


def test_passing_outline_goes_to_writing_with_no_tools():
    result, seen_tools, llm = run([tool_call("s1", "search_docs", query="최대 스텝"), final(OUTLINE), final(f"최대 스텝은 10이다 [{CITE}]")])

    assert (result.status, result.answer) == ("completed", f"최대 스텝은 10이다 [{CITE}]")
    assert "search_docs" in seen_tools[0] and not seen_tools[0] & {"ask_user", "delegate"}  # 개요 단계
    assert seen_tools[2] == set()  # 본문 단계는 Tool이 없다(Q2-1)
    write_prompt = llm.calls[2][0].content  # 본문 단계는 새 이력: 사용자 프롬프트 하나
    assert len(llm.calls[2]) == 1 and OUTLINE in write_prompt and "최대 스텝은 10이다." in write_prompt


def test_failing_gate_stops_before_writing():
    result, seen_tools, _ = run([tool_call("s1", "search_docs", query="날씨"), final("관련 문서를 찾지 못했다.")])

    assert result.status == "gate_failed" and "개요 항목 0개" in result.answer
    assert len(seen_tools) == 2  # 개요 단계의 호출 두 번뿐, 본문 단계는 부르지 않았다(Q4)
