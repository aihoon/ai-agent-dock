"""evaluator-optimizer 워크플로 테스트(C17, 가짜 모델): 답 작성 → evaluator가 기준표로 평가 → 실패한 기준의 이유로 고쳐 쓰기.

가짜 모델과 실행 도우미는 C15 테스트의 것을 쓴다(요청 내용으로 답한다).
"""

import pytest
from fakes import TEST_CONFIG, final, tool_call
from langgraph.checkpoint.memory import InMemorySaver
from test_orchestration import CITE, FakeStore
from test_parallelization import POOL, ByContent, start, tool

from ai_agent_dock.core.agents import MAIN
from ai_agent_dock.core.orchestration import WorkflowContext, WorkflowRunner
from ai_agent_dock.core.retrieval import Knowledge
from ai_agent_dock.core.runtime.types import AgentConfig
from ai_agent_dock.workflows import WORKFLOWS

NAME = "evaluator-optimizer"
TOOLS = [*POOL, tool("ask_user")]
ALL_PASS = "질문에 답함: 통과\n근거에 맞음: 통과\n출처: 통과\n간결: 통과"
NO_CITATION = "질문에 답함: 통과\n근거에 맞음: 통과\n출처: 실패 - 출처가 없다\n간결: 통과"
CITED = f"최대 스텝은 10이다 [{CITE}]"


def knowledge():
    return Knowledge(store=FakeStore(), seen=set(), files=1, chunks=1)


def run(writer, evaluator, max_revisions=4, **kw):
    llm = ByContent({MAIN: writer, "evaluator": evaluator})
    result, lines = start(NAME, "최대 스텝은?", llm, tools=TOOLS, knowledge=knowledge(), max_revisions=max_revisions, **kw)
    return result, lines, llm


def test_failed_criterion_reason_goes_into_the_rewrite_and_the_second_answer_passes():
    writer = [tool_call("s1", "search_docs", query="최대 스텝"), final("최대 스텝은 10이다"), tool_call("s2", "search_docs", query="최대 스텝"), final(CITED)]
    result, lines, llm = run(writer, [final(NO_CITATION), final(ALL_PASS)])

    assert (result.status, result.revisions, result.answer) == ("passed", 1, CITED)
    rewrite = next(r.messages[0].content for r in llm.of(MAIN) if "[이전 답]" in r.messages[0].content)
    assert rewrite.startswith("최대 스텝은?\n\n[이전 답]\n최대 스텝은 10이다\n\n[고칠 점]\n- 출처: 출처가 없다\n\n")
    assert lines == [
        "답 작성",
        "평가: 통과 3/4 (실패: 출처 - 출처가 없다)",
        "코드 출처 검사: 출처 없음(참고)",
        "고쳐 쓰기 1/4",
        "답 다시 작성 (고칠 점: 출처)",
        "평가: 통과 4/4",
        "코드 출처 검사: 통과(참고)",
    ]


def test_evaluator_sees_question_answer_and_every_tool_result_the_writer_got():
    writer = [tool_call("s1", "search_docs", query="최대 스텝"), tool_call("w1", "web__tavily_search", q="loop"), final(CITED)]
    _, _, llm = run(writer, [final(ALL_PASS)])

    [evaluator] = llm.of("evaluator")
    prompt = evaluator.messages[0].content
    assert evaluator.tools == [] and "질문: 최대 스텝은?" in prompt
    assert f'<untrusted source="answer">\n{CITED}\n</untrusted>' in prompt
    assert f'<untrusted source="search_docs">\n[{CITE}]\n최대 스텝은 10이다.' in prompt and '<untrusted source="web__tavily_search">\nok' in prompt


def test_writer_gets_main_tools_without_asking_or_delegating_plus_web():
    _, _, llm = run([final(CITED)], [final(ALL_PASS)])
    names = {t.name for t in llm.of(MAIN)[0].tools}
    assert {"search_docs", "calculate", "web__tavily_search"} <= names and not names & {"ask_user", "delegate"}


def test_code_citation_check_is_only_a_note():  # 평가자가 통과면 코드 검사가 실패여도 통과(Q11)
    result, lines, _ = run([tool_call("s1", "search_docs", query="최대 스텝"), final("출처 없는 답")], [final(ALL_PASS)])
    assert result.status == "passed" and "코드 출처 검사: 출처 없음(참고)" in lines


def test_code_citation_check_is_skipped_when_docs_were_not_searched():  # Q33
    _, lines, _ = run([tool_call("w1", "web__tavily_search", q="mcp"), final("웹 답 https://example.com")], [final(ALL_PASS)])
    assert "코드 출처 검사: 문서 검색 안 함(건너뜀)" in lines


def test_still_failing_at_the_limit_ends_with_the_failed_criteria():
    result, lines, llm = run([final("출처 없는 답")], [final(NO_CITATION)], max_revisions=2)
    assert (result.status, result.revisions) == ("limit", 2)
    assert result.answer == "출처 없는 답\n\n아직 통과하지 못한 기준:\n- 출처: 출처가 없다"
    assert len(llm.of("evaluator")) == 3 and "고쳐 쓰기 2/2" in lines


def test_evaluator_error_ends_with_the_current_answer_without_rewriting():
    result, lines, llm = run([final(CITED)], [RuntimeError("끊김")])
    assert (result.status, result.answer, result.revisions) == ("eval_failed", CITED, 0)
    assert len(llm.of(MAIN)) == 1 and lines[-1] == "평가 실패(RuntimeError: 끊김) → 지금 답으로 끝냅니다"


def test_unavailable_without_agent_team():
    with pytest.raises(ValueError, match="에이전트"):
        WorkflowRunner(WORKFLOWS, WorkflowContext(config=AgentConfig(llm_config=TEST_CONFIG, tools=POOL)), InMemorySaver()).start(NAME, "x")
