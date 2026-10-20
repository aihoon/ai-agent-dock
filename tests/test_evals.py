"""평가 판정과 실행 테스트(가짜 모델): 규칙 채점, 회귀, 케이스 파일, 미리 적은 답, 결과 저장."""

import json
from pathlib import Path

from fakes import TEST_CONFIG, FakeLLM, final, tool_call

from ai_agent_dock.core.evals import (
    Case,
    check,
    evaluate,
    last_passed,
    load_cases,
    regressions,
    save_run,
    scripted_approver,
    scripted_input,
    summarize,
)
from ai_agent_dock.core.evals.run import CaseResult
from ai_agent_dock.core.llm import AssistantMessage, Tool, ToolCall, ToolMessage
from ai_agent_dock.core.runtime.types import AgentResult, AgentSettings
from ai_agent_dock.core.tools import BUILTIN_TOOLS

CASES = Path(__file__).resolve().parent.parent / "evals" / "cases.jsonl"


def result(answer, *tools):
    calls = [ToolCall(id=f"c{i}", name=name, arguments={}) for i, name in enumerate(tools)]
    return AgentResult(answer=answer, status="completed", steps=1, history=[AssistantMessage(tool_calls=calls)])


def case(**rules):
    return Case(id="x", category="tool", question="q", **rules)


# --- 규칙 채점 ---


def test_all_rules_met_passes():
    assert check(case(expect_tools=["calculate"], expect_contains=["391"]), result("답은 391", "calculate")) == []


def test_missing_expected_tool_fails():
    assert check(case(expect_tools=["calculate"]), result("391")) == ["툴 calculate을 부르지 않았다"]


def test_forbidden_tool_fails():
    assert check(case(forbid_tools=["load_skill"]), result("2", "load_skill")) == ["부르면 안 되는 툴 load_skill을 불렀다"]


def test_missing_word_fails():
    assert check(case(expect_contains=["1024"]), result("천이십사")) == ["답에 '1024'가 없다"]


def test_any_one_of_the_expressions_is_enough():
    refuse = case(expect_any=["모르", "확인할 수 없", "확인해 드릴 수 없"])
    assert check(refuse, result("날씨는 확인해 드릴 수 없습니다")) == []
    assert check(refuse, result("잘 모르겠습니다")) == []
    assert check(refuse, result("내일은 맑고 20도입니다")) == ["답에 ['모르', '확인할 수 없', '확인해 드릴 수 없'] 중 어느 표현도 없다"]


def test_a_forbidden_expression_in_the_answer_fails():
    weather = case(expect_any=["수 없"], forbid_contains=["°C", "도 정도"])
    assert check(weather, result("내일 날씨는 알려드릴 수 없습니다")) == []
    assert check(weather, result("확인할 수 없지만 대략 18°C 정도일 거예요")) == ["답에 있으면 안 되는 '°C'가 있다"]


def test_citation_required_and_forbidden():
    cited = "최대 스텝은 10이다 [에이전트 루프, 최대 스텝]"
    assert check(case(expect_citation=True), result(cited)) == []
    assert check(case(expect_citation=True), result("최대 스텝은 10이다")) == ["답에 출처 [문서 제목, 위치]가 없다"]
    assert check(case(expect_citation=False), result("모릅니다")) == []
    assert check(case(expect_citation=False), result(cited)) == ["근거가 없는데 출처를 붙였다"]


def with_tool_result(content, is_error):
    call = ToolCall(id="c1", name="files__read_text_file", arguments={"path": "notes.md"})
    history = [AssistantMessage(tool_calls=[call]), ToolMessage(tool_call_id="c1", content=content, is_error=is_error)]
    return AgentResult(answer="답", status="completed", steps=2, history=history)


def test_a_tool_that_ran_and_failed_fails_the_case():
    failures = check(case(expect_tools=["files__read_text_file"]), with_tool_result("ToolError: ENOENT: no such file", True))
    assert failures == ["툴 files__read_text_file이 오류를 냈다: ToolError: ENOENT: no such file"]


def test_a_denied_or_blocked_tool_is_not_counted_as_an_error():
    for content in ["사용자가 실행을 거부했다. 이유: 테스트", "사용자가 다르게 요청했다: 다른 파일", "이 턴에서 사용자가 files__read_text_file 실행을 이미 거부했다."]:
        assert check(case(expect_tools=["files__read_text_file"]), with_tool_result(content, True)) == []


def test_denial_wording_in_the_loop_matches_what_the_scorer_skips():
    """루프가 실제로 만드는 거부 관찰이 판정에서 오류로 잡히지 않아야 한다."""
    write = Tool("files__write_file", "쓴다", {"type": "object", "properties": {}}, lambda **kwargs: "기록함")
    llm = FakeLLM([tool_call("c1", "files__write_file"), tool_call("c2", "files__write_file"), final("거부되었습니다")])
    settings = AgentSettings(config=TEST_CONFIG, tools=[write], llm=llm, approve=scripted_approver({"files__write_file": "deny"}))

    outcome = evaluate(case(expect_tools=["files__write_file"], expect_contains=["거부"]), lambda _c: settings)

    assert outcome.passed, outcome.failures


def test_regression_is_a_case_that_passed_before_and_fails_now():
    assert regressions({"a", "b"}, {"b", "c"}) == {"b"}


# --- 케이스 파일 ---


def test_case_file_covers_every_category_and_builtin_tool():
    cases = load_cases(CASES)
    assert len({c.id for c in cases}) == len(cases)  # id는 겹치지 않는다
    assert {c.category for c in cases} == {"tool", "docs", "approval", "clarify", "skill", "memory", "workflow", "safety"}
    expected = {name for c in cases for name in c.expect_tools}
    builtin = {t.name for t in BUILTIN_TOOLS} | {"ask_user", "search_docs", "load_skill"}
    assert builtin <= expected  # 빌트인 Tool마다 최소 한 케이스


# --- 미리 적은 답 ---


def test_scripted_approver_answers_by_tool_name_and_approves_unlisted_tools():
    approve = scripted_approver({"files__write_file": "deny"})
    call = ToolCall(id="c", name="files__write_file", arguments={})
    assert approve(call).kind == "deny" and approve(ToolCall(id="d", name="files__list_allowed_directories", arguments={})).kind == "approve"


def test_scripted_input_answers_in_order_then_empty():
    ask = scripted_input(["회의 메모"])
    assert [ask("q"), ask("q")] == ["회의 메모", ""]


# --- 실행과 저장 ---


def calc_settings(_case):
    llm = FakeLLM([tool_call("c1", "calculate", expression="17*23"), final("답은 391입니다")])
    return AgentSettings(config=TEST_CONFIG, tools=BUILTIN_TOOLS, llm=llm)


def test_evaluate_runs_the_case_and_counts_passes_over_repeats():
    outcome = evaluate(case(expect_tools=["calculate"], expect_contains=["391"]), calc_settings, repeat=2)
    assert (outcome.runs, outcome.passes, outcome.passed, outcome.tools) == (2, 2, True, ["calculate"])


def test_save_run_and_read_the_last_passed_ids(tmp_path):
    assert last_passed(tmp_path) is None
    results = [CaseResult(id="a", category="tool", runs=1, passes=1), CaseResult(id="b", category="docs", runs=1, passes=0, failures=["답에 'x'가 없다"])]

    path = save_run(tmp_path, "eval-1", "fake", "fake-model", 1, results)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert (data["provider"], data["model"], data["pass_rate"]) == ("fake", "fake-model", 0.5)
    assert last_passed(tmp_path) == {"a"}


def test_summary_shows_rate_failures_and_regressions():
    results = [CaseResult(id="a", category="tool", runs=1, passes=1), CaseResult(id="b", category="docs", runs=1, passes=0, failures=["이유"])]
    text = summarize(results, regressed={"b"})
    assert "통과 1/2 (50%)" in text and "실패 b [회귀]" in text and "회귀 1개: b" in text
