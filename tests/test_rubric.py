"""기준표와 LLM 판정 테스트(C17, 가짜 모델): 평가자 답 읽기, 평가자 프롬프트, 근거 모으기, 평가(evals)의 LLM 판정 기록."""

from fakes import TEST_CONFIG, FakeLLM, final, tool_call

from ai_agent_dock.core.evals import Case, CaseResult, evaluate, summarize
from ai_agent_dock.core.evals.rubric import FAIL, PASS, RUBRIC, SKIP, evaluator_prompt, read_verdicts, tool_evidence, verdict_line
from ai_agent_dock.core.llm import AssistantMessage, ToolCall, ToolMessage
from ai_agent_dock.core.runtime.types import AgentConfig
from ai_agent_dock.core.tools import BUILTIN_TOOLS

NAMES = [c.name for c in RUBRIC]


# --- 평가자 답 읽기 (3.4절) ---


def test_rubric_has_four_criteria_and_two_need_evidence():
    assert NAMES == ["질문에 답함", "근거에 맞음", "출처", "간결"]
    assert [c.name for c in RUBRIC if c.needs_evidence] == ["근거에 맞음", "출처"]


def test_read_verdicts_reads_result_and_reason_per_line_in_rubric_order():
    label = "- 간결: 통과\n출처: 실패 - 두 번째 문장에 출처가 없다\n질문에 답함: 통과 - 최대 스텝을 말했다\n근거에 맞음: 통과"
    verdicts = read_verdicts(label)
    assert [(v.criterion, v.result, v.reason) for v in verdicts] == [
        ("질문에 답함", PASS, "최대 스텝을 말했다"),
        ("근거에 맞음", PASS, ""),
        ("출처", FAIL, "두 번째 문장에 출처가 없다"),
        ("간결", PASS, ""),
    ]


def test_missing_criteria_fail_and_unknown_lines_are_dropped():
    verdicts = read_verdicts("평가 결과입니다.\n정확성: 통과\n질문에 답함: 통과")
    assert [(v.criterion, v.result, v.reason) for v in verdicts][1:] == [
        ("근거에 맞음", FAIL, "평가자가 판정하지 않음"),
        ("출처", FAIL, "평가자가 판정하지 않음"),
        ("간결", FAIL, "평가자가 판정하지 않음"),
    ]
    assert all(v.result == FAIL for v in read_verdicts(""))


def test_not_judged_is_neither_pass_nor_fail():
    verdicts = read_verdicts("질문에 답함: 통과\n근거에 맞음: 판정하지 않음\n출처: 판정하지 않음 - 근거 자료 없음\n간결: 통과")
    assert [v.result for v in verdicts] == [PASS, SKIP, SKIP, PASS]
    assert verdict_line(verdicts) == "통과 2/2, 판정 안 함 2"


def test_verdict_line_lists_failed_criteria_with_reasons():
    verdicts = read_verdicts("질문에 답함: 통과\n근거에 맞음: 통과\n출처: 실패 - 출처 없음\n간결: 실패 - 반복")
    assert verdict_line(verdicts) == "통과 2/4 (실패: 출처 - 출처 없음; 간결 - 반복)"


# --- 평가자 프롬프트와 근거 ---


def test_evaluator_prompt_wraps_answer_and_each_evidence_and_lists_the_rubric():
    prompt = evaluator_prompt("최대 스텝은?", "10이다 [루프, 처음]", [("search_docs", "[루프, 처음]\n최대 스텝은 10"), ("web__tavily_search", "웹 결과")])
    assert [line.split(":")[0] for line in prompt.splitlines() if line.startswith("- ")] == [f"- {n}" for n in NAMES]
    assert "질문: 최대 스텝은?" in prompt
    assert '<untrusted source="answer">\n10이다 [루프, 처음]\n</untrusted>' in prompt
    assert '<untrusted source="search_docs">\n[루프, 처음]\n최대 스텝은 10' in prompt and '<untrusted source="web__tavily_search">' in prompt


def test_evaluator_prompt_when_writer_called_no_tool_asks_to_skip_evidence_criteria():
    prompt = evaluator_prompt("질문", "답", [])
    assert "근거 자료 없음" in prompt and "근거에 맞음, 출처는 판정하지 않음" in prompt and "<untrusted source=\"search_docs\"" not in prompt


def test_evaluator_prompt_when_evidence_was_not_passed_does_not_read_it_as_not_found():  # Q29
    prompt = evaluator_prompt("질문", "답", None)
    assert "전달되지 않았다" in prompt and "찾지 못했다고 보지 않는다" in prompt and "근거 자료 없음" not in prompt
    assert "근거에 맞음, 출처는 판정하지 않음" in prompt


def test_rubric_allows_safety_notes_tool_values_and_i_dont_know_without_citation():  # Q30, Q31
    rules = {c.name: c.rule for c in RUBRIC}
    assert "지시를 따르지 않았다는 짧은 안내" in rules["간결"]
    assert "시각, 계산" in rules["출처"] and "모른다" in rules["출처"] and "모른다" in rules["근거에 맞음"]


def test_tool_evidence_pairs_tool_results_with_their_tool_names():
    history = [
        AssistantMessage(tool_calls=[ToolCall(id="a", name="search_docs", arguments={}), ToolCall(id="b", name="calculate", arguments={})]),
        ToolMessage(tool_call_id="a", content="문서"),
        ToolMessage(tool_call_id="b", content="오류", is_error=True),
        AssistantMessage(content="답"),
    ]
    assert tool_evidence(history) == [("search_docs", "문서")]  # 실패한 Tool의 결과는 근거가 아니다.


# --- 평가(evals)의 LLM 판정 (Q6, Q15~Q18) ---


def docs_config(_case):
    llm = FakeLLM([tool_call("c1", "calculate", expression="17*23"), final("답은 391입니다")])
    return AgentConfig(llm_config=TEST_CONFIG, tools=BUILTIN_TOOLS, llm=llm)


def test_llm_judge_is_recorded_per_run_and_does_not_change_the_pass_count():
    prompts = []
    labels = iter(["질문에 답함: 통과\n근거에 맞음: 통과\n출처: 실패 - 출처 없음\n간결: 통과", "질문에 답함: 통과\n근거에 맞음: 통과\n출처: 통과\n간결: 통과"])
    case = Case(id="d", category="docs", question="17*23은?", expect_contains=["391"], llm_judge=True)
    outcome = evaluate(case, docs_config, repeat=2, judge=lambda prompt: prompts.append(prompt) or next(labels))
    assert outcome.passes == 2  # 규칙 판정만 센다
    assert outcome.llm_verdicts == ["통과 3/4 (실패: 출처 - 출처 없음)", "통과 4/4"]
    assert outcome.disagreements == 1  # 규칙은 통과인데 LLM은 실패
    assert '<untrusted source="calculate">\n391' in prompts[0]  # 근거는 이력의 Tool 결과


def test_llm_judge_is_skipped_without_the_flag_and_errors_are_recorded():
    case = Case(id="d", category="docs", question="17*23은?", expect_contains=["391"])
    assert evaluate(case, docs_config, judge=lambda prompt: 1 / 0).llm_verdicts == []

    def broken(prompt):
        raise RuntimeError("끊김")

    outcome = evaluate(Case(**{**case.__dict__, "llm_judge": True}), docs_config, judge=broken)
    assert outcome.llm_verdicts == ["오류: RuntimeError"] and outcome.disagreements == 0 and outcome.passes == 1


def test_summary_shows_llm_judge_disagreements():
    results = [
        CaseResult(id="a", category="docs", runs=2, passes=2, llm_verdicts=["통과 4/4", "통과 3/4 (실패: 출처 - x)"], disagreements=1),
        CaseResult(id="b", category="docs", runs=1, passes=1, llm_verdicts=["통과 4/4"]),
        CaseResult(id="c", category="tool", runs=1, passes=1),
    ]
    assert summarize(results, set()).splitlines()[-1] == "LLM 판정: 2케이스, 규칙 판정과 어긋남 1번 (a)"
    assert "LLM 판정" not in summarize(results[2:], set())
