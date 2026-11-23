"""reflexion 워크플로 테스트(C19, 가짜 모델): act(main) → evaluate(evaluator) → 실패면 reflect(reflector) → act … → 통과면 교훈 저장.

가짜 모델은 C15 테스트의 것을 쓴다(요청 내용으로 답한다). 교훈 저장소는 임시 폴더의 MemoryStore다.
"""

import pytest
from fakes import TEST_CONFIG, final, tool_call
from langgraph.checkpoint.memory import InMemorySaver
from test_parallelization import AGENTS, POOL, ByContent, tool

from ai_agent_dock.core.agents import MAIN, AgentTeam, load_agents
from ai_agent_dock.core.evals.rubric import RUBRIC, evaluator_prompt, read_verdicts
from ai_agent_dock.core.memory import MemoryStore
from ai_agent_dock.core.orchestration import WorkflowContext, WorkflowRunner
from ai_agent_dock.core.runtime.types import AgentConfig
from ai_agent_dock.workflows import WORKFLOWS
from ai_agent_dock.workflows.reflexion import CRITERIA, MAX_SAVED, act_prompt, read_lesson, save_lessons

NAME = "reflexion"
COMMON_PASS = "질문에 답함: 통과\n근거에 맞음: 통과\n출처: 통과\n간결: 통과"
ALL_PASS = COMMON_PASS + "\n결론 먼저: 통과\n확인한 곳: 통과"
RULES_FAIL = COMMON_PASS + "\n결론 먼저: 실패 - 첫 줄이 결론으로 시작하지 않는다\n확인한 곳: 실패 - 마지막 줄이 없다"
LESSON = "답의 첫 줄은 '결론:'으로 시작하고, 마지막 줄에 '확인한 곳:'과 부른 Tool 이름을 쓴다"
GOOD = "결론: 10이다.\n최대 스텝은 10이다.\n확인한 곳: search_docs"


def run(answers, store, question="최대 스텝은?", max_revisions=4):
    llm = ByContent(answers)
    team = AgentTeam(load_agents(AGENTS))
    base = AgentConfig(llm_config=TEST_CONFIG, tools=[*POOL, tool("ask_user")], llm=llm)
    lines = []
    context = WorkflowContext(
        config=team.config_for(MAIN, base), team=team, base=base, progress=lines.append, max_revisions=max_revisions, reflections=store
    )
    return WorkflowRunner(WORKFLOWS, context, InMemorySaver()).start(NAME, question), lines, llm


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "reflections.md")


def test_fail_reflect_retry_pass_saves_the_lesson(store):
    writer = [tool_call("s1", "search_docs", query="최대 스텝"), final("최대 스텝은 10이다."), tool_call("s2", "search_docs", query="최대 스텝"), final(GOOD)]
    result, lines, llm = run({MAIN: writer, "evaluator": [final(RULES_FAIL), final(ALL_PASS)], "reflector": final(f"- \"{LESSON}\"\n설명")}, store)

    assert (result.status, result.revisions, result.answer) == ("passed", 1, GOOD)
    assert store.items() == [LESSON]
    first, retry = dict.fromkeys(r.messages[0].content for r in llm.of(MAIN))  # 시도마다 첫 메시지(스텝마다 같은 요청이 쌓인다)
    assert first == "최대 스텝은?"
    assert retry == act_prompt("최대 스텝은?", [LESSON]) and "최대 스텝은 10이다." not in retry  # 이전 답 없이 처음부터(Q11)
    reflect = llm.of("reflector")[0]
    assert reflect.tools == []
    prompt = reflect.messages[0].content
    assert "- 결론 먼저: 첫 줄이 결론으로 시작하지 않는다" in prompt and "부른 Tool: search_docs" in prompt and "이미 있는 교훈:\n없음" in prompt
    assert lines == [
        "워크플로 시작: reflexion (저장된 교훈 0개)",
        "Act 1/5 시작",
        "Act 1/5 완료",
        "Evaluate 시작",
        "Evaluate 완료: 통과 4/6 (실패: 결론 먼저 - 첫 줄이 결론으로 시작하지 않는다; 확인한 곳 - 마지막 줄이 없다)",
        "Reflect 시작",
        f"Reflect 완료\n  {LESSON}",
        "Act 2/5 시작 (교훈 1개)",
        "Act 2/5 완료",
        "Evaluate 시작",
        "Evaluate 완료: 통과 6/6",
        "교훈 저장: 1개 (모두 1개)",
    ]


def test_saved_lessons_go_into_the_first_act_and_first_try_pass_saves_nothing(store):
    store.add(LESSON)
    result, lines, llm = run({MAIN: final(GOOD), "evaluator": final(ALL_PASS)}, store, question="2+2는?")
    assert (result.status, result.revisions) == ("passed", 0)
    assert llm.of(MAIN)[0].messages[0].content == f"[지난 실패에서 배운 교훈]\n- {LESSON}\n\n[질문]\n2+2는?"
    assert store.items() == [LESSON] and llm.of("reflector") == []
    assert lines[:2] == ["워크플로 시작: reflexion (저장된 교훈 1개)", "Act 1/5 시작 (교훈 1개)"] and not any(line.startswith("교훈 저장") for line in lines)


def test_limit_saves_nothing(store):
    result, lines, llm = run({MAIN: final("답"), "evaluator": final(RULES_FAIL), "reflector": final(LESSON)}, store, max_revisions=2)
    assert (result.status, result.revisions) == ("limit", 2) and len(llm.of(MAIN)) == 3
    assert result.answer == "답\n\n아직 통과하지 못한 기준:\n- 결론 먼저: 첫 줄이 결론으로 시작하지 않는다\n- 확인한 곳: 마지막 줄이 없다"
    assert lines[-1] == "Act 상한(3번)에 이르러 지금 답으로 끝냅니다" and store.items() == []


def test_evaluator_error_ends_with_the_current_answer(store):
    result, lines, _ = run({MAIN: final("답"), "evaluator": RuntimeError("끊김")}, store)
    assert (result.status, result.answer) == ("eval_failed", "답")
    assert lines[-1] == "Evaluate 실패(RuntimeError: 끊김) → 지금 답으로 끝냅니다" and store.items() == []


def test_reflector_error_or_empty_uses_the_failed_reasons_and_does_not_save(store):
    for reflector in [RuntimeError("끊김"), final("  ")]:
        result, lines, llm = run({MAIN: [final("답"), final(GOOD)], "evaluator": [final(RULES_FAIL), final(ALL_PASS)], "reflector": reflector}, store)
        fallback = "결론 먼저: 첫 줄이 결론으로 시작하지 않는다; 확인한 곳: 마지막 줄이 없다"
        assert result.status == "passed" and store.items() == []
        assert any(line.startswith("Reflect 실패(") and line.endswith(f"→ 실패 이유를 교훈으로\n  {fallback}") for line in lines)
        assert fallback in llm.of(MAIN)[-1].messages[0].content


def test_evaluator_gets_the_team_rules(store):
    _, _, llm = run({MAIN: final(GOOD), "evaluator": final(ALL_PASS)}, store)
    prompt = llm.of("evaluator")[0].messages[0].content
    assert '- 결론 먼저: 첫 줄은 "결론:"으로' in prompt and "- 확인한 곳:" in prompt and "- 출처:" in prompt


# --- 부품 ---


def test_rubric_is_a_parameter_and_defaults_to_the_c17_rubric():
    assert [v.criterion for v in read_verdicts(ALL_PASS)] == [c.name for c in RUBRIC]
    assert [v.result for v in read_verdicts(COMMON_PASS, CRITERIA)][-2:] == ["실패", "실패"]  # 빠진 팀 규칙은 실패
    assert "결론 먼저" not in evaluator_prompt("q", "a", []) and "결론 먼저" in evaluator_prompt("q", "a", [], CRITERIA)


def test_save_lessons_skips_duplicates_and_keeps_the_newest(store):
    store.add("옛 교훈")
    assert save_lessons(store, ["옛  교훈", "새 교훈", "새 교훈"]) == 1 and store.items() == ["옛 교훈", "새 교훈"]
    save_lessons(store, [f"교훈 {n}" for n in range(MAX_SAVED)])
    assert len(store.items()) == MAX_SAVED and store.items()[0] == "교훈 0" and store.items()[-1] == f"교훈 {MAX_SAVED - 1}"


def test_read_lesson_takes_the_first_line_without_marks():
    assert read_lesson('\n1. "규칙을 지킨다"\n설명') == "규칙을 지킨다" and read_lesson("") == ""


def test_unavailable_without_store_or_agents():
    team = AgentTeam(load_agents(AGENTS))
    base = AgentConfig(llm_config=TEST_CONFIG, tools=POOL)
    with pytest.raises(ValueError, match="교훈 저장소"):
        WorkflowRunner(WORKFLOWS, WorkflowContext(config=base, team=team, base=base), InMemorySaver()).start(NAME, "x")
    with pytest.raises(ValueError, match="에이전트"):
        WorkflowRunner(WORKFLOWS, WorkflowContext(config=base), InMemorySaver()).start(NAME, "x")
