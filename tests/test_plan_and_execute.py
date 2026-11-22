"""plan-and-execute 워크플로 테스트(C18, 가짜 모델): planner가 계획 → worker가 한 단계씩 실행 → 단계마다 replan → aggregator가 합침.

가짜 모델과 실행 도우미는 C15 테스트의 것을 쓴다(요청 내용으로 답한다). 단계 프롬프트에 앞 단계의 작업 글이 들어가므로
worker의 답은 질문 글이 아니라 에이전트 이름(researcher, main)으로 고른다. planner의 답은 JSON이다(Q23).
"""

import json
from pathlib import Path

import pytest
from fakes import TEST_CONFIG, final
from langgraph.checkpoint.memory import InMemorySaver
from test_parallelization import POOL, ByContent, start, tool

from ai_agent_dock.core.agents import MAIN
from ai_agent_dock.core.orchestration import WorkflowContext, WorkflowRunner
from ai_agent_dock.core.runtime.types import AgentConfig
from ai_agent_dock.workflows import WORKFLOWS
from ai_agent_dock.workflows.common import ONLY_MINE
from ai_agent_dock.workflows.plan_and_execute import MAX_RUNS, plan_prompt, read_plan, replan_input

AGENTS = Path(__file__).resolve().parent.parent / "agents"
NAME = "plan-and-execute"
WITH_ASKING = [*POOL, tool("ask_user"), tool("delegate"), tool("files__read_text_file")]


def plan(*steps, finished=False):
    """planner의 출력 JSON. steps는 (worker, task)."""
    return final(json.dumps({"finished": finished, "steps": [{"worker": w, "task": t} for w, t in steps]}, ensure_ascii=False))


END = plan(finished=True)


# --- 출력 JSON 읽기 (Q25) ---


def test_read_plan_reads_json_even_with_fence_and_text_around():
    label = '계획입니다.\n```json\n{"finished": false, "steps": [{"worker": " Researcher ", "task": " 찾기 "}]}\n```'
    assert read_plan(label) == (False, [("researcher", "찾기")])
    assert read_plan('{"finished": true, "steps": []}') == (True, [])


def test_read_plan_rejects_non_json_and_wrong_shapes():
    for label in ["모르겠다", "", "researcher: 찾기", '{"steps": []}', '{"finished": "no", "steps": []}', '{"finished": false, "steps": "x"}',
                  '{"finished": false, "steps": [{"worker": "researcher"}]}', '{"finished": false, "steps": [{"worker": "", "task": "찾기"}]}',
                  '{"finished": false, "steps": ["researcher: 찾기"]}', '{"finished": false, "steps": [{"worker": 1, "task": "찾기"}]}']:
        assert read_plan(label) is None, label


def test_read_plan_keeps_five_steps():
    label = json.dumps({"finished": False, "steps": [{"worker": "researcher", "task": f"작업{n}"} for n in range(1, 8)]})
    assert [task for _, task in read_plan(label)[1]] == ["작업1", "작업2", "작업3", "작업4", "작업5"]


def test_agent_md_examples_read_and_input_fields_are_documented():  # 형식이 AGENT.md와 코드에서 어긋나지 않게(4.3절)
    text = (AGENTS / "planner" / "AGENT.md").read_text(encoding="utf-8")
    examples = [line.strip() for line in text.splitlines() if line.strip().startswith('{"finished"')]
    assert len(examples) == 2 and all(read_plan(e) is not None for e in examples)
    state = {"question": "q", "plan": [], "done": [{"n": 1, "worker": "w", "task": "t", "ok": True, "result": "r"}], "remaining": []}
    data = replan_input(state)
    assert all(f"- {key}:" in text for key in data) and all(f'"{key}"' in text for key in data["done"][0])


# --- 그래프 ---


def test_plan_execute_replan_fix_execute_done_then_aggregate():
    llm = ByContent({
        "planner": [
            plan(("researcher", "C17 고쳐 쓰기 상한을 찾는다"), ("utility", "상한의 제곱을 계산한다")),
            plan(("utility", "1단계에서 찾은 상한 4의 제곱을 계산한다")),  # 고침
            END,
        ],
        "researcher": final("상한은 4 [C17, 3절]"),
        MAIN: final("16"),
        "aggregator": final("답은 16 [C17, 3절]"),
    })
    result, lines = start(NAME, "상한의 제곱은?", llm, tools=WITH_ASKING)

    assert (result.status, result.workers, result.revisions, result.answer) == ("completed", ["researcher", "utility"], 1, "답은 16 [C17, 3절]")
    first, second, third = llm.of("planner")
    assert first.tools == [] and first.messages[0].content == plan_prompt("상한의 제곱은?")
    replan = second.messages[0].content
    assert replan.startswith("worker 목록:\n- researcher:")
    assert json.loads(replan[replan.index("{"):]) == {  # 다시 계획의 입력은 JSON(Q24)
        "question": "상한의 제곱은?",
        "plan": [{"worker": "researcher", "task": "C17 고쳐 쓰기 상한을 찾는다"}, {"worker": "utility", "task": "상한의 제곱을 계산한다"}],
        "done": [{"n": 1, "worker": "researcher", "task": "C17 고쳐 쓰기 상한을 찾는다", "ok": True, "result": "상한은 4 [C17, 3절]"}],
        "remaining": [{"worker": "utility", "task": "상한의 제곱을 계산한다"}],
        "runs_left": MAX_RUNS - 1,
    }

    [utility] = llm.of(MAIN)
    assert utility.messages[0].content == (  # 뒤 단계가 앞 단계 결과를 글로 받는다(Q10). worker는 JSON을 보지 않는다
        "원래 질문: 상한의 제곱은?\n\n앞 단계 결과:\n[researcher-1] 단계: C17 고쳐 쓰기 상한을 찾는다\n결과:\n상한은 4 [C17, 3절]\n\n"
        f"맡은 단계: 1단계에서 찾은 상한 4의 제곱을 계산한다\n{ONLY_MINE}"
    )
    assert {t.name for t in utility.tools} == {"calculate", "get_current_time"}
    assert "[utility-2] 맡은 작업: 1단계에서 찾은 상한 4의 제곱을 계산한다\n결과:\n16" in llm.of("aggregator")[0].messages[0].content
    assert lines == [  # 4.5절: 단계마다 시작/완료 줄
        "워크플로 시작: plan-and-execute",
        "Plan 시작",
        "Plan 완료: 2단계\n  researcher: C17 고쳐 쓰기 상한을 찾는다\n  utility: 상한의 제곱을 계산한다",
        "Execute 1/8 시작: researcher-1\n  맡은 단계: C17 고쳐 쓰기 상한을 찾는다",
        "Execute 1/8 완료: researcher-1",
        "Replan 시작",
        "Replan 완료: 고침\n  utility: 1단계에서 찾은 상한 4의 제곱을 계산한다",
        "Execute 2/8 시작: utility-2\n  맡은 단계: 1단계에서 찾은 상한 4의 제곱을 계산한다",
        "Execute 2/8 완료: utility-2",
        "Replan 시작",
        "Replan 완료: 끝",
        "Aggregate 시작: aggregator",
    ]


def test_same_remaining_plan_is_kept_and_not_counted():
    llm = ByContent({
        "planner": [plan(("researcher", "가 찾기"), ("researcher", "나 찾기")), plan(("researcher", "나 찾기")), END],
        "researcher": final("찾음"),
        "aggregator": final("답"),
    })
    result, lines = start(NAME, "질문", llm)
    assert (result.revisions, result.workers) == (0, ["researcher", "researcher"])
    assert "Replan 완료: 유지" in lines and lines[-2] == "Replan 완료: 끝"


def test_execution_limit_stops_without_replan_and_lists_undone_steps():  # Q8
    llm = ByContent({
        "planner": [plan(("researcher", "찾기")), plan(("researcher", "또 찾기"), ("researcher", "더 찾기"))],  # 마지막 답을 계속 낸다.
        "researcher": final("못 찾음"),
        "aggregator": final("지금까지의 답"),
    })
    result, lines = start(NAME, "질문", llm)
    assert len(llm.of("researcher")) == MAX_RUNS and len(llm.of("planner")) == MAX_RUNS  # 처음 계획 1번 + replan 7번
    assert result.status == "limit" and result.answer == "지금까지의 답\n\n실행 상한에 이르러 하지 못한 단계:\n- researcher: 더 찾기"
    assert lines[-2:] == [f"Execute 상한({MAX_RUNS}번)에 이르러 지금까지로 답합니다", "Aggregate 시작: aggregator"]


def test_failed_step_goes_to_replan_as_failure():  # Q11
    llm = ByContent({"planner": [plan(("researcher", "찾기"), ("utility", "계산")), END], "researcher": RuntimeError("끊김")})
    result, lines = start(NAME, "질문", llm)
    assert "Execute 1/8 실패: researcher-1 (RuntimeError: 끊김)" in lines
    replan = llm.of("planner")[1].messages[0].content
    assert json.loads(replan[replan.index("{"):])["done"] == [{"n": 1, "worker": "researcher", "task": "찾기", "ok": False, "result": "실행 실패: RuntimeError: 끊김"}]
    assert llm.of("aggregator") == [] and result.answer == "researcher-1: 실행 실패: RuntimeError: 끊김"  # 모두 실패면 합치지 않는다


def test_unreadable_or_finished_first_plan_or_planner_error_goes_to_general_main():
    for answer in [final("모르겠다"), END, plan()]:  # JSON이 아님, 처음 계획의 finished: true(Q23), 빈 계획
        llm = ByContent({"planner": answer, "원래 질문": final("main의 답")})
        result, lines = start(NAME, "원래 질문", llm)
        assert (result.status, result.workers, result.answer) == ("completed", [], "main의 답")
        assert lines[2].startswith("Plan 완료: 실패(") and lines[3] == "general 경로 실행"

    result, lines = start(NAME, "원래 질문", ByContent({"planner": RuntimeError("끊김"), "원래 질문": final("답")}))
    assert result.answer == "답" and lines[2] == 'Plan 완료: 실패("오류: RuntimeError") → general'


def test_unreadable_replan_ends_with_what_was_done():
    llm = ByContent({"planner": [plan(("researcher", "찾기"), ("researcher", "또")), final("잘 모르겠다")], "researcher": final("찾음"), "aggregator": final("답")})
    result, lines = start(NAME, "질문", llm)
    assert (result.status, result.workers, result.answer) == ("completed", ["researcher"], "답")
    assert 'Replan 완료: 읽기 실패("잘 모르겠다") → 끝' in lines


def test_steps_do_not_ask_delegate_or_write_and_unknown_worker_goes_to_general():  # Q20, C16 Q14
    llm = ByContent({
        "planner": [plan(("docs", "문서 찾기"), ("researcher", "파일 찾기")), plan(("researcher", "파일 찾기")), END],
        "researcher": final("b"), MAIN: final("a"), "aggregator": final("ab"),
    })
    result, lines = start(NAME, "질문", llm, tools=WITH_ASKING)
    assert result.workers == ["general", "researcher"] and "알 수 없는 worker docs → general" in lines
    general = {t.name for t in llm.of(MAIN)[0].tools}
    assert "search_docs" in general and not general & {"ask_user", "delegate", "files__write_file"}
    assert {t.name for t in llm.of("researcher")[0].tools} == {"search_docs", "files__read_text_file"}


def test_unavailable_without_agent_team():
    base = AgentConfig(llm_config=TEST_CONFIG, tools=POOL)
    with pytest.raises(ValueError, match="에이전트"):
        WorkflowRunner(WORKFLOWS, WorkflowContext(config=base), InMemorySaver()).start(NAME, "x")
