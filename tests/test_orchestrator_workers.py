"""orchestrator-workers 워크플로 테스트(C16, 가짜 모델): orchestrator가 작업과 worker를 정함 → worker들 동시 실행 → aggregator가 합침.

가짜 모델과 실행 도우미는 C15 테스트의 것을 쓴다(요청 내용으로 답한다).
"""

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import TEST_CONFIG, final, tool_call
from langgraph.checkpoint.memory import InMemorySaver
from test_parallelization import POOL, ByContent, start, tool

from ai_agent_dock.core.agents import MAIN, AgentTeam, load_agents
from ai_agent_dock.core.evals import Case
from ai_agent_dock.core.evals.score import check_workflow
from ai_agent_dock.core.observability import Tracer
from ai_agent_dock.core.orchestration import WorkflowContext, WorkflowRunner
from ai_agent_dock.core.runtime.types import AgentConfig
from ai_agent_dock.workflows import WORKFLOWS
from ai_agent_dock.workflows.orchestrator_workers import ONLY_MINE, WORKERS, plan_prompt, read_tasks

AGENTS = Path(__file__).resolve().parent.parent / "agents"
NAME = "orchestrator-workers"
WITH_ASKING = [*POOL, tool("ask_user"), tool("delegate"), tool("files__read_text_file")]


# --- orchestrator 답 읽기 (3.5절) ---


def test_read_tasks_reads_worker_and_task_per_line_and_allows_the_same_worker_again():
    label = "researcher: C14 경로 표\n2. Web: Send 공식 설명\n- researcher: C15 voting 규칙\n* utility) 지금 몇 시"
    assert read_tasks(label) == [("researcher", "C14 경로 표"), ("web", "Send 공식 설명"), ("researcher", "C15 voting 규칙")]


def test_read_tasks_keeps_unknown_worker_names_for_the_graph_to_hand_to_general():
    assert read_tasks("docs: 문서 찾기\nweb: 웹 찾기") == [("docs", "문서 찾기"), ("web", "웹 찾기")]


def test_read_tasks_drops_lines_without_colon_empty_tasks_and_prose():
    label = "작업을 나눴다\nresearcher:\n참고: 이건 설명이다\n경로 docs: 띄어쓴 이름\nweb: 웹 찾기"
    assert read_tasks(label) == [("web", "웹 찾기")]  # 이름은 영어 소문자 한 단어만(설명 줄을 작업으로 읽지 않는다)
    assert read_tasks("") == [] and read_tasks("모르겠다") == []


def test_read_tasks_keeps_five():
    label = "\n".join(f"researcher: 작업{n}" for n in range(1, 8))
    assert [task for _, task in read_tasks(label)] == ["작업1", "작업2", "작업3", "작업4", "작업5"]


def test_plan_prompt_shows_the_three_workers():
    prompt = plan_prompt("질문")
    assert [w.name for w in WORKERS] == ["researcher", "web", "utility"]
    assert [line.split(":")[0] for line in prompt.splitlines() if line.startswith("- ")] == ["- researcher", "- web", "- utility"]
    assert prompt.endswith("질문: 질문")


# --- 그래프 ---


def test_tasks_run_with_their_worker_and_aggregator_gets_them_in_task_order():
    llm = ByContent({
        "orchestrator": final("researcher: C14 경로 표\nweb: Send 공식 설명\nresearcher: C15 voting 규칙"),
        "C14 경로": [tool_call("c1", "search_docs", q="경로"), final("경로 답 [C14, 경로 표]")],
        "Send 공식": final("Send 답 [Send](https://x.dev)"),
        "C15 voting": final("voting 답 [C15, voting]"),
        "aggregator": final("보고서"),
    })
    result, lines = start(NAME, "보고서 써 줘", llm, tools=WITH_ASKING)

    [orchestrator] = llm.of("orchestrator")
    assert orchestrator.tools == [] and orchestrator.messages[0].content == plan_prompt("보고서 써 줘")
    researchers = llm.of("researcher")
    assert {r.messages[0].content for r in researchers} == {
        f"원래 질문: 보고서 써 줘\n맡은 작업: C14 경로 표\n{ONLY_MINE}",
        f"원래 질문: 보고서 써 줘\n맡은 작업: C15 voting 규칙\n{ONLY_MINE}",
    }
    assert {t.name for t in researchers[0].tools} == {"search_docs", "files__read_text_file"}  # 웹 Tool은 web worker만(Q12)
    assert "web__tavily_search" in [t.name for t in llm.of("web")[0].tools]
    [aggregator] = llm.of("aggregator")
    prompt = aggregator.messages[0].content
    assert prompt.startswith(
        "원래 질문: 보고서 써 줘\n\n"
        "[researcher-1] 맡은 작업: C14 경로 표\n결과:\n경로 답 [C14, 경로 표]\n\n"
        "[web-2] 맡은 작업: Send 공식 설명\n결과:\nSend 답 [Send](https://x.dev)\n\n"
        "[researcher-3] 맡은 작업: C15 voting 규칙\n결과:\nvoting 답 [C15, voting]\n\n"
    )
    assert "보고서를 원하면" in prompt  # 형식은 사용자 프롬프트로(Q10-1)
    assert (result.status, result.workers, result.answer) == ("completed", ["researcher", "web", "researcher"], "보고서")
    assert lines[:3] == [
        "나누기: orchestrator가 worker 3종으로 작업을 정한다",
        "작업 3개:\n  researcher-1: C14 경로 표\n  web-2: Send 공식 설명\n  researcher-3: C15 voting 규칙",
        "동시 실행: researcher-1, web-2, researcher-3",
    ]
    assert sorted(lines[3:6]) == ["researcher-1 완료", "researcher-3 완료", "web-2 완료"] and lines[6] == "합치기: aggregator"


def test_unknown_worker_goes_to_general_without_asking_or_delegating():  # Q14
    llm = ByContent({"orchestrator": final("docs: 문서 찾기\nutility: 시각 확인"), "문서 찾기": final("a"), "시각 확인": final("b"), "aggregator": final("ab")})
    result, lines = start(NAME, "둘 다", llm, tools=WITH_ASKING)
    assert result.workers == ["general", "utility"] and "알 수 없는 worker docs → general" in lines
    general = next(r for r in llm.of(MAIN) if "문서 찾기" in r.messages[0].content)
    names = {t.name for t in general.tools}
    assert "search_docs" in names and not names & {"ask_user", "delegate", "files__write_file"}


def test_one_task_still_goes_through_aggregator():  # Q7
    llm = ByContent({"orchestrator": final("researcher: 하나만"), "하나만": final("a"), "aggregator": final("보고서")})
    result, lines = start(NAME, "질문", llm)
    assert result.answer == "보고서" and len(llm.of("aggregator")) == 1
    assert lines[1] == "작업 1개:\n  researcher-1: 하나만" and not any(line.startswith("동시 실행") for line in lines)


def test_no_tasks_or_orchestrator_error_goes_to_general_main():
    llm = ByContent({"orchestrator": final("모르겠다"), "원래 질문": final("main의 답")})
    result, lines = start(NAME, "원래 질문", llm)
    assert (result.workers, result.answer) == ([], "main의 답")
    assert lines[1] == '나누기 실패("모르겠다") → general' and llm.of(MAIN)[0].messages[0].content == "원래 질문"

    result, lines = start(NAME, "원래 질문", ByContent({"orchestrator": RuntimeError("끊김"), "원래 질문": final("답")}))
    assert result.answer == "답" and lines[1].startswith('나누기 실패("오류: RuntimeError")')


def test_a_failed_worker_is_reported_and_aggregator_still_runs():  # Q9
    llm = ByContent({"orchestrator": final("researcher: 문서 작업\nweb: 웹 작업"), "문서 작업": final("문서 답"), "웹 작업": RuntimeError("끊김"), "aggregator": final("합친 답")})
    result, lines = start(NAME, "둘 다", llm)
    assert result.answer == "합친 답" and "web-2 실패(RuntimeError: 끊김)" in lines
    assert "[web-2] 맡은 작업: 웹 작업\n결과:\n조사 실패: RuntimeError: 끊김" in llm.of("aggregator")[0].messages[0].content


def test_all_workers_failing_skips_aggregator():  # Q9
    llm = ByContent({"orchestrator": final("researcher: 문서 작업\nweb: 웹 작업"), "문서 작업": RuntimeError("a"), "웹 작업": RuntimeError("b")})
    result, lines = start(NAME, "둘 다", llm)
    assert llm.of("aggregator") == [] and "합치기: aggregator" not in lines
    assert result.answer == "researcher-1: 조사 실패: RuntimeError: a\nweb-2: 조사 실패: RuntimeError: b"


def test_aggregated_answer_citations_are_checked_against_search_results():
    llm = ByContent({"orchestrator": final("researcher: 문서 작업\nweb: 웹 작업"), "문서 작업": final("a"), "웹 작업": final("b"), "aggregator": final("답 [MCP, 개요] [지어낸 문서, 1절]")})
    _, lines = start(NAME, "둘 다", llm, knowledge=SimpleNamespace(seen={"MCP, 개요"}))
    assert lines[-1] == "※ 검색 결과에 없는 출처: 지어낸 문서, 1절"


def test_workers_really_run_at_the_same_time():
    barrier = threading.Barrier(3, timeout=5)  # 차례로 돌면 첫 가지가 나머지를 기다리다 시간 초과로 실패한다.
    llm = ByContent({"orchestrator": final("researcher: 작업 가\nresearcher: 작업 나\nutility: 작업 다"), "작업 가": final("a"), "작업 나": final("b"), "작업 다": final("c"), "aggregator": final("abc")}, barrier)
    result, lines = start(NAME, "셋", llm)
    assert result.answer == "abc" and not any("실패" in line for line in lines)


def test_branch_trace_and_progress_names_carry_worker_and_task_number(tmp_path):  # Q15
    tracer = Tracer(tmp_path)
    llm = ByContent({"orchestrator": final("researcher: 작업 가\nresearcher: 작업 나"), "작업 가": final("a"), "작업 나": final("b"), "aggregator": final("ab")})
    start(NAME, "둘", llm, tracer=tracer)
    rows = [json.loads(line) for line in tracer.path.read_text(encoding="utf-8").splitlines()]
    assert sorted((r.get("agent") for r in rows if r["kind"] == "llm"), key=str) == [None, None, "researcher-1", "researcher-2"]

    names = []
    team = AgentTeam(load_agents(AGENTS), wrap_llm=lambda name, llm: lambda request, on_text_delta=None: names.append(name) or llm(request))
    base = AgentConfig(llm_config=TEST_CONFIG, tools=list(POOL), llm=llm)
    WorkflowRunner(WORKFLOWS, WorkflowContext(config=team.config_for(MAIN, base), team=team, base=base), InMemorySaver()).start(NAME, "둘")
    assert names[0] == "orchestrator" and sorted(names[1:3]) == ["researcher-1", "researcher-2"] and names[3] == "aggregator"


def test_unavailable_without_agent_team():
    base = AgentConfig(llm_config=TEST_CONFIG, tools=POOL)
    with pytest.raises(ValueError, match="에이전트"):
        WorkflowRunner(WORKFLOWS, WorkflowContext(config=base), InMemorySaver()).start(NAME, "x")


# --- 평가 (Q17) ---


def test_eval_checks_that_every_expected_worker_was_used():
    case = Case(id="o", category="orchestrator", question="x", workflow=NAME, expect_workers=["researcher", "web"])
    assert check_workflow(case, "completed", 0, workers=["web", "researcher", "researcher", "utility"]) == []
    assert check_workflow(case, "completed", 0, workers=["researcher"]) == ["worker web을 쓰지 않았다(쓴 worker: ['researcher'])"]
    assert check_workflow(case, "completed", 0, workers=None) == ["worker researcher, web을 쓰지 않았다(쓴 worker: [])"]


def test_orchestrator_workers_does_not_import_other_workflows():
    text = (Path(__file__).resolve().parent.parent / "src/ai_agent_dock/workflows/orchestrator_workers.py").read_text(encoding="utf-8")
    others = ["answer_check", "prompt_chaining", "routing", "sectioning", "voting"]
    assert [o for o in others if f"from .{o} import" in text or f"import {o}" in text] == []
