"""워크플로 테스트(C10, 가짜 모델): answer-check의 분기와 상한, 승인 대기와 재개, 추적 기록, 의존 방향."""

import json
import re
import sqlite3
from pathlib import Path

from fakes import TEST_CONFIG, FakeLLM, final, tool_call
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from ai_agent_dock.core.observability import Tracer
from ai_agent_dock.core.orchestration import WAITING, WorkflowContext, WorkflowRunner
from ai_agent_dock.core.retrieval import Hit, Knowledge
from ai_agent_dock.core.runtime.types import AgentSettings
from ai_agent_dock.core.tools import BUILTIN_TOOLS
from ai_agent_dock.workflows import WORKFLOWS
from ai_agent_dock.workflows.answer_check import NO_EVIDENCE, citation_problem

SRC = Path(__file__).resolve().parent.parent / "src" / "ai_agent_dock"
CITE = "에이전트 루프, 최대 스텝"


class FakeStore:
    def search(self, query, top_k=3):
        return [Hit(cite=CITE, section="최대 스텝", text="최대 스텝은 10이다.", score=0.9)]


def cited_answer():
    return [tool_call("s1", "search_docs", query="최대 스텝"), final(f"최대 스텝은 10이다 [{CITE}]")]


def make_runner(responses, checkpointer=None, tracer=None, knowledge=True, repeat_last=False):
    settings = AgentSettings(config=TEST_CONFIG, tools=BUILTIN_TOOLS, llm=FakeLLM(responses, repeat_last=repeat_last), tracer=tracer)
    kb = Knowledge(store=FakeStore(), seen=set(), files=1, chunks=1) if knowledge else None
    return WorkflowRunner(WORKFLOWS, WorkflowContext(settings=settings, knowledge=kb, max_revisions=2), checkpointer or InMemorySaver())


def test_citation_check_passes_only_with_searched_citations():
    assert citation_problem("답", {CITE}) == "출처 없음"
    assert citation_problem("답 [지어낸 문서, 1절]", {CITE}) == "검색에 없던 출처: 지어낸 문서, 1절"
    assert citation_problem(f"답 [{CITE}]", {CITE}) == ""


def test_cited_answer_goes_straight_to_approval_and_approve_finishes():
    runner = make_runner(cited_answer())
    waiting = runner.start("answer-check", "최대 스텝은?")
    assert (waiting.status, waiting.revisions) == (WAITING, 0) and CITE in waiting.draft

    done = runner.resume(waiting.run_id, "approve")
    assert done.status == "approved" and CITE in done.answer and runner.pending() == []


def test_answer_without_citation_is_revised_up_to_the_limit_then_gives_up():
    runner = make_runner([final("출처 없는 답")], repeat_last=True)
    done = runner.start("answer-check", "최대 스텝은?")
    assert (done.status, done.revisions) == ("no_evidence", 2)
    assert done.answer.startswith(NO_EVIDENCE)
    retry_prompt = runner.context.settings.llm.calls[1][0].content  # 두 번째 답 작성의 첫 메시지
    assert "[이전 답]" in retry_prompt and "출처 없음" in retry_prompt


def test_nodes_report_progress_lines():  # C10 D24
    lines = []
    runner = make_runner([final("출처 없는 답"), *cited_answer()])
    runner.context.progress = lines.append
    runner.start("answer-check", "최대 스텝은?")
    assert lines == [
        "답 작성",
        "출처 검사: 출처 없음 (검색한 출처 0개)",
        "고쳐 쓰기 1/2",
        "답 다시 작성 (고칠 점: 출처 없음)",
        "출처 검사: 통과 → 사람 승인 (검색한 출처 1개)",
    ]


def test_reject_ends_without_an_answer_and_revise_goes_back_to_writing():
    runner = make_runner(cited_answer() + cited_answer())
    first = runner.start("answer-check", "최대 스텝은?")
    again = runner.resume(first.run_id, "revise:더 짧게")
    assert (again.status, again.revisions) == (WAITING, 1)
    assert "사용자 요청: 더 짧게" in runner.context.settings.llm.calls[2][0].content

    rejected = runner.resume(again.run_id, "reject")
    assert rejected.status == "rejected" and rejected.answer is None


def test_waiting_run_survives_a_restart_with_a_new_checkpointer(tmp_path):
    db = tmp_path / "checkpoints.sqlite"
    conn = sqlite3.connect(db, check_same_thread=False)
    waiting = make_runner(cited_answer(), SqliteSaver(conn)).start("answer-check", "최대 스텝은?")
    conn.close()

    conn = sqlite3.connect(db, check_same_thread=False)  # 프로그램을 껐다 켠 것처럼 새 연결과 새 실행기
    restarted = make_runner([], SqliteSaver(conn))
    [found] = restarted.pending()
    assert (found.run_id, found.question, found.status) == (waiting.run_id, "최대 스텝은?", WAITING)
    assert restarted.resume(found.run_id, "approve").status == "approved"  # 모델을 다시 부르지 않는다(응답 없음)
    conn.close()


def test_unknown_workflow_and_missing_index_are_refused():
    runner = make_runner([], knowledge=False)
    for name, message in [("nope", "없는 워크플로"), ("answer-check", "문서 색인이 없습니다")]:
        try:
            runner.start(name, "질문")
            raise AssertionError("시작하면 안 된다")
        except ValueError as e:
            assert message in str(e)


def test_trace_records_workflow_steps(tmp_path):
    tracer = Tracer(tmp_path)
    runner = make_runner(cited_answer(), tracer=tracer)
    runner.resume(runner.start("answer-check", "최대 스텝은?").run_id, "approve")
    rows = [json.loads(line) for line in tracer.path.read_text(encoding="utf-8").splitlines()]
    kinds = [r["kind"] for r in rows if r["kind"].startswith("workflow_")]
    assert kinds[0] == "workflow_start" and kinds[-1] == "workflow_end"
    assert {"workflow_interrupt", "workflow_resume"} <= set(kinds)
    assert [r["node"] for r in rows if r["kind"] == "workflow_node"] == ["write", "check", "approve"]
    assert "llm" in {r["kind"] for r in rows}  # 노드 안 에이전트의 줄도 남는다


def test_only_orchestration_and_workflows_import_langgraph():
    allowed = {SRC / "core" / "orchestration", SRC / "workflows"}
    for path in SRC.rglob("*.py"):
        if not any(folder in path.parents for folder in allowed):
            assert not re.search(r"^\s*(from|import) langgraph", path.read_text(encoding="utf-8"), re.M), path
    for path in (SRC / "core").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "ai_agent_dock.workflows" not in text and "..workflows" not in text, path


# --- 채팅 명령(C10 D5, D6) ---


def make_ctx(tmp_path, responses):
    from ai_agent_dock.core.commands import CommandContext
    from ai_agent_dock.core.memory import MemoryStore
    from ai_agent_dock.core.session import Session
    from ai_agent_dock.core.session.store import SessionStore
    from ai_agent_dock.core.tools import ToolRegistry

    runner = make_runner(responses)
    return CommandContext(
        session=Session(runner.context.settings),
        registry=ToolRegistry(),
        skills=[],
        store=SessionStore(tmp_path / "sessions"),
        memory=MemoryStore(tmp_path / "memory.md"),
        make_prompt=lambda: "지침",
        summarize=lambda text: "요약",
        knowledge=runner.context.knowledge,
        workflows=runner,
    )


def run(ctx, line):
    from ai_agent_dock.core.commands import find_command

    command, arg = find_command(line)
    return command.run(ctx, arg)


def test_workflow_command_waits_for_approval_and_approved_result_joins_the_chat(tmp_path):
    ctx = make_ctx(tmp_path, cited_answer())
    text = run(ctx, "/workflow answer-check 최대 스텝은?")
    assert text.startswith("[승인 요청] answer-check") and ctx.waiting is not None
    assert ctx.session.history == []  # 중간 단계는 채팅 이력에 넣지 않는다.

    done = ctx.answer_workflow("approve")
    assert done.startswith("워크플로 끝: 승인") and ctx.waiting is None
    assert [m.role for m in ctx.session.history] == ["user", "assistant"]
    assert ctx.session.history[0].content == "최대 스텝은?" and ctx.store.list()  # 세션도 저장된다.


def test_rejected_workflow_does_not_join_the_chat(tmp_path):
    ctx = make_ctx(tmp_path, cited_answer())
    run(ctx, "/workflow answer-check 최대 스텝은?")
    assert ctx.answer_workflow("reject") == "워크플로 끝: 거부 (고쳐 쓴 횟수 0)"
    assert ctx.session.history == []


def test_workflows_lists_waiting_runs_and_resume_brings_back_the_approval_request(tmp_path):
    ctx = make_ctx(tmp_path, cited_answer())
    run(ctx, "/workflow answer-check 최대 스텝은?")
    ctx.waiting = None  # 메뉴에서 답하지 않고 넘어간 것과 같다.

    listing = run(ctx, "/workflows")
    assert "answer-check:" in listing and "1. answer-check  최대 스텝은?" in listing
    assert run(ctx, "/workflow-resume 1").startswith("[승인 요청]") and ctx.waiting is not None
    assert run(ctx, "/workflow-resume 9").startswith("사용법: /workflow-resume 번호")
    assert run(ctx, "/workflow answer-check").startswith("사용법: /workflow 이름 요청")


# --- 평가(C10 D22) ---


def test_workflow_case_answers_approvals_in_order_and_checks_status_and_revisions():
    from ai_agent_dock.core.evals import Case, evaluate

    def make_settings(_case):
        return AgentSettings(config=TEST_CONFIG, tools=BUILTIN_TOOLS, llm=FakeLLM(cited_answer() + cited_answer()))

    def workflows(settings):
        kb = Knowledge(store=FakeStore(), seen=set(), files=1, chunks=1)
        return WorkflowRunner(WORKFLOWS, WorkflowContext(settings=settings, knowledge=kb), InMemorySaver())

    case = Case(id="w", category="workflow", question="최대 스텝은?", workflow="answer-check",
                workflow_decisions=["revise:짧게", "approve"], expect_status="approved", expect_max_revisions=1, expect_citation=True)
    assert evaluate(case, make_settings, workflows=workflows).passed

    strict = Case(**{**case.__dict__, "expect_max_revisions": 0, "expect_status": "rejected"})
    failures = evaluate(strict, make_settings, workflows=workflows).failures
    assert failures == ["워크플로가 rejected가 아니라 approved로 끝났다", "고쳐 쓴 횟수 1가 상한 0를 넘었다"]


def test_answer_check_agent_gets_search_docs_but_not_ask_user():
    from ai_agent_dock.core.hitl import make_ask_user_tool

    runner = make_runner(cited_answer())
    seen_tools = []
    real_llm = runner.context.settings.llm

    def spy(request, on_text_delta=None):
        seen_tools.append({t.name for t in request.tools})
        return real_llm(request)

    runner.context.settings.tools = [*BUILTIN_TOOLS, make_ask_user_tool(lambda prompt: "")]
    runner.context.settings.llm = spy
    runner.start("answer-check", "최대 스텝은?")
    assert "search_docs" in seen_tools[0] and "ask_user" not in seen_tools[0]  # C10 D23


def test_safety_applies_inside_the_workflow_agent():  # C11 D17
    from ai_agent_dock.core.safety import with_safety

    runner = make_runner(cited_answer())
    runner.context.settings, _ = with_safety(runner.context.settings)
    runner.start("answer-check", "최대 스텝은?")
    search_result = runner.context.settings.llm.calls[1][-1].content  # 두 번째 호출에 실린 search_docs 결과
    assert search_result.startswith('<untrusted source="search_docs">')
