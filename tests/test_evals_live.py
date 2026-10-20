"""실제 모델 평가: evals/cases.jsonl의 케이스를 실제 모델로 실행한다. uv run pytest --live 또는 uv run evals로만 실행된다.

툴 집합은 채팅과 같다(빌트인, 스킬, 문서 검색, mcp.json의 MCP 서버). 승인과 되묻기는 케이스에 적힌 답을 쓴다.
결과는 data/evals/실행번호.json, 추적은 logs/sessions/eval-실행번호.jsonl에 남는다.
"""

import shutil
from contextlib import ExitStack
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

from ai_agent_dock.core.evals import evaluate, last_passed, load_cases, regressions, save_run, scripted_approver, scripted_input, summarize
from ai_agent_dock.core.llm import LLMRequest, UserMessage, call_llm, config_from_env
from ai_agent_dock.core.llm.tokens import request_tokens
from ai_agent_dock.core.session.compact import compact
from ai_agent_dock.core.observability import Tracer, new_session_id
from ai_agent_dock.core.orchestration import WorkflowContext, WorkflowRunner
from ai_agent_dock.core.retrieval import KnowledgeIndex, OpenAIEmbedder, open_knowledge
from ai_agent_dock.core.runtime.prompt import build_system_prompt
from ai_agent_dock.core.runtime.types import AgentSettings
from ai_agent_dock.core.safety import load_safety_policy, with_safety
from ai_agent_dock.core.skills import scan_skills
from ai_agent_dock.core.toolkit import make_builtin_tools
from ai_agent_dock.core.tools import DEFAULT_MAX_RESULT_CHARS, ToolRegistry
from ai_agent_dock.env import env_bool, env_float, env_int, env_str
from ai_agent_dock.workflows import WORKFLOWS

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "evals" / "fixtures"  # safety 평가 전용 자료(C11 D14). 일반 대화의 knowledge/, data/index.sqlite와 섞지 않는다.


def prepare_fixtures():
    """평가용 파일을 파일 서버 폴더(data/sandbox/eval/)에 복사하고, 평가용 문서 색인(data/evals/index.sqlite)을 만든다."""
    shutil.copytree(FIXTURES / "sandbox", ROOT / "data" / "sandbox", dirs_exist_ok=True)
    path = ROOT / "data" / "evals" / "index.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    store = KnowledgeIndex(path, OpenAIEmbedder())
    try:
        store.refresh(FIXTURES / "knowledge")  # 바뀐 문서만 다시 임베딩한다.
    finally:
        store.close()
    return open_knowledge(path)

pytestmark = pytest.mark.live


def connect_mcp(stack: ExitStack) -> dict:
    """mcp.json(또는 MCP_CONFIG)에 적은 서버에 연결한다. 설정 파일이 없으면 MCP 툴 없이 평가한다."""
    path = Path(env_str("MCP_CONFIG", "") or ROOT / "mcp.json")
    if not path.is_file():
        return {}
    from ai_agent_dock.core.tools.mcp import connect_mcp_servers, load_config

    return stack.enter_context(
        connect_mcp_servers(
            load_config(path),
            call_timeout=env_float("MCP_TOOL_TIMEOUT", 30.0),
            start_timeout=env_float("MCP_START_TIMEOUT", 60.0),
        )
    )


def test_cases_with_real_model(request):
    load_dotenv(ROOT / ".env")
    repeat = request.config.getoption("--repeat")
    config = config_from_env(request.config.getoption("--provider"))
    cases = load_cases(ROOT / "evals" / "cases.jsonl")
    skills = scan_skills(ROOT / "skills")
    knowledge = open_knowledge(ROOT / "data" / "index.sqlite")
    fixture_knowledge = prepare_fixtures()
    system_prompt = build_system_prompt(skills=skills, cite_docs=knowledge is not None)
    summarize_config = config  # 컴팩션 요약도 같은 모델로 한다(평가에서는 COMPACT_LLM_* 설정을 쓰지 않는다)

    def summarize_text(text):  # 컴팩션 요약. 평가 결과 요약(summarize)과 이름이 겹치지 않게 한다.
        return call_llm(LLMRequest(config=summarize_config, messages=[UserMessage(content=text)])).text or ""
    run_id = new_session_id("eval-")
    tracer = Tracer(ROOT / "logs" / "sessions" if env_bool("TRACE_ENABLED", True) else None, session_id=run_id)

    with ExitStack() as stack:
        mcp_tools = connect_mcp(stack)
        policy = load_safety_policy(FIXTURES / "safety.json") if "files" in mcp_tools else None  # 안전 정책은 파일 서버 Tool을 막는다.

        def make_settings(case):
            kb = fixture_knowledge if case.knowledge == "fixtures" else knowledge
            registry = ToolRegistry(max_result_chars=env_int("TOOL_RESULT_MAX_CHARS", DEFAULT_MAX_RESULT_CHARS))
            for source, tool in make_builtin_tools(skills=skills, knowledge=kb, input_fn=scripted_input(case.user_inputs)):
                registry.add(tool, source=source)
            for server, tools in mcp_tools.items():
                registry.add_all(tools, source=f"mcp:{server}")
            prompt = build_system_prompt(skills=skills, cite_docs=kb is not None, memory=case.memory_items)
            settings = AgentSettings(
                config=config,
                tools=registry.tools(),
                system_prompt=prompt,
                approve=scripted_approver(case.approvals),
                tracer=tracer,
            )
            return with_safety(settings, policy=policy)[0]  # 채팅과 같은 Safety(C11 D17)

        first = make_settings(cases[0])
        tracer.record(
            "session_start",
            None,
            system_prompt=system_prompt,
            tools=[{"name": t.name, "description": t.description, "parameters": t.parameters} for t in first.tools],
            model=config.model,
            max_steps=first.max_steps,
        )
        def compactor(history):
            tools = first.tools
            count = lambda h: request_tokens(LLMRequest(config=config, messages=h, system_prompt=system_prompt, tools=tools))
            return compact(history, keep_turns=env_int("COMPACT_KEEP_TURNS", 4), summarize=summarize_text, count=count)[0]

        def workflows(settings):  # 워크플로 케이스(C10): 체크포인트는 메모리에 둔다(평가는 재개하지 않는다).
            context = WorkflowContext(settings=settings, knowledge=knowledge, max_revisions=env_int("WORKFLOW_MAX_REVISIONS", 2))
            return WorkflowRunner(WORKFLOWS, context, InMemorySaver())

        results = [evaluate(case, make_settings, repeat, compactor=compactor, workflows=workflows) for case in cases]

    results_dir = ROOT / "data" / "evals"
    previous = last_passed(results_dir)  # 저장하기 전에 직전 실행을 읽는다.
    regressed = regressions(previous or set(), {r.id for r in results if not r.passed})
    path = save_run(results_dir, run_id, config.provider, config.model, repeat, results)
    summary = summarize(results, regressed)
    print(f"\n[평가 {run_id}] {config.provider}/{config.model}, 반복 {repeat}회\n{summary}\n결과 파일: {path}")
    assert all(r.passed for r in results), summary
