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

from ai_agent_dock.core.agents import MAIN, AgentTeam, load_agents
from ai_agent_dock.core.evals import CaseResult, evaluate, last_passed, load_cases, regressions, save_run, scripted_approver, scripted_input, summarize
from ai_agent_dock.core.llm import LLMRequest, UserMessage, call_llm, config_from_env
from ai_agent_dock.core.llm.tokens import request_tokens
from ai_agent_dock.core.memory import MemoryStore
from ai_agent_dock.core.session import Session
from ai_agent_dock.core.session.compact import compact
from ai_agent_dock.core.observability import Tracer, new_session_id
from ai_agent_dock.core.evals.rubric import EVALUATOR
from ai_agent_dock.core.orchestration import WorkflowContext, WorkflowRunner, ask_agent
from ai_agent_dock.core.retrieval import KnowledgeIndex, OpenAIEmbedder, open_knowledge
from ai_agent_dock.core.runtime.types import AgentConfig
from ai_agent_dock.core.safety import load_safety_policy, with_safety
from ai_agent_dock.core.skills import scan_skills
from ai_agent_dock.core.tools import DEFAULT_MAX_RESULT_CHARS, build_pool, connect_mcp
from ai_agent_dock.env import env_bool, env_int, env_str
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


def test_cases_with_real_model(request, tmp_path):
    load_dotenv(ROOT / ".env")
    repeat = request.config.getoption("--repeat")
    llm_config = config_from_env(request.config.getoption("--provider"))
    cases = load_cases(ROOT / "evals" / "cases.jsonl")
    category = request.config.getoption("--category")
    if category:  # 한 카테고리만 먼저 돌린다(C20 Q20). 회귀는 돌린 케이스만 본다.
        cases = [c for c in cases if c.category == category]
        assert cases, f"카테고리 {category}의 케이스가 없다"
    skills = scan_skills(ROOT / "skills")
    knowledge = open_knowledge(ROOT / "data" / "index.sqlite")
    fixture_knowledge = prepare_fixtures()
    notion_knowledge = open_knowledge(ROOT / "data" / "notion.sqlite")  # C22 케이스(knowledge "notion")의 색인. scripts/ingest_notion.py가 만든다.
    indexes = {"fixtures": fixture_knowledge, "notion": notion_knowledge}
    delegated = []  # 이번 실행에서 맡긴 하위 에이전트의 결과. 판정에 하위 에이전트의 Tool 호출도 센다(C12).
    team = AgentTeam(
        load_agents(ROOT / "agents"),
        max_depth=env_int("SUBAGENT_MAX_DEPTH", 1),
        max_steps=env_int("SUBAGENT_MAX_STEPS", 6),
        max_result_chars=env_int("TOOL_RESULT_MAX_CHARS", DEFAULT_MAX_RESULT_CHARS),
        on_end=lambda name, result: delegated.append(result),
    )

    def take_delegated():
        taken = list(delegated)
        delegated.clear()
        return taken

    summarize_config = llm_config  # 컴팩션 요약도 같은 모델로 한다(평가에서는 COMPACT_LLM_* 설정을 쓰지 않는다)

    def summarize_text(text):  # 컴팩션 요약. 평가 결과 요약(summarize)과 이름이 겹치지 않게 한다.
        return call_llm(LLMRequest(config=summarize_config, messages=[UserMessage(content=text)])).text or ""
    run_id = new_session_id("eval-")
    tracer = Tracer(ROOT / "logs" / "sessions" if env_bool("TRACE_ENABLED", True) else None, session_id=run_id)

    with ExitStack() as stack:
        mcp_path = Path(env_str("MCP_CONFIG", "") or ROOT / "mcp.json")  # 설정 파일이 없으면 MCP Tool 없이 평가한다.
        # 평가는 웹 검색(Tavily)에 기대지 않는다. 결과가 매번 달라진다(C12 Q13).
        mcp_tools = connect_mcp(stack, mcp_path if mcp_path.is_file() else None, skip_servers=("web",))
        policy = load_safety_policy(FIXTURES / "safety.json") if "files" in mcp_tools else None  # 안전 정책은 파일 서버 Tool을 막는다.

        last = {}  # 마지막으로 만든 base. evaluate는 make_config(case) 바로 뒤에 workflows를 부른다.

        def make_config(case):
            kb = indexes.get(case.knowledge, knowledge)
            pool = build_pool(skills=skills, knowledge=kb, mcp_tools=mcp_tools, input_fn=scripted_input(case.user_inputs))  # 채팅과 같은 pool(C17 Q21)
            config = AgentConfig(
                llm_config=llm_config,
                tools=pool.tools(),
                approve=scripted_approver(case.approvals),  # 하위 에이전트의 승인도 Tool 이름으로 답한다.
                tracer=tracer,
            )
            base = with_safety(config, policy=policy)[0]  # 채팅과 같은 Safety(C11 D17)
            last["base"], last["case"], last["kb"] = base, case, kb  # routing 케이스가 경로마다 에이전트를 만들 때(C14), reflexion 케이스가 교훈을 꺼낼 때(C19) 쓴다.
            return team.config_for(MAIN, base, skills=skills, memory=case.memory_items)  # 채팅과 같은 main(C12)

        first = make_config(cases[0])
        system_prompt = first.system_prompt
        tracer.record(
            "session_start",
            None,
            system_prompt=system_prompt,
            tools=[{"name": t.name, "description": t.description, "parameters": t.parameters} for t in first.tools],
            model=llm_config.model,
            max_steps=first.max_steps,
        )
        def compactor(history):
            tools = first.tools
            count = lambda h: request_tokens(LLMRequest(config=llm_config, messages=h, system_prompt=system_prompt, tools=tools))
            return compact(history, keep_turns=env_int("COMPACT_KEEP_TURNS", 4), summarize=summarize_text, count=count)[0]

        def workflows(config):  # 워크플로 케이스(C10): 체크포인트는 메모리에 둔다(평가는 재개하지 않는다).
            reflections = MemoryStore(tmp_path / f"reflections-{new_session_id()}.md")  # 실행마다 케이스의 교훈으로 새로 시작한다(C19 Q19).
            for text in last["case"].reflections:
                reflections.add(text)
            context = WorkflowContext(
                config=config, knowledge=last["kb"], max_revisions=env_int("WORKFLOW_MAX_REVISIONS", 4), team=team, base=last["base"], reflections=reflections
            )
            return WorkflowRunner(WORKFLOWS, context, InMemorySaver())

        def session(config):  # 채팅과 같은 대화 세션: main이 넘기면 넘겨받은 에이전트가 이어 답한다(C21 Q20).
            def agent_config(name):
                return team.config_for(name, last["base"], skills=skills, memory=last["case"].memory_items, handoff=True)

            return Session(agent_config(MAIN), agents=agent_config)

        def judge(prompt):  # llm_judge 케이스의 LLM 판정(C17). 기록만 하고 통과/실패에는 넣지 않는다.
            return ask_agent(team.config_for(EVALUATOR, last["base"]), prompt)

        def run_case(case):
            if case.knowledge == "notion" and notion_knowledge is None:  # 건너뛰지 않고 실패로 남긴다(C22 4.6절)
                return CaseResult(id=case.id, category=case.category, runs=repeat, passes=0, failures=["Notion 색인이 없다(uv run python scripts/ingest_notion.py)"])
            return evaluate(case, make_config, repeat, compactor=compactor, workflows=workflows, delegated=take_delegated, judge=judge, session=session)

        results = [run_case(case) for case in cases]

    results_dir = ROOT / "data" / "evals"
    previous = last_passed(results_dir)  # 저장하기 전에 직전 실행을 읽는다.
    regressed = regressions(previous or set(), {r.id for r in results if not r.passed})
    path = save_run(results_dir, run_id, llm_config.provider, llm_config.model, repeat, results)
    summary = summarize(results, regressed)
    print(f"\n[평가 {run_id}] {llm_config.provider}/{llm_config.model}, 반복 {repeat}회\n{summary}\n결과 파일: {path}")
    assert all(r.passed for r in results), summary
