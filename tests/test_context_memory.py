"""C9 테스트: 세션 저장소, 컴팩션, 장기 메모리, 프롬프트의 메모리, 캐싱, 토큰 합계, memory 평가 실행."""

import json

from fakes import TEST_CONFIG, FakeLLM, final, tool_call

from ai_agent_dock.core.evals import Case, evaluate
from ai_agent_dock.core.llm import AssistantMessage, LLMRequest, LLMResponse, ToolCall, ToolMessage, UserMessage
from ai_agent_dock.core.llm.providers import anthropic as anthropic_provider
from ai_agent_dock.core.llm.providers import openai as openai_provider
from ai_agent_dock.core.memory import MemoryStore
from ai_agent_dock.core.runtime.loop import run_agent
from ai_agent_dock.core.runtime.prompt import DEFAULT_SYSTEM_PROMPT, build_system_prompt
from ai_agent_dock.core.runtime.types import AgentSettings
from ai_agent_dock.core.session import Session
from ai_agent_dock.core.session.compact import SUMMARY_HEADER, compact, split_turns
from ai_agent_dock.core.session.store import SessionRecord, SessionStore, auto_name
from ai_agent_dock.core.tools import BUILTIN_TOOLS


def sample_history():
    call = ToolCall(id="c1", name="calculate", arguments={"expression": "2+3"})
    return [
        UserMessage(content="2+3은?"),
        AssistantMessage(content=None, tool_calls=[call], raw=[{"type": "function_call", "call_id": "c1"}]),
        ToolMessage(tool_call_id="c1", content="5"),
        AssistantMessage(content="5입니다"),
    ]


# --- 세션 저장소 ---


def test_session_round_trip_keeps_text_tool_calls_and_results(tmp_path):
    store = SessionStore(tmp_path)
    store.save(SessionRecord(id="s1", name="계산", provider="openai", history=sample_history()))

    loaded = store.load("s1", "openai")

    assert loaded.name == "계산" and loaded.turns == 1
    assert loaded.history[1].tool_calls[0].arguments == {"expression": "2+3"}
    assert loaded.history[2].content == "5" and loaded.history[1].raw  # 같은 제공자면 raw를 쓴다


def test_loading_with_another_provider_drops_raw(tmp_path):
    store = SessionStore(tmp_path)
    store.save(SessionRecord(id="s1", name="계산", provider="openai", history=sample_history()))
    assert store.load("s1", "anthropic").history[1].raw == []


def test_rename_rejects_a_taken_name_and_only_changes_the_header(tmp_path):
    store = SessionStore(tmp_path)
    a = SessionRecord(id="a", name="첫째", provider="openai", history=sample_history())
    b = SessionRecord(id="b", name="둘째", provider="openai")
    store.save(a), store.save(b)

    store.rename(a, "새 이름")

    data = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    assert data["name"] == "새 이름" and "새 이름" not in json.dumps(data["history"], ensure_ascii=False)
    try:
        store.rename(b, "새 이름")
        raise AssertionError("겹치는 이름을 거절해야 한다")
    except ValueError:
        pass


def test_list_is_newest_first_and_skips_broken_files(tmp_path):
    store = SessionStore(tmp_path)
    store.save(SessionRecord(id="old", name="옛", provider="openai"))
    store.save(SessionRecord(id="new", name="새", provider="openai"))
    (tmp_path / "broken.json").write_text("{깨짐", encoding="utf-8")
    assert [r.id for r in store.list()][0] == "new" and len(store.list()) == 2


def test_auto_name_and_unique_name(tmp_path):
    store = SessionStore(tmp_path)
    name = auto_name("아주 긴 첫 질문이 여기에 들어갑니다 그리고 더 길어요")
    assert len(name.split(" ", 1)[1]) == 20
    store.save(SessionRecord(id="a", name=name, provider="openai"))
    assert store.unique_name(name) == f"{name} (2)"


# --- 컴팩션 ---


def turn(n, tool_result="결과"):
    call = ToolCall(id=f"c{n}", name="files__read_text_file", arguments={})
    return [UserMessage(content=f"질문 {n}"), AssistantMessage(tool_calls=[call]), ToolMessage(tool_call_id=f"c{n}", content=tool_result), AssistantMessage(content=f"답 {n}")]


def test_compact_summarizes_old_turns_and_keeps_recent_turns_verbatim():
    history = [m for n in range(1, 7) for m in turn(n, "x" * 2000)]
    seen = []

    new, report = compact(history, keep_turns=2, summarize=lambda text: seen.append(text) or "요약 내용", count=len)

    assert report.summarized_turns == 4 and report.trimmed_results == 4
    assert "컴팩션으로 잘림" in seen[0] and "x" * 600 not in seen[0]  # 1단계: 큰 Tool 결과를 잘라서 요약에 넘긴다
    assert new[0].content.startswith(SUMMARY_HEADER) and new[0].content.endswith("질문 5")
    assert new[1:] == history[-7:]  # 최근 턴은 원문 그대로
    assert len(split_turns(new)) == 2 and report.after_tokens < report.before_tokens


def test_compact_keeps_tool_call_and_result_pairs():
    history = [m for n in range(1, 6) for m in turn(n)]
    new, _ = compact(history, keep_turns=2, summarize=lambda text: "요약", count=len)
    call_ids = [c.id for m in new if isinstance(m, AssistantMessage) for c in m.tool_calls]
    result_ids = [m.tool_call_id for m in new if isinstance(m, ToolMessage)]
    assert call_ids == result_ids


def test_compact_does_nothing_when_there_are_only_recent_turns():
    history = turn(1)
    new, report = compact(history, keep_turns=4, summarize=lambda text: "요약", count=len)
    assert new == history and report.summarized_turns == 0


# --- 장기 메모리와 프롬프트 ---


def test_memory_add_remove_and_renumber(tmp_path):
    memory = MemoryStore(tmp_path / "memory.md")
    assert memory.items() == []
    memory.add("이름은 훈")
    memory.add("반려견은\n콩")
    memory.add("답은 짧게")

    assert memory.remove(1) == "이름은 훈"
    assert (tmp_path / "memory.md").read_text(encoding="utf-8") == "1. 반려견은 콩\n2. 답은 짧게\n"


def test_memory_goes_right_after_the_base_prompt():
    from pathlib import Path

    from ai_agent_dock.core.retrieval.tool import CITATION_RULE
    from ai_agent_dock.core.skills import SkillMeta

    skill = SkillMeta(name="요약", description="글을 짧게 줄인다", path=Path("x"))
    text = build_system_prompt(skills=[skill], cite_docs=True, memory=["반려견은 콩"])
    assert text.index(DEFAULT_SYSTEM_PROMPT) < text.index("반려견은 콩") < text.index("글을 짧게") < text.index(CITATION_RULE)
    assert "장기 메모리" not in build_system_prompt()


# --- 캐싱과 토큰 ---


def test_anthropic_request_has_cache_marks_and_openai_does_not():
    request = LLMRequest(config=TEST_CONFIG, messages=[UserMessage(content="안녕")], system_prompt="지침", tools=BUILTIN_TOOLS)
    a = anthropic_provider.build_request(request)
    o = openai_provider.build_request(request)
    assert a["system"][0]["cache_control"] == {"type": "ephemeral"} and a["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in json.dumps(o)


def test_cached_tokens_are_copied_from_the_responses():
    from types import SimpleNamespace as NS

    anthropic_usage = NS(input_tokens=10, output_tokens=5, cache_read_input_tokens=2000, cache_creation_input_tokens=0)
    a = anthropic_provider.parse(NS(stop_reason="end_turn", content=[NS(type="text", text="hi")], usage=anthropic_usage))
    assert (a.input_tokens, a.cached_input_tokens) == (2010, 2000)  # 입력은 캐시분을 포함해 맞춘다

    openai_usage = NS(input_tokens=2010, output_tokens=5, input_tokens_details=NS(cached_tokens=2000))
    o = openai_provider.parse(NS(status="completed", output=[], output_text="hi", usage=openai_usage))
    assert (o.input_tokens, o.cached_input_tokens) == (2010, 2000)


def test_session_tracks_last_turn_and_cumulative_usage_separately_from_history_size():
    responses = [
        LLMResponse(text=None, tool_calls=[ToolCall(id="c1", name="calculate", arguments={"expression": "2+3"})], input_tokens=100, output_tokens=10, cached_input_tokens=80),
        LLMResponse(text="5", input_tokens=120, output_tokens=5),
        LLMResponse(text="안녕", input_tokens=150, output_tokens=3),
    ]
    session = Session(AgentSettings(config=TEST_CONFIG, tools=BUILTIN_TOOLS, llm=FakeLLM(responses)))

    session.ask("2+3은?")
    session.ask("안녕")

    assert (session.last_usage.input_tokens, session.last_usage.output_tokens) == (150, 3)
    assert (session.total_usage.input_tokens, session.total_usage.output_tokens, session.total_usage.cached_input_tokens) == (370, 18, 80)
    assert 0 < session.history_tokens() < session.total_usage.input_tokens


# --- memory 평가 실행 ---


def test_evaluate_sends_setup_turns_compacts_and_checks_only_the_last_turn():
    llm = FakeLLM([final("기억할게요"), tool_call("c1", "calculate", expression="1+1"), final("2"), final("훈입니다")])
    compacted = []

    def compactor(history):
        compacted.append(len(history))
        return history

    case = Case(id="m", category="memory", question="내 이름은?", setup_turns=["내 이름은 훈", "1+1은?"], compact_before=True, forbid_tools=["calculate"], expect_contains=["훈"])
    outcome = evaluate(case, lambda _c: AgentSettings(config=TEST_CONFIG, tools=BUILTIN_TOOLS, llm=llm), compactor=compactor)

    assert outcome.passed, outcome.failures  # 앞선 턴의 calculate는 판정에 넣지 않는다
    assert compacted == [6]
