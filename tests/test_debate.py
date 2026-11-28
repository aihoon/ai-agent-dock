"""debate 워크플로 테스트(C20, 가짜 모델): propose(토론자 3명 동시) → tally → debate … → agreed 또는 decide(aggregator).

토론자 셋은 같은 질문을 받아 요청만으로는 누가 누구인지 모른다. 그래서 가지 이름(debater-1~3)으로 감싼 모델 호출(wrap_llm)로 답을 고른다.
"""

import threading

from fakes import TEST_CONFIG, final
from langgraph.checkpoint.memory import InMemorySaver
from test_parallelization import AGENTS, POOL, tool

from ai_agent_dock.core.agents import MAIN, AgentTeam, load_agents
from ai_agent_dock.core.orchestration import WorkflowContext, WorkflowRunner
from ai_agent_dock.core.runtime.types import AgentConfig
from ai_agent_dock.workflows import WORKFLOWS
from ai_agent_dock.workflows.debate import MAX_ROUNDS, REVIEW, normalize, read_conclusion, without_review

NAME = "debate"
Q = "공 값은?"


class ByName:
    """이름(debater-1, aggregator 등) → 응답 목록. 차례로 쓰고 마지막 것은 되풀이한다. 예외면 던진다."""

    def __init__(self, answers):
        self.answers = {k: list(v) if isinstance(v, list) else [v] for k, v in answers.items()}
        self.requests = []
        self.lock = threading.Lock()

    def wrap(self, name, llm):
        def call(request, on_text_delta=None):
            with self.lock:
                self.requests.append((name, request))
                answers = self.answers[name]
                response = answers.pop(0) if len(answers) > 1 else answers[0]
            if isinstance(response, Exception):
                raise response
            return response

        return call

    def of(self, name):
        return [r for n, r in self.requests if n == name]


def run(answers, question=Q):
    fake = ByName(answers)
    team = AgentTeam(load_agents(AGENTS), wrap_llm=fake.wrap)
    base = AgentConfig(llm_config=TEST_CONFIG, tools=[*POOL, tool("ask_user")], llm=lambda r, on_text_delta=None: None)
    lines = []
    context = WorkflowContext(config=team.config_for(MAIN, base), team=team, base=base, progress=lines.append)
    return WorkflowRunner(WORKFLOWS, context, InMemorySaver()).start(NAME, question), lines, fake


def say(conclusion, why="근거"):
    return final(f"결론: {conclusion}\n{why}")


def test_first_round_agreement_ends_without_debate_or_aggregator():
    result, lines, fake = run({"debater-1": say("50원"), "debater-2": say("**50**원입니다."), "debater-3": say("50 원")})
    assert (result.status, result.revisions, result.answer) == ("agreed", 0, "결론: 50원\n근거")
    assert fake.of("aggregator") == [] and len(fake.requests) == 3
    assert lines == [
        "워크플로 시작: debate (토론자 3명, 고치기 최대 4번)",
        "Propose 시작: debater-1, debater-2, debater-3",
        "Propose 완료: 50원 | **50**원입니다. | 50 원 → 모두 같음",
    ]


def test_split_then_debate_once_then_agree():
    result, lines, fake = run(
        {"debater-1": [say("50원"), say("50원")], "debater-2": [say("100원", "틀린 근거"), say("50원", "고침")], "debater-3": say("50원")}
    )
    assert (result.status, result.revisions) == ("agreed", 1)
    first = fake.of("debater-2")[0].messages[0].content
    retry = fake.of("debater-2")[1].messages[0].content
    assert first == Q
    assert "[당신의 지난 답]\n결론: 100원\n틀린 근거" in retry
    assert "debater-1:\n<untrusted" in retry and "debater-3:\n<untrusted" in retry and "debater-2:\n" not in retry
    assert lines[2:] == ["Propose 완료: 50원 | 100원 | 50원 → 갈림", "Debate 1/4 시작", "Debate 1/4 완료: 50원 | 50원 | 50원 → 모두 같음"]


def test_agreed_answer_drops_the_review_section_but_others_see_it():
    reviewed = f"결론: 아니요\n근거\n\n{REVIEW}\ndebater-2의 처음 답은 틀렸다"
    result, _, fake = run({"debater-1": [say("아니요"), final(reviewed)], "debater-2": [say("예"), say("아니요")], "debater-3": [say("아니요"), say("아니요")]})
    assert (result.status, result.answer) == ("agreed", "결론: 아니요\n근거")
    retry = fake.of("debater-1")[1].messages[0].content
    assert f"맨 끝 {REVIEW} 절에만" in retry and "예/아니요 질문이면 예 또는 아니요만" in retry


def test_split_until_the_limit_goes_to_the_aggregator_with_the_last_round_only():
    answers = {"debater-1": say("SQLite 유지"), "debater-2": say("지금은 SQLite 유지"), "debater-3": [say("Postgres", "첫 근거"), say("Postgres로 이전")]}
    result, lines, fake = run({**answers, "aggregator": final("과반: SQLite 유지")})
    assert (result.status, result.revisions, result.answer) == ("decided", MAX_ROUNDS, "과반: SQLite 유지")
    assert len(fake.requests) == 3 * (MAX_ROUNDS + 1) + 1
    prompt = fake.of("aggregator")[0].messages[0].content
    assert "Postgres로 이전" in prompt and "첫 근거" not in prompt and f"원래 질문: {Q}" in prompt
    assert lines[-3:] == [f"Debate 4/4 완료: SQLite 유지 | 지금은 SQLite 유지 | Postgres로 이전 → 갈림", "Decide 시작: aggregator (갈린 채 상한)", "Decide 완료"]


def test_aggregator_error_falls_back_to_the_first_debater():
    result, lines, _ = run({"debater-1": say("A"), "debater-2": say("B"), "debater-3": say("C"), "aggregator": RuntimeError("끊김")})
    assert result.status == "decided" and result.answer == "정리 실패: debater-1의 답입니다.\n\n결론: A\n근거"
    assert lines[-1] == "Decide 실패(RuntimeError: 끊김) → debater-1의 답으로 끝냅니다"


def test_a_failed_debater_is_skipped_and_all_failing_first_round_ends_failed():
    result, lines, fake = run({"debater-1": say("50원"), "debater-2": RuntimeError("끊김"), "debater-3": say("50원")})
    assert (result.status, result.revisions) == ("agreed", 0) and lines[-1] == "Propose 완료: 50원 | 실패 | 50원 → 모두 같음"

    result, lines, _ = run({f"debater-{n}": final("  ") for n in (1, 2, 3)})
    assert result.status == "failed" and result.answer.startswith("토론자가 모두 답하지 못했습니다: 실패: ValueError: 빈 답")
    assert lines[-1] == "Propose 완료: 실패 | 실패 | 실패 → 답 없음"


def test_failed_debater_shows_no_answer_to_the_others():
    _, _, fake = run({"debater-1": [say("A"), say("A")], "debater-2": [RuntimeError("끊김"), say("A")], "debater-3": [say("B"), say("A")]})
    retry = fake.of("debater-1")[1].messages[0].content
    assert "debater-2:\n(답 없음)" in retry
    assert "[당신의 지난 답]\n(답 없음)" in fake.of("debater-2")[1].messages[0].content


def test_debaters_get_writer_tools_without_approval_asking_or_delegating():
    _, _, fake = run({f"debater-{n}": say("50원") for n in (1, 2, 3)})
    names = {t.name for t in fake.of("debater-1")[0].tools}
    assert {"calculate", "get_current_time", "search_docs", "files__list_directory", "web__tavily_search"} <= names
    assert not names & {"files__write_file", "ask_user", "delegate"}
    assert fake.of("debater-1")[0].system_prompt.startswith("당신은 토론에 참여한 토론자다")


# --- 결론 줄 ---


def test_read_conclusion_and_normalize():
    assert read_conclusion("\n결론: **1,027**입니다.\n근거") == "**1,027**입니다."
    assert read_conclusion("**결론: 금요일**\n근거") == "금요일"
    assert read_conclusion("공은 50원이다.") == "공은 50원이다."  # 결론 줄이 없으면 첫 줄 전체
    assert normalize("**1,027**입니다.") == normalize("1027") == "1027"
    assert normalize("'SQLite 유지'") == normalize("SQLite유지") == "sqlite유지"
    assert normalize("아니요.") == normalize("**아니요**") == "아니요"
    assert without_review(f"결론: 예\n근거\n{REVIEW}\n지적") == "결론: 예\n근거" and without_review("결론: 예") == "결론: 예"
    assert normalize("50원") != normalize("공은 50원이다")  # 같은 뜻을 다르다고 볼 수는 있다(4.3절)


def test_normalize_keeps_signs_decimals_and_fractions():  # C90 리뷰 A3: 다른 뜻을 같다고 보지 않는다
    assert normalize("-3") != normalize("3")
    assert normalize("3.5") != normalize("35") and normalize("1/2") != normalize("12")
    assert normalize("**3.5**입니다.") == normalize("3.5") and normalize("1,027") == "1027"
