"""corrective-rag 워크플로 테스트(C22, 가짜 모델, 가짜 색인): retrieve → grade(grader) → 재검색 또는 generate → check.

가짜 모델과 실행 도우미는 C15 테스트의 것을 쓴다(요청 내용으로 답한다). 가짜 색인은 검색어마다 정해 둔 조각을 돌려준다.
"""

import json
from pathlib import Path

from fakes import final
from test_parallelization import ByContent, start

from ai_agent_dock.core.agents import MAIN
from ai_agent_dock.core.evals import Case
from ai_agent_dock.core.evals.score import check, check_workflow
from ai_agent_dock.core.retrieval import Hit, Knowledge
from ai_agent_dock.core.runtime.types import AgentResult
from ai_agent_dock.workflows.corrective_rag import MAX_RETRIES, NO_EVIDENCE, TOP_K, read_grades

AGENTS = Path(__file__).resolve().parent.parent / "agents"
NAME = "corrective-rag"
QUESTION = "plan-and-execute는 실행을 최대 몇 번까지 해?"
RUNS = Hit(cite="C18.  Plan-and-Execute, 세부 설계", section="세부 설계", text="실행은 모두 합쳐 8번(MAX_RUNS)까지다.", score=0.6)
REPLAN = Hit(cite="C18.  Plan-and-Execute, 흐름", section="흐름", text="단계마다 replan이 돈다.", score=0.5)
WORKERS = Hit(cite="C16.  Orchestrator-workers, 세부 설계", section="세부 설계", text="작업은 최대 5개다.", score=0.5)
CITED = "실행은 8번까지입니다 [C18.  Plan-and-Execute, 세부 설계]."


class Store:
    """검색어마다 정해 둔 조각을 돌려주는 가짜 색인. 받은 검색어와 top_k를 모은다."""

    def __init__(self, hits):
        self.hits, self.queries = hits, []

    def search(self, query, top_k=3):
        self.queries.append((query, top_k))
        return self.hits.get(query, [])


def grades(*values, rewrite=""):
    return final(json.dumps({"grades": list(values), "rewrite": rewrite}, ensure_ascii=False))


def run(hits, grader, writer=final(CITED)):
    store = Store(hits)
    llm = ByContent({"grader": grader, MAIN: writer})
    result, lines = start(NAME, QUESTION, llm, knowledge=Knowledge(store=store, seen=set(), files=1, chunks=1))
    return result, lines, llm, store


def chunks_given(llm):
    return llm.of(MAIN)[0].messages[0].content


# --- grader 답 읽기 (Q12) ---


def test_read_grades_reads_json_even_with_text_around_and_checks_the_length():
    label = '평가입니다.\n```json\n{"grades": [" Correct ", "incorrect"], "rewrite": " MAX_RUNS "}\n```'
    assert read_grades(label, 2) == (["correct", "incorrect"], "MAX_RUNS")
    assert read_grades('{"grades": ["ambiguous"]}', 1) == (["ambiguous"], "")
    for label in ["모르겠다", "", '{"grades": ["correct"]}', '{"grades": ["맞음", "correct"]}', '{"grades": "correct"}', '{"rewrite": "x"}']:
        assert read_grades(label, 2) is None, label


def test_agent_md_examples_read():  # 형식이 AGENT.md와 코드에서 어긋나지 않게
    text = (AGENTS / "grader" / "AGENT.md").read_text(encoding="utf-8")
    examples = [line.strip() for line in text.splitlines() if line.strip().startswith('{"grades"')]
    assert len(examples) == 2 and all(read_grades(e, 3) is not None for e in examples)


# --- 그래프 ---


def test_correct_on_first_search_answers_without_retrieving_again_and_gets_only_correct_and_ambiguous():
    result, lines, llm, store = run({QUESTION: [RUNS, REPLAN, WORKERS]}, grades("correct", "ambiguous", "incorrect"))
    assert (result.status, result.revisions, result.answer) == ("completed", 0, CITED)
    assert store.queries == [(QUESTION, TOP_K)]
    given = chunks_given(llm)
    assert RUNS.text in given and REPLAN.text in given and WORKERS.text not in given
    assert llm.of(MAIN)[0].tools == []  # 고른 조각만으로 답한다(Tool 없음)
    grader_prompt = llm.of("grader")[0].messages[0].content
    assert '<untrusted source="search">' in grader_prompt and f"[{RUNS.cite}]" in grader_prompt
    assert any("Grade 완료: correct 1 · ambiguous 1 · incorrect 1 → generate" in line for line in lines)
    assert any("출처 검사: 통과" in line for line in lines)


def test_no_correct_rewrites_the_query_and_retrieves_again():
    hits = {QUESTION: [WORKERS, REPLAN], "MAX_RUNS 실행 상한": [REPLAN, RUNS]}
    grader = [grades("incorrect", "ambiguous", rewrite="MAX_RUNS 실행 상한"), grades("correct")]
    result, lines, llm, store = run(hits, grader)
    assert (result.status, result.revisions) == ("completed", 1)
    assert [q for q, _ in store.queries] == [QUESTION, "MAX_RUNS 실행 상한"]
    second = llm.of("grader")[1].messages[0].content
    assert RUNS.text in second and REPLAN.text not in second  # 이미 평가한 조각은 다시 보내지 않는다
    assert f"- {QUESTION}\n- MAX_RUNS 실행 상한" in second  # 지금까지 쓴 검색어를 보여 준다
    given = chunks_given(llm)
    assert RUNS.text in given and REPLAN.text in given and WORKERS.text not in given  # 앞 라운드의 ambiguous도 모은다
    assert any("Retrieve 2/3 완료: 새 조각 1개 (검색어: MAX_RUNS 실행 상한)" in line for line in lines)


def test_limit_with_only_ambiguous_answers_with_them():
    hits = {QUESTION: [REPLAN], "a": [WORKERS], "b": []}
    grader = [grades("ambiguous", rewrite="a"), grades("incorrect", rewrite="b")]
    result, _, llm, store = run(hits, grader)
    assert (result.status, result.revisions) == ("completed", MAX_RETRIES)
    assert len(store.queries) == MAX_RETRIES + 1 and len(llm.of("grader")) == 2  # 새 조각이 없으면 grader를 부르지 않는다
    assert REPLAN.text in chunks_given(llm)


def test_limit_without_usable_chunks_ends_with_no_evidence_and_does_not_write():
    result, lines, llm, _ = run({QUESTION: [WORKERS]}, grades("incorrect", rewrite="x"))
    assert (result.status, result.answer, result.revisions) == ("no_evidence", NO_EVIDENCE, MAX_RETRIES)
    assert llm.of(MAIN) == []


def test_empty_or_same_rewrite_falls_back_to_the_question():
    result, _, _, store = run({QUESTION: [WORKERS]}, grades("incorrect", rewrite=QUESTION))
    assert [q for q, _ in store.queries] == [QUESTION] * (MAX_RETRIES + 1)

    result, _, _, store = run({QUESTION: [WORKERS], "a": [REPLAN]}, [grades("incorrect", rewrite="a"), grades("incorrect", rewrite="a")])
    assert [q for q, _ in store.queries] == [QUESTION, "a", QUESTION]  # 이미 쓴 검색어도 원래 질문으로


def test_unreadable_grades_ask_again_once_then_count_as_ambiguous():
    result, _, llm, _ = run({QUESTION: [RUNS]}, [final("모르겠다"), final("여전히 모르겠다")])
    assert (result.status, len(llm.of("grader"))) == ("completed", 2)  # 모두 ambiguous. 다시 찾아도 새 조각이 없어 grader를 더 부르지 않는다
    assert RUNS.text in chunks_given(llm)

    result, _, llm, _ = run({QUESTION: [RUNS]}, [final("모르겠다"), grades("correct")])
    assert (result.status, result.revisions, len(llm.of("grader"))) == ("completed", 0, 2)


def test_grader_error_counts_the_round_as_ambiguous():
    result, _, llm, _ = run({QUESTION: [RUNS]}, RuntimeError("down"))
    assert result.status == "completed" and RUNS.text in chunks_given(llm)


def test_citation_problem_is_reported_without_rewriting():
    result, lines, llm, _ = run({QUESTION: [RUNS]}, grades("correct"), writer=final("8번입니다 [지어낸 문서, 1절]"))
    assert result.status == "completed" and len(llm.of(MAIN)) == 1
    assert any("※ 검색 결과에 없는 출처: 지어낸 문서, 1절" in line for line in lines)


def test_unavailable_without_index():
    llm = ByContent({MAIN: final("x")})
    try:
        start(NAME, QUESTION, llm, knowledge=None)
    except ValueError as e:
        assert "문서 색인" in str(e)
    else:
        raise AssertionError("색인 없이 시작했다")


# --- 평가 칸 (Q14, Q16) ---


def test_expect_source_and_min_revisions():
    case = Case(id="x", category="corrective-rag", question="q", expect_source="C18", expect_min_revisions=1)
    answer = lambda text: AgentResult(answer=text, status="completed", steps=0, history=[])
    assert check(case, answer(CITED)) == []
    assert check(case, answer("8번 [C16.  Orchestrator-workers, 세부 설계]")) == ["출처에 C18 문서가 없다"]
    assert check(case, answer("8번")) == ["출처에 C18 문서가 없다"]
    assert check_workflow(case, "completed", 0) == ["다시 찾은 횟수 0가 하한 1보다 적다"]
    assert check_workflow(case, "completed", 1) == []
