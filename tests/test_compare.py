"""패턴 비교표 테스트(C90): 결과 파일의 compare 케이스를 질문 × 실행 방법 줄로 만들고 CSV로 쓴다."""

import csv
import json

from ai_agent_dock.core.evals import Case, CaseResult, load_cases, save_run
from ai_agent_dock.core.evals.compare import compare_rows, write_csv


def result(id, passes=3, llm_calls=2.0, input_tokens=1000.0, seconds=10.0, llm_verdicts=(), shape="noevidence", level="easy", category="compare"):
    return CaseResult(
        id=id, category=category, runs=3, passes=passes, llm_verdicts=list(llm_verdicts),
        llm_calls=llm_calls, input_tokens=input_tokens, output_tokens=100.0, seconds=seconds, shape=shape, level=level,
    )


def saved(tmp_path, results):
    return json.loads(save_run(tmp_path, "eval-x", "openai", "m", 3, results).read_text(encoding="utf-8"))


# --- 케이스 칸 ---


def test_case_reads_shape_and_level_and_the_result_file_keeps_them(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id":"compare-simple-1-main","category":"compare","question":"q","shape":"simple","level":"easy"}\n'
        '{"id":"tool-1","category":"tool","question":"q"}\n',
        encoding="utf-8",
    )
    compare, other = load_cases(path)
    assert (compare.shape, compare.level) == ("simple", "easy")
    assert (other.shape, other.level) == (None, None)  # 기존 케이스는 그대로
    data = saved(tmp_path, [result("compare-simple-1-main", shape="simple")])
    assert data["cases"][0]["shape"] == "simple" and data["cases"][0]["level"] == "easy"


# --- 비교표 줄 ---


def test_rows_are_one_per_question_and_runner_with_ratios_against_main(tmp_path):
    data = saved(tmp_path, [
        result("compare-noevidence-1-main", passes=1, llm_calls=4.0, input_tokens=2000.0, seconds=10.0, llm_verdicts=["통과 2/4 (실패: …)", "통과 4/4"]),
        result("compare-noevidence-1-corrective-rag", llm_calls=2.0, input_tokens=1000.0, seconds=15.0, llm_verdicts=["통과 4/4"]),
        result("tool-calc-1", category="tool", shape=None, level=None),  # compare가 아니면 뺀다
    ])
    main, crag = compare_rows(data)
    assert (main["shape"], main["level"], main["question"], main["runner"]) == ("noevidence", "easy", 1, "main")
    assert (main["passes"], main["runs"], main["judge"]) == (1, 3, 0.75)  # (2/4 + 4/4) / 2
    assert (main["calls_x"], main["tokens_x"], main["seconds_x"]) == (1.0, 1.0, 1.0)
    assert crag["runner"] == "corrective-rag"  # 하이픈이 든 워크플로 이름
    assert (crag["calls_x"], crag["tokens_x"], crag["seconds_x"]) == (0.5, 0.5, 1.5)
    assert crag["judge"] == 1.0


def test_judge_is_the_passed_share_of_judged_criteria_and_skips_errors_and_missing():
    rows = compare_rows({"run": "r", "cases": [
        {**vars(result("compare-judgment-2-main", llm_verdicts=["통과 2/2, 판정 안 함 2", "오류: TimeoutError"])), "passed": True},
        {**vars(result("compare-judgment-2-debate", llm_verdicts=[])), "passed": True},
    ]})
    assert rows[0]["judge"] == 1.0  # 판정한 기준 2개 중 2개(판정 안 함은 분모에서 뺀다, Q17). 오류는 평균에서 뺀다
    assert rows[1]["judge"] == ""  # LLM 판정이 없으면 비운다


def test_ratios_are_blank_without_a_main_row_or_with_zero_main_cost():
    rows = compare_rows({"run": "r", "cases": [
        {**vars(result("compare-classify-1-voting")), "passed": True},
        {**vars(result("compare-simple-1-main", llm_calls=0.0, input_tokens=0.0, seconds=0.0, shape="simple")), "passed": True},
        {**vars(result("compare-simple-1-routing", shape="simple")), "passed": True},
    ]})
    voting, main, routing = rows
    assert (voting["calls_x"], voting["tokens_x"]) == ("", "")
    assert routing["calls_x"] == ""  # main이 0이면 나눌 수 없다


def test_rows_are_ordered_by_shape_question_and_main_first():
    rows = compare_rows({"run": "r", "cases": [
        {**vars(result("compare-simple-2-routing", shape="simple")), "passed": True},
        {**vars(result("compare-simple-1-routing", shape="simple")), "passed": True},
        {**vars(result("compare-simple-1-main", shape="simple")), "passed": True},
    ]})
    assert [(r["question"], r["runner"]) for r in rows] == [(1, "main"), (1, "routing"), (2, "routing")]


# --- CSV ---


def test_csv_has_a_bom_for_excel_and_no_question_text(tmp_path):
    data = saved(tmp_path, [result("compare-noevidence-1-main"), result("compare-noevidence-1-corrective-rag")])
    path = write_csv(tmp_path, data)
    assert path.name == "eval-x-compare.csv"
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM: 엑셀에서 한글이 깨지지 않는다
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    assert [r["runner"] for r in rows] == ["main", "corrective-rag"]
    assert rows[0]["run"] == "eval-x" and rows[1]["calls_x"] == "1.0"
    assert "question_text" not in rows[0] and set(rows[0]) >= {"shape", "level", "question", "passes", "judge", "seconds"}


def test_case_defaults_do_not_break_existing_cases():
    assert Case(id="a", category="tool", question="q").shape is None
