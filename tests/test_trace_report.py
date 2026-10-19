"""추적 리포트 테스트: 세션 목록, 세션별 통계, 전체 통계, 세션별 비교."""

import json

from ai_agent_dock.core.observability.report import all_stats, format_compare, format_list, format_session, format_total


def write_session(directory, session, turns):
    """turns: 턴마다 (llm 호출 수, tool 호출 수, 시간 ms, 입력 토큰, 출력 토큰, 출처)."""
    rows = [{"session": session, "time": "2026-10-18T10:00:00.000", "turn": 0, "kind": "session_start", "model": "m"}]
    for n, (llm, tools, ms, tin, tout, source) in enumerate(turns, start=1):
        rows.append({"session": session, "time": f"2026-10-18T10:0{n}:00.000", "turn": n, "kind": "turn_start"})
        for _ in range(llm):
            rows.append({"session": session, "time": "t", "turn": n, "kind": "llm", "input_tokens": tin, "output_tokens": tout, "input_tokens_source": source, "output_tokens_source": source})
        for _ in range(tools):
            rows.append({"session": session, "time": "t", "turn": n, "kind": "tool"})
        rows.append({"session": session, "time": f"2026-10-18T10:0{n}:30.000", "turn": n, "kind": "turn_end", "duration_ms": ms})
    (directory / f"{session}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def make(tmp_path):
    write_session(tmp_path, "20261018-100000-aaaa", [(2, 1, 100.0, 10, 5, "provider"), (1, 0, 50.0, 20, 4, "tiktoken")])
    write_session(tmp_path, "20261018-110000-bbbb", [(3, 2, 300.0, 30, 6, "provider")])
    return all_stats(tmp_path)


def test_session_stats_count_calls_time_and_tokens_per_turn(tmp_path):
    first = make(tmp_path)[0]
    assert [(t.llm_calls, t.tool_calls, t.duration_ms) for t in first.turns] == [(2, 1, 100.0), (1, 0, 50.0)]
    assert (first.total("input_tokens"), first.total("output_tokens"), first.total("estimated")) == (40, 14, 2)
    assert first.model == "m"


def test_list_shows_every_session_with_its_turn_count(tmp_path):
    text = format_list(make(tmp_path))
    assert "세션 2개" in text and "20261018-100000-aaaa" in text and "턴 2" in text and "턴 1" in text


def test_session_view_has_a_row_per_turn_and_a_total(tmp_path):
    text = format_session(make(tmp_path)[0])
    assert text.count("\n") == 4  # 제목, 머리줄, 턴 2줄, 합계
    assert "합계" in text


def test_total_sums_all_sessions_and_averages_per_turn(tmp_path):
    text = format_total(make(tmp_path))
    assert "세션 2개, 턴 3개" in text
    assert "LLM 호출: 합계 6.0, 턴당 평균 2.0" in text
    assert "tiktoken으로 계산한 토큰 값: 2개" in text


def test_compare_has_one_row_per_session(tmp_path):
    text = format_compare(make(tmp_path))
    assert len(text.splitlines()) == 3 and "20261018-110000-bbbb" in text


def test_cached_tokens_are_summed_and_command_records_are_not_turns(tmp_path):
    rows = [
        {"session": "s", "time": "t", "turn": 1, "kind": "llm", "input_tokens": 2010, "output_tokens": 5, "cached_input_tokens": 2000},
        {"session": "s", "time": "t", "turn": 1, "kind": "compact", "before_tokens": 10, "after_tokens": 5},
        {"session": "s", "time": "t", "turn": 1, "kind": "session_rename", "old_name": "a", "new_name": "b"},
        {"session": "s", "time": "t", "turn": 1, "kind": "workflow_node", "node": "write"},
    ]
    (tmp_path / "s.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    stats = all_stats(tmp_path)[0]
    assert stats.total("cached_tokens") == 2000 and stats.total("llm_calls") == 1
    assert "캐시 입력 토큰: 합계 2,000.0" in format_total([stats])


def test_list_shows_saved_session_names(tmp_path):
    text = format_list(make(tmp_path), {"20261018-100000-aaaa": "고래 테스트"})
    assert "20261018-100000-aaaa" in text and "[고래 테스트]" in text and text.count("[") == 1
