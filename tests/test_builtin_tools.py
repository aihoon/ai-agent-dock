"""기본 제공 Tool 테스트. 계산기가 안전하게 동작하는지(허용하지 않는 입력과 폭주하는 수식 포함)를 확인한다."""

from datetime import datetime

import pytest

from ai_agent_dock.core.tools import BUILTIN_TOOLS, calculate, get_current_time


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("17*23", 391), ("2**10", 1024), ("-3+5", 2), ("10/4", 2.5), ("(1+2)*3", 9), ("--4", 4)],
)
def test_calculate_basic_arithmetic(expression, expected):
    assert calculate(expression) == expected


def test_calculate_division_by_zero_raises_so_the_loop_can_report_it():
    with pytest.raises(ZeroDivisionError):
        calculate("1/0")


@pytest.mark.parametrize("expression", ["2**101", "9**9**9", "2**-101"])
def test_calculate_rejects_runaway_exponents_quickly(expression):
    with pytest.raises(ValueError, match="지수"):
        calculate(expression)


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('echo hi')", "abs(-1)", "a+1", "'x'*3", "True+1", "[1]+[2]", "1 if 1 else 2", "1<2"],
)
def test_calculate_rejects_anything_but_plain_arithmetic(expression):
    with pytest.raises(ValueError, match="지원하지 않는"):
        calculate(expression)


def test_calculate_reports_malformed_expression_as_syntax_error():
    with pytest.raises(SyntaxError):
        calculate("1+")


def test_get_current_time_is_iso_format_with_timezone():
    parsed = datetime.fromisoformat(get_current_time())

    assert parsed.tzinfo is not None


def test_builtin_tools_are_well_formed_and_callable_through_their_schema():
    by_name = {t.name: t for t in BUILTIN_TOOLS}

    assert set(by_name) == {"get_current_time", "calculate"}
    assert all(t.description and t.parameters["type"] == "object" for t in BUILTIN_TOOLS)
    assert by_name["calculate"].parameters["required"] == ["expression"]
    assert by_name["calculate"].func(expression="1+1") == 2
