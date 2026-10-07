"""환경 변수 읽기 도우미 테스트."""

import pytest

from ai_agent_dock.env import env_bool, env_int, env_str


def test_env_str_uses_default_when_unset_or_empty(monkeypatch):
    monkeypatch.delenv("X_NAME", raising=False)
    assert env_str("X_NAME", "기본") == "기본"

    monkeypatch.setenv("X_NAME", "")
    assert env_str("X_NAME", "기본") == "기본"

    monkeypatch.setenv("X_NAME", "값")
    assert env_str("X_NAME", "기본") == "값"


def test_env_int_uses_default_when_unset_or_blank(monkeypatch):
    monkeypatch.delenv("X_NUM", raising=False)
    assert env_int("X_NUM", 7) == 7
    assert env_int("X_NUM") is None

    monkeypatch.setenv("X_NUM", "  ")
    assert env_int("X_NUM", 7) == 7


def test_env_int_reads_a_valid_value(monkeypatch):
    monkeypatch.setenv("X_NUM", "42")

    assert env_int("X_NUM", 7) == 42


@pytest.mark.parametrize("bad", ["abc", "1.5", "0", "-3"])
def test_env_int_rejects_bad_values_and_names_the_variable(monkeypatch, bad):
    monkeypatch.setenv("X_NUM", bad)

    with pytest.raises(ValueError, match="X_NUM"):
        env_int("X_NUM", 7)


def test_env_bool_uses_default_when_unset_or_blank(monkeypatch):
    monkeypatch.delenv("X_FLAG", raising=False)
    assert env_bool("X_FLAG", True) is True
    assert env_bool("X_FLAG", False) is False

    monkeypatch.setenv("X_FLAG", "  ")
    assert env_bool("X_FLAG", True) is True


@pytest.mark.parametrize(("raw", "expected"), [("true", True), ("TRUE", True), ("1", True), ("false", False), ("False", False), ("0", False), (" true ", True)])
def test_env_bool_reads_valid_values_ignoring_case_and_spaces(monkeypatch, raw, expected):
    monkeypatch.setenv("X_FLAG", raw)

    assert env_bool("X_FLAG", not expected) is expected


@pytest.mark.parametrize("bad", ["yes", "on", "2", "ture"])
def test_env_bool_rejects_other_values_and_names_the_variable(monkeypatch, bad):
    monkeypatch.setenv("X_FLAG", bad)

    with pytest.raises(ValueError, match="X_FLAG"):
        env_bool("X_FLAG", True)
