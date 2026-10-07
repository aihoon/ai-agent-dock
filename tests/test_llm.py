"""LLM 호출 계층의 제공자 선택 테스트. 네트워크는 쓰지 않는다."""

from types import SimpleNamespace

import pytest

from ai_agent_dock.core.llm import client as llm_client


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nope")

    with pytest.raises(ValueError, match="nope"):
        llm_client.call_llm([], [])


@pytest.mark.parametrize("name", ["openai", "anthropic"])
def test_registered_providers_load_once_and_expose_call(name):
    module = llm_client._provider(name)

    assert callable(module.call)
    assert llm_client._provider(name) is module  # 두 번째부터는 캐시된 모듈을 돌려준다.


def test_call_llm_dispatches_to_selected_provider(monkeypatch):
    seen = []
    fake = SimpleNamespace(call=lambda history, tools, system: seen.append(system) or "응답")
    monkeypatch.setattr(llm_client, "_provider", lambda name: seen.append(name) or fake)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")

    assert llm_client.call_llm([], [], "시스템") == "응답"
    assert seen == ["anthropic", "시스템"]


def test_default_provider_is_openai_when_unset(monkeypatch):
    seen = []
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr(llm_client, "_provider", lambda name: seen.append(name) or SimpleNamespace(call=lambda *a: None))

    llm_client.call_llm([], [])

    assert seen == ["openai"]
