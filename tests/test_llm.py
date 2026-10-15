"""LLM 호출 계층의 제공자 선택 테스트. 네트워크는 쓰지 않는다.

제공자는 설정(LLMConfig.provider)으로 고른다. 환경 변수 LLM_PROVIDER는 config_from_env가 설정을 만들 때만 읽는다.
"""

from types import SimpleNamespace

import pytest

from ai_agent_dock.core.llm import LLMConfig, LLMRequest
from ai_agent_dock.core.llm import client as llm_client


def request(provider="openai"):
    return LLMRequest(config=LLMConfig(provider=provider, model="m", max_output_tokens=None), messages=[])


def test_unknown_provider_is_rejected_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nope")

    with pytest.raises(ValueError, match="nope"):
        llm_client.config_from_env()


def test_unknown_provider_in_config_is_rejected():
    with pytest.raises(ValueError, match="nope"):
        llm_client.call_llm(request("nope"))


@pytest.mark.parametrize("name", ["openai", "anthropic"])
def test_registered_providers_load_once_and_expose_call_and_config(name):
    module = llm_client._provider(name)

    assert callable(module.call)
    assert callable(module.config_from_env)
    assert llm_client._provider(name) is module  # 두 번째부터는 캐시된 모듈을 돌려준다.


def test_call_llm_dispatches_by_config_provider(monkeypatch):
    seen = []
    fake = SimpleNamespace(call=lambda req, on_text_delta: seen.append(req.config.model) or "응답")
    monkeypatch.setattr(llm_client, "_provider", lambda name: seen.append(name) or fake)

    assert llm_client.call_llm(request("anthropic")) == "응답"
    assert seen == ["anthropic", "m"]


def test_config_provider_argument_beats_environment_variable(monkeypatch):
    seen = []
    monkeypatch.setattr(llm_client, "_provider", lambda name: seen.append(name) or SimpleNamespace(config_from_env=lambda: None))
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    llm_client.config_from_env(provider="anthropic")

    assert seen == ["anthropic"]


def test_unknown_provider_argument_is_rejected_even_when_env_is_valid(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    with pytest.raises(ValueError, match="nope"):
        llm_client.config_from_env(provider="nope")


def test_default_provider_is_openai_when_unset(monkeypatch):
    seen = []
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr(llm_client, "_provider", lambda name: seen.append(name) or SimpleNamespace(config_from_env=lambda: None))

    llm_client.config_from_env()

    assert seen == ["openai"]
