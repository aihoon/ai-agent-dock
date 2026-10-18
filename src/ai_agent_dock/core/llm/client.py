"""제공자에 독립적인 LLM 호출. 제공자는 LLMConfig.provider로 고른다.

환경 변수 LLM_PROVIDER는 config_from_env()가 읽어 설정을 만들 때만 쓰이고, 호출 자체는 설정만 본다.
.env를 읽는 일은 이 모듈의 몫이 아니다. 프로그램 시작점(CLI, 서버 등)이 load_dotenv()를 한 번 호출한다.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from importlib import import_module
from types import ModuleType

from ai_agent_dock.env import env_str

from .tokens import fill_tokens
from .types import LLMConfig, LLMRequest, LLMResponse

# 제공자 이름 → 어댑터 모듈. 새 제공자는 providers/ 에 어댑터를 만들고 여기에 한 줄 추가한다.
_PROVIDERS = {
    "openai": ".providers.openai",
    "anthropic": ".providers.anthropic",
}


@cache
def _provider(name: str) -> ModuleType:
    """어댑터 모듈을 처음 쓸 때만 가져온다. 선택되지 않은 제공자의 SDK는 가져오지 않는다."""
    if name not in _PROVIDERS:
        raise ValueError(f"지원하지 않는 LLM_PROVIDER: {name!r} (지원: {', '.join(_PROVIDERS)})")
    return import_module(_PROVIDERS[name], __package__)


def config_from_env(provider: str | None = None) -> LLMConfig:
    """환경 변수로 LLM 설정을 만든다. 시작할 때 한 번 호출한다.

    제공자 우선순위: provider 인자 > 환경 변수 LLM_PROVIDER > 기본값 openai.
    """
    return _provider(provider or env_str("LLM_PROVIDER", "openai")).config_from_env()


def call_llm(request: LLMRequest, on_text_delta: Callable[[str], None] | None = None) -> LLMResponse:
    """모델을 한 번 호출한다. 제공자는 request.config.provider로 고른다.

    on_text_delta를 주면 스트리밍으로 받으며, 답의 글자 조각이 도착할 때마다 호출한다(Tool 인자 조각은 전달하지 않는다).
    반환값은 스트리밍 여부와 관계없이 같은 형식의 LLMResponse다. 스트리밍 여부를 어떻게 정할지는 호출하는 쪽의 몫이다.
    """
    response = _provider(request.config.provider).call(request, on_text_delta)
    return fill_tokens(request, response)  # 제공자가 토큰 수를 주지 않았으면 tiktoken으로 채운다.
