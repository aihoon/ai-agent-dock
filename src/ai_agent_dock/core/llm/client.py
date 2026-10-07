"""제공자에 독립적인 LLM 호출. 제공자는 환경 변수 LLM_PROVIDER로 고른다(없으면 openai).

.env를 읽는 일은 이 모듈의 몫이 아니다. 프로그램 시작점(CLI, 서버 등)이 load_dotenv()를 한 번 호출한다.
"""

from __future__ import annotations

from functools import cache
from importlib import import_module
from types import ModuleType

from ai_agent_dock.env import env_str

from .types import LLMResponse, Message, Tool

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


def call_llm(history: list[Message], tools: list[Tool], system: str | None = None) -> LLMResponse:
    return _provider(env_str("LLM_PROVIDER", "openai")).call(history, tools, system)
