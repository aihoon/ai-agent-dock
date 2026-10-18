"""간단한 CLI 데모. 사용법: uv run python examples/cli.py "지금 몇 시야? 17*23은?"

실제 모델을 호출한다(.env의 LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL 사용).
Tool은 플랫폼의 기본 제공 Tool(ai_agent_dock.core.tools)을 쓴다.
"""

import logging
import sys

from dotenv import load_dotenv

from ai_agent_dock.core.llm import config_from_env
from ai_agent_dock.core.runtime.loop import run_agent
from ai_agent_dock.core.runtime.prompt import build_system_prompt
from ai_agent_dock.core.runtime.types import AgentSettings
from ai_agent_dock.core.tools import BUILTIN_TOOLS

if __name__ == "__main__":
    load_dotenv()  # 프로그램 시작점에서 .env를 한 번만 읽는다. LLM_PROVIDER 등이 여기서 정해진다.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    question = " ".join(sys.argv[1:]) or "지금 몇 시야? 그리고 17*23은 얼마야?"
    settings = AgentSettings(config=config_from_env(), tools=BUILTIN_TOOLS, system_prompt=build_system_prompt())
    result = run_agent(settings, question)
    print(result.answer)
    print(f"[status={result.status}, steps={result.steps}]")
