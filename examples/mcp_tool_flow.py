"""MCP 흐름 디버그 실습: 내장 Tool과 MCP Tool을 한 레지스트리에 넣고, 실제 LLM이 그 Tool을 고르고 루프가 실행하는 턴을 돌린다.

VS Code에서 이 파일을 열고 브레이크포인트를 찍은 뒤 'Python 디버그'로 실행한다.
.env의 LLM 키가 필요하다(모델 호출은 턴 하나에 보통 2회). MCP 서버는 시험용 서버(tests/mcp_test_server.py)를 자식 프로세스로 띄운다.
"""

import json
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

from dotenv import load_dotenv

from ai_agent_dock.core.llm import config_from_env
from ai_agent_dock.core.runtime.types import AgentConfig
from ai_agent_dock.core.session import Session
from ai_agent_dock.core.tools import BUILTIN_TOOLS, DEFAULT_MAX_RESULT_CHARS, ToolRegistry
from ai_agent_dock.core.tools.mcp import ServerSpec, connect_mcp_servers, load_config
from ai_agent_dock.env import env_float, env_int

SERVER = Path(__file__).resolve().parent.parent / "tests" / "mcp_test_server.py"
QUESTION = (
    "다음 Tool을 각각 한 번씩 호출해 줘: test__echo(text='안녕하세요'), calculate(expression='2+3*4'), "
    "get_current_time(), test__fail(). 결과를 항목별로 알려 줘. 실패한 것은 실패라고 말해 줘."
)


def main():
    load_dotenv()
    registry = ToolRegistry(max_result_chars=env_int("TOOL_RESULT_MAX_CHARS", DEFAULT_MAX_RESULT_CHARS))

    print("1) 내장 Tool 등록")
    registry.add_all(BUILTIN_TOOLS)

    with ExitStack() as stack:
        print("2) MCP 설정 파일 작성 후 읽기")
        config_dir = stack.enter_context(tempfile.TemporaryDirectory())
        config_path = Path(config_dir) / "mcp.json"
        config = {"servers": {"test": {"command": sys.executable, "args": [str(SERVER)]}}}
        config_path.write_text(json.dumps(config), encoding="utf-8")
        servers = load_config(config_path)

        print("3) MCP 서버 연결과 MCP Tool 등록")
        connected = stack.enter_context(
            connect_mcp_servers(
                servers,
                call_timeout=env_float("MCP_TOOL_TIMEOUT", 30.0),
                start_timeout=env_float("MCP_START_TIMEOUT", 60.0),
            )
        )
        for server, tools in connected.items():
            registry.add_all(tools, source=f"mcp:{server}")

        print("4) 등록된 Tool 목록(출처 포함)")
        for tool in registry.tools():
            print(f"  {tool.name:<22} [{registry.source_of(tool.name)}]")

        print("5) LLM 턴 실행: 모델이 Tool을 고르고, 루프가 실행하고, 결과를 모델에 돌려준다")
        session = Session(AgentConfig(llm_config=config_from_env(), tools=registry.tools()))
        result = session.ask(QUESTION)
        print(f"  상태: {result.status}, 모델 호출: {result.steps}회")
        for message in result.history[1:]:
            if message.role == "assistant" and message.tool_calls:
                for call in message.tool_calls:
                    print(f"  [모델 요청] {call.name}({call.arguments})")
            elif message.role == "tool":
                tag = "오류 관찰" if message.is_error else "결과"
                print(f"  [{tag}] {(message.content or '')[:100]!r}")
            elif message.role == "assistant":
                print(f"  [최종 답변]\n{message.content}")

        print("6) 직접 호출: 인자 오류, 긴 결과 자르기, 점이 든 이름")
        by_name = {tool.name: tool for tool in registry.tools()}
        for label, name, arguments in [
            ("calculate 인자 누락", "calculate", {}),
            ("긴 결과", "test__long_text", {"chars": 25000}),
            ("점이 든 이름", "test__dotted_name", {}),
        ]:
            try:
                output = by_name[name].func(**arguments)
                print(f"  {label}: {len(output)}자, 끝부분 {output[-60:]!r}")
            except Exception as e:
                print(f"  {label}: {type(e).__name__}: {e}")

        print("7) 시간 초과: 제한 0.5초, 서버는 2초 걸리는 Tool 호출")
        slow = stack.enter_context(connect_mcp_servers({"slow": servers["test"]}, call_timeout=0.5, start_timeout=60))
        try:
            output = {t.name: t for t in slow["slow"]}["slow__sleep"].func(seconds=2)
            print(f"  결과: {output!r}")
        except Exception as e:
            print(f"  예외: {type(e).__name__}: {e}")

        print("8) 서버 시작 실패: 없는 명령을 가진 서버는 경고를 내고 건너뛴다")
        broken = stack.enter_context(
            connect_mcp_servers({"broken": ServerSpec("no-such-command-xyz", [])}, call_timeout=5, start_timeout=5)
        )
        print(f"  연결된 서버: {list(broken)}")


if __name__ == "__main__":
    main()
