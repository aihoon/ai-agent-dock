"""대화형 CLI. 여러 턴의 대화를 이어 가며, 앞의 질문과 답을 기억한다.

사용법:
  uv run python examples/chat.py                       # .env의 LLM_PROVIDER로 대화
  uv run python examples/chat.py --provider anthropic  # 제공자를 인자로 지정(환경 변수보다 우선)
  uv run python examples/chat.py --max-steps 5         # 한 턴에서 모델을 호출하는 횟수 상한
  uv run python examples/chat.py --no-stream           # 일괄 응답(생략하면 .env의 AGENT_STREAM, 없으면 스트리밍)

명령: /exit(종료), /reset(대화 이력 비우기), /history(이력 보기), /tools(사용 중인 Tool 보기), /help(명령 목록)
스트리밍이면 답의 글자를 도착하는 대로 보여 주고, 일괄 응답이면 턴이 끝난 뒤 한 번에 보여 준다.
어느 쪽이든 진행 상황(모델 호출, Tool 실행)은 한 줄씩 보여 준다. 턴 도중 Ctrl+C는 그 턴만 취소한다.
"""

import argparse
import logging
from dataclasses import replace
from functools import partial

from dotenv import load_dotenv

from ai_agent_dock.core.llm import Message, Tool, call_llm
from ai_agent_dock.core.session import Session
from ai_agent_dock.core.tools import BUILTIN_TOOLS
from ai_agent_dock.env import env_bool

HELP = "/exit 종료 | /reset 대화 이력 비우기 | /history 이력 보기 | /tools 사용 중인 Tool 보기 | /help 명령 목록"
SHOW = 80  # /history에서 한 항목을 이만큼만 보여 준다.


class Progress:
    """모델 호출과 Tool 실행을 감싸서 진행 상황을 한 줄씩 출력한다. 루프 코드는 건드리지 않는다.

    스트리밍으로 글자 조각이 출력 중이면 줄이 끝나지 않은 상태(open)다. 다음 진행 줄은 그 줄을 먼저 끝내고 출력한다.
    """

    def __init__(self):
        self.step = 0
        self.open = False  # 글자 조각을 출력하다 줄바꿈을 아직 못 한 상태

    def begin_turn(self):
        self.step = 0

    def end_line(self):
        """출력 중인 글자 줄이 있으면 줄바꿈으로 끝낸다."""
        if self.open:
            print(flush=True)
            self.open = False

    def on_text_delta(self, text: str):
        print(text, end="", flush=True)
        self.open = True

    def _say(self, message: str):
        self.end_line()
        print(message, flush=True)

    def wrap_llm(self, llm):
        def traced(history, tools, system_prompt, on_text_delta=None):
            self.step += 1
            self._say(f"  · 모델 호출 중 (스텝 {self.step})")
            if on_text_delta is None:
                return llm(history, tools, system_prompt)
            return llm(history, tools, system_prompt, on_text_delta=on_text_delta)

        return traced

    def wrap_tool(self, tool: Tool) -> Tool:
        def traced(**kwargs):
            self._say(f"  · Tool 실행: {tool.name}")
            return tool.func(**kwargs)

        return replace(tool, func=traced)


def brief(message: Message) -> str:
    text = (message.content or "").replace("\n", " ")
    text = text if len(text) <= SHOW else text[:SHOW] + "…"
    if message.role == "assistant" and message.tool_calls:
        text += f" [Tool 요청: {', '.join(c.name for c in message.tool_calls)}]"
    if message.role == "tool" and message.is_error:
        text = "(오류) " + text
    return f"{message.role:9s}: {text}"


def describe_tool(tool: Tool) -> str:
    """Tool 하나를 `이름(인자, ...)`과 설명 두 줄로 설명한다. 필수 인자 뒤에는 *를 붙인다."""
    schema = tool.parameters
    required = set(schema.get("required", []))
    args = ", ".join(f"{name}{'*' if name in required else ''}" for name in schema.get("properties", {}))
    return f"{tool.name}({args})\n      {tool.description}"


def main():
    parser = argparse.ArgumentParser(description="대화형 CLI")
    parser.add_argument("--provider", help="LLM 제공자(openai, anthropic). 생략하면 .env의 LLM_PROVIDER, 없으면 openai")
    parser.add_argument("--max-steps", type=int, help="한 턴에서 모델을 호출하는 횟수 상한(생략하면 .env의 AGENT_MAX_STEPS 또는 10)")
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        help="스트리밍(--stream) 또는 일괄 응답(--no-stream). 생략하면 .env의 AGENT_STREAM, 없으면 스트리밍",
    )
    args = parser.parse_args()

    load_dotenv()  # 시작점에서 .env를 한 번만 읽는다.
    logging.basicConfig(level=logging.ERROR)  # 진행 표시와 섞이지 않게 로그는 오류만 보인다.
    try:
        # 우선순위: 인자 > 환경 변수 AGENT_STREAM > 기본값(스트리밍)
        stream = args.stream if args.stream is not None else env_bool("AGENT_STREAM", True)
    except ValueError as e:
        raise SystemExit(f"설정 오류: {e}")

    progress = Progress()
    llm = progress.wrap_llm(partial(call_llm, provider=args.provider))
    session = Session([progress.wrap_tool(t) for t in BUILTIN_TOOLS], max_steps=args.max_steps, llm=llm)

    print(f"대화를 시작합니다({'스트리밍' if stream else '일괄 응답'}). 명령을 보려면 /help 를 입력하세요.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            if line == "/exit":
                break
            elif line == "/reset":
                session.reset()
                print("대화 이력을 비웠습니다.")
            elif line == "/history":
                print(f"(메시지 {len(session.history)}개)")
                for m in session.history:
                    print(f"  {brief(m)}")
            elif line == "/tools":
                print(f"(Tool {len(session.tools)}개, *는 필수 인자)")
                for t in session.tools:
                    print(f"  {describe_tool(t)}")
            elif line == "/help":
                print(HELP)
            else:
                print(f"알 수 없는 명령입니다: {line}  ({HELP})")
            continue

        progress.begin_turn()
        try:
            result = session.ask(line, on_text_delta=progress.on_text_delta if stream else None)
        except KeyboardInterrupt:  # Ctrl+C는 이 턴만 취소한다. 세션 이력은 이 턴 이전 그대로 남는다.
            progress.end_line()
            print("(중단됨: 이 턴은 이력에 반영되지 않았습니다)")
            continue
        except Exception as e:  # 모델 호출 실패, 스트림 끊김 등. 세션 이력은 이 턴 이전 그대로 남는다.
            progress.end_line()
            print(f"오류: {type(e).__name__}: {e}\n(이 턴은 이력에 반영되지 않았습니다)")
            continue
        progress.end_line()
        if not stream or not result.answer:  # 스트리밍이면 답은 이미 화면에 나왔으므로 다시 출력하지 않는다.
            print(result.answer or "(답변 없음)")
        if result.status == "max_steps":
            print(f"※ 최대 스텝({result.steps})에 도달해 중단되었습니다. 이어서 '계속해'라고 말해 보세요.")


if __name__ == "__main__":
    main()
