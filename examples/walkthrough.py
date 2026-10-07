"""Agent Loop를 한 단계씩 눈으로 확인하는 데모. 단계마다 무슨 일이 일어나는지 출력한다.

사용법:
  uv run python examples/walkthrough.py                 # 실제 모델로 실행(기본 질문)
  uv run python examples/walkthrough.py "질문 내용"      # 실제 모델, 질문 지정
  uv run python examples/walkthrough.py --scenario error  # 모델 없이: Tool 오류가 나도 계속되는 모습
  uv run python examples/walkthrough.py --scenario loop   # 모델 없이: 끝없이 Tool만 요청해도 종료되는 모습

scenario error/loop는 미리 정한 응답을 쓰는 가짜 모델이라 API 호출과 비용이 없다.
"""

import argparse
import logging
from dataclasses import replace

from cli import TOOLS  # examples/cli.py의 데모 Tool(get_current_time, calculate)
from dotenv import load_dotenv

from ai_agent_dock.core.llm import LLMResponse, Message, Tool, ToolCall, call_llm
from ai_agent_dock.core.runtime.loop import run_agent

SHOW = 200  # 긴 내용은 이만큼만 보여 준다.
SYSTEM = "계산과 시간 조회는 반드시 Tool을 사용해 답한다."


def clip(text, n=SHOW):
    text = str(text).replace("\n", " ")
    return text if len(text) <= n else text[:n] + f"… (+{len(text) - n}자)"


def banner(title):
    print(f"\n{'━' * 8} {title} {'━' * 8}")


def describe(m: Message) -> str:
    """이력 한 항목을 한 줄로 설명한다."""
    if m.role == "user":
        return f"user      : {clip(m.content)}"
    if m.role == "assistant":
        calls = ", ".join(f"{c.name}({c.arguments})" for c in m.tool_calls) or "(없음)"
        return f"assistant : 텍스트={clip(m.content) if m.content else '(없음)'} / Tool 요청={calls}"
    flag = "오류" if m.is_error else "성공"
    return f"tool      : [{flag}] id={m.tool_call_id} → {clip(m.content)}"


class Tracer:
    """모델 호출과 Tool 실행을 감싸서, 지나가는 데이터를 출력한다."""

    def __init__(self, llm):
        self._llm = llm
        self.turn = 0
        self._seen = 0  # 이미 보여 준 이력 항목 수

    def llm(self, history, tools, system):
        self.turn += 1
        banner(f"턴 {self.turn}")
        new = history[self._seen :]
        if new:
            label = "이력에 새로 들어온 항목" if self.turn > 1 else "첫 입력"
            print(f"[{self.turn}-1] {label}:")
            for m in new:
                print(f"       {describe(m)}")
        print(f"[{self.turn}-2] 모델 호출 → 이력 {len(history)}개, Tool {len(tools)}개를 보낸다")
        response = self._llm(history, tools, system)
        print(f"[{self.turn}-3] 모델 응답:")
        print(f"       텍스트   : {clip(response.text) if response.text else '(없음)'}")
        if response.tool_calls:
            for c in response.tool_calls:
                note = f"  ⚠ 오류 표시: {c.error}" if c.error else ""
                print(f"       Tool 요청: {c.name}({c.arguments}) id={c.id}{note}")
            print("       → Tool 요청이 있으므로 루프가 계속된다")
        else:
            print("       → Tool 요청이 없으므로 이 응답이 최종 답변이다")
        self._seen = len(history) + 1  # 방금 응답(assistant)은 여기서 보여 줬으므로 다음엔 건너뛴다.
        return response

    def wrap_tool(self, tool: Tool) -> Tool:
        def traced(**kwargs):
            print(f"[{self.turn}-4] Tool 실행: {tool.name}({kwargs})")
            try:
                result = tool.func(**kwargs)
            except Exception as e:
                print(f"       → 예외 발생: {type(e).__name__}: {e}  (루프는 이 오류를 모델에 알리고 계속한다)")
                raise
            print(f"       → 결과: {clip(result)}")
            return result

        return replace(tool, func=traced)


def scripted(*responses):
    """미리 정한 응답을 차례로 돌려주는 가짜 모델. 응답이 바닥나면 마지막 응답을 반복한다."""
    items = list(responses)
    state = {"n": 0}

    def llm(history, tools, system):
        i = min(state["n"], len(items) - 1)
        state["n"] += 1
        return items[i]

    return llm


def ask(call_id, name, **arguments):
    return LLMResponse(text=None, tool_calls=[ToolCall(call_id, name, arguments)])


SCENARIOS = {
    "error": (
        "1/0을 계산한 뒤, 다시 1+1을 계산해 줘.",
        scripted(
            ask("c1", "calculate", expression="1/0"),
            ask("c2", "calculate", expression="1+1"),
            LLMResponse(text="1/0은 계산할 수 없었고, 1+1은 2입니다."),
        ),
        None,
    ),
    "loop": (
        "시간을 계속 알려 줘.",
        scripted(ask("c1", "get_current_time")),  # 끝없이 같은 Tool만 요청한다.
        3,
    ),
}


def main():
    parser = argparse.ArgumentParser(description="Agent Loop 단계별 출력 데모")
    parser.add_argument("question", nargs="*", help="실제 모델에 보낼 질문(생략하면 기본 질문)")
    parser.add_argument("--scenario", choices=SCENARIOS, help="모델 없이 시나리오 실행")
    parser.add_argument("--max-turns", type=int, help="모델 호출 횟수 상한(생략하면 .env의 AGENT_MAX_TURNS 또는 10)")
    args = parser.parse_args()

    load_dotenv()  # 시작점에서 .env를 한 번만 읽는다.
    logging.basicConfig(level=logging.ERROR)  # 단계 출력이 이 데모의 목적이라, 로그는 오류만 보인다.

    if args.scenario:
        question, model, scenario_turns = SCENARIOS[args.scenario]
        mode = f"가짜 모델(시나리오: {args.scenario})"
    else:
        question = " ".join(args.question) or "지금 몇 시야? 그리고 17*23은 얼마야? 두 결과를 한 문장으로 말해 줘."
        model, scenario_turns = call_llm, None
        mode = "실제 모델(.env의 LLM_PROVIDER)"
    max_turns = args.max_turns or scenario_turns

    tracer = Tracer(model)
    tools = [tracer.wrap_tool(t) for t in TOOLS]

    banner("시작")
    print(f"모드      : {mode}")
    print(f"질문      : {question}")
    print(f"시스템    : {SYSTEM}")
    print(f"Tool      : {', '.join(t.name for t in tools)}")
    print(f"최대 턴   : {max_turns or '(.env의 AGENT_MAX_TURNS 또는 기본 10)'}")

    result = run_agent(question, tools, system=SYSTEM, max_turns=max_turns, llm=tracer.llm)

    banner("결과")
    print(f"종료 사유 : {result.status}  ({'최종 답변으로 종료' if result.status == 'completed' else '최대 턴에 도달해 강제 종료'})")
    print(f"모델 호출 : {result.turns}회")
    print(f"최종 답변 : {result.answer if result.answer else '(없음)'}")
    banner("최종 이력(모델이 마지막에 본 전체 대화)")
    for i, m in enumerate(result.history, 1):
        print(f"{i:2d}. {describe(m)}")


if __name__ == "__main__":
    main()
