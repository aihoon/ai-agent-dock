"""여러 턴의 대화가 이어지는 과정을 한 단계씩 눈으로 확인하는 데모.

각 턴이 시작될 때 세션 이력에 무엇이 들어 있는지, 스텝마다 무슨 일이 일어나는지, 턴이 끝난 뒤 이력이
어떻게 바뀌는지 출력한다. (C1의 walkthrough.py가 "한 턴 안"을 보여 준다면, 이것은 "턴과 턴 사이"를 보여 준다.)

사용법:
  uv run python examples/chat_walkthrough.py --scenario memory     # 모델 없이: 앞 대화를 기억하는 모습
  uv run python examples/chat_walkthrough.py --scenario fail       # 모델 없이: 도중에 실패해도 이력이 안 깨지는 모습
  uv run python examples/chat_walkthrough.py --scenario maxsteps   # 모델 없이: 최대 스텝 뒤에도 대화가 이어지는 모습
  uv run python examples/chat_walkthrough.py "17*23은 얼마야?" "거기에 2를 곱해 줘"   # 실제 모델로 여러 턴
  uv run python examples/chat_walkthrough.py --provider anthropic "안녕" "내가 방금 뭐라고 했지?"

scenario는 미리 정한 응답을 쓰는 가짜 모델이라 API 호출과 비용이 없다. 질문을 여러 개 주면 한 세션에서 차례로 묻는다.
"""

import argparse
import logging
from functools import partial

from dotenv import load_dotenv
from walkthrough import Tracer, ask, banner, describe, scripted  # C1 워크스루의 도구를 재사용한다.

from ai_agent_dock.core.llm import LLMResponse, call_llm
from ai_agent_dock.core.session import Session
from ai_agent_dock.core.tools import BUILTIN_TOOLS


class TurnTracer(Tracer):
    """Tracer를 턴마다 스텝 번호를 다시 세도록 확장한다."""

    def begin_turn(self):
        self.step = 0


def final(text):
    return LLMResponse(text=text)


def failing_after_tool():
    """도중에 실패하는 가짜 모델: Tool 요청까지는 정상이다가 다음 호출에서 예외를 던진다."""
    items = [ask("c1", "calculate", expression="1+1"), RuntimeError("네트워크 오류(가상)")]
    state = {"n": 0}

    def llm(history, tools, system_prompt):
        item = items[min(state["n"], len(items) - 1)]
        state["n"] += 1
        if isinstance(item, Exception):
            raise item
        return item

    return llm


# (질문 목록, 가짜 모델, 턴당 최대 스텝)
SCENARIOS = {
    "memory": (
        ["17*23은 얼마야?", "거기에 2를 곱해 줘"],
        scripted(
            ask("c1", "calculate", expression="17*23"),
            final("391입니다."),
            ask("c2", "calculate", expression="391*2"),
            final("782입니다."),  # 두 번째 턴이 첫 턴의 391을 이력에서 알고 있어야 나올 수 있는 답
        ),
        None,
    ),
    "fail": (
        ["1+1은?", "다시 한 번 1+1은?"],
        None,  # 아래에서 턴별로 다른 모델을 쓴다(첫 턴 중간 실패 → 두 번째 턴 정상)
        None,
    ),
    "maxsteps": (
        ["시간을 계속 알려 줘", "계속해"],
        scripted(
            ask("c1", "get_current_time"),
            ask("c2", "get_current_time"),
            final("그만 알려 드립니다."),
        ),
        2,
    ),
}


def show_history(session, title):
    print(f"\n[{title}] 세션 이력 {len(session.history)}개")
    for i, m in enumerate(session.history, 1):
        print(f"   {i:2d}. {describe(m)}")


def main():
    parser = argparse.ArgumentParser(description="여러 턴 대화의 단계별 출력 데모")
    parser.add_argument("questions", nargs="*", help="실제 모델에 차례로 보낼 질문들(생략하면 기본 두 질문)")
    parser.add_argument("--scenario", choices=SCENARIOS, help="모델 없이 시나리오 실행")
    parser.add_argument("--provider", help="LLM 제공자(openai, anthropic). 생략하면 .env의 LLM_PROVIDER")
    parser.add_argument("--max-steps", type=int, help="한 턴에서 모델을 호출하는 횟수 상한")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.ERROR)

    if args.scenario:
        questions, model, scenario_steps = SCENARIOS[args.scenario]
        mode = f"가짜 모델(시나리오: {args.scenario})"
    else:
        questions = args.questions or ["17*23은 얼마야?", "거기에 2를 곱해 줘"]
        model, scenario_steps = partial(call_llm, provider=args.provider), None
        mode = f"실제 모델(제공자: {args.provider or '.env의 LLM_PROVIDER'})"
    max_steps = args.max_steps or scenario_steps

    # fail 시나리오는 첫 턴에서 도중에 실패하고, 두 번째 턴은 정상으로 끝나도록 턴별 모델을 따로 만든다.
    turn_models = None
    if args.scenario == "fail":
        turn_models = [failing_after_tool(), scripted(ask("c9", "calculate", expression="1+1"), final("2입니다."))]

    holder = {"llm": model}
    tracer = TurnTracer(lambda history, tools, system_prompt: holder["llm"](history, tools, system_prompt))
    session = Session([tracer.wrap_tool(t) for t in BUILTIN_TOOLS], max_steps=max_steps, llm=tracer.llm)

    banner("시작")
    print(f"모드    : {mode}")
    print(f"질문 수 : {len(questions)}턴")
    print(f"Tool    : {', '.join(t.name for t in session.tools)}")
    print(f"턴당 최대 스텝: {max_steps or '(.env의 AGENT_MAX_STEPS 또는 기본 10)'}")

    for turn, question in enumerate(questions, 1):
        if turn_models:
            holder["llm"] = turn_models[turn - 1]
        tracer.begin_turn()
        tracer._seen = len(session.history)  # 이번 턴 시작 전까지의 이력은 "이미 본 것"으로 취급한다.
        banner(f"턴 {turn}: {question}")
        print(f"턴 시작 전 세션 이력: {len(session.history)}개 (앞 턴의 대화를 이 이력으로 모델에 전달한다)")
        try:
            result = session.ask(question)
        except Exception as e:
            print(f"\n✗ 이 턴은 실패했다: {type(e).__name__}: {e}")
            print("   → 세션은 실패한 턴의 변경을 버리고, 턴 시작 전 이력 그대로 유지한다.")
            show_history(session, f"턴 {turn} 실패 뒤")
            continue
        print(f"\n✓ 턴 {turn} 종료: {result.status}, 스텝 {result.steps}회, 답변: {result.answer or '(없음)'}")
        show_history(session, f"턴 {turn} 종료 뒤")

    banner("끝")
    print(f"최종 세션 이력: {len(session.history)}개")


if __name__ == "__main__":
    main()
