"""되묻기 Tool(ask_user): 모델이 정보가 부족하다고 판단하면 사용자에게 묻고 답을 돌려준다. 승인 대상이 아니다."""

from __future__ import annotations

from collections.abc import Callable

from ..llm import Tool

CANCELLED = "사용자가 질문에 답하지 않고 취소했다. 다시 묻지 말고, 지금 아는 것으로 짧게 답을 끝낸다."


def make_ask_user_tool(input_fn: Callable[[str], str] = input) -> Tool:
    def ask_user(question: str) -> str:
        answer = input_fn(f"[질문] {question}\n> ").strip()
        if answer.startswith("/"):  # 슬래시 명령은 답이 아니라 취소로 본다(C10 D25).
            print(f"질문을 취소했습니다. 명령({answer.split()[0]})은 답이 끝난 뒤 다시 입력하세요.")
            return CANCELLED
        return answer

    return Tool(
        name="ask_user",
        description=(
            "정보가 부족해서 추측할 수 없을 때 사용자에게 질문하고 답을 받는다. "
            "무엇을, 어디에, 어떤 내용으로 할지 같은 필요한 정보가 빠졌으면 답 문장으로 묻고 끝내지 말고 이 툴로 묻는다. "
            "질문은 한 번에 하나만 짧게 한다."
        ),
        parameters={
            "type": "object",
            "properties": {"question": {"type": "string", "description": "사용자에게 할 질문"}},
            "required": ["question"],
        },
        read_only=True,
        trusted=True,
        func=ask_user,
    )
