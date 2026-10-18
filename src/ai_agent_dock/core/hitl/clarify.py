"""되묻기 Tool(ask_user): 모델이 정보가 부족하다고 판단하면 사용자에게 묻고 답을 돌려준다. 승인 대상이 아니다."""

from __future__ import annotations

from collections.abc import Callable

from ..llm import Tool


def make_ask_user_tool(input_fn: Callable[[str], str] = input) -> Tool:
    def ask_user(question: str) -> str:
        return input_fn(f"[질문] {question}\n> ").strip()

    return Tool(
        name="ask_user",
        description="정보가 부족해서 추측할 수 없을 때 사용자에게 질문하고 답을 받는다.",
        parameters={
            "type": "object",
            "properties": {"question": {"type": "string", "description": "사용자에게 할 질문"}},
            "required": ["question"],
        },
        read_only=True,
        trusted=True,
        func=ask_user,
    )
