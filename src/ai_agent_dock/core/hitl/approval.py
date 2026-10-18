"""사용자 승인: 높음 등급 Tool을 실행하기 전에 번호 메뉴로 묻고, 결정을 돌려준다.

승인 선택지는 코드가 정한다. 모델이 만든 선택지를 쓰지 않는다.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..llm import ToolCall

PREVIEW_CHARS = 60  # 승인 메뉴에 보여 줄 인자 미리보기 글자 수


@dataclass(frozen=True)
class Approval:
    kind: str  # "approve"(승인하고 실행), "deny"(거부), "revise"(다르게 요청)
    reason: str = ""

    @classmethod
    def approve(cls) -> Approval:
        return cls("approve")

    @classmethod
    def deny(cls, reason: str = "") -> Approval:
        return cls("deny", reason)

    @classmethod
    def revise(cls, reason: str) -> Approval:
        return cls("revise", reason)


def ask_in_terminal(
    call: ToolCall,
    *,
    interactive: bool,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> Approval:
    """번호 메뉴로 승인을 묻는다. 대화형 입력이 없으면 승인을 받을 수 없으므로 거부한다(D6)."""
    if not interactive:
        return Approval.deny("승인을 받을 수 없는 실행이라 거부했다")
    print_fn(f"[승인 요청] {call.name}")
    print_fn(f"  인자: {_preview(call.arguments)}")
    print_fn("  1) 승인하고 실행")
    print_fn("  2) 거부")
    print_fn("  3) 다르게 요청(이유 입력)")
    while True:
        choice = input_fn("> ").strip()
        if choice == "1":
            return Approval.approve()
        if choice == "2":
            return Approval.deny()
        if choice == "3":
            return Approval.revise(input_fn("다르게 요청할 내용: ").strip())
        print_fn("1, 2, 3 중에서 고르세요.")


def _preview(arguments: dict[str, Any]) -> str:
    text = json.dumps(arguments, ensure_ascii=False)
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "…"
