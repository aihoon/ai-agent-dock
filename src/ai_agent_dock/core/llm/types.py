"""모델 호출에 쓰는, 제공자에 독립적인 공통 형식."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """모델이 요청한 Tool 호출 하나."""

    id: str
    name: str
    arguments: dict[str, Any]
    # 어댑터가 인자를 해석하지 못했거나 응답이 잘렸을 때 사유를 담는다. 루프는 이 Tool을 실행하지 않고 오류로 돌려준다.
    error: str | None = None


@dataclass
class LLMResponse:
    """call_llm이 돌려주는 모델 응답."""

    text: str | None  # 최종 답변. Tool 호출만 있는 응답이면 없을 수 있다.
    tool_calls: list[ToolCall] = field(default_factory=list)
    # 제공자 원본 항목. OpenAI의 추론 항목처럼 다음 호출에 그대로 돌려줘야 하는 데이터를 보관한다.
    raw: list[Any] = field(default_factory=list)


@dataclass
class Message:
    """대화 이력의 한 항목. role은 "user", "assistant", "tool" 중 하나."""

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant 전용
    raw: list[Any] = field(default_factory=list)  # assistant 전용, LLMResponse.raw를 그대로 보관
    tool_call_id: str | None = None  # tool 전용, 어떤 ToolCall의 결과인지 짝을 맞추는 값
    is_error: bool = False  # tool 전용, 실행이 실패했는지


@dataclass
class Tool:
    """모델에 알려 주고, 루프가 실행하는 Tool. 레지스트리는 C2에서 만든다."""

    name: str
    description: str
    parameters: dict[str, Any]  # 인자의 JSON Schema
    func: Callable[..., Any]
