"""모델 호출에 쓰는, 제공자에 독립적인 공통 형식."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass(frozen=True)
class LLMConfig:
    """모델 호출의 정적 설정. 시작할 때 한 번 만들고 모든 호출에 같이 쓴다. 환경 변수는 어댑터의 config_from_env가 읽는다."""

    provider: str  # 어떤 어댑터를 쓸지 정한다(예: "openai", "anthropic")
    model: str
    max_output_tokens: int | None  # None이면 제공자 기본값을 쓴다


@dataclass
class Tool:
    """모델에 알려 주고, 루프가 실행하는 Tool. 여러 Tool을 모아 관리하는 곳은 core/tools/registry.py의 ToolRegistry다."""

    name: str
    description: str
    parameters: dict[str, Any]  # 인자의 JSON Schema
    func: Callable[..., Any]
    read_only: bool = False  # 읽기 전용 표시(MCP는 서버가 준 readOnlyHint). 기본은 False.
    trusted: bool = False  # 이 툴을 제공하는 서버를 신뢰하는지(mcp.json의 trusted, 빌트인은 True로 명시). 기본은 믿지 않는다. 자동 실행은 read_only와 trusted가 모두 True일 때만 한다.


@dataclass
class LLMRequest:
    """모델을 한 번 호출하는 전체 입력. 정적 설정(config)과 호출마다 달라지는 값(이력, Tool 목록, 지침)을 묶는다."""

    config: LLMConfig
    messages: list[Message]
    system_prompt: str | None = None
    tools: list[Tool] = field(default_factory=list)


@dataclass
class UserMessage:
    """사용자가 보낸 질문. 우리가 만들어 이력에 넣는 인풋이다."""

    role: ClassVar[str] = "user"
    content: str


@dataclass
class AssistantMessage:
    """모델이 낸 응답(텍스트 답이나 Tool 요청). LLMResponse를 이력에 저장한 아웃풋이다."""

    role: ClassVar[str] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: list[Any] = field(default_factory=list)  # 제공자 원본 항목. 다음 호출에 그대로 돌려준다.


@dataclass
class ToolMessage:
    """우리가 Tool을 실행한 결과. 어떤 ToolCall의 결과인지 tool_call_id로 짝을 맞춘다."""

    role: ClassVar[str] = "tool"
    tool_call_id: str
    content: str
    is_error: bool = False  # 실행이 실패했는지


# 대화 이력의 한 항목. 이력 리스트에는 세 종류가 발생한 순서대로 섞여 들어간다.
Message = UserMessage | AssistantMessage | ToolMessage

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
    # 토큰 수. 제공자 응답의 값을 쓰고, 없으면 tiktoken으로 계산한다(call_llm이 채운다). 출처는 provider 또는 tiktoken이다.
    input_tokens: int | None = None
    output_tokens: int | None = None
    input_tokens_source: str | None = None
    output_tokens_source: str | None = None