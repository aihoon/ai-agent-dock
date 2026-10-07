"""Anthropic(Claude) 어댑터: 공통 형식과 Messages API 형식을 서로 변환한다.

조절 가능한 환경 변수: ANTHROPIC_API_KEY, ANTHROPIC_MODEL, ANTHROPIC_MAX_TOKENS
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any

from anthropic import Anthropic  # 절대 import이므로 이 파일 이름(anthropic.py)과 충돌하지 않는다.

from ai_agent_dock.env import env_int, env_str

from ..types import LLMResponse, Message, Tool, ToolCall

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_MAX_TOKENS = 16000  # 스트리밍을 쓰지 않는 요청에 권장되는 기본값


@cache
def _client() -> Anthropic:
    # ANTHROPIC_API_KEY 등 인증 정보는 SDK가 환경에서 읽는다. 연결을 재사용하려고 한 번만 만든다.
    return Anthropic()


def to_messages(history: list[Message]) -> list[dict[str, Any]]:
    """공통 대화 이력을 Messages API의 messages 목록으로 바꾼다."""
    out: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []  # 연속된 tool 메시지는 user 메시지 하나로 묶어야 한다.

    def flush() -> None:
        if results:
            out.append({"role": "user", "content": list(results)})
            results.clear()

    for m in history:
        if m.role == "tool":
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id,
                "content": m.content or "",
            }
            if m.is_error:
                block["is_error"] = True
            results.append(block)
            continue
        flush()
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            if m.raw:
                # 응답 블록(사고 블록 포함)을 그대로 돌려줘야 모델이 이전 판단을 이어 간다.
                content: list[Any] = list(m.raw)
            else:
                content = []
                if m.content:
                    content.append({"type": "text", "text": m.content})
                for c in m.tool_calls:
                    content.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments})
            out.append({"role": "assistant", "content": content})
        else:
            raise ValueError(f"알 수 없는 role: {m.role!r}")
    flush()
    return out


def to_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    return [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]


def parse(response: Any) -> LLMResponse:
    """Messages API 응답을 공통 형식으로 바꾼다."""
    if response.stop_reason == "refusal":
        raise RuntimeError("모델이 응답을 거부했다(stop_reason=refusal)")
    # 출력이 잘리면 Tool 인자가 불완전할 수 있다. 실행하지 않고 오류로 돌려 모델이 다시 판단하게 한다.
    truncated = None
    if response.stop_reason == "max_tokens":
        truncated = "응답이 max_tokens에서 잘려 Tool 인자가 불완전하다"
    text = "".join(b.text for b in response.content if b.type == "text") or None
    calls = [
        ToolCall(id=b.id, name=b.name, arguments=dict(b.input), error=truncated)
        for b in response.content
        if b.type == "tool_use"
    ]
    return LLMResponse(text=text, tool_calls=calls, raw=list(response.content))


def build_request(history: list[Message], tools: list[Tool], system_prompt: str | None = None) -> dict[str, Any]:
    """환경 설정과 이력으로 messages.create 인자를 만든다. 네트워크 호출은 하지 않는다."""
    kwargs: dict[str, Any] = {
        "model": env_str("ANTHROPIC_MODEL", DEFAULT_MODEL),
        "max_tokens": env_int("ANTHROPIC_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        "messages": to_messages(history),
    }
    if tools:
        kwargs["tools"] = to_tools(tools)
    if system_prompt:
        kwargs["system"] = system_prompt
    return kwargs


def call(
    history: list[Message],
    tools: list[Tool],
    system_prompt: str | None = None,
    on_text_delta: Callable[[str], None] | None = None,
) -> LLMResponse:
    """모델을 한 번 호출한다. on_text_delta를 주면 스트리밍으로 받으며 글자 조각이 올 때마다 호출한다."""
    kwargs = build_request(history, tools, system_prompt)
    if on_text_delta is None:
        return parse(_client().messages.create(**kwargs))
    with _client().messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            on_text_delta(text)
        # 글자 조각만 우리가 전달하고, 최종 응답은 SDK가 조립한 결과를 기존 parse()로 바꾼다(Tool 인자 포함).
        return parse(stream.get_final_message())
