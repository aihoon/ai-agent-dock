"""OpenAI 어댑터: 공통 형식과 Responses API 형식을 서로 변환한다.

조절 가능한 환경 변수: OPENAI_API_KEY, OPENAI_MODEL, OPENAI_MAX_OUTPUT_TOKENS
"""

from __future__ import annotations

import json
from functools import cache
from typing import Any

from openai import OpenAI  # 절대 import이므로 이 파일 이름(openai.py)과 충돌하지 않는다.

from ai_agent_dock.env import env_int, env_str

from ..types import LLMResponse, Message, Tool, ToolCall

DEFAULT_MODEL = "gpt-5-mini"


@cache
def _client() -> OpenAI:
    # OPENAI_API_KEY는 SDK가 환경에서 읽는다. 연결을 재사용하려고 한 번만 만든다.
    return OpenAI()


def to_input(history: list[Message]) -> list[Any]:
    """공통 대화 이력을 Responses API의 input 항목 목록으로 바꾼다."""
    items: list[Any] = []
    for m in history:
        if m.role == "user":
            items.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            if m.raw:
                # 원본 항목(추론 항목 포함)을 그대로 돌려줘야 모델이 이전 판단을 이어 간다.
                items.extend(m.raw)
            else:
                if m.content:
                    items.append({"role": "assistant", "content": m.content})
                for c in m.tool_calls:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": c.id,
                            "name": c.name,
                            "arguments": json.dumps(c.arguments, ensure_ascii=False),
                        }
                    )
        elif m.role == "tool":
            items.append(
                {"type": "function_call_output", "call_id": m.tool_call_id, "output": m.content or ""}
            )
        else:
            raise ValueError(f"알 수 없는 role: {m.role!r}")
    return items


def to_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    # strict 기본값에 기대지 않고 명시한다. strict=True는 스키마 제약이 까다롭다.
    return [
        {
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
            "strict": False,
        }
        for t in tools
    ]


def parse(response: Any) -> LLMResponse:
    """Responses API 응답을 공통 형식으로 바꾼다."""
    # 출력 한도에서 잘린 응답이면 Tool 인자가 불완전할 수 있다.
    truncated = None
    if getattr(response, "status", None) == "incomplete":
        truncated = "응답이 출력 토큰 한도에서 잘려 Tool 인자가 불완전하다"
    calls: list[ToolCall] = []
    for item in response.output:
        if item.type != "function_call":
            continue
        error = truncated
        try:
            args = json.loads(item.arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("인자가 JSON 객체가 아님")
        except ValueError as e:  # JSONDecodeError도 ValueError의 하위 클래스다.
            args, error = {}, error or f"잘못된 Tool 인자: {e}"
        calls.append(ToolCall(id=item.call_id, name=item.name, arguments=args, error=error))
    return LLMResponse(
        text=response.output_text or None,
        tool_calls=calls,
        raw=list(response.output),
    )


def build_request(history: list[Message], tools: list[Tool], system: str | None = None) -> dict[str, Any]:
    """환경 설정과 이력으로 responses.create 인자를 만든다. 네트워크 호출은 하지 않는다."""
    kwargs: dict[str, Any] = {"model": env_str("OPENAI_MODEL", DEFAULT_MODEL), "input": to_input(history)}
    max_output = env_int("OPENAI_MAX_OUTPUT_TOKENS")  # 없으면 API 기본값을 쓴다.
    if max_output:
        kwargs["max_output_tokens"] = max_output
    if tools:
        kwargs["tools"] = to_tools(tools)
    if system:
        kwargs["instructions"] = system
    return kwargs


def call(history: list[Message], tools: list[Tool], system: str | None = None) -> LLMResponse:
    return parse(_client().responses.create(**build_request(history, tools, system)))
