"""토큰 수 채우기: 제공자 응답에 값이 없으면 tiktoken으로 계산하고, 값에 출처 태그(provider 또는 tiktoken)를 붙인다.

OpenAI는 tiktoken이 쓰는 인코딩 규칙과 같아서 값이 거의 정확하다. Anthropic은 공개된 토크나이저가 없어서 같은 인코딩으로
근사한다. 그래서 Anthropic의 계산값도 tiktoken 태그로 구분해 둔다.
"""

from __future__ import annotations

import json
from functools import cache

import tiktoken

from .types import AssistantMessage, LLMRequest, LLMResponse

PROVIDER = "provider"  # 제공자 응답의 값을 그대로 썼다.
TIKTOKEN = "tiktoken"  # tiktoken으로 계산했다.


@cache
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("o200k_base")


def count(text: str) -> int:
    return len(_encoding().encode(text))


def _request_text(request: LLMRequest) -> str:
    """입력 토큰을 셀 대상: 시스템 프롬프트, 이력(툴 요청 포함), Tool 스키마."""
    parts = [request.system_prompt or ""]
    for message in request.messages:
        if isinstance(message, AssistantMessage):
            parts.append(message.content or "")
            parts += [json.dumps({"name": c.name, "arguments": c.arguments}, ensure_ascii=False) for c in message.tool_calls]
        else:
            parts.append(message.content)
    parts += [json.dumps({"name": t.name, "description": t.description, "parameters": t.parameters}, ensure_ascii=False) for t in request.tools]
    return "\n".join(parts)


def _response_text(response: LLMResponse) -> str:
    """출력 토큰을 셀 대상: 답 텍스트와 툴 요청."""
    parts = [response.text or ""]
    parts += [json.dumps({"name": c.name, "arguments": c.arguments}, ensure_ascii=False) for c in response.tool_calls]
    return "\n".join(parts)


def fill_tokens(request: LLMRequest, response: LLMResponse) -> LLMResponse:
    """토큰 수가 비어 있으면 채우고, 모든 값에 출처 태그를 붙인다. 응답 객체를 고쳐서 돌려준다."""
    if response.input_tokens is None:
        response.input_tokens = count(_request_text(request))
        response.input_tokens_source = TIKTOKEN
    elif response.input_tokens_source is None:
        response.input_tokens_source = PROVIDER
    if response.output_tokens is None:
        response.output_tokens = count(_response_text(response))
        response.output_tokens_source = TIKTOKEN
    elif response.output_tokens_source is None:
        response.output_tokens_source = PROVIDER
    return response
