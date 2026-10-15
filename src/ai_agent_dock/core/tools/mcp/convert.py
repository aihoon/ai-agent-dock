"""MCP의 이름과 결과를 우리 Tool 형식으로 바꾸는 규칙(네트워크 없이 시험할 수 있는 순수 함수)."""

from __future__ import annotations

import json
import re
from typing import Any

from ..errors import ToolError

MAX_NAME_LENGTH = 128  # Claude API의 Tool 이름 규칙: ^[a-zA-Z0-9_-]{1,128}$
_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def model_tool_name(server: str, tool: str) -> str:
    """모델에 알려 줄 이름: `서버이름__Tool이름`. 허용되지 않는 문자(MCP가 허용하는 `.` 등)는 `_`로 바꾼다."""
    name = _INVALID_NAME_CHARS.sub("_", f"{server}__{tool}")
    if len(name) > MAX_NAME_LENGTH:
        raise ValueError(f"Tool 이름이 {MAX_NAME_LENGTH}자를 넘는다: {name!r}")
    return name


def result_to_text(result: Any) -> str:
    """CallToolResult를 모델에 줄 문자열로 바꾼다. 결과가 오류(is_error)면 ToolError를 던져 루프가 오류 관찰로 처리하게 한다.

    텍스트 블록은 줄바꿈으로 잇고, 텍스트가 아닌 블록은 `[image: image/png]` 같은 자리표시로 바꾼다
    (이미지 등을 모델에 그대로 넘기는 것은 이번 범위가 아니다). 텍스트가 없고 structured_content만 있으면 그 JSON을 쓴다.
    """
    parts = [_block_text(block) for block in result.content]
    has_text = any(block.type == "text" for block in result.content)
    if not has_text and result.structured_content is not None:
        parts = [json.dumps(result.structured_content, ensure_ascii=False)]
    text = "\n".join(parts) if parts else "(빈 결과)"
    if result.is_error:
        raise ToolError(text)
    return text


def _block_text(block: Any) -> str:
    kind = block.type
    if kind == "text":
        return block.text
    if kind in ("image", "audio"):
        return f"[{kind}: {block.mime_type}]"
    if kind == "resource_link":
        return f"[resource_link: {block.uri}]"
    if kind == "resource":  # 임베디드 리소스: 텍스트면 그대로, 바이너리면 자리표시
        resource = block.resource
        text = getattr(resource, "text", None)
        return text if text is not None else f"[resource: {resource.uri}, {getattr(resource, 'mime_type', None)}]"
    return f"[{kind}]"
