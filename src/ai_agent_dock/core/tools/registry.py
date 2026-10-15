"""Tool 레지스트리: 내장 Tool과 MCP Tool을 한곳에 모아 관리한다.

등록할 때 두 가지를 보장한다.
  1. 이름이 겹치면 오류를 낸다(루프의 이름 표가 조용히 덮어쓰는 것을 막는다).
  2. 각 Tool의 func를 감싸서, 호출 전에 인자를 parameters(JSON Schema)로 검증하고 호출 뒤에 결과 길이를 제한한다.
루프(run_agent)는 고치지 않는다. Session에는 tools()가 돌려주는 목록을 넘긴다.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from jsonschema.validators import validator_for

from ..llm import Tool
from .errors import ToolArgumentError

DEFAULT_MAX_RESULT_CHARS = 20000  # 실제 값은 호출하는 쪽이 TOOL_RESULT_MAX_CHARS로 읽어 넘긴다.


class ToolRegistry:
    def __init__(self, *, max_result_chars: int = DEFAULT_MAX_RESULT_CHARS):
        self._max_result_chars = max_result_chars
        self._tools: dict[str, Tool] = {}
        self._sources: dict[str, str] = {}

    # 등록 시점에 이름 중복과 스키마를 검사해, 잘못된 Tool은 대화 중이 아니라 시작할 때 드러나게 한다.
    def add(self, tool: Tool, source: str = "builtin") -> None:
        """Tool 하나를 등록한다. source는 출처 표시("builtin", "mcp:서버이름")이며 오류 메시지와 /tools에 쓴다."""
        if tool.name in self._tools:
            raise ValueError(f"Tool 이름이 겹친다: {tool.name!r} ({self._sources[tool.name]} 와 {source})")
        # 스키마가 JSON Schema 2020-12가 기본이라는 MCP 스펙에 맞춰, $schema가 없으면 2020-12로 검증한다.
        validator_class = validator_for(tool.parameters, default=Draft202012Validator)
        try:
            validator_class.check_schema(tool.parameters)
        except SchemaError as e:
            raise ValueError(f"Tool {tool.name!r}({source})의 parameters가 올바른 JSON Schema가 아니다: {e.message}") from None
        guarded = _guard(tool, validator_class(tool.parameters), self._max_result_chars)
        self._tools[tool.name] = replace(tool, func=guarded)
        self._sources[tool.name] = source

    # 기본 Tool 묶음이나 MCP 서버 하나가 내놓은 Tool 목록을 한 번에 넣는다. 출처를 같이 받아 서버별로 구분한다.
    def add_all(self, tools: Iterable[Tool], source: str = "builtin") -> None:
        for tool in tools:
            self.add(tool, source)

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    # /tools 목록과 오류 메시지에서 "이 Tool이 어디서 왔는지"를 보여 주려고 둔다.
    def source_of(self, name: str) -> str:
        return self._sources[name]

    def __len__(self) -> int:
        return len(self._tools)


# 루프가 부르는 func를 감싼다. 인자 검증이 실패하면 원래 func를 호출하지 않고 ToolArgumentError를 던진다.
def _guard(tool: Tool, validator: Any, max_chars: int):
    def run(**arguments: Any) -> str:
        errors = sorted(validator.iter_errors(arguments), key=lambda e: [str(p) for p in e.path])
        if errors:
            details = "; ".join(_describe(e) for e in errors)
            raise ToolArgumentError(f"{tool.name}의 인자가 형식에 맞지 않는다: {details}")
        return _truncate(str(tool.func(**arguments)), max_chars)

    return run


# jsonschema의 영문 메시지 대신 "어느 인자가 왜 틀렸는지"만 짧게 모아, 모델이 고칠 지점을 알 수 있게 한다.
def _describe(error: ValidationError) -> str:
    where = ".".join(str(p) for p in error.path) or "(인자 전체)"
    return f"{where}: {error.message}"


# 큰 결과가 대화 이력과 컨텍스트를 채우지 않도록 자르고, 잘렸다는 사실을 모델에 알린다.
def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n\n[결과가 길어 잘렸습니다: 전체 {len(text)}자 중 앞 {limit}자만 전달함]"
