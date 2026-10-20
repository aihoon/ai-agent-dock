"""안전 정책(C11 D4, D8, D25): safety.json으로 Tool을 막거나(deny), 승인을 받게 하거나(ask), 경로 인자를 패턴으로 제한한다.

정책은 코드 기본값(read_only, trusted)보다 더 엄격하게만 바꿀 수 있다. allow처럼 느슨하게 바꾸는 값은 오류다.
형식: {"tools": {"files__move_file": "deny", "files__write_file": {"mode": "ask", "paths": ["*.md"]}}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from fnmatch import fnmatch
from pathlib import Path

from ..llm import Tool, ToolCall

MODES = ("deny", "ask")
PATH_ARGS = ("path", "source", "destination", "paths")  # Filesystem MCP 서버의 경로 인자(C11 V3)


@dataclass(frozen=True)
class Rule:
    mode: str = "ask"
    paths: list[str] = field(default_factory=list)  # 비어 있으면 경로를 보지 않는다


@dataclass(frozen=True)
class SafetyPolicy:
    rules: dict[str, Rule] = field(default_factory=dict)

    def apply(self, tools: list[Tool]) -> list[Tool]:
        """정책에 적힌 Tool이 모두 있는지 보고, 적힌 Tool은 승인을 받도록 신뢰 표시를 끈다."""
        names = {t.name for t in tools}
        missing = sorted(set(self.rules) - names)
        if missing:
            raise ValueError(f"정책에 없는 Tool 이름이 있습니다: {', '.join(missing)}")
        return [replace(t, trusted=False) if t.name in self.rules else t for t in tools]

    def check(self, call: ToolCall, tool: Tool) -> str | None:
        """Tool 실행 전 검사. 막을 이유를 돌려주고, 통과면 None."""
        rule = self.rules.get(call.name)
        if rule is None:
            return None
        if rule.mode == "deny":
            return f"정책이 {call.name}을 막았다."
        for value in _path_values(call):
            if not any(fnmatch(value, pattern) for pattern in rule.paths):
                return f"경로 {value}는 허용된 패턴({', '.join(rule.paths)})에 맞지 않는다."
        return None


def _path_values(call: ToolCall) -> list[str]:
    values = []
    for name in PATH_ARGS:
        value = call.arguments.get(name)
        values += value if isinstance(value, list) else [value] if isinstance(value, str) else []
    return values


def load_safety_policy(path: str | Path) -> SafetyPolicy:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rules = {}
    for name, raw in data.get("tools", {}).items():
        rule = Rule(mode=raw) if isinstance(raw, str) else Rule(mode=raw.get("mode", "ask"), paths=list(raw.get("paths", [])))
        if rule.mode not in MODES:
            raise ValueError(f"{name}: mode는 {', '.join(MODES)} 중 하나입니다(정책은 더 엄격하게만 바꿀 수 있습니다): {rule.mode}")
        rules[name] = rule
    return SafetyPolicy(rules)
