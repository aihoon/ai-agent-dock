"""Hook(C11 D19~D21): 사용자가 hooks.json으로 설치해 에이전트 루프에 더하는 확장. Safety(항상 거는 고정 검사)와 다르다.

자리(이벤트)는 3개다: before_tool(실행 직전, 승인보다 먼저), after_tool(실행 직후), after_answer(최종 답을 이력에 넣기 전).
hooks.json의 Hook은 외부 명령이다. 표준 입력으로 JSON을 받고, 표준 출력으로 JSON을 돌려준다.
  받는 것: {"event", "tool", "arguments", "result", "answer"}
  돌려주는 것: {"block": "이유"}(before_tool), {"result": "새 결과"}(after_tool), {"answer": "새 답"}(after_answer), 빈 출력은 통과
명령이 실패하거나 시간을 넘기면 before_tool은 막고(검사를 못 했으면 안전한 쪽), 나머지는 원래 값을 두고 경고한다.
코드에서 함수로 붙일 수도 있다(HookSet의 목록에 넣는다). 이 모듈은 runtime을 import하지 않는다.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from ..llm import Tool, ToolCall

EVENTS = ("before_tool", "after_tool", "after_answer")
DEFAULT_TIMEOUT = 10.0  # HOOK_TIMEOUT

BeforeTool = Callable[[ToolCall, Tool], "str | None"]  # 막을 이유를 돌려주면 막는다.
AfterTool = Callable[[ToolCall, Tool, str], str]  # 모델에게 넘길 결과를 돌려준다.
AfterAnswer = Callable[[str], str]  # 이력에 넣을 답을 돌려준다.


@dataclass(frozen=True)
class HookSpec:
    """hooks.json의 Hook 하나."""

    event: str
    command: list[str]
    match: str = "*"  # Tool 이름 패턴(after_answer에는 쓰지 않는다)


@dataclass
class HookSet:
    """이벤트별 Hook 목록. 루프는 등록 순서대로 부르기만 한다."""

    before_tool: list[BeforeTool] = field(default_factory=list)
    after_tool: list[AfterTool] = field(default_factory=list)
    after_answer: list[AfterAnswer] = field(default_factory=list)
    specs: list[HookSpec] = field(default_factory=list)  # hooks.json에서 읽은 것(/hooks가 보여 준다)


def load_hooks(
    path: str | Path,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    record: Callable[..., None] | None = None,
    warn: Callable[[str], None] = print,
) -> HookSet:
    """hooks.json을 읽어 HookSet을 만든다. 명령은 설정 파일이 있는 폴더에서 실행한다."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    hooks = HookSet()
    for event, entries in data.get("hooks", {}).items():
        if event not in EVENTS:
            raise ValueError(f"없는 Hook 이벤트입니다: {event} (쓸 수 있는 이벤트: {', '.join(EVENTS)})")
        for entry in entries:
            command = entry.get("command")
            if not command or not isinstance(command, list):
                raise ValueError(f"{event}: command는 명령과 인자의 목록이어야 합니다: {command}")
            spec = HookSpec(event=event, command=[str(c) for c in command], match=entry.get("match", "*"))
            runner = _Runner(spec, path.parent, timeout, record or (lambda **fields: None), warn)
            hooks.specs.append(spec)
            getattr(hooks, event).append(getattr(runner, event))
    return hooks


@dataclass
class _Runner:
    spec: HookSpec
    cwd: Path
    timeout: float
    record: Callable[..., None]
    warn: Callable[[str], None]

    def _run(self, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        """명령을 실행해 (출력 JSON, 오류)를 돌려준다."""
        try:
            done = subprocess.run(
                self.spec.command, input=json.dumps(payload, ensure_ascii=False), capture_output=True,
                text=True, timeout=self.timeout, cwd=self.cwd,
            )
            if done.returncode != 0:
                return None, f"종료 코드 {done.returncode}: {done.stderr.strip()[:200]}"
            out = done.stdout.strip()
            return (json.loads(out) if out else {}), None
        except subprocess.TimeoutExpired:
            return None, f"{self.timeout:g}초 안에 끝나지 않았다"
        except (OSError, json.JSONDecodeError) as e:
            return None, f"{type(e).__name__}: {e}"

    def _call(self, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        output, error = self._run(payload)
        self.record(event=self.spec.event, command=" ".join(self.spec.command), tool=payload.get("tool"), output=output, error=error)
        return output, error

    def before_tool(self, call: ToolCall, tool: Tool) -> str | None:
        if not fnmatch(call.name, self.spec.match):
            return None
        output, error = self._call({"event": "before_tool", "tool": call.name, "arguments": call.arguments})
        if error:
            return f"Hook({self.spec.command[-1]})을 실행하지 못해 막았다: {error}"
        return output.get("block") or None

    def after_tool(self, call: ToolCall, tool: Tool, content: str) -> str:
        if not fnmatch(call.name, self.spec.match):
            return content
        output, error = self._call({"event": "after_tool", "tool": call.name, "arguments": call.arguments, "result": content})
        if error:
            self.warn(f"⚠ Hook({self.spec.command[-1]}) 실패, 결과를 그대로 둡니다: {error}")
            return content
        return str(output.get("result", content))

    def after_answer(self, answer: str) -> str:
        output, error = self._call({"event": "after_answer", "answer": answer})
        if error:
            self.warn(f"⚠ Hook({self.spec.command[-1]}) 실패, 답을 그대로 둡니다: {error}")
            return answer
        return str(output.get("answer", answer))
