"""감사 로그: Tool 실행 시도와 결정을 JSON Lines 파일에 한 줄씩 남긴다.

인자 값은 ARG_CHARS 글자까지만 남긴다. 비밀값이 그대로 새지 않게 하려는 최소한의 조치다(전체 마스킹은 C11).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ARG_CHARS = 20


@dataclass
class AuditLog:
    path: Path
    turn: int = 0  # 현재 턴 번호. 채팅이 턴을 시작할 때 begin_turn으로 올린다.

    def begin_turn(self) -> None:
        self.turn += 1

    def record(
        self,
        *,
        tool: str,
        risk: str,
        decision: str,
        arguments: dict[str, Any],
        result_chars: int,
        is_error: bool,
    ) -> None:
        line = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "turn": self.turn,
            "tool": tool,
            "risk": risk,
            "decision": decision,  # "auto"(자동 실행), "approve", "deny", "revise"
            "args": {key: str(value)[:ARG_CHARS] for key, value in arguments.items()},
            "result_chars": result_chars,
            "is_error": is_error,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)  # 로그 폴더가 없으면 만든다.
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
