"""추적 기록: 턴, 스텝, LLM 호출, Tool 실행을 세션별 파일(logs/sessions/세션번호.jsonl)에 한 줄씩 덧붙인다.

세션 번호는 세션을 시작할 때 한 번 만든다. 모든 줄에 세션 번호를 넣고, 시각은 밀리초로 찍는다.
내용(질문, 답, Tool 인자, Tool 결과)을 그대로 남긴다. 민감한 정보가 섞일 수 있으므로 이 폴더는 git에 올리지 않는다.
이 모듈은 runtime을 import하지 않는다(의존 방향: runtime이 이 모듈을 쓴다).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..safety.mask import mask_value


def new_session_id(prefix: str = "") -> str:
    """시각 기반 세션 번호. 예: 20261018-150405-a1b2 (평가 실행이면 prefix로 eval-을 붙인다)."""
    return f"{prefix}{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


@dataclass
class Tracer:
    directory: Path | None  # 세션 파일을 둘 폴더(logs/sessions). None이면 기록하지 않는다.
    session_id: str = field(default_factory=new_session_id)
    turn: int = 0
    trace_id: str = ""

    @property
    def path(self) -> Path | None:
        return None if self.directory is None else self.directory / f"{self.session_id}.jsonl"

    def begin_turn(self) -> None:
        self.turn += 1
        self.trace_id = uuid.uuid4().hex[:12]

    def record(self, kind: str, step: int | None, duration_ms: float | None = None, **fields: Any) -> None:
        """한 줄을 쓴다. kind는 session_start, turn_start, llm, tool, turn_end 중 하나다."""
        if self.path is None:
            return
        line = {
            "session": self.session_id,
            "time": datetime.now().isoformat(timespec="milliseconds"),
            "turn": self.turn,
            "trace_id": self.trace_id,
            "step": step,
            "kind": kind,
            "duration_ms": None if duration_ms is None else round(duration_ms, 1),
            **mask_value(fields),  # 키와 개인정보는 가려서 남긴다(C11 D5).
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
