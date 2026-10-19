"""세션 저장소: 세션을 data/sessions/내부번호.json에 저장하고, 목록을 보여 주고, 불러오고, 이름을 바꾼다.

파일 이름은 바뀌지 않는 내부 번호다. 사용자에게 보이는 이름은 파일 안 맨 위에 한 번만 적는다(C9 D10, D12).
이 모듈은 runtime을 import하지 않는다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..llm import AssistantMessage, Message, ToolCall, ToolMessage, UserMessage

NAME_CHARS = 20  # 자동 이름에 쓰는 첫 질문 글자 수


@dataclass
class SessionRecord:
    id: str  # 내부 번호(추적 기록의 세션 번호와 같다)
    name: str
    provider: str
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated: str = ""
    history: list[Message] = field(default_factory=list)

    @property
    def turns(self) -> int:
        return sum(isinstance(m, UserMessage) for m in self.history)


def auto_name(question: str) -> str:
    """자동 이름: 날짜 + 첫 질문 앞 20자."""
    text = " ".join(question.split())
    return f"{datetime.now():%Y-%m-%d} {text[:NAME_CHARS]}"


def _dump_raw(item: Any) -> Any:
    """제공자 원본 항목을 JSON으로 바꾼다. SDK 객체면 model_dump, 이미 dict면 그대로 둔다."""
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json", exclude_none=True)
    return item


def message_to_dict(m: Message) -> dict[str, Any]:
    if isinstance(m, UserMessage):
        return {"role": "user", "content": m.content}
    if isinstance(m, AssistantMessage):
        return {
            "role": "assistant",
            "content": m.content,
            "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments, "error": c.error} for c in m.tool_calls],
            "raw": [_dump_raw(item) for item in m.raw],
        }
    return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content, "is_error": m.is_error}


def message_from_dict(d: dict[str, Any], keep_raw: bool) -> Message:
    if d["role"] == "user":
        return UserMessage(content=d["content"])
    if d["role"] == "assistant":
        calls = [ToolCall(id=c["id"], name=c["name"], arguments=c["arguments"], error=c.get("error")) for c in d.get("tool_calls", [])]
        return AssistantMessage(content=d.get("content"), tool_calls=calls, raw=list(d.get("raw", [])) if keep_raw else [])
    return ToolMessage(tool_call_id=d["tool_call_id"], content=d["content"], is_error=d.get("is_error", False))


class SessionStore:
    def __init__(self, directory: Path):
        self.directory = directory

    def _path(self, session_id: str) -> Path:
        return self.directory / f"{session_id}.json"

    def save(self, record: SessionRecord) -> None:
        record.updated = datetime.now().isoformat(timespec="seconds")
        self.directory.mkdir(parents=True, exist_ok=True)
        data = {
            "id": record.id,
            "name": record.name,
            "provider": record.provider,
            "created": record.created,
            "updated": record.updated,
            "history": [message_to_dict(m) for m in record.history],
        }
        tmp = self._path(record.id).with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self._path(record.id))  # 다 쓴 뒤 바꿔 끼워서 도중에 꺼져도 이전 파일이 깨지지 않게 한다.

    def load(self, session_id: str, provider: str) -> SessionRecord:
        """세션을 불러온다. 저장한 제공자와 지금 제공자가 다르면 raw를 버리고 텍스트와 Tool 요청만 쓴다(D15)."""
        data = json.loads(self._path(session_id).read_text(encoding="utf-8"))
        keep_raw = data["provider"] == provider
        return SessionRecord(
            id=data["id"],
            name=data["name"],
            provider=provider,
            created=data["created"],
            updated=data["updated"],
            history=[message_from_dict(d, keep_raw) for d in data["history"]],
        )

    def list(self) -> list[SessionRecord]:
        """저장된 세션을 마지막 시각이 최근인 순서로 돌려준다. 깨진 파일은 건너뛴다."""
        records = []
        for path in self.directory.glob("*.json") if self.directory.is_dir() else []:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                history = [message_from_dict(d, keep_raw=False) for d in data["history"]]
                records.append(SessionRecord(id=data["id"], name=data["name"], provider=data["provider"], created=data["created"], updated=data["updated"], history=history))
            except (ValueError, KeyError):
                continue
        return sorted(records, key=lambda r: r.updated, reverse=True)

    def names(self) -> set[str]:
        return {r.name for r in self.list()}

    def unique_name(self, name: str) -> str:
        """자동 이름이 겹치면 뒤에 (2), (3)을 붙인다."""
        taken = self.names()
        candidate, n = name, 2
        while candidate in taken:
            candidate, n = f"{name} ({n})", n + 1
        return candidate

    def rename(self, record: SessionRecord, new_name: str) -> None:
        """이름을 바꾼다. 다른 세션과 이름이 겹치면 ValueError. 파일 맨 위의 이름 한 칸만 바뀐다."""
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("이름이 비어 있습니다.")
        if any(r.name == new_name and r.id != record.id for r in self.list()):
            raise ValueError(f"같은 이름의 세션이 이미 있습니다: {new_name}")
        record.name = new_name
        self.save(record)
