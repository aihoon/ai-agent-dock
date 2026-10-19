"""장기 메모리: 세션을 넘어 남길 내용을 data/memory.md에 번호 목록으로 저장한다(C9 D16~D19).

한 줄에 메모 하나, "1. 내용" 형식이다. 사용자가 /memory-add로 쓴 문장을 그대로 저장한다(다듬지 않는다).
"""

from __future__ import annotations

import re
from pathlib import Path

NUMBERED = re.compile(r"^\s*\d+\.\s*")


class MemoryStore:
    def __init__(self, path: Path):
        self.path = path

    def items(self) -> list[str]:
        if not self.path.is_file():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [NUMBERED.sub("", line).strip() for line in lines if line.strip()]

    def _write(self, items: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("".join(f"{n}. {text}\n" for n, text in enumerate(items, start=1)), encoding="utf-8")

    def add(self, text: str) -> int:
        """맨 끝에 더하고 새 번호를 돌려준다. 줄바꿈은 한 칸 띄어쓰기로 바꾼다(한 줄에 메모 하나)."""
        text = " ".join(text.split())
        if not text:
            raise ValueError("메모 내용이 비어 있습니다.")
        items = [*self.items(), text]
        self._write(items)
        return len(items)

    def remove(self, number: int) -> str:
        """번호의 메모를 지우고 그 내용을 돌려준다. 남은 메모는 번호를 다시 매긴다."""
        items = self.items()
        if not 1 <= number <= len(items):
            raise ValueError(f"없는 번호입니다: {number} (1~{len(items)})")
        removed = items.pop(number - 1)
        self._write(items)
        return removed
