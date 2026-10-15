"""스킬 계층의 데이터 클래스와 예외. 스킬 관련 함수는 이 클래스들로만 값을 주고받는다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SkillMeta:
    """SKILL.md 머리말에서 읽은 스킬 정보. 본문은 담지 않는다(본문은 load_skill을 불렀을 때만 읽는다)."""

    name: str
    description: str
    path: Path


@dataclass(frozen=True)
class Skill:
    """본문까지 읽은 스킬."""

    meta: SkillMeta
    body: str


class SkillFormatError(ValueError):
    """SKILL.md의 머리말이나 이름이 규칙에 맞지 않는다."""


class SkillNotFoundError(ValueError):
    """목록에 없는 스킬 이름을 불렀다."""
