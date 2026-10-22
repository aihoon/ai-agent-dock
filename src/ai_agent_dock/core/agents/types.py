"""에이전트 정의의 데이터 클래스와 예외."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentSpec:
    """agents/<이름>/AGENT.md에서 읽은 에이전트 정의(C12 Q1)."""

    name: str
    description: str  # 맡기는 쪽의 에이전트 목록에 한 줄로 보인다.
    tools: tuple[str, ...]  # 쓸 Tool의 이름 패턴. "*"는 모두, "!패턴"은 제외(fnmatch)
    body: str  # 역할 지침. 시스템 프롬프트의 기본 지침 자리에 들어간다.
    path: Path
    max_steps: int | None = None  # 없으면 main은 AGENT_MAX_STEPS, 하위 에이전트는 SUBAGENT_MAX_STEPS


class AgentFormatError(ValueError):
    """AGENT.md의 머리말이나 이름, Tool 목록이 규칙에 맞지 않는다."""
