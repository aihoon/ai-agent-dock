"""시스템 프롬프트 조립: 기본 지침, 메모리, 스킬 목록, 에이전트 목록, 출처 규칙, 외부 내용 규칙을 이 파일에서만 이어 붙인다.

순서는 항상 같다: 기본 지침 → 장기 메모리(있을 때) → 스킬 목록(load_skill이 있을 때) → 에이전트 목록(맡길 수 있을 때)
→ 출처 규칙(search_docs가 있을 때) → 외부 내용 규칙(결과를 감싸는 Tool이 있을 때).
입력은 원재료(AGENT.md 본문, Tool 목록, 스킬, 메모리, 맡길 에이전트)다. 조각의 글은 그 조각을 만드는 코드 옆에 있고,
이 파일은 순서와 조건만 갖는다. runtime은 시스템 프롬프트를 문자열로만 안다(C11 D19).
"""

from __future__ import annotations

from ..llm import Tool
from ..memory import build_memory_index
from ..retrieval.tool import CITATION_RULE
from ..safety import UNTRUSTED_RULE
from ..skills import SkillMeta, build_skill_index
from .loader import build_agent_index
from .types import AgentSpec


def build_system_prompt(
    base: str,
    tools: list[Tool],
    *,
    skills: list[SkillMeta] | None = None,
    memory: list[str] | None = None,
    targets: list[AgentSpec] | None = None,
) -> str:
    """시스템 프롬프트 문자열. base는 기본 지침(AGENT.md 본문), tools는 이 에이전트가 받는 Tool(delegate 포함)이다."""
    names = {t.name for t in tools}
    parts = [
        base,
        build_memory_index(memory or []),  # 장기 메모리는 기본 지침 바로 뒤(C9 D20)
        build_skill_index(skills or []) if "load_skill" in names else "",
        build_agent_index(targets or []),  # 맡길 수 있는 에이전트 목록(C12 D7)
        CITATION_RULE if "search_docs" in names else "",
        UNTRUSTED_RULE if any(t.untrusted_output for t in tools) else "",  # 마지막(C11 D9)
    ]
    return "\n\n".join(part for part in parts if part)
