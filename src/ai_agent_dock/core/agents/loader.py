"""에이전트 정의 읽기: agents/<이름>/AGENT.md를 찾고, 머리말을 검사하고, Tool 패턴을 고르고, 에이전트 목록 문장을 만든다.

형식은 스킬(SKILL.md)과 같다. 머리말: name, description, tools, max_steps(생략 가능). 본문: 역할 지침.
tools는 [a, b] 또는 a, b로 쓴다. "*"는 모든 Tool, "!web__*"처럼 !로 시작하면 제외다.
"""

from __future__ import annotations

from collections.abc import Iterable
from fnmatch import fnmatchcase
from pathlib import Path

from ..llm import Tool
from ..skills.loader import NAME_MAX_LENGTH, NAME_PATTERN, split_frontmatter
from ..skills.types import SkillFormatError
from .types import AgentSpec, AgentFormatError

AGENT_FILE = "AGENT.md"
MAIN = "main"  # 사용자와 대화하는 에이전트. 다른 에이전트가 main에게 맡길 수는 없다.
AGENT_INDEX_HEAD = (
    "일을 맡길 수 있는 에이전트 목록이다. 검색을 여러 번 해야 하는 조사(여러 문서를 찾아 비교하거나 정리하는 일)는 "
    "직접 search_docs를 여러 번 부르지 말고 delegate(agent, task)로 맡긴다. 한 번 찾아 답할 수 있는 질문은 직접 한다. "
    "맡길 때는 필요한 맥락을 task에 모두 적는다(맡은 에이전트는 이 대화를 보지 못한다)."
)


def load_agents(root: Path) -> dict[str, AgentSpec]:
    """root 아래의 에이전트 정의를 이름순으로 읽는다. main이 없으면 시작할 수 없다(코드에 기본 지침이 없다, C12 Q11)."""
    agents = {}
    if root.is_dir():
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            path = folder / AGENT_FILE
            if path.is_file():
                agents[folder.name] = _read(path, folder.name)
    if MAIN not in agents:
        raise AgentFormatError(f"메인 에이전트 정의가 없다: {root / MAIN / AGENT_FILE}")
    return agents


def matches(patterns: Iterable[str], name: str) -> bool:
    """Tool 이름이 패턴에 맞는지. 하나라도 포함 패턴에 맞고, 제외 패턴에는 하나도 맞지 않아야 한다."""
    patterns = list(patterns)
    include = [p for p in patterns if not p.startswith("!")]
    exclude = [p[1:] for p in patterns if p.startswith("!")]
    return any(fnmatchcase(name, p) for p in include) and not any(fnmatchcase(name, p) for p in exclude)


def select_tools(patterns: Iterable[str], tools: list[Tool]) -> list[Tool]:
    patterns = list(patterns)
    return [t for t in tools if matches(patterns, t.name)]


def build_agent_index(agents: list[AgentSpec]) -> str:
    """시스템 프롬프트에 넣을 에이전트 목록 문장. 이름과 한 줄 설명만 넣고 역할 지침은 넣지 않는다(C5 D2와 같은 원칙)."""
    if not agents:
        return ""
    return "\n".join([AGENT_INDEX_HEAD, *(f"- {a.name}: {a.description}" for a in agents)])


def _read(path: Path, folder_name: str) -> AgentSpec:
    try:
        fields, body = split_frontmatter(path.read_text(encoding="utf-8"), path)
    except SkillFormatError as e:  # 머리말 형식 오류는 스킬과 같은 문장으로 알린다.
        raise AgentFormatError(str(e)) from None
    name, description = fields.get("name", ""), fields.get("description", "")
    if not name or not description:
        raise AgentFormatError(f"{path}: 머리말에 name과 description이 있어야 한다")
    if len(name) > NAME_MAX_LENGTH or not NAME_PATTERN.fullmatch(name):
        raise AgentFormatError(f"{path}: name {name!r}은 소문자, 숫자, 하이픈만 쓰고 64자 이하여야 한다")
    if name != folder_name:
        raise AgentFormatError(f"{path}: name {name!r}이 폴더 이름 {folder_name!r}과 같아야 한다")
    tools = tuple(t.strip().strip("\"'") for t in fields.get("tools", "").strip().strip("[]").split(",") if t.strip())
    if not tools:
        raise AgentFormatError(f"{path}: 머리말에 tools가 없다(모든 Tool이면 \"*\")")
    max_steps = fields.get("max_steps")
    if max_steps is not None and (not max_steps.isdigit() or int(max_steps) < 1):
        raise AgentFormatError(f"{path}: max_steps는 1 이상의 정수여야 한다: {max_steps!r}")
    if not body.strip():
        raise AgentFormatError(f"{path}: 본문(역할 지침)이 비어 있다")
    return AgentSpec(
        name=name,
        description=description,
        tools=tools,
        body=body.strip(),
        path=path,
        max_steps=int(max_steps) if max_steps else None,
    )
