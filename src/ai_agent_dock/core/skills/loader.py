"""스킬 읽기: skills/<이름>/SKILL.md를 찾고, 머리말을 검사하고, 목록 문장과 load_skill Tool을 만든다.

머리말은 표준 라이브러리로 읽는다. name과 description 두 줄만 쓰고, description은 한 줄이어야 한다.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..llm import Tool
from .types import Skill, SkillFormatError, SkillMeta, SkillNotFoundError

SKILL_FILE = "SKILL.md"
NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")  # 소문자, 숫자, 하이픈 하나씩 사이에 두고, 양 끝에는 두지 않는다.
NAME_MAX_LENGTH = 64


def scan_skills(root: Path) -> list[SkillMeta]:
    """root 아래의 스킬 폴더를 이름순으로 찾아 머리말만 읽는다. root가 없으면 빈 목록이다."""
    if not root.is_dir():
        return []
    metas = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        path = folder / SKILL_FILE
        if path.is_file():
            metas.append(_read_meta(path, folder.name))
    return metas


def read_skill(meta: SkillMeta) -> Skill:
    """스킬 파일에서 본문을 읽는다. 머리말 아래의 내용만 돌려준다."""
    _, body = _split_frontmatter(meta.path.read_text(encoding="utf-8"), meta.path)
    return Skill(meta=meta, body=body.strip())


def build_index(metas: list[SkillMeta]) -> str:
    """시스템 프롬프트에 넣을 스킬 목록 문장. 스킬이 없으면 빈 문자열이다(본문은 넣지 않는다)."""
    if not metas:
        return ""
    lines = ["사용할 수 있는 스킬 목록이다. 필요하면 load_skill(name)으로 그 스킬의 본문을 불러와 따른다."]
    lines += [f"- {m.name}: {m.description}" for m in metas]
    return "\n".join(lines)


def make_load_skill_tool(metas: list[SkillMeta]) -> Tool:
    """스킬 본문을 돌려주는 내장 Tool. 호출될 때마다 파일을 읽는다."""
    by_name = {m.name: m for m in metas}

    def load_skill(name: str) -> str:
        meta = by_name.get(name)
        if meta is None:
            available = ", ".join(by_name) or "없음"
            raise SkillNotFoundError(f"없는 스킬: {name!r}. 사용 가능한 스킬: {available}")
        return read_skill(meta).body

    return Tool(
        name="load_skill",
        description=(
            "스킬 목록에 있는 이름의 스킬 본문(절차와 지침)을 불러온다. "
            "스킬 목록에 있는 작업을 요청받으면, 직접 답하기 전에 이 툴로 그 스킬의 본문을 불러와 정해진 형식과 절차를 따른다."
        ),
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "스킬 목록에 있는 이름"}},
            "required": ["name"],
        },
        read_only=True,
        trusted=True,
        func=load_skill,
    )


def _read_meta(path: Path, folder_name: str) -> SkillMeta:
    fields, _ = _split_frontmatter(path.read_text(encoding="utf-8"), path)
    name = fields.get("name", "")
    description = fields.get("description", "")
    if not name:
        raise SkillFormatError(f"{path}: 머리말에 name이 없다")
    if not description:
        raise SkillFormatError(f"{path}: 머리말에 description이 없다")
    if len(name) > NAME_MAX_LENGTH or not NAME_PATTERN.fullmatch(name):
        raise SkillFormatError(f"{path}: name {name!r}은 소문자, 숫자, 하이픈만 쓰고 64자 이하여야 한다")
    if name != folder_name:
        raise SkillFormatError(f"{path}: name {name!r}이 폴더 이름 {folder_name!r}과 같아야 한다")
    return SkillMeta(name=name, description=description, path=path)


def _split_frontmatter(text: str, path: Path) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillFormatError(f"{path}: 맨 위에 --- 로 시작하는 머리말이 없다")
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        raise SkillFormatError(f"{path}: 머리말이 --- 로 닫히지 않았다")
    fields = {}
    for line in lines[1:end]:
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip().strip("\"'")
    return fields, "\n".join(lines[end + 1 :])
