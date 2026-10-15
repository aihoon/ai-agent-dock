"""스킬 계층 테스트: 머리말 검사, 목록 문장, load_skill Tool, 대화 루프 안에서의 사용."""

from pathlib import Path

import pytest
from fakes import TEST_CONFIG, FakeLLM, final

from ai_agent_dock.core.llm import LLMResponse, ToolCall
from ai_agent_dock.core.runtime.types import AgentSettings
from ai_agent_dock.core.session import Session
from ai_agent_dock.core.skills import (
    SkillFormatError,
    SkillNotFoundError,
    build_index,
    make_load_skill_tool,
    read_skill,
    scan_skills,
)
from ai_agent_dock.core.tools import ToolArgumentError, ToolRegistry


def write_skill(root: Path, folder: str, text: str) -> Path:
    path = root / folder / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def skill_text(name="debug-checklist", description="버그를 좁혀 찾는 절차", body="1. 재현한다."):
    return f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n"


def test_scan_reads_meta_and_body(tmp_path):
    write_skill(tmp_path, "debug-checklist", skill_text())

    [meta] = scan_skills(tmp_path)

    assert (meta.name, meta.description) == ("debug-checklist", "버그를 좁혀 찾는 절차")
    assert read_skill(meta).body == "1. 재현한다."


def test_missing_root_and_folders_without_skill_file_are_ignored(tmp_path):
    assert scan_skills(tmp_path / "없음") == []
    (tmp_path / "빈폴더").mkdir()
    write_skill(tmp_path, "debug-checklist", skill_text())

    assert [m.name for m in scan_skills(tmp_path)] == ["debug-checklist"]


@pytest.mark.parametrize(
    "text",
    [
        "name: debug-checklist\ndescription: b\n",  # 머리말 없음
        "---\nname: debug-checklist\ndescription: b\n",  # 닫는 --- 없음
        "---\ndescription: b\n---\n",  # name 없음
        "---\nname: debug-checklist\n---\n",  # description 없음
    ],
)
def test_broken_frontmatter_is_rejected(tmp_path, text):
    write_skill(tmp_path, "debug-checklist", text)

    with pytest.raises(SkillFormatError):
        scan_skills(tmp_path)


@pytest.mark.parametrize("name", ["Debug", "debug--check", "-debug", "debug-", "debug_check", "디버그"])
def test_invalid_name_is_rejected(tmp_path, name):
    write_skill(tmp_path, name, skill_text(name=name))

    with pytest.raises(SkillFormatError):
        scan_skills(tmp_path)


def test_name_must_match_folder(tmp_path):
    write_skill(tmp_path, "other", skill_text(name="debug-checklist"))

    with pytest.raises(SkillFormatError, match="폴더 이름"):
        scan_skills(tmp_path)


def test_index_is_empty_without_skills():
    assert build_index([]) == ""


def test_index_lists_name_and_description_without_body(tmp_path):
    write_skill(tmp_path, "debug-checklist", skill_text(body="비밀 본문"))

    index = build_index(scan_skills(tmp_path))

    assert "debug-checklist: 버그를 좁혀 찾는 절차" in index
    assert "비밀 본문" not in index


def test_load_skill_returns_body_and_lists_available_names_on_error(tmp_path):
    write_skill(tmp_path, "debug-checklist", skill_text())
    tool = make_load_skill_tool(scan_skills(tmp_path))

    assert tool.func(name="debug-checklist") == "1. 재현한다."
    with pytest.raises(SkillNotFoundError, match="debug-checklist"):
        tool.func(name="없음")


def test_load_skill_reads_file_at_call_time(tmp_path):
    path = write_skill(tmp_path, "debug-checklist", skill_text(body="처음"))
    tool = make_load_skill_tool(scan_skills(tmp_path))

    path.write_text(skill_text(body="나중"), encoding="utf-8")

    assert tool.func(name="debug-checklist") == "나중"


def test_registry_validates_load_skill_arguments(tmp_path):
    write_skill(tmp_path, "debug-checklist", skill_text())
    registry = ToolRegistry()
    registry.add(make_load_skill_tool(scan_skills(tmp_path)), source="skills")
    tool = registry.tools()[0]

    with pytest.raises(ToolArgumentError):
        tool.func()  # name 누락
    assert tool.func(name="debug-checklist") == "1. 재현한다."


def test_registry_truncates_long_skill_body(tmp_path):
    write_skill(tmp_path, "debug-checklist", skill_text(body="가" * 50))
    registry = ToolRegistry(max_result_chars=20)
    registry.add(make_load_skill_tool(scan_skills(tmp_path)), source="skills")

    output = registry.tools()[0].func(name="debug-checklist")

    assert output.startswith("가" * 20)
    assert "잘렸습니다" in output


def test_model_loads_skill_through_the_loop(tmp_path):
    write_skill(tmp_path, "debug-checklist", skill_text())
    registry = ToolRegistry()
    registry.add(make_load_skill_tool(scan_skills(tmp_path)), source="skills")
    request = LLMResponse(text=None, tool_calls=[ToolCall(id="c1", name="load_skill", arguments={"name": "debug-checklist"})])
    llm = FakeLLM([request, final("끝")])
    session = Session(AgentSettings(config=TEST_CONFIG, tools=registry.tools(), llm=llm))

    result = session.ask("오류를 찾아 줘")

    tool_message = next(m for m in result.history if m.role == "tool")
    assert tool_message.content == "1. 재현한다."
    assert result.answer == "끝"


def test_add_user_message_does_not_call_the_model():
    llm = FakeLLM([])  # 호출되면 AssertionError가 난다.
    session = Session(AgentSettings(config=TEST_CONFIG, llm=llm))

    session.add_user_message("지침")

    assert [m.role for m in session.history] == ["user"]
    assert llm.calls == []
