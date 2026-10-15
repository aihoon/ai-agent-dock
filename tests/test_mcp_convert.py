"""MCP 변환 규칙과 설정 파일 읽기 테스트(네트워크·프로세스 없이). SDK의 실제 타입(mcp_types)으로 결과를 만든다."""

import json

import mcp_types as t
import pytest

from ai_agent_dock.core.tools import ToolError
from ai_agent_dock.core.tools.mcp import ServerSpec, load_config
from ai_agent_dock.core.tools.mcp.convert import MAX_NAME_LENGTH, model_tool_name, result_to_text


def text(s):
    return t.TextContent(type="text", text=s)


# --- 이름 ---


def test_model_tool_name_prefixes_the_server_name():
    assert model_tool_name("demo", "word_count") == "demo__word_count"


@pytest.mark.parametrize(("server", "tool", "expected"), [("files", "read.file", "files__read_file"), ("a b", "x/y", "a_b__x_y")])
def test_characters_the_model_api_rejects_are_replaced(server, tool, expected):
    assert model_tool_name(server, tool) == expected


def test_too_long_name_is_rejected():
    with pytest.raises(ValueError, match=str(MAX_NAME_LENGTH)):
        model_tool_name("s", "x" * MAX_NAME_LENGTH)


# --- 결과 ---


def test_text_blocks_are_joined_with_newlines():
    assert result_to_text(t.CallToolResult(content=[text("첫째"), text("둘째")])) == "첫째\n둘째"


def test_non_text_blocks_become_placeholders():
    result = t.CallToolResult(
        content=[
            text("본문"),
            t.ImageContent(type="image", data="AAAA", mime_type="image/png"),
            t.AudioContent(type="audio", data="BBBB", mime_type="audio/wav"),
            t.ResourceLink(type="resource_link", name="main", uri="file:///a/main.rs"),
            t.EmbeddedResource(type="resource", resource=t.TextResourceContents(uri="file:///a.txt", text="임베디드 본문")),
            t.EmbeddedResource(
                type="resource", resource=t.BlobResourceContents(uri="file:///b.bin", blob="CCCC", mime_type="application/octet-stream")
            ),
        ]
    )

    assert result_to_text(result).splitlines() == [
        "본문",
        "[image: image/png]",
        "[audio: audio/wav]",
        "[resource_link: file:///a/main.rs]",
        "임베디드 본문",
        "[resource: file:///b.bin, application/octet-stream]",
    ]


def test_structured_content_is_used_when_there_is_no_text():
    result = t.CallToolResult(content=[], structured_content={"온도": 22.5})

    assert json.loads(result_to_text(result)) == {"온도": 22.5}


def test_text_wins_over_structured_content():
    result = t.CallToolResult(content=[text("요약")], structured_content={"x": 1})

    assert result_to_text(result) == "요약"


def test_empty_result_has_a_visible_placeholder():
    assert result_to_text(t.CallToolResult(content=[])) == "(빈 결과)"


def test_error_result_raises_tool_error_with_the_servers_message():
    with pytest.raises(ToolError, match="날짜가 과거다"):
        result_to_text(t.CallToolResult(content=[text("날짜가 과거다")], is_error=True))


# --- 설정 파일 ---


def write(tmp_path, data):
    path = tmp_path / "mcp.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def test_valid_config_is_read_with_optional_fields_defaulted(tmp_path):
    path = write(tmp_path, {"servers": {"a": {"command": "uvx", "args": ["x"], "env": {"K": "v"}}, "b": {"command": "run"}}})

    assert load_config(path) == {"a": ServerSpec("uvx", ["x"], {"K": "v"}), "b": ServerSpec("run", [], {})}


def test_missing_config_file_names_the_path(tmp_path):
    with pytest.raises(ValueError, match="nope.json"):
        load_config(tmp_path / "nope.json")


def test_config_that_is_not_json_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="JSON"):
        load_config(write(tmp_path, "{깨짐"))


@pytest.mark.parametrize(
    ("data", "mentions"),
    [
        ([], "servers"),
        ({"servers": []}, "servers"),
        ({"servers": {"a": "uvx"}}, "'a'"),
        ({"servers": {"a": {}}}, "command"),
        ({"servers": {"a": {"command": ""}}}, "command"),
        ({"servers": {"a": {"command": "x", "args": "문자열"}}}, "args"),
        ({"servers": {"a": {"command": "x", "args": [1]}}}, "args"),
        ({"servers": {"a": {"command": "x", "env": {"K": 1}}}}, "env"),
        ({"servers": {"": {"command": "x"}}}, "이름"),
    ],
)
def test_bad_config_says_which_part_is_wrong(tmp_path, data, mentions):
    with pytest.raises(ValueError, match=mentions):
        load_config(write(tmp_path, data))
