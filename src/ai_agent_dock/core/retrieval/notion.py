"""Notion 수집: Notion API가 주는 페이지 마크다운을 검색용 md로 바꾸고, 하위 페이지를 폴더 구조로 모은다.

네트워크는 부르는 쪽(scripts/ingest_notion.py)이 맡는다. 여기서는 받은 글만 다룬다.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

FENCE = re.compile(r"^```(\w*)[^\n]*\n.*?^```[ \t]*$\n?", re.M | re.S)  # 코드 블록(언어 이름 포함)
PAGE = re.compile(r'<page\s+url="[^"]*?([0-9a-f]{32}|[0-9a-f-]{36})[^"]*"[^>]*>(.*?)</page>', re.S)  # 하위 페이지
TAG = re.compile(r"</?[a-zA-Z][\w-]*(?:\s[^<>]*)?/?>")  # 영문 이름의 태그만(<태그> 같은 본문 글은 남긴다)
MERMAID_STYLE = re.compile(r"^[ \t]*(?:classDef|style|linkStyle|class)\s.*\n?|:::[\w-]+", re.M)  # 색과 모양 꾸밈. 검색에 잡음이다.


def to_markdown(raw: str) -> str:
    """Notion 마크다운을 검색용 md로: Mermaid는 꾸밈만 빼고 다른 코드 블록은 그대로, 표는 | 표로, 나머지 태그는 글만 남긴다."""
    parts, start = [], 0
    for m in FENCE.finditer(raw):
        parts.append(_clean(raw[start : m.start()]))
        fence = m.group(0)
        parts.append(MERMAID_STYLE.sub("", fence) if m.group(1).lower() == "mermaid" else fence)
        start = m.end()
    parts.append(_clean(raw[start:]))
    return re.sub(r"\n{3,}", "\n\n", "".join(parts)).strip() + "\n"


def child_pages(raw: str) -> list[tuple[str, str]]:
    """본문에 있는 하위 페이지의 (id, 제목)."""
    return [(page_id.replace("-", ""), title.strip()) for page_id, title in PAGE.findall(raw)]


def file_name(title: str) -> str:
    """페이지 제목을 파일 이름으로. 경로에 쓸 수 없는 글자는 _로 바꾼다."""
    return re.sub(r'[\\/:*?"<>|]', "_", title).strip()


def collect(fetch: Callable[[str], str], page_id: str, title: str, folder: Path = Path()) -> list[tuple[Path, str]]:
    """페이지와 그 아래 페이지 전부의 (상대 경로, md 내용). 하위 페이지가 있으면 같은 이름의 폴더에 둔다.

    ponytail: 같은 부모 아래 제목이 같은 페이지는 파일 하나로 겹친다. 생기면 id를 이름에 붙인다.
    """
    raw = fetch(page_id)
    name = file_name(title)
    files = [(folder / f"{name}.md", f"# {title}\n\n출처: https://www.notion.so/{page_id}\n\n{to_markdown(raw)}")]
    for child_id, child_title in child_pages(raw):
        files += collect(fetch, child_id, child_title, folder / name)
    return files


def _clean(text: str) -> str:
    text = PAGE.sub("", text)  # 하위 페이지는 자기 파일이 된다
    text = re.sub(r"<database\b[^>]*>.*?</database>", "", text, flags=re.S)
    text = re.sub(r"\\([\\`*_{}\[\]()#+\-.!~|$<>])", r"\1", text)  # Notion의 \~ \$ 같은 이스케이프를 푼다
    text = re.sub(r"\[([^\[\]]+)\]\(https?://\1/?\)", r"\1", text)  # Notion이 AGENT.md 같은 글을 주소로 보고 만든 링크
    text = re.sub(r"<table\b[^>]*>(.*?)</table>", _table, text, flags=re.S)
    text = re.sub(r'[ \t]*\{(?:color|toggle)="[^}]*\}', "", text)  # 블록 색, 토글 표시
    text = TAG.sub("", text)
    return "\n".join(line.rstrip().lstrip("\t") for line in text.split("\n"))


def _table(m: re.Match) -> str:
    rows = []
    for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", m.group(1), re.S):
        cells = [" ".join(TAG.sub("", c).split()).replace("|", "\\|") for c in re.findall(r"<td\b[^>]*>(.*?)</td>", row, re.S)]
        rows.append("| " + " | ".join(cells) + " |")
        if len(rows) == 1:
            rows.append("|" + " --- |" * len(cells))
    return "\n" + "\n".join(rows) + "\n"
