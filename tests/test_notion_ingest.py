"""Notion 수집 테스트: Notion 마크다운을 검색용 md로 바꾸기, 하위 페이지를 폴더 구조로 모으기, 하위 폴더 색인. 네트워크는 쓰지 않는다."""

from pathlib import Path

from ai_agent_dock.core.retrieval import HashingEmbedder, KnowledgeIndex
from ai_agent_dock.core.retrieval.notion import child_pages, collect, file_name, to_markdown

RAW = """<callout icon="📌" color="gray_bg">
<p>
<b>플랫폼</b>을 먼저 만든다. C0\\~C14 완료, 태그 <code>c14</code>.
</p>
</callout>

## 1. 목표 {color="blue"}

### 4.1  [AGENT.md](http://AGENT.md) 머리말, [문서](https://example.com)

```mermaid
flowchart LR
  A["시작"]:::src --> B
  classDef src fill:#eef2ff,stroke:#6366f1
  style B fill:#fff
```

```bash
uv run python scripts/ingest_notion.py
```

<table header-row="true">
<colgroup>
<col />
<col width="452" />
</colgroup>
<tr>
<td>
항목
</td>
<td>
값
</td>
</tr>
<tr>
<td>
모델
</td>
<td>
gpt-6-luna | 기본
</td>
</tr>
</table>

용어는 <mention url="https://app.notion.com/p/abc" /> 참고. <mention-page url="https://app.notion.com/p/def">C14</mention-page> 참고.
`git push origin <태그>`

<page url="https://app.notion.com/p/3f03b44cd6f18115bf3fff6bd6e8be37">
C0.  프로젝트 기반
</page>
<page url="https://www.notion.so/C1-Agent-Loop-3f03b44cd6f18161a4ccd35f561de2e7">C1/C2 메모</page>
"""


def test_to_markdown_strips_mermaid_styles_and_tags_and_keeps_code_and_tables():
    md = to_markdown(RAW)

    assert "```mermaid\nflowchart LR\n  A[\"시작\"] --> B\n```" in md  # 그림은 남기고 꾸밈(classDef, style, :::)만 뺀다
    assert "```bash\nuv run python scripts/ingest_notion.py\n```" in md  # 코드 블록은 남긴다
    assert "| 항목 | 값 |\n| --- | --- |\n| 모델 | gpt-6-luna \\| 기본 |" in md
    assert "플랫폼을 먼저 만든다. C0~C14 완료, 태그 c14." in md
    assert "## 1. 목표\n" in md
    assert "### 4.1  AGENT.md 머리말, [문서](https://example.com)" in md  # 글자와 주소가 같은 자동 링크만 글자로
    assert "C14 참고" in md and "<" not in md.replace("<태그>", "")
    assert "프로젝트 기반" not in md  # 하위 페이지는 자기 파일이 된다


def test_child_pages_reads_ids_and_titles():
    assert child_pages(RAW) == [
        ("3f03b44cd6f18115bf3fff6bd6e8be37", "C0.  프로젝트 기반"),
        ("3f03b44cd6f18161a4ccd35f561de2e7", "C1/C2 메모"),
    ]


def test_file_name_replaces_characters_not_allowed_in_paths():
    assert file_name("C1/C2: 메모?  ") == "C1_C2_ 메모_"


def test_collect_follows_the_notion_tree_as_folders():
    pages = {
        "root": '본문\n<page url="https://app.notion.com/p/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa">자식</page>',
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa": '<page url="https://app.notion.com/p/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb">손자</page>',
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb": "손자 본문",
    }

    files = dict(collect(pages.__getitem__, "root", "AI Agent Dock"))

    assert list(files) == [Path("AI Agent Dock.md"), Path("AI Agent Dock/자식.md"), Path("AI Agent Dock/자식/손자.md")]
    assert files[Path("AI Agent Dock.md")].startswith("# AI Agent Dock\n\n출처: https://www.notion.so/root\n\n본문")
    assert files[Path("AI Agent Dock/자식/손자.md")].endswith("손자 본문\n")


def test_refresh_reads_subfolders_and_keeps_them_apart_by_path(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "top.md").write_text("# 위\n내용", encoding="utf-8")
    (tmp_path / "a" / "top.md").write_text("# 아래\n내용", encoding="utf-8")  # 이름이 같아도 경로가 다르면 다른 문서
    index = KnowledgeIndex(":memory:", HashingEmbedder())

    report = index.refresh(tmp_path)

    assert report.added == 2 and index.stats() == (2, 2)
