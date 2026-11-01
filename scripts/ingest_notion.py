"""Notion 수집: AI Agent Dock 페이지와 그 아래 페이지 전부를 md로 받아 data/notion/에 쓰고, data/notion.sqlite로 색인한다.

사용법: uv run python scripts/ingest_notion.py
실행할 때마다 전부 새로 만든다(고친 페이지, 새 페이지, 지운 페이지가 모두 반영된다). 받기가 끝난 뒤에 이전 결과를 지운다.
.env의 NOTION_API_KEY(읽기 전용 connection의 API token)와 OPENAI_API_KEY(임베딩)가 필요하다.
채팅에서 쓰려면 .env에 DOCS_INDEX=data/notion.sqlite를 넣는다. 평가는 data/index.sqlite를 그대로 쓴다.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from ai_agent_dock.core.retrieval import KnowledgeIndex, OpenAIEmbedder
from ai_agent_dock.core.retrieval.notion import collect

ROOT = Path(__file__).resolve().parent.parent
ROOT_PAGE = "3f03b44cd6f1814db324cf119065852f"  # Notion의 AI Agent Dock 페이지
NOTION_DIR = ROOT / "data" / "notion"  # 받은 md(Notion 페이지 구조 그대로). data/는 git에서 제외한다.
INDEX_PATH = ROOT / "data" / "notion.sqlite"  # Notion 문서 색인. 평가용 색인(data/index.sqlite)과 따로 둔다.
NOTION_VERSION = "2026-03-11"


def notion_get(path: str) -> dict:
    request = urllib.request.Request(
        f"https://api.notion.com/v1/{path}",
        headers={"Authorization": f"Bearer {os.environ['NOTION_API_KEY']}", "Notion-Version": NOTION_VERSION},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Notion API 오류 {e.code} ({path}): {e.read().decode('utf-8', 'replace')}")


def page_markdown(page_id: str) -> str:
    data = notion_get(f"pages/{page_id}/markdown")
    if data.get("truncated"):
        print(f"  ⚠ 일부 블록을 받지 못했습니다({page_id}): {len(data.get('unknown_block_ids', []))}개")
    print(f"  받음: {page_id}", flush=True)
    return data["markdown"]


def page_title(page_id: str) -> str:
    properties = notion_get(f"pages/{page_id}")["properties"].values()
    return "".join(t["plain_text"] for p in properties if p["type"] == "title" for t in p["title"])


def main() -> None:
    load_dotenv()  # NOTION_API_KEY, OPENAI_API_KEY를 .env에서 읽는다.
    if not os.environ.get("NOTION_API_KEY"):
        raise SystemExit(".env에 NOTION_API_KEY가 없습니다. Notion connection의 API token을 넣으세요.")

    print("Notion에서 받는 중...")
    files = collect(page_markdown, ROOT_PAGE, page_title(ROOT_PAGE))

    shutil.rmtree(NOTION_DIR, ignore_errors=True)
    INDEX_PATH.unlink(missing_ok=True)
    for path, content in files:
        target = NOTION_DIR / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    print(f"md 저장: {len(files)}개 ({NOTION_DIR.relative_to(ROOT)}/)")

    store = KnowledgeIndex(INDEX_PATH, OpenAIEmbedder())
    try:
        store.refresh(NOTION_DIR)
        docs, chunks = store.stats()
    except Exception as e:
        raise SystemExit(f"색인 실패: {type(e).__name__}: {e}")
    finally:
        store.close()
    print(f"색인 완료: 문서 {docs}개, 청크 {chunks}개 ({INDEX_PATH.relative_to(ROOT)})")


if __name__ == "__main__":
    main()
