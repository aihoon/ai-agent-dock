"""지식 설정: 이미 만들어 둔 색인을 연다. 색인이 없거나 비어 있으면 None(C5와 같은 동작).

문서를 임베딩하는 일은 scripts/index_knowledge.py가 한다. 채팅은 색인을 읽기만 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .embedding import Embedder, OpenAIEmbedder
from .index import KnowledgeIndex
from .types import KnowledgeSearch


@dataclass
class Knowledge:
    store: KnowledgeSearch
    seen: set[str]  # 세션 동안 검색으로 나온 출처. /reset 때 비운다(이력을 이어 쓰는 턴의 출처도 정당하다).
    files: int  # 색인된 문서 수
    chunks: int  # 색인된 청크 수
    calls: list[str] = field(default_factory=list)  # search_docs 호출 기록(검색어). 턴마다 길이 차이로 검색 여부를 안다.


def open_knowledge(db_path: str | Path, embedder: Embedder | None = None) -> Knowledge | None:
    path = Path(db_path)
    if not path.exists():
        return None
    store = KnowledgeIndex(path, embedder or OpenAIEmbedder())
    files, chunks = store.stats()
    if files == 0:
        store.close()
        return None
    return Knowledge(store=store, seen=set(), files=files, chunks=chunks)
