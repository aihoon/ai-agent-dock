"""지식 색인: 문서 조각과 그 벡터를 SQLite에 저장하고, 질문과 가까운 조각을 찾는다."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .chunking import chunk_markdown
from .embedding import Embedder
from .types import Hit, RefreshReport

DEFAULT_THRESHOLD = 0.3  # 시작점. 점수가 이보다 낮은 조각은 관련 없다고 보고 버린다. 실제 문서로 확인해 조정한다.


class KnowledgeIndex:
    def __init__(self, db_path: str | Path, embedder: Embedder):
        self._db = sqlite3.connect(str(db_path))
        self._embedder = embedder
        columns = [row[1] for row in self._db.execute("PRAGMA table_info(chunks)")]
        if columns and "title" not in columns:
            # 옛 스키마(출처 번호 방식)는 행을 비우고 문서 제목 컬럼을 추가한다. 색인은 문서에서 다시 만들 수 있다.
            self._db.execute("DELETE FROM chunks")
            self._db.execute("DELETE FROM files")
            self._db.execute("ALTER TABLE chunks ADD COLUMN title TEXT NOT NULL DEFAULT ''")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (source TEXT PRIMARY KEY, hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS chunks (
                source TEXT NOT NULL,
                n INTEGER NOT NULL,
                title TEXT NOT NULL,
                section TEXT NOT NULL,
                text TEXT NOT NULL,
                vector TEXT NOT NULL,
                PRIMARY KEY (source, n)
            );
            """
        )

    def close(self) -> None:
        self._db.close()

    def refresh(self, root: Path) -> RefreshReport:
        """폴더의 .md 파일과 색인을 맞춘다. 내용이 바뀐 파일만 다시 임베딩한다."""
        files = {p.name: p for p in sorted(root.glob("*.md"))} if root.is_dir() else {}
        stored = dict(self._db.execute("SELECT source, hash FROM files").fetchall())
        added = updated = unchanged = 0
        for name, path in files.items():
            content = path.read_text(encoding="utf-8")
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if stored.get(name) == digest:
                unchanged += 1
                continue
            if name in stored:
                updated += 1
            else:
                added += 1
            self._replace(name, digest, content)
        removed = [name for name in stored if name not in files]
        for name in removed:
            self._delete(name)
        self._db.commit()
        return RefreshReport(added=added, updated=updated, removed=len(removed), unchanged=unchanged)

    def search(self, query: str, top_k: int = 3, threshold: float = DEFAULT_THRESHOLD) -> list[Hit]:
        """질문과 가까운 조각을 점수 순으로 돌려준다. 임계값보다 낮은 조각은 뺀다."""
        [query_vector] = self._embedder.embed([query])
        rows = self._db.execute("SELECT title, section, text, vector FROM chunks").fetchall()
        hits = []
        for title, section, text, vector in rows:
            score = _cosine(query_vector, json.loads(vector))
            if score >= threshold:
                hits.append(Hit(cite=_cite(title, section), section=section, text=text, score=score))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def stats(self) -> tuple[int, int]:
        """(색인된 문서 수, 청크 수)를 돌려준다."""
        files = self._db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunks = self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return files, chunks

    def _replace(self, name: str, digest: str, content: str) -> None:
        chunks = chunk_markdown(content)
        title = _document_title(content, name)
        vectors = self._embedder.embed([f"{c.section}\n{c.text}" for c in chunks])
        self._delete(name)
        for n, (chunk, vector) in enumerate(zip(chunks, vectors), start=1):
            self._db.execute(
                "INSERT INTO chunks (source, n, title, section, text, vector) VALUES (?, ?, ?, ?, ?, ?)",
                (name, n, title, chunk.section, chunk.text, json.dumps(vector)),
            )
        self._db.execute("INSERT INTO files (source, hash) VALUES (?, ?)", (name, digest))

    def _delete(self, name: str) -> None:
        self._db.execute("DELETE FROM chunks WHERE source = ?", (name,))
        self._db.execute("DELETE FROM files WHERE source = ?", (name,))


def _document_title(content: str, fallback: str) -> str:
    """문서의 첫 제목(# 한 개)을 문서 제목으로 쓴다. 없으면 파일 이름을 쓴다."""
    for line in content.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return Path(fallback).stem


def _cite(title: str, section: str) -> str:
    """출처 표기: [문서 제목, 위치]. 위치가 문서 제목과 같으면 제목만 쓴다."""
    if section and section != title:
        return f"{title}, {section}"
    return title


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
