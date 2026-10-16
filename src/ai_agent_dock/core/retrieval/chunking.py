"""문서를 검색 단위(청크)로 나눈다. 제목 경계로 먼저 나누고, 너무 긴 부분은 겹침을 두고 자른다."""

from __future__ import annotations

from dataclasses import dataclass

MAX_CHARS = 800  # 한 청크의 최대 글자 수
OVERLAP = 100  # 잘린 청크끼리 겹치는 글자 수(문장이 잘리는 경계를 덜 놓치게 한다)


@dataclass(frozen=True)
class Chunk:
    section: str  # 속한 제목. 제목 앞 내용이면 빈 문자열
    text: str  # 제목을 뺀 본문


def chunk_markdown(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[Chunk]:
    """마크다운을 제목(#) 단위로 나누고, 각 부분을 max_chars 이하로 자른다."""
    chunks: list[Chunk] = []
    section = ""
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            chunks.extend(Chunk(section, piece) for piece in _split(content, max_chars, overlap))
        body.clear()

    for line in text.splitlines():
        if line.startswith("#"):
            flush()
            section = line.lstrip("#").strip()
        else:
            body.append(line)
    flush()
    return chunks


def _split(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    step = max_chars - overlap
    pieces = []
    start = 0
    while True:
        pieces.append(text[start : start + max_chars])
        if start + max_chars >= len(text):
            return pieces
        start += step
