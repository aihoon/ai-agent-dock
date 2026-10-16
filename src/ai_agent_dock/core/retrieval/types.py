"""지식 검색의 공용 타입과 저장소 약속. 구현(SQLite, 나중에는 다른 DB)은 이 약속만 지키면 된다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Hit:
    cite: str  # 출처 표기 [문서 제목, 위치]. 예: 에이전트 루프, 최대 스텝
    section: str
    text: str
    score: float


@dataclass(frozen=True)
class RefreshReport:
    added: int  # 새로 색인한 파일 수
    updated: int  # 내용이 바뀌어 다시 색인한 파일 수
    removed: int  # 폴더에서 사라져 색인에서 지운 파일 수
    unchanged: int  # 그대로라 건너뛴 파일 수


class KnowledgeSearch(Protocol):
    """검색만 하는 약속. 검색 Tool은 이것만 안다. 구현은 SQLite일 수도, 원격 벡터 DB일 수도 있다."""

    def search(self, query: str, top_k: int = 3) -> list[Hit]:
        """질문과 가까운 조각을 점수 순으로 돌려준다."""
        ...


class KnowledgeIndexer(Protocol):
    """색인을 만드는 약속. 스크립트만 안다. 원격 저장소라면 이 역할은 그 시스템이 맡는다."""

    def refresh(self, root: Path) -> RefreshReport:
        """폴더의 문서와 저장소를 맞춘다."""
        ...
