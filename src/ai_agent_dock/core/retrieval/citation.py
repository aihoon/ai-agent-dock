"""출처 검사: 답에 적힌 [파일이름#번호]가 실제 검색 결과에 있었는지 코드로 확인한다.

프롬프트만으로는 출처를 지어낼 수 있으므로, 답이 나온 뒤에 한 번 더 대조한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# [문서 제목, 위치]. 쉼표가 있어야 출처로 본다. 바로 뒤에 (URL)이 오면 웹 링크라 뺀다(C15 5.1).
CITE = re.compile(r"\[([^\[\]\n,]+,[^\[\]\n]+)\](?!\()")


@dataclass(frozen=True)
class CitationCheck:
    cited: list[str]  # 답에 적힌 출처
    unknown: list[str]  # 이번 턴의 검색 결과에 없던 출처(지어낸 것으로 본다)


def check_citations(answer: str, seen: set[str]) -> CitationCheck:
    cited = [part.strip() for group in CITE.findall(answer) for part in group.split(";")]  # [가, 1절; 나, 2절]은 출처 둘이다.
    known = {_squash(cite) for cite in seen}
    unknown = [cite for cite in dict.fromkeys(cited) if _squash(cite) not in known]
    return CitationCheck(cited=cited, unknown=unknown)


def _squash(cite: str) -> str:
    """연속 공백을 한 칸으로, 굵게 표시(**)는 지운다. 모델은 제목의 두 칸 공백(예: Notion의 "C11.  Safety")과 **를 빼고 옮겨 쓴다."""
    return " ".join(cite.replace("**", "").split())
