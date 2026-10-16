"""출처 검사: 답에 적힌 [파일이름#번호]가 실제 검색 결과에 있었는지 코드로 확인한다.

프롬프트만으로는 출처를 지어낼 수 있으므로, 답이 나온 뒤에 한 번 더 대조한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CITE = re.compile(r"\[([^\[\]\n,]+,[^\[\]\n]+)\]")  # [문서 제목, 위치]. 쉼표가 있어야 출처로 본다.


@dataclass(frozen=True)
class CitationCheck:
    cited: list[str]  # 답에 적힌 출처
    unknown: list[str]  # 이번 턴의 검색 결과에 없던 출처(지어낸 것으로 본다)


def check_citations(answer: str, seen: set[str]) -> CitationCheck:
    cited = CITE.findall(answer)
    unknown = [cite for cite in dict.fromkeys(cited) if cite not in seen]
    return CitationCheck(cited=cited, unknown=unknown)
