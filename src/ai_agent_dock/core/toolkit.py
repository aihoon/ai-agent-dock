"""빌트인 Tool 조립: 기본 제공 Tool과 기능별 Tool(스킬, 문서 검색, 되묻기)을 한 목록으로 모은다.

의존 방향(순환 방지): 이 파일만 skills, retrieval, hitl, tools를 import한다.
그 모듈들은 이 파일을 import하지 않는다. 진입점(examples)은 이 파일을 통해서만 빌트인 Tool을 얻는다.
"""

from __future__ import annotations

from collections.abc import Callable

from .hitl import make_ask_user_tool
from .llm import Tool
from .retrieval import Knowledge, make_search_docs_tool
from .skills import SkillMeta, make_load_skill_tool
from .tools import BUILTIN_TOOLS


def make_builtin_tools(
    *,
    skills: list[SkillMeta] | None = None,
    knowledge: Knowledge | None = None,
    input_fn: Callable[[str], str] = input,
) -> list[tuple[str, Tool]]:
    """(출처, Tool) 목록을 돌려준다. 출처는 /tools와 오류 메시지에 쓰는 이름이다.

    스킬이 없으면 load_skill을, 문서 색인이 열리지 않았으면 search_docs를 넣지 않는다.
    ask_user는 항상 넣는다. input_fn은 되묻기의 입력 함수다(테스트에서 바꿀 수 있다).
    """
    entries = [("builtin", tool) for tool in BUILTIN_TOOLS]
    entries.append(("builtin", make_ask_user_tool(input_fn)))  # 되묻기 Tool. 승인 대상이 아니다.
    if skills:
        entries.append(("skills", make_load_skill_tool(skills)))
    if knowledge:
        entries.append(("knowledge", make_search_docs_tool(knowledge.store, knowledge.seen, calls=knowledge.calls)))
    return entries
