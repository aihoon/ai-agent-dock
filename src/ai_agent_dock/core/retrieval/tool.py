"""문서 검색 Tool(search_docs)과 출처 규칙. 루프는 이 Tool이 무엇을 하는지 모른다."""

from __future__ import annotations

from ..llm import Tool
from .types import KnowledgeSearch

CITATION_RULE = (
    "문서 내용으로 답할 때는 search_docs로 먼저 찾는다. 쓴 내용의 근거마다 [문서 제목, 위치]를 붙인다(예: [에이전트 루프, 최대 스텝]). "
    "검색 결과에 근거가 없으면 모른다고 답하고, 근거를 지어내지 않는다."
)


def make_search_docs_tool(
    store: KnowledgeSearch, seen: set[str], top_k: int = 3, calls: list[str] | None = None
) -> Tool:
    """검색 결과의 출처를 seen에 모아 둔다. 출처 검사가 이 집합과 답을 대조한다."""

    def search_docs(query: str) -> str:
        if calls is not None:
            calls.append(query)  # 호출 기록. 채팅이 이번 턴에 검색했는지 알 때 쓴다.
        hits = store.search(query, top_k=top_k)
        if not hits:
            return "관련 문서를 찾지 못했다."
        seen.update(hit.cite for hit in hits)
        return "\n\n".join(f"[{hit.cite}]\n{hit.text}" for hit in hits)

    return Tool(
        name="search_docs",
        description="프로젝트 문서(knowledge 폴더)에서 질문과 관련된 내용을 찾는다. 결과의 [문서 제목, 위치]를 답의 출처로 쓴다.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "찾을 내용을 짧게 적은 검색어"}},
            "required": ["query"],
        },
        func=search_docs,
    )
