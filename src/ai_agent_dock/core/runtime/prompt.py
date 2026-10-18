"""시스템 프롬프트 조립: 기본 지침, 스킬 목록, 출처 규칙을 이 파일에서만 이어 붙인다.

순서는 항상 같다: 기본 지침 → 스킬 목록(스킬이 있을 때) → 출처 규칙(문서 검색을 켰을 때).
진입점(chat, cli 등)은 필요한 조각을 골라 넘기기만 하고, 조각을 직접 이어 붙이지 않는다.
"""

from __future__ import annotations

from ..retrieval.tool import CITATION_RULE
from ..skills.loader import build_index
from ..skills.types import SkillMeta

# 일반 대화형 어시스턴트의 최소 기본 지침. 모든 진입점이 이 값을 기본으로 쓴다.
DEFAULT_SYSTEM_PROMPT = (
    "당신은 도움이 되는 어시스턴트다. 사용자의 언어로 답한다. "
    "모르거나 확인할 수 없는 것은 지어내지 말고 모른다고 답한다. "
    "날씨, 주가, 환율, 뉴스처럼 실시간으로 바뀌는 정보는 그 정보를 가져오는 Tool이 없으면 답하지 않는다. 대략적인 수치나 일반적인 예상치도 내지 않고, 확인할 수 없다고 말한다. "
    "사용자가 Tool 실행을 거부하면 같은 Tool을 다시 호출하지 않고, 거부된 사실을 알리고 다음 행동을 사용자에게 묻는다. "
    "Tool로 읽지 못한 내용은 있는 것처럼 요약하거나 지어내지 않는다."
)


def build_system_prompt(
    *, skills: list[SkillMeta] | None = None, cite_docs: bool = False, base: str = DEFAULT_SYSTEM_PROMPT
) -> str:
    """시스템 프롬프트 문자열을 만든다.

    skills가 비었으면 스킬 목록을 넣지 않는다. cite_docs가 False면 출처 규칙을 넣지 않는다.
    base를 주면 기본 지침 대신 그 문장을 쓴다(테스트나 특수한 예제용).
    """
    parts = [base]
    index = build_index(skills or [])
    if index:
        parts.append(index)
    if cite_docs:
        parts.append(CITATION_RULE)
    return "\n\n".join(parts)
