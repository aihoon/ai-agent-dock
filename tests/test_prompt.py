"""시스템 프롬프트 조립 테스트: 조각이 정해진 순서로 들어가고, 조건에 맞을 때만 들어간다."""

from pathlib import Path

from ai_agent_dock.core.retrieval.tool import CITATION_RULE
from ai_agent_dock.core.runtime.prompt import DEFAULT_SYSTEM_PROMPT, UNTRUSTED_RULE, build_system_prompt
from ai_agent_dock.core.skills import SkillMeta


def test_default_and_untrusted_rule_only_when_no_skills_and_no_docs():
    assert build_system_prompt() == f"{DEFAULT_SYSTEM_PROMPT}\n\n{UNTRUSTED_RULE}"  # 외부 내용 규칙은 항상 마지막(C11 D9)


def test_parts_are_joined_in_fixed_order():
    skill = SkillMeta(name="요약", description="글을 짧게 줄인다", path=Path("요약/SKILL.md"))

    text = build_system_prompt(skills=[skill], cite_docs=True)

    assert text.index(DEFAULT_SYSTEM_PROMPT) < text.index("요약") < text.index(CITATION_RULE)


def test_citation_rule_only_when_docs_are_on():
    assert CITATION_RULE not in build_system_prompt(skills=[])
    assert build_system_prompt(cite_docs=True).endswith(f"{CITATION_RULE}\n\n{UNTRUSTED_RULE}")


def test_base_can_be_replaced_for_special_entry_points():
    assert build_system_prompt(base="다른 지침") == f"다른 지침\n\n{UNTRUSTED_RULE}"
