"""위임(C12): 에이전트 정의로 에이전트 설정(AgentConfig)을 만들고, 맡길 수 있으면 delegate Tool을 준다.

main과 하위 에이전트는 같은 함수(config_for)로 만든다. 차이는 코드 규칙 하나다(C12 Q11):
delegate로 불린 에이전트(깊이 1 이상)는 장기 메모리, 스킬 목록, ask_user, load_skill을 받지 않는다(컨텍스트 격리, Q4, Q7).
하위 에이전트는 새 이력으로 task만 받고 최종 답만 돌려준다(Q5). 승인, Safety, Hook, 감사 로그는 부모 것을 그대로 쓴다(D6).
delegate와 에이전트 목록은 함께 붙고 함께 빠진다. 붙는 조건: tools에 delegate가 맞고, 깊이 < SUBAGENT_MAX_DEPTH이고,
부를 에이전트(자기 자신, 부모 줄, main, delegate: false를 뺀 나머지)가 남을 때(Q12, C14 Q2-2).
워크플로는 only로 Tool을 더 좁힐 수 있다(C14 Routing의 경로 표, Q2-3). AGENT.md의 model이 있으면 그 모델로 돈다(C14 Q3).
의존 방향: core/agents → core/runtime. runtime은 agents를 모른다(delegate는 Tool일 뿐이다).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace

from ..hitl import Approval
from ..llm import Tool, ToolCall, model_config
from ..observability import Tracer
from ..runtime.loop import run_agent
from ..runtime.types import AgentResult, AgentConfig
from ..skills import SkillMeta
from ..tools import DEFAULT_MAX_RESULT_CHARS, ToolRegistry
from .loader import MAIN, matches, select_tools
from .system_prompt import build_system_prompt
from .types import AgentSpec, AgentFormatError

DELEGATE = "delegate"
PARENT_ONLY = {"ask_user", "load_skill"}  # 되묻기와 스킬은 main만 쓴다. 하위 에이전트는 사용자를 볼 수 없다.
MAX_DEPTH = 3  # SUBAGENT_MAX_DEPTH의 최대값
GLOB = "*?["


@dataclass
class AgentTeam:
    """설치된 에이전트들과 위임 규칙. 응용은 config_for(MAIN, base)로 main 설정을 만든다."""

    agents: dict[str, AgentSpec]
    max_depth: int = 1  # SUBAGENT_MAX_DEPTH. 1이면 main만 맡긴다.
    max_steps: int | None = None  # SUBAGENT_MAX_STEPS. AGENT.md의 max_steps가 우선한다.
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS  # 하위 에이전트의 답 길이 상한(TOOL_RESULT_MAX_CHARS, Q5)
    approve_for: Callable[[str], Callable[[ToolCall], Approval]] | None = None  # 하위 에이전트 이름 → 승인 함수(메뉴에 [이름], Q7)
    on_start: Callable[[str, str], None] | None = None  # (에이전트, task). 응용이 진행 줄을 보여 준다(F5).
    on_end: Callable[[str, AgentResult], None] | None = None  # (에이전트, 결과). 평가가 하위 에이전트의 Tool 호출을 모은다.
    wrap_llm: Callable[[str, Callable], Callable] | None = None  # (에이전트, 모델 호출 함수) → 감싼 함수. 응용이 진행 줄에 에이전트 이름을 붙인다.

    def __post_init__(self) -> None:
        if not 1 <= self.max_depth <= MAX_DEPTH:
            raise ValueError(f"SUBAGENT_MAX_DEPTH는 1~{MAX_DEPTH} 사이여야 한다: {self.max_depth}")

    def check(self, tools: list[Tool]) -> list[str]:
        """시작할 때 하위 에이전트의 Tool을 검사한다. 쓰기 Tool이 있으면 오류, 연결되지 않은 이름은 경고 문장으로 돌려준다."""
        warnings = []
        names = {t.name for t in tools}
        for agent in self.agents.values():
            if agent.name == MAIN:
                continue
            writes = [t.name for t in self._tools_of(agent, tools, depth=1) if not t.read_only]
            if writes and agent.delegate:  # 맡길 수 없는 에이전트(경로 전용)는 하위 에이전트가 아니다.
                raise AgentFormatError(
                    f"{agent.path}: 하위 에이전트는 읽기 전용 Tool만 쓴다(C12 D2): {', '.join(writes)}. "
                    '서버가 읽기 전용 힌트를 주지 않는 Tool이면 mcp.json의 서버 항목에 "read_only"로 적는다.'
                )
            missing = [p for p in agent.tools if p != DELEGATE and not p.startswith("!") and not any(c in p for c in GLOB) and p not in names]
            if missing:  # MCP 서버를 껐거나 문서 색인이 없으면 그 Tool 없이 일한다.
                warnings.append(f"에이전트 {agent.name}: 연결되지 않은 Tool은 빼고 씁니다: {', '.join(missing)}")
        return warnings

    def targets(self, name: str, depth: int = 0, parents: tuple[str, ...] = ()) -> list[AgentSpec]:
        """name이 지금 맡길 수 있는 에이전트들. 비어 있으면 delegate도 에이전트 목록도 주지 않는다."""
        if depth >= self.max_depth or not matches(self.agents[name].tools, DELEGATE):
            return []
        blocked = {MAIN, name, *parents}  # 자기 자신과 부모 줄을 부르면 순환한다.
        return [a for a in self.agents.values() if a.name not in blocked and a.delegate]

    def config_for(
        self,
        name: str,
        base: AgentConfig,
        *,
        skills: list[SkillMeta] | None = None,
        memory: list[str] | None = None,
        depth: int = 0,
        parents: tuple[str, ...] = (),
        tracer: Tracer | None = None,
        only: Iterable[str] | None = None,
    ) -> AgentConfig:
        """에이전트 하나의 설정. base는 모든 에이전트가 함께 쓰는 설정(모델, 설치된 전체 Tool, 승인, Safety, Hook, 감사 로그, 추적)이다.

        main은 depth=0으로 부르고 skills와 memory를 넘긴다. 하위 에이전트는 delegate가 depth와 parents, tracer를 넘긴다.
        only를 주면 AGENT.md의 Tool 중 only에도 맞는 것만 준다(교집합). delegate도 only에 맞아야 붙는다.
        """
        agent = self.agents[name]
        if depth == 0:
            tracer = base.tracer
        tools = self._tools_of(agent, base.tools, depth)
        if only is not None:
            only = list(only)
            tools = select_tools(only, tools)
        targets = self.targets(name, depth, parents) if only is None or matches(only, DELEGATE) else []
        if targets:
            tools.append(self._delegate_tool(base, depth, (*parents, name), targets, tracer))
        # 하위 에이전트는 load_skill이 없어 스킬 목록이 빠진다. 메모리는 여기서 뺀다(Q4).
        prompt = build_system_prompt(agent.body, tools, skills=skills, memory=memory if depth == 0 else None, targets=targets)
        if depth == 0:  # main: 실행 인자(--max-steps) > AGENT.md > AGENT_MAX_STEPS
            max_steps, approve = base.max_steps or agent.max_steps, base.approve
        else:  # 하위 에이전트: AGENT.md > SUBAGENT_MAX_STEPS
            max_steps = agent.max_steps or self.max_steps
            approve = self.approve_for(name) if self.approve_for else base.approve
        llm_config = model_config(agent.model, base.llm_config) if agent.model else base.llm_config
        llm = self.wrap_llm(name, base.llm) if self.wrap_llm else base.llm
        return replace(base, llm_config=llm_config, llm=llm, tools=tools, system_prompt=prompt, max_steps=max_steps, tracer=tracer, approve=approve)

    def _tools_of(self, agent: AgentSpec, tools: list[Tool], depth: int) -> list[Tool]:
        pool = tools if depth == 0 else [t for t in tools if t.name not in PARENT_ONLY]
        return select_tools(agent.tools, pool)

    def _delegate_tool(
        self, base: AgentConfig, depth: int, chain: tuple[str, ...], targets: list[AgentSpec], tracer: Tracer | None
    ) -> Tool:
        def delegate(agent: str, task: str) -> str:
            delegation = uuid.uuid4().hex[:8]
            if tracer is not None:
                tracer.record("delegate", None, id=delegation, target=agent, task=task)
            child_tracer = tracer.child(agent, delegation) if tracer is not None else None
            config = self.config_for(agent, base, depth=depth + 1, parents=chain, tracer=child_tracer)
            if self.on_start:
                self.on_start(agent, task)
            result = run_agent(config, task)  # 새 이력. 하위 에이전트의 Tool 결과는 부모 이력에 들어가지 않는다.
            if self.on_end:
                self.on_end(agent, result)
            answer = result.answer or "(답 없음)"
            if result.status == "max_steps":
                return f"[{agent}가 스텝 상한({result.steps})에 걸려 끝내지 못했다. 지금까지의 답] {answer}"
            return answer

        tool = Tool(
            name=DELEGATE,
            description=(
                "에이전트 목록에 있는 다른 에이전트에게 일을 맡기고 그 에이전트의 최종 답을 받는다. "
                "맡은 에이전트는 이 대화를 보지 못하므로, task에 필요한 맥락(질문, 조건, 원하는 답의 형식)을 모두 적는다."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "enum": [a.name for a in targets], "description": "맡길 에이전트 이름"},
                    "task": {"type": "string", "description": "맡길 일과 필요한 맥락"},
                },
                "required": ["agent", "task"],
            },
            func=delegate,
            read_only=True,  # 맡기는 것 자체는 승인을 묻지 않는다. 하위 에이전트의 Tool이 각자 승인을 거친다.
            trusted=True,
            untrusted_output=True,  # 외부 자료를 읽고 쓴 답이라 외부 내용으로 감싸고 지시문을 탐지한다(Q6).
        )
        registry = ToolRegistry(max_result_chars=self.max_result_chars)  # 인자 검사(enum)와 답 길이 제한을 다른 Tool과 같게 한다.
        registry.add(tool, source="agents")
        return registry.tools()[0]
