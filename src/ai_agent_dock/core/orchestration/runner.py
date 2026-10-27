"""워크플로 실행(C10 D4, D9, D10, D15): 워크플로 그래프를 체크포인트와 함께 시작하고, 사람 승인에서 멈추고, 이어 간다.

워크플로 정의(그래프)는 core 밖(src/ai_agent_dock/workflows/)에 있고, 응용이 목록을 넘긴다. 이 모듈은 정의를 로드만 한다.
체크포인트 저장 방식(checkpointer)은 밖에서 받는다. 채팅은 SQLite, 테스트는 InMemorySaver, 서비스는 C12에서 Postgres.
승인 단계의 답은 글자로 넘긴다: "approve", "reject", "revise:이유"(C10 D7, D22).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.types import Command

from ..observability import new_session_id
from ..retrieval import Knowledge
from ..runtime.loop import run_agent
from ..runtime.types import AgentConfig

if TYPE_CHECKING:
    from ..agents import AgentTeam

WAITING = "waiting"  # 사람 승인에서 멈춘 상태. 나머지 상태(approved, rejected, no_evidence)는 워크플로가 정한다.
SHOW = 200  # 추적 기록에 노드 결과를 이만큼만 남긴다(전체 답은 에이전트의 llm 줄에 있다).


@dataclass
class WorkflowContext:
    """워크플로가 노드 안에서 쓰는 것. 채팅과 같은 에이전트 설정(시스템 프롬프트, Tool, 모델)을 쓴다(C10 D13)."""

    config: AgentConfig
    knowledge: Knowledge | None = None
    max_revisions: int = 2  # WORKFLOW_MAX_REVISIONS(C10 D8)
    progress: Callable[[str], None] = lambda message: None  # 노드가 알리는 진행 줄(C10 D24). 응용이 화면에 보여 준다.
    team: AgentTeam | None = None  # 에이전트 정의. 노드마다 다른 에이전트를 쓰는 워크플로(C14 routing)가 쓴다.
    base: AgentConfig | None = None  # 모든 에이전트가 함께 쓰는 설정(설치된 전체 Tool, 승인, Safety, 추적). team.config_for에 넘긴다.


@dataclass(frozen=True)
class Workflow:
    name: str
    description: str
    build: Callable[[WorkflowContext], StateGraph]  # 그래프를 만든다(컴파일은 실행기가 체크포인트와 함께 한다).
    unavailable: Callable[[WorkflowContext], str | None] = lambda ctx: None  # 시작할 수 없는 이유. None이면 시작한다(C10 D16).


@dataclass
class Outcome:
    """시작이나 재개의 결과. status가 waiting이면 draft를 보여 주고 승인 답을 받아 resume을 부른다."""

    run_id: str
    workflow: str
    question: str
    status: str
    answer: str | None = None  # 끝났을 때의 최종 답(승인, 근거 없음). 거부면 None
    draft: str | None = None  # 승인 대기일 때 보여 줄 초안
    revisions: int = 0
    started: str = ""
    route: str | None = None  # routing 워크플로가 고른 경로(C14). 평가가 expect_route와 비교한다.


def ask_agent(config: AgentConfig, prompt: str) -> str:
    """에이전트 노드: 독(run_agent)을 새 이력으로 한 번 부르고 답만 돌려준다. 채팅 이력은 보지 않는다(C10 D13)."""
    return run_agent(config, prompt).answer or ""


class WorkflowRunner:
    def __init__(self, workflows: list[Workflow], context: WorkflowContext, checkpointer: BaseCheckpointSaver):
        self.workflows = {w.name: w for w in workflows}
        self.context = context
        self.checkpointer = checkpointer
        self._apps: dict[str, object] = {}

    def _app(self, name: str):
        if name not in self._apps:
            self._apps[name] = self.workflows[name].build(self.context).compile(checkpointer=self.checkpointer)
        return self._apps[name]

    def _record(self, kind: str, **fields) -> None:
        tracer = self.context.config.tracer
        if tracer is not None:
            tracer.record(kind, None, **fields)

    def start(self, name: str, question: str) -> Outcome:
        """워크플로를 시작한다. 없는 이름이나 시작할 수 없는 이유가 있으면 ValueError."""
        if name not in self.workflows:
            raise ValueError(f"없는 워크플로입니다: {name} (목록은 /workflows)")
        reason = self.workflows[name].unavailable(self.context)
        if reason:
            raise ValueError(reason)
        run_id = new_session_id("wf-")
        started = datetime.now().isoformat(timespec="seconds")
        self._record("workflow_start", run=run_id, workflow=name, question=question)
        state = {"workflow": name, "question": question, "started": started, "status": "running", "revisions": 0}
        return self._run(name, run_id, state)

    def resume(self, run_id: str, decision: str) -> Outcome:
        """승인 대기 실행을 승인 답으로 이어 간다. 프로그램을 껐다 켠 뒤에도 같은 run_id로 부른다."""
        name = self._workflow_of(run_id)
        self._record("workflow_resume", run=run_id, workflow=name, decision=decision)
        return self._run(name, run_id, Command(resume=decision))

    def pending(self) -> list[Outcome]:
        """승인을 기다리는 실행. 시작 시각이 최근인 순서다(C10 D15)."""
        # ponytail: 체크포인트를 전부 훑는다. 실행이 많아지면 실행 목록 표를 따로 둔다.
        run_ids = {c.config["configurable"]["thread_id"] for c in self.checkpointer.list(None)}
        waiting = []
        for run_id in run_ids:
            name = self._workflow_of(run_id)
            if name in self.workflows:
                outcome = self._outcome(name, run_id)
                if outcome.status == WAITING:
                    waiting.append(outcome)
        return sorted(waiting, key=lambda o: o.started, reverse=True)

    def _workflow_of(self, run_id: str) -> str:
        saved = self.checkpointer.get_tuple(_config(run_id))
        if saved is None:
            raise ValueError(f"없는 실행입니다: {run_id}")
        return saved.checkpoint["channel_values"].get("workflow", "")

    def _run(self, name: str, run_id: str, payload) -> Outcome:
        app = self._app(name)
        for update in app.stream(payload, _config(run_id), stream_mode="updates"):
            for node, values in update.items():
                if node != "__interrupt__":
                    self._record("workflow_node", run=run_id, node=node, update=_brief(values or {}))
        outcome = self._outcome(name, run_id)
        if outcome.status == WAITING:
            self._record("workflow_interrupt", run=run_id, revisions=outcome.revisions)
        else:
            self._record("workflow_end", run=run_id, status=outcome.status, revisions=outcome.revisions)
        return outcome

    def _outcome(self, name: str, run_id: str) -> Outcome:
        snapshot = self._app(name).get_state(_config(run_id))
        values = snapshot.values
        waiting = bool(snapshot.interrupts)
        return Outcome(
            run_id=run_id,
            workflow=name,
            question=values.get("question", ""),
            status=WAITING if waiting else values.get("status", "running"),
            answer=None if waiting else values.get("answer"),
            draft=snapshot.interrupts[0].value.get("draft") if waiting else None,
            revisions=values.get("revisions", 0),
            started=values.get("started", ""),
            route=values.get("route"),
        )


def _config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _brief(values: dict) -> dict:
    return {k: (v[:SHOW] if isinstance(v, str) else v) for k, v in values.items()}
