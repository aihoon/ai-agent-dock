"""추적 리포트 계산: logs/sessions/의 세션 파일을 읽어 세션 목록, 세션별 통계, 전체 통계, 세션별 비교를 만든다.

통계는 질문과 답 같은 내용은 보여 주지 않는다. 호출 수, 걸린 시간, 토큰 수만 센다.
내용은 스텝 보기(format_steps)에서만 보여 준다(C13 T6).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..llm.tokens import TIKTOKEN


@dataclass
class TurnStats:
    turn: int
    llm_calls: int = 0
    tool_calls: int = 0
    duration_ms: float = 0.0  # 턴 끝 줄(turn_end)의 턴 전체 시간
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0  # 입력 토큰 중 캐시에서 읽은 토큰(C9)
    estimated: int = 0  # tiktoken으로 계산한 토큰 값의 개수(제공자가 주지 않은 값)


@dataclass
class SessionStats:
    session: str
    start: str
    end: str
    model: str | None
    turns: list[TurnStats]
    safety: dict[str, int] = field(default_factory=dict)  # Safety 기록 종류별 개수(C11 D13)
    hooks: int = 0  # 사용자 Hook 실행 수(C11 D21)
    agents: dict[str, list[int]] = field(default_factory=dict)  # 에이전트 이름 → [LLM 호출, 입력 토큰, 출력 토큰](C12 F4)
    delegations: int = 0  # delegate 호출 수

    def total(self, name: str) -> float:
        return sum(getattr(t, name) for t in self.turns)


def load_sessions(directory: Path) -> dict[str, list[dict]]:
    """세션 번호 → 그 세션의 줄 목록. 파일 이름(세션 번호) 순서대로 돌려준다."""
    sessions = {}
    for path in sorted(directory.glob("*.jsonl")) if directory.is_dir() else []:
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            sessions[path.stem] = lines
    return sessions


def session_stats(session: str, lines: list[dict]) -> SessionStats:
    turns: dict[int, TurnStats] = {}
    safety: dict[str, int] = {}
    hooks = delegations = 0
    agents: dict[str, list[int]] = {}
    model = None
    for line in lines:
        kind = line["kind"]
        if kind == "session_start":
            model = line.get("model")
            continue
        if kind == "delegate":
            delegations += 1
            continue
        if kind == "llm":  # 하위 에이전트의 호출도 그 턴에 더하고, 에이전트별로도 따로 센다.
            counts = agents.setdefault(line.get("agent") or "main", [0, 0, 0])
            counts[0] += 1
            counts[1] += line.get("input_tokens") or 0
            counts[2] += line.get("output_tokens") or 0
        if kind == "safety":
            safety[line.get("check", "?")] = safety.get(line.get("check", "?"), 0) + 1
            continue
        if kind == "hook":
            hooks += 1
            continue
        if kind not in ("turn_start", "llm", "tool", "turn_end"):
            continue  # compact, session_resume 같은 명령 기록은 턴 통계에 넣지 않는다.
        stats = turns.setdefault(line["turn"], TurnStats(turn=line["turn"]))
        if kind == "llm":
            stats.llm_calls += 1
            stats.input_tokens += line.get("input_tokens") or 0
            stats.output_tokens += line.get("output_tokens") or 0
            stats.cached_tokens += line.get("cached_input_tokens") or 0
            stats.estimated += (line.get("input_tokens_source") == TIKTOKEN) + (line.get("output_tokens_source") == TIKTOKEN)
        elif kind == "tool":
            stats.tool_calls += 1
        elif kind == "turn_end" and not line.get("agent"):  # 턴 시간은 main의 턴 끝 줄만 쓴다.
            stats.duration_ms = line.get("duration_ms") or 0.0
    return SessionStats(
        session=session,
        start=lines[0]["time"],
        end=lines[-1]["time"],
        model=model,
        turns=[turns[k] for k in sorted(turns)],
        safety=safety,
        hooks=hooks,
        agents=agents,
        delegations=delegations,
    )


def all_stats(directory: Path) -> list[SessionStats]:
    return [session_stats(session, lines) for session, lines in load_sessions(directory).items()]


def format_list(stats: list[SessionStats], names: dict[str, str] | None = None) -> str:
    """세션 목록: 세션 수, 세션마다 시작, 끝, 턴 수. names(세션 번호 → 세션 이름)를 주면 이름도 보여 준다(C9 D26)."""
    names = names or {}
    rows = [f"세션 {len(stats)}개"]
    rows += [f"  {s.session}  {s.start} ~ {s.end}  턴 {len(s.turns)}{f'  [{names[s.session]}]' if s.session in names else ''}" for s in stats]
    return "\n".join(rows)


def format_session(s: SessionStats) -> str:
    """세션 하나의 통계: 턴별 LLM 호출 수, Tool 호출 수, 걸린 시간, 토큰 수와 합계."""
    rows = [f"세션 {s.session} (모델 {s.model or '-'})", "  턴  LLM  Tool   시간(ms)   입력토큰  (캐시)  출력토큰  계산값"]
    for t in s.turns:
        rows.append(f"  {t.turn:>2}  {t.llm_calls:>3}  {t.tool_calls:>4}  {t.duration_ms:>9.1f}  {t.input_tokens:>9}  {t.cached_tokens:>6}  {t.output_tokens:>8}  {t.estimated:>6}")
    rows.append(
        f"  합계 {s.total('llm_calls'):>3.0f}  {s.total('tool_calls'):>4.0f}  {s.total('duration_ms'):>9.1f}  "
        f"{s.total('input_tokens'):>9.0f}  {s.total('cached_tokens'):>6.0f}  {s.total('output_tokens'):>8.0f}  {s.total('estimated'):>6.0f}"
    )
    if s.safety:
        rows.append("  Safety: " + ", ".join(f"{k} {v}" for k, v in sorted(s.safety.items())))
    if s.hooks:
        rows.append(f"  Hook 실행: {s.hooks}")
    if s.delegations:  # 위임이 있던 세션만 에이전트별로 나눠 보여 준다(C12 F4).
        rows.append(f"  위임: {s.delegations}번")
        rows += [f"    {name}: LLM {llm}, 입력 {tin:,}, 출력 {tout:,}" for name, (llm, tin, tout) in s.agents.items()]
    return "\n".join(rows)


def format_total(stats: list[SessionStats]) -> str:
    """전체 통계: 모든 세션의 합계와 턴당 평균."""
    turns = [t for s in stats for t in s.turns]
    if not turns:
        return "기록된 턴이 없습니다."
    rows = [f"세션 {len(stats)}개, 턴 {len(turns)}개"]
    for name, label in [("llm_calls", "LLM 호출"), ("tool_calls", "Tool 호출"), ("duration_ms", "시간(ms)"), ("input_tokens", "입력 토큰"), ("cached_tokens", "캐시 입력 토큰"), ("output_tokens", "출력 토큰")]:
        total = sum(getattr(t, name) for t in turns)
        rows.append(f"  {label}: 합계 {total:,.1f}, 턴당 평균 {total / len(turns):,.1f}")
    rows.append(f"  tiktoken으로 계산한 토큰 값: {sum(t.estimated for t in turns)}개")
    return "\n".join(rows)


def format_compare(stats: list[SessionStats]) -> str:
    """세션별 비교: 세션마다 한 줄."""
    rows = ["  세션                        턴  LLM  Tool   시간(ms)   입력토큰  출력토큰"]
    for s in stats:
        rows.append(
            f"  {s.session:<26}  {len(s.turns):>2}  {s.total('llm_calls'):>3.0f}  {s.total('tool_calls'):>4.0f}  "
            f"{s.total('duration_ms'):>9.1f}  {s.total('input_tokens'):>9.0f}  {s.total('output_tokens'):>8.0f}"
        )
    return "\n".join(rows)


STEP_TEXT_LIMIT = 400  # 스텝 보기에서 글 하나를 줄이는 길이(글자)


def _short(value: object) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = "\n      ".join(row for row in text.splitlines() if row.strip())  # 여러 줄 글은 빈 줄을 빼고 들여 쓴다.
    return text if len(text) <= STEP_TEXT_LIMIT else f"{text[:STEP_TEXT_LIMIT]}… (+{len(text) - STEP_TEXT_LIMIT}자)"


def format_steps(lines: list[dict]) -> str:
    """세션 하나를 기록 순서대로: 에이전트에 준 프롬프트, 스텝마다 모델 출력과 Tool 결과, 워크플로 노드의 상태 변경."""
    rows = []
    for line in lines:
        kind, who = line["kind"], f"[{line['agent']}] " if line.get("agent") else ""
        if kind == "workflow_start":
            rows.append(f"━━ 워크플로 {line.get('workflow')}: {line.get('question')}")
        elif kind == "turn_start":
            rows.append(f"{who}[턴 {line['turn']}] 프롬프트: {_short(line.get('question'))}")
        elif kind == "llm":
            calls = ", ".join(f"{c['name']}({_short(c.get('arguments'))})" for c in line.get("tool_calls") or [])
            out = f"Tool 요청 {calls}" if calls else _short(line.get("text"))
            rows.append(f"  {who}스텝 {line['step']} 모델 → {out}  (입력 {line.get('input_tokens')}, 출력 {line.get('output_tokens')}, {line.get('duration_ms')}ms)")
        elif kind == "tool":
            rows.append(f"    {who}Tool {line['name']} → {_short(line.get('result'))}")
        elif kind == "delegate":
            rows.append(f"    위임 → {line.get('target')}: {_short(line.get('task'))}")
        elif kind == "workflow_node":
            rows.append(f"▸ 노드 {line['node']}: " + ", ".join(f"{k}={_short(v)}" for k, v in (line.get("update") or {}).items()))
        elif kind == "workflow_end":
            rows.append(f"━━ 워크플로 끝: {line.get('status')}")
    return "\n".join(rows)
