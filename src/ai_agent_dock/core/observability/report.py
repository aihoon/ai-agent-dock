"""추적 리포트 계산: logs/sessions/의 세션 파일을 읽어 세션 목록, 세션별 통계, 전체 통계, 세션별 비교를 만든다.

질문과 답 같은 내용은 보여 주지 않는다. 호출 수, 걸린 시간, 토큰 수만 센다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
    estimated: int = 0  # tiktoken으로 계산한 토큰 값의 개수(제공자가 주지 않은 값)


@dataclass
class SessionStats:
    session: str
    start: str
    end: str
    model: str | None
    turns: list[TurnStats]

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
    model = None
    for line in lines:
        kind = line["kind"]
        if kind == "session_start":
            model = line.get("model")
            continue
        stats = turns.setdefault(line["turn"], TurnStats(turn=line["turn"]))
        if kind == "llm":
            stats.llm_calls += 1
            stats.input_tokens += line.get("input_tokens") or 0
            stats.output_tokens += line.get("output_tokens") or 0
            stats.estimated += (line.get("input_tokens_source") == TIKTOKEN) + (line.get("output_tokens_source") == TIKTOKEN)
        elif kind == "tool":
            stats.tool_calls += 1
        elif kind == "turn_end":
            stats.duration_ms = line.get("duration_ms") or 0.0
    return SessionStats(
        session=session,
        start=lines[0]["time"],
        end=lines[-1]["time"],
        model=model,
        turns=[turns[k] for k in sorted(turns)],
    )


def all_stats(directory: Path) -> list[SessionStats]:
    return [session_stats(session, lines) for session, lines in load_sessions(directory).items()]


def format_list(stats: list[SessionStats]) -> str:
    """세션 목록: 세션 수, 세션마다 시작, 끝, 턴 수."""
    rows = [f"세션 {len(stats)}개"]
    rows += [f"  {s.session}  {s.start} ~ {s.end}  턴 {len(s.turns)}" for s in stats]
    return "\n".join(rows)


def format_session(s: SessionStats) -> str:
    """세션 하나의 통계: 턴별 LLM 호출 수, Tool 호출 수, 걸린 시간, 토큰 수와 합계."""
    rows = [f"세션 {s.session} (모델 {s.model or '-'})", "  턴  LLM  Tool   시간(ms)   입력토큰  출력토큰  계산값"]
    for t in s.turns:
        rows.append(f"  {t.turn:>2}  {t.llm_calls:>3}  {t.tool_calls:>4}  {t.duration_ms:>9.1f}  {t.input_tokens:>9}  {t.output_tokens:>8}  {t.estimated:>6}")
    rows.append(
        f"  합계 {s.total('llm_calls'):>3.0f}  {s.total('tool_calls'):>4.0f}  {s.total('duration_ms'):>9.1f}  "
        f"{s.total('input_tokens'):>9.0f}  {s.total('output_tokens'):>8.0f}  {s.total('estimated'):>6.0f}"
    )
    return "\n".join(rows)


def format_total(stats: list[SessionStats]) -> str:
    """전체 통계: 모든 세션의 합계와 턴당 평균."""
    turns = [t for s in stats for t in s.turns]
    if not turns:
        return "기록된 턴이 없습니다."
    rows = [f"세션 {len(stats)}개, 턴 {len(turns)}개"]
    for name, label in [("llm_calls", "LLM 호출"), ("tool_calls", "Tool 호출"), ("duration_ms", "시간(ms)"), ("input_tokens", "입력 토큰"), ("output_tokens", "출력 토큰")]:
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
