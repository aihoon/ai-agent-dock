"""MCP 서버 설정 파일 읽기. 우리가 서버를 만드는 것이 아니라, 클라이언트가 어느 서버에 접속할지만 적는다.

형식(mcp.example.json 참고):
  {"servers": {"서버이름": {"command": "uvx", "args": ["mcp-server-time"], "env": {"KEY": "값"}}}}
args와 env는 생략할 수 있다. stdio 서버만 지원한다(원격 서버는 C15).
env 값에 "${이름}"을 쓰면 그 환경 변수(.env)의 값으로 바꾼다. 키를 mcp.json에 직접 적지 않기 위해서다(C12 Q13).
"read_only": ["Tool 이름"]은 서버가 읽기 전용 힌트를 주지 않는 Tool을 운영자가 읽기 전용으로 지정한다(C12 V6).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

ENV_REF = re.compile(r"\$\{(\w+)\}")


@dataclass(frozen=True)
class ServerSpec:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    trusted: bool = False  # 이 서버의 읽기 전용 힌트를 믿어도 되는지. 기본은 믿지 않는다(승인 절차를 거친다).
    read_only: list[str] = field(default_factory=list)  # 서버의 Tool 이름(접두어 없이). 힌트가 없어도 읽기 전용으로 본다.


def load_config(path: str | Path) -> dict[str, ServerSpec]:
    """설정 파일을 읽어 서버 이름 → ServerSpec 표를 돌려준다. 잘못된 곳은 파일과 항목을 밝혀 ValueError를 낸다."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"MCP 설정 파일이 없다: {path} (mcp.example.json을 복사해 만든다)") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"MCP 설정 파일이 JSON이 아니다: {path} ({e})") from None

    servers = data.get("servers") if isinstance(data, dict) else None
    if not isinstance(servers, dict):
        raise ValueError(f'MCP 설정 파일 {path}: 최상위에 "servers" 객체가 있어야 한다')

    specs: dict[str, ServerSpec] = {}
    for name, raw in servers.items():
        where = f"MCP 설정 파일 {path}의 서버 {name!r}"
        if not name:
            raise ValueError(f"MCP 설정 파일 {path}: 서버 이름이 비어 있다")
        if not isinstance(raw, dict):
            raise ValueError(f"{where}: 객체여야 한다")
        command = raw.get("command")
        if not isinstance(command, str) or not command:
            raise ValueError(f'{where}: "command"는 비어 있지 않은 문자열이어야 한다')
        args = raw.get("args", [])
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise ValueError(f'{where}: "args"는 문자열 배열이어야 한다')
        env = raw.get("env", {})
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise ValueError(f'{where}: "env"는 문자열 값의 객체여야 한다')
        trusted = raw.get("trusted", False)
        if not isinstance(trusted, bool):
            raise ValueError(f'{where}: "trusted"는 true 또는 false여야 한다')
        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f'{where}: "enabled"는 true 또는 false여야 한다')
        read_only = raw.get("read_only", [])
        if not isinstance(read_only, list) or not all(isinstance(t, str) for t in read_only):
            raise ValueError(f'{where}: "read_only"는 Tool 이름(문자열) 배열이어야 한다')
        if not enabled:
            continue  # 끈 서버는 연결하지 않는다. 설정 항목은 파일에 그대로 남긴다. 끈 서버의 ${...}은 읽지 않는다.
        env = {key: _expand(value, f"{where}의 env {key!r}") for key, value in env.items()}
        specs[name] = ServerSpec(command=command, args=list(args), env=env, trusted=trusted, read_only=list(read_only))
    return specs


def _expand(value: str, where: str) -> str:
    """'${이름}'을 환경 변수 값으로 바꾼다. 없거나 비어 있으면 시작할 때 실패한다(서버가 키 없이 뜨지 않게)."""

    def lookup(match: re.Match) -> str:
        found = os.environ.get(match.group(1))
        if not found:
            raise ValueError(f"{where}: 환경 변수 {match.group(1)}이 없다(.env에 값을 넣는다)")
        return found

    return ENV_REF.sub(lookup, value)
