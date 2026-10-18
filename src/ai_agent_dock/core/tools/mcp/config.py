"""MCP 서버 설정 파일 읽기. 우리가 서버를 만드는 것이 아니라, 클라이언트가 어느 서버에 접속할지만 적는다.

형식(mcp.example.json 참고):
  {"servers": {"서버이름": {"command": "uvx", "args": ["mcp-server-time"], "env": {"KEY": "값"}}}}
args와 env는 생략할 수 있다. stdio 서버만 지원한다(원격 서버는 C12).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ServerSpec:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    trusted: bool = False  # 이 서버의 읽기 전용 힌트를 믿어도 되는지. 기본은 믿지 않는다(승인 절차를 거친다).


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
        if not enabled:
            continue  # 끈 서버는 연결하지 않는다. 설정 항목은 파일에 그대로 남긴다.
        specs[name] = ServerSpec(command=command, args=list(args), env=dict(env), trusted=trusted)
    return specs
