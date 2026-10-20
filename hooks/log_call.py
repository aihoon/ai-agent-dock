"""예시 Hook(before_tool): Tool 호출을 logs/hook_calls.jsonl에 한 줄씩 남긴다. 막지 않는다.

hooks.json에 적으면 플랫폼이 Tool 실행 직전에 이 스크립트를 띄우고, 표준 입력으로
{"event": "before_tool", "tool": 이름, "arguments": {...}}를 넘긴다.
막으려면 표준 출력으로 {"block": "이유"}를 쓴다. 아무것도 쓰지 않으면 통과다.
인자 값은 비밀이 섞일 수 있어 이름만 남긴다.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

event = json.load(sys.stdin)
log = Path(__file__).resolve().parent.parent / "logs" / "hook_calls.jsonl"
log.parent.mkdir(parents=True, exist_ok=True)
with log.open("a", encoding="utf-8") as f:
    line = {"time": datetime.now().isoformat(timespec="seconds"), "tool": event.get("tool"), "args": sorted(event.get("arguments", {}))}
    f.write(json.dumps(line, ensure_ascii=False) + "\n")
