"""MCP 서버 하나와의 연결. SDK는 async 전용이고 우리 루프는 동기라서, 서버마다 백그라운드 스레드에서
이벤트 루프를 돌리고 위쪽에는 동기 함수(start, call_tool, close)만 보여 준다.

연결은 start()에서 열어 close()까지 유지한다(stdio 서버를 호출마다 다시 띄우지 않기 위해서).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from mcp import Client, StdioServerParameters

from ..errors import ToolError
from .config import ServerSpec

MAX_LIST_PAGES = 50  # 서버가 next_cursor를 끝없이 돌려주는 경우를 막는다.
CLOSE_WAIT = 10.0  # close()가 스레드 종료를 기다리는 최대 시간(초)


class McpConnection:
    def __init__(self, name: str, spec: ServerSpec, *, call_timeout: float):
        self.name = name
        self.spec = spec
        self.call_timeout = call_timeout
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()  # 연결과 Tool 목록 수신이 끝났거나(성공) 실패했을 때 켠다.
        self._closing = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._client: Client | None = None
        self._tools: list[Any] = []
        self._error: Exception | None = None

    def start(self, timeout: float) -> list[Any]:
        """서버를 띄우고 연결해 Tool 목록(MCP Tool 정의)을 돌려준다. 실패하면 예외를 던지고 정리한다."""
        self._thread = threading.Thread(target=self._run, name=f"mcp-{self.name}", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            self.close()
            raise TimeoutError(f"MCP 서버 {self.name!r} 시작이 {timeout:g}초 안에 끝나지 않았다")
        if self._error is not None:
            self.close()
            raise self._error
        return self._tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """서버의 Tool을 호출해 CallToolResult를 돌려준다. 제한 시간을 넘기면 ToolError를 던진다."""
        loop, client = self._loop, self._client
        if loop is None or client is None or self._closing:
            raise ToolError(f"MCP 서버 {self.name!r}와 연결되어 있지 않다")
        future = asyncio.run_coroutine_threadsafe(client.call_tool(name, arguments), loop)
        try:
            return future.result(self.call_timeout)
        except TimeoutError:
            future.cancel()
            raise ToolError(f"MCP 서버 {self.name!r}의 {name} 호출이 {self.call_timeout:g}초 안에 끝나지 않았다") from None

    def close(self) -> None:
        """연결을 닫고 서버 프로세스를 정리한다. 여러 번 불러도 안전하다."""
        self._closing = True  # 순서 중요: 플래그를 먼저 켜고 task를 읽는다(_main은 task를 먼저 저장하고 플래그를 본다).
        loop, task = self._loop, self._task
        if loop is not None and task is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass  # 확인 직후 루프가 끝난 경우. 이미 정리된 것이므로 할 일이 없다.
        if self._thread is not None:
            self._thread.join(CLOSE_WAIT)

    # --- 아래는 백그라운드 스레드에서 실행된다 ---

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as e:  # 연결 도중 실패. start()가 기다리고 있으면 그쪽으로 전달된다.
            self._error = self._error or _root_cause(e)
        finally:
            self._client = None
            self._ready.set()

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.current_task()
        if self._closing:
            return
        params = StdioServerParameters(command=self.spec.command, args=self.spec.args, env=self.spec.env or None)
        try:
            # legacy: server/discover를 먼저 묻지 않고 initialize로 연결한다. 최신 서버 중 discover를 모르는 서버가 경고를 남기지 않게 한다.
            async with Client(params, mode="legacy") as client:
                self._tools = await _list_all_tools(client)
                self._client = client
                self._ready.set()
                await asyncio.Future()  # close()가 task를 취소할 때까지 연결을 유지한다.
        except asyncio.CancelledError:
            pass  # close()에 의한 정상 종료. async with를 빠져나오며 서버 프로세스가 정리된다.


def _root_cause(error: Exception) -> Exception:
    """SDK가 TaskGroup으로 겹겹이 감싼 예외(ExceptionGroup)를 가장 안쪽의 실제 원인으로 푼다."""
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return error


async def _list_all_tools(client: Client) -> list[Any]:
    tools: list[Any] = []
    cursor: str | None = None
    for _ in range(MAX_LIST_PAGES):
        page = await client.list_tools(cursor=cursor)
        tools.extend(page.tools)
        cursor = page.next_cursor
        if not cursor:
            return tools
    raise RuntimeError(f"Tool 목록이 {MAX_LIST_PAGES}쪽을 넘는다")
