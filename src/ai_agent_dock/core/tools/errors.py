"""Tool 계층의 예외. 루프는 Tool이 던지는 모든 예외를 오류 관찰(is_error)로 바꿔 모델에 전달한다."""

# ValueError, RuntimeError는 내장 예외라 import하지 않는다.
class ToolArgumentError(ValueError):
    """모델이 준 인자가 Tool의 parameters(JSON Schema)에 맞지 않는다."""


class ToolError(RuntimeError):
    """Tool이 실행됐으나 스스로 실패를 알렸다(예: MCP 결과의 isError). 메시지는 모델에게 그대로 전달된다."""
