"""플랫폼이 기본으로 제공하는 Tool(도메인 지식이 없는 범용 Tool).

Tool 레지스트리, 입력 스키마 검증, MCP 연동은 C4에서 이 폴더에 추가한다.
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime

from ..llm import Tool

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}
MAX_EXPONENT = 100  # 9**9**9 같은 수식이 계산을 멈추지 않게 하는 상한


def _eval(node):
    # bool은 int의 하위 클래스라서 따로 거른다(True+1 같은 수식을 허용하지 않기 위해).
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"지수가 너무 크다({MAX_EXPONENT} 이하만 허용)")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("지원하지 않는 수식")


def calculate(expression: str):
    """숫자와 + - * / ** 만 허용하는 안전한 계산기. eval을 쓰지 않는다."""
    return _eval(ast.parse(expression, mode="eval").body)


def get_current_time() -> str:
    """현재 시각을 시간대가 포함된 ISO 형식(초 단위)으로 돌려준다."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


BUILTIN_TOOLS = [
    Tool(
        name="get_current_time",
        description="현재 날짜와 시각을 ISO 형식으로 돌려준다.",
        parameters={"type": "object", "properties": {}},
        func=get_current_time,
    ),
    Tool(
        name="calculate",
        description="수식을 계산한다. 숫자와 + - * / ** 만 사용할 수 있다.",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "예: 17*23"}},
            "required": ["expression"],
        },
        func=calculate,
    ),
]
