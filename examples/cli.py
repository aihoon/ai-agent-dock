"""간단한 CLI 데모. 사용법: uv run python examples/cli.py "지금 몇 시야? 17*23은?"

실제 모델을 호출한다(.env의 LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL 사용).
"""

import ast
import logging
import operator
import sys
from datetime import datetime

from dotenv import load_dotenv

from ai_agent_dock.core.llm import Tool
from ai_agent_dock.core.runtime.loop import run_agent

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("지수가 너무 크다(100 이하만 허용)")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("지원하지 않는 수식")


def calculate(expression: str):
    """숫자와 + - * / ** 만 허용하는 안전한 계산기. eval을 쓰지 않는다."""
    return _eval(ast.parse(expression, mode="eval").body)


def get_current_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


TOOLS = [
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

if __name__ == "__main__":
    load_dotenv()  # 프로그램 시작점에서 .env를 한 번만 읽는다. LLM_PROVIDER 등이 여기서 정해진다.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    question = " ".join(sys.argv[1:]) or "지금 몇 시야? 그리고 17*23은 얼마야?"
    result = run_agent(question, TOOLS, system="계산과 시간 조회는 반드시 Tool을 사용해 답한다.")
    print(result.answer)
    print(f"[status={result.status}, steps={result.steps}]")
