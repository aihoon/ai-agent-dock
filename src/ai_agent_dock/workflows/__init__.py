"""워크플로 정의(C10 D9): 플랫폼 위에서 순서를 코드로 정한 그래프들. core는 이 폴더를 import하지 않고, 응용이 WORKFLOWS를 넘긴다."""

from . import answer_check, prompt_chaining, routing

WORKFLOWS = [answer_check.WORKFLOW, prompt_chaining.WORKFLOW, routing.WORKFLOW]

__all__ = ["WORKFLOWS"]
