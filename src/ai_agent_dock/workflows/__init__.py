"""워크플로 정의(C10 D9): 플랫폼 위에서 순서를 코드로 정한 그래프들. core는 이 폴더를 import하지 않고, 응용이 WORKFLOWS를 넘긴다."""

from . import answer_check, evaluator_optimizer, orchestrator_workers, plan_and_execute, prompt_chaining, reflexion, routing, sectioning, voting

WORKFLOWS = [
    answer_check.WORKFLOW, 
    prompt_chaining.WORKFLOW, 
    routing.WORKFLOW, 
    sectioning.WORKFLOW, 
    voting.WORKFLOW,
    orchestrator_workers.WORKFLOW,
    evaluator_optimizer.WORKFLOW,
    plan_and_execute.WORKFLOW,
    reflexion.WORKFLOW,
    ]

__all__ = ["WORKFLOWS"]
