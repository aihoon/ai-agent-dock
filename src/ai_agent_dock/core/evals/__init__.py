"""평가: 케이스(프로젝트 루트 evals/)를 실행해 규칙으로 판정하고, 결과를 data/evals/에 저장한다."""

from .run import CaseResult, evaluate, last_passed, save_run, scripted_approver, scripted_input, summarize
from .score import Case, called_tools, check, load_cases, regressions

__all__ = [
    "Case",
    "CaseResult",
    "called_tools",
    "check",
    "evaluate",
    "last_passed",
    "load_cases",
    "regressions",
    "save_run",
    "scripted_approver",
    "scripted_input",
    "summarize",
]
