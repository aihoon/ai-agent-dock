"""uv run evals: 실제 모델 평가를 실행하는 짧은 명령. uv run pytest tests/test_evals_live.py --live -s와 같다.

뒤에 붙인 인자는 pytest에 그대로 넘긴다. 예: uv run evals --provider anthropic --repeat 3
프로젝트 루트에서 실행한다.
"""

import sys
from pathlib import Path


def main() -> None:
    import pytest  # 개발 의존성이라 실행할 때만 가져온다.

    target = Path.cwd() / "tests" / "test_evals_live.py"
    sys.exit(pytest.main([str(target), "--live", "-s", "-q", *sys.argv[1:]]))
