"""pytest 설정: 실제 모델 평가(live 표시)는 --live를 줄 때만 실행한다.

uv run pytest        → live 테스트는 건너뛴다(가짜 모델 테스트만, 비용과 네트워크 없음).
uv run pytest --live → live 테스트만 실행한다(실제 모델 호출). 짧은 명령: uv run evals
"""

import pytest


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", default=False, help="실제 모델로 평가 케이스를 실행한다(비용과 네트워크가 든다).")
    parser.addoption("--provider", default=None, help="평가에 쓸 제공자(기본: .env의 LLM_PROVIDER).")
    parser.addoption("--repeat", type=int, default=1, help="평가 케이스마다 반복할 횟수(기본 1). 통과율은 반복 중 통과 비율이다.")


def pytest_configure(config):
    config.addinivalue_line("markers", "live: 실제 모델을 부르는 평가. --live를 줄 때만 실행한다.")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        deselected = [item for item in items if "live" not in item.keywords]
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = [item for item in items if "live" in item.keywords]
        return
    skip = pytest.mark.skip(reason="실제 모델 평가는 --live(또는 uv run evals)로만 실행한다")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
