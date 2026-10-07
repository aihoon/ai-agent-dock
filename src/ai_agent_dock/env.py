"""환경 변수 읽기 도우미. 에이전트 코어의 기능이 아니라 공용 도구라서 core 밖에 둔다.

.env는 프로그램 시작점이 load_dotenv()로 먼저 읽어 둔다. 이 모듈은 이미 읽힌 환경 변수만 본다.
원칙: 코드에는 기본값만 두고, 실제 조절은 .env에서 한다. 값이 없거나 비어 있으면 기본값을 쓴다.
"""

from __future__ import annotations

import os


def env_str(name: str, default: str) -> str:
    return os.environ.get(name) or default


def env_int(name: str, default: int | None = None) -> int | None:
    """양의 정수 설정을 읽는다. 값이 없거나 비어 있으면 default, 잘못된 값이면 이름을 밝혀 오류를 낸다."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name}은 정수여야 한다: {raw!r}") from None
    if value <= 0:
        raise ValueError(f"{name}은 1 이상이어야 한다: {value}")
    return value
