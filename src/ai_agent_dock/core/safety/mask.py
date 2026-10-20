"""비밀과 개인정보 가리기(C11 D5, D16). 정규식이라 형식이 다른 값은 놓치고, 비슷한 숫자는 잘못 가릴 수 있다.

기록(추적, 감사 로그)에는 다섯 종류를 모두 가리고, 답에는 API 키만 가린다.
이 모듈은 다른 모듈을 import하지 않는다(observability, hitl이 이 모듈을 쓴다).
"""

from __future__ import annotations

import re
from typing import Any

KEY = ("[KEY]", re.compile(r"\b(?:sk-(?:ant-)?[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})"))
PATTERNS = [
    KEY,
    ("[EMAIL]", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("[RRN]", re.compile(r"\b\d{6}-[1-4]\d{6}\b")),
    ("[CARD]", re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b")),
    ("[PHONE]", re.compile(r"\b0\d{1,2}-\d{3,4}-\d{4}\b")),
]


def mask_text(text: str) -> str:
    """다섯 종류를 모두 가린다(기록용)."""
    for label, pattern in PATTERNS:
        text = pattern.sub(label, text)
    return text


def mask_keys(text: str) -> str:
    """API 키 모양만 가린다(답용)."""
    label, pattern = KEY
    return pattern.sub(label, text)


def mask_value(value: Any) -> Any:
    """기록에 넣을 값 안의 글자를 모두 가린다. dict와 list는 안쪽까지 본다."""
    if isinstance(value, str):
        return mask_text(value)
    if isinstance(value, dict):
        return {k: mask_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [mask_value(v) for v in value]
    return value
